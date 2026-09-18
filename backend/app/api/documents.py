import hashlib
import uuid

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import User, current_user, forbid_if_not_owner
from app.config import get_settings
from app.db.models import Chunk, DocStatus, Document
from app.db.session import SessionLocal
from app.schemas.documents import (
    ChunkOut,
    DocumentOut,
    SearchHitOut,
    SearchResponse,
    StatsOut,
)
from app.services import ingest
from app.services.embeddings import get_embeddings
from app.services.parsing import UnsupportedFileType, detect_kind
from app.services.retrieval import retrieve
from app.services.storage import StorageError, get_storage
from app.services.vectorstore import get_vector_store

log = structlog.get_logger()

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


def _owner_match(owner_id: str | None):
    """Predicate for "this row belongs to this caller".

    Spelled with `is_(None)` rather than `==` because `x = NULL` is NULL, never
    true. With auth off every row has a NULL owner, so a plain `==` matches
    nothing at all -- the lookup would report every upload as new and the whole
    dedupe would be dead in precisely the configuration it ships in.

    Pulled out of the query so a test can compile it without a database.
    """
    if owner_id is None:
        return Document.owner_id.is_(None)
    return Document.owner_id == owner_id


async def _find_by_digest(
    session: AsyncSession, digest: str, owner_id: str | None
) -> Document | None:
    """The caller's existing copy of these bytes, if any.

    Scoped by owner even though the digest is global: one tenant must not learn
    that another holds a file, and must not inherit a row it cannot delete.
    """
    result = await session.execute(
        select(Document)
        .where(Document.content_sha256 == digest, _owner_match(owner_id))
        # Oldest wins, so repeated uploads keep converging on one row rather
        # than hopping between near-simultaneous inserts.
        .order_by(Document.created_at.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    background: BackgroundTasks,
    response: Response,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
) -> Document:
    """Accept a file, return immediately, embed in the background.

    Embedding is rate-limited to ~133 chunks/minute, so holding the request
    open would time out any realistic proxy. The client polls GET /documents.

    Already-indexed content is returned as-is with **200** instead of 201, so
    re-dropping a folder is a cheap sync rather than a corpus-corrupting event.
    Duplicates are not merely untidy: each one spends real embedding quota, and
    it puts two copies of every passage in the vector store under DIFFERENT
    chunk ids, which defeats the retrieval-level dedupe and quietly burns a
    top-k slot on a passage the model has already been given.
    """
    filename = file.filename or "untitled"

    # Validate the type before reading the body, so a junk upload is cheap.
    try:
        detect_kind(filename, file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(data) // 1024 // 1024}MB; limit is "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024}MB.",
        )

    digest = hashlib.sha256(data).hexdigest()

    async with SessionLocal() as session:
        existing = await _find_by_digest(session, digest, user.owner_id)
        if existing is not None and existing.status is not DocStatus.failed:
            response.status_code = 200
            log.info(
                "upload_duplicate",
                document_id=str(existing.id),
                filename=filename,
                existing_filename=existing.filename,
            )
            return existing

        if existing is not None:
            # A FAILED row is a duplicate of nothing useful -- it holds no
            # chunks and no vectors. Treating it as one would make re-uploading
            # after a transient parse or embedding failure impossible: the user
            # would get a cheerful 200 and a document that stays broken for
            # ever. Reuse the row and run ingestion again.
            existing.status = DocStatus.pending
            existing.error = None
            existing.filename = filename
            existing.n_pages = existing.n_chunks = existing.n_embedded = 0
            await session.commit()
            await session.refresh(existing)
            background.add_task(ingest.ingest_document, existing.id, data)
            log.info("upload_retry_failed", document_id=str(existing.id))
            return existing

        doc = Document(
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(data),
            status=DocStatus.pending,
            owner_id=user.owner_id,
            content_sha256=digest,
        )
        session.add(doc)
        try:
            await session.commit()
        except IntegrityError:
            # Two uploads of the same bytes in flight at once: both SELECTed
            # before either INSERTed, so the check above cleared both and the
            # index caught the loser. The uploader sends files sequentially,
            # but two browser tabs do not coordinate.
            await session.rollback()
            winner = await _find_by_digest(session, digest, user.owner_id)
            if winner is None:
                raise
            response.status_code = 200
            log.info("upload_duplicate_race", document_id=str(winner.id))
            return winner
        await session.refresh(doc)

    # FastAPI runs this after the response is sent. Fine for a demo; a real
    # deployment would hand this to a worker queue so a restart mid-ingest
    # doesn't silently abandon the document in `embedding` state.
    background.add_task(ingest.ingest_document, doc.id, data)

    log.info("upload_accepted", document_id=str(doc.id), filename=filename, bytes=len(data))
    return doc


@router.get("/documents", response_model=list[DocumentOut])
async def get_documents(user: User = Depends(current_user)) -> list[Document]:
    return await ingest.list_documents(owner_id=user.owner_id)


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID, user: User = Depends(current_user)
) -> Document:
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        forbid_if_not_owner(doc.owner_id, user)
        return doc


@router.get("/documents/{document_id}/file")
async def get_document_file(
    document_id: uuid.UUID, user: User = Depends(current_user)
) -> Response:
    """The ORIGINAL upload, so a citation can be checked against its source.

    A redirect when the store can sign a URL, and the bytes themselves when it
    cannot. R2 signs; the local directory has no public host to sign for, so
    the API serves it. The caller does not need to know which -- following a
    redirect is what a browser does anyway.

    `inline` rather than `attachment`: the point is to OPEN page four, not to
    download a copy of the handbook.
    """
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        forbid_if_not_owner(doc.owner_id, user)
        key, filename, content_type = doc.storage_key, doc.filename, doc.content_type

    if not key:
        # Uploaded before storage existed, or stored while the backend was
        # unreachable. Said plainly, because "not found" would suggest the
        # document itself is gone when its text is perfectly well indexed.
        raise HTTPException(
            status_code=404,
            detail="The original file was not kept for this document.",
        )

    store = get_storage()
    signed = await store.signed_url(key)
    if signed:
        return RedirectResponse(signed, status_code=307)

    try:
        data = await store.get(key)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=data,
        media_type=content_type or "application/octet-stream",
        headers={
            # The filename is quoted and stripped of quotes of its own, because
            # it is user-supplied and this header is parsed.
            "Content-Disposition": (
                f'inline; filename="{filename.replace(chr(34), "")}"'
            )
        },
    )


@router.delete("/documents/{document_id}", status_code=204)
async def remove_document(
    document_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    # Ownership is checked before deletion, not inside it, so a mismatched
    # caller gets a 404 without any write happening.
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        forbid_if_not_owner(doc.owner_id, user)

    deleted = await ingest.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, tags=["chunks"])
async def get_chunk(
    chunk_id: uuid.UUID, user: User = Depends(current_user)
) -> ChunkOut:
    """One chunk by id. Backs clickable citations.

    Reads from Postgres rather than the Qdrant payload: same text, but no
    vector round-trip and no dependency on the payload's shape.
    """
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(Chunk, Document.filename, Document.owner_id)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id == chunk_id)
            )
        ).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Chunk not found.")
        chunk, filename, owner_id = row
        # A chunk id is a UUID from a citation, so this is the endpoint most
        # likely to be probed with someone else's id.
        forbid_if_not_owner(owner_id, user)
        return ChunkOut.model_validate(chunk).model_copy(update={"filename": filename})


@router.get(
    "/documents/{document_id}/chunks", response_model=list[ChunkOut], tags=["chunks"]
)
async def get_document_chunks(
    document_id: uuid.UUID, user: User = Depends(current_user)
) -> list[ChunkOut]:
    """Every chunk of a document, in order.

    Lets you see exactly how a document was split -- the fastest way to
    diagnose a wrong answer, since chunking is the biggest lever on retrieval.
    """
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        forbid_if_not_owner(doc.owner_id, user)
        rows = (
            await session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            )
        ).scalars()
        return [
            ChunkOut.model_validate(c).model_copy(update={"filename": doc.filename})
            for c in rows
        ]


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    limit: int | None = Query(None, ge=1, le=20),
    document_id: list[uuid.UUID] | None = Query(None),
    multi_query: bool | None = Query(
        None, description="Rewrite into variations and fuse with RRF"
    ),
    user: User = Depends(current_user),
) -> SearchResponse:
    """Retrieval only, no generation.

    This exists to inspect retrieval on its own. When an answer is wrong, this
    is how you tell whether retrieval or generation is at fault -- and it's the
    cheap way to compare multi_query on/off without spending a model call on
    the answer.
    """
    hits = await retrieve(
        q,
        top_k=limit,
        document_ids=document_id,
        owner_id=user.owner_id,
        multi_query=multi_query,
    )
    return SearchResponse(
        query=q,
        hits=[
            SearchHitOut(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                filename=h.filename,
                page=h.page,
                chunk_index=h.chunk_index,
                heading=h.heading,
                text=h.text,
                score=h.score,
                meta=h.meta,
                rrf_score=h.rrf_score,
                found_by=h.found_by,
            )
            for h in hits
        ],
    )


@router.get("/stats", response_model=StatsOut)
async def stats(user: User = Depends(current_user)) -> StatsOut:
    """Cross-check Postgres against Qdrant. A mismatch means a failed ingest.

    Postgres counts are per-owner; the Qdrant count is collection-wide, so the
    two only line up in single-user mode. Worth knowing before reading a
    mismatch as a failed ingest.
    """
    settings = get_settings()
    async with SessionLocal() as session:
        doc_q = select(func.count()).select_from(Document)
        chunk_q = (
            select(func.count())
            .select_from(Chunk)
            .join(Document, Document.id == Chunk.document_id)
        )
        if user.owner_id is not None:
            doc_q = doc_q.where(Document.owner_id == user.owner_id)
            chunk_q = chunk_q.where(Document.owner_id == user.owner_id)

        n_docs = await session.scalar(doc_q) or 0
        n_chunks = await session.scalar(chunk_q) or 0

    try:
        n_vectors = await get_vector_store().count()
    except Exception:
        n_vectors = 0  # collection not created until the first ingest

    return StatsOut(
        documents=n_docs,
        chunks_in_postgres=n_chunks,
        vectors_in_qdrant=n_vectors,
        embedding_provider=settings.embedding_provider,
        embedding_dim=get_embeddings().dim,
    )

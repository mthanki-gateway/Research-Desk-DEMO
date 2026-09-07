import uuid

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy import func, select

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
from app.services.vectorstore import get_vector_store

log = structlog.get_logger()

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
) -> Document:
    """Accept a file, return immediately, embed in the background.

    Embedding is rate-limited to ~133 chunks/minute, so holding the request
    open would time out any realistic proxy. The client polls GET /documents.
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

    async with SessionLocal() as session:
        doc = Document(
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(data),
            status=DocStatus.pending,
            owner_id=user.owner_id,
        )
        session.add(doc)
        await session.commit()
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

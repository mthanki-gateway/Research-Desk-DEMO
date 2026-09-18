"""Ingestion pipeline, as explicit stages.

    bytes -> [parse] -> pages -> [chunk] -> chunks -> [enrich] -> [embed] -> [store]

Each stage is a plain function with typed in/out and no knowledge of its
neighbours. That's deliberate: the planned LLM categorisation step is an
`Enricher` appended to ENRICHERS below, and nothing else in the file changes.
Enrichers must NOT call the embedding API -- they get their own rate limiter,
because generation and embedding are separate quotas.

Runs as a background task because embedding is slow on the free tier (~133
chunks/minute), so a large PDF takes minutes. The upload request returns
immediately with a document id; the client polls for progress.
"""

from __future__ import annotations

import uuid
from typing import Protocol

import structlog
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chunk, DocStatus, Document
from app.db.session import SessionLocal
from app.services.chunking import TextChunk, chunk_pages
from app.services.embeddings import get_embeddings
from app.services.lexical import invalidate_lexical_index
from app.services.parsing import Page, parse
from app.services.storage import get_storage, key_for
from app.services.vectorstore import ChunkPayload, get_vector_store

log = structlog.get_logger()

# Vectors are written to Qdrant in slices of this many, so n_embedded advances
# steadily in the UI instead of jumping from 0 to done.
FLUSH_EVERY = 100


class Enricher(Protocol):
    """Adds document-level metadata before embedding.

    A Protocol, not a base class: any object with a matching signature
    qualifies, so an enricher needs no import from this module.

    The returned dict is merged into Document.meta and copied onto every one of
    that document's vector payloads, which is what makes it filterable later.
    """

    async def __call__(self, *, filename: str, pages: list[Page], chunks: list[TextChunk]) -> dict:
        ...


# Empty by design. LLM categorisation will append here.
ENRICHERS: list[Enricher] = []


async def ingest_document(document_id: uuid.UUID, data: bytes) -> None:
    """Entry point for the background task. Never raises -- records failure."""
    settings = get_settings()
    log_ = log.bind(document_id=str(document_id))

    try:
        async with SessionLocal() as session:
            doc = await session.get(Document, document_id)
            if doc is None:
                log_.error("ingest_document_missing")
                return

            # --- stage 0: keep the original ---
            #
            # BEFORE parsing, because parsing is the step most likely to fail
            # and the original is exactly what somebody needs in order to work
            # out why. Storing it also makes re-chunking a background job
            # rather than an apology: improve the chunker and every document
            # can be re-ingested without anybody re-uploading anything.
            #
            # NEVER FATAL. A store that is full, misconfigured or unreachable
            # costs the archival copy; it must not cost the indexing, because
            # the text is what answers questions.
            try:
                store = get_storage()
                doc.storage_key = await store.put(
                    key_for(doc.owner_id, doc.id, doc.filename),
                    data,
                    doc.content_type,
                )
                await session.commit()
            except Exception as exc:  # noqa: BLE001
                log_.warning("ingest_store_failed", error=str(exc)[:200])

            # --- stage 1: parse ---
            doc.status = DocStatus.parsing
            await session.commit()
            pages = parse(data, doc.filename, doc.content_type)

            # --- stage 2: chunk ---
            chunks = chunk_pages(
                pages,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                min_chars=settings.min_chunk_chars,
            )
            if not chunks:
                raise ValueError("Document produced no chunks.")

            doc.n_pages = len(pages)
            doc.n_chunks = len(chunks)
            await session.commit()
            log_.info("chunked", pages=len(pages), chunks=len(chunks))

            # --- stage 3: enrich (no-op until an enricher is registered) ---
            meta: dict = dict(doc.meta or {})
            for enricher in ENRICHERS:
                try:
                    meta.update(
                        await enricher(filename=doc.filename, pages=pages, chunks=chunks)
                    )
                except Exception:
                    # Enrichment is additive, never load-bearing. A failing
                    # categoriser must not cost us the document.
                    log_.exception("enricher_failed", enricher=type(enricher).__name__)
            doc.meta = meta
            await session.commit()

            # Persist chunk rows first so their UUIDs can be reused as Qdrant
            # point ids -- one identity across both stores.
            rows = [
                Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    chunk_index=c.index,
                    page=c.page,
                    heading=c.heading,
                    text=c.text,
                    n_chars=len(c.text),
                )
                for c in chunks
            ]
            session.add_all(rows)

            doc.status = DocStatus.embedding
            await session.commit()

            # --- stages 4 + 5: embed and store, in windows ---
            embeddings = get_embeddings()
            store = get_vector_store()
            await store.ensure_collection()

            created_at = doc.created_at.isoformat()

            for start in range(0, len(rows), FLUSH_EVERY):
                window = rows[start : start + FLUSH_EVERY]
                texts = [r.text for r in window]

                vectors = await embeddings.embed_documents(texts)

                await store.upsert(
                    payloads=[
                        ChunkPayload(
                            chunk_id=r.id,
                            document_id=document_id,
                            filename=doc.filename,
                            content_type=doc.content_type,
                            page=r.page,
                            chunk_index=r.chunk_index,
                            heading=r.heading,
                            text=r.text,
                            n_chars=r.n_chars,
                            created_at=created_at,
                            owner_id=doc.owner_id,
                            meta=meta,
                        )
                        for r in window
                    ],
                    vectors=vectors,
                )

                doc.n_embedded = start + len(window)
                await session.commit()
                log_.info("embedded_progress", done=doc.n_embedded, total=len(rows))

            doc.status = DocStatus.ready
            doc.error = None
            await session.commit()
            # The BM25 index is built from these rows and cached per scope, so
            # without this a document is searchable by vector the moment it is
            # ready and invisible to lexical search until the process restarts.
            # That asymmetry would be near-impossible to spot: hybrid results
            # would simply be a little worse for the newest document.
            invalidate_lexical_index()
            log_.info("ingest_complete", chunks=len(rows))

    except Exception as exc:
        log_.exception("ingest_failed")
        await _mark_failed(document_id, str(exc))


async def _mark_failed(document_id: uuid.UUID, message: str) -> None:
    """Separate session: the one that failed may be in a broken transaction."""
    try:
        async with SessionLocal() as session:
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = DocStatus.failed
                doc.error = message[:2000]
                await session.commit()
    except Exception:  # pragma: no cover
        log.exception("could_not_mark_failed", document_id=str(document_id))


async def delete_document(document_id: uuid.UUID) -> bool:
    """Remove from both stores. Qdrant first: an orphaned vector would surface
    in search results with no row to render, whereas an orphaned row is inert.
    """
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return False
        storage_key = doc.storage_key

        await get_vector_store().delete_document(document_id)
        await session.delete(doc)  # chunks cascade
        await session.commit()

        # AND THE ORIGINAL. Without this every delete leaks a file: the row and
        # its vectors go, and the only record of where the bytes were goes with
        # them, so nothing can ever find them again. On a 1GB bucket that is a
        # quota filling up with objects no query can name.
        #
        # AFTER the commit, and never fatal. The row is the thing that makes a
        # document exist; an object store that is briefly unreachable must not
        # leave a document half-deleted and still listed.
        if storage_key:
            try:
                await get_storage().delete(storage_key)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "delete_stored_file_failed",
                    key=storage_key,
                    error=str(exc)[:200],
                )
        # A stale lexical index would keep serving chunk ids for a deleted
        # document. They no longer resolve to a dense hit so nothing would be
        # shown -- but the BM25 half would be silently scoring against text that
        # is gone, which is a worse bug than it looks: deletion should mean
        # deletion everywhere.
        invalidate_lexical_index()
        return True


async def list_documents(owner_id: str | None = None) -> list[Document]:
    """Owner's documents, or all of them when owner_id is None (auth disabled)."""
    async with SessionLocal() as session:
        query = select(Document).order_by(Document.created_at.desc())
        if owner_id is not None:
            query = query.where(Document.owner_id == owner_id)
        result = await session.execute(query)
        return list(result.scalars())

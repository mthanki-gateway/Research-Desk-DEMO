"""Qdrant access. The only module that knows the vector DB exists."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings

log = structlog.get_logger()


@dataclass(slots=True)
class ChunkPayload:
    """Everything stored alongside one vector.

    Deliberately denormalised: retrieval must be able to render a full citation
    (file, section, page) straight from the search result, without a second
    round-trip to Postgres for every hit.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content_type: str
    page: int | None
    chunk_index: int
    heading: str | None
    text: str
    n_chars: int
    created_at: str
    owner_id: str | None
    meta: dict


@dataclass(slots=True)
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    heading: str | None
    text: str
    score: float
    meta: dict
    # Set only when results came from fused rankings (multi-query, later
    # hybrid). Kept separate from `score` because an RRF score is on a
    # completely different scale to cosine similarity -- around 0.03 vs 0.7 --
    # and conflating them would make the UI's numbers meaningless.
    rrf_score: float | None = None
    # Which query variations found this chunk. A chunk found by several is
    # strong evidence; this is also how you see whether rewriting helped.
    found_by: list[str] | None = None

    # --- provenance ---
    # "document" for a retrieved chunk, "web" for a search result.
    #
    # Web results reuse this dataclass rather than getting their own type, and
    # that is deliberate: `merge_evidence`, `build_context`, RRF, citation
    # numbering and message persistence all take `list[SearchHit]`, so a second
    # type would mean touching every one of them. A web result fills
    # `filename` with the page title, `text` with the snippet, and carries a
    # synthetic `chunk_id` so dedupe-by-id keeps working.
    source: str = "document"
    # Set only for web results. The UI opens this instead of the chunk panel,
    # because there is no chunk to open.
    url: str | None = None


class VectorStore:
    def __init__(self, dim: int) -> None:
        s = get_settings()
        self._collection = s.qdrant_collection
        self._dim = dim
        self._client = AsyncQdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key or None,
            prefer_grpc=False,
        )

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)

        if exists:
            info = await self._client.get_collection(self._collection)
            actual = info.config.params.vectors.size
            if actual != self._dim:
                # Collections are fixed-dimension. Switching embedding provider
                # or EMBEDDING_DIM without re-ingesting would otherwise fail
                # deep inside an upsert with an opaque error.
                raise RuntimeError(
                    f"Qdrant collection '{self._collection}' has dim {actual}, but the "
                    f"embedding provider produces {self._dim}. Delete the collection and "
                    f"re-ingest: docker compose exec api python -m app.scripts.reset_vectors"
                )
        else:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dim,
                    distance=models.Distance.COSINE,
                ),
            )
            log.info("collection_created", name=self._collection, dim=self._dim)

        # Payload indexes make filtered search cheap. Without them Qdrant
        # full-scans the payload, which also makes per-document deletion slow
        # as the collection grows.
        #
        # meta.* fields are intentionally NOT indexed yet: Qdrant needs to know
        # the field type, and nothing writes them until an enricher does. Add
        # an index here when categorisation lands.
        for field in ("document_id", "owner_id", "filename"):
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    async def upsert(
        self, payloads: list[ChunkPayload], vectors: list[list[float]]
    ) -> None:
        """One vector per payload, in order.

        strict=True on zip catches a length mismatch loudly. Silently pairing
        the wrong vector with the wrong text would produce a store that looks
        healthy and returns subtly wrong results forever.
        """
        points = [
            models.PointStruct(
                id=str(p.chunk_id),
                vector=vector,
                payload={
                    "document_id": str(p.document_id),
                    "filename": p.filename,
                    "content_type": p.content_type,
                    "page": p.page,
                    "chunk_index": p.chunk_index,
                    "heading": p.heading,
                    "text": p.text,
                    "n_chars": p.n_chars,
                    "created_at": p.created_at,
                    "owner_id": p.owner_id,
                    # Nested dict. Qdrant filters nested keys with dotted
                    # paths, so a future filter is `meta.category`.
                    "meta": p.meta,
                },
            )
            for p, vector in zip(payloads, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[uuid.UUID] | None = None,
        owner_id: str | None = None,
        meta_filters: dict[str, str | list[str]] | None = None,
    ) -> list[SearchHit]:
        """Vector search with optional payload filters.

        Qdrant applies filters *during* the ANN traversal, not as a post-filter,
        so asking for the top 5 within one document really does return that
        document's 5 best chunks -- not the global top 5 filtered down to
        whatever survives.
        """
        conditions: list[models.FieldCondition] = []

        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=[str(d) for d in document_ids]),
                )
            )
        if owner_id:
            conditions.append(
                models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))
            )
        for key, value in (meta_filters or {}).items():
            match = (
                models.MatchAny(any=value)
                if isinstance(value, list)
                else models.MatchValue(value=value)
            )
            conditions.append(models.FieldCondition(key=f"meta.{key}", match=match))

        query_filter = models.Filter(must=conditions) if conditions else None

        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: list[SearchHit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=uuid.UUID(str(point.id)),
                    document_id=uuid.UUID(payload["document_id"]),
                    filename=payload.get("filename", ""),
                    page=payload.get("page"),
                    chunk_index=payload.get("chunk_index", 0),
                    heading=payload.get("heading"),
                    text=payload.get("text", ""),
                    score=point.score,
                    meta=payload.get("meta") or {},
                )
            )
        return hits

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def claim_unowned(self, owner_id: str) -> int:
        """Assign every ownerless vector to `owner_id`.

        Counterpart to the Postgres update in api/auth.py. Vectors carry
        owner_id in their payload rather than joining to a table, so enabling
        auth would otherwise make pre-auth documents invisible to search while
        their rows still existed -- a confusing half-state.

        `IsEmptyCondition` is the Qdrant way to match a missing-or-null payload
        field; a MatchValue against None does not work.
        """
        unowned = models.Filter(
            must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="owner_id"))]
        )
        before = await self._client.count(
            collection_name=self._collection, count_filter=unowned, exact=True
        )
        if before.count == 0:
            return 0

        await self._client.set_payload(
            collection_name=self._collection,
            payload={"owner_id": owner_id},
            points=models.FilterSelector(filter=unowned).filter,
            wait=True,
        )
        log.info("claimed_vectors", owner_id=owner_id, n=before.count)
        return before.count

    async def all_vectors(
        self, owner_id: str | None, *, limit: int = 5_000
    ) -> list[tuple[list[float], dict]]:
        """Every stored vector with its payload, for whole-corpus analysis.

        `scroll`, not `search`: there is no query here. Search would need a
        probe vector and would return results ranked by distance from it,
        which is the wrong shape for "show me everything" -- and on an HNSW
        index it is also approximate, so some points would simply never
        appear.

        Capped because the caller projects and cross-multiplies these in
        memory: a similarity matrix is O(n^2), so 5,000 chunks is already 25
        million cosines. Far past the point where a scatter plot means
        anything to look at, but a limit beats an unbounded allocation.
        """
        points: list[tuple[list[float], dict]] = []
        offset = None
        while len(points) < limit:
            batch, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="owner_id",
                                match=models.MatchValue(value=owner_id),
                            )
                        ]
                    )
                    if owner_id
                    else None
                ),
                # The whole reason for this method. Payload alone would give
                # the labels and none of the geometry.
                with_vectors=True,
                with_payload=True,
                limit=min(256, limit - len(points)),
                offset=offset,
            )
            points.extend(
                # The point id IS the chunk id (see `id=str(p.chunk_id)` in
                # upsert) and it is NOT repeated in the payload, so dropping it
                # here left every caller with no way to say which chunk a
                # vector belonged to. The Atlas has been shipping empty
                # chunk_ids because of it -- invisible, because nothing
                # downstream used them until the query ray needed to match a
                # stored citation against a plotted point.
                (list(pt.vector), {**dict(pt.payload or {}), "chunk_id": str(pt.id)})
                for pt in batch
                if pt.vector is not None
            )
            if offset is None:
                break
        return points

    async def count(self) -> int:
        result = await self._client.count(collection_name=self._collection, exact=True)
        return result.count

    async def drop_collection(self) -> None:
        await self._client.delete_collection(self._collection)

    async def aclose(self) -> None:
        await self._client.close()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        from app.services.embeddings import get_embeddings

        _store = VectorStore(dim=get_embeddings().dim)
    return _store


async def close_vector_store() -> None:
    global _store
    if _store is not None:
        await _store.aclose()
        _store = None

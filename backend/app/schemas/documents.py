import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import DocStatus


class DocumentOut(BaseModel):
    """The API contract for a document. from_attributes lets it read an ORM row."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocStatus
    error: str | None
    n_pages: int
    n_chunks: int
    n_embedded: int
    created_at: datetime
    owner_id: str | None
    meta: dict

    @property
    def progress(self) -> float:
        if self.n_chunks == 0:
            return 0.0
        return self.n_embedded / self.n_chunks


class ChunkOut(BaseModel):
    """A stored chunk, for citation drill-down and library inspection."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page: int | None
    heading: str | None
    text: str
    n_chars: int
    # Joined in, so the UI can render a citation without a second request.
    filename: str = ""


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    heading: str | None
    text: str
    score: float
    meta: dict
    # Only populated for fused (multi-query) results.
    rrf_score: float | None = None
    found_by: list[str] | None = None
    # "document" or "web". The UI opens a chunk for the former and the url
    # for the latter, because a web result has no chunk to open.
    source: str = "document"
    url: str | None = None

    @property
    def citation(self) -> str:
        """Human-readable source, e.g. 'acme-report.md · ## Risks · p.3'."""
        parts = [self.filename]
        if self.heading:
            parts.append(self.heading)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " · ".join(parts)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitOut]


class StatsOut(BaseModel):
    documents: int
    chunks_in_postgres: int
    vectors_in_qdrant: int
    embedding_provider: str
    embedding_dim: int

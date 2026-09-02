import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Role
from app.schemas.documents import SearchHitOut


class SessionCreate(BaseModel):
    title: str | None = Field(None, max_length=256)
    # Empty or omitted = search all documents.
    document_ids: list[uuid.UUID] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    """All optional -- PATCH semantics. None means 'leave unchanged'."""

    title: str | None = Field(None, max_length=256)
    document_ids: list[uuid.UUID] | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Role
    content: str
    sources: list = Field(default_factory=list)
    agent_meta: dict = Field(default_factory=dict)
    created_at: datetime


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    document_ids: list[uuid.UUID]
    summary: str | None
    summarised_upto: int
    owner_id: str | None
    created_at: datetime
    updated_at: datetime
    n_messages: int = 0


class SessionDetail(SessionOut):
    messages: list[MessageOut] = Field(default_factory=list)


class TurnRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    multi_query: bool | None = None
    # Overrides the session's scope for this one turn only.
    document_ids: list[uuid.UUID] | None = None


class TurnResponse(BaseModel):
    session_id: uuid.UUID
    question: str
    answer: str
    sources: list[SearchHitOut]
    sources_used: list[int]
    sub_questions: list[str]
    critique: str
    sufficient: bool
    iterations: int
    trace: list[dict]
    # Visibility into what history was actually sent, since that's the thing
    # you'll want to inspect when tuning token usage.
    context_chars: int

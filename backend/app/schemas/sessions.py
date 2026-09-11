import uuid
from datetime import datetime
from typing import Literal

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
    # None = use AGENT_CLARIFY from config. Explicit true lets this turn pause
    # to ask what the request means, if the request is too vague to search on.
    clarify: bool | None = None
    # None = use REACT_DEFAULT. True gathers evidence with the tool-calling
    # loop (multi-hop, can search the web) instead of planning every lookup
    # up front.
    react: bool | None = None
    # DEVELOPMENT ONLY -- ignored entirely unless APP_ENV=dev.
    #
    # Run this one turn on a different model family. "gemma" trades quality for
    # a 30 rpm / 14,400-per-day budget, which is the only way to exercise the
    # agent loop repeatedly; the answer model's 5 rpm allows about one question
    # every two minutes.
    #
    # Ignored rather than rejected in prod: it is a developer convenience, and a
    # client that sends it should get a normal answer on the configured model
    # rather than a 400 telling it the field exists.
    model_profile: Literal["gemini", "gemma"] | None = None


class ResumeRequest(BaseModel):
    """A user's answer to a paused graph.

    `thread_id` is required and comes from the paused turn's response. It is
    the checkpoint key -- without it there is nothing to resume, and it is why
    a pause survives across requests, workers and deploys.
    """

    thread_id: str = Field(..., min_length=1, max_length=200)
    # answer = use `answer` to narrow the search
    # skip   = search the original request as written
    # cancel = stop without searching
    action: Literal["answer", "skip", "cancel"] = "skip"
    # Only read when action == "answer". Either a chosen option's label or
    # whatever the user typed into the free-text box -- the graph treats both
    # identically, which is what keeps "something else" from being a special
    # case.
    answer: str = Field("", max_length=500)
    # Echoed back so the resumed turn is persisted against the right question;
    # the graph also holds it, but re-sending keeps the endpoint self-contained.
    question: str = Field(..., min_length=1, max_length=2000)


class ClarifyOption(BaseModel):
    label: str
    description: str = ""


class InterruptOut(BaseModel):
    """What the graph is waiting for. Present only on a paused turn."""

    type: str
    # The clarifying question, written by the model.
    question: str = ""
    # Concrete choices, grounded in the headings the documents actually have.
    options: list[ClarifyOption] = Field(default_factory=list)
    # What the user originally typed, so the UI can show it alongside.
    original: str = ""
    actions: list[str] = Field(default_factory=list)


class TurnResponse(BaseModel):
    session_id: uuid.UUID
    question: str
    answer: str
    sources: list[SearchHitOut]
    sources_used: list[int]
    sub_questions: list[str]
    critique: str
    sufficient: bool
    # The answer is knowingly incomplete: `resolve` kept what the sources
    # supported and named the gap. Distinct from `sufficient`, which is True by
    # then precisely because resolve produced the final answer.
    partial: bool = False
    iterations: int
    trace: list[dict]
    # Visibility into what history was actually sent, since that's the thing
    # you'll want to inspect when tuning token usage.
    context_chars: int
    # --- human-in-the-loop ---
    # Non-null means the turn is NOT finished: it is parked at a checkpoint
    # waiting for POST /resume. Nothing has been persisted and no answer
    # exists yet.
    interrupt: InterruptOut | None = None
    # The checkpoint key to resume with. Only returned while paused -- a
    # completed turn deletes its thread, so the id would be a dangling
    # reference.
    thread_id: str | None = None
    # What the user said when asked to clarify. Null on an ordinary turn.
    clarification: str | None = None
    # Langfuse trace id, so the client can attach feedback to this turn later.
    # Null when tracing is not configured.
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    """A thumbs up/down on one assistant message.

    Identifies the turn by MESSAGE id, not trace id. The trace id is looked up
    server-side from the stored message -- accepting one from the client would
    let anyone score any trace, including another tenant's.
    """

    message_id: str = Field(..., min_length=1, max_length=64)
    helpful: bool
    comment: str | None = Field(None, max_length=1000)

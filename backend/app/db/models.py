import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocStatus(enum.StrEnum):
    pending = "pending"
    parsing = "parsing"
    embedding = "embedding"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)

    status: Mapped[DocStatus] = mapped_column(
        Enum(DocStatus, name="doc_status"), default=DocStatus.pending, index=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Reserved for auth (Firebase/Clerk/etc). Nullable and unused today, but
    # adding the column now costs nothing and avoids a migration later --
    # every vector already carries it, so per-user filtering will work without
    # re-ingesting anything.
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Free-form document-level metadata, copied onto every vector's payload.
    # This is the extension point for LLM-based categorisation: an enricher
    # writes {"category": "...", "tags": [...]} here and retrieval can filter
    # on it without any schema change.
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")

    # Progress counters so the UI can show "142 / 300 chunks embedded" instead of
    # a spinner. Ingestion takes minutes on the free tier, so this matters.
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_embedded: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Chunk text lives here AND in the Qdrant payload, deliberately.

    Qdrant answers "what is semantically near this vector"; Postgres answers
    "give me chunk 7 of document X" and, in step 4, supports lexical/BM25
    search for hybrid retrieval. Duplicating a few KB of text now avoids a
    full re-ingest later.
    """

    __tablename__ = "chunks"

    # Same UUID is used as the Qdrant point id, so the two stores stay linked
    # without a translation table.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The section heading this chunk came from, e.g. "## Risks". Stored so a
    # citation can say *where* in the document a claim came from, not just
    # which file.
    heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    n_chars: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChatSession(Base):
    """A research conversation.

    Named ChatSession, not Session, to avoid confusion with SQLAlchemy's
    Session -- a genuine footgun when both are imported in one module.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), default="New session")

    # Which documents this session may search. Empty list = all of them.
    # This is the "scope" control, and it feeds straight into the
    # document_ids filter VectorStore.search already supports.
    document_ids: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")

    # Rolling summary of turns already evicted from the verbatim window, plus
    # how many messages it covers. Gemma is 16K tokens/MINUTE, so resending a
    # long history is not merely expensive -- past ~30 turns it cannot be sent
    # inside one minute at all.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summarised_upto: Mapped[int] = mapped_column(Integer, default=0)

    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Role(enum.StrEnum):
    user = "user"
    assistant = "assistant"


class Message(Base):
    """One turn.

    Note the deliberate duplication with LangGraph's checkpointer: the
    checkpointer persists *graph state* so a run can resume mid-execution,
    while this table is the queryable transcript for display and for building
    prompt context. Reading history out of checkpoint blobs would mean coupling
    the UI to LangGraph's internal state shape.

    Only the question and the answer are stored -- never the retrieved chunk
    text. Re-embedding a query is cheap (100 RPM); generation tokens are the scarce
    resource, so replaying stored context into every later turn is the one
    thing that must not happen.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[Role] = mapped_column(Enum(Role, name="message_role"))
    content: Mapped[str] = mapped_column(Text)

    # Assistant-only provenance: citation numbers, the sources they point at,
    # and how the agent got there. JSONB so the shape can evolve without a
    # migration.
    sources: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    agent_meta: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")

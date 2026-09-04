import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.agent.checkpointer import discard_thread
from app.agent.graph import AgentResult, run_agent, stream_agent
from app.auth import User, current_user, forbid_if_not_owner
from app.db.models import ChatSession, Message, Role
from app.db.session import SessionLocal
from app.schemas.documents import SearchHitOut
from app.schemas.sessions import (
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionOut,
    SessionUpdate,
    TurnRequest,
    TurnResponse,
)
from app.services.history import build_chat_context, update_summary
from app.services.llm import LLMError
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    req: SessionCreate, user: User = Depends(current_user)
) -> SessionOut:
    async with SessionLocal() as db:
        chat = ChatSession(
            title=req.title or "New session",
            document_ids=[str(d) for d in req.document_ids],
            owner_id=user.owner_id,
        )
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return _session_out(chat, 0)


@router.get("", response_model=list[SessionOut])
async def list_sessions(user: User = Depends(current_user)) -> list[SessionOut]:
    async with SessionLocal() as db:
        # One grouped query for the counts rather than N+1 lazy loads.
        counts = dict(
            (
                await db.execute(
                    select(Message.session_id, func.count()).group_by(Message.session_id)
                )
            ).all()
        )
        query = select(ChatSession).order_by(ChatSession.updated_at.desc())
        if user.owner_id is not None:
            query = query.where(ChatSession.owner_id == user.owner_id)
        rows = (await db.execute(query)).scalars()
        return [_session_out(s, counts.get(s.id, 0)) for s in rows]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID, user: User = Depends(current_user)
) -> SessionDetail:
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        return SessionDetail(
            **_session_out(chat, len(chat.messages)).model_dump(),
            messages=[MessageOut.model_validate(m) for m in chat.messages],
        )


@router.patch("/{session_id}", response_model=SessionOut)
async def update_session(
    session_id: uuid.UUID, req: SessionUpdate, user: User = Depends(current_user)
) -> SessionOut:
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        if req.title is not None:
            chat.title = req.title
        if req.document_ids is not None:
            chat.document_ids = [str(d) for d in req.document_ids]
        await db.commit()
        await db.refresh(chat)
        return _session_out(chat, len(chat.messages))


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        await db.delete(chat)  # messages cascade
        await db.commit()


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------


@router.post("/{session_id}/messages", response_model=TurnResponse)
async def add_turn(
    session_id: uuid.UUID, req: TurnRequest, user: User = Depends(current_user)
) -> TurnResponse:
    """Ask a question inside a session. Blocking; see /stream for progress."""
    prep = await _prepare_turn(session_id, req, user)

    try:
        result = await run_agent(
            req.question,
            top_k=req.top_k,
            document_ids=prep["scope"],
            owner_id=user.owner_id,
            multi_query=req.multi_query,
            chat_context=prep["context"],
            thread_id=prep["thread_id"],
        )
    except LLMError as exc:
        log.warning("turn_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Model error: {exc}") from exc

    await _persist_turn(session_id, req.question, result, prep, user)

    return TurnResponse(
        session_id=session_id,
        question=req.question,
        answer=result.answer,
        sources=[_hit_out(h) for h in result.evidence],
        sources_used=result.citations,
        sub_questions=result.sub_questions,
        critique=result.critique,
        sufficient=result.sufficient,
        iterations=result.iterations,
        trace=result.trace,
        context_chars=len(prep["context"]),
    )


@router.post("/{session_id}/stream")
async def stream_turn(
    session_id: uuid.UUID, req: TurnRequest, user: User = Depends(current_user)
) -> StreamingResponse:
    """Same as /messages, but emits SSE progress as each node completes.

    Node-level, not token-level: with responseSchema output there is no partial
    prose to stream. Events: `progress` per node, then one `done` with the full
    result, or `error`.
    """
    # Ownership resolves BEFORE the response starts streaming, so an
    # unauthorised caller gets a clean 404. Once the stream opens the status
    # code is already sent and a failure can only travel as an event.
    prep = await _prepare_turn(session_id, req, user)

    async def events() -> AsyncIterator[str]:
        final_state: dict = {}
        try:
            async for kind, node, payload in stream_agent(
                req.question,
                top_k=req.top_k,
                document_ids=prep["scope"],
                # MUST be passed. `stream_agent` defaults owner_id to None, and
                # None means "do not filter by owner" in the vector store -- so
                # omitting it here (as this call did) made a streamed turn
                # search EVERY user's chunks. The document scope masked it
                # whenever a session had documents selected, but a session with
                # no scope resolves to `scope=None`, and then nothing constrained
                # retrieval at all. /messages passed it; /stream did not, and
                # /stream is the path the UI uses.
                owner_id=user.owner_id,
                multi_query=req.multi_query,
                chat_context=prep["context"],
                thread_id=prep["thread_id"],
            ):
                if kind == "state":
                    final_state = payload
                    continue
                yield _sse("progress", {"node": node, "detail": _describe(node, payload)})

            result = AgentResult(final_state)
            await _persist_turn(session_id, req.question, result, prep, user)
            yield _sse(
                "done",
                {
                    "answer": result.answer,
                    "sources": [_hit_out(h).model_dump(mode="json") for h in result.evidence],
                    "sources_used": result.citations,
                    "sub_questions": result.sub_questions,
                    "critique": result.critique,
                    "sufficient": result.sufficient,
                    "iterations": result.iterations,
                    "trace": result.trace,
                    "context_chars": len(prep["context"]),
                },
            )
        except Exception as exc:
            # The response has already started, so an HTTP error code is no
            # longer available -- the failure has to travel as an event.
            log.exception("stream_turn_failed")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Render and other proxies buffer by default, which would hold the
            # whole stream until completion and defeat the point.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _load(db, session_id: uuid.UUID, user: User) -> ChatSession:
    """Load a session, or 404 if it does not exist OR is not the caller's.

    The ownership check lives here rather than at each call site so that no
    endpoint can forget it -- every session read goes through this function.
    """
    chat = (
        await db.execute(
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
    ).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    forbid_if_not_owner(chat.owner_id, user)
    return chat


async def _prepare_turn(
    session_id: uuid.UUID, req: TurnRequest, user: User
) -> dict:
    """Resolve scope and build the history context, before any LLM work."""
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        messages = list(chat.messages)

        # Fold anything newly evicted from the verbatim window into the summary
        # BEFORE building context, so this turn sees an up-to-date summary.
        summary, upto = await update_summary(chat, messages)
        if summary != (chat.summary or "") or upto != chat.summarised_upto:
            chat.summary = summary or None
            chat.summarised_upto = upto
            await db.commit()
            await db.refresh(chat)

        context = build_chat_context(chat, messages)

        # Per-turn override wins; otherwise the session's scope. Empty = all.
        if req.document_ids is not None:
            scope = req.document_ids or None
        else:
            scope = [uuid.UUID(d) for d in chat.document_ids] or None

        # One thread per ATTEMPT, not per session and not per turn.
        #
        # Per-session was the original bug: append reducers made evidence and
        # sub-questions pile up across turns, letting an earlier question's
        # chunks contaminate a later answer.
        #
        # Per-turn was still wrong, because a FAILED turn persists no messages,
        # so a retry computes the same turn index and the same thread. Measured:
        # a second attempt on one thread duplicated sub_questions and trace,
        # and polluted tried_queries -- which makes critique refuse to re-run a
        # query it thinks was already attempted. The random suffix guarantees
        # each attempt starts clean; the thread is deleted once the answer is
        # persisted.
        turn = len(messages) // 2
        thread_id = f"{session_id}:{turn}:{uuid.uuid4().hex[:8]}"

        return {
            "context": context,
            "scope": scope,
            "was_empty": not messages,
            "thread_id": thread_id,
        }


async def _persist_turn(
    session_id: uuid.UUID,
    question: str,
    result: AgentResult,
    prep: dict,
    user: User,
) -> None:
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)

        db.add(Message(session_id=session_id, role=Role.user, content=question))
        db.add(
            Message(
                session_id=session_id,
                role=Role.assistant,
                content=result.answer,
                # Citation targets only -- never the chunk text. Storing that
                # would re-send retrieved context on every later turn, which is
                # the fastest way to blow the token budget.
                sources=[
                    {
                        "n": i + 1,
                        "chunk_id": str(h.chunk_id),
                        "document_id": str(h.document_id),
                        "filename": h.filename,
                        "heading": h.heading,
                        "page": h.page,
                        "score": round(h.score, 4),
                    }
                    for i, h in enumerate(result.evidence)
                ],
                agent_meta={
                    "sources_used": result.citations,
                    "sub_questions": result.sub_questions,
                    "iterations": result.iterations,
                    "sufficient": result.sufficient,
                    "critique": result.critique,
                },
            )
        )

        # Title the session from its first question, so the list is readable
        # without asking the user to name anything.
        if prep.get("was_empty") and chat.title == "New session":
            chat.title = question[:80] + ("..." if len(question) > 80 else "")

        await db.commit()

    # The turn is durable now, so its checkpoints are dead weight. Deleting
    # here is what keeps the checkpoint tables from growing without bound.
    thread_id = prep.get("thread_id")
    if thread_id:
        await discard_thread(thread_id)


def _session_out(chat: ChatSession, n_messages: int) -> SessionOut:
    return SessionOut(
        id=chat.id,
        title=chat.title,
        document_ids=[uuid.UUID(d) for d in (chat.document_ids or [])],
        summary=chat.summary,
        summarised_upto=chat.summarised_upto,
        owner_id=chat.owner_id,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        n_messages=n_messages,
    )


def _hit_out(h: SearchHit) -> SearchHitOut:
    return SearchHitOut(
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _describe(node: str, payload: dict) -> str:
    """Human-readable one-liner per node, for the progress display."""
    if node == "plan":
        subs = payload.get("sub_questions") or []
        return f"planned {len(subs)} sub-question" + ("" if len(subs) == 1 else "s")
    if node == "retrieve":
        trace = (payload.get("trace") or [{}])[-1]
        queries = trace.get("queries") or []
        total = sum(q.get("n", 0) for q in queries)
        return f"retrieved {total} chunks across {len(queries)} queries"
    if node == "draft":
        cited = payload.get("citations") or []
        return f"drafted, cited {len(cited)} sources"
    if node == "critique":
        if payload.get("sufficient"):
            return "critique passed"
        missing = payload.get("missing") or []
        return f"critique found {len(missing)} gap(s), retrying"
    return node


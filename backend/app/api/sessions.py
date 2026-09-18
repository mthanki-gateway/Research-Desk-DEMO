import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from app.agent.checkpointer import discard_thread
from app.agent.graph import (
    AgentResult,
    resume_agent,
    run_agent,
    stream_agent,
    stream_resume,
)
from app.auth import User, current_user, forbid_if_not_owner
from app.config import get_settings, use_model_profile
from app.db.models import ChatSession, Message, Role
from app.db.session import SessionLocal
from app.schemas.documents import SearchHitOut
from app.schemas.sessions import (
    FeedbackRequest,
    InterruptOut,
    MessageOut,
    ResumeRequest,
    SessionCreate,
    SessionDetail,
    SessionOut,
    SessionUpdate,
    TurnRequest,
    TurnResponse,
)
from app.services import preferences, tracing
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
async def list_sessions(
    response: Response,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(current_user),
) -> list[SessionOut]:
    """Research Desk's chats. NOT Parley's conversations.

    FILTERED BY KIND, which it was not. Every table in this app stores its
    conversations in `chat_sessions`, and Parley's three modes are rows there
    too -- so spoken conversations were appearing in the Research Desk sidebar,
    where opening one lands on a chat page for something that was never typed.
    They are separate apps and this is the line between them.

    PAGINATED, because this list grows for ever and had no ceiling: every chat
    anybody had ever started was fetched, counted and serialised on every page
    load. The total goes in a header so the response stays a plain array and
    existing callers are unaffected.
    """
    async with SessionLocal() as db:
        owned = select(ChatSession).where(
            # `IS NULL` as well: rows predating the column are Research Desk's,
            # because Parley did not exist when they were written.
            or_(ChatSession.kind == "chat", ChatSession.kind.is_(None))
        )
        if user.owner_id is not None:
            owned = owned.where(ChatSession.owner_id == user.owner_id)

        total = (
            await db.execute(
                select(func.count()).select_from(owned.subquery())
            )
        ).scalar() or 0

        rows = list(
            (
                await db.execute(
                    owned.order_by(ChatSession.updated_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars()
        )

        # Counted for THIS PAGE only. The grouped count over every message in
        # the database was cheap at twenty sessions and is not at twenty
        # thousand, and nothing off the page needs a number.
        counts: dict = {}
        if rows:
            counts = dict(
                (
                    await db.execute(
                        select(Message.session_id, func.count())
                        .where(Message.session_id.in_([r.id for r in rows]))
                        .group_by(Message.session_id)
                    )
                ).all()
            )

    response.headers["X-Total-Count"] = str(total)
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


def _trace_turn(
    session_id: uuid.UUID,
    req: TurnRequest | ResumeRequest,
    user: User,
    *,
    resumed: bool = False,
):
    """Trace-level attributes for one turn. A no-op when tracing is off.

    `session_id` is what turns a pile of independent traces into conversations,
    and multi-turn failures are invisible without it: "the answer was wrong"
    often means "turn 4 misresolved a pronoun from turn 2", which can only be
    read in order.

    `user_id` is the Supabase uuid -- an opaque internal id, deliberately not
    an email. A trace store is a third-party PII surface, and the id is enough
    for per-tenant cost and a support workflow.

    Tags carry the retrieval configuration, so a quality change can be
    attributed to a setting rather than guessed at: filter traces by
    `multi-query` and compare faithfulness against those without it.
    """
    tags = ["resumed" if resumed else "turn"]
    if isinstance(req, TurnRequest):
        if req.multi_query:
            tags.append("multi-query")
        if req.clarify:
            tags.append("clarify")
        if req.react:
            tags.append("react")

    return tracing.turn(
        name="chat.resume" if resumed else "chat.turn",
        session_id=str(session_id),
        user_id=user.owner_id,
        tags=tags,
        metadata={"top_k": getattr(req, "top_k", None)},
        # The question, so the trace LIST is scannable. Without it every row
        # shows a blank Input column and you have to open each trace to find
        # out what it was about.
        input=req.question,
    )


def _requested_profile(req: TurnRequest) -> str | None:
    """The model profile this turn asked for, or None to use the configured one.

    The dev gate lives HERE rather than inside `use_model_profile`, so the rule
    sits next to the request that carries the field. Buried in config it would
    be an assumption; here it is one readable line at the boundary.
    """
    if get_settings().app_env != "dev":
        return None
    return req.model_profile


@router.post("/{session_id}/messages", response_model=TurnResponse)
async def add_turn(
    session_id: uuid.UUID, req: TurnRequest, user: User = Depends(current_user)
) -> TurnResponse:
    """Ask a question inside a session. Blocking; see /stream for progress."""
    prep = await _prepare_turn(session_id, req, user)

    # Profile OUTSIDE the trace: the override changes which model every node
    # calls, so it has to be active before anything reads settings -- including
    # the trace metadata that records which model answered.
    with use_model_profile(_requested_profile(req)), _trace_turn(session_id, req, user):
        try:
            result = await run_agent(
                req.question,
                top_k=req.top_k,
                document_ids=prep["scope"],
                owner_id=user.owner_id,
                multi_query=req.multi_query,
                chat_context=prep["context"],
                session_id=str(session_id),
                preferences=prep.get("preferences", ""),
                thread_id=prep["thread_id"],
                clarify=req.clarify,
                react=req.react,
            )
        except LLMError as exc:
            log.warning("turn_failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"Model error: {exc}") from exc

        prep["trace_id"] = tracing.current_trace_id()
        tracing.set_turn_io(output=result.answer or result.interrupt)
        if result.paused:
            tracing.annotate_turn(
                status="paused for clarification",
                metadata={"paused": True},
                level="DEFAULT",
            )
        return await _finish_turn(session_id, req.question, result, prep, user)


@router.post("/{session_id}/resume", response_model=TurnResponse)
async def resume_turn(
    session_id: uuid.UUID, req: ResumeRequest, user: User = Depends(current_user)
) -> TurnResponse:
    """Answer the clarifying question: narrow, skip, or cancel.

    A separate request from the one that started the turn, and that is the
    whole point -- the pause lives in the checkpoint, not in a held-open
    connection, so it survives a deploy or a user who wandered off for ten
    minutes.
    """
    # Ownership first. `thread_id` is a client-supplied checkpoint key, so it
    # must never be the only thing authorising a resume -- checking the session
    # is what stops one user continuing another's paused graph.
    prep = await _prepare_resume(session_id, req, user)

    with _trace_turn(session_id, req, user, resumed=True):
        try:
            result = await resume_agent(
                req.thread_id,
                {"action": req.action, "answer": req.answer},
            )
        except LLMError as exc:
            log.warning("resume_failed", error=str(exc))
            raise HTTPException(status_code=502, detail=f"Model error: {exc}") from exc

        prep["trace_id"] = tracing.current_trace_id()
        tracing.set_turn_io(output=result.answer or result.interrupt)
        if result.paused:
            tracing.annotate_turn(
                status="paused for clarification",
                metadata={"paused": True},
                level="DEFAULT",
            )
        return await _finish_turn(session_id, req.question, result, prep, user)


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
        # Set INSIDE the generator, not around it. `use_model_profile` sets a
        # ContextVar, and a generator body runs in whatever context each
        # `__anext__` is driven from -- entering the block around the call that
        # merely CREATES the generator would set and reset the override before a
        # single node ran.
        with use_model_profile(_requested_profile(req)):
            async for chunk in _stream_events(
                stream_agent(
                    req.question,
                    top_k=req.top_k,
                    document_ids=prep["scope"],
                    # MUST be passed. `stream_agent` defaults owner_id to None,
                    # and None means "do not filter by owner" in the vector
                    # store -- so omitting it here (as this call once did) made
                    # a streamed turn search EVERY user's chunks. The document
                    # scope masked it whenever a session had documents
                    # selected, but a session with no scope resolves to
                    # `scope=None`, and then nothing constrained retrieval at
                    # all. /messages passed it; /stream did not, and /stream is
                    # the path the UI uses.
                    owner_id=user.owner_id,
                    multi_query=req.multi_query,
                    chat_context=prep["context"],
                    session_id=str(session_id),
                    preferences=prep.get("preferences", ""),
                    thread_id=prep["thread_id"],
                    clarify=req.clarify,
                    react=req.react,
                ),
                session_id=session_id,
                question=req.question,
                prep=prep,
                user=user,
                trace=_trace_turn(session_id, req, user),
            ):
                yield chunk

    return _sse_response(events())


@router.post("/{session_id}/resume/stream")
async def resume_stream(
    session_id: uuid.UUID, req: ResumeRequest, user: User = Depends(current_user)
) -> StreamingResponse:
    """SSE resume, so a continued turn streams exactly like a fresh one."""
    prep = await _prepare_resume(session_id, req, user)

    async def events() -> AsyncIterator[str]:
        async for chunk in _stream_events(
            stream_resume(
                req.thread_id,
                {"action": req.action, "answer": req.answer},
            ),
            session_id=session_id,
            question=req.question,
            prep=prep,
            user=user,
            trace=_trace_turn(session_id, req, user, resumed=True),
        ):
            yield chunk

    return _sse_response(events())


@router.post("/{session_id}/feedback", status_code=204)
async def submit_feedback(
    session_id: uuid.UUID,
    req: FeedbackRequest,
    user: User = Depends(current_user),
) -> None:
    """Attach a thumbs up/down to a turn's trace.

    Per §7.3.5 this is the single best quality signal available -- it is the
    only one that reflects what the USER thought, and it costs a button. Every
    model-based metric is a proxy for it.

    The trace id is read from the stored message rather than taken from the
    client. A client-supplied trace id would let anyone score any trace,
    including another tenant's, and scores are what the dashboards aggregate.
    """
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        message = next(
            (m for m in chat.messages if str(m.id) == req.message_id), None
        )

    if message is None or message.role != Role.assistant:
        raise HTTPException(status_code=404, detail="Message not found.")

    trace_id = (message.agent_meta or {}).get("trace_id")
    if not trace_id:
        # Tracing was off when this turn ran, so there is nothing to score.
        # Not an error -- feedback on an untraced turn is simply a no-op.
        log.info("feedback_untraced", message_id=req.message_id)
        return

    tracing.score(
        "user_feedback",
        # 1/0 rather than the raw string: numeric scores aggregate into a rate,
        # which is the form the question "is quality improving?" needs.
        1 if req.helpful else 0,
        trace_id=trace_id,
        data_type="NUMERIC",
        comment=req.comment or None,
    )
    log.info("feedback_recorded", helpful=req.helpful, trace_id=trace_id)


async def _stream_events(
    source: AsyncIterator[tuple],
    *,
    session_id: uuid.UUID,
    question: str,
    prep: dict,
    user: User,
    trace: Any = None,
) -> AsyncIterator[str]:
    """Turn agent stream items into SSE, shared by start and resume.

    Three terminal shapes: `interrupt` (paused, nothing persisted), `done`
    (finished and persisted), or `error`.

    `trace` is the trace context manager, entered HERE rather than in the
    endpoint. An SSE endpoint returns its StreamingResponse immediately and the
    generator body runs afterwards, so a `with` in the endpoint would have
    exited before a single node ran -- and every span would have landed outside
    the trace.
    """
    final_state: dict = {}
    stack = contextlib.ExitStack()
    if trace is not None:
        stack.enter_context(trace)
    try:
        async for kind, node, payload in source:
            if kind == "state":
                final_state = payload
                continue
            if kind == "interrupt":
                # A pause is terminal FOR THIS STREAM. The turn is not
                # finished, so nothing is persisted and the thread is not
                # discarded -- it is the only way back to this state.
                # A pause IS the outcome for this trace, so record it as the
                # output -- otherwise a paused turn looks like a turn that
                # produced nothing.
                tracing.set_turn_io(output={"paused": payload})
        # The LangGraph child span is marked ERROR by the callback handler
                # because an interrupt is signalled as an exception. Say plainly
                # on the ROOT span that this was a pause, not a failure.
                tracing.annotate_turn(
                    status="paused for clarification",
                    metadata={"paused": True},
                    level="DEFAULT",
                )
                yield _sse(
                    "interrupt",
                    {**payload, "thread_id": prep.get("thread_id")},
                )
                return
            if kind == "progress":
                # Fine-grained, from inside a node: which search is running,
                # which rerank. Passed through with its own shape rather than
                # squeezed into the node event -- the UI phrases these
                # differently ("Searching the web for ...") and needs the
                # fields, not a pre-rendered sentence.
                yield _sse("activity", payload)
                continue
            yield _sse("progress", {"node": node, "detail": _describe(node, payload)})

        result = AgentResult(final_state)
        # Captured inside the trace context, before it closes.
        prep["trace_id"] = tracing.current_trace_id()
        tracing.set_turn_io(output=result.answer)
        await _persist_turn(session_id, question, result, prep, user)
        _score_turn(result, prep["trace_id"])
        yield _sse(
            "done",
            {
                "answer": result.answer,
                "sources": [_hit_out(h).model_dump(mode="json") for h in result.evidence],
                "sources_used": result.citations,
                "sub_questions": result.sub_questions,
                "critique": result.critique,
                "sufficient": result.sufficient,
                "partial": result.partial,
                "intent": result.intent,
                "memory_saved": result.memory_saved,
                "iterations": result.iterations,
                "trace": result.trace,
                "context_chars": len(prep["context"]),
                "clarification": result.clarification,
                # Returned so the client can attach feedback later. Without it,
                # a thumbs-down arriving two minutes after the answer has
                # nothing to point at.
                "trace_id": prep["trace_id"],
            },
        )
    except Exception as exc:
        # The response has already started, so an HTTP error code is no
        # longer available -- the failure has to travel as an event.
        log.exception("stream_turn_failed")
        yield _sse("error", {"detail": str(exc)})
    finally:
        stack.close()


def _sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
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

        prefs = await preferences.preferences_in_force(
            owner_id=user.owner_id, session_id=session_id
        )
        return {
            "context": context,
            "scope": scope,
            "was_empty": not messages,
            "thread_id": thread_id,
            # Rendered here rather than in a node so both the blocking and the
            # streaming path get it from one place -- they have diverged before.
            "preferences": preferences.render_for_prompt(prefs),
            "n_preferences": len(prefs),
        }


async def _prepare_resume(
    session_id: uuid.UUID, req: ResumeRequest, user: User
) -> dict:
    """Authorise a resume and rebuild the context a finished turn needs.

    Two things matter here.

    **Ownership is checked against the SESSION, never the thread id.** The
    thread id arrives from the client and is only a checkpoint key -- treating
    it as proof of anything would let one user continue another's paused graph
    by guessing or replaying an id. `_load` does the real check.

    **The thread id must belong to this session.** Thread ids are formatted
    `<session>:<turn>:<random>`, so a mismatched prefix is a client sending a
    valid-but-unrelated thread, which is rejected rather than resumed.
    """
    async with SessionLocal() as db:
        chat = await _load(db, session_id, user)
        messages = list(chat.messages)

    if not req.thread_id.startswith(f"{session_id}:"):
        raise HTTPException(status_code=400, detail="Thread does not belong to this session.")

    return {
        # The resumed run reuses the checkpointed chat_context; nothing needs
        # rebuilding, and rebuilding it would risk a different value than the
        # one the paused graph is holding.
        "context": "",
        "scope": None,
        "was_empty": not messages,
        "thread_id": req.thread_id,
    }


def _score_turn(result: AgentResult, trace_id: str | None) -> None:
    """Programmatic scores, computed from data the turn already produced.

    These cost nothing -- no model call, no extra work -- and they are the whole
    of §7.3.6's "cheap proxies" turned into a queryable dimension. Until now
    they went into a log line and evaporated; as scores they become a dashboard:
    zero-citation rate over time, mean iterations, how often the critic is
    unsatisfied.

    BOOLEAN rather than 0/1 where the thing is a fact, so Langfuse renders and
    aggregates it as a proportion instead of a meaningless average.
    """
    if trace_id is None:
        return

    # THE hallucination proxy: an answer citing nothing, from a system whose
    # entire contract is citation. Cheap to compute, and the single most
    # useful production signal here.
    tracing.score(
        "cited_sources",
        bool(result.citations),
        trace_id=trace_id,
        data_type="BOOLEAN",
    )
    # How hard the agent had to work. A rising mean means retrieval is
    # degrading -- the critic is sending it back more often.
    tracing.score("iterations", result.iterations, trace_id=trace_id)
    # The critic's own verdict. False means it gave up rather than succeeded.
    tracing.score(
        "sufficient", bool(result.sufficient), trace_id=trace_id, data_type="BOOLEAN"
    )
    if result.clarification:
        # Only present when a human was asked and answered, so its RATE tells
        # you how often questions arrive too vague to serve -- a fact about
        # your users, not your retriever.
        tracing.score(
            "clarified",
            True,
            trace_id=trace_id,
            data_type="BOOLEAN",
            comment=result.clarification[:200],
        )


async def _finish_turn(
    session_id: uuid.UUID,
    question: str,
    result: AgentResult,
    prep: dict,
    user: User,
) -> TurnResponse:
    """Persist a completed turn, or report a pause without persisting."""
    if result.paused:
        # Deliberately NOT persisted and NOT discarded: the turn has no answer
        # yet, and the checkpoint is the only route back to this state. A
        # paused thread is cleaned up by scripts/prune_checkpoints.py if the
        # human never comes back.
        log.info("turn_paused", session_id=str(session_id), thread=prep["thread_id"])
        return TurnResponse(
            session_id=session_id,
            question=question,
            answer="",
            sources=[],
            sources_used=[],
            sub_questions=result.sub_questions,
            critique="",
            sufficient=False,
            iterations=result.iterations,
            trace=result.trace,
            context_chars=len(prep["context"]),
            interrupt=InterruptOut(**result.interrupt),
            thread_id=prep["thread_id"],
        )

    await _persist_turn(session_id, question, result, prep, user)
    _score_turn(result, prep.get("trace_id"))
    return TurnResponse(
        session_id=session_id,
        question=question,
        answer=result.answer,
        sources=[_hit_out(h) for h in result.evidence],
        sources_used=result.citations,
        sub_questions=result.sub_questions,
        critique=result.critique,
        sufficient=result.sufficient,
        partial=result.partial,
        iterations=result.iterations,
        trace=result.trace,
        context_chars=len(prep["context"]),
        clarification=result.clarification,
        trace_id=prep.get("trace_id"),
    )


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
                        # Provenance travels with the stored citation, so
                        # a reloaded transcript still knows which sources
                        # were web pages and can link them.
                        "source": h.source,
                        "url": h.url,
                    }
                    for i, h in enumerate(result.evidence)
                ],
                agent_meta={
                    "sources_used": result.citations,
                    "sub_questions": result.sub_questions,
                    "iterations": result.iterations,
                    "sufficient": result.sufficient,
                    "partial": result.partial,
                    "intent": result.intent,
                    "memory_saved": result.memory_saved,
                    "critique": result.critique,
                    # Absent on an ordinary turn, so "the question was clear" and
                    # "the user clarified it" stay distinguishable after the
                    # fact. Without it a clarified turn is indistinguishable from
                    # an automatic one in the transcript.
                    "clarification": result.clarification,
                    # Stored so feedback arriving LATER -- a thumbs-down two
                    # minutes after the answer -- has a trace to attach to.
                    # Without this the score would have nowhere to land.
                    "trace_id": prep.get("trace_id"),
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
        source=h.source,
        url=h.url,
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


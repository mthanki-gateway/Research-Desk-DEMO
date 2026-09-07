"""Langfuse tracing. The single seam between this app and the tracer.

WHY A SEAM AND NOT DIRECT SDK CALLS EVERYWHERE

Two reasons, and the second is the important one.

1. Tracing is OPTIONAL. With no keys configured the app must behave exactly as
   it did before observability existed -- no imports at module scope, no
   failures, no noise. Every function here is a no-op in that state, so call
   sites read identically whether tracing is on or off.

2. **Tracing must never break a request.** An observability outage taking down
   the product is the classic own-goal: you added a system to find out when
   things go wrong and it became the thing that went wrong. Every entry point
   here swallows its own errors and degrades to not-tracing.

WHAT LANGFUSE GIVES THIS APP THAT structlog CANNOT

A turn is 3-5 model calls across four graph nodes with a retry loop. Flat log
lines cannot show that `critique` fired twice because retrieval missed on the
first pass, because a log line has no parent. A trace does:

    TRACE  chat turn                     session=... user=...  41.2s  9,840 tok
    |- SPAN  clarify
    |  \- GENERATION gemma  in 890 out 140
    |- SPAN  plan
    |- SPAN  retrieve
    |  |- EMBEDDING  embed query
    |  \- RETRIEVER  qdrant.search  top_k=5 -> 5 hits
    |- SPAN  draft
    \- SPAN  critique   sufficient=false   <- this is why it looped

VERSION NOTE: langfuse 4.x is OpenTelemetry-based and its API differs sharply
from the 2.x tutorials still dominating search results. 2.x used
`langfuse.trace()` / `langfuse.generation()` and
`from langfuse.callback import CallbackHandler`. 4.x uses
`start_as_current_observation(as_type=...)`, `propagate_attributes()`, and
`from langfuse.langchain import CallbackHandler`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import structlog

from app.config import get_settings

log = structlog.get_logger()

_client: Any = None
_checked = False


def enabled() -> bool:
    return get_settings().tracing_enabled


def _close_quietly(stack: contextlib.ExitStack, what: str) -> None:
    """Unwind tracing context managers, swallowing THEIR failures.

    `with ExitStack()` would let an exception raised inside Langfuse's own
    `__exit__` escape into the request -- which is how the original failure
    happened: it surfaced at the END of a turn, during unwinding, not while the
    app was working. That breaks the one contract this module has.

    The cost is that the inner context managers are exited with no exception
    info, so Langfuse cannot mark a span as errored automatically. That is
    acceptable here because the places where it matters set `level="ERROR"`
    explicitly -- see `GenAIClient.generate`.
    """
    try:
        stack.close()
    except Exception:
        log.warning("tracing_exit_failed", observation=what, exc_info=True)


def get_tracer() -> Any:
    """The Langfuse client, or None when tracing is off or unavailable.

    Imported lazily and cached. `_checked` guards against retrying a failed
    init on every request -- a wrong host would otherwise pay a connection
    timeout per turn.
    """
    global _client, _checked
    if _checked:
        return _client
    _checked = True

    if not enabled():
        log.info("tracing_disabled", reason="no langfuse keys configured")
        return None

    settings = get_settings()
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            # Tags every trace with dev/prod, so local experimentation does not
            # pollute the numbers you actually report on.
            environment=settings.app_env,
        )
        log.info("tracing_ready", host=settings.langfuse_host)
    except Exception:
        log.exception("tracing_init_failed")
        _client = None
    return _client


def langchain_handler() -> Any:
    """Callback handler for the LangGraph agent, or None.

    This one line instruments the ENTIRE graph -- every node becomes a span,
    every edge the tree structure, every state update captured. It is the
    strongest practical argument for having built on LangGraph at all: the
    instrumentation points already exist and this just attaches a listener.

    It does NOT see inside `GenAIClient`, which talks to the provider over raw
    httpx rather than through LangChain. Those are instrumented by hand in
    llm.py, which is why both exist.
    """
    if get_tracer() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        log.exception("tracing_handler_failed")
        return None


def callbacks() -> list[Any]:
    """Callback list for a LangGraph config. Empty when tracing is off."""
    handler = langchain_handler()
    return [handler] if handler is not None else []


@contextlib.contextmanager
def turn(
    *,
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    input: Any = None,
) -> Iterator[None]:
    """Trace-level attributes for one user turn.

    `session_id` groups turns into a conversation, which is what makes
    multi-turn failures visible at all: "the answer was wrong" often means
    "turn 4 misresolved a pronoun from turn 2", and that is only readable in
    order. `user_id` gives per-tenant cost and a support workflow.

    `user_id` MUST be an opaque internal id, never an email -- a trace store is
    a third-party PII surface.
    """
    tracer = get_tracer()
    if tracer is None:
        yield
        return

    # BOTH context managers, and in this order.
    #
    # `propagate_attributes` does NOT create a span -- it only decorates
    # whatever trace is already active. Using it alone left no root span at
    # all: `get_current_trace_id()` returned None and every score was rejected
    # with "Bad request", having no trace to attach to. So the attributes go on
    # the outside and an explicit root observation inside. `as_type="agent"` is
    # the semantic type for a turn of an agentic system.
    stack = contextlib.ExitStack()
    try:
        # ONLY the setup is guarded, and the guard does NOT yield.
        #
        # This previously wrapped the `yield` in `try/except Exception: yield`,
        # which is broken two ways. An exception raised by the CALLER's body
        # propagates back through the yield point, so that except clause caught
        # the app's own errors -- and then yielded a SECOND time. A
        # @contextmanager generator must yield exactly once; yielding twice
        # raises `RuntimeError: generator didn't stop after throw()`, which
        # replaced the real error with a meaningless one and surfaced to the
        # user as a failed turn.
        try:
            from langfuse import propagate_attributes

            stack.enter_context(
                propagate_attributes(
                    trace_name=name,
                    session_id=session_id,
                    user_id=user_id,
                    tags=tags or [],
                    metadata=metadata or {},
                )
            )
            stack.enter_context(
                tracer.start_as_current_observation(
                    name=name, as_type="agent", input=input
                )
            )
            # Also set it at TRACE level. The root observation carrying an
            # input is not the same thing as the trace having one, and it is
            # the trace that the list view renders.
            tracer.set_current_trace_io(input=input)
        except Exception:
            log.exception("tracing_turn_failed")

        # Reached whether setup succeeded or not, exactly once. An exception
        # from the body propagates normally to the caller.
        yield
    finally:
        _close_quietly(stack, name)


@contextlib.contextmanager
def observe(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
    model: str | None = None,
    model_parameters: dict | None = None,
) -> Iterator[Any]:
    """One nested observation. Yields a handle, or None when tracing is off.

    Callers must tolerate a None handle -- that is the whole point of the seam,
    and it keeps `if span is not None` as the only tracing-aware line in
    otherwise ordinary code.

    `as_type` carries semantics Langfuse renders differently: "generation" for a
    model call (gets token/cost accounting), "embedding", "retriever", "tool".
    Using the right one is free and makes the trace readable at a glance.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Same single-yield discipline as `turn()` above, and for the same reason:
    # wrapping the yield in `except Exception: yield None` caught the caller's
    # own exceptions and then yielded twice.
    stack = contextlib.ExitStack()
    span = None
    try:
        try:
            span = stack.enter_context(
                tracer.start_as_current_observation(
                    name=name,
                    as_type=as_type,  # type: ignore[arg-type]
                    input=input,
                    metadata=metadata,
                    model=model,
                    model_parameters=model_parameters,
                )
            )
        except Exception:
            log.exception("tracing_observe_failed", observation=name)

        yield span
    finally:
        _close_quietly(stack, name)


def update(span: Any, **fields: Any) -> None:
    """Write fields onto an observation. Safe with a None span."""
    if span is None:
        return
    try:
        span.update(**fields)
    except Exception:
        log.warning("tracing_update_failed", exc_info=True)


def set_turn_io(*, input: Any = None, output: Any = None) -> None:
    """Set the TRACE-level input/output, i.e. the question and the answer.

    Without this the trace list is unusable. Langfuse shows Input and Output as
    columns, and they come from the trace, not from its nested observations --
    so a root span opened with neither leaves every row blank and identical.
    You then have to open each trace to discover what it was even about, which
    defeats the point of a list.

    Called twice per turn: the question on the way in, the answer on the way
    out. Kept separate from `turn()` because the answer does not exist yet when
    the context is entered.
    """
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        tracer.set_current_trace_io(input=input, output=output)
    except Exception:
        log.warning("tracing_set_io_failed", exc_info=True)


def annotate_turn(
    *,
    status: str | None = None,
    metadata: dict | None = None,
    level: str | None = None,
) -> None:
    """Record a status on the turn's root span.

    Exists because of one upstream artifact worth knowing about.

    When the graph interrupts for human-in-the-loop, LangGraph signals it
    INTERNALLY AS AN EXCEPTION (GraphInterrupt). Langfuse's LangChain callback
    handler sees `on_chain_error` and marks the `LangGraph` span
    `level=ERROR` -- with an empty status message, because nothing actually
    went wrong. So a perfectly successful pause is recorded as a failure, which
    pollutes the "Observations by Level" chart and makes every clarification
    look like a broken turn.

    That child span belongs to the handler and cannot be corrected from here,
    so instead the ROOT span is annotated explicitly. Anyone reading the trace
    sees "paused for clarification" at the top, next to the misleading ERROR
    underneath it.
    """
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        tracer.update_current_span(
            status_message=status,
            metadata=metadata,
            level=level,  # type: ignore[arg-type]
        )
    except Exception:
        log.warning("tracing_annotate_failed", exc_info=True)


def current_trace_id() -> str | None:
    """The active trace id, so a turn can be scored or linked to later.

    Returned to the client and stored on the message: without it, user feedback
    arriving minutes later has nothing to attach to.
    """
    tracer = get_tracer()
    if tracer is None:
        return None
    try:
        return tracer.get_current_trace_id()
    except Exception:
        return None


def score(
    name: str,
    value: float | str,
    *,
    trace_id: str | None = None,
    comment: str | None = None,
    data_type: str | None = None,
) -> None:
    """Attach a quality judgement to a trace.

    This is the join that makes observability worth more than logging: a
    faithfulness of 0.4 in a spreadsheet tells you the number, whereas the same
    score ON the trace lets you click through to the exact prompt, the exact
    retrieved chunks and the exact completion that earned it. Debugging quality
    is the point, not measuring it.
    """
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        tracer.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            comment=comment,
            data_type=data_type,  # type: ignore[arg-type]
        )
    except Exception:
        log.warning("tracing_score_failed", score=name, exc_info=True)


def flush() -> None:
    """Send anything still buffered.

    The SDK batches in a background thread, so a process that exits promptly
    drops its last traces. Called from the FastAPI lifespan shutdown.
    """
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        tracer.flush()
    except Exception:
        log.warning("tracing_flush_failed", exc_info=True)


def reset_for_tests() -> None:
    """Drop the cached client so a test can change settings and re-init."""
    global _client, _checked
    _client = None
    _checked = False

"""Fine-grained progress, from inside a node to the SSE stream.

WHY THIS EXISTS

Progress used to be per NODE: `clarify`, `plan`, `retrieve`, `draft`. That says
which stage is running and nothing about what it is doing, so a turn that spent
six seconds searching the web showed one motionless line reading "retrieve" --
and the web search, which is the most interesting thing the agent does, was
invisible.

HOW IT GETS OUT

LangGraph's `get_stream_writer()` returns a callable that publishes onto the
graph's "custom" stream mode, which the caller drains alongside "updates" and
"values". That is the supported way for code deep inside a node to reach the
consumer without threading a queue through every signature.

WHY IT IS A NO-OP OUTSIDE A GRAPH

The same retrieval functions run from the evaluation harness, from scripts and
from tests, where there is no graph and no stream. `get_stream_writer` raises
there, so every call is guarded: emitting progress must never be a reason a
search fails.
"""

from __future__ import annotations

import contextlib
from typing import Any

import structlog

log = structlog.get_logger()


def emit(kind: str, **fields: Any) -> None:
    """Publish one progress event. Silent when there is nothing listening.

    `kind` is a short machine-readable verb -- "search", "search_done",
    "rerank" -- and the UI decides how to phrase it. Sending prose from here
    would put user-facing copy in a service module and make it untranslatable.
    """
    with contextlib.suppress(Exception):
        # Imported lazily: this module is imported by retrieval, which the
        # evaluation harness loads without LangGraph's runtime in scope.
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer is not None:
            writer({"progress": {"kind": kind, **fields}})


def searching(source: str, query: str) -> None:
    """A search is STARTING. Emitted before the call, not after.

    The point is to show work in flight -- an event published once the results
    are back tells the user what already happened, which is the one moment they
    did not need telling.
    """
    emit("search", source=source, query=query[:120])


def searched(source: str, query: str, n: int) -> None:
    emit("search_done", source=source, query=query[:120], n=n)


def looked_up(tool: str) -> None:
    """A METADATA tool is starting -- names, counts, sizes, not contents.

    A separate verb from `searching` on purpose. These were emitting nothing,
    so a turn that answered "how many documents do we have?" entirely from the
    collection metadata showed an empty activity list, and the only visible
    trace of the agent's work was whatever searches happened to run alongside
    it. The user reasonably read that as the agent having searched for the
    answer it in fact looked up.

    Distinguishing them in the UI matters for the same reason it matters in the
    answer: reading a document and counting documents are different claims.
    """
    emit("lookup", tool=tool)


def looked_up_done(tool: str) -> None:
    emit("lookup_done", tool=tool)


def remembering(text: str) -> None:
    """A preference is being stored. The one tool that WRITES."""
    emit("remember", text=text[:120])

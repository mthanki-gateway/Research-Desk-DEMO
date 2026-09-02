"""The graph.

    START → plan → retrieve → draft → critique ─┬→ END
                      ▲                         │
                      └──────── retry ──────────┘

The cycle is the reason this is a graph and not four awaits. LangChain's LCEL
builds DAGs, and a DAG cannot loop; expressing "critique decides whether to go
back" as a chain is impossible, and as a hand-rolled while-loop it means
threading a mutable dict through every step by hand.

The checkpointer slot (step 6) is the other reason: compiling with a
checkpointer snapshots state after every node, which is what turns this into
resumable chat sessions for free.
"""

from __future__ import annotations

import uuid
from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import critique, draft, plan, retrieve_node
from app.agent.state import ResearchState
from app.config import get_settings
from app.services.vectorstore import SearchHit

log = structlog.get_logger()


def should_continue(state: ResearchState) -> Literal["retrieve", "__end__"]:
    """The conditional edge: the model's own verdict decides control flow.

    This is the thing a static pipeline cannot express -- whether to loop is
    data produced at runtime, not a decision made when the code was written.
    """
    settings = get_settings()

    if state.get("sufficient", True):
        return END
    if state.get("iterations", 0) >= settings.agent_max_iterations:
        # Cap, not a judgement of quality. Each cycle is ~2 Gemma calls, and an
        # unbounded loop is the easiest way to spend a daily quota.
        log.info("iteration_cap_reached", iterations=state.get("iterations"))
        return END
    if not state.get("pending_queries"):
        return END  # nothing left to search for
    return "retrieve"


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)

    builder.add_node("plan", plan)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("draft", draft)
    builder.add_node("critique", critique)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "critique")
    builder.add_conditional_edges(
        "critique", should_continue, {"retrieve": "retrieve", END: END}
    )

    # checkpointer=None means no persistence -- fine for step 4, which is
    # single-turn. Step 6 passes AsyncPostgresSaver here and nothing else in
    # this file changes.
    return builder.compile(checkpointer=checkpointer)


_graph = None


def set_graph(compiled) -> None:
    """Install a graph built with a checkpointer (called from lifespan)."""
    global _graph
    _graph = compiled


def get_graph():
    global _graph
    if _graph is None:
        # No checkpointer: single-turn only, no resume. Used if the
        # checkpointer failed to initialise -- degrade rather than 500.
        _graph = build_graph()
        log.info("agent_graph_compiled", checkpointer=False)
    return _graph


class AgentResult:
    """Flattened view of the final state, for the API layer."""

    def __init__(self, state: dict) -> None:
        self.question: str = state.get("question", "")
        self.answer: str = state.get("draft", "")
        self.evidence: list[SearchHit] = state.get("evidence", [])
        self.citations: list[int] = state.get("citations", [])
        self.sub_questions: list[str] = state.get("sub_questions", [])
        self.critique: str = state.get("critique", "")
        self.sufficient: bool = state.get("sufficient", True)
        self.iterations: int = state.get("iterations", 0)
        self.trace: list[dict] = state.get("trace", [])


def initial_state(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    chat_context: str = "",
) -> ResearchState:
    return {
        "question": question,
        "top_k": top_k or get_settings().retrieval_top_k,
        "document_ids": [str(d) for d in document_ids] if document_ids else None,
        "owner_id": owner_id,
        "multi_query": multi_query,
        "chat_context": chat_context,
        # These cannot be reset by passing []: `evidence`, `sub_questions`,
        # `tried_queries` and `trace` all have append-style reducers, and a
        # reducer applies to the INPUT too -- so [] appends nothing rather than
        # clearing. Isolation between turns comes from a per-turn thread_id
        # instead; see thread_config().
        "evidence": [],
        "sub_questions": [],
        "tried_queries": [],
        "trace": [],
        "unanswered": [],
        "iterations": 0,
    }


def thread_config(thread_id: str | None) -> dict | None:
    """The checkpointer keys state by thread_id. Without one, no persistence.

    Callers pass a thread id scoped to ONE TURN (`<session>:<n>`), not one per
    session. That is deliberate:

    * accumulators (evidence, sub_questions, trace) have append reducers, so
      reusing a thread across turns made them grow forever -- and, worse, let
      evidence retrieved for an earlier question leak into a later answer
    * conversation memory does not live in graph state anyway. It lives in the
      `messages` table and is passed in as `chat_context`, which keeps the
      transcript queryable and independent of LangGraph's state shape

    So the checkpointer's job here is durability *within* a run -- resume after
    a crash mid-graph, and `interrupt()` for human-in-the-loop -- not carrying
    the conversation.
    """
    if not thread_id:
        return None
    return {"configurable": {"thread_id": thread_id}}


async def run_agent(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    chat_context: str = "",
    thread_id: str | None = None,
) -> AgentResult:
    state = initial_state(
        question,
        top_k=top_k,
        document_ids=document_ids,
        owner_id=owner_id,
        multi_query=multi_query,
        chat_context=chat_context,
    )
    final = await get_graph().ainvoke(state, config=thread_config(thread_id))
    result = AgentResult(final)
    log.info(
        "agent_done",
        iterations=result.iterations,
        n_evidence=len(result.evidence),
        sufficient=result.sufficient,
    )
    return result


async def stream_agent(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    chat_context: str = "",
    thread_id: str | None = None,
):
    """Yield (node_name, state_update) as each node completes.

    Node-level progress, not token-level. With responseSchema output there is
    no partial prose to stream -- you would be streaming half a JSON object.
    For an agent this is arguably the better signal anyway: "retrieving 2 of 3"
    says what is happening, where a token crawl only proves it is alive.
    """
    state = initial_state(
        question,
        top_k=top_k,
        document_ids=document_ids,
        owner_id=owner_id,
        multi_query=multi_query,
        chat_context=chat_context,
    )
    # Two stream modes at once: "updates" gives per-node deltas for progress,
    # "values" gives the full state after each node so the caller ends up with
    # the final state without a second aget_state() round-trip.
    async for mode, chunk in get_graph().astream(
        state, config=thread_config(thread_id), stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            for node, update in chunk.items():
                yield "node", node, update
        elif mode == "values":
            yield "state", None, chunk

"""The graph.

    START ─┬─(no tools)→ route ─┬─(instruction only)────────────→ END
           │                    └─→ clarify
           └─(agent)──────────────→ clarify ─┬─(vague)→ ask_human ─→ …
                                             │
                                             ├─→ react ─┬─(no search)→ END
                                             │          └─→ draft ──┐
                                             └─→ plan → retrieve → draft
                                                          ▲          │
                                                          │      critique
                                                          │      │    │
                                              END ←───────┴──────┘    │
                                                        └── retry ────┘

WHO DECIDES WHETHER TO RETRIEVE

The agent does, when it has tools. `react` calls no search for a greeting, a
question about the assistant, or a bare instruction, and writes the reply
itself -- so "hi" costs one model call instead of a plan, three sub-questions
and five web searches. `route` remains the entry point only for the paths that
cannot make that call for themselves: the Gemma profile, which emits no
functionCall parts, and the planned path the evaluation harness measures.

The cycle is the reason this is a graph and not four awaits. LangChain's LCEL
builds DAGs, and a DAG cannot loop; expressing "critique decides whether to go
back" as a chain is impossible, and as a hand-rolled while-loop it means
threading a mutable dict through every step by hand.

The checkpointer slot (step 6) is the other reason: compiling with a
checkpointer snapshots state after every node, which is what turns this into
resumable chat sessions for free -- and, in step 7, into a graph that can PAUSE
mid-run to ask the user a question and be resumed by a different request.

`clarify` is a no-op unless the turn asked for it, and even then it only routes
to `ask_human` when the request is genuinely too vague to search -- so one
compiled graph serves every mode.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agent.nodes import (
    REMEMBER,
    UNANSWERABLE,
    UNSUPPORTED_CLAIM,
    ask_human,
    critique,
    draft,
    plan,
    resolve,
    retrieve_node,
    route,
)
from app.agent.nodes import (
    clarify as clarify_node,
)
from app.agent.react import react
from app.agent.state import ResearchState
from app.config import get_settings
from app.services import tracing
from app.services.vectorstore import SearchHit

log = structlog.get_logger()


def should_continue(
    state: ResearchState,
) -> Literal["retrieve", "draft", "resolve", "__end__"]:
    """The conditional edge: the model's own verdict decides control flow.

    This is the thing a static pipeline cannot express -- whether to loop is
    data produced at runtime, not a decision made when the code was written.

    THE REMEDY DEPENDS ON THE FAILURE, and that is the whole reason `critique`
    now returns a category instead of a boolean:

        unsupported_claim  -> draft      rewrite from the SAME evidence.
                                         Searching again cannot fix a sentence
                                         that says more than its source.
        missing_evidence   -> retrieve   find what is not there yet. Rewriting
                                         cannot fix words that were never in
                                         any passage.
        unanswerable       -> resolve    say what IS known and name the gap.

    Every exhausted budget also lands on `resolve` rather than END. Stopping at
    END would ship whatever draft happened to exist, complete with the claims
    the critic just rejected -- the budget running out is not a reason to
    publish a criticised answer unchanged.
    """
    settings = get_settings()

    if state.get("sufficient", True):
        return END

    mode = state.get("failure_mode", "")

    if mode == UNSUPPORTED_CLAIM:
        # One rewrite, not a conversation. A second pass at the same evidence
        # rarely differs, and a critic that rejects it twice is usually
        # disagreeing about tone rather than support.
        if state.get("regen_count", 0) < settings.agent_max_regens:
            return "draft"
        log.info("regen_cap_reached", regens=state.get("regen_count"))
        return "resolve"

    if mode == UNANSWERABLE:
        return "resolve"

    # missing_evidence, and anything unrecognised -- the conservative default,
    # since looking again is recoverable and asserting unanswerable is not.
    if state.get("iterations", 0) >= settings.agent_max_iterations:
        # Cap, not a judgement of quality. Each cycle costs ~2 model calls, and
        # an unbounded loop is the easiest way to spend a daily quota.
        log.info("iteration_cap_reached", iterations=state.get("iterations"))
        return "resolve"
    if not state.get("pending_queries"):
        return "resolve"  # nothing left to search for
    return "retrieve"


def _uses_react(state: ResearchState) -> bool:
    """Will this turn gather with the ReAct agent?

    Shared by `entry` and `gather_strategy` so the two cannot disagree. They
    did while this was inlined twice: `entry` skipped the router on a turn that
    then fell back to `plan` because the model had no tool calling, leaving the
    turn with neither a router nor an agent able to handle an instruction.
    """
    return bool(state.get("react")) and get_settings().supports_tool_calling


def gather_strategy(state: ResearchState) -> Literal["react", "plan"]:
    """Which evidence-gathering strategy this turn uses.

    `plan` decomposes every lookup up front; `react` lets the model choose each
    one after seeing the last result. The second is what multi-hop needs -- you
    cannot look up a company before a search names it -- and the first is what
    the evaluation harness measures, so both stay.

    CAPABILITY BEATS PREFERENCE. ReAct is built on native functionCall parts,
    and the Gemma profile's model emits none -- asked to use tools it narrates
    its intentions as prose, which `generate_tools` reads as "no calls
    requested". The loop would exit on round 1 with no evidence and `draft`
    would answer from nothing: a silent wrong answer rather than a visible
    misconfiguration. So a turn that asks for ReAct on a model that cannot do
    it falls back to the planned path instead.
    """
    if not state.get("react"):
        return "plan"
    if not get_settings().supports_tool_calling:
        log.info("react_unavailable", reason="model has no native tool calling")
        return "plan"
    return "react"


def after_react(state: ResearchState) -> Literal["draft", "__end__"]:
    """A turn that needed no sources is already answered.

    `react` writes its own reply when it called no tools and gathered nothing
    -- a greeting, a question about the assistant, a bare instruction. Sending
    that to `draft` is what made "hi" run three sub-questions and five web
    searches: `draft` exists to compose an answer FROM EVIDENCE, so handed none
    it either says nothing was found or reaches for whatever was lexically
    nearest.
    """
    if state.get("answered_directly"):
        return END
    return "draft"


def entry(state: ResearchState) -> Literal["route", "clarify"]:
    """Whether the standalone router runs at all.

    THE AGENT OWNS THIS JOB WHEN IT CAN DO IT. `route` is a model call in front
    of every turn that classifies the message before anything else has run --
    paid on all of them, including the overwhelming majority that store
    nothing. The ReAct agent makes the same judgement with `remember_preference`
    at the moment it actually has something to store, and it composes: "tell me
    about X and always cite pages" is one search call plus one remember call in
    a single round, where the router first had to split the message in two.

    It stays for the path that CANNOT do it. Tool calling is what the agent
    version rests on, and the Gemma profile emits no functionCall parts -- so
    on that profile, and on the planned path the evaluation harness measures,
    the router is still the only thing standing between "remember to cite
    pages" and a document search for the phrase.
    """
    return "clarify" if _uses_react(state) else "route"


def after_route(state: ResearchState) -> Literal["clarify", "__end__"]:
    """Only a PURE instruction ends the turn here.

    `route` already wrote the confirmation into `draft`, and for an instruction
    alone there is nothing to retrieve -- an instruction about how to answer is
    not a question about the documents. Falling through to the search path is
    the bug this node exists to prevent: it produced "your documents do not
    mention personal preferences regarding search behaviour".

    `both` deliberately does NOT end. One message often carries a question AND
    a standing instruction ("tell me about X, and always say which facts came
    from the web"), and treating that as either one alone silently drops half
    of what was asked. `route` has already stored the preference and rewritten
    `question` to the question part, so the rest of the graph runs normally --
    with the new instruction already in force.
    """
    if state.get("intent") == REMEMBER:
        return END
    return "clarify"


def needs_human(state: ResearchState) -> Literal["ask_human", "plan"]:
    """Ask the user only when `clarify` actually produced a question.

    A cheap dict read, deliberately: the judgement was made in the node and
    written to state, and the router just reads the verdict. Routers stay pure
    so control flow is testable without a model.
    """
    if state.get("pending_clarification"):
        return "ask_human"
    return gather_strategy(state)  # type: ignore[return-value]


def after_human(state: ResearchState) -> Literal["react", "plan", "__end__"]:
    """The user declined to answer, so nothing is searched.

    Separate from `should_continue` because they answer different questions:
    this one is "did the user stop us", that one is "is the answer good
    enough".
    """
    if state.get("cancelled"):
        return END
    return gather_strategy(state)


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)

    builder.add_node("route", route)
    builder.add_node("clarify", clarify_node)
    builder.add_node("ask_human", ask_human)
    builder.add_node("plan", plan)
    builder.add_node("react", react)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("draft", draft)
    builder.add_node("critique", critique)
    builder.add_node("resolve", resolve)

    builder.add_conditional_edges(
        START, entry, {"route": "route", "clarify": "clarify"}
    )
    builder.add_conditional_edges(
        "route", after_route, {"clarify": "clarify", END: END}
    )
    builder.add_conditional_edges(
        "clarify",
        needs_human,
        {"ask_human": "ask_human", "plan": "plan", "react": "react"},
    )
    builder.add_conditional_edges(
        "ask_human", after_human, {"plan": "plan", "react": "react", END: END}
    )
    builder.add_edge("plan", "retrieve")
    # ReAct does its own retrieval through tools, so it goes STRAIGHT to
    # drafting -- it produces the evidence that `retrieve` would have. Unless
    # it gathered nothing on purpose, in which case it has already answered.
    builder.add_conditional_edges(
        "react", after_react, {"draft": "draft", END: END}
    )
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "critique")
    builder.add_conditional_edges(
        "critique",
        should_continue,
        {"retrieve": "retrieve", "draft": "draft", "resolve": "resolve", END: END},
    )
    # `resolve` is terminal. It has already produced the final answer and set
    # sufficient=True; routing it back to critique would re-review a draft that
    # was written specifically to satisfy the objection, and a critic asked to
    # review its own instructions being followed tends to find something new.
    builder.add_edge("resolve", END)

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


def interrupt_payload(state: dict) -> dict | None:
    """The pending human-in-the-loop request, if the graph paused.

    LangGraph reports a pause by putting `__interrupt__` in the returned state,
    holding one or more Interrupt objects whose `.value` is whatever the node
    passed to `interrupt()`. Read defensively -- the exact container type is an
    internal detail, and a shape change here should degrade to "not paused"
    rather than raise inside an API handler.
    """
    raw = state.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, list | tuple) else raw
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"type": "unknown", "value": str(value)}


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
        # Human-in-the-loop. `interrupt` is None on every ordinary turn, so
        # existing callers (the evaluation harness among them) are unaffected;
        # they read `.answer` and never look here.
        self.interrupt: dict | None = interrupt_payload(state)
        # What the user said when asked to clarify. None on an ordinary turn,
        # so "the question was clear" stays distinguishable from "the user
        # clarified it".
        # True when `resolve` answered from partial evidence and named the gap.
        # Distinct from `sufficient`, which by then is True precisely because
        # resolve produced the final answer -- without this the UI cannot tell a
        # complete answer from a knowingly incomplete one.
        self.partial: bool = bool(state.get("partial"))
        # Preferences this turn stored, so the UI can say memory changed
        # rather than leaving it to be inferred from the prose.
        self.memory_saved: list[str] = state.get("memory_saved") or []
        self.intent: str = state.get("intent", "")
        # `answer` is an error message rather than an answer. Callers that are
        # not a chat window -- the evaluation harness above all -- must be able
        # to tell the difference; see `generation_failed` in state.py.
        self.failed: bool = bool(state.get("generation_failed"))
        self.clarification: str | None = state.get("clarification")
        self.original_question: str | None = state.get("original_question")
        self.cancelled: bool = bool(state.get("cancelled"))

    @property
    def paused(self) -> bool:
        """True when the graph is waiting for a human, not finished."""
        return self.interrupt is not None


def initial_state(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    chat_context: str = "",
    session_id: str | None = None,
    preferences: str = "",
    clarify: bool | None = None,
    react: bool | None = None,
    resumable: bool = True,
) -> ResearchState:
    settings = get_settings()
    ask = settings.agent_clarify if clarify is None else clarify
    # A pause is only meaningful if there is a thread to resume. Without one
    # the checkpointer stores nothing, `interrupt()` has nowhere to record the
    # pause, and the turn would simply stop with no way to continue it. Forcing
    # clarification off is the honest degradation.
    if ask and not resumable:
        # Debug, not warning: a caller with no thread simply cannot pause. That
        # is the evaluation harness on every question, and it is correct
        # behaviour rather than a misconfiguration.
        log.debug("clarify_skipped", reason="no thread_id")
        ask = False

    return {
        "question": question,
        "top_k": top_k or settings.retrieval_top_k,
        "document_ids": [str(d) for d in document_ids] if document_ids else None,
        "owner_id": owner_id,
        "multi_query": multi_query,
        "chat_context": chat_context,
        "session_id": session_id,
        "preferences": preferences,
        "intent": "",
        "memory_saved": [],
        "clarify": ask,
        "react": (
            settings.react_default if react is None else react
        ),
        "cancelled": False,
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


def thread_config(thread_id: str | None) -> dict:
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
    # `callbacks` is what instruments the whole graph: every node becomes a
    # span, every edge the tree structure. One line, because LangGraph already
    # fires callbacks at each boundary -- the strongest practical payoff of
    # having built on it rather than hand-rolling the loop. Empty list when
    # tracing is unconfigured.
    #
    # Returned even with no thread_id, so an un-checkpointed run (the
    # evaluation harness) is still traced.
    config: dict = {"callbacks": tracing.callbacks()}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return config


async def run_agent(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    chat_context: str = "",
    session_id: str | None = None,
    preferences: str = "",
    thread_id: str | None = None,
    clarify: bool | None = None,
    react: bool | None = None,
) -> AgentResult:
    state = initial_state(
        question,
        top_k=top_k,
        document_ids=document_ids,
        owner_id=owner_id,
        multi_query=multi_query,
        chat_context=chat_context,
        session_id=session_id,
        preferences=preferences,
        clarify=clarify,
        react=react,
        resumable=bool(thread_id),
    )
    final = await get_graph().ainvoke(state, config=thread_config(thread_id))
    result = AgentResult(final)
    log.info(
        "agent_done",
        iterations=result.iterations,
        n_evidence=len(result.evidence),
        sufficient=result.sufficient,
        paused=result.paused,
    )
    return result


async def resume_agent(thread_id: str, decision: dict) -> AgentResult:
    """Continue a paused graph with a human's answer.

    `Command(resume=...)` replaces the graph's input entirely: LangGraph loads
    the checkpoint for `thread_id`, replays the interrupted node, and this time
    `interrupt()` returns `decision` instead of pausing. Nothing before it in
    the graph runs again -- the plan is not re-planned.

    Note the process boundary. The pause did not hold a coroutine open; the
    request that started this turn has long since returned. This is a fresh
    request, possibly on a different worker, and it works because the state was
    PERSISTED rather than parked in memory.
    """
    # Checked on the ARGUMENT, not on `thread_config(...) is None`.
    # `thread_config` now always returns a dict (it carries tracing callbacks
    # even without a thread), so the old `config is None` guard silently
    # stopped guarding anything.
    if not thread_id:
        raise ValueError("resume needs a thread_id")

    final = await get_graph().ainvoke(
        Command(resume=decision), config=thread_config(thread_id)
    )
    result = AgentResult(final)
    log.info(
        "agent_resumed",
        thread_id=thread_id,
        clarification=result.clarification,
        cancelled=result.cancelled,
        still_paused=result.paused,
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
    session_id: str | None = None,
    preferences: str = "",
    thread_id: str | None = None,
    clarify: bool | None = None,
    react: bool | None = None,
):
    """Yield (kind, node_name, payload) as each node completes.

    Node-level progress, not token-level. With responseSchema output there is
    no partial prose to stream -- you would be streaming half a JSON object.
    For an agent this is arguably the better signal anyway: "retrieving 2 of 3"
    says what is happening, where a token crawl only proves it is alive.

    Three kinds are yielded: `node` per completed node, `state` with the full
    state after each node, and `interrupt` if the graph pauses for a human.
    """
    state = initial_state(
        question,
        top_k=top_k,
        document_ids=document_ids,
        owner_id=owner_id,
        multi_query=multi_query,
        chat_context=chat_context,
        session_id=session_id,
        preferences=preferences,
        clarify=clarify,
        react=react,
        resumable=bool(thread_id),
    )
    async for item in _astream(state, thread_id):
        yield item


async def stream_resume(thread_id: str, decision: dict):
    """Resume a paused graph, streaming the rest of the run.

    Deliberately the same generator shape as `stream_agent`, so the SSE
    endpoint that consumes it needs no second code path -- from the client's
    point of view a resumed turn streams exactly like a fresh one.
    """
    async for item in _astream(Command(resume=decision), thread_id):
        yield item


async def _astream(inputs: Any, thread_id: str | None):
    """Shared streaming loop. `inputs` is a fresh state or a resume Command."""
    # Two stream modes at once: "updates" gives per-node deltas for progress,
    # "values" gives the full state after each node so the caller ends up with
    # the final state without a second aget_state() round-trip.
    async for mode, chunk in get_graph().astream(
        inputs, config=thread_config(thread_id), stream_mode=["updates", "values", "custom"]
    ):
        if mode == "custom":
            # Fine-grained progress published from INSIDE a node -- which
            # search is running, which rerank. Node-level events say
            # "retrieve" for six seconds and nothing about what is being
            # looked up. See services/progress.py.
            payload = (chunk or {}).get("progress") if isinstance(chunk, dict) else None
            if payload:
                yield "progress", payload.get("kind"), payload
            continue
        if mode == "updates":
            # A pause arrives on the "updates" channel under the same
            # `__interrupt__` key `ainvoke` uses, NOT as a node result -- so it
            # has to be filtered out here or the UI would render a progress
            # line for a node called "__interrupt__".
            for node, update in chunk.items():
                if node == "__interrupt__":
                    payload = interrupt_payload({"__interrupt__": update})
                    if payload is not None:
                        yield "interrupt", None, payload
                    continue
                yield "node", node, update
        elif mode == "values":
            yield "state", None, chunk

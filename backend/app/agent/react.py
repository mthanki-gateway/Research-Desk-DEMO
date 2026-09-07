"""The ReAct research node: native tool calling in a bounded loop.

    ┌──────────────────────────────────────────────────────┐
    │  model                                               │
    │    ├─ functionCall search_documents("competitors")   │  round 1
    │    └─ observation: "Bolt, Cadence, Dyna"             │
    │    ├─ functionCall search_web("Bolt funding")   ─┐   │
    │    ├─ functionCall search_web("Cadence funding") ├── │  round 2, PARALLEL
    │    └─ functionCall search_web("Dyna funding")   ─┘   │
    │    └─ no more calls → done; draft writes the answer  │  round 3
    └──────────────────────────────────────────────────────┘

WHAT THIS DOES THAT `plan` CANNOT

`plan` decomposes the question into every lookup UP FRONT. That is the right
shape when the lookups are knowable in advance, and it cannot express a
dependency: you cannot write "look up Bolt's funding" until a search has told
you Bolt exists. Here each round sees the previous round's results, so the
second query is chosen with knowledge the first produced.

Both paths are kept. The planned graph is what every recall and faithfulness
number in the evaluation harness measures, and replacing it would silently
invalidate all of them.

WHY IT IS ONE NODE AND NOT A SUBGRAPH

The loop is a plain `while` over model calls, which LangGraph does not need to
own -- there is no conditional routing to express and no state to merge between
rounds. Making each round a node would put the transcript in graph state and
snapshot the whole conversation to Postgres on every round, for no benefit. The
node boundary is where the useful state lives: evidence in, answer out.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog

from app.agent import tools
from app.agent.state import ResearchState
from app.config import get_settings
from app.services.llm import LLMError, get_llm
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

REACT_SYSTEM = """You GATHER SOURCES for a question using tools. You do not \
write the final answer -- a later step does that from what you collect.

YOUR SOURCES
You have two, and they are EQUALS. The user's uploaded documents hold their \
private material; the web holds everything public. Neither is a boundary on \
what can be answered, and neither is a fallback for the other. Choose by where \
the answer actually lives:
- specific to this user or their organisation -> their documents
- general knowledge, definitions, background, public figures, current events \
-> the web
- a question that spans both ("how does ours compare to the industry figure") \
-> BOTH, and gather each part from where it lives

If their documents do not cover something, that is not a dead end -- search the \
web for it. If the web is not available to you, say what is missing rather than \
filling the gap from memory.

HOW TO WORK
1. Search for what is needed, using whichever tool fits each part.
2. Read the results. If they are not relevant, search again with different \
wording rather than giving up.
3. When finding one fact DEPENDS on what another search returns, do them in \
order across separate turns -- you cannot look up a company before a search \
has told you its name. When several lookups are INDEPENDENT, request them \
together in one turn so they run at the same time.
4. Stop calling tools once the retrieved passages cover every part of the \
question, then reply with one short sentence saying what you found. That \
sentence is not shown to the user.

Never answer from memory, and never write citation markers -- your job is \
retrieval, not composition. Widening your sources does not weaken this: a \
claim you did not retrieve is still a claim you cannot make."""


async def react(state: ResearchState) -> dict:
    """Gather evidence by calling tools until the model stops asking."""
    settings = get_settings()
    question = state["question"]
    top_k = state.get("top_k") or settings.retrieval_top_k
    raw_ids = state.get("document_ids")
    document_ids = [uuid.UUID(d) for d in raw_ids] if raw_ids else None
    owner_id = state.get("owner_id")

    chat_context = state.get("chat_context") or ""
    opening = (f"{chat_context}\n\n" if chat_context else "") + f"Question: {question}"

    # The running transcript. It must contain the model's own tool REQUESTS as
    # well as their results -- append its `content` verbatim each round, or it
    # loses track of what it asked for and repeats itself.
    contents: list[dict] = [{"role": "user", "parts": [{"text": opening}]}]

    specs = tools.tool_specs()
    evidence: list[SearchHit] = []
    seen: set[uuid.UUID] = set()
    trace: list[dict] = []

    for round_no in range(1, tools.max_rounds() + 1):
        try:
            calls, text, model_content = await get_llm().generate_tools(
                contents, tools=specs, system=REACT_SYSTEM
            )
        except LLMError as exc:
            log.warning("react_failed", round=round_no, error=str(exc))
            trace.append({"round": round_no, "error": str(exc)})
            break

        if not calls:
            # No tools requested: gathering is finished. The prose is a
            # note-to-self and is deliberately DISCARDED -- `draft` composes
            # the answer, because only it sees the final numbering of the
            # accumulated evidence. Letting this model write the answer would
            # mean either no inline citations or invented ones.
            trace.append({"round": round_no, "done": True, "note": text[:160]})
            break

        contents.append(model_content)

        # Capped, and TRUNCATED rather than rejected: a greedy response asking
        # for twenty searches should still get its first few, because
        # cancelling the whole round teaches the model nothing.
        calls = calls[: tools.max_calls_per_round()]

        # Independent lookups run CONCURRENTLY. This is the payoff of the
        # model requesting several calls in one turn -- three competitor
        # lookups are one wall-clock step, not three.
        results = await asyncio.gather(
            *(
                tools.run_tool(
                    call.get("name", ""),
                    call.get("args") or {},
                    top_k=top_k,
                    document_ids=document_ids,
                    owner_id=owner_id,
                )
                for call in calls
            )
        )

        response_parts = []
        for call, (hits, observation) in zip(calls, results, strict=True):
            for hit in hits:
                # Dedupe across rounds. Without this the same chunk retrieved
                # by two phrasings occupies two citation slots and is handed to
                # the model twice, inflating its apparent importance.
                if hit.chunk_id not in seen:
                    seen.add(hit.chunk_id)
                    evidence.append(hit)
            response_parts.append(
                {
                    "functionResponse": {
                        "name": call.get("name", ""),
                        "response": {"result": observation},
                    }
                }
            )
            trace.append(
                {
                    "round": round_no,
                    "tool": call.get("name"),
                    "query": (call.get("args") or {}).get("query"),
                    "n_hits": len(hits),
                }
            )

        # Tool results go back as a `user` turn. That is the API's convention
        # for functionResponse parts, not a modelling choice.
        contents.append({"role": "user", "parts": response_parts})
    else:
        # Round cap reached while it was still asking for tools. Not an error:
        # whatever was gathered still goes to drafting, which beats discarding
        # six rounds of retrieval.
        log.info("react_round_cap", rounds=tools.max_rounds())
        trace.append({"round_cap": tools.max_rounds()})

    n_web = sum(1 for h in evidence if h.source == "web")
    # Two DIFFERENT numbers, and conflating them is what made an early version
    # of this log read "rounds=7" under a cap of 6: one round can request
    # several calls, so counting tool entries counts calls, not rounds.
    n_calls = len([t for t in trace if "tool" in t])
    log.info(
        "react_done",
        rounds=len({t["round"] for t in trace if "round" in t}),
        tool_calls=n_calls,
        n_evidence=len(evidence),
        n_web=n_web,
    )

    # EVIDENCE ONLY. `draft` writes the answer and `critique` reviews it, both
    # unchanged -- which is the whole reason this node gathers rather than
    # answers: citation numbering, `sources_used`, the no-sources-cited badge
    # and Tier 2 faithfulness all keep working with no special case for this
    # path.
    return {
        "evidence": evidence,
        "pending_queries": [],
        # Surfaced in the trace panel, so a ReAct turn shows what it searched
        # for -- the same slot the planned path fills with its sub-questions.
        "sub_questions": [t["query"] for t in trace if t.get("query")],
        "iterations": 0,
        "trace": [{"node": "react", "rounds": trace, "n_web": n_web}],
    }

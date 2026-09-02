"""The four nodes. Each takes state, returns only the keys it changed.

Every LLM call goes through a responseSchema, because Gemma 4 has no
function-calling and no thinking channel -- asked for prose it writes its whole
reasoning trace into the reply. Schemas are this app's substitute for tool use.
"""

from __future__ import annotations

import uuid

import structlog

from app.agent.state import ResearchState
from app.config import get_settings
from app.services.llm import LLMError, extract_string_list, get_llm
from app.services.retrieval import build_context, retrieve

log = structlog.get_logger()


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower().rstrip("?.")
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# --------------------------------------------------------------------------
# 1. plan
# --------------------------------------------------------------------------

# No `reasoning` field, deliberately. It was in here, and Gemma produced four
# correct sub_questions then degenerated inside `reasoning`, truncating the
# response and invalidating the whole object -- so a good plan was thrown away.
# Every extra field is another chance to derail; ask only for what is used.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Self-contained lookups needed to answer the question.",
        },
    },
    "required": ["sub_questions"],
}

PLAN_SYSTEM = """You break a research question into the separate lookups needed \
to answer it.

A document search can only retrieve a few passages per query, so a question \
asking about two different things must be split -- otherwise one of them is \
never retrieved.

Rules:
- One sub-question per distinct fact being requested.
- Each must be SELF-CONTAINED: no "it", "that", "the above", "the prior year". \
If conversation history is provided, resolve every reference against it. \
"And the year before?" must become "What was operating income in 2023?" -- the \
search engine has no memory of the conversation.
- If the question asks for only one thing, return it as a single sub-question, \
rephrased for search.
- Never invent requirements the question did not ask for.
- At most 3 sub-questions."""


async def plan(state: ResearchState) -> dict:
    question = state["question"]
    settings = get_settings()
    chat_context = state.get("chat_context") or ""

    # History goes to `plan` above all: this is where a follow-up like "and the
    # prior year?" gets turned into a standalone query. Retrieval itself is
    # stateless, so if the reference is not resolved here it never will be.
    history_block = (
        f"Conversation so far:\n{chat_context}\n\n" if chat_context else ""
    )
    prompt = (
        f"{history_block}"
        f"Break this question into the lookups needed to answer it.\n\n"
        f"Question: {question}"
    )
    try:
        # generate + lenient parse, not generate_json: a truncated response
        # still usually contains a complete sub_questions array, and losing a
        # good plan to a strict parse failure is worse than salvaging it.
        raw = await get_llm().generate(
            prompt,
            schema=PLAN_SCHEMA,
            system=PLAN_SYSTEM,
            temperature=0.0,
            max_output_tokens=600,
        )
        # Dedupe: the planner sometimes emits the same lookup twice (seen when
        # resolving a follow-up, where the resolved and original phrasings
        # collide). Each duplicate is a wasted embedding call and a wasted
        # retrieval slot.
        subs = _dedupe(extract_string_list(raw, "sub_questions"))[
            : settings.agent_max_subquestions
        ]
    except LLMError as exc:
        # Planning is an optimisation over asking the question as-is. Degrade to
        # the baseline rather than failing the request.
        log.warning("plan_failed", error=str(exc))
        subs = []

    if not subs:
        subs = [question]

    log.info("planned", n=len(subs), sub_questions=subs)
    return {
        "pending_queries": subs,
        "sub_questions": subs,
        "tried_queries": subs,
        "iterations": 0,
        "trace": [{"node": "plan", "sub_questions": subs}],
    }


# --------------------------------------------------------------------------
# 2. retrieve
# --------------------------------------------------------------------------


async def retrieve_node(state: ResearchState) -> dict:
    """Retrieve for every pending query. This is the node the cycle re-enters.

    The key difference from baseline RAG: it runs once per sub-question and the
    results accumulate, so a two-part question gets two retrievals even at
    top_k=1.
    """
    queries = state.get("pending_queries") or [state["question"]]
    top_k = state.get("top_k") or get_settings().retrieval_top_k
    raw_ids = state.get("document_ids")
    document_ids = [uuid.UUID(d) for d in raw_ids] if raw_ids else None

    gathered = []
    per_query = []
    for query in queries:
        hits = await retrieve(
            query,
            top_k=top_k,
            document_ids=document_ids,
            owner_id=state.get("owner_id"),
            multi_query=state.get("multi_query"),
        )
        gathered.extend(hits)
        per_query.append({"query": query, "n": len(hits)})

    log.info("retrieved_for_plan", n_queries=len(queries), n_hits=len(gathered))
    return {
        "evidence": gathered,  # merge_evidence dedupes against what we have
        "pending_queries": [],
        "trace": [{"node": "retrieve", "queries": per_query}],
    }


# --------------------------------------------------------------------------
# 3. draft
# --------------------------------------------------------------------------

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, with [n] citations inline.",
        },
        "sources_used": {"type": "array", "items": {"type": "integer"}},
        "unanswered": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parts of the question the sources do not cover.",
        },
    },
    "required": ["answer", "sources_used"],
}

DRAFT_SYSTEM = """You answer questions using ONLY the numbered sources provided.

Rules:
- Cite the source number in square brackets after each claim, e.g. [1] or [2].
- Quote figures exactly as they appear. Never round, adjust or infer a number.
- Answer every part of the question that the sources support.
- List any part you could NOT answer from the sources in `unanswered`. Do not \
guess, and do not use knowledge from outside the sources.
- Be concise."""


async def draft(state: ResearchState) -> dict:
    evidence = state.get("evidence") or []
    if not evidence:
        return {
            "draft": "No documents have been indexed yet. Upload one first.",
            "citations": [],
            "sufficient": True,  # nothing to retry with
            "trace": [{"node": "draft", "skipped": "no evidence"}],
        }

    # Ordering is deliberate: history first, then sources, then the question
    # LAST. Models attend most strongly to the start and end of a context, so
    # the question sits in the strongest position and history -- the least
    # critical part -- takes the weak middle.
    chat_context = state.get("chat_context") or ""
    history_block = (
        f"Conversation so far:\n{chat_context}\n\n" if chat_context else ""
    )
    prompt = (
        f"{history_block}"
        f"Sources:\n{build_context(evidence)}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer using only the sources above."
    )
    result = await get_llm().generate_json(
        prompt,
        schema=DRAFT_SCHEMA,
        system=DRAFT_SYSTEM,
        temperature=0.1,
        max_output_tokens=900,
    )

    answer = str(result.get("answer", "")).strip()
    citations = [int(n) for n in result.get("sources_used", []) if isinstance(n, int)]
    unanswered = [str(u) for u in result.get("unanswered", []) if isinstance(u, str)]

    log.info("drafted", cited=citations, unanswered=unanswered)
    return {
        "draft": answer,
        "citations": citations,
        "unanswered": unanswered,
        "trace": [
            {
                "node": "draft",
                "n_sources": len(evidence),
                "cited": citations,
                "unanswered": unanswered,
            }
        ],
    }


# --------------------------------------------------------------------------
# 4. critique
# --------------------------------------------------------------------------

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": (
                "True if the answer fully addresses the question and every "
                "claim is supported."
            ),
        },
        "assessment": {"type": "string", "description": "One or two sentences."},
        "missing": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Self-contained search queries that would fill the gaps.",
        },
    },
    "required": ["sufficient", "assessment"],
}

CRITIQUE_SYSTEM = """You review a draft answer against the sources it was built \
from.

CRITICAL CONTEXT: the sources shown are ONLY the passages retrieved so far, not \
the whole document library. More passages can still be fetched. So "the sources \
do not contain X" does NOT mean X is unavailable -- it usually means the right \
passage has not been retrieved yet.

Judge two things:
1. Completeness -- does the draft answer every part of the question?
2. Support -- is every claim backed by the cited sources?

Set sufficient=false when any part of the question is unanswered, and put \
specific self-contained SEARCH QUERIES in `missing` that would retrieve the \
absent facts. Queries, not instructions.

Set sufficient=true only when either:
- every part of the question is answered and supported, or
- the listed queries have already been tried and still returned nothing, so the \
information genuinely is not in the library.

If the draft cited NO sources at all, the retrieval phrasing almost certainly \
failed rather than the information being absent. In that case set \
sufficient=false and propose queries worded DIFFERENTLY from the ones already \
tried -- different vocabulary, synonyms, a fuller sentence.

Never repeat a query that has already been tried."""


async def critique(state: ResearchState) -> dict:
    settings = get_settings()
    iterations = state.get("iterations", 0) + 1
    evidence = state.get("evidence") or []

    tried = state.get("tried_queries") or []
    unanswered = state.get("unanswered") or []

    cited = state.get("citations") or []
    cited_note = (
        "The draft cited NO sources -- treat the retrieval phrasing as the "
        "likely problem.\n\n"
        if not cited
        else f"Sources the draft cited: {cited}\n\n"
    )

    prompt = (
        f"Question: {state['question']}\n\n"
        f"Draft answer:\n{state.get('draft', '')}\n\n"
        f"{cited_note}"
        f"Parts the drafter could not answer: {unanswered or 'none reported'}\n\n"
        f"Queries already tried: {tried}\n\n"
        f"Sources retrieved so far:\n{build_context(evidence)}\n\n"
        "Is this answer complete and fully supported?"
    )
    try:
        result = await get_llm().generate_json(
            prompt,
            schema=CRITIQUE_SCHEMA,
            system=CRITIQUE_SYSTEM,
            temperature=0.0,
            max_output_tokens=600,
        )
        sufficient = bool(result.get("sufficient", True))
        assessment = str(result.get("assessment", ""))
        missing = [
            str(m).strip()
            for m in result.get("missing", [])
            if isinstance(m, str) and m.strip()
        ]
    except LLMError as exc:
        # If the critic fails, accept the draft. Better a good answer with no
        # review than a 502.
        log.warning("critique_failed", error=str(exc))
        sufficient, assessment, missing = True, f"critique unavailable ({exc})", []

    # The drafter's own `unanswered` list is more reliable than the critic's
    # inference -- it knows exactly what it couldn't support. If it reported
    # gaps, trust that over a sufficient=true verdict.
    if unanswered and not missing:
        missing = list(unanswered)
        if sufficient:
            sufficient = False
            assessment += " (drafter reported unanswered parts)"

    # Never re-run a query that already came back empty-handed; that's how a
    # cycle becomes an infinite loop that spends quota to learn nothing.
    tried_lower = {q.strip().lower() for q in tried}
    missing = [m for m in missing if m.strip().lower() not in tried_lower][
        : settings.agent_max_subquestions
    ]

    # No new queries to run means nothing to retry with, whatever the verdict.
    if not sufficient and not missing:
        sufficient = True
        assessment += " (no untried follow-up queries, accepting draft)"

    log.info("critiqued", sufficient=sufficient, iterations=iterations, missing=missing)
    return {
        "sufficient": sufficient,
        "critique": assessment,
        "missing": missing,
        "pending_queries": missing,
        "tried_queries": missing,
        "iterations": iterations,
        "trace": [
            {
                "node": "critique",
                "sufficient": sufficient,
                "assessment": assessment,
                "missing": missing,
                "iteration": iterations,
            }
        ],
    }

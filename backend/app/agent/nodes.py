"""The four nodes. Each takes state, returns only the keys it changed.

Every LLM call goes through a responseSchema, because Gemma 4 has no
function-calling and no thinking channel -- asked for prose it writes its whole
reasoning trace into the reply. Schemas are this app's substitute for tool use.
"""

from __future__ import annotations

import re
import uuid

import structlog
from langgraph.types import interrupt

from app.agent.state import ResearchState
from app.config import get_settings
from app.services.llm import (
    LLMError,
    extract_int_list,
    extract_string,
    extract_string_list,
    get_llm,
)
from app.services.retrieval import build_context, document_outline, retrieve

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
# 0. clarify + ask_human  (human-in-the-loop)
#
# TWO nodes, not one, and the split is the whole point.
#
# `interrupt()` does not suspend a function mid-body. On resume LangGraph
# re-executes the node FROM THE TOP and `interrupt()` returns the supplied
# value on that second pass. So anything above the call runs TWICE.
#
# Deciding whether a question is ambiguous costs an LLM call and a SQL lookup.
# Putting that in the same node as the interrupt would pay for it again every
# time a user answered -- silently, and only in the human-in-the-loop path.
# So the expensive half lives in `clarify`, which never interrupts, and
# `ask_human` does nothing at all before its `interrupt()`.
# --------------------------------------------------------------------------

CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "ambiguous": {
            "type": "boolean",
            "description": "True only if the request cannot be searched as written.",
        },
        "question": {
            "type": "string",
            "description": "The clarifying question to ask the user. One sentence.",
        },
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "2-6 words."},
                    "description": {
                        "type": "string",
                        "description": "One short line on what this would cover.",
                    },
                },
                "required": ["label", "description"],
            },
        },
    },
    "required": ["ambiguous"],
}

CLARIFY_SYSTEM = """You decide whether a request to a document search system is \
specific enough to answer, and if not, what to ask.

You are given the section headings of the documents available. Every option you \
offer MUST correspond to something those headings show the documents actually \
cover -- an option the corpus cannot answer wastes the user's choice.

Set ambiguous=true ONLY when the request does not say enough to search on:
- it names a broad topic with no particular aspect ("tell me about the report")
- it explicitly defers the specifics ("the specific thing I want to know")
- it could mean two clearly different things the documents treat separately
- it asks for "details" or "information" without saying about what

Set ambiguous=false when the request names a specific fact, figure, event, \
section or entity -- even if it is short. "What was operating income?" is \
specific. So is "summarize this document": that is a clear instruction, not an \
ambiguous one.

When ambiguous=true:
- `question` is ONE short sentence asking what they want. Never apologise, \
never restate their question back to them.
- `options` are 2 to 4 CONCRETE choices, each a real aspect of the documents \
drawn from the headings. `label` is 2-6 words. `description` is one short line \
saying what that option would cover.
- Options must be genuinely different from each other, not rephrasings.

When ambiguous=false, return ambiguous=false and nothing else."""

# Actions the user may take. A closed set rather than free text, for the same
# reason `scope` is an enum in retrieval.py: an unrecognised value must have
# one obvious, safe meaning.
CLARIFY_ACTIONS = ("answer", "skip", "cancel")

_MAX_OPTIONS = 4
_MAX_LABEL = 60
_MAX_DESCRIPTION = 120

# Gemma leaks LaTeX into string fields. Measured, straight from a real option
# label rendered to the user:
#
#   $$ ext{Norwegian Cod Fisheries and Ancient Monuments}}{ ext{Description...
#
# It is reaching for \text{...} from maths-heavy training data, and the
# backslashes get eaten somewhere in the JSON round trip leaving `ext{`.
# Unwrap what is recoverable, then reject anything still carrying markup --
# an option label is 2-6 words of plain English, so a brace or a backslash in
# one means the model was not writing English.
_TEX_WRAPPER = re.compile(r"\\?(?:text|mathrm|mathit|bf)?\{([^{}]*)\}")
_TEX_NOISE = re.compile(r"[{}\\$]|\bext\b")


def _detex(value: str) -> str:
    """Unwrap \\text{...} and strip stray TeX punctuation. Best effort."""
    # Repeatedly unwrap, so nested \text{\text{x}} collapses rather than
    # leaving one layer behind.
    for _ in range(3):
        unwrapped = _TEX_WRAPPER.sub(r"\1", value)
        if unwrapped == value:
            break
        value = unwrapped
    return value.replace("$", "").strip()


def _clean_options(raw: object) -> list[dict]:
    """Keep only well-formed {label, description} pairs.

    Rejects rather than repairs anything still malformed after `_detex`. A
    garbled option is worse than a missing one: the user has to read it, decide
    it is nonsense, and lose confidence in the other three. Dropping it costs
    one choice; showing it costs trust.
    """
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue

        label = _detex(str(item.get("label", "")))
        if not label or len(label) > _MAX_LABEL or _TEX_NOISE.search(label):
            log.debug("clarify_option_rejected", label=str(item.get("label", ""))[:80])
            continue
        # Two options saying the same thing is not a choice.
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)

        description = _detex(str(item.get("description", "")))
        if _TEX_NOISE.search(description):
            description = ""

        out.append({"label": label, "description": description[:_MAX_DESCRIPTION]})
    return out[:_MAX_OPTIONS]


async def clarify(state: ResearchState) -> dict:
    """Decide whether to ask the user what they mean, and draft the options.

    Never interrupts, so it is safe for this to be the expensive node.

    Fails soft in every direction: a quota error, a malformed object, or fewer
    than two usable options all resolve to "not ambiguous", which is exactly
    the behaviour the graph has when clarification is switched off. A broken
    clarifier must never block a search.
    """
    question = state["question"]
    if not state.get("clarify"):
        return {}

    raw_ids = state.get("document_ids")
    document_ids = [uuid.UUID(d) for d in raw_ids] if raw_ids else None
    # Same grounding trick as query expansion: given the real headings, the
    # model offers aspects that exist instead of inventing plausible ones.
    outline = await document_outline(document_ids, state.get("owner_id"))
    outline_block = (
        "Sections in the documents being searched:\n"
        + "\n".join(f"- {h}" for h in outline)
        + "\n\n"
        if outline
        else ""
    )

    try:
        result = await get_llm().generate_json(
            f"{outline_block}Request: {question}",
            schema=CLARIFY_SCHEMA,
            system=CLARIFY_SYSTEM,
            temperature=0.0,
            max_output_tokens=700,
        )
    except LLMError as exc:
        log.warning("clarify_failed", error=str(exc))
        return {"trace": [{"node": "clarify", "skipped": f"llm error: {exc}"}]}

    options = _clean_options(result.get("options"))
    ask = str(result.get("question", "")).strip()

    # Two options is the minimum that constitutes a choice. One option is not a
    # question, it is a guess with extra steps -- better to just search.
    if not bool(result.get("ambiguous")) or len(options) < 2 or not ask:
        log.info("clarify_not_needed", n_options=len(options))
        return {"trace": [{"node": "clarify", "ambiguous": False}]}

    log.info("clarify_needed", question=ask, n_options=len(options))
    return {
        "pending_clarification": {"question": ask, "options": options},
        "trace": [
            {"node": "clarify", "ambiguous": True, "question": ask, "options": options}
        ],
    }


def apply_clarification(decision: object, original: str) -> dict:
    """Turn the user's answer into a state update. Pure, hence testable.

    Kept out of the node because `interrupt()` cannot run outside a graph, and
    this is where the interesting logic is.

    A malformed decision is treated as SKIP, never as an error. Failing the
    turn would throw away work because a client sent the wrong shape, and skip
    is what the graph would have done with clarification switched off -- so a
    bad payload degrades to the pre-existing behaviour rather than a new one.
    """
    if not isinstance(decision, dict):
        return {"trace": [{"node": "ask_human", "decision": "skip (malformed)"}]}

    action = str(decision.get("action", "skip")).strip().lower()

    if action == "cancel":
        return {
            "cancelled": True,
            "pending_queries": [],
            # Something must land in `draft`: it is what the API reads as the
            # answer, and the graph is about to skip every node that writes it.
            "draft": "Stopped without searching, at your request.",
            "sufficient": True,
            "trace": [{"node": "ask_human", "decision": "cancelled"}],
        }

    answer = str(decision.get("answer", "")).strip()
    if action != "answer" or not answer:
        # "Search anyway" -- an explicit, reasonable choice, not a failure.
        return {"trace": [{"node": "ask_human", "decision": "skipped"}]}

    return {
        # `question` is REWRITTEN, and this is the point of the whole feature:
        # everything downstream -- plan, retrieval, drafting -- sees the
        # clarified request. Overwrite works because `question` carries no
        # reducer. The original is preserved separately for the transcript.
        "question": f"{original}\n\nSpecifically: {answer}",
        "original_question": original,
        "clarification": answer,
        "trace": [{"node": "ask_human", "decision": "answered", "answer": answer}],
    }


async def ask_human(state: ResearchState) -> dict:
    """Put the question to the user and wait.

    NOTHING happens before `interrupt()` -- see the section comment above. This
    node exists only to hold that call.
    """
    pending = state.get("pending_clarification") or {}
    original = state["question"]

    # `interrupt()` raises internally. The graph stops here, the checkpointer
    # persists everything, and the process is FREE -- this is not a coroutine
    # blocked on a socket. The answer can arrive minutes later, from a
    # different worker, after a deploy. Without a checkpointer it cannot work.
    decision = interrupt(
        {
            "type": "clarification",
            "question": pending.get("question", ""),
            "options": pending.get("options", []),
            "original": original,
            "actions": list(CLARIFY_ACTIONS),
        }
    )

    update = apply_clarification(decision, original)
    log.info("clarification_answered", clarification=update.get("clarification"))
    return update


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
    # `generate` + lenient parsing, NOT `generate_json`.
    #
    # Gemma's characteristic failure is a repetition loop that runs until the
    # token cap, which truncates the JSON. A strict parse then discarded an
    # answer that was already complete before the loop began. Measured, from a
    # real turn: "...which specific pyramid you are asking about. [No source
    # provided for this clarification/refusal/unanswered part of the
    # question/question/question/..." -- a usable first sentence followed by
    # noise, surfaced to the user as `model returned invalid JSON`.
    #
    # This is the same trade `plan` already makes, and `draft` should have made
    # it first: it is the node whose output the user actually reads.
    try:
        raw = await get_llm().generate(
            prompt,
            schema=DRAFT_SCHEMA,
            system=DRAFT_SYSTEM,
            temperature=0.1,
            max_output_tokens=900,
        )
    except LLMError as exc:
        # Quota, safety block, recitation. Nothing to salvage, but a turn that
        # says why beats a 502 -- the retrieved evidence is still on screen.
        log.warning("draft_failed", error=str(exc))
        return {
            "draft": f"The answer could not be generated ({exc}).",
            "citations": [],
            "sufficient": True,  # a retry would hit the same wall
            "trace": [{"node": "draft", "error": str(exc)}],
        }

    answer = extract_string(raw, "answer")
    citations = extract_int_list(raw, "sources_used")
    unanswered = extract_string_list(raw, "unanswered")

    if not answer:
        # Parsed, but there is no answer in it -- degeneration from the first
        # token, or the schema ignored entirely.
        log.warning("draft_unusable", raw=raw[:200])
        return {
            "draft": "The model did not return a usable answer. Try asking again.",
            "citations": [],
            "sufficient": True,
            "trace": [{"node": "draft", "error": "no answer field"}],
        }

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

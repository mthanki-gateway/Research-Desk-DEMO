"""The four nodes. Each takes state, returns only the keys it changed.

Every LLM call goes through a responseSchema.

That began as a necessity -- Gemma has no function calling and no thinking
channel, so asked for prose it wrote its whole reasoning trace into the reply.
It is now a CHOICE: every Gemini Flash model here supports native
functionDeclarations, verified live. Schemas are kept because these nodes are
not tool calls -- the graph decides control flow, the model fills in fields --
and because a schema is what stops reasoning leaking into user-facing text.
"""

from __future__ import annotations

import re
import uuid

import structlog
from langgraph.types import interrupt

from app.agent.state import ResearchState
from app.config import get_settings
from app.services import preferences, websearch
from app.services.llm import (
    LLMError,
    extract_bool,
    extract_int_list,
    extract_object_list,
    extract_string,
    extract_string_list,
    get_llm,
    is_repetitive,
)
from app.services.pool import get_pool
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

A search can only retrieve a few passages per query, so a question asking about \
two different things must be split -- otherwise one of them is never retrieved.

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
    # All three REQUIRED. With only `ambiguous` required, the model emitted
    # `{"ambiguous":true,"options":[...]}` and skipped `question` entirely --
    # so the node had good options and no header sentence, and rejected the
    # lot. A responseSchema is a contract; leaving a field optional is telling
    # the model it may omit it.
    "required": ["ambiguous", "question", "options"],
}

CLARIFY_SYSTEM = """You decide whether to ASK THE USER A QUESTION instead of \
answering. Almost always, the answer is no.

ASKING IS A FAILURE MODE, NOT A COURTESY. It stops the user, costs them a \
round trip, and makes the assistant feel obstructive. Searching and being \
partly wrong is nearly always better: they can see what you did and correct it \
in one line.

THE TEST: could ANY reasonable answer be given without asking?
If yes -- ambiguous=false. Always.

APPLY THESE FIRST, and they settle nearly every case:

1. TWO POSSIBLE MEANINGS AND BOTH ARE ANSWERABLE -> ANSWER BOTH.
   Never ask someone to choose between things you could simply cover.
   "tell me about pyramids and also the pain points" -> answer both.
   "you never answered the other question" -> answer the outstanding one.

2. A BROAD REQUEST IS NOT AMBIGUOUS, it is broad. Summarise, cover the main \
aspects, and say what you covered.
   "tell me about the report" -> summarise the report.

3. A FOLLOW-UP TAKES ITS SUBJECT FROM THE CONVERSATION. Read the history.
   "and the year before?", "what about the other one", "now try the internet".

4. AN INSTRUCTION ABOUT HOW TO ANSWER IS NOT A QUESTION ABOUT WHAT TO ANSWER.
   "summarise that", "answer from the web instead", "try again", "be shorter".

5. IF YOU WOULD HAVE TO INVENT THE OPTIONS, there is nothing to ask about.

ambiguous=true ONLY when BOTH of these hold:
- there is genuinely NO sensible default -- not merely several possibilities, \
but no way to pick or combine them; AND
- answering the wrong reading would be COSTLY or MISLEADING, not just \
imperfect. Deleting something, a figure that would be wrong in a way the user \
could not spot, or a question that names an entity you cannot identify at all.

Concretely, that is nearly only this shape: the request defers its own \
specifics ("the specific thing I want to know", "you know the one") and the \
conversation does not say which.

When in doubt, ambiguous=false. A user who wanted something narrower will say \
so; a user who was stopped for no reason just loses time.

When ambiguous=true:
- `question` is ONE short sentence asking what they want. Never apologise, \
never restate their question back to them.
- `options` are 2 to 4 CONCRETE choices, every one anchored in something REAL: \
a section heading from their documents, or the specific subject already under \
discussion. NEVER offer generic categories -- "Latest Technology Trends", \
"Global Financial Markets", "Historical Architecture" are worthless, because \
they tell the user nothing and the assistant cannot act on them. If you cannot \
name 2 concrete, anchored options, return ambiguous=false and let the search \
run.
- Options must be genuinely different from each other, not rephrasings.

When ambiguous=false, return ambiguous=false and nothing else."""

# Actions the user may take. A closed set rather than free text, for the same
# reason `scope` is an enum in retrieval.py: an unrecognised value must have
# one obvious, safe meaning.
CLARIFY_ACTIONS = ("answer", "skip", "cancel")

# Used when the model produced usable options but no question sentence.
_DEFAULT_ASK = "What would you like to know about?"

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
        "Sections in the user's own documents (not the only thing searchable):\n"
        + "\n".join(f"- {h}" for h in outline)
        + "\n\n"
        if outline
        else ""
    )

    # Lenient parsing, NOT generate_json. Measured failure: Gemma degenerated
    # inside the THIRD option{APOS}s description ("Information about the about
    # the...") and truncated the document, so a strict parse discarded
    # `ambiguous: true` AND two complete, usable options -- and the turn ran
    # without pausing. A repetition loop in a field nobody reads silently
    # disabled the whole feature.
    # The conversation, which this node used to judge WITHOUT.
    #
    # That omission produced the worst clarifying question in the app's
    # history. Asked "then answer from the internet!" as a follow-up about a
    # pyramid, `clarify` saw six words and no topic, correctly concluded it
    # could not tell what was wanted, and invented three categories out of thin
    # air: "Latest Technology Trends", "Global Financial Markets", "Historical
    # Architecture". Every other node already gets history -- `plan` needs it
    # to resolve "the year before", `draft` to resolve pronouns -- and the one
    # node whose entire job is judging whether a request is clear was the one
    # node judging it out of context.
    chat_context = state.get("chat_context") or ""
    history_block = f"Conversation so far:\n{chat_context}\n\n" if chat_context else ""

    # Stated, not assumed. An option about public information is useless if the
    # web cannot actually be reached, and the clarifier has no other way to
    # know -- it would offer "look it up online" on a deployment where that is
    # impossible.
    coverage = (
        "Searchable: the user's own documents, and the public web.\n\n"
        if websearch.enabled()
        else "Searchable: the user's own documents ONLY -- web search is not "
        "configured, so do not offer options that require public "
        "information.\n\n"
    )

    try:
        raw = await get_llm().generate(
            f"{history_block}{outline_block}{coverage}Latest request: {question}",
            schema=CLARIFY_SCHEMA,
            system=CLARIFY_SYSTEM,
            temperature=0.0,
            # 700 truncated mid-options on a five-document corpus. The
            # descriptions are the bulk of the response and the model is
            # verbose in them, so this is headroom rather than a fix -- the
            # salvage above is what makes truncation survivable.
            max_output_tokens=900,
        )
    except LLMError as exc:
        log.warning("clarify_failed", error=str(exc))
        return {"trace": [{"node": "clarify", "skipped": f"llm error: {exc}"}]}

    options = _clean_options(extract_object_list(raw, "options"))
    # Fall back rather than discard. If two or more grounded options survived,
    # a missing header sentence is not a reason to throw them away -- the
    # options ARE the question.
    ask = extract_string(raw, "question") or _DEFAULT_ASK

    # Two options is the minimum that constitutes a choice. One option is not a
    # question, it is a guess with extra steps -- better to just search.
    if not extract_bool(raw, "ambiguous") or len(options) < 2 or not ask:
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


# Phrases that describe WHERE a thing should have been, not WHAT it is.
#
# The critic writes gaps as sentences about the corpus -- "A height for the Red
# Pyramid FROM YOUR DOCUMENTS" -- and those sentences were being embedded and
# searched verbatim. Every one of these words pulls the query vector towards
# passages that talk about documents and away from passages that state a
# height. The retry then returns nothing, and the turn concludes the fact does
# not exist.
_META = re.compile(
    r"\b("
    r"(?:in|from|within|according to|per)\s+"
    r"(?:the\s+|your\s+|these\s+|his\s+|her\s+|their\s+)?"
    r"(?:user'?s?\s+)?"
    r"(?:own\s+)?"
    r"(?:provided\s+|uploaded\s+|attached\s+|supplied\s+)?"
    r"(?:documents?|files?|corpus|sources?|library|passages?|context)"
    r"|document'?s?\s+(?:mention|statement)s?\s+of"
    r")\b",
    re.IGNORECASE,
)

# A leading article or hedge on a NOUN PHRASE gap: "A height for...",
# "The original height of...", "Any mention of...".
_LEAD = re.compile(
    r"^\s*(?:a|an|the|any|some|specific|explicit|separate|exact)\s+", re.IGNORECASE
)


def as_query(text: str) -> str:
    """Turn a critic's gap description into something worth searching for.

    A gap is written to be READ by the drafter -- "A separate height for the
    Great Pyramid of Khufu distinct from the Great Pyramid of Giza in your
    documents" -- and was being sent to the embedder unchanged. Most of that
    sentence is about the SHAPE of the omission, and it dominates the vector.

    Conservative on purpose. It strips phrases that refer to the corpus itself
    and a leading article, and leaves everything else alone: a gap that is
    already a decent query must come through unharmed, and over-trimming a
    query is as bad as not trimming it.
    """
    out = _META.sub(" ", text or "")
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.;:")
    out = _LEAD.sub("", out).strip()
    # If the strip ate everything, the original was pure meta-language and the
    # original is still the better of two bad options.
    return out or (text or "").strip()


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

    # Chunks already shown to the model this turn are EXCLUDED on a retry.
    #
    # Without this a second pass is close to wasted: the critique asks for a
    # different angle, retrieval obliges with differently worded queries, and
    # `merge_evidence` then dedupes the results back down to the set the first
    # pass already had. The cycle costs a full round of calls and adds nothing.
    # Excluding what was seen is what lets the retry actually differ.
    #
    # Only on a RETRY -- on the first pass the set is empty and this is a no-op.
    seen = set(state.get("seen_chunk_ids") or ())

    # A RETRY, which is what `seen` being non-empty means.
    retry = bool(seen)

    gathered = []
    per_query = []
    for raw_query in queries:
        query = as_query(raw_query)
        hits = await retrieve(
            query,
            top_k=top_k,
            document_ids=document_ids,
            owner_id=state.get("owner_id"),
            multi_query=state.get("multi_query"),
            exclude_chunk_ids=seen,
        )
        gathered.extend(hits)
        entry = {"query": query, "n": len(hits)}

        # THE WEB IS RETRIED TOO, and this is the fix for a whole class of
        # wrong answer.
        #
        # Measured on "the height of the Great Pyramid and the Red Pyramid":
        # the critic correctly identified the gap -- no Red Pyramid height --
        # and this node then re-searched THE SAME CORPUS with a reworded query,
        # got nothing, and the turn concluded "your documents do not give the
        # height of the Red Pyramid". They never would. The corpus does not
        # contain it and no rewording can make it. One web search returns
        # "Height 105 m (344 ft)" as the first result.
        #
        # Re-asking a corpus that has already been asked is the one retry that
        # cannot possibly succeed: the critic raised the gap precisely because
        # the corpus did not answer it. Only a different SOURCE can.
        #
        # Retry only. On the first pass the ReAct loop already has `search_web`
        # as a tool and chooses for itself; duplicating it here would double
        # every web call on every turn.
        if retry and not document_ids and websearch.enabled():
            try:
                web_hits = await websearch.search_web(query, limit=top_k)
                gathered.extend(web_hits)
                entry["web"] = len(web_hits)
            except Exception as exc:  # noqa: BLE001 - the web is optional
                # A failed web search must not fail the turn. The document
                # results are still worth drafting from.
                log.warning("retry_web_failed", query=query[:80], error=str(exc))
                entry["web"] = 0

        per_query.append(entry)

    log.info(
        "retrieved_for_plan",
        n_queries=len(queries),
        n_hits=len(gathered),
        n_excluded=len(seen),
        retry=retry,
    )
    return {
        "evidence": gathered,  # merge_evidence dedupes against what we have
        "pending_queries": [],
        "trace": [
            {"node": "retrieve", "queries": per_query, "excluded": len(seen)}
        ],
    }


# --------------------------------------------------------------------------
# 3. draft
# --------------------------------------------------------------------------

# The rules for text the USER READS: what to report, and how to shape it.
#
# They lived only in DRAFT_SYSTEM, and `resolve` -- which produces the final
# answer whenever the critique loop exhausts its budget -- had none of them.
# So the turns most likely to be long and hard to read were exactly the ones
# rendered as a single unbroken block, and the formatting work looked like it
# had silently stopped applying. Measured on "tell me all about pyramids":
# two critique cycles, `iteration_cap_reached`, then `resolve` wrote the
# answer with zero line breaks in it.
#
# One constant appended to both, rather than the text copied into each: a
# rule improved in one place and not the other is how they drifted apart the
# first time.
ANSWER_RULES = """SAY WHAT YOU DID, FIRST

Open with ONE short sentence reporting the work, then a blank line, then the \
answer. The reader cannot see the retrieval, so without this they cannot tell \
a thin answer from a thin corpus -- "I don't know" reads identically whether \
nothing was searched or everything was.

The "Search coverage" line below says what this turn ACTUALLY did. Report that \
and nothing else. It is a record of work PERFORMED, never a list of what was \
available -- if it does not say the web was searched, the web was not \
searched, and claiming otherwise is a false statement about your own \
behaviour.

Looking up collection metadata -- how many documents exist, their names, their \
sizes -- is NOT a search. Nothing inside them was read, so do not report it as \
reading them. Say precisely what happened:

    I checked your document list without searching inside the documents.

    I searched your documents.

    I searched your documents and the web.

    I searched your documents and found nothing on this, so the answer below \
is from the web.

If a memory update is reported to you below, SAY SO in that same opening -- \
plainly, and quoting what was stored:

    I've remembered that you always want a table when comparing numbers, and \
searched your documents and the web.

Never claim a search you were not told about, never pad this into a paragraph, \
and never repeat it at the end. If a later turn asks what you searched, answer \
from what the coverage line said -- never dismiss a report you made as \
boilerplate.

FORMAT IT SO IT CAN BE READ

Write markdown, and structure it. A correct answer delivered as one unbroken \
block is a worse answer -- nobody reads it, and the parts they wanted are \
buried.

Use REAL line breaks -- an actual blank line between paragraphs, each list \
item on its own actual line. Never type the characters backslash-n; they \
appear on screen exactly as written and the answer reads "records [5] \
.\\n\\nRegarding the operations...". Structure that is not on separate lines \
does not survive either: "intro: - first - second" on one line is the other \
half of this same failure.

- PARAGRAPHS FIRST. This matters more than everything below it combined. One \
idea per paragraph, separated by a blank line, and never more than about five \
sentences before a break. Prose broken into paragraphs is the DEFAULT shape of \
an answer; lists, tables and headings are exceptions you reach for when the \
content genuinely has that shape.
- ONE PARAGRAPH PER PART OF THE QUESTION. If the user asked two things, answer \
the first, break, then answer the second -- in the order they asked. Do not \
weave the parts together into one paragraph, and do not answer them in one \
paragraph just because both answers are short. A question with three parts \
gets at least three paragraphs.
- A LIST when you are enumerating. If you catch yourself writing "(1) ... (2) \
... (3)" inside a sentence, those are list items -- put each on its own line \
starting with "- " or "1. ". Do not inline them. But do not reach for a list \
where two sentences would do: a list of three fragments is harder to read \
than the paragraph it replaced.
- A TABLE when you are comparing things across the same dimensions -- figures \
by period, options against criteria, documents against what each covers. Use \
markdown pipes:

    | Metric | 2023 | 2024 |
    | --- | --- | --- |
    | Gross margin | 58.7% [1] | 62.1% [1] |

- A `### heading` only when the answer covers genuinely separate topics. Two \
paragraphs do not need headings.
- **Bold** for a figure or term the reader is looking for. Sparingly; bolding \
everything is the same as bolding nothing.

Citations go INSIDE the structure -- at the end of the sentence, the list item, \
or the table cell they support. A list of citations at the end tells the reader \
nothing about which claim came from where.

Be concise: structure is not permission to write more. Prefer a short answer \
with three clear paragraphs over a long one with three headings.

When in doubt, use a paragraph break."""


DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            # ASK FOR LINE BREAKS, NEVER FOR THE ESCAPE THAT ENCODES THEM.
            #
            # Both failures here were measured, and they are opposite:
            #
            #   silent   Told nothing, the model emitted NO line breaks at all
            #            -- a literal newline is illegal inside a JSON string
            #            literal, so every structural instruction in
            #            DRAFT_SYSTEM evaporated at the envelope and answers
            #            arrived as "...pain points: - Customer concentration
            #            ... - Hardware supply chain...", markers intact and
            #            not one \n in the field.
            #
            #   literal  Told to "write the two-character escape \n", it
            #            escaped the BACKSLASH -- emitting "\\n" in the JSON,
            #            which decodes to the two visible characters \ and n.
            #            The user read "records [5] .\n\nRegarding the
            #            technical operations..." on screen.
            #
            # Encoding is the serialiser's job and it does it correctly. Ask
            # only for the intent -- blank lines between paragraphs -- and let
            # the JSON layer represent them. `_unescape_newlines` in llm.py is
            # the net under the second failure.
            "description": (
                "The answer in MARKDOWN, with [n] citations inline. Use real "
                "line breaks: a blank line between paragraphs, and each list "
                "item on its own line. Do not write the characters backslash-n "
                "-- press a real newline and let the encoding handle it. An "
                "answer that is one unbroken block is wrong."
            ),
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

DRAFT_SYSTEM = (
    """You are a research assistant. You answer from the numbered \
sources provided, and you are helpful about what they do and do not contain.

Each source is marked with its KIND:
- `document` -- one of the user's own uploaded files
- `web` -- a public page, with its url

Both are legitimate sources and neither outranks the other. Treat them as one \
pool of evidence.

WHAT MUST COME FROM THE SOURCES
Every FACT you assert -- figures, dates, events, findings, quantities, names of \
things that happened. Cite each one as [1], [2]. Never invent a fact, never \
adjust a number, and never present your own knowledge as though a source said \
it.

WHAT YOU MAY USE YOUR OWN KNOWLEDGE FOR
Understanding the question and connecting it to the sources. Specifically:
- RECOGNISING THAT TWO NAMES MEAN THE SAME THING. If the user asks about \
"Akhet Khufu" and a source describes the largest tomb built for Khufu at Giza, \
those are the same monument -- say so and answer from that source. Refusing \
because the exact string is absent is a failure, not caution.
- Knowing what a term, acronym, place or person is, well enough to find the \
relevant source.
- One clause of framing so the answer makes sense.
Mark this kind of statement as your own -- "commonly known as", "this is the \
same structure as" -- and do NOT put a citation on it. A citation means "a \
source said this".

WHEN THE SOURCES FALL SHORT
Never answer with a bare refusal. Say what IS there and what is missing, in \
that order: "Your documents describe X and Y [1] but do not give Z." A reader \
should learn something from every answer, including the answers that cannot be \
complete. Put the genuinely missing parts in `unanswered`.

ATTRIBUTION
- Name the origin in the sentence. For the web, the site or publication ("per \
the Postgres documentation [3]"); for their own files, the file or section \
("the Q1 review [1]"). A reader must be able to tell which claims rest on \
their own material and which on a public page, without opening anything.
- Never blur the two. Do not let a web figure stand as if it came from their \
documents, and do not present their internal numbers as public knowledge.
"""
    # Appended rather than inlined, so `resolve` gets the identical block.
    + "\n\n"
    + ANSWER_RULES
)


# "[1]", "[2, 3]", "[1][4]" -- the shapes the drafter actually produces. The
# frontend already parses the same markers to render citation chips, so keeping
# this permissive is what stops the two views disagreeing.
_CITE_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _cited_in_text(answer: str, n_sources: int) -> list[int]:
    """Citation numbers appearing inline, in order, deduped.

    Numbers out of range are DROPPED rather than kept: a [7] against six
    sources is a hallucinated citation, and passing it on would have the UI
    resolve it to nothing or -- worse -- to the wrong source.
    """
    out: list[int] = []
    for group in _CITE_MARKER.findall(answer):
        for part in group.split(","):
            n = int(part.strip())
            if 1 <= n <= n_sources and n not in out:
                out.append(n)
    return out


def _memory_block(state: ResearchState) -> str:
    """What was stored this turn, for whichever node writes the answer.

    SHARED BY `draft` AND `resolve`, and that sharing is the point. It lived
    only in `draft`, so a turn that stored an instruction and then exhausted
    the critique loop had its answer written by `resolve` -- which knew nothing
    about the memory and never mentioned it. Measured: "tell me all about
    pyramids ... and update my preference to be more stoic" stored the
    preference and answered without a word about it, which from the outside is
    indistinguishable from having ignored the request.

    Reports BOTH outcomes, and keeps them apart. An instruction that was
    already in force is skipped rather than stored, and saying nothing about it
    is how a restatement comes back looking ignored -- which is precisely what
    makes someone state it a third time. But it must not be reported as newly
    saved either, or the assistant claims to have done something it explicitly
    declined to do.

    Empty string when neither happened, so the prompt gains no line at all on
    the overwhelming majority of turns -- a "nothing was remembered" note is
    itself something the model reasons about, and small models apologise for
    it.
    """
    saved = state.get("memory_saved") or []
    known = state.get("memory_known") or []
    if not saved and not known:
        return ""

    parts = ["Memory, this turn -- report this in your opening sentence:"]
    if saved:
        listed = "\n".join(f"- {s}" for s in saved)
        parts.append(f"STORED, now in force. Quote it back:\n{listed}")
    if known:
        listed = "\n".join(f"- {s}" for s in known)
        parts.append(
            "ALREADY IN FORCE, so nothing was added. Say it was already "
            f"remembered -- do NOT say you saved it:\n{listed}"
        )
    return "\n\n".join(parts) + "\n\n"


def _coverage_block(state: ResearchState) -> str:
    """What was ACTUALLY searched this turn, as a fact rather than a capability.

    THE BUG THIS REPLACES

    This line used to read "Both the user's documents and the web were
    SEARCHABLE for this question" whenever web search was configured. The
    answer rules directly above it tell the model to report its work from this
    line -- so on a turn that touched only the metadata tools, the model dutifully
    opened with "I searched your documents and the web." Nothing had been
    searched at all. Asked about it on the next turn it correctly said no web
    search happened, which reads as the model contradicting itself when in fact
    it was told two different things.

    Availability is not use. This reports use, derived from the evidence that
    actually came back, and states availability only where its absence changes
    what an honest answer can claim.
    """
    evidence = state.get("evidence") or []
    used_web = any(getattr(h, "source", "document") == "web" for h in evidence)
    used_docs = any(getattr(h, "source", "document") != "web" for h in evidence)
    used_metadata = bool(state.get("corpus_facts"))

    did: list[str] = []
    if used_docs:
        did.append("searched the user's documents")
    if used_web:
        did.append("searched the web")
    if used_metadata:
        # Named precisely, because it is the thing the model kept mis-reporting
        # as a search. Looking up how many documents exist is not retrieval.
        did.append(
            "looked up collection metadata (document names, counts, sizes) "
            "WITHOUT searching their contents"
        )

    if did:
        done = "This turn: " + "; ".join(did) + "."
    else:
        done = "This turn: nothing was searched or looked up."

    # Only stated when it changes what may honestly be claimed. On a turn that
    # did search the web, saying the web is available is noise.
    caveat = ""
    if not websearch.enabled() and not used_web:
        caveat = (
            " Web search is NOT configured on this deployment, so public "
            "information could not be looked up at all. If part of the question "
            "needs it, say that it is not in their documents AND that web "
            "search is not enabled -- do not imply the information does not "
            "exist."
        )
    elif websearch.enabled() and not used_web:
        caveat = " The web was available but was NOT used; do not claim it was."

    return f"Search coverage -- {done}{caveat}\n\n"


def _facts_block(state: ResearchState) -> str:
    """Metadata results, labelled so they are not mistaken for sources.

    The label does real work. Dropped into the prompt unmarked, these read as
    just more context and the drafter either attaches a citation to them --
    picking whichever [n] happens to be nearby, which is a fabricated citation
    for a figure no source contains -- or omits the figure as uncitable, which
    loses the very thing that was asked for.

    Shared by `draft` and `resolve` for the same reason `_memory_block` is:
    `resolve` rewrites the answer from scratch when the critique loop runs out
    of budget, so anything only `draft` was told would be discarded one node
    later.
    """
    facts = state.get("corpus_facts") or []
    if not facts:
        return ""
    joined = "\n\n".join(facts)
    return (
        "FACTS FROM THIS APPLICATION'S OWN DATABASE. These are not sources and "
        "carry no number -- state them directly and NEVER put a [n] citation "
        "on them. They are exact; do not round, hedge or re-describe them as "
        "approximate:\n"
        f"{joined}\n\n"
    )


async def draft(state: ResearchState) -> dict:
    settings = get_settings()
    evidence = state.get("evidence") or []
    # NO PASSAGES IS NOT THE SAME AS NOTHING TO SAY.
    #
    # A metadata tool answers without retrieving anything -- "you have 5
    # documents, 39 passages, 18,506 characters" is a complete answer backed by
    # no passage at all. Bailing out here on an empty evidence list would have
    # replied "nothing was found" to a question this app had already answered
    # exactly, which is the same class of mistake as sending "hi" to retrieval.
    if not evidence and not (state.get("corpus_facts") or []):
        # Deliberately does NOT say "upload a document". That was right while
        # the corpus was the whole universe, but a search can now come back
        # empty with documents present and the web searched -- and telling
        # someone to upload a file when the real problem was phrasing sends
        # them off to fix the wrong thing.
        return {
            "draft": (
                "Nothing was found for this question. If no documents have "
                "been uploaded yet, add one; otherwise try rephrasing."
            ),
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
    # A REGENERATION carries the critic's objection, and nothing else changes.
    #
    # This is the remedy for `unsupported_claim`: the passages were right and
    # the sentence overstated them, so the fix is to rewrite from the SAME
    # evidence. Re-retrieving cannot help -- the words the draft needed were
    # never in any source -- which is why this path exists separately from the
    # critique -> retrieve cycle and has its own budget.
    overclaims = state.get("unsupported_claims") or []
    redo_block = ""
    if overclaims:
        listed = "\n".join(f"- {c}" for c in overclaims)
        redo_block = (
            "A reviewer found these statements go further than the sources "
            f"support:\n{listed}\n\n"
            "Rewrite the answer keeping everything the sources DO support, and "
            "either drop each statement above or weaken it to what the source "
            "actually says. Do not add new claims.\n\n"
        )

    # Omitted entirely rather than left empty when a metadata-only turn brought
    # no passages. A bare "Sources:" heading with nothing under it reads as a
    # failed search, and the drafter opens by apologising for finding nothing
    # -- directly above the figures that answer the question.
    sources_block = f"Sources:\n{build_context(evidence)}\n\n" if evidence else ""

    prompt = (
        f"{history_block}"
        f"{sources_block}"
        f"{_coverage_block(state)}"
        f"{_memory_block(state)}"
        f"{_facts_block(state)}"
        f"{redo_block}"
        f"Question: {state['question']}\n\n"
        "Answer from the sources above, per your instructions."
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
        # The ANSWER model -- the strongest one available, used only here and
        # in baseline RAG. This is the text the user reads, and it is 1-2 calls
        # per turn, which is what makes a 5 rpm budget affordable where the
        # four-call agent would not fit.
        raw = await get_pool("answer").generate(
            prompt,
            schema=DRAFT_SCHEMA,
            system=DRAFT_SYSTEM + (state.get("preferences") or ""),
            temperature=0.1,
            # Profile-dependent, because the right ceiling differs by model.
            #
            # 900 was too tight on Gemini, and the failure was silent rather
            # than loud: `sources_used` is emitted AFTER `answer`, so a long
            # answer that hit the cap lost its citation list entirely -- lenient
            # parsing salvaged the prose and returned `citations=[]`, surfacing
            # as "no sources cited" on an answer that cited in every sentence.
            # Measured on a two-part question spanning a document and a web
            # page: the answer cut off at "According to the Root Cause section
            # of acme-incident-2024".
            #
            # On Gemma the pressure runs the other way -- 2000 tokens is a third
            # of a minute's entire budget. See MODEL_PROFILES.
            #
            # Both halves are covered regardless: `_cited_in_text` below no
            # longer depends on the tail of the JSON surviving.
            max_output_tokens=settings.draft_max_output_tokens,
        )
    except LLMError as exc:
        # Quota, safety block, recitation. Nothing to salvage, but a turn that
        # says why beats a 502 -- the retrieved evidence is still on screen.
        log.warning("draft_failed", error=str(exc))
        return {
            "draft": f"The answer could not be generated ({exc}).",
            "citations": [],
            "sufficient": True,  # a retry would hit the same wall
            # Flagged, so callers that are not a chat window can tell this
            # apart from an answer. Without it the evaluation harness cached
            # this string and scored it for faithfulness.
            "generation_failed": True,
            "trace": [{"node": "draft", "error": str(exc)}],
        }

    answer = extract_string(raw, "answer")
    citations = extract_int_list(raw, "sources_used")
    unanswered = extract_string_list(raw, "unanswered")

    # The inline [n] markers are the ground truth, so derive from them when the
    # declared list is missing.
    #
    # `sources_used` is a SUMMARY of what the prose already says, and it is the
    # part most likely to be lost: it comes last in the JSON, so truncation
    # takes it first. The markers, by contrast, are what the reader actually
    # sees and what the UI turns into clickable citations -- an answer full of
    # [1]s that reports citing nothing is just wrong, and the text is right
    # there to check.
    if not citations:
        derived = _cited_in_text(answer, len(evidence))
        if derived:
            log.info("citations_derived_from_text", cited=derived)
            citations = derived

    # Two failure modes, one message.
    #
    # `not answer` -- nothing parsed, or degeneration from the first token.
    #
    # `is_repetitive` -- SECOND LINE OF DEFENCE, and it exists because the
    # first one leaked. `strip_degeneration` cuts a trailing loop, but a
    # response that is mostly loop still leaves a fragment behind, and one
    # reached the UI as several hundred repetitions of "the-the". Cutting is
    # not the same as judging: if what survives is still mostly repetition,
    # there is no answer here and saying so is better than showing it.
    if not answer or is_repetitive(answer):
        log.warning("draft_unusable", raw=raw[:200], salvaged=answer[:120])
        return {
            "draft": "The model did not return a usable answer. Try asking again.",
            "citations": [],
            "sufficient": True,
            "generation_failed": True,
            "trace": [{"node": "draft", "error": "no usable answer"}],
        }

    regen = state.get("regen_count", 0) + (1 if overclaims else 0)
    log.info("drafted", cited=citations, unanswered=unanswered, regen=regen)
    return {
        "draft": answer,
        "citations": citations,
        "unanswered": unanswered,
        # Everything the model was shown, whether or not it cited it. Cited-only
        # would let an uncited chunk be retrieved again on the retry and count
        # as a "new angle" when the drafter had already read it and passed.
        "seen_chunk_ids": {str(h.chunk_id) for h in evidence},
        "regen_count": regen,
        # Cleared so the next critique starts fresh. Left in place, a fixed
        # over-claim would be re-injected into every later draft as though it
        # were still present.
        "unsupported_claims": [],
        "trace": [
            {
                "node": "draft",
                "n_sources": len(evidence),
                "cited": citations,
                "unanswered": unanswered,
                "regenerated": bool(overclaims),
            }
        ],
    }


# --------------------------------------------------------------------------
# 5. resolve  (partial answer / abstain)
#
# ABSTENTION IS FOR "NO EVIDENCE EXISTS", NOT FOR "THE QUESTION IS IMPERFECT".
#
# A system that refuses whenever a question is not perfectly shaped is
# performing rigour rather than being useful, and people stop asking it things.
# Every response should leave the user with a next move, so there is exactly one
# hard stop -- nothing was retrieved at all -- and everything else is an answer
# with its limits named.
# --------------------------------------------------------------------------

RESOLVE_SYSTEM = (
    """You are finishing an answer that could not be completed.

You are given a draft, the sources behind it, and what a reviewer said was \
missing or unsupported.

Produce the most useful honest answer available:
- KEEP every claim the sources support, with its [n] citations intact.
- REMOVE or weaken anything the reviewer flagged as unsupported.
- Then state plainly, in one or two sentences at the end, what could not be \
answered and why -- "your documents do not give X".

Never apologise at length, never refuse outright when some of the question was \
answerable, and never invent a fact to fill the gap. A partial answer with its \
limits named is far more useful than a refusal."""
    # THE SAME RULES THE DRAFTER GETS.
    #
    # This node writes the final answer whenever the critique loop runs out of
    # budget, and it had no formatting guidance at all -- so the longest,
    # most-worked turns were exactly the ones that arrived as a single
    # unbroken block, and every fix to DRAFT_SYSTEM looked like it had
    # silently stopped applying.
    + "\n\n"
    + ANSWER_RULES
)

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            # Same wording as DRAFT_SCHEMA's, and for the same reason: the
            # field description is where the model learns this is markdown
            # that may contain line breaks. Without it the schema quietly
            # implies a single-line string.
            "description": (
                "The answer in MARKDOWN, [n] citations kept. Use real line "
                "breaks: a blank line between paragraphs, each list item on "
                "its own line. Do not write the characters backslash-n."
            ),
        },
        "sources_used": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer"],
}


async def resolve(state: ResearchState) -> dict:
    """Answer partially, or abstain if there is genuinely nothing."""
    settings = get_settings()
    evidence = state.get("evidence") or []
    draft_text = state.get("draft") or ""
    missing = state.get("missing") or state.get("unanswered") or []
    overclaims = state.get("unsupported_claims") or []

    # THE ONE HARD STOP. No passages at all means there is nothing to be
    # partially right about.
    if not evidence:
        searched = state.get("tried_queries") or [state["question"]]
        return {
            "draft": (
                "I could not find anything to answer this. Searched: "
                + "; ".join(f"“{q}”" for q in searched[:3])
                + ". Try naming a document or section, or rephrasing."
            ),
            "citations": [],
            "sufficient": True,
            "partial": False,
            "trace": [{"node": "resolve", "outcome": "abstained"}],
        }

    gaps = "\n".join(f"- {m}" for m in (missing or ["(not specified)"]))
    flagged = "\n".join(f"- {c}" for c in overclaims) if overclaims else "(none)"
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Sources:\n{build_context(evidence)}\n\n"
        f"Draft answer:\n{draft_text}\n\n"
        f"Reviewer says these are unsupported:\n{flagged}\n\n"
        f"Reviewer says these are missing:\n{gaps}\n\n"
        # This node replaces the draft wholesale, so anything the draft was
        # told to mention has to be repeated here or it is simply dropped. The
        # memory confirmation was exactly that: stored, mentioned by `draft`,
        # then written out of existence by a `resolve` that had never heard of
        # it.
        f"{_memory_block(state)}"
        f"{_facts_block(state)}"
        # The OTHER half of the same bug. `resolve` shares ANSWER_RULES, which
        # says to report the work from the "Search coverage" line -- and
        # `resolve` was never given one. So on every turn the critique loop
        # exhausted, the opening sentence was invented from nothing.
        f"{_coverage_block(state)}"
        "Write the most useful honest answer available."
    )

    try:
        raw = await get_pool("answer").generate(
            prompt,
            schema=RESOLVE_SCHEMA,
            system=RESOLVE_SYSTEM + (state.get("preferences") or ""),
            temperature=0.1,
            max_output_tokens=settings.draft_max_output_tokens,
        )
        answer = extract_string(raw, "answer")
        citations = extract_int_list(raw, "sources_used")
    except LLMError as exc:
        # Fall back to the draft that already exists rather than losing it. It
        # is imperfect -- that is why we are here -- but an imperfect grounded
        # answer beats an error message.
        log.warning("resolve_failed", error=str(exc))
        answer, citations = draft_text, state.get("citations") or []

    if not answer or is_repetitive(answer):
        answer, citations = draft_text, state.get("citations") or []

    if not citations:
        citations = _cited_in_text(answer, len(evidence))

    log.info("resolved", cited=citations, n_gaps=len(missing))
    return {
        "draft": answer,
        "citations": citations,
        "sufficient": True,  # this IS the final answer; nothing follows
        "partial": True,
        "trace": [
            {"node": "resolve", "outcome": "partial", "gaps": missing},
        ],
    }


# --------------------------------------------------------------------------
# 4. critique
# --------------------------------------------------------------------------

# Failure modes, and they exist because the remedies are OPPOSITE.
#
# An over-claim needs the draft rewritten from the SAME evidence -- searching
# again cannot fix it, because the passages were already correct. A gap needs
# new evidence -- rewriting cannot fix it, because the words are not there.
# A single `sufficient: false` collapsed both into "go and search again", so
# the only remedy the graph had was the wrong one half the time.
UNSUPPORTED_CLAIM = "unsupported_claim"
MISSING_EVIDENCE = "missing_evidence"
UNANSWERABLE = "unanswerable"
FAILURE_MODES = (UNSUPPORTED_CLAIM, MISSING_EVIDENCE, UNANSWERABLE)

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
        "failure_mode": {
            "type": "string",
            "enum": list(FAILURE_MODES),
            "description": (
                "Only when sufficient is false. unsupported_claim = the answer "
                "says more than the sources support. missing_evidence = the "
                "sources do not cover part of the question. unanswerable = no "
                "search could help."
            ),
        },
        "assessment": {"type": "string", "description": "One or two sentences."},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Claims the cited sources do not actually support.",
        },
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

CRITICAL CONTEXT: the sources shown are ONLY what has been retrieved so far, \
not everything that could be. More can still be fetched, from the user's \
documents AND from the public web. So "the sources do not contain X" does NOT \
mean X is unavailable -- it usually means the right search has not been run \
yet.

The user's documents are not a boundary. A gap that their files cannot fill may \
still be answerable from public sources, so propose a query for it rather than \
concluding the information does not exist.

Judge two things:
1. Completeness -- does the draft answer every part of the question?
2. Support -- is every claim backed by the cited sources? A claim attributed \
to the wrong KIND of source -- a public figure presented as coming from the \
user's own documents, or the reverse -- is NOT supported.

When sufficient=false you MUST also name the `failure_mode`, because the fix \
differs completely:

- unsupported_claim -- the evidence is fine, the DRAFT overstates it. List the \
offending sentences in `unsupported_claims`. Do NOT propose searches; more \
passages cannot fix a sentence that says more than its source.
- missing_evidence -- the draft is honest but part of the question is not \
covered. Put self-contained SEARCH QUERIES in `missing`. Queries, not \
instructions.
- unanswerable -- no search would help: the question asks for something no \
document could contain, or asks about a future or a private fact.

Set sufficient=true only when either:
- every part of the question is answered and supported, or
- the listed queries have already been tried and still returned nothing, so the \
information is genuinely unavailable from any source.

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
        mode = str(result.get("failure_mode", "") or "")
        missing = [
            str(m).strip()
            for m in result.get("missing", [])
            if isinstance(m, str) and m.strip()
        ]
        overclaims = [
            str(c).strip()
            for c in result.get("unsupported_claims", [])
            if isinstance(c, str) and c.strip()
        ]
    except LLMError as exc:
        # If the critic fails, accept the draft. Better a good answer with no
        # review than a 502.
        log.warning("critique_failed", error=str(exc))
        sufficient, assessment = True, f"critique unavailable ({exc})"
        mode, missing, overclaims = "", [], []

    # The drafter's own `unanswered` list is more reliable than the critic's
    # inference -- it knows exactly what it couldn't support. If it reported
    # gaps, trust that over a sufficient=true verdict.
    if unanswered and not missing:
        missing = list(unanswered)
        if sufficient:
            sufficient = False
            mode = mode or MISSING_EVIDENCE
            assessment += " (drafter reported unanswered parts)"

    # Never re-run a query that already came back empty-handed; that's how a
    # cycle becomes an infinite loop that spends quota to learn nothing.
    tried_lower = {q.strip().lower() for q in tried}
    missing = [m for m in missing if m.strip().lower() not in tried_lower][
        : settings.agent_max_subquestions
    ]

    # Infer the mode when the model left it out, rather than defaulting to one.
    # Which remedy applies is the whole decision, and guessing wrong sends the
    # graph to re-retrieve an answer that only needed rewording -- or to reword
    # an answer that was honest about a real gap.
    if not sufficient and mode not in FAILURE_MODES:
        mode = UNSUPPORTED_CLAIM if overclaims and not missing else MISSING_EVIDENCE

    # Nothing left to search for. Note this NO LONGER forces sufficient=True:
    # an over-claim is still fixable by regenerating, and the old code accepted
    # the draft here precisely when the critic had found a real problem it had
    # no queries for.
    if not sufficient and mode == MISSING_EVIDENCE and not missing:
        mode = UNANSWERABLE
        assessment += " (no untried follow-up queries)"

    log.info(
        "critiqued",
        sufficient=sufficient,
        failure_mode=mode or None,
        iterations=iterations,
        missing=missing,
        n_overclaims=len(overclaims),
    )
    return {
        "sufficient": sufficient,
        "critique": assessment,
        "failure_mode": mode,
        "missing": missing,
        "unsupported_claims": overclaims,
        "pending_queries": missing,
        "tried_queries": missing,
        "iterations": iterations,
        "trace": [
            {
                "node": "critique",
                "sufficient": sufficient,
                "failure_mode": mode or None,
                "assessment": assessment,
                "missing": missing,
                "unsupported_claims": overclaims,
                "iteration": iterations,
            }
        ],
    }


# --------------------------------------------------------------------------
# 0a. route  (what KIND of turn is this)
#
# THE GAP THIS CLOSES.
#
# Every turn used to be treated as a retrieval question. Asked "remember my
# preference to always search both the internet and my documents", the graph
# dutifully searched the documents FOR THAT PREFERENCE and answered "your
# documents do not mention personal preferences regarding search behaviour" --
# which is both true and completely useless.
#
# `clarify` could not catch it: it decides whether a request is specific enough
# to SEARCH, which already assumes searching is the right response. The missing
# question was one level up: is this a question at all?
# --------------------------------------------------------------------------

ASK = "ask"
REMEMBER = "remember"
BOTH = "both"

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [ASK, REMEMBER, BOTH],
            "description": (
                "ask = a question only. remember = an instruction only. "
                "both = the message does both at once."
            ),
        },
        "preference": {
            "type": "string",
            "description": (
                "The standing instruction, in the second person, keeping the "
                "user's own specifics. Empty when intent=ask."
            ),
        },
        "scope": {
            "type": "string",
            "enum": ["user", "session"],
            "description": "user = always. session = this conversation only.",
        },
        "question": {
            "type": "string",
            "description": (
                "The part that is actually a QUESTION, with the instruction "
                "removed. Empty when intent=remember."
            ),
        },
    },
    "required": ["intent"],
}

ROUTE_SYSTEM = """You decide what KIND of message this is. You do not answer it.

intent = "remember" when the user is telling you HOW to behave from now on, \
rather than asking for information. Signals: "remember", "always", "from now \
on", "never", "going forward", "in future", "make sure you", "stop doing".

    "always search the web too"                      -> remember
    "remember I prefer short answers"                 -> remember
    "from now on cite page numbers"                   -> remember
    "never mention the incident report"               -> remember

intent = "ask" for everything else -- questions, follow-ups, instructions about \
THIS answer only, and anything you are unsure about.

    "what drove the margin improvement?"              -> ask
    "summarise that in one line"                      -> ask   (this answer only)
    "what do my documents say about preferences?"     -> ask   (a real question)

intent = "both" when ONE message does BOTH -- a question AND a standing \
instruction. This is common and must not be collapsed into either one:

    "tell me about pyramids and always say which facts came from the web"
        -> both. question: "tell me about pyramids"
                 preference: "Always say which facts came from the web."

    "what was revenue? and from now on give me the figure first"
        -> both. question: "what was revenue?"
                 preference: "Give the figure first."

Picking one would silently drop the other half of what they asked for.

BIAS TOWARDS ANSWERING. If a message contains anything question-shaped, the \
intent is "ask" or "both", never "remember" alone -- losing the answer is far \
worse than losing the instruction, which they can restate.

When there is a preference ("remember" or "both"):
- `preference` is ONE imperative sentence addressed to you, keeping every \
specific the user gave. "always search both the documents and the web, and say \
which facts came from which" -- not "the user prefers thorough search".
- `scope` is "user" unless they clearly limited it to this conversation \
("for this chat", "just here"). Default to "user": people say "always" and mean \
it.

When intent = "both":
- `question` is the message with the instruction REMOVED, and nothing else \
changed. It becomes the search query, so leaving "and remember to always..." in \
it would send that phrase to a retrieval engine as if it were a topic.

ALREADY-REMEMBERED INSTRUCTIONS may be listed below. If the user is restating \
one of them -- in any wording -- leave `preference` EMPTY. People repeat \
themselves when they think they were not heard, and storing a second copy makes \
the instruction look twice as emphatic while telling them nothing new. Only \
emit a preference that ADDS something: a new rule, or a genuine change to an \
existing one (including reversing it).

A RESTATEMENT WITH NO QUESTION IN IT is intent="remember" with `preference` \
empty. Do NOT call it "ask": there is nothing to look up, and searching the \
documents for an instruction the user just gave finds nothing and wastes their \
time. If the restatement is bundled with a real question, use "both" and put \
the question in `question`."""


async def route(state: ResearchState) -> dict:
    """Classify the turn, and capture a preference when that is what it is.

    Fails soft to `ask` in every direction. A broken router must never stop a
    question being answered -- that is a far worse failure than missing a
    preference the user can restate.
    """
    question = state["question"]
    settings = get_settings()
    if not settings.agent_route:
        return {}

    chat_context = state.get("chat_context") or ""
    history = f"Conversation so far:\n{chat_context}\n\n" if chat_context else ""

    # The router already sees what is stored, so the "is this new?" judgement
    # is made by a model that understands synonyms rather than by token
    # overlap. Measured: lexical dedupe kept "always tell me what's from the
    # internet" and "always let me know what info is from the internet" as two
    # rows, because they share almost no content words despite being one
    # instruction.
    known = state.get("preferences") or ""
    already = (
        f"ALREADY REMEMBERED:\n{known}\n\n"
        if known
        else "Nothing is remembered for this user yet.\n\n"
    )

    try:
        raw = await get_llm().generate(
            f"{history}{already}Message: {question}",
            schema=ROUTE_SCHEMA,
            system=ROUTE_SYSTEM,
            temperature=0.0,
            max_output_tokens=300,
        )
    except LLMError as exc:
        log.warning("route_failed", error=str(exc))
        return {"trace": [{"node": "route", "intent": ASK, "error": str(exc)}]}

    intent = (extract_string(raw, "intent") or ASK).strip().lower()
    if intent not in (REMEMBER, BOTH):
        return {"trace": [{"node": "route", "intent": ASK}]}

    text = extract_string(raw, "preference").strip()
    scope = (extract_string(raw, "scope") or "user").strip().lower()
    asked = extract_string(raw, "question").strip()

    if not text:
        # No preference to store. Two quite different reasons, and they need
        # different endings.
        if intent == BOTH or asked:
            # A restatement bundled with a real question. Answer the question;
            # the instruction is already in force.
            log.info("route_restated_with_question", question=(asked or question)[:60])
            return {
                "question": asked or question,
                "intent": BOTH,
                "trace": [{"node": "route", "intent": BOTH, "saved": False}],
            }
        # A pure restatement. Confirming beats searching for it -- which is
        # exactly the failure this node exists to prevent -- and beats silence,
        # which reads as not having listened.
        log.info("route_restated", question=question[:60])
        return {
            "draft": (
                "Already remembered, so nothing changed. You can see everything "
                "I have stored in your profile."
            ),
            "citations": [],
            "sufficient": True,
            "intent": REMEMBER,
            "memory_saved": [],
            "trace": [{"node": "route", "intent": REMEMBER, "saved": False}],
        }

    raw_session = state.get("session_id")
    session_uuid = uuid.UUID(raw_session) if raw_session else None

    # Checked against what is STORED, not against the block in the prompt.
    #
    # `state["preferences"]` is the rendered instruction block, and
    # `render_for_prompt` caps it at MAX_IN_PROMPT. Deduping against that would
    # make every preference past the cap invisible to this check -- so a user
    # with a long list would start accumulating duplicates of exactly the older
    # instructions they had most likely forgotten stating.
    stored = await preferences.preferences_in_force(
        owner_id=state.get("owner_id"), session_id=session_uuid
    )
    if await preferences.already_covered(text, [p.text for p in stored]):
        log.info("route_duplicate_preference", text=text[:60])
        if intent == BOTH or asked:
            # Already in force, so there is nothing to store -- but there IS a
            # question, and it still gets answered.
            return {
                "question": asked or question,
                "intent": BOTH,
                "trace": [
                    {"node": "route", "intent": BOTH, "saved": False, "duplicate": True}
                ],
            }
        return {
            "draft": (
                "Already remembered, so nothing changed. You can see everything "
                "I have stored in your profile."
            ),
            "citations": [],
            "sufficient": True,
            "intent": REMEMBER,
            "memory_saved": [],
            "trace": [
                {
                    "node": "route",
                    "intent": REMEMBER,
                    "saved": False,
                    "duplicate": True,
                }
            ],
        }

    pref = await preferences.remember(
        text,
        owner_id=state.get("owner_id"),
        session_id=session_uuid,
        scope="session" if scope == "session" else "user",
        source_message=question,
    )
    saved = [text] if pref is not None else []

    # APPLIED TO THIS TURN, not merely stored for the next one.
    #
    # "tell me about X and always say which facts came from the web" plainly
    # means "including now". Storing it and answering without it would ignore
    # the instruction in the very message that gave it, which reads as the
    # assistant not listening.
    merged = (state.get("preferences") or "") + preferences.render_for_prompt(
        [pref] if pref is not None else []
    )

    if intent == BOTH:
        # The question with the instruction stripped out. It becomes the search
        # query, and leaving "and remember to always..." in would send that
        # phrase to a retrieval engine as though it were a topic.
        target = asked or question
        log.info("routed_both", scope=scope, saved=bool(pref), question=target[:60])
        return {
            # Rewritten, so every node downstream sees the question alone. The
            # original is still in the transcript.
            "question": target,
            "original_question": question,
            "intent": BOTH,
            "preferences": merged,
            "memory_saved": saved,
            "trace": [
                {
                    "node": "route",
                    "intent": BOTH,
                    "scope": scope,
                    "saved": bool(pref),
                    "question": target,
                }
            ],
        }

    where = "this conversation" if scope == "session" else "all conversations"
    answer = (
        f"Noted, and saved for {where}:\n\n> {text}\n\nI will apply this from now on."
        if pref is not None
        else f"Already remembered, so nothing changed:\n\n> {text}"
    )

    log.info("routed_remember", scope=scope, saved=pref is not None)
    return {
        # Terminal ONLY for a pure instruction: there is no question to answer,
        # nothing to retrieve, and running the search path would produce exactly
        # the "your documents do not mention your preferences" answer this node
        # exists to prevent.
        "draft": answer,
        "citations": [],
        "sufficient": True,
        "intent": REMEMBER,
        "preferences": merged,
        "memory_saved": saved,
        "trace": [
            {"node": "route", "intent": REMEMBER, "scope": scope, "saved": bool(pref)}
        ],
    }

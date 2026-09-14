"""Tool declarations and dispatch for the ReAct research node.

WHY THE DESCRIPTIONS MATTER MORE THAN THE CODE

The model picks a tool from its **name, description and parameter names** and
nothing else. Most "the agent chose the wrong tool" problems are
tool-description problems, not model problems -- so each description below says
what the tool is for AND when not to use it, which is the part people leave
out.

TWO TOOLS, NOT MORE

Tool choice degrades somewhere around 10-20 options, and every tool is another
thing that can be picked wrongly. Two clearly-separated tools -- the user's own
documents, and the public web -- is a distinction the model can always get
right, and it maps onto the only thing that actually differs: whether the
answer is private or public.

PEERS, NOT A FALLBACK

An earlier version of these descriptions said "USE search_documents FIRST" and
"use search_web ONLY when the documents cannot contain the answer". That made
the corpus a boundary and the web a last resort, which is wrong in both
directions: it wasted a round searching the documents for "what is HNSW", and
it discouraged the web on exactly the questions that need both ("how does our
p99 compare to the published benchmark"). The descriptions now describe each
tool's DOMAIN and let the model route on where the answer actually lives.

What did NOT change is the grounding contract. Widening the sources does not
license answering from memory: everything still has to come back through a
tool and be citable. See DRAFT_SYSTEM in nodes.py.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from app.config import get_settings
from app.services import preferences, websearch
from app.services.parents import read_around
from app.services.retrieval import retrieve
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

SEARCH_DOCUMENTS = "search_documents"
SEARCH_WEB = "search_web"
READ_AROUND = "read_around"
REMEMBER_PREFERENCE = "remember_preference"


def tool_specs() -> list[dict[str, Any]]:
    """functionDeclarations for the models that support them.

    `search_web` is omitted entirely when unconfigured rather than declared and
    made to fail. A tool the model can see but cannot use is worse than no
    tool: it will keep choosing it, get nothing back, and burn rounds.
    """
    declarations: list[dict[str, Any]] = [
        {
            "name": SEARCH_DOCUMENTS,
            "description": (
                "Search the user's own uploaded documents: their reports, "
                "incidents, transcripts and internal data. This is the only "
                "place their private material exists, so it is the right tool "
                "for anything specific to them or their organisation. Returns "
                "passages that can be cited. Call it again with different "
                "wording if the first results are not relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "One sentence describing the information needed, "
                            "phrased as it would appear in the document. Not a "
                            "keyword list."
                        ),
                    }
                },
                "required": ["query"],
            },
        }
    ]

    # THE MIDDLE RUNG OF THE CONTEXT LADDER.
    #
    #   the matched passage        returned by search_documents
    #   its whole section          automatic, when parent retrieval is on
    #   a little more, bounded     read_around          <- this
    #   the whole document         not offered: at this corpus size it would
    #                              usually mean stuffing the entire file
    #
    # Without it the model's only move when a passage refers to something it
    # cannot see ("this represented a sharp reversal") is to search again with
    # terms taken from the very sentence it does not understand -- which
    # retrieves the same passage back.
    declarations.append(
        {
            "name": READ_AROUND,
            "description": (
                "Read the passages immediately before and after one you already "
                "retrieved, in document order. Use when a passage refers to "
                "something you cannot see -- 'this figure', 'the reversal "
                "above', a pronoun with no antecedent -- and you need the "
                "surrounding text rather than a different search. Crosses "
                "section boundaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The id of a passage from an earlier search result.",
                    },
                    "before": {"type": "integer", "description": "Passages before. 0-3."},
                    "after": {"type": "integer", "description": "Passages after. 0-3."},
                },
                "required": ["chunk_id"],
            },
        }
    )

    if websearch.enabled():
        declarations.append(
            {
                "name": SEARCH_WEB,
                "description": (
                    "Search the public web: general knowledge, definitions, "
                    "technical background, public companies, current events, "
                    "standards and benchmarks. Use it whenever the answer lives "
                    "outside the user's own files, and use it ALONGSIDE "
                    "search_documents when a question spans both -- for example "
                    "comparing something in their documents against public "
                    "figures. Results must be cited like any other source."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A web search query, as typed into a search engine.",
                        }
                    },
                    "required": ["query"],
                },
            }
        )

    # Memory as a TOOL rather than a node in front of the graph.
    #
    # A router node had to classify every message before anything else ran, so
    # it decided "is this an instruction?" without having seen a single
    # document -- and the cost of that guess was paid on every turn, including
    # the overwhelming majority that store nothing. As a tool it is the same
    # judgement made by the agent that is already reading the message, at the
    # moment it has something to store, and it composes: "tell me about X and
    # always cite pages" is one search call plus one remember call, rather than
    # a router that has to split the message before either can happen.
    declarations.append(
        {
            "name": REMEMBER_PREFERENCE,
            "description": (
                "Store a STANDING INSTRUCTION about how to answer, so it "
                "applies to this and every future turn. Use it when the user "
                "tells you how to behave from now on -- 'always cite page "
                "numbers', 'keep answers short', 'never mention X', 'remember "
                "that I prefer...'. Call it as well as searching when one "
                "message both asks something and gives an instruction. Do NOT "
                "use it for a one-off request about the current answer "
                "('summarise that in one line'), and do NOT use it to store "
                "facts, documents or answers -- it is only for instructions "
                "about your own behaviour. Storing something already covered "
                "is harmless: it is detected and skipped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "One imperative sentence addressed to you, keeping "
                            "every specific the user gave. 'Always say which "
                            "facts came from the web.' -- not 'the user "
                            "prefers thorough search'."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["user", "session"],
                        "description": (
                            "'user' (the default) applies it to every "
                            "conversation; 'session' only to this one. Use "
                            "'session' only when the user clearly limited it "
                            "-- 'for this chat', 'just here'. People say "
                            "'always' and mean it."
                        ),
                    },
                },
                "required": ["instruction"],
            },
        }
    )

    return [{"functionDeclarations": declarations}]


async def run_tool(
    name: str,
    args: dict[str, Any],
    *,
    top_k: int,
    document_ids: list[uuid.UUID] | None,
    owner_id: str | None,
    session_id: uuid.UUID | None = None,
    remembered: list[str] | None = None,
) -> tuple[list[SearchHit], str]:
    """Execute one tool call. Returns (hits, text_for_the_model).

    Never raises. A tool that throws would end the loop, and the model can
    recover from "no results" by rephrasing -- so failures come back as text it
    can read and react to, which is the whole point of the observation step.

    `owner_id` is threaded through rather than read from a context var for the
    same reason it is on the graph state: tenant scoping must be an explicit
    argument on every retrieval path.
    """
    # Checked BEFORE the query guard below: this is the one tool that takes no
    # query, and falling through would reject every call with "the query
    # parameter was empty".
    if name == REMEMBER_PREFERENCE:
        return [], await _remember(
            args, owner_id=owner_id, session_id=session_id, remembered=remembered
        )

    query = str(args.get("query", "")).strip()
    if not query:
        return [], "Error: the query parameter was empty. Provide a query."

    try:
        if name == SEARCH_DOCUMENTS:
            hits = await retrieve(query, top_k=top_k, document_ids=document_ids, owner_id=owner_id)
            if not hits:
                return [], (
                    f"No passages matched {query!r}. Either nothing in the "
                    "documents covers it, or the wording is too far from how "
                    "the document puts it -- try different terms."
                )
            return hits, f"{_coverage_note(hits, top_k)}\n\n{_render_for_model(hits)}"

        if name == READ_AROUND:
            raw_id = str(args.get("chunk_id", "")).strip()
            try:
                anchor = uuid.UUID(raw_id)
            except ValueError:
                return [], (
                    f"{raw_id!r} is not a passage id. Use an id shown in an "
                    "earlier search result."
                )
            hits = await read_around(
                anchor,
                before=int(args.get("before", 1) or 0),
                after=int(args.get("after", 1) or 0),
                owner_id=owner_id,
                document_ids=document_ids,
            )
            if not hits:
                # Covers both "no such passage" and "not yours". Deliberately
                # one message: confirming that an id exists but belongs to
                # someone else is a leak even without the content.
                return [], f"No passage with id {raw_id}. Use an id from a search result."
            return hits, _render_for_model(hits)

        if name == SEARCH_WEB:
            if not websearch.enabled():
                return [], "Web search is not configured."
            hits = await websearch.search_web(query)
            if not hits:
                return [], f"No web results for {query!r}. Try different wording."
            return hits, _render_for_model(hits)

        # The model invented a tool. Say so plainly -- it can correct itself.
        return [], f"Unknown tool {name!r}. Available: {SEARCH_DOCUMENTS}, {SEARCH_WEB}."

    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the loop
        log.warning("tool_failed", tool=name, query=query[:60], error=str(exc))
        return [], f"The {name} tool failed: {type(exc).__name__}. Try again or rephrase."


async def _remember(
    args: dict[str, Any],
    *,
    owner_id: str | None,
    session_id: uuid.UUID | None,
    remembered: list[str] | None,
) -> str:
    """Store a standing instruction, and tell the model what happened.

    The observation is written for the AGENT to act on, not for the user: it
    says plainly whether the instruction was new or already covered, because
    the agent has to report that difference in its reply and cannot tell
    otherwise. "Saved" and "you already had this" are different things to say.

    Appends to `remembered` so the caller knows what was stored this turn
    without re-reading the database -- the answer has to confirm it, and a
    second query could race with a concurrent turn.
    """
    text = " ".join(str(args.get("instruction", "")).split())
    if not text:
        return "Error: the instruction parameter was empty."

    scope = str(args.get("scope") or "user").strip().lower()
    if scope not in ("user", "session"):
        scope = "user"

    try:
        stored = await preferences.preferences_in_force(
            owner_id=owner_id, session_id=session_id
        )
        if await preferences.already_covered(text, [p.text for p in stored]):
            log.info("preference_already_covered", text=text[:60])
            return (
                f"Already covered by a stored instruction, so nothing was "
                f"added: {text!r}. Tell the user it is already remembered "
                "rather than claiming you saved it."
            )

        pref = await preferences.remember(
            text,
            owner_id=owner_id,
            session_id=session_id,
            scope=scope,
            source_message=None,
        )
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the loop
        log.warning("remember_tool_failed", error=str(exc))
        return f"Could not store the instruction: {type(exc).__name__}."

    if pref is None:
        return (
            f"Already remembered, so nothing changed: {text!r}. Say it is "
            "already in force rather than claiming you saved it."
        )

    if remembered is not None:
        remembered.append(text)
    where = "this conversation" if scope == "session" else "every conversation"
    return f"Stored for {where}: {text!r}. Confirm this to the user."


def _coverage_note(hits: list[SearchHit], top_k: int) -> str:
    """What this search did NOT return.

    Surfacing the gap is the single cheapest way to get a second hop. A model
    shown only results assumes it has them all and stops; a model told "these
    are 5 passages from 2 of your 4 documents" has an obvious next move, and it
    takes no extra call to say so.
    """
    files = sorted({h.filename for h in hits if h.source == "document"})
    parts = [f"Showing {len(hits)} passages"]
    if files:
        parts.append("from " + ", ".join(files[:4]) + ("..." if len(files) > 4 else ""))
    if len(hits) >= top_k:
        # A full page is evidence there may be more behind it. Fewer than asked
        # for means the pool was genuinely exhausted, and saying "there may be
        # more" then would invite a pointless extra round.
        parts.append("(more may exist -- narrow the query or ask for a different aspect)")
    return " ".join(parts)


def _render_for_model(hits: list[SearchHit]) -> str:
    """Results as text the model can reason over.

    Deliberately WITHOUT citation numbers. Numbering is assigned once, at
    drafting time, over the whole accumulated evidence set -- if each tool
    result carried its own [1]..[n] the model would cite numbers that mean
    something different in every round, and the citations would point at the
    wrong sources in the final answer.
    """
    blocks = []
    for hit in hits:
        where = hit.url or hit.filename
        if hit.source == "document" and hit.heading:
            where = f"{hit.filename} › {hit.heading.lstrip('# ').strip()}"
        # The id is a HANDLE, not a citation marker. Citation numbers are
        # assigned once at drafting time over the accumulated evidence set --
        # per-result numbers would mean something different every round. An id
        # is stable and means the same thing everywhere, which is what makes
        # `read_around` callable at all: without it the model has no way to
        # name the passage it wants more context around.
        prefix = f"[id {hit.chunk_id}] " if hit.source == "document" else ""
        blocks.append(f"{prefix}{where}\n{hit.text}")
    return "\n\n".join(blocks)


def max_rounds() -> int:
    return get_settings().react_max_rounds


def max_calls_per_round() -> int:
    return get_settings().react_max_calls_per_round

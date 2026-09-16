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

import asyncio
import uuid
from typing import Any

import structlog

from app.config import get_settings
from app.services import corpus, preferences, progress, websearch
from app.services.parents import read_around
from app.services.retrieval import retrieve
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

SEARCH_DOCUMENTS = "search_documents"
SEARCH_WEB = "search_web"
READ_AROUND = "read_around"
REMEMBER_PREFERENCE = "remember_preference"
LIST_DOCUMENTS = "list_documents"
CORPUS_STATS = "corpus_stats"
CONVERSATION_STATS = "conversation_stats"

# Tools that do not RETRIEVE anything.
#
# The distinction drives control flow, not bookkeeping. `react` uses it to
# decide whether the turn "searched": a turn that only listed documents or
# stored a preference has no passages to draft from, so it must answer in its
# own words rather than being sent to `draft` with an empty evidence set and
# told to compose from nothing.
NON_RETRIEVAL = frozenset(
    {REMEMBER_PREFERENCE, LIST_DOCUMENTS, CORPUS_STATS, CONVERSATION_STATS}
)


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
                "wording if the first results are not relevant. Pass "
                "`filename` to search ONE document, which is worth doing "
                "alongside an unscoped search when list_documents shows an "
                "obviously relevant file."
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
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict the search to this one "
                            "document, as named by list_documents. Omit it to "
                            "search everything, which is the default and "
                            "usually right."
                        ),
                    },
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

    # QUESTIONS ABOUT THE CORPUS, not questions answered FROM it.
    #
    # Semantic search cannot answer "how many documents are there", "what is
    # the average conversation length" or "what do these files broadly cover" --
    # embedding those questions retrieves passages that happen to discuss
    # counting, which is not the same thing at all. Before these existed the
    # agent could only deflect "what documents do you have?" while the UI
    # listed all five by name beside it.
    #
    # Three tools rather than one `stats(kind)` with a mode parameter: the
    # model picks from name and description, and a name that says exactly what
    # comes back is chosen correctly far more often than an enum it has to
    # reason about. Three clear names cost about as many tokens as one vague
    # one with three documented modes.
    declarations.append(
        {
            "name": LIST_DOCUMENTS,
            "description": (
                "List the user's uploaded documents: filename, size, how many "
                "passages each holds, when it was uploaded, whether it "
                "finished indexing, and the section headings across all of "
                "them. Use it for 'what documents do you have', 'what is in my "
                "library', 'what are these files about', 'what themes do they "
                "cover', and before saying something is not in their "
                "documents. It returns no passages and nothing citable -- to "
                "quote or cite content, use search_documents."
            ),
            "parameters": {"type": "object", "properties": {}},
        }
    )
    declarations.append(
        {
            "name": CORPUS_STATS,
            "description": (
                "Aggregate statistics over the whole document collection: how "
                "many documents, total and average length, passage counts, "
                "indexing status breakdown, and the first and latest upload "
                "dates. Use it for counting and sizing questions about the "
                "library as a whole -- 'how much have I uploaded', 'how long "
                "is the average document'. Not for the contents of any "
                "document."
            ),
            "parameters": {"type": "object", "properties": {}},
        }
    )
    declarations.append(
        {
            "name": CONVERSATION_STATS,
            "description": (
                "Aggregate statistics over the user's past chat sessions: how "
                "many conversations, average and median messages per "
                "conversation, longest conversation, average message length "
                "split by who wrote it, and first and last activity dates. Use "
                "it for questions about their usage and history -- 'how many "
                "chats have I had', 'what is the average length of my "
                "sessions'. It reports counts only and never the content of "
                "any past conversation."
            ),
            "parameters": {"type": "object", "properties": {}},
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
    already_known: list[str] | None = None,
    facts: list[str] | None = None,
) -> tuple[list[SearchHit], str]:
    """Execute one tool call. Returns (hits, text_for_the_model).

    Never raises. A tool that throws would end the loop, and the model can
    recover from "no results" by rephrasing -- so failures come back as text it
    can read and react to, which is the whole point of the observation step.

    `owner_id` is threaded through rather than read from a context var for the
    same reason it is on the graph state: tenant scoping must be an explicit
    argument on every retrieval path.
    """
    # Checked BEFORE the query guard below: these take no query, and falling
    # through would reject every call with "the query parameter was empty".
    if name == REMEMBER_PREFERENCE:
        progress.remembering(str(args.get("text") or ""))
        return [], await _remember(
            args,
            owner_id=owner_id,
            session_id=session_id,
            remembered=remembered,
            already_known=already_known,
        )

    if name in (LIST_DOCUMENTS, CORPUS_STATS, CONVERSATION_STATS):
        progress.looked_up(name)
        observation = await _metadata(name, owner_id=owner_id)
        progress.looked_up_done(name)
        # Recorded so the fact survives to whichever node writes the answer.
        #
        # The agent's own prose is discarded on the retrieval path -- `draft`
        # composes from `evidence` alone -- so a turn that both searched AND
        # counted would otherwise lose the count entirely. Returning it as a
        # SearchHit instead was the alternative and is worse: it would join the
        # numbered citation list, and "[3]" pointing at a figure this app
        # computed rather than at a passage anyone can open is exactly the kind
        # of unverifiable citation the grounding rules exist to prevent.
        if facts is not None:
            facts.append(observation)
        return [], observation

    query = str(args.get("query", "")).strip()
    if not query:
        return [], "Error: the query parameter was empty. Provide a query."

    try:
        if name == SEARCH_DOCUMENTS:
            scope = document_ids
            note = ""
            wanted = str(args.get("filename", "")).strip()
            if wanted:
                found = await corpus.resolve_document(wanted, owner_id)
                if found is None:
                    # NOT a silent fallback to searching everything. The model
                    # asked for one document; quietly answering from all of
                    # them would return passages it would then attribute to a
                    # file it never actually searched.
                    names = ", ".join(
                        d["filename"] for d in await corpus.documents(owner_id)
                    )
                    return [], (
                        f"No document matching {wanted!r}. Available: {names}. "
                        "Retry with one of those, or omit filename to search "
                        "everything."
                    )
                doc_id, actual = found
                scope = [uuid.UUID(doc_id)]
                # Says which file it really got, because the match is fuzzy and
                # a wrong resolution is otherwise invisible.
                note = f"Searched only {actual}. "

            hits = await retrieve(query, top_k=top_k, document_ids=scope, owner_id=owner_id)
            if not hits:
                return [], note + (
                    f"No passages matched {query!r}. Either nothing in the "
                    "documents covers it, or the wording is too far from how "
                    "the document puts it -- try different terms."
                )
            body = f"{_coverage_note(hits, top_k)}\n\n{_render_for_model(hits)}"
            return hits, note + body

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


async def _metadata(name: str, *, owner_id: str | None) -> str:
    """Run one metadata tool and render its result for the model.

    Rendered as labelled lines rather than returned as JSON. The model reads
    this back as an observation and then writes prose from it; JSON invites it
    to echo the structure -- answers came back as key/value dumps -- while
    lines it can read are lines it rewrites.

    Never raises, for the same reason the search tools do not: a failed tool
    should come back as text the model can react to, not end the loop.
    """
    try:
        if name == LIST_DOCUMENTS:
            rows = await corpus.documents(owner_id)
            outline = await corpus.headings(owner_id) if rows else []
            return corpus.render_documents(rows, outline)

        if name == CORPUS_STATS:
            return corpus.render_stats(
                "Statistics for the whole document collection:",
                await corpus.corpus_stats(owner_id),
            )

        return corpus.render_stats(
            "Statistics over the user's past conversations:",
            await corpus.conversation_stats(owner_id),
        )
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the loop
        log.warning("metadata_tool_failed", tool=name, error=str(exc))
        return f"The {name} tool failed: {type(exc).__name__}."


# Serialises the check-then-write below. See `_remember`.
_REMEMBER_LOCK = asyncio.Lock()


async def _remember(
    args: dict[str, Any],
    *,
    owner_id: str | None,
    session_id: uuid.UUID | None,
    remembered: list[str] | None,
    already_known: list[str] | None = None,
) -> str:
    """Store a standing instruction, and tell the model what happened.

    The observation is written for the AGENT to act on, not for the user: it
    says plainly whether the instruction was new or already covered, because
    the agent has to report that difference in its reply and cannot tell
    otherwise. "Saved" and "you already had this" are different things to say.

    Appends to `remembered` so the caller knows what was stored this turn
    without re-reading the database -- the answer has to confirm it, and a
    second query could race with a concurrent turn.

    SERIALISED, because the deduplication is a check-then-write and `react`
    dispatches a round's tool calls through `asyncio.gather`. Two paraphrases
    of one instruction requested in the SAME round both read the table before
    either wrote, so both passed the "already covered?" test and both were
    stored -- measured: "Always keep answers short." and "Please be brief in
    your replies." landed as two rows. The lock makes the second call read the
    first one's write.

    It is per-process, not a database lock, and that is the right size for
    this: the race is between two calls in one `gather`, and the lexical guard
    inside `remember` still catches the rarer cross-process case of one user
    running two turns at once.
    """
    text = " ".join(str(args.get("instruction", "")).split())
    if not text:
        return "Error: the instruction parameter was empty."

    scope = str(args.get("scope") or "user").strip().lower()
    if scope not in ("user", "session"):
        scope = "user"

    try:
        async with _REMEMBER_LOCK:
            stored = await preferences.preferences_in_force(
                owner_id=owner_id, session_id=session_id
            )
            # `remembered` is included alongside the stored rows -- belt and
            # braces. The lock already makes the second caller see the first
            # one's committed write, but this also covers the case where the
            # first write was skipped as a duplicate of something a THIRD call
            # stored, and it costs one list concatenation.
            known = [p.text for p in stored] + list(remembered or [])
            if await preferences.already_covered(text, known):
                log.info("preference_already_covered", text=text[:60])
                # Recorded, not merely logged. "Already in force" is a real
                # outcome the user needs told -- restating an instruction and
                # getting no acknowledgement reads as not having been heard,
                # which is the exact thing that makes people restate it again.
                if already_known is not None:
                    already_known.append(text)
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
            # INSIDE the lock: the next caller's `known` list has to include
            # this, and appending after releasing would reopen the same race
            # one level up.
            if pref is not None and remembered is not None:
                remembered.append(text)
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the loop
        log.warning("remember_tool_failed", error=str(exc))
        return f"Could not store the instruction: {type(exc).__name__}."

    if pref is None:
        return (
            f"Already remembered, so nothing changed: {text!r}. Say it is "
            "already in force rather than claiming you saved it."
        )

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

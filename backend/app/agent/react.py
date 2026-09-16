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

REACT_SYSTEM = """You handle the user's message. You decide what it needs: \
sometimes a search, sometimes nothing at all.

FIRST DECIDE WHETHER ANYTHING NEEDS LOOKING UP

Call NO TOOLS AT ALL, and simply write the reply yourself, when the message \
does not depend on any source:
- a greeting or small talk -- "hi", "hey", "thanks", "how are you"
- a question about YOU: what you can do, how you work, what documents you have
- a pure instruction about how to answer, once you have stored it
- a message you cannot act on until they say more

"hi" is answered with a greeting. Searching the documents and the web for it \
wastes their time, returns whatever happens to be lexically nearest, and \
produces a paragraph about a transcript they did not ask for. This is the \
single most common way to get this wrong.

When you do answer directly, write the ACTUAL REPLY -- a sentence or two, \
addressed to the user, in ordinary friendly English. Do not invent facts about \
their documents; if you have not searched, you do not know what is in them.

TALK LIKE A PERSON HERE. A greeting is answered like a greeting:

    "hi"                -> "Hi! What would you like to look into?"
    "thanks"            -> "You're welcome."
    "good job"          -> "Thanks! Anything else you want me to dig into?"

Not "Hi." Not "Hello." A bare full-stopped word is not concise, it is curt, \
and three of them in a row read as a broken machine.

NEVER NARRATE YOUR OWN BEHAVIOUR. "I repeat myself when I acknowledge \
repetitive praise without new input to address" is not a reply to anything -- \
it is you describing yourself in the third person. If a message needs no real \
answer, say something ordinary and brief and stop.

THE USER'S STANDING INSTRUCTIONS ABOUT HOW ANSWERS ARE PRESENTED DO NOT GOVERN \
SMALL TALK. Anything about length, structure, citations, formatting or the \
tone to take with findings -- "be stoic, facts first", "keep it short", \
"always use tables" -- describes an ANSWER BUILT FROM SOURCES. Applied to "hi" \
it produces exactly the broken replies above. Those instructions resume the \
moment there is a real answer to write.

Instructions about the CHANNEL itself still apply everywhere: what language to \
use, what to call the user, anything about how you address them. Those are not \
about answers, they are about talking to them at all.

Everything below applies only when the message DOES need sources.

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

KNOW WHAT IS THERE BEFORE ASSUMING WHERE IT IS

For a broad question about the user's own material -- "what should I know \
about X", "summarise our approach to Y" -- call list_documents FIRST. It is \
one cheap call and it tells you what the corpus actually contains, including \
which file is large enough to hold most of the answer.

Then search BOTH ways in the same round:
- an unscoped search_documents, which covers everything;
- a search_documents with `filename` set to the document that obviously \
covers the topic.

Both, not either. The unscoped search finds the paragraph in a file you did \
not expect; the scoped one stops a single large document being crowded out of \
the results by every other file also matching weakly. A corpus with one \
64-chunk handbook and five small documents will return a thin, scattered set \
for a handbook-shaped question unless the handbook is also searched on its own.

Do NOT scope when the question does not point at one document. A guess that \
narrows the search to the wrong file is worse than not narrowing at all, \
because the answer will look complete.

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

When sources ARE involved you do not write the final answer -- a later step \
composes it from what you collect. Never answer from memory, and never write \
citation markers: your job there is retrieval, not composition. Widening your \
sources does not weaken this: a claim you did not retrieve is still a claim \
you cannot make.

STANDING INSTRUCTIONS

When the user tells you how to behave from now on, call remember_preference. \
Do it ALONGSIDE searching when one message does both -- "tell me about X and \
always cite pages" is a search and a remember, not a choice between them. \
Then say what you stored, in your own reply if you are answering directly."""


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

    raw_session = state.get("session_id")
    session_id = uuid.UUID(raw_session) if raw_session else None

    specs = tools.tool_specs()
    evidence: list[SearchHit] = []
    seen: set[uuid.UUID] = set()
    trace: list[dict] = []
    # Filled by the remember_preference tool. Mutable and passed in rather than
    # parsed back out of the observations, because the answer has to confirm
    # what was stored and re-reading the table could race with another turn.
    remembered: list[str] = []
    # Instructions the user restated that were ALREADY in force. Tracked
    # separately from `remembered` because the answer has to say something
    # different about each -- "saved" and "you already had this" are not
    # the same message, and merging them would have the assistant claim to
    # have stored something it deliberately did not.
    already_known: list[str] = []
    # Facts the metadata tools computed -- counts, file lists, averages.
    # Carried separately from `evidence` because they are not passages and
    # cannot be cited: they were produced by this application from its own
    # database, so there is nothing for a [n] marker to point at.
    facts: list[str] = []
    # The reply the model wrote when it decided nothing needed looking up.
    direct: str = ""
    # Whether any SEARCH ran, which is not the same as whether evidence exists.
    #
    # "No evidence" has two causes that must not share a code path: the agent
    # never looked (a greeting), or it looked and found nothing. Only the first
    # is safe to answer from the model's own words. The second has to go to
    # `draft`, which is where "your documents cover X but not Z" is written and
    # where `critique` still reviews the result -- otherwise a failed search
    # becomes licence to answer from memory, ungrounded and unreviewed.
    searched = False

    for round_no in range(1, tools.max_rounds() + 1):
        try:
            calls, text, model_content = await get_llm().generate_tools(
                contents,
                tools=specs,
                system=REACT_SYSTEM + (state.get("preferences") or ""),
            )
        except LLMError as exc:
            log.warning("react_failed", round=round_no, error=str(exc))
            trace.append({"round": round_no, "error": str(exc)})
            break

        if not calls:
            # No tools requested. What that MEANS depends on whether anything
            # was gathered, and the two cases could not be more different.
            #
            # With evidence, gathering is finished and the prose is a
            # note-to-self, deliberately DISCARDED -- `draft` composes the
            # answer, because only it sees the final numbering of the
            # accumulated evidence. Letting this model write it would mean
            # either no inline citations or invented ones.
            #
            # With NOTHING gathered, the model judged that the message needs no
            # sources -- a greeting, a question about the assistant itself, a
            # bare instruction. Then the prose IS the answer and is kept.
            # Sending that case to `draft` is what made "hi" run three
            # sub-questions and five web searches, and answer with whatever
            # happened to be lexically nearest.
            if not searched:
                direct = text.strip()
            trace.append({"round": round_no, "done": True, "note": text[:160]})
            break

        contents.append(model_content)

        # Capped, and TRUNCATED rather than rejected: a greedy response asking
        # for twenty searches should still get its first few, because
        # cancelling the whole round teaches the model nothing.
        calls = calls[: tools.max_calls_per_round()]

        # Counting documents or storing an instruction is not looking
        # anything up, so a turn that only did those can still answer in
        # its own words -- there is no evidence for `draft` to work from.
        if any(c.get("name") not in tools.NON_RETRIEVAL for c in calls):
            searched = True

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
                    session_id=session_id,
                    remembered=remembered,
                    already_known=already_known,
                    facts=facts,
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

    # NOTHING TO GROUND: the agent judged the message needed no sources, so its
    # own prose is the answer and the turn ends here.
    #
    # `sufficient` is True so `critique` is skipped entirely -- a critic asked
    # whether "Hello, how can I help?" is supported by its sources would
    # correctly find that it is not, and send a greeting round the
    # missing-evidence loop. Grounding is a rule about CLAIMS, and there are
    # none here.
    if direct and not searched:
        log.info("react_direct", remembered=len(remembered), chars=len(direct))
        return {
            "evidence": [],
            "pending_queries": [],
            "sub_questions": [],
            "iterations": 0,
            "draft": direct,
            "citations": [],
            "sufficient": True,
            "answered_directly": True,
            "memory_saved": remembered,
            "memory_known": already_known,
            "corpus_facts": facts,
            "trace": [
                {
                    "node": "react",
                    "direct": True,
                    "remembered": remembered,
                    "already_known": already_known,
                }
            ],
        }

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
        # Carried so `draft` can confirm what was stored in its opening line.
        "memory_saved": remembered,
        "memory_known": already_known,
        # Metadata results reach `draft` HERE, and this is the whole reason
        # they are a separate state key. On this path the node's own prose is
        # discarded, so a turn that both searched and counted would otherwise
        # keep the passages and silently lose the count.
        "corpus_facts": facts,
        "trace": [{"node": "react", "rounds": trace, "n_web": n_web}],
    }

"""The conversation that designs an interview.

WHY THIS IS A CHAT AND NOT A FORM

The first version was two text boxes and a Preview button. It worked and it was
wrong, because writing a brief is not data entry -- it is the part where you
work out what you actually want to know. A form asks you to arrive already
knowing; a conversation gets there with you, and the questions it asks ("who
are you talking to?", "is budget a number or a range?") are exactly the ones
that turn a vague intention into a schema worth interviewing against.

HOW IT WORKS

One model call per turn, returning everything at once: a reply to say, and
revised artefacts when the conversation has moved them on. The alternative --
a chat model with tools to set the brief, set the participant and regenerate
the fields -- is more moving parts to do the same thing, and every extra tool
is another round trip in a conversation that should feel immediate.

The fields it produces are a DRAFT that can keep changing while the project is
being designed. That does not contradict freezing them: a conversation
snapshots the schema when it STARTS (see `adopt_project`), so an interview
already under way is unaffected by later edits, while a link that has not been
used yet picks up whatever the project says now. Which is exactly what someone
means by "I changed my mind, the link should gather the new things too".
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.services.blueprint import (
    MAX_BRIEF_CHARS,
    MAX_PARTICIPANT_CHARS,
    normalise,
)
from app.services.llm import get_llm, strip_degeneration
from app.services.websearch import search_web

log = structlog.get_logger()

# Enough to design an interview, bounded so a long session cannot grow the
# prompt without limit. The artefacts carry the substance; the transcript only
# has to carry the reasoning that produced them.
MAX_TURNS = 40

# What one lookup is worth carrying. Enough to name the things a field ought to
# cover, not so much that the findings crowd out the conversation that asked
# for them -- the point is a better draft, not a literature review.
RESEARCH_HITS = 5
RESEARCH_CHARS = 700

# TWO THINGS HERE ARE SCARS, not style.
#
# `fields` IS REQUIRED. With only `reply` required, the cheapest valid
# completion is a reply and nothing else, and that is what came back turn after
# turn -- including turns whose reply said "I have drafted a few initial data
# points" while returning none. Requiring them is what actually makes the panel
# fill up as you talk, and restating them every turn is not waste: it IS the
# revision mechanism.
#
# THERE IS NO `title`. Asked for one inline, flash-lite degenerated into it --
# sixty words of near-synonyms one turn ("Properly Accurately Correctly Exactly
# Precisely…"), a Korean phrase repeated thirty times the next -- which ate the
# token budget, truncated the JSON, and cost the whole turn. `maxLength` did
# not hold it. The project is named by `title_for` instead, in its own call,
# where a runaway costs one short string and nothing else.
SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "maxLength": 1200},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 40},
                    "label": {"type": "string", "maxLength": 60},
                    "description": {"type": "string", "maxLength": 400},
                    "type": {"type": "string", "enum": ["STRING", "NUMBER", "ARRAY"]},
                    "required": {"type": "boolean"},
                },
                "required": ["name", "label", "description", "type", "required"],
            },
        },
        "brief": {"type": "string", "maxLength": MAX_BRIEF_CHARS},
        "participant": {"type": "string", "maxLength": MAX_PARTICIPANT_CHARS},
        # Spellings for the microphone, not facts for the profile.
        "vocabulary": {
            "type": "array",
            "items": {"type": "string", "maxLength": 60},
            "description": (
                "Words and names the PARTICIPANT is likely to say that a "
                "speech recogniser gets wrong: jargon, tools, acronyms, proper "
                "nouns, anything spelled unusually. Written exactly as they "
                "should appear. Drawn from THIS subject only."
            ),
        },
        "ready": {"type": "boolean"},
        # Its own decision to finish, rather than waiting to be told. Described
        # for the same reason `research` is: it changes what happens next
        # instead of what gets stored, and an undescribed key is one the model
        # never touches.
        "synthesise": {
            "type": "boolean",
            "description": (
                "True to settle the data points NOW and generate the link, "
                "exactly as if they had pressed Synthesise. Set it when you "
                "have enough to interview against and there is no important "
                "question left -- not on every turn you happen to be ready."
            ),
        },
        # DESCRIBED, unlike its neighbours, because it is the one key whose
        # meaning is not obvious from its name and the only one that changes
        # what happens next rather than what gets stored. Left bare it was
        # never once used, however firmly the system prompt asked for it.
        "research": {
            "type": "string",
            "maxLength": 200,
            "description": (
                "A web search to run BEFORE you answer, when a specialist "
                "would know specifics you would otherwise guess at -- what a "
                "role is actually hired on, what an audit normally covers, "
                "what a process usually involves. You will be shown the "
                "results and asked again. Empty when you do not need one."
            ),
        },
    },
    "required": ["reply", "fields"],
}

SYSTEM = """You help someone design a spoken interview. They tell you what they
want to find out; you turn it into the data points worth interviewing against.

YOU ARE NOT THE INTERVIEWER. You are designing the interview somebody else will
be given. Never ask the user the questions you are drafting -- ask them about
the questions.

DO THE WORK. DO NOT INTERVIEW THEM ABOUT IT.

This is the difference between useful and exhausting. Somebody who says "I want
to interview candidates for a software engineering role" has told you plenty:
draft the whole set of data points, right there, as an expert in that field
would. Do not ask them which ones they want, one at a time, and assemble the
list out of their answers -- if they had the list already they would not need
you.

So, every turn: make the confident, well-informed draft first. Then ask AT MOST
ONE question, and only one that would genuinely change the design -- a fork you
cannot call ("are these graduates or seniors?"), not a detail you can decide
("should the description mention frameworks?"). If nothing important is
genuinely uncertain, ask nothing and say what you would change next.

Assume, and say what you assumed. "I've assumed mid-level and weighted it
towards React and Node since you mentioned AI — say the word and I'll shift it"
is worth ten questions. They correct assumptions far more readily than they
answer interrogations.

USE THE WEB rather than guess. Put a query in `research` and you will be shown
results and asked again, before anything you write is seen. Reach for it
whenever the draft would otherwise be generic -- and the FIRST turn about a new
subject is nearly always one of those, because that is the turn where you know
least and are about to commit to a whole list.

"Software engineer" should send you to look up what those roles are actually
hired on this year, not to write "technical skills" and "communication". Same
for a supplier audit, a tenancy check, a clinical intake, a grant application:
every field has specifics that an insider names and an outsider talks around,
and the difference between the two is this one search.

Skip it only when the subject is genuinely about their own business -- their
pricing, their team, their customers -- which the web cannot tell you.

TONE

Warm, brisk, a colleague rather than a form. Contractions are fine. Short
turns. No preamble, no "great question", no bulleted summaries of what you just
did -- they can see the data points, they are on the screen beside you.

WHAT TO RETURN, EVERY TURN

`reply` is what you say. Always present.

`fields` are the data points. Each needs a snake_case name, a readable label, a
type of STRING, NUMBER or ARRAY, whether it is required, and a description that
tells the interviewer what counts as an answer and what to do with an awkward
one. Write descriptions as instruction, not definition: "Their approximate
annual budget, as a number in their own currency. A range is fine -- take the
midpoint and put the range in a note" is useful; "the budget" is not.

FILL THEM OUT PROPERLY from the first turn. Five to eight considered data
points on turn one, not two placeholders and a question. Then revise them every
turn the conversation moves them on; returning fields does not end anything.

Mark as few required as you honestly can -- three to six is usually right.
Every required field is one the interviewer will keep pushing for, and an
interview that cannot end until it has all twelve is an interrogation.

`vocabulary` IS FOR THE MICROPHONE. Speech recognition mangles exactly the
words that matter most: "React" becomes "react", "GCP" becomes "GCP" or "G C
P" or "jeep", "Node.js" becomes "node J S". List the terms this particular
participant is likely to say, spelled the way they should be written, and the
interviewer will hear them correctly.

Draw them FROM THE SUBJECT, always. An interview about front-end engineering
wants React, Node.js, TypeScript, Kubernetes, GCP. One about Norse mythology
wants Ragnarok, Yggdrasil, Snorri Sturluson, Skaldskaparmal. One about
cardiology wants echocardiogram, atrial fibrillation, NSTEMI. A list of
programming languages in a mythology interview is worse than no list at all --
it biases the recogniser towards words nobody is going to say.

Twenty to forty terms is right, and they should be the ones an outsider would
misspell. There is no value in listing "budget" or "team".

`brief` is the instruction as you now understand it, rewritten in full each time
it changes. Write it as an instruction to an interviewer.

`participant` is what is known about the person being interviewed, as a
paragraph. Context only -- it is never read back to them.

`ready` is true once the fields are good enough to run with, which is usually
straight away. Say so, and make clear they can keep changing things.

`synthesise` FINISHES THE JOB YOURSELF. Set it when the data points would
stand up to a real interview and you have nothing important left to ask -- you
do not need permission, and making somebody hunt for a button to confirm what
you have just told them is finished is the form this replaced. It settles the
schema and generates the link in the same turn.

Set it ONCE, when it is genuinely ready. Not on the first turn just because you
produced some fields, and not again on every later turn -- they have the link
by then, and it keeps working as the data points change.

WHERE THINGS ARE ON THEIR SCREEN

You are the Design tab of a project. Beside this conversation, a panel shows
the data points as you write them and who is being interviewed, so there is no
need to list them back -- they are already on screen.

Two more tabs along the top:

  Links    one link per person, and whether each has been opened
  Results  what each participant said, filed against the data points

When you generate a link, SAY WHERE IT IS -- "the link is under Links" -- and
that it can be sent to anyone, with no account needed at their end. Point at
Results when there would be something there to read. Do not describe the
interface beyond that, and never name a button that is not one of these.

`brief` and `participant` may be omitted on a turn that does not change them.
`fields` may not: return the whole list every time, including the ones you are
leaving alone. A partial list is read as the new list, and the fields you left
out would be deleted."""


def _history(messages: list[dict]) -> str:
    lines = []
    for m in messages[-MAX_TURNS:]:
        who = "THEM" if m.get("role") == "user" else "YOU"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines)


def _state(project: Any) -> str:
    """What has been agreed so far, so the model revises rather than restarts."""
    fields = project.fields or []
    shown = (
        "\n".join(
            f"  {f['name']} ({f.get('type', 'STRING')}, "
            f"{'required' if f.get('required') else 'optional'}): "
            f"{f.get('description', '')}"
            for f in fields
        )
        or "  (none yet)"
    )
    return (
        f"Current brief:\n{project.brief or '(none yet)'}\n\n"
        f"Current participant notes:\n{project.participant or '(none yet)'}\n\n"
        f"Current fields:\n{shown}"
    )


# Appended when somebody presses "Synthesise now". It overrides the designer's
# instinct to ask one more question, which is the right instinct right up until
# the moment the person says they have had enough.
FINISH = """
THEY HAVE PRESSED "SYNTHESISE NOW". Do not ask another question first.

Return the COMPLETE `fields` list, along with `brief` and `participant`, built
from everything the conversation has given you and your best judgement for
whatever it has not. Set `ready` to true.

In `reply`, say in a sentence or two what you produced, and name anything you
had to guess at -- that is what they will correct, and they can only correct
what you admit to. They can keep talking afterwards and the data points will
follow along."""


async def respond(
    project: Any, messages: list[dict], message: str, force: bool = False
) -> dict:
    """One designer turn. Returns the reply and whatever it revised.

    `force` is "synthesise now": the same call, told to stop asking and commit.
    It is the same code path rather than a separate one-shot generator so that
    what comes out is the schema THIS CONVERSATION built, rather than a second
    reading of the brief that quietly discards everything said around it.

    Never raises on a bad model response: a designer that dies on malformed
    JSON loses the conversation that produced the schema, which is far more
    expensive than a turn that says it did not follow.
    """
    prompt = (
        f"{_state(project)}\n\n"
        f"Conversation so far:\n{_history(messages)}\n\n"
        f"THEM: {message}"
        f"{FINISH if force else ''}"
    )

    parsed = await _ask(prompt)
    if parsed is None:
        return {"reply": "Sorry, I lost that. Could you say it again?"}

    # THE SECOND PASS: look something up, then draft again knowing it.
    #
    # Not a tool call. Gemini will not take `tools` and `responseSchema`
    # together, and the structured output is the whole contract here -- so the
    # query comes back as a field, we run it, and ask again with the findings
    # in front of it.
    #
    # THE OPENING TURN SEARCHES WHETHER OR NOT IT ASKED TO. Left to choose, the
    # model essentially never asks: it is not uncertain, it is confidently
    # generic, and "Technical Skills / Communication / Problem Solving" is
    # exactly what that confidence produces. The first turn about a subject is
    # also the one where it knows least and is about to commit to a whole list,
    # so that is the one turn worth spending a lookup on by default. Afterwards
    # it is back to the model's judgement, because by then the conversation --
    # not the web -- is what it is missing.
    opening = not (project.fields or [])
    query = str(parsed.get("research") or "").strip() or (message if opening else "")
    findings = await _research(query)
    if findings:
        parsed = await _ask(f"{prompt}\n\n{findings}") or parsed

    out: dict[str, Any] = {
        "reply": str(parsed.get("reply") or "").strip()
        or "Tell me more about what you want to find out."
    }
    if isinstance(parsed.get("brief"), str) and parsed["brief"].strip():
        out["brief"] = parsed["brief"].strip()[:MAX_BRIEF_CHARS]
    if isinstance(parsed.get("participant"), str) and parsed["participant"].strip():
        out["participant"] = parsed["participant"].strip()[:MAX_PARTICIPANT_CHARS]
    if isinstance(parsed.get("fields"), list) and parsed["fields"]:
        # Through the SAME validator the one-shot path uses. These become a
        # tool schema, and a malformed field name is not a bad label -- it is a
        # setup message the Live API rejects, which fails the session outright.
        cleaned = normalise(parsed["fields"])
        if cleaned:
            out["fields"] = cleaned
    if isinstance(parsed.get("vocabulary"), list):
        out["vocabulary"] = clean_vocabulary(parsed["vocabulary"])
    out["ready"] = bool(parsed.get("ready")) or bool(out.get("fields"))
    # Only meaningful alongside a schema. A model that asks to finish while
    # returning nothing to finish is asking for an empty interview.
    out["synthesise"] = bool(parsed.get("synthesise")) and bool(out.get("fields"))

    log.info(
        "designer_turn",
        revised=[
            k for k in ("brief", "participant", "fields", "vocabulary") if k in out
        ],
        ready=out["ready"],
        synthesise=out["synthesise"],
        researched=bool(findings),
    )
    return out


async def _ask(prompt: str) -> dict | None:
    """One structured call. None when the answer could not be read.

    Never raises on a bad model response: a designer that dies on malformed
    JSON loses the conversation that produced the schema, which is far more
    expensive than a turn that says it did not follow.
    """
    raw = await get_llm().generate(
        prompt,
        system=SYSTEM,
        schema=SCHEMA,
        temperature=0.3,
        # Twelve fields with a real description each is most of this budget,
        # and a turn that revises the whole schema writes all of them. At 2000
        # the forced turn -- the one that always writes every field -- was the
        # one that truncated, and truncated JSON is unparsable JSON.
        max_output_tokens=6000,
    )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        log.warning("designer_unparsable", chars=len(str(raw)), tail=str(raw)[-200:])
        return None
    return parsed if isinstance(parsed, dict) else None


async def _research(query: str) -> str:
    """What the web says about `query`, as prompt text. "" when there is none.

    Never load-bearing, in either direction: no search key, a provider outage
    or a useless query all degrade to designing from the model's own knowledge,
    which is what it did before this existed.
    """
    query = query.strip()[:200]
    if not query:
        return ""

    try:
        hits = await search_web(query, limit=RESEARCH_HITS)
    except Exception as exc:  # noqa: BLE001 - a failed lookup is not a failed turn
        log.warning("designer_research_failed", error=str(exc)[:200])
        return ""
    if not hits:
        return ""

    log.info("designer_research", query=query[:80], hits=len(hits))
    lines = "\n\n".join(
        f"{h.filename}\n{(h.text or '')[:RESEARCH_CHARS]}" for h in hits
    )
    return (
        f'You asked to look up "{query}". Here is what came back.\n\n'
        f"{lines}\n\n"
        "Use it to make the data points SPECIFIC -- the things this subject "
        "actually turns on, named the way people in it would name them. Do not "
        "quote it, cite it, or mention searching; it is background, and the "
        "user only wants the better data points that came of it. Leave "
        "`research` empty now."
    )


# The API takes up to 1000 phrases and recommends staying near 100; past that
# the bias is spread so thin it stops helping. Forty is about what one subject
# actually has, and leaves room for the operator to think about the list.
MAX_VOCABULARY = 60


def clean_vocabulary(raw: list) -> list[str]:
    """Deduplicated, trimmed, and capped. Case is PRESERVED.

    The casing is the point: "Node.js" and "node js" are the same phrase to a
    recogniser and different things to whoever reads the profile afterwards,
    and this list is what the interviewer copies the spelling from.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        term = " ".join(item.split())[:60]
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= MAX_VOCABULARY:
            break
    return out


async def title_for(brief: str) -> str:
    """A short name for the project, in its own call.

    Separate from the designer turn because naming is the one thing that
    reliably sends this model into a repetition loop, and inline it took the
    data points down with it. Here a runaway costs a bad title: the budget is
    forty tokens, `strip_degeneration` cuts the loop, and anything still
    implausible is dropped for the caller's fallback.
    """
    text = (brief or "").strip()
    if not text:
        return ""

    try:
        raw = await get_llm().generate(
            f"Brief:\n{text[:MAX_BRIEF_CHARS]}",
            system=(
                "Name this interview project the way someone would refer to it "
                "in a corridor: two to five words, no quotes, no version "
                'numbers, no trailing punctuation. For example "SMB churn '
                'interviews" or "Inbound lead qualification". Reply with the '
                "name and nothing else."
            ),
            temperature=0.2,
            max_output_tokens=40,
        )
    except Exception as exc:  # noqa: BLE001 - a nameless project still works
        log.warning("title_failed", error=str(exc)[:200])
        return ""

    name = strip_degeneration(str(raw or "")).strip().strip('"').splitlines()[:1]
    name = (name[0] if name else "").strip().strip('"').rstrip(".")
    # Eight words is already twice what was asked for. Past that it is not a
    # short name that came out long, it is the loop again.
    return name[:60] if name and len(name.split()) <= 8 else ""

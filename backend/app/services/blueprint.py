"""Turn a plain-English brief into the schema an interview will fill.

THE DESIGN QUESTION THIS ANSWERS

Speak and Interview have a fixed field list written in Python. Howler does not:
the operator says "find out their budget, their timeline, who signs off, and
what they use today", and those become the data points. So where does the
schema come from, and when?

IT IS GENERATED ONCE AND THEN FROZEN. Not improvised turn by turn. Four things
depend on the schema being stable for the life of a conversation:

  * COMPLETENESS NEEDS A DENOMINATOR. "Four of six" is meaningless if six can
    change; an interview whose target moves can never be finished, and an agent
    that invents a new requirement every turn will interview somebody for ever.

  * THE TOOL DECLARATION IS FIXED AT CONNECT TIME. The Live API takes its
    function declarations in the setup message. Fields discovered mid-session
    could not be declared, so the model would have nowhere to put them.

  * RESUMPTION HAS TO FIND THE SAME SHAPE. A conversation continued tomorrow
    reloads its schema from the row; regenerating it from the brief would
    quietly produce a different one, and the half-filled profile would no
    longer line up with it.

  * THE CARD NEEDS STABLE KEYS. A column that appears and disappears between
    renders is not a profile, it is a rumour.

So: the brief is turned into a field list once, stored on the session, and read
back every time. The model can still record things the schema did not
anticipate -- they land in `other` notes, exactly as they do in Interview,
which is the escape hatch that makes a frozen schema safe rather than limiting.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from app.services.llm import extract_object_list, get_llm

log = structlog.get_logger()

# Room for a real brief, and a ceiling so one cannot become the system prompt.
MAX_BRIEF_CHARS = 4_000
MAX_PARTICIPANT_CHARS = 4_000

# More than this and the interview cannot finish inside a session, and the card
# stops being readable at a glance -- which is the entire point of a card.
MAX_FIELDS = 12

TYPES = {"STRING", "NUMBER", "ARRAY"}

SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "type": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["name", "label", "description", "type", "required"],
            },
        }
    },
    "required": ["fields"],
}

SYSTEM = """You turn a brief into the fields an interviewer should fill.

The brief says what someone wants to find out from a conversation. Your job is
to name the data points that answer it -- not to plan the conversation, and not
to write questions.

RULES

Return between three and twelve fields. Fewer than three is not worth an
interview; more than twelve cannot be gathered in one sitting, and a card with
twenty rows is not read.

`name` is snake_case, lowercase, no spaces, and is how the data will be stored.
`label` is what a person reads: "Budget range", "Decision makers".

`description` tells the interviewer what counts as an answer, and what to do
with an awkward one. Write it as instruction, not as definition. "Their
approximate annual budget for this, as a number in their own currency. A range
is fine -- record the midpoint and put the range in a note" is useful.
"The budget" is not.

`type` is STRING, NUMBER or ARRAY. Use ARRAY only where several distinct
answers are genuinely expected -- tools they use, people involved. A single
answer with detail is a STRING; the detail belongs in a note.

`required` marks what the interview is FOR. Mark as few as you honestly can:
every required field is a thing the interviewer will keep pushing for, and an
interview that cannot end until it has all twelve becomes an interrogation.
Three to six required is usually right. Anything the brief calls "if possible",
"ideally" or "nice to have" is not required.

DO NOT add fields the brief did not ask for. A name field, a contact field, a
catch-all "other notes" -- the interviewer captures all of that anyway, and
inventing requirements makes the conversation longer for nobody's benefit."""


def _slug(name: str) -> str:
    """A safe, stable key. The model writes these and they become dict keys."""
    out = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return out[:40]


# Names the profile machinery uses for its own purposes. A field called `notes`
# would be merged into the note bucket and never appear as a field.
RESERVED = {"notes", "quotes", "summary", "ended", "completed_on", "general", "other"}


def normalise(raw: list[dict]) -> list[dict[str, Any]]:
    """Validate and clean what the model returned.

    Everything here is defensive because the output becomes a TOOL SCHEMA. A
    malformed field name is not a bad label, it is a setup message the Live API
    rejects -- and the session then fails to open at all, which is how an
    invalid `items` on an array took out the whole interview mode once already.
    """
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _slug(item.get("name") or item.get("label") or "")
        if not name or name in seen or name in RESERVED:
            continue
        kind = str(item.get("type") or "STRING").upper()
        if kind not in TYPES:
            kind = "STRING"
        fields.append(
            {
                "name": name,
                "label": str(item.get("label") or name.replace("_", " ")).strip()[:60],
                "description": str(item.get("description") or "").strip()[:400],
                "type": kind,
                "required": bool(item.get("required")),
            }
        )
        seen.add(name)
        if len(fields) >= MAX_FIELDS:
            break

    # Nothing required means nothing to finish. The model sometimes marks every
    # field optional when the brief is phrased gently, and the interview then
    # has no completion condition at all -- so the first few become required.
    if fields and not any(f["required"] for f in fields):
        for field in fields[: min(3, len(fields))]:
            field["required"] = True
        log.info("blueprint_forced_required", n=min(3, len(fields)))

    return fields


async def from_brief(brief: str) -> list[dict[str, Any]]:
    """The data points a brief is asking for. Raises on an unusable result."""
    text = (brief or "").strip()[:MAX_BRIEF_CHARS]
    if not text:
        raise ValueError("A brief is required.")

    raw = await get_llm().generate(
        f"Brief:\n{text}",
        system=SYSTEM,
        schema=SCHEMA,
        temperature=0.1,
        max_output_tokens=1500,
    )
    fields = normalise(extract_object_list(raw, "fields"))
    if len(fields) < 2:
        raise ValueError(
            "That brief did not produce a usable set of fields. Try describing "
            "what you want to find out in a sentence or two."
        )

    log.info(
        "blueprint_built",
        n=len(fields),
        required=[f["name"] for f in fields if f["required"]],
    )
    return fields

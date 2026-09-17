"""The profile an interview is trying to build, and the tool that fills it.

WHY A TOOL AND NOT AN EXTRACTION PASS

The obvious design is to let the interview run and parse a profile out of the
transcript afterwards. This records it AS IT IS LEARNED instead, through a tool
the model calls, for two reasons:

  * The model knows what it just heard. A later parser is guessing from prose
    what was already unambiguous in the moment -- "about five years, maybe six"
    is trivially `5` to the thing that heard it and a coin flip to a regex.

  * THE TOOL'S RESULT IS THE STEERING. Every call returns the merged profile
    and what is still missing, so the same action that records an answer tells
    the interviewer what to ask next. Nothing has to re-derive "what do I still
    need" from a long conversation, which is exactly the reasoning that decays
    as a transcript grows and that context compression would eventually destroy
    outright.

WHY IT IS NOT SENT BACK ON EVERY TURN

An earlier sketch of this echoed the whole profile into the model on each turn.
It is unnecessary: a live session already retains its own context, and the tool
result covers the case where it does not. Sending it unprompted would spend
tokens on every turn to restate something that changed on few of them.
"""

from __future__ import annotations

from typing import Any

# The fields an interview is trying to fill.
#
# REQUIRED ones decide when the interview is done; the rest are welcome and
# never block. The split matters: an interviewer that treats every field as
# mandatory interrogates, and one that treats none as mandatory never finishes.
FIELDS: list[dict[str, Any]] = [
    {
        "name": "full_name",
        "required": True,
        "description": "The person's name, as they said it.",
    },
    {
        "name": "current_role",
        "required": True,
        "description": (
            "Their job title or what they do now, in their own words. "
            "'Between roles' or 'student' are valid answers."
        ),
    },
    {
        "name": "years_experience",
        "required": True,
        "type": "NUMBER",
        "description": (
            "Total years of professional experience, as a number. Round to the "
            "nearest year; 'about five or six' is 5."
        ),
    },
    {
        "name": "skills",
        "required": True,
        "type": "ARRAY",
        "description": (
            "Skills, tools and technologies they actually work with. Record "
            "what they say, not what you infer from their job title."
        ),
    },
    {
        "name": "interests",
        "required": True,
        "type": "ARRAY",
        "description": (
            "What they are interested in working on or learning. Subjects and "
            "problems, not hobbies, unless the hobby is relevant to the work."
        ),
    },
    {
        "name": "work_setup",
        "required": True,
        "description": (
            "Their preference: 'remote', 'office', or 'hybrid'. If they give a "
            "detail like 'hybrid, two days in', record 'hybrid' and put the "
            "detail in notes."
        ),
    },
    {
        "name": "location",
        "required": False,
        "description": "Where they are based, if they mention it.",
    },
    {
        "name": "availability",
        "required": False,
        "description": (
            "When they could start, or their notice period, if it comes up."
        ),
    },
    {
        "name": "looking_for",
        "required": False,
        "description": "What they want from their next role, if they say.",
    },
]

# Notes are NOT a field. They are a second channel that runs alongside every
# field, and the reason is the difference between a profile and a spreadsheet.
#
# "work_setup: hybrid" is true and nearly useless. "hybrid -- was firm about
# it, mentioned a long commute and sounded like he had negotiated it before"
# is the same answer with the thing that makes it actionable still attached.
# Sentiment, hesitation, enthusiasm, the aside that explains the answer: all of
# it is present when the model hears it and gone by the time anyone reads the
# table.
#
# Keyed by field so an observation stays attached to what it is about, with
# "general" for anything that belongs to the person rather than to one answer.
NOTES_KEY = "notes"
GENERAL = "general"

NOTE_ITEM = {
    "type": "OBJECT",
    "properties": {
        "field": {
            "type": "STRING",
            "description": (
                "Which field this is about -- one of the field names above, "
                f"or '{GENERAL}' for something about them overall."
            ),
        },
        "note": {
            "type": "STRING",
            "description": (
                "What was notable about HOW they answered: hesitation, "
                "enthusiasm, a caveat, an aside, a reason. One sentence. Do "
                "not restate the answer itself."
            ),
        },
    },
    "required": ["field", "note"],
}

# What the model calls the text, in practice.
#
# The schema says `note`. It has been observed sending `observation` as well,
# and bare strings with no object at all -- the API does not enforce an array's
# item schema, so whatever the model produces is what arrives. Reading only the
# declared key cost every note in a full interview: they were recorded, the
# tool reported success, and the stored profile came back empty.
_NOTE_KEYS = ("note", "observation", "text", "value", "comment")

REQUIRED = [f["name"] for f in FIELDS if f["required"]]
FIELD_NAMES = [f["name"] for f in FIELDS]
TOOL_NAME = "record_profile"


def declaration() -> dict[str, Any]:
    """The tool spec, generated from FIELDS so the two cannot drift.

    EVERY FIELD IS OPTIONAL in the schema, deliberately, even the required
    ones. The model records what it has learned so far and calls again as it
    learns more; a schema that demanded all of them would force it to either
    invent the missing ones or never call the tool at all.
    """
    return {
        "name": TOOL_NAME,
        "description": (
            "Record what you have learned about the person so far. Call this "
            "as soon as you learn something, not at the end -- it returns the "
            "profile built up so far and exactly which fields are still "
            "missing, so it is how you know what to ask next. Only pass the "
            "fields you actually learned; leave the rest out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                field["name"]: (
                    {
                        "type": field.get("type", "STRING"),
                        "description": field["description"],
                        **(
                            {"items": {"type": "STRING"}}
                            if field.get("type") == "ARRAY"
                            else {}
                        ),
                    }
                )
                for field in FIELDS
            }
            | {
                NOTES_KEY: {
                    "type": "ARRAY",
                    "items": NOTE_ITEM,
                    "description": (
                        "What was notable about HOW they answered, attached to "
                        "the field it is about. Sentiment, hesitation, "
                        "enthusiasm, a caveat, the reason behind an answer. "
                        "This is what makes the profile more than a "
                        "spreadsheet, so record it whenever there is anything "
                        "to record -- not only when asked."
                    ),
                }
            },
            "required": [],
        },
    }


def merge(existing: dict, update: dict) -> dict:
    """Fold a tool call's arguments into the profile so far.

    LAST ANSWER WINS for scalars, because people correct themselves -- "five
    years, sorry, six" must end at six. Lists are UNIONED instead, because a
    second mention of skills is nearly always additional rather than a
    correction, and dropping the first set would silently lose half of them.
    """
    merged = dict(existing or {})

    # Notes take their own path: they arrive as a list of {field, observation}
    # and are stored keyed by field, because an observation is only worth
    # keeping while it is still attached to what it is about.
    incoming_notes = (update or {}).get(NOTES_KEY) or []
    if incoming_notes:
        notes = {k: list(v) for k, v in (merged.get(NOTES_KEY) or {}).items()}
        for note in incoming_notes:
            # BOTH SHAPES ARE ACCEPTED, and this is not defensiveness.
            #
            # The tool declares notes as {field, observation} objects. The model
            # frequently sends bare strings instead -- measured, on the very
            # first call -- and the API does not enforce the item schema. The
            # original merge skipped anything that was not a dict, so every note
            # was silently discarded: the model dutifully recorded them, the
            # tool reported success, and the stored profile came back `{}`.
            #
            # A note that arrives unattached is filed under `general`. Losing
            # the field it belonged to is a small harm; losing the observation
            # is the whole feature.
            if isinstance(note, str):
                note = {"field": GENERAL, "note": note}
            if not isinstance(note, dict):
                continue
            field = str(note.get("field") or GENERAL).strip() or GENERAL
            # An unknown field name is filed under `general` rather than
            # dropped. A misattributed observation is still an observation;
            # a discarded one is gone.
            if field not in FIELD_NAMES and field != GENERAL:
                field = GENERAL
            text = ""
            for key in _NOTE_KEYS:
                if note.get(key):
                    text = str(note[key]).strip()
                    break
            if not text:
                # A dict with neither `field` nor a known text key is most
                # likely {field_name: observation}. Reading it that way
                # recovers the note rather than discarding it.
                extras = {
                    k: v
                    for k, v in note.items()
                    if k != "field" and isinstance(v, str) and v.strip()
                }
                if len(extras) == 1:
                    field, value = next(iter(extras.items()))
                    text = value.strip()
            if not text:
                continue
            bucket = notes.setdefault(field, [])
            if text.lower() not in {n.lower() for n in bucket}:
                bucket.append(text)
        merged[NOTES_KEY] = notes

    for key, value in (update or {}).items():
        if key == NOTES_KEY:
            continue
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            seen = list(merged.get(key) or [])
            for item in value:
                text = str(item).strip()
                # Case-insensitive, because "Python" and "python" from two
                # different turns are one skill.
                if text and text.lower() not in {s.lower() for s in seen}:
                    seen.append(text)
            merged[key] = seen
        else:
            merged[key] = value
    return merged


def missing(profile: dict) -> list[str]:
    """Required fields with nothing in them yet.

    Notes are never required. An interview that will not finish until every
    answer has an observation attached is one that invents observations.
    """
    return [
        name
        for name in REQUIRED
        if not (profile or {}).get(name) and (profile or {}).get(name) != 0
    ]


def complete(profile: dict) -> bool:
    return not missing(profile)


def render(profile: dict) -> str:
    """The tool's reply to the model: what is known, and what is not.

    Written as lines rather than returned as JSON for the same reason the
    corpus tools are: a model reads lines back as facts and rewrites them,
    and reads JSON back as a structure to echo.
    """
    notes = (profile or {}).get(NOTES_KEY) or {}
    lines = ["Profile so far:"]
    for field in FIELDS:
        value = (profile or {}).get(field["name"])
        if value in (None, "", []):
            continue
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else value
        lines.append(f"- {field['name']}: {shown}")
        # Echoed back so the model can see what it has already observed and
        # does not record the same thing three times in different words.
        for note in notes.get(field["name"], []):
            lines.append(f"    note: {note}")
    if len(lines) == 1:
        lines.append("- (nothing recorded yet)")

    for note in notes.get(GENERAL, []):
        lines.append(f"- general note: {note}")

    gaps = missing(profile)
    if gaps:
        lines.append("")
        lines.append("STILL MISSING, ask about these next: " + ", ".join(gaps))
    else:
        lines.append("")
        lines.append(
            "ALL REQUIRED FIELDS ARE NOW FILLED. Tell the person you have "
            "everything you need, thank them, and offer them a chance to add "
            "anything you did not ask about."
        )
    return "\n".join(lines)

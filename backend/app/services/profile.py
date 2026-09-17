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
    {
        "name": "notes",
        "required": False,
        "description": (
            "Anything else worth knowing that has no field of its own. Short."
        ),
    },
]

REQUIRED = [f["name"] for f in FIELDS if f["required"]]
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
    for key, value in (update or {}).items():
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
    """Required fields with nothing in them yet."""
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
    lines = ["Profile so far:"]
    for field in FIELDS:
        value = (profile or {}).get(field["name"])
        if value in (None, "", []):
            continue
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else value
        lines.append(f"- {field['name']}: {shown}")
    if len(lines) == 1:
        lines.append("- (nothing recorded yet)")

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

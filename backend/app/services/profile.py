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
QUOTES_KEY = "quotes"
GENERAL = "general"
# Anything they said that belongs to no field. NOT a dumping ground: it is
# where the interesting half of an interview usually ends up, because the
# fields were chosen in advance and the person was not.
OTHER = "other"

# Buckets a note may be filed under, beyond the fields themselves.
EXTRA_BUCKETS = (GENERAL, OTHER)

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
                "What was notable: tone, energy, hesitation, enthusiasm, "
                "reluctance, pride, frustration, a caveat, a reason, an aside, "
                "a correction, something volunteered unasked. Anything a person "
                "reading this later would want to know."
            ),
        },
    },
    "required": ["field", "note"],
}

QUOTE_ITEM = {
    "type": "OBJECT",
    "properties": {
        "field": {
            "type": "STRING",
            "description": (
                "Which field this is about -- a field name, "
                f"'{GENERAL}', or '{OTHER}'."
            ),
        },
        "quote": {
            "type": "STRING",
            "description": (
                "Their own words, as close to verbatim as you can manage. "
                "Short -- a phrase or a sentence, not a paragraph."
            ),
        },
    },
    "required": ["field", "quote"],
}

# What the model calls the text, in practice.
#
# The schema says `note`. It has been observed sending `observation` as well,
# and bare strings with no object at all -- the API does not enforce an array's
# item schema, so whatever the model produces is what arrives. Reading only the
# declared key cost every note in a full interview: they were recorded, the
# tool reported success, and the stored profile came back empty.
_NOTE_KEYS = ("note", "observation", "text", "value", "comment")
_QUOTE_KEYS = ("quote", "text", "said", "value")

REQUIRED = [f["name"] for f in FIELDS if f["required"]]
FIELD_NAMES = [f["name"] for f in FIELDS]


# ---------------------------------------------------------------------------
# The field list is a PARAMETER, not a constant.
#
# Interview has a fixed set written above. Howler generates its own from a
# brief and stores it on the session, so every function here takes the fields
# it should work against and defaults to the built-in ones. That default is
# what keeps Interview unchanged: one code path, two sources of schema.
# ---------------------------------------------------------------------------


def _fields(fields: list[dict] | None) -> list[dict]:
    return fields if fields else FIELDS


def names_of(fields: list[dict] | None = None) -> list[str]:
    return [f["name"] for f in _fields(fields)]


def required_of(fields: list[dict] | None = None) -> list[str]:
    return [f["name"] for f in _fields(fields) if f.get("required")]
TOOL_NAME = "record_profile"
END_TOOL = "end_interview"


def end_declaration() -> dict[str, Any]:
    """The tool that closes an interview.

    SEPARATE FROM COMPLETENESS, deliberately. "Every required field is filled"
    and "this conversation is over" are different facts: the interviewer fills
    the last field, then asks whether there is anything to add, and the answer
    to THAT is often the most useful thing in the profile. Ending on
    completeness would cut the conversation off exactly there.

    So the model says when it is done, and the microphone stops reopening.
    """
    return {
        "name": END_TOOL,
        "description": (
            "End the interview. Call this only after you have told the person "
            "you have everything you need, asked whether they want to add "
            "anything, and recorded whatever they added. After this the "
            "microphone will not reopen, so do not call it while you are still "
            "expecting an answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "STRING",
                    "description": (
                        "One sentence on who this person is, for the top of "
                        "the card."
                    ),
                },
                # HOW IT WAS SAID, gathered at the end because it is a
                # judgement about the whole conversation. The per-answer notes
                # carry the moments; this is the shape they add up to, and
                # without it a reader has to reconstruct it from twenty
                # fragments.
                "demeanour": {
                    "type": "STRING",
                    "description": (
                        "How they came across over the whole conversation, in "
                        "a sentence or two -- energy, warmth, confidence, "
                        "guardedness, how comfortable they seemed. Describe "
                        "what you HEARD, not what it means about them: "
                        "'quiet and careful, took time over each answer' is "
                        "an observation; 'lacks confidence' is a verdict you "
                        "are not in a position to reach."
                    ),
                },
                "notable_moments": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": (
                        "The two to five moments where their delivery changed "
                        "and it meant something -- what they warmed up about, "
                        "what they hurried past, where they hesitated, where "
                        "they became precise. Each one names the subject and "
                        "what changed: 'lit up describing the migration they "
                        "led -- fastest and most detailed answer'. Leave it "
                        "empty rather than inventing any."
                    ),
                },
            },
            "required": [],
        },
    }


def declaration(fields: list[dict] | None = None) -> dict[str, Any]:
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
                for field in _fields(fields)
            }
            | {
                QUOTES_KEY: {
                    "type": "ARRAY",
                    "items": QUOTE_ITEM,
                    "description": (
                        "Their actual words, attached to the field they are "
                        "about. A quote survives every summary you might write "
                        "later, and it is the one thing a reader trusts "
                        "completely. Capture them generously."
                    ),
                },
                NOTES_KEY: {
                    "type": "ARRAY",
                    "items": NOTE_ITEM,
                    "description": (
                        "EVERYTHING worth knowing that is not a field value: "
                        "tone, emotion, energy, hesitation, enthusiasm, "
                        "reluctance, pride, frustration, humour, caveats, "
                        "reasons, asides, corrections, context, anything they "
                        "volunteered. Use the field it relates to, "
                        f"'{GENERAL}' for how they came across overall, or "
                        f"'{OTHER}' for anything that fits no field at all. "
                        "Be generous -- this is the most valuable part of the "
                        "profile and there is no penalty for recording too "
                        "much."
                    ),
                }
            },
            "required": [],
        },
    }


def merge(existing: dict, update: dict, fields: list[dict] | None = None) -> dict:
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
    for bucket_key, text_keys in (
        (NOTES_KEY, _NOTE_KEYS),
        (QUOTES_KEY, _QUOTE_KEYS),
    ):
        merged = _fold(
            merged,
            (update or {}).get(bucket_key) or [],
            bucket_key,
            text_keys,
            names_of(fields),
        )

    for key, value in (update or {}).items():
        if key in (NOTES_KEY, QUOTES_KEY):
            continue
        if value is None or value == "" or value == []:
            continue

        # AN INVENTED FIELD BECOMES A NOTE, rather than a column nobody sees.
        #
        # The model passes whatever it likes: `salary_expectation`,
        # `favourite_language`. Those used to be stored verbatim as profile
        # keys -- and the card renders only the declared fields, so they were
        # written to the database and then invisible to everyone, which is the
        # worst of both. Filed under `other`, the information survives and
        # shows up where the reader is already looking.
        if key not in names_of(fields):
            merged = _fold(
                merged,
                [{"field": OTHER, "note": f"{key.replace('_', ' ')}: {_flat(value)}"}],
                NOTES_KEY,
                _NOTE_KEYS,
                names_of(fields),
            )
            continue

        if isinstance(value, list):
            seen = list(merged.get(key) or [])
            for item in value:
                text = str(item).strip()
                if text and text.lower() not in {s.lower() for s in seen}:
                    seen.append(text)
            merged[key] = seen
        else:
            merged[key] = value
    return merged


def _flat(value: Any) -> str:
    """A tool argument as one readable phrase."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _fold(
    merged: dict,
    incoming: list,
    bucket_key: str,
    text_keys: tuple,
    known: list[str] | None = None,
) -> dict:
    """Fold notes or quotes into their field-keyed bucket.

    One function for both because they differ only in which key holds the text.
    """
    known = known if known is not None else FIELD_NAMES
    if incoming:
        notes = {k: list(v) for k, v in (merged.get(bucket_key) or {}).items()}
        for note in incoming:
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
                note = {"field": GENERAL, text_keys[0]: note}
            if not isinstance(note, dict):
                continue
            field = str(note.get("field") or GENERAL).strip() or GENERAL
            text = ""
            for key in text_keys:
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

            # VALIDATED LAST, not first. The fallback above can REASSIGN the
            # field, and checking before it ran let an invented bucket through:
            # a quote sent as {"note": "..."} was filed under a bucket called
            # "note" instead of landing in `other`.
            #
            # An unknown field is filed rather than dropped. A misattributed
            # observation is still an observation; a discarded one is gone.
            if field not in known and field not in EXTRA_BUCKETS:
                field = OTHER

            bucket = notes.setdefault(field, [])
            if text.lower() not in {n.lower() for n in bucket}:
                bucket.append(text)
        merged = dict(merged)
        merged[bucket_key] = notes
    return merged


def _blank(value: Any) -> bool:
    """Nothing recorded. `0` is a value -- see `missing`."""
    return value is None or value == "" or value == []


def missing(profile: dict, fields: list[dict] | None = None) -> list[str]:
    """Required fields with nothing in them yet.

    Notes are never required. An interview that will not finish until every
    answer has an observation attached is one that invents observations.
    """
    return [
        name
        for name in required_of(fields)
        if not (profile or {}).get(name) and (profile or {}).get(name) != 0
    ]


def complete(profile: dict, fields: list[dict] | None = None) -> bool:
    return not missing(profile, fields)


def render(profile: dict, fields: list[dict] | None = None) -> str:
    """The tool's reply to the model: what is known, and what is not.

    Written as lines rather than returned as JSON for the same reason the
    corpus tools are: a model reads lines back as facts and rewrites them,
    and reads JSON back as a structure to echo.
    """
    notes = (profile or {}).get(NOTES_KEY) or {}
    quotes = (profile or {}).get(QUOTES_KEY) or {}
    lines = ["Profile so far:"]
    for field in _fields(fields):
        value = (profile or {}).get(field["name"])
        if value in (None, "", []):
            continue
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else value
        lines.append(f"- {field['name']}: {shown}")
        # Echoed back so the model can see what it has already observed and
        # does not record the same thing three times in different words.
        for note in notes.get(field["name"], []):
            lines.append(f"    note: {note}")
        for quote in quotes.get(field["name"], []):
            lines.append(f'    quote: "{quote}"')
    if len(lines) == 1:
        lines.append("- (nothing recorded yet)")

    for bucket, label in ((GENERAL, "general"), (OTHER, "other")):
        for note in notes.get(bucket, []):
            lines.append(f"- {label} note: {note}")
        for quote in quotes.get(bucket, []):
            lines.append(f'- {label} quote: "{quote}"')

    gaps = missing(profile, fields)
    # Optional fields the model has not asked about.
    #
    # They were invisible before: nothing in the tool result mentioned them, so
    # the only place they existed was one line of the prompt, and they were
    # never asked for -- "location" went unfilled in every interview. Naming
    # them here puts them in front of the model at the moment it is deciding
    # what to ask next, which is the only moment it matters.
    spare = [
        f["name"]
        for f in _fields(fields)
        if not f["required"] and _blank((profile or {}).get(f["name"]))
    ]

    if gaps:
        lines.append("")
        lines.append("STILL MISSING, ask about these next: " + ", ".join(gaps))
        if spare:
            lines.append(
                "Also still empty, and worth asking if it fits the "
                "conversation: " + ", ".join(spare)
            )
    else:
        lines.append("")
        if spare:
            # Asked BEFORE the closing question, not after -- once the model
            # has said it has everything it needs, going back to ask more is
            # confusing and reads as having misled them.
            lines.append(
                "All required fields are filled. These are still empty: "
                + ", ".join(spare)
                + ". Ask about any you have not already raised, one at a "
                "time. If you have asked and they did not want to say, that "
                "is an answer -- leave it and finish."
            )
        else:
            lines.append(
                "ALL FIELDS ARE NOW FILLED. Tell the person you have "
                "everything you need, thank them, and offer them a chance to "
                "add anything you did not ask about."
            )

    # COUNTED, not merely listed. The fields fill up early and then stop
    # moving, so without this the model gets no signal that the rich half of
    # the profile is thin -- and a thin profile is the real failure mode
    # here, not a missing field.
    n_notes = sum(len(v) for v in notes.values())
    n_quotes = sum(len(v) for v in quotes.values())
    lines.append("")
    lines.append(f"Recorded so far: {n_notes} note(s), {n_quotes} quote(s).")
    if n_notes < 6:
        lines.append(
            "That is THIN. You are almost certainly letting detail go "
            "unrecorded -- tone, reasons, asides, things they volunteered, "
            "their own words. Record more."
        )
    return "\n".join(lines)


# Field names that hold A PERSON'S name, as an ALLOWLIST rather than a pattern.
#
# Interview has a fixed `full_name`. Howler's schema is written from a brief, so
# it arrives as `candidate_name`, `respondent_name`, or nothing at all -- which
# is why a Howler conversation stayed called "Interview" and its link stayed
# "Unnamed" long after the interview had finished.
#
# Matching `*_name` instead would be shorter and wrong: `company_name`,
# `product_name` and `tool_name` are all names, and none of them is the
# person's. Naming somebody's interview after their employer is worse than
# leaving it unnamed, so a new spelling gets added here deliberately.
_NAME_FIELDS = (
    "full_name",
    "name",
    "candidate_name",
    "participant_name",
    "interviewee_name",
    "respondent_name",
    "applicant_name",
    "contact_name",
    "person_name",
)


def person_name(profile: dict) -> str:
    """The name of whoever was interviewed, or "" if it was never recorded.

    `_blank` FIRST, before `_flat`. `_flat` is a formatter, not a validator --
    it stringifies whatever it is handed, so a missing field came back as the
    literal "None" and passed every check below it. That renamed an
    unnamed conversation to "None", which the tests caught.
    """
    data = profile or {}
    for key in _NAME_FIELDS:
        raw = data.get(key)
        if _blank(raw):
            continue
        value = _flat(raw).strip()
        # A NAME, not a sentence. The model sometimes files a whole
        # introduction under a name field -- "John Doe, a senior engineer
        # looking for remote work" -- and that is a summary, not a title.
        if value and len(value) <= 60 and len(value.split()) <= 5:
            return value
    return ""

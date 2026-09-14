"""Durable instructions about HOW to answer.

WHY THIS IS NOT THE CONVERSATION SUMMARY

Both are "memory", and treating them as one thing is why a stated preference
gets lost:

    summary      compression of turns that scrolled out of the window.
                 Regenerated as the conversation grows, lossy by design, about
                 the SUBJECT MATTER.
    preference   an instruction, stated once, kept verbatim, applied to every
                 later turn, about BEHAVIOUR.

"Always search the web too" put through a summariser becomes "the user asked
about search behaviour" -- which preserves the topic and destroys the
imperative. So preferences are stored separately and never summarised.

WHY IT IS NOT JUST A LONGER SYSTEM PROMPT

Because it is per user and per conversation. A preference set in one chat
should not silently change another unless the user said "always".
"""

from __future__ import annotations

import re
import uuid

import structlog
from sqlalchemy import or_, select

from app.db.models import Preference
from app.db.session import SessionLocal
from app.services.llm import LLMError, extract_bool, get_llm

log = structlog.get_logger()

# A hard ceiling on what reaches a prompt. Preferences accumulate, and an
# unbounded list would eventually crowd out the retrieved passages the answer
# has to cite -- the instructions would win an argument with the evidence.
MAX_IN_PROMPT = 12
MAX_LENGTH = 400


async def preferences_in_force(
    *, owner_id: str | None, session_id: uuid.UUID | None
) -> list[Preference]:
    """Everything that applies to this turn: user-level plus this session's.

    One query with an OR rather than two calls, because "what is in force right
    now" is a single question and answering it in two round-trips invites the
    two halves to drift apart.
    """
    stmt = select(Preference).where(Preference.active.is_(True))
    if owner_id is not None:
        stmt = stmt.where(Preference.owner_id == owner_id)
    stmt = stmt.where(
        or_(Preference.session_id.is_(None), Preference.session_id == session_id)
    )
    # Session-scoped last, so a conversation-specific instruction appears after
    # the general one and wins when they conflict -- models weight later
    # instructions more heavily.
    stmt = stmt.order_by(Preference.session_id.is_(None).desc(), Preference.created_at)

    async with SessionLocal() as db:
        return list((await db.execute(stmt)).scalars().all())


def render_for_prompt(prefs: list[Preference]) -> str:
    """The block injected into a system prompt. Empty string when there are none.

    Empty rather than a placeholder like "(no preferences)", because a line
    saying there are no instructions is itself an instruction the model will
    reason about -- and on a small model it reliably produces a sentence
    apologising for having no preferences.
    """
    if not prefs:
        return ""
    lines = "\n".join(f"- {p.text}" for p in prefs[:MAX_IN_PROMPT])
    return (
        "\nSTANDING INSTRUCTIONS FROM THIS USER. They apply to every answer and "
        "override your defaults where they conflict:\n"
        f"{lines}\n"
    )


# Stop words for the near-duplicate test only. Two phrasings of one instruction
# differ almost entirely in these, so comparing with them in makes paraphrases
# look distinct.
_NOISE = frozenset(
    "a an and are as at be by for from in is it its me my of on or please that "
    "the this to what when whats with you your".split()
)
_WORD = re.compile(r"[a-z0-9']+")

# Jaccard overlap above which two instructions are treated as the same one.
# 0.8 is deliberately high: merging two genuinely different preferences loses
# one silently, which is worse than keeping a near-duplicate.
_SIMILAR = 0.8


# Words that reverse an instruction. Deliberately generous: a false "these
# differ" keeps a near-duplicate, which is harmless, while a false "these match"
# discards the reversal the user just asked for.
_NEGATIONS = frozenset(
    "never no not dont don't avoid stop without except exclude omit skip".split()
)


def _polarity(text: str) -> bool:
    """True when the instruction reads as a prohibition."""
    words = _WORD.findall(text.lower())
    # Both forms, so "don't" matches whether or not the apostrophe survived.
    return bool(_NEGATIONS & (set(words) | {w.replace("'", "") for w in words}))


def _fingerprint(text: str) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    return frozenset(w.replace("'", "") for w in words if w not in _NOISE)


def _is_near_duplicate(a: str, b: str) -> bool:
    """Do two instructions say the same thing?

    Exact matching is not enough, and this is measured rather than theoretical:
    the router paraphrases slightly every time, so restating one preference
    produced

        "Always let me know what info is from the internet and what is from the docs."
        "Always let me know what info is from internet and what's from the docs."

    as two rows. Each takes a slot in the prompt and makes the instruction look
    twice as emphatic as it is.
    """
    # NEGATION IS CHECKED SEPARATELY, and it has to be.
    #
    # Bag-of-words similarity scales with length, so one flipped word is a
    # large difference in a short instruction and a negligible one in a long
    # sentence: "always cite the page number and section heading for every
    # claim" against the same line starting "never" overlaps ~0.82 and would be
    # merged, silently discarding a reversal of the instruction. A preference
    # and its opposite are never the same preference, whatever the overlap.
    if _polarity(a) != _polarity(b):
        return False

    first, second = _fingerprint(a), _fingerprint(b)
    if not first or not second:
        return first == second
    overlap = len(first & second) / len(first | second)
    return overlap >= _SIMILAR


_DEDUPE_SCHEMA = {
    "type": "object",
    "properties": {
        "already_covered": {
            "type": "boolean",
            "description": "True if a stored instruction already says this.",
        },
    },
    "required": ["already_covered"],
}

_DEDUPE_SYSTEM = """You compare ONE new instruction against instructions already \
stored, and answer a single question: would storing it change anything?

already_covered = true when an existing instruction ALREADY tells you to do \
this, however differently worded.

    new:      "Always tell me what came from the internet."
    existing: "Always let me know which facts are from the web."
    -> true. Same instruction, different words.

    new:      "Keep answers brief."
    existing: "Always prefer short answers."
    -> true.

already_covered = false when it ADDS something -- a new rule, a narrower or \
wider version of an existing one, or a REVERSAL.

    new:      "Never use tables."
    existing: "Always show me a table when comparing numbers."
    -> false. This reverses it. A reversal is never a duplicate.

    new:      "Cite the page number too."
    existing: "Always cite your sources."
    -> false. More specific, so it adds something.

    new:      "Always search the web."
    existing: "Always answer in short paragraphs."
    -> false. Unrelated.

When genuinely unsure, answer false. A duplicate that slips through is one \
redundant line; a real instruction discarded as a duplicate is silently \
ignoring what the user just asked for."""


async def already_covered(text: str, existing: list[str]) -> bool:
    """Does a stored instruction already say this?

    A DEDICATED CALL, rather than a clause in a bigger prompt.

    The router that used to own this judgement was doing four jobs in one pass
    -- classify the intent, extract the instruction, choose its scope, strip the
    question out -- and the dedupe clause at the end of that prompt was
    measurably the one that got dropped: it kept emitting paraphrases of
    instructions listed directly above it in the same prompt. One small model,
    one yes/no question, nothing else competing for attention, is what makes
    this reliable.

    It costs one call only on turns that actually store something, which is
    rare compared with asking.

    Fails OPEN -- on an LLM error the preference is stored. `remember` still
    applies its lexical guard underneath, and an extra row is a far smaller
    failure than dropping an instruction the user just gave.
    """
    if not existing:
        return False

    listed = "\n".join(f"- {t}" for t in existing)
    try:
        raw = await get_llm().generate(
            f"ALREADY STORED:\n{listed}\n\nNEW INSTRUCTION: {text}",
            schema=_DEDUPE_SCHEMA,
            system=_DEDUPE_SYSTEM,
            temperature=0.0,
            max_output_tokens=80,
        )
    except LLMError as exc:
        log.warning("dedupe_failed", error=str(exc))
        return False
    return extract_bool(raw, "already_covered")


async def remember(
    text: str,
    *,
    owner_id: str | None,
    session_id: uuid.UUID | None,
    scope: str = "user",
    source_message: str | None = None,
) -> Preference | None:
    """Store a preference. Returns it, or None if it duplicates an existing one.

    Deduplicated on MEANING, not exact text -- see `_is_near_duplicate`.
    """
    text = " ".join(text.split())[:MAX_LENGTH]
    if not text:
        return None

    target_session = session_id if scope == "session" else None

    async with SessionLocal() as db:
        # Compared against everything ALREADY IN FORCE HERE, not only rows of
        # the same scope.
        #
        # The asymmetry is deliberate. A user-level instruction already applies
        # to this conversation, so storing a session-scoped copy of it adds a
        # row that changes no behaviour whatsoever -- pure duplication, and the
        # commonest kind, because someone restating an instruction mid-chat
        # often phrases it as "in this chat, ...". The reverse is not true: a
        # session-scoped preference does NOT cover other conversations, so a
        # user-level instruction that resembles it is a genuine broadening and
        # must be stored.
        # `is_(None)` rather than `.in_([None, ...])`: in SQL, `x IN (NULL)` is
        # never true even for a NULL x, so the list form would have matched no
        # user-level rows at all and quietly disabled this whole check.
        scope_filter = Preference.session_id.is_(None)
        if target_session is not None:
            scope_filter = or_(scope_filter, Preference.session_id == target_session)
        rows = (
            (
                await db.execute(
                    select(Preference).where(
                        Preference.owner_id == owner_id,
                        scope_filter,
                        Preference.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        existing = next((p for p in rows if _is_near_duplicate(p.text, text)), None)
        if existing is not None:
            log.info(
                "preference_duplicate", new=text[:60], existing=existing.text[:60]
            )
            return None

        pref = Preference(
            owner_id=owner_id,
            session_id=target_session,
            text=text,
            source_message=source_message,
        )
        db.add(pref)
        await db.commit()
        await db.refresh(pref)

    log.info("preference_saved", scope=scope, text=text[:80])
    return pref


async def forget(preference_id: uuid.UUID, *, owner_id: str | None) -> bool:
    """Deactivate a preference.

    Soft delete: the profile view can show what was once in force, and an
    instruction the user cancels is itself worth keeping a record of. Scoped by
    owner, because the id arrives from a client.
    """
    async with SessionLocal() as db:
        stmt = select(Preference).where(Preference.id == preference_id)
        if owner_id is not None:
            stmt = stmt.where(Preference.owner_id == owner_id)
        pref = (await db.execute(stmt)).scalar_one_or_none()
        if pref is None:
            return False
        pref.active = False
        await db.commit()
    return True

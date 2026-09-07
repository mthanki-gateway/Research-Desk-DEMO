"""Deciding what conversation history to actually send the model.

Persistence and prompt context are separate concerns: every turn is stored
forever in `messages`, but only a small slice of it goes into any prompt.

HISTORY IS NOW SENT WHOLE. The compression machinery below is a fallback.

It was mandatory under Gemma: 16K tokens per MINUTE across 3-5 calls per turn
left roughly 1.5K for history, about 8-10 plain turns. Gemini Flash reports a
1,048,576-token window and 250K tokens/minute, so the binding constraint is
gone and `history_full` sends the entire transcript.

The summary path is kept because the budget is a real ceiling, but it is
strictly worse when avoidable -- a summary is a lossy rewrite, and pronoun
resolution is precisely what breaks when the referent was compressed away.

Layers, cheapest first:

  1. system prompt + retrieved chunks     (always, built elsewhere)
  2. rolling summary of evicted turns     (1 model call per eviction)
  3. verbatim last N exchanges            (always, uncompressed)
  4. semantically retrieved old turns     (NOT IMPLEMENTED -- see below)

Layer 4 is deliberately absent. It needs each exchange embedded into Qdrant
with its thread_id, and gating on a *relative* score margin (an irrelevant
match scored 0.570 against this corpus, so an absolute threshold cannot work).
Layers 2 and 3 cover ordinary conversations; layer 4 only matters when someone
returns to a topic from 30 turns ago. `retrieve_old_turns()` below marks the
seam.
"""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.db.models import ChatSession, Message, Role
from app.services.limiter import estimate_tokens
from app.services.llm import LLMError, get_llm

log = structlog.get_logger()

# How many recent messages survive compression, when compression happens at
# all. Configurable now rather than a constant, and much larger: 40 messages is
# twenty exchanges against the old three.
#
# Three was sized for Gemma leaving roughly 1.5K tokens for history. This
# window is what pronouns and follow-ups resolve against -- the one thing that
# must never be compressed -- so with the token floor gone there is no reason
# to keep it tight.
def _verbatim() -> int:
    return get_settings().verbatim_messages

# Only summarise once there is a worthwhile amount to fold in; summarising one
# stray message per turn would spend a Gemma call to save ~40 tokens.
SUMMARISE_THRESHOLD = 4

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What was discussed, 3-4 sentences, past tense.",
        },
        "established_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific facts already established, with figures.",
        },
    },
    "required": ["summary"],
}

SUMMARY_SYSTEM = """You maintain a running summary of a research conversation.

You receive the previous summary (if any) and the exchanges being folded into \
it. Produce an updated summary covering both.

Keep SPECIFICS. "Discussed revenue" is useless; "2024 revenue was $847.3m, up \
18.4%" is what makes the summary worth its tokens. Put such facts, with their \
figures, in established_facts.

Be compact. The summary must stay under 150 words however long the \
conversation gets."""


def _render(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        who = "User" if m.role == Role.user else "Assistant"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


async def update_summary(session: ChatSession, messages: list[Message]) -> tuple[str, int]:
    """Fold newly-evicted messages into the rolling summary.

    Returns (summary, summarised_upto). Incremental on purpose: it summarises
    *the previous summary plus the newly evicted turns*, never the whole
    history again, so cost stays flat as the conversation grows.
    """
    # Nothing to compress while the whole transcript fits the budget. This
    # saves a model call per eviction on every ordinary conversation -- the
    # summary only earns its cost once history stops fitting.
    if get_settings().history_full:
        return session.summary or "", session.summarised_upto or 0

    older = messages[:-_verbatim()] if len(messages) > _verbatim() else []
    already = session.summarised_upto or 0
    newly_evicted = older[already:]

    if len(newly_evicted) < SUMMARISE_THRESHOLD:
        return session.summary or "", already

    previous = session.summary or "(none)"
    prompt = (
        f"Previous summary:\n{previous}\n\n"
        f"New exchanges to fold in:\n{_render(newly_evicted)}\n\n"
        "Produce the updated summary."
    )
    try:
        result = await get_llm().generate_json(
            prompt,
            schema=SUMMARY_SCHEMA,
            system=SUMMARY_SYSTEM,
            temperature=0.1,
            max_output_tokens=500,
        )
    except LLMError as exc:
        # Keep the old summary rather than failing the user's turn. Worst case
        # the window just slides without compression.
        log.warning("summary_failed", error=str(exc))
        return session.summary or "", already

    summary = str(result.get("summary", "")).strip()
    facts = [f for f in result.get("established_facts", []) if isinstance(f, str)]
    if facts:
        summary += "\n\nEstablished facts:\n" + "\n".join(f"- {f}" for f in facts)

    upto = already + len(newly_evicted)
    log.info("summary_updated", messages_folded=len(newly_evicted), upto=upto)
    return summary, upto


def build_chat_context(session: ChatSession, messages: list[Message]) -> str:
    """Assemble the history block for a prompt. Empty string on turn one.

    Sends the WHOLE transcript when it fits the budget, which on Gemini Flash
    it almost always does -- a 200K-token budget against a 1M window is
    hundreds of turns.

    The summary path is kept as a fallback rather than deleted, because the
    budget is a real ceiling and a conversation can eventually exceed it. It is
    also strictly worse when avoidable: a summary is a lossy rewrite, and
    pronoun resolution is exactly what breaks when the referent was compressed
    away.
    """
    if not messages:
        return ""

    settings = get_settings()

    if settings.history_full:
        whole = _render(messages)
        if estimate_tokens(whole) <= settings.history_max_tokens:
            return f"Conversation so far:\n{whole}"
        log.info(
            "history_truncated",
            n_messages=len(messages),
            estimated_tokens=estimate_tokens(whole),
            budget=settings.history_max_tokens,
        )

    parts: list[str] = []
    if session.summary:
        parts.append(f"Earlier in this conversation:\n{session.summary}")

    recent = messages[-_verbatim():]
    if recent:
        parts.append(f"Recent exchanges:\n{_render(recent)}")

    return "\n\n".join(parts)


async def retrieve_old_turns(session_id, query: str) -> list[Message]:
    """Layer 4 seam -- intentionally a no-op.

    To implement: embed each exchange on completion into a `conversations`
    Qdrant collection keyed by session_id, retrieve 1-2 here, and keep them
    only if they beat the top document-chunk score. Never gate on an absolute
    score: measured on this corpus, an irrelevant match reached 0.570 while a
    correct one reached 0.615, so the ranges overlap.
    """
    return []

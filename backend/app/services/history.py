"""Deciding what conversation history to actually send the model.

Persistence and prompt context are separate concerns: every turn is stored
forever in `messages`, but only a small slice of it goes into any prompt.

The constraint is Gemma's **16K tokens per minute** -- throughput, not context
window. Since the agent makes 3-5 calls per question, each call needs to stay
around 3.5K tokens, which leaves roughly 1.5K for history. That's ~8-10 plain
turns, so anything longer has to be compressed.

Layers, cheapest first:

  1. system prompt + retrieved chunks     (always, built elsewhere)
  2. rolling summary of evicted turns     (1 Gemma call per eviction)
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

from app.db.models import ChatSession, Message, Role
from app.services.llm import LLMError, get_llm

log = structlog.get_logger()

# 3 exchanges = 6 messages. Three rather than two because a clarification
# often spans two turns ("do you mean 2023?" / "yes"), and this window is what
# pronouns and follow-ups resolve against -- the one thing that must never be
# compressed.
VERBATIM_MESSAGES = 6

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
    older = messages[:-VERBATIM_MESSAGES] if len(messages) > VERBATIM_MESSAGES else []
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
    """Assemble the history block for a prompt. Empty string on turn one."""
    if not messages:
        return ""

    parts: list[str] = []
    if session.summary:
        parts.append(f"Earlier in this conversation:\n{session.summary}")

    recent = messages[-VERBATIM_MESSAGES:]
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

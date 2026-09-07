"""Step 3: baseline RAG. Retrieve once, generate once. No agent.

Deliberately naive, and kept as a permanent point of comparison for the
LangGraph agent in step 4. Its known weaknesses are the reason step 4 exists:

  * one retrieval pass, so a question needing two lookups gets one
  * no self-check, so if the context doesn't support an answer the only defence
    is a prompt instruction -- and vector search always returns *something*
    (a question about quantum computing scored 0.570 against this corpus)
  * no query rewriting, so a poorly-phrased question retrieves poorly
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog

from app.config import get_settings
from app.services.llm import (
    extract_int_list,
    extract_string,
    get_llm,
    is_repetitive,
)
from app.services.retrieval import build_context, retrieve
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

SYSTEM = """You answer questions using ONLY the numbered sources provided.

Rules:
- Cite the source number in square brackets after each claim, e.g. [1] or [2].
- Quote figures exactly as they appear. Never round, adjust or infer a number.
- If the sources do not contain the answer, say exactly: "The provided \
documents do not contain this information." Do not guess, and do not use \
knowledge from outside the sources.
- Be concise. Two or three sentences is usually enough."""

PROMPT = """Sources:
{context}

Question: {question}

Answer using only the sources above, citing them as [1], [2], etc."""

# Gemma 4 has no "thinking" channel: asked for prose, it writes its entire
# reasoning trace -- constraint checklists, drafts, self-corrections -- straight
# into the reply, then runs out of tokens before finishing. A responseSchema
# fixes it: reasoning has nowhere to go except the declared string field, so
# what comes back is the answer alone.
#
# This is the same lesson as elsewhere in the app -- with Gemma, structure the
# output or get a monologue.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, 2-3 sentences, with [n] citations inline.",
        },
        "sources_used": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Source numbers actually cited in the answer.",
        },
    },
    "required": ["answer", "sources_used"],
}


@dataclass(slots=True)
class RagAnswer:
    question: str
    answer: str
    hits: list[SearchHit]
    sources_used: list[int]


async def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
) -> RagAnswer:
    hits = await retrieve(
        question,
        top_k=top_k,
        document_ids=document_ids,
        owner_id=owner_id,
        multi_query=multi_query,
    )

    if not hits:
        # Only happens with an empty collection -- see the module docstring:
        # a populated store always returns something, however irrelevant.
        return RagAnswer(
            question=question,
            answer="No documents have been indexed yet. Upload one first.",
            hits=[],
            sources_used=[],
        )

    # Lenient parsing, for the same reason the agent's `draft` node uses it: a
    # Gemma repetition loop runs until the token cap and truncates the JSON,
    # and a strict parse throws away an answer that was complete before the
    # loop started. See `strip_degeneration` in llm.py.
    raw = await get_llm(get_settings().answer_model).generate(
        PROMPT.format(context=build_context(hits), question=question),
        schema=ANSWER_SCHEMA,
        system=SYSTEM,
        temperature=0.1,
        max_output_tokens=900,
    )

    answer = extract_string(raw, "answer")
    sources_used = extract_int_list(raw, "sources_used")
    # `is_repetitive` as well as emptiness: cutting a trailing loop is not the
    # same as judging what survived. See the draft node.
    if not answer or is_repetitive(answer):
        answer = "The model did not return a usable answer. Try asking again."
        sources_used = []

    log.info(
        "rag_answered",
        question=question[:80],
        n_sources=len(hits),
        cited=sources_used,
    )
    return RagAnswer(
        question=question, answer=answer, hits=hits, sources_used=sources_used
    )

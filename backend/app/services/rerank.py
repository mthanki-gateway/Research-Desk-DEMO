"""Listwise LLM reranking, with relevance grading merged in.

WHY RERANK

Retrieval uses a BI-ENCODER: the question and each chunk are embedded
separately and compared by distance. That is what makes it fast -- the chunk
vectors are precomputed -- and it is also its limit, because the two texts never
meet. The score says "these two summaries of meaning are close", not "this
passage answers that question".

A reranker sees them TOGETHER. Far more accurate, far too slow to run over a
corpus. So: retrieve broadly and cheaply, then rerank precisely.

WHY LISTWISE

    pointwise   one call per candidate    20 calls   model cannot compare;
                                                     eight chunks all score 7
    listwise    one call, all candidates   1 call    ranking is comparative
    pairwise    every pair               ~190 calls  absurd

WHY GRADING IS THE SAME CALL

Grading is the binary question "is this relevant at all", and its job is to
make "I found nothing useful" a possible outcome. A listwise reranker told
"return fewer if fewer are genuinely useful" is already deciding exactly that.
Keeping a separate grading node would pay twice for one judgement. A
cross-encoder could not have offered this -- it scores, it does not decide.
"""

from __future__ import annotations

import random

import structlog

from app.config import get_settings
from app.services import progress, tracing
from app.services.llm import LLMError, extract_int_list, get_llm
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Passage numbers, most useful first. Fewer is fine.",
        },
    },
    "required": ["ranked_ids"],
}

RERANK_SYSTEM = """You rank passages by how well they help ANSWER a question.

Rules:
- Judge only whether the passage helps answer the question asked. Do NOT reward \
a passage for being on the same topic.
- Return the passage numbers, most useful first.
- Return FEWER than asked if fewer are genuinely useful. Returning padding is \
worse than returning a short list -- a passage that does not help is noise that \
makes the answer vaguer.
- If NONE of the passages help, return an empty list.
- Never invent a number that was not shown to you."""


def _excerpt(hit: SearchHit, limit: int) -> str:
    where = hit.url or hit.filename
    if hit.source == "document" and hit.heading:
        where = f"{hit.filename} > {hit.heading.lstrip('# ').strip()}"
    return f"{where}\n{hit.text[:limit]}"


async def rerank_hits(
    query: str,
    hits: list[SearchHit],
    *,
    top_k: int,
) -> list[SearchHit]:
    """Reorder `hits` by usefulness, keeping at most `top_k`.

    Never raises and never returns more than it was given. On any failure it
    returns the first `top_k` of the original order, which is the behaviour the
    app had before reranking existed -- a reranker that breaks retrieval is
    worse than no reranker.
    """
    settings = get_settings()
    if len(hits) <= 1:
        return hits[:top_k]

    # SHUFFLED before presenting. Models over-favour whatever sits at the start
    # and end of a list, so showing them in retrieval order invites the model to
    # rubber-stamp the ranking it was given -- which would make the whole step
    # an expensive no-op that still looks like it is working.
    order = list(range(len(hits)))
    random.shuffle(order)

    passages = "\n\n".join(
        f"[{n}] {_excerpt(hits[idx], settings.rerank_excerpt_chars)}"
        for n, idx in enumerate(order, start=1)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Passages:\n{passages}\n\n"
        f"Return at most {top_k} passage numbers, most useful first."
    )

    with tracing.observe(
        "rerank",
        as_type="generation",
        input=query,
        metadata={"n_candidates": len(hits), "top_k": top_k},
    ) as span:
        progress.emit("rerank", n=len(hits))
        try:
            raw = await get_llm().generate(
                prompt,
                schema=RERANK_SCHEMA,
                system=RERANK_SYSTEM,
                temperature=0.0,
                # Only a list of integers comes back, so this is generous.
                max_output_tokens=200,
            )
        except LLMError as exc:
            log.warning("rerank_failed", error=str(exc))
            tracing.update(span, level="ERROR", status_message=str(exc)[:200])
            return hits[:top_k]

        picked = extract_int_list(raw, "ranked_ids")

        # VALIDATED against what was actually sent. Models hallucinate ids, and
        # an out-of-range number would either crash the lookup or -- worse --
        # silently resolve to the wrong passage. Duplicates are dropped for the
        # same reason: one passage occupying two slots is a fabricated second
        # source.
        seen: set[int] = set()
        chosen: list[SearchHit] = []
        invalid = 0
        for n in picked:
            if not (1 <= n <= len(order)):
                invalid += 1
                continue
            if n in seen:
                continue
            seen.add(n)
            chosen.append(hits[order[n - 1]])

        dropped = len(hits) - len(chosen)
        tracing.update(
            span,
            output={"kept": len(chosen), "dropped": dropped, "invalid_ids": invalid},
            metadata={"invalid_id_rate": invalid / max(1, len(picked))},
        )

        if invalid:
            # Worth its own line: a rising rate means model or prompt drift, and
            # it is invisible in the results -- the answer just gets slightly
            # worse.
            log.warning("rerank_invalid_ids", n=invalid, of=len(picked))

        if not chosen:
            # Two very different situations, one outcome, so log which.
            #
            # An EMPTY list is a real verdict -- the grading half saying nothing
            # here is useful -- and honouring it is what lets the agent abstain.
            # But a parse failure looks identical, and treating that as "nothing
            # is relevant" would silently discard good retrieval.
            if picked == []:
                log.info("rerank_graded_all_out", query=query[:60], n=len(hits))
                return []
            log.warning("rerank_unusable", raw=raw[:160])
            return hits[:top_k]

        log.info(
            "reranked",
            query=query[:60],
            candidates=len(hits),
            kept=len(chosen),
            dropped=dropped,
        )
        return chosen[:top_k]

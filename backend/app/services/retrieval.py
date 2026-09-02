"""Retrieval, query expansion, rank fusion, and prompt-context assembly.

Kept separate from both the vector store and the chat endpoint so that adding
hybrid search (BM25 fused with vectors) later means changing only this file --
callers keep getting a ranked list of SearchHit.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog

from app.config import get_settings
from app.services.embeddings import get_embeddings
from app.services.llm import LLMError, extract_string_list, get_llm
from app.services.vectorstore import SearchHit, get_vector_store

log = structlog.get_logger()


# --------------------------------------------------------------------------
# query expansion
# --------------------------------------------------------------------------

VARIATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "variations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Alternative phrasings of the question.",
        }
    },
    "required": ["variations"],
}

VARIATIONS_SYSTEM = """You rewrite a search question into alternative phrasings \
for a document search engine.

The search is semantic, not keyword-based, and longer more specific phrasings \
retrieve better than short ones. So each variation should:
- use different vocabulary and synonyms than the original
- be a complete, specific question or statement, not keywords
- approach the information need from a different angle
- stay faithful to what was actually asked

If the question has several parts, write one variation per part."""


async def expand_query(question: str, n: int | None = None) -> list[str]:
    """Return `n` alternative phrasings. Falls back to [] on any failure.

    Query expansion is an optimisation, never load-bearing: if Gemma is out of
    quota we still want the original query to run.
    """
    settings = get_settings()
    count = n or settings.query_variations

    prompt = (
        f"Write {count} alternative phrasings of this question. "
        f"Each must be one sentence, under 25 words.\n\nQuestion: {question}"
    )
    try:
        raw = await get_llm().generate(
            prompt,
            schema=VARIATIONS_SCHEMA,
            system=VARIATIONS_SYSTEM,
            # 0.7 sent Gemma into a repetition loop that truncated the JSON.
            # Some diversity is wanted here, but 0.35 is where it stays coherent.
            temperature=0.35,
            top_p=0.9,
            max_output_tokens=500,
        )
    except LLMError as exc:
        log.warning("query_expansion_failed", error=str(exc))
        return []

    candidates = extract_string_list(raw, "variations")
    original = question.strip().lower()
    variations = [v for v in candidates if v.lower() != original]

    log.info(
        "query_expanded", original=question[:60], n=len(variations), variations=variations
    )
    return variations[:count]


# --------------------------------------------------------------------------
# rank fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: dict[str, list[SearchHit]], *, k: int | None = None
) -> list[SearchHit]:
    """Fuse several ranked lists into one, using rank position only.

        score(chunk) = Σ over lists of 1 / (k + rank)

    Position rather than score is deliberate. Cosine scores are not comparable
    across queries -- we measured an irrelevant match at 0.570 and a correct one
    at 0.615 -- and a BM25 score would be on a different scale again. Rank is
    the only currency the lists share, which is also what makes RRF need no
    tuning or normalisation.

    A chunk that several variations agree on accumulates contributions and
    rises, which is exactly the signal we want.
    """
    damping = k if k is not None else get_settings().rrf_k

    scores: dict[uuid.UUID, float] = {}
    best: dict[uuid.UUID, SearchHit] = {}
    found_by: dict[uuid.UUID, list[str]] = {}

    for label, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (damping + rank)
            found_by.setdefault(hit.chunk_id, []).append(label)
            # Keep the copy with the strongest cosine score, so the reported
            # similarity is the best evidence we saw for that chunk.
            if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                best[hit.chunk_id] = hit

    fused: list[SearchHit] = []
    for chunk_id, score in scores.items():
        hit = best[chunk_id]
        hit.rrf_score = score
        hit.found_by = found_by[chunk_id]
        fused.append(hit)

    fused.sort(key=lambda h: h.rrf_score or 0.0, reverse=True)
    return fused


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


async def _search_one(
    query: str,
    *,
    limit: int,
    document_ids: list[uuid.UUID] | None,
    owner_id: str | None,
) -> list[SearchHit]:
    vector = await get_embeddings().embed_query(query)
    return await get_vector_store().search(
        vector, limit=limit, document_ids=document_ids, owner_id=owner_id
    )


async def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
) -> list[SearchHit]:
    """Retrieve up to `top_k` chunks, optionally via multi-query + RRF."""
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    use_multi = settings.multi_query if multi_query is None else multi_query

    if not use_multi:
        hits = await _search_one(
            query, limit=k, document_ids=document_ids, owner_id=owner_id
        )
        log.info(
            "retrieved",
            mode="single",
            query=query[:60],
            n=len(hits),
            top_score=round(hits[0].score, 4) if hits else None,
        )
        return hits

    variations = await expand_query(query)
    queries = [query, *variations]

    # Each variation retrieves its own top k. The candidate pool is therefore
    # up to len(queries)*k, and fusion picks the best k of those -- so the win
    # is in WHICH k you end up with, not in how many.
    results = await asyncio.gather(
        *(
            _search_one(q, limit=k, document_ids=document_ids, owner_id=owner_id)
            for q in queries
        )
    )

    rankings = {q: hits for q, hits in zip(queries, results, strict=True)}
    fused = reciprocal_rank_fusion(rankings)[:k]

    log.info(
        "retrieved",
        mode="multi",
        query=query[:60],
        n_queries=len(queries),
        candidates=sum(len(r) for r in results),
        returned=len(fused),
    )
    return fused


def build_context(hits: list[SearchHit]) -> str:
    """Render hits as numbered sources the model can cite as [1], [2].

    Ordering note: hits arrive best-first and are kept that way. Models attend
    most strongly to the start and end of a long context ("lost in the
    middle"), so with a small top_k the strongest evidence leading is the
    simplest good default.
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        where = hit.filename
        if hit.heading:
            where += f" › {hit.heading.lstrip('# ').strip()}"
        if hit.page is not None:
            where += f" › p.{hit.page}"
        blocks.append(f"[{i}] {where}\n{hit.text}")
    return "\n\n".join(blocks)

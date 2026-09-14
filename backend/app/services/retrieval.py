"""Retrieval, query expansion, rank fusion, and prompt-context assembly.

Kept separate from both the vector store and the chat endpoint so that adding
hybrid search (BM25 fused with vectors) later means changing only this file --
callers keep getting a ranked list of SearchHit.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.services import progress, tracing
from app.services.embeddings import get_embeddings
from app.services.lexical import lexical_search
from app.services.llm import LLMError, extract_string_list, get_llm
from app.services.parents import expand_to_parents
from app.services.rerank import rerank_hits
from app.services.vectorstore import SearchHit, get_vector_store

log = structlog.get_logger()


# --------------------------------------------------------------------------
# query expansion
# --------------------------------------------------------------------------

VARIATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        # A constrained enum, NOT free text. The plan node learned this the hard
        # way: an open-ended `reasoning` field sent Gemma into degeneration and
        # truncated the whole object, discarding good output. A two-value enum
        # costs a handful of tokens and cannot derail.
        "scope": {
            "type": "string",
            "enum": ["specific", "broad"],
            "description": "specific = particular facts; broad = the document overall.",
        },
        "variations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Search queries to run.",
        },
    },
    "required": ["scope", "variations"],
}

VARIATIONS_SYSTEM = """You turn a request into queries for a semantic document \
search engine.

The search matches meaning, not keywords, and returns only a handful of \
passages per query. Longer, more specific phrasings retrieve better than short \
ones.

First decide which kind of request this is, and report it as `scope`.

SPECIFIC -- asks for particular facts. "What was operating income?", "Who are \
the largest customers?", "How many employees?"
  Write variations that approach the SAME information need from different \
angles: different vocabulary, synonyms, related phrasing. Stay faithful to \
what was asked.

BROAD -- asks about the document as a whole. "Summarize this", "what does this \
cover?", "key takeaways", "what should I know about this company?"
  Do NOT paraphrase the request. A phrase like "summarize the document" \
describes an ACTION; it shares no meaning with the document's contents, so \
searching for it returns arbitrary passages. Instead write one query per major \
TOPIC the document covers, so that together they retrieve passages spanning the \
whole document instead of clustering in one section. Each query should read \
like a statement about that topic as it would actually appear in the text.

If a section outline is given, use it: cover the most substantive sections and \
skip boilerplate such as tables of contents, legal notices and page headers. \
With no outline, infer the likely topics from the request itself.

If the request has several distinct parts, write one query per part.

Every query must be one sentence, specific, and never a keyword list. Never \
invent facts or figures -- describe what you are looking for, not what you \
expect to find."""


@dataclass(frozen=True, slots=True)
class Expansion:
    """Result of query expansion.

    `broad` matters to the caller, not just to logging: for a broad request the
    ORIGINAL question must be dropped from the fused query set. See `retrieve`.
    """

    variations: list[str]
    broad: bool

    @classmethod
    def empty(cls) -> Expansion:
        return cls(variations=[], broad=False)


# Cap on headings sent to the rewriter. Each is only a few tokens, but a long
# document can carry hundreds of them and the budget here is 16K tokens/minute
# for the whole app. 40 is enough to convey what a document covers.
_OUTLINE_LIMIT = 40


async def document_outline(
    document_ids: list[uuid.UUID] | None,
    owner_id: str | None,
    *,
    limit: int = _OUTLINE_LIMIT,
) -> list[str]:
    """Distinct section headings for the documents in scope.

    This is what turns intent-aware rewriting from guesswork into something
    grounded. Told only "summarize the document", the model has to invent
    plausible topics -- fine for a financial report it can guess at, useless for
    an arbitrary document. Given the actual headings, it writes queries against
    sections that really exist.

    Cheap on purpose: one indexed SQL query, no embedding and no LLM call, using
    the heading metadata the chunker already stores. It is also best-effort --
    any failure returns [] and the rewriter falls back to inferring topics,
    because a broken outline must never fail a search.
    """
    try:
        async with SessionLocal() as session:
            stmt = (
                select(Chunk.heading)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.heading.isnot(None))
                .order_by(Chunk.document_id, Chunk.chunk_index)
                .limit(limit * 4)  # room to dedupe below
            )
            if document_ids:
                stmt = stmt.where(Chunk.document_id.in_(document_ids))
            if owner_id is not None:
                stmt = stmt.where(Document.owner_id == owner_id)

            rows = (await session.execute(stmt)).scalars().all()
    except Exception:
        log.warning("outline_lookup_failed", exc_info=True)
        return []

    # Order-preserving dedupe: headings repeat across the chunks of one
    # section, and document order is more useful to the model than any sort.
    seen: set[str] = set()
    out: list[str] = []
    for raw in rows:
        heading = (raw or "").lstrip("# ").strip()
        key = heading.lower()
        if heading and key not in seen:
            seen.add(key)
            out.append(heading)
        if len(out) >= limit:
            break
    return out


def _scope_is_broad(raw: str) -> bool:
    """Lenient read of the `scope` field.

    Deliberately tolerant of a truncated or slightly malformed object, for the
    same reason the plan node salvages its list: losing a good set of queries to
    a strict parse failure is worse than accepting a near-miss. Unrecognised or
    missing means `specific`, which is the behaviour this code had before scope
    existed -- so a parse failure degrades to the old path rather than a new one.
    """
    try:
        value = json.loads(raw)
        if isinstance(value, dict) and isinstance(value.get("scope"), str):
            return value["scope"].strip().lower() == "broad"
    except (json.JSONDecodeError, TypeError):
        pass
    return bool(re.search(r'"scope"\s*:\s*"\s*broad\s*"', raw, re.IGNORECASE))


async def expand_query(
    question: str,
    n: int | None = None,
    *,
    outline: list[str] | None = None,
) -> Expansion:
    """Rewrite `question` into `n` search queries. Never raises.

    Query expansion is an optimisation, never load-bearing: if Gemma is out of
    quota we still want the original query to run.
    """
    settings = get_settings()
    count = n or settings.query_variations

    outline_block = ""
    if outline:
        listed = "\n".join(f"- {h}" for h in outline)
        outline_block = (
            f"Sections present in the documents being searched:\n{listed}\n\n"
        )

    prompt = (
        f"{outline_block}"
        f"Write {count} search queries for this request, and report its scope. "
        f"Each query must be one sentence, under 25 words.\n\n"
        f"Request: {question}"
    )
    try:
        # The REWRITER model, not the workhorse. Query rewriting is the one
        # call where throughput beats quality: it fires N times per turn, emits
        # short phrases rather than prose, and nothing a user reads comes from
        # it -- so it stays on the model with the largest request budget.
        raw = await get_llm(settings.rewriter_model).generate(
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
        return Expansion.empty()

    candidates = extract_string_list(raw, "variations")
    original = question.strip().lower()
    variations = [v for v in candidates if v.lower() != original][:count]
    broad = _scope_is_broad(raw)

    log.info(
        "query_expanded",
        original=question[:60],
        scope="broad" if broad else "specific",
        n=len(variations),
        outline_used=len(outline or []),
        variations=variations,
    )
    return Expansion(variations=variations, broad=broad)


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


def exclude_seen(hits: list[SearchHit], seen: set[str] | None) -> list[SearchHit]:
    """Drop chunks already served this turn.

    Applied BEFORE the floor and the fusion, not after. Filtering at the end
    would let already-seen chunks occupy the candidate pool and the reranker's
    budget, so a retry would fetch 20, discard 15 it had already shown, and
    rerank the 5 that were left -- which is how a "different angle" ends up
    being the same angle with fewer options.
    """
    if not seen:
        return hits
    return [h for h in hits if str(h.chunk_id) not in seen]


def apply_floor(hits: list[SearchHit], floor: float) -> list[SearchHit]:
    """Drop hits below an absolute similarity floor.

    THE POINT IS TO MAKE "NOTHING RELEVANT" REACHABLE. RRF reads rank and
    discards magnitude, so it always produces a confident top-k: every
    candidate could be a terrible match and the output would look exactly like
    a perfect run. Without an absolute check somewhere, "the corpus cannot
    answer this" is not an outcome the pipeline can ever produce.

    Applied to the DENSE scores only, and deliberately before fusion. A BM25
    score has no absolute meaning to compare against -- it is unbounded and
    corpus-dependent -- and an RRF score is a rank artefact, so neither can
    carry a threshold.
    """
    if floor <= 0.0:
        return hits
    kept = [h for h in hits if h.score >= floor]
    if len(kept) != len(hits):
        log.info("floor_rejected", floor=floor, dropped=len(hits) - len(kept))
    return kept


async def _lexical_hits(
    query: str,
    *,
    limit: int,
    document_ids: list[uuid.UUID] | None,
    owner_id: str | None,
    by_id: dict[uuid.UUID, SearchHit],
) -> list[SearchHit]:
    """BM25 results, expressed as SearchHits so RRF can fuse them.

    Only chunks the DENSE side already returned are usable here: a SearchHit
    carries text, heading and filename, and BM25 returns an id and a score.
    Hydrating the rest would mean a second SQL round-trip per query.

    That is a real limitation and worth naming: BM25 can currently only
    RE-RANK the dense candidate pool, not introduce a chunk dense retrieval
    missed entirely -- which is precisely the case it exists for
    ("INC-2024-1183"). Widening the dense pool (`hybrid_candidates`) is what
    makes that overlap likely enough to be useful for now; hydrating from
    Postgres is the honest fix and is a small change to make later.
    """
    lexical = await lexical_search(
        query, limit=limit, owner_id=owner_id, document_ids=document_ids
    )
    out: list[SearchHit] = []
    for hit in lexical:
        found = by_id.get(hit.chunk_id)
        if found is not None:
            out.append(found)
    return out


async def _finalise(
    query: str, hits: list[SearchHit], *, k: int, narrows: bool
) -> list[SearchHit]:
    """The last stage shared by both retrieval paths: rerank, then cut to k.

    `narrows` records whether anything upstream deliberately over-fetched. It
    exists so the slice to k happens exactly once, in one place -- the earlier
    shape sliced in two branches and made it easy to add a third that forgot.
    """
    settings = get_settings()
    if settings.rerank and len(hits) > 1:
        hits = await rerank_hits(query, hits[: settings.rerank_candidates], top_k=k)
    elif narrows or len(hits) > k:
        hits = hits[:k]

    # LAST, deliberately, and after the cut to k.
    #
    # Expanding before the reranker would hand it whole sections to judge, which
    # is both far more tokens and a worse judgement -- relevance is a property
    # of the passage that matched, and a long section dilutes it exactly as it
    # dilutes an embedding. Expanding before the slice would waste assembly on
    # chunks about to be discarded.
    #
    # Dedupe on parent can return FEWER than k, which is correct: three children
    # of one section are one source, and padding the list back to k would be
    # inventing corroboration.
    return await expand_to_parents(hits)


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
    # Traced as a RETRIEVER, which is the granularity that makes a trace
    # readable: one observation per query carrying what was asked and what came
    # back. Instrumenting embed_query and search separately would be more
    # faithful to the call stack and much worse to read -- with multi-query on,
    # a turn would show eight sibling spans with no indication of which query
    # produced which hits.
    with tracing.observe(
        "retrieve",
        as_type="retriever",
        input=query,
        metadata={"limit": limit, "scoped": bool(document_ids)},
    ) as span:
        progress.searching("documents", query)
        vector = await get_embeddings().embed_query(query)
        hits = await get_vector_store().search(
            vector, limit=limit, document_ids=document_ids, owner_id=owner_id
        )
        progress.searched("documents", query, len(hits))
        tracing.update(
            span,
            # Chunk ids and scores, NOT the chunk text. Two reasons: the text
            # is already on the draft generation's prompt, so this would double
            # every trace's size; and it keeps document contents out of the
            # trace store, which is a third-party PII surface.
            output=[
                {
                    "chunk_id": str(h.chunk_id),
                    "file": h.filename,
                    "heading": h.heading,
                    "score": round(h.score, 4),
                }
                for h in hits
            ],
            metadata={
                "n_hits": len(hits),
                # A sharp drop from the top score is the cheap signal that the
                # corpus cannot serve this query -- worth having as a
                # filterable dimension in production.
                "top_score": round(hits[0].score, 4) if hits else None,
            },
        )
        return hits


async def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    owner_id: str | None = None,
    multi_query: bool | None = None,
    exclude_chunk_ids: set[str] | None = None,
) -> list[SearchHit]:
    """Retrieve up to `top_k` chunks, optionally via multi-query + RRF.

    The full pipeline, when everything is switched on:

        query -> [expand] -> dense  -+-> floor -> RRF -> rerank -> top_k
                              BM25  -+

    Every stage after the dense search is opt-in and degrades to the stage
    before it, so the default path is exactly the behaviour this function had
    before any of them existed.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    use_multi = settings.multi_query if multi_query is None else multi_query

    # Fetch deeper than k when a later stage will narrow it again. A reranker
    # can only reorder what it is handed, so retrieving exactly k and then
    # "reranking" is an expensive no-op -- and RRF is worse, since a chunk one
    # position past a list's end contributes nothing at all.
    narrows = settings.rerank or (settings.hybrid_search and not use_multi)
    fetch = max(k, settings.rerank_candidates if settings.rerank else k)
    if settings.hybrid_search:
        fetch = max(fetch, settings.hybrid_candidates)

    # INTENT IS CLASSIFIED EVEN WITH MULTI-QUERY OFF.
    #
    # The gap this closes: `scope` (specific vs broad) used to be produced only
    # on the multi-query path, so with the toggle off a request like
    # "summarize this document" went straight to a single dense search for the
    # literal phrase -- and "summarize" describes an ACTION that shares no
    # meaning with the document's contents, so it retrieves arbitrary passages.
    # The exact failure the classifier exists to prevent was reachable by
    # turning off an unrelated switch.
    #
    # Scope and the rewrites come from the SAME model call, so intent cannot be
    # had without paying for the rewrite. That is why this is one call either
    # way, and why the toggle now controls FAN-OUT rather than whether the
    # query is understood at all.
    #
    # A broad request then uses the topic queries regardless of the toggle: a
    # single search structurally cannot serve "tell me about the whole
    # document", so honouring `multi_query=False` there would mean honouring a
    # setting into a known-bad result.
    classify = use_multi or settings.intent_always
    expansion = Expansion.empty()
    if classify:
        # The outline is fetched before rewriting so the rewriter can target
        # real sections. One SQL query, and [] on any failure.
        outline = await document_outline(document_ids, owner_id)
        expansion = await expand_query(query, outline=outline)

    fan_out = use_multi or (expansion.broad and bool(expansion.variations))

    if not fan_out:
        dense = await _search_one(
            query, limit=fetch, document_ids=document_ids, owner_id=owner_id
        )
        dense = apply_floor(
            exclude_seen(dense, exclude_chunk_ids), settings.retrieval_score_floor
        )

        hits = dense
        if settings.hybrid_search and dense:
            by_id = {h.chunk_id: h for h in dense}
            sparse = await _lexical_hits(
                query,
                limit=settings.hybrid_candidates,
                document_ids=document_ids,
                owner_id=owner_id,
                by_id=by_id,
            )
            # Fused even when sparse is empty, so `rrf_score` and `found_by`
            # are populated consistently on both paths -- the UI reads them.
            hits = reciprocal_rank_fusion({"dense": dense, "bm25": sparse})

        hits = await _finalise(query, hits, k=k, narrows=narrows)
        log.info(
            "retrieved",
            mode="hybrid" if settings.hybrid_search else "single",
            scope="broad" if expansion.broad else "specific",
            classified=classify,
            query=query[:60],
            n=len(hits),
            top_score=round(hits[0].score, 4) if hits else None,
        )
        return hits

    # For a BROAD request the original question is dropped from the query set.
    # This is not a tidy-up -- keeping it actively poisons the results. RRF
    # gives every list an equal vote, so a query like "summarize the document"
    # contributes a full ranking of passages chosen for their similarity to the
    # *word* "summarize", and those arbitrary chunks then compete with the
    # topic-targeted ones on equal footing. Removing the one query known to
    # carry no signal is the whole point of classifying scope.
    #
    # If rewriting produced nothing (quota, parse failure), the original is all
    # there is, so it stays -- a mediocre search beats no search.
    if expansion.broad and expansion.variations:
        queries = list(expansion.variations)
    else:
        queries = [query, *expansion.variations]

    # Each variation retrieves its own top k. The candidate pool is therefore
    # up to len(queries)*k, and fusion picks the best k of those -- so the win
    # is in WHICH k you end up with, not in how many.
    #
    # NOTE for when the token budget allows it: a broad request deserves a
    # larger k per query than a specific one, since the whole aim is coverage
    # rather than precision. It is held at k here because Gemma's 16K
    # tokens/minute is the binding constraint -- the draft step has to fit every
    # retrieved passage into one prompt.
    results = await asyncio.gather(
        *(
            _search_one(q, limit=fetch, document_ids=document_ids, owner_id=owner_id)
            for q in queries
        )
    )
    results = [
        apply_floor(exclude_seen(r, exclude_chunk_ids), settings.retrieval_score_floor)
        for r in results
    ]

    rankings: dict[str, list[SearchHit]] = {
        q: hits for q, hits in zip(queries, results, strict=True)
    }

    # BM25 joins the fusion as ONE MORE LIST, not as a separate stage. That is
    # the property that makes RRF the right fuser here: adding a retriever is
    # adding a list, with no weights to tune and no scores to normalise.
    if settings.hybrid_search:
        by_id = {h.chunk_id: h for r in results for h in r}
        for q in queries:
            sparse = await _lexical_hits(
                q,
                limit=settings.hybrid_candidates,
                document_ids=document_ids,
                owner_id=owner_id,
                by_id=by_id,
            )
            if sparse:
                rankings[f"bm25:{q}"] = sparse

    fused = reciprocal_rank_fusion(rankings)
    fused = await _finalise(query, fused, k=k, narrows=narrows)

    log.info(
        "retrieved",
        mode="multi",
        scope="broad" if expansion.broad else "specific",
        query=query[:60],
        n_queries=len(queries),
        original_dropped=expansion.broad and bool(expansion.variations),
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
    return "\n\n".join(context_blocks(hits))


def context_blocks(hits: list[SearchHit]) -> list[str]:
    """The same rendered passages `build_context` joins, as a list.

    Exists so the evaluation harness can record EXACTLY what the model was
    shown. It previously stored bare `hit.text`, which omitted the
    `[n] document · filename › heading` label -- and since the drafter is
    required to name its source in the sentence, RAGAS then decomposed
    "According to acme-report.md, ..." into a claim about a filename that
    appeared nowhere in the context it was given.

    Measured: faithfulness 0.0 on an answer whose substantive claim was fully
    supported. Nothing was wrong with the answer or the retrieval; the judge was
    handed a different context than the model.
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        # EVERY source carries its kind, not just the web ones.
        #
        # It used to label only `web`, leaving an unlabelled source to mean
        # "document" by omission. That was fine while documents were the whole
        # universe and the web was the exception. They are peers now, so the
        # asymmetry actively misleads: the drafter is asked to tell the reader
        # where each fact came from, and it cannot do that reliably from a
        # marking that is present on one kind and absent on the other.
        if hit.source == "web":
            where = f"web · {hit.filename}"
            if hit.url:
                where += f" · {hit.url}"
        else:
            where = f"document · {hit.filename}"
            if hit.heading:
                where += f" › {hit.heading.lstrip('# ').strip()}"
            if hit.page is not None:
                where += f" › p.{hit.page}"
        blocks.append(f"[{i}] {where}\n{hit.text}")
    return blocks

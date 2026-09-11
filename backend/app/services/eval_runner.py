"""Tier 1 evaluation runner: retrieval quality, measured.

Tier 1 is the cheap half of evaluation. It makes **one embedding call per
question** and no LLM calls at all (unless multi-query is on), which is what
makes it runnable on demand from the Lab UI rather than as an overnight job.

The efficiency that makes that true: retrieval runs **once** per question at
`max(k_values)`, and every k is computed by slicing that single ranked list.
Measuring recall@1, @3, @5 and @10 therefore costs exactly the same as
measuring recall@10 alone.

Tier 2 (faithfulness, answer relevancy — the LLM-judged metrics) is separate,
because a full agent run per question is ~4 model calls against a 16K
tokens/minute budget and cannot be a web request.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

import structlog
from sqlalchemy import select

from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.services.evaluation import Aggregate, QuestionScores
from app.services.golden import ChunkSpec, GoldenQuestion, MatchResult, match
from app.services.retrieval import retrieve

log = structlog.get_logger()

# Defaults chosen to expose the retrieval-vs-ranking diagnosis: if recall@10 is
# high while recall@3 is low, retrieval is finding the right chunks and RANKING
# is burying them -- which is the signal that a reranker would help. If both
# are low, retrieval itself is failing and a reranker would change nothing.
DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(slots=True)
class HitSummary:
    """One retrieved chunk, trimmed for display.

    `preview` is truncated because a report of 20 questions x 10 hits would
    otherwise carry the better part of the corpus over the wire. `chunk_id` is
    included so the UI can fetch the FULL text on demand through the existing
    chunk panel — the same drill-down citations already use.
    """

    rank: int
    chunk_id: str
    filename: str
    heading: str | None
    score: float
    relevant: bool
    preview: str
    n_chars: int


@dataclass(slots=True)
class ExpectedFact:
    """A labelled fact the question needs, and whether retrieval found it.

    Sent to the UI so a failure can say *what* was missed rather than just
    "fact 1 not found". Without the label, a red row is unactionable — you
    cannot tell whether retrieval missed something obvious or something
    genuinely obscure.
    """

    index: int
    file: str
    must_contain: list[str]
    # None means no retrieved chunk satisfied it at any depth.
    found_at_rank: int | None
    # Chunks in the corpus that DO satisfy this label, resolved by direct SQL
    # rather than by retrieval. Two jobs:
    #
    #   1. The UI can open the chunk retrieval *should* have returned, which is
    #      the only way to judge whether a miss was reasonable.
    #   2. It validates the label. An empty list means NO chunk in the corpus
    #      matches, so the question can never pass -- either the fixture is not
    #      ingested or `must_contain` has a typo. That failure is otherwise
    #      indistinguishable from a retrieval failure, and would send you
    #      optimising retrieval against an impossible target.
    matching_chunk_ids: list[str]

    @property
    def label_resolves(self) -> bool:
        return bool(self.matching_chunk_ids)


@dataclass(slots=True)
class QuestionResult:
    id: str
    question: str
    tags: list[str]
    answerable: bool
    total_specs: int
    specs_satisfied: int
    # Spec index -> rank at which it was first found. A spec satisfied at rank
    # 9 with top_k=5 is a RANKING failure, not a retrieval one, and the UI
    # should say so.
    satisfied_at: dict[int, int]
    unsatisfied_specs: list[int]
    hits: list[HitSummary]
    # What the question was labelled as needing, and where each was found.
    expected: list[ExpectedFact]
    # The reference answer from the golden set, so the UI can show what a
    # correct response looks like next to what retrieval supplied.
    expected_answer: str
    expect_refusal: bool
    # Per-k scores, keyed by k as a string so this serialises to JSON cleanly.
    scores: dict[str, dict] = field(default_factory=dict)


@dataclass(slots=True)
class Tier1Report:
    config: dict
    questions: list[QuestionResult]
    aggregates: dict[str, dict]
    elapsed_seconds: float
    n_embedding_calls: int

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "n_embedding_calls": self.n_embedding_calls,
            "aggregates": self.aggregates,
            "questions": [asdict(q) for q in self.questions],
        }


async def _resolve_spec(
    spec: ChunkSpec, owner_id: str | None, *, limit: int = 5
) -> list[str]:
    """Find chunks in the corpus that satisfy a golden label.

    Direct SQL, not retrieval -- the point is to know what SHOULD have come
    back, independently of whether search found it. `ILIKE` mirrors the
    case-insensitive substring test in `golden.ChunkSpec.matches`, so the two
    cannot disagree about what counts as a match.
    """
    async with SessionLocal() as db:
        stmt = (
            select(Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.filename == spec.file)
            .order_by(Chunk.chunk_index)
            .limit(limit)
        )
        if owner_id is not None:
            stmt = stmt.where(Document.owner_id == owner_id)
        for needle in spec.must_contain:
            # Escape LIKE wildcards so a label containing % or _ still means
            # itself. Percent signs are common in this corpus ("62.1%").
            escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            stmt = stmt.where(Chunk.text.ilike(f"%{escaped}%", escape="\\"))
        rows = (await db.execute(stmt)).scalars().all()
    return [str(r) for r in rows]


def _score_at(
    question: GoldenQuestion, m: MatchResult, k: int
) -> QuestionScores:
    """Build per-k scores from the two relevance vectors.

    `QuestionScores.compute` takes one vector, but the metrics do not all want
    the same one -- see MatchResult. The rule is whether the metric NORMALISES
    AGAINST `total_relevant`:

        precision, hit    precision_relevance   no denominator from specs; a
                                                spec matching three chunks
                                                legitimately fills three slots
        recall, NDCG, MAP recall_relevance      denominator IS the spec count,
                                                so each spec may contribute at
                                                most one credit

    Getting that wrong produced NDCG@10 = 1.007 -- impossible, since NDCG is
    DCG/IDCG. Two retrieved chunks both satisfied one spec, so the actual DCG
    counted two gains while the ideal DCG was built from `total_relevant` = 1.
    MAP was inflated the same way and less visibly, having no hard ceiling to
    violate.

    The bug predates the reranker and was unreachable until retrieval got good
    enough to return two chunks for one spec inside k.

    MRR is unaffected by the choice: the first chunk matching any spec is
    necessarily the first for that spec, so both vectors have their first True
    at the same index.
    """
    precision_view = QuestionScores.compute(
        question.id, m.precision_relevance, k=k, total_relevant=question.total_relevant
    )
    recall_view = QuestionScores.compute(
        question.id, m.recall_relevance, k=k, total_relevant=question.total_relevant
    )
    return QuestionScores(
        question_id=precision_view.question_id,
        k=precision_view.k,
        n_retrieved=precision_view.n_retrieved,
        total_relevant=precision_view.total_relevant,
        hit=precision_view.hit,
        precision=precision_view.precision,
        recall=recall_view.recall,
        reciprocal_rank=precision_view.reciprocal_rank,
        average_precision=recall_view.average_precision,
        ndcg=recall_view.ndcg,
    )


async def run_tier1(
    questions: list[GoldenQuestion],
    *,
    top_k: int | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    multi_query: bool = False,
    owner_id: str | None = None,
    document_ids: list[uuid.UUID] | None = None,
    preview_chars: int = 140,
    filters: dict | None = None,
) -> Tier1Report:
    """Retrieve for every question and score the results.

    `top_k` defaults to max(k_values) so a single retrieval serves every k.
    Passing a larger top_k is legitimate — it widens the candidate pool the
    metrics can see, which is how you measure whether a reranker would help.

    `filters` records HOW the caller narrowed the suite (by tag, by id) and is
    copied into the report config. Callers do the filtering themselves, so
    without this the report cannot tell a full-suite run from a two-question
    one — and a saved JSON report would look like a complete result. Diffing a
    filtered "before" against an unfiltered "after" would then compare
    different question sets and read as a huge improvement.
    """
    fetch = top_k or max(k_values)
    started = time.perf_counter()
    results: list[QuestionResult] = []
    embedding_calls = 0

    for question in questions:
        hits = await retrieve(
            question.question,
            top_k=fetch,
            document_ids=document_ids,
            owner_id=owner_id,
            multi_query=multi_query,
        )
        # One query embedding, plus N more if multi-query expanded it.
        embedding_calls += 1 if not multi_query else 1 + len(
            {h for hit in hits for h in (hit.found_by or [])}
        )

        pairs = [(hit.filename, hit.text) for hit in hits]
        m = match(question, pairs)

        summaries = [
            HitSummary(
                rank=i,
                chunk_id=str(hit.chunk_id),
                filename=hit.filename,
                heading=(hit.heading or "").lstrip("# ").strip() or None,
                score=round(hit.score, 4),
                relevant=m.precision_relevance[i - 1],
                preview=hit.text[:preview_chars].replace("\n", " "),
                n_chars=len(hit.text),
            )
            for i, hit in enumerate(hits, start=1)
        ]

        result = QuestionResult(
            id=question.id,
            question=question.question,
            tags=list(question.tags),
            answerable=question.answerable,
            total_specs=question.total_relevant,
            specs_satisfied=len(m.satisfied_at),
            satisfied_at={str(i): r for i, r in m.satisfied_at.items()},
            unsatisfied_specs=m.unsatisfied_specs,
            hits=summaries,
            expected=[
                ExpectedFact(
                    index=i,
                    file=spec.file,
                    must_contain=list(spec.must_contain),
                    found_at_rank=m.satisfied_at.get(i),
                    matching_chunk_ids=await _resolve_spec(spec, owner_id),
                )
                for i, spec in enumerate(question.expect_chunks)
            ],
            expected_answer=question.expected_answer,
            expect_refusal=question.expect_refusal,
        )
        for k in k_values:
            result.scores[str(k)] = asdict(_score_at(question, m, k))
        results.append(result)

    # Aggregate per k. Rebuilt from the stored dicts so the report and the
    # aggregate can never disagree about what was scored.
    aggregates: dict[str, dict] = {}
    for k in k_values:
        scores = [
            QuestionScores(**{**r.scores[str(k)]})  # type: ignore[arg-type]
            for r in results
        ]
        aggregates[str(k)] = asdict(Aggregate.over(scores, k=k))

    elapsed = time.perf_counter() - started
    report = Tier1Report(
        config={
            "top_k": fetch,
            "k_values": list(k_values),
            "multi_query": multi_query,
            "n_questions": len(questions),
            "scoped_to_documents": [str(d) for d in document_ids] if document_ids else None,
            # Empty dict = the full suite ran. Anything else means these
            # metrics cover a SUBSET and must not be compared against a
            # full-suite number.
            "filters": filters or {},
            "question_ids": [q.id for q in questions],
        },
        questions=results,
        aggregates=aggregates,
        elapsed_seconds=elapsed,
        n_embedding_calls=embedding_calls,
    )

    log.info(
        "tier1_complete",
        n=len(questions),
        top_k=fetch,
        multi_query=multi_query,
        seconds=round(elapsed, 2),
        recall_at_max_k=aggregates[str(max(k_values))]["recall"],
    )
    return report

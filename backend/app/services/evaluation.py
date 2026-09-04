"""Retrieval evaluation metrics.

Tier 1 of evaluation: **no LLM, no network, no state.** Every function here is
pure arithmetic over a list of booleans, which is why this module is the one
part of the eval stack that can be unit-tested exhaustively and run on every
push.

The input contract is deliberately narrow. Callers resolve "which retrieved
chunks were relevant" into a rank-ordered `list[bool]` before calling anything
here, so the metrics never need to know how relevance was decided -- by chunk
id, by heading, by substring match. That separation is what keeps the labels in
`fixtures/golden.yaml` robust to re-chunking: change the chunker and the
matching logic adapts, while these formulas do not move.

    relevance[0] is the result at RANK 1.

`total_relevant` is how many relevant chunks the golden label says exist, which
can exceed `len(relevance)` -- that is precisely the case recall is there to
detect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def hit_at_k(relevance: list[bool], k: int) -> bool:
    """Did at least one relevant chunk make the top k?

    The bluntest useful metric, and the one non-engineers understand: "87% of
    questions retrieved something usable."
    """
    return any(relevance[:k])


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Of the k returned, what fraction were relevant.

    Divides by `k`, NOT by the number actually returned. If retrieval returns 3
    results when 5 were asked for, the 2 missing slots count against it --
    otherwise a system that returns one perfect result would score 1.0 and beat
    one that returns five good ones.
    """
    if k <= 0:
        return 0.0
    return sum(relevance[:k]) / k


def recall_at_k(relevance: list[bool], k: int, total_relevant: int) -> float | None:
    """Of all relevant chunks, what fraction reached the top k.

    **The metric that matters most in RAG.** A chunk that was not retrieved
    cannot be cited, reranked, or reasoned over -- every other quality problem
    is fixable downstream, missing recall is not.

    Returns None when `total_relevant` is 0. That is not a failure: it is an
    unanswerable question, where recall is mathematically undefined (0/0) and
    the thing to measure instead is whether the system correctly declined. The
    aggregator skips None rather than scoring it as 0.0, which would punish a
    system for behaving correctly.
    """
    if total_relevant <= 0:
        return None
    return sum(relevance[:k]) / total_relevant


def reciprocal_rank(relevance: list[bool], k: int) -> float:
    """1 / rank of the first relevant result WITHIN the top k, else 0.

    Averaged over questions this is MRR@k. The `k` is not optional: without
    truncation this scans the whole retrieved list and returns the same value
    at every k, which makes a per-k table silently meaningless. If the first
    relevant result sits at rank 8, MRR@5 must be 0 -- at k=5 the user never
    saw it.

    Worth knowing it is a poor fit for RAG anyway: when five chunks all go into
    the prompt, whether the good one was rank 1 or rank 3 barely matters. It is
    a search metric, kept because interviews ask for it.
    """
    for index, is_relevant in enumerate(relevance[:k], start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def average_precision(relevance: list[bool], k: int, total_relevant: int) -> float | None:
    """Mean of precision@i at each rank within the top k holding a relevant item.

    Averaged over questions this is MAP@k. Truncated at k for the same reason
    as reciprocal_rank: a relevant result the user never saw cannot count
    towards the score.
    """
    if total_relevant <= 0:
        return None
    hits = 0
    running = 0.0
    for index, is_relevant in enumerate(relevance[:k], start=1):
        if is_relevant:
            hits += 1
            running += hits / index
    return running / total_relevant


def dcg_at_k(gains: list[float], k: int) -> float:
    """Discounted Cumulative Gain: Σ gain_i / log2(i + 1).

    The log discount is what makes it rank-aware -- a relevant result at rank 1
    contributes 1/log2(2) = 1.0, at rank 4 only 1/log2(5) = 0.43.
    """
    return sum(gain / math.log2(index + 1) for index, gain in enumerate(gains[:k], start=1))


def ndcg_at_k(relevance: list[bool], k: int, total_relevant: int) -> float | None:
    """DCG normalised by the best achievable DCG for this question.

    Normalisation is what makes scores comparable across questions: a question
    with one relevant chunk and one with five cannot be compared on raw DCG.

    Binary relevance here. NDCG's real advantage is GRADED relevance
    (perfect / good / marginal), but producing graded judgments is an annotation
    cost most RAG projects rightly decline -- so this uses 1.0/0.0 gains and is
    kept mainly because interviews ask for it.
    """
    if total_relevant <= 0:
        return None
    gains = [1.0 if r else 0.0 for r in relevance]
    actual = dcg_at_k(gains, k)
    # Ideal ordering: every relevant chunk first, capped by k and by how many
    # relevant chunks actually exist.
    ideal = dcg_at_k([1.0] * min(k, total_relevant), k)
    if ideal == 0.0:
        return None
    return actual / ideal


@dataclass(frozen=True, slots=True)
class QuestionScores:
    """Per-question retrieval scores at one k."""

    question_id: str
    k: int
    n_retrieved: int
    total_relevant: int
    hit: bool
    precision: float
    # None for unanswerable questions -- see recall_at_k.
    recall: float | None
    reciprocal_rank: float
    average_precision: float | None
    ndcg: float | None

    @classmethod
    def compute(
        cls, question_id: str, relevance: list[bool], k: int, total_relevant: int
    ) -> QuestionScores:
        return cls(
            question_id=question_id,
            k=k,
            n_retrieved=len(relevance),
            total_relevant=total_relevant,
            hit=hit_at_k(relevance, k),
            precision=precision_at_k(relevance, k),
            recall=recall_at_k(relevance, k, total_relevant),
            reciprocal_rank=reciprocal_rank(relevance, k),
            average_precision=average_precision(relevance, k, total_relevant),
            ndcg=ndcg_at_k(relevance, k, total_relevant),
        )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Suite-level means at one k.

    Metrics that are None for a question are EXCLUDED from that metric's mean
    rather than counted as zero. Unanswerable questions have no defined recall,
    and averaging them in as 0.0 would make a correctly-abstaining system look
    broken.
    """

    k: int
    n_questions: int
    n_scored: int
    hit_rate: float | None
    precision: float | None
    recall: float | None
    mrr: float | None
    map: float | None
    ndcg: float | None

    @classmethod
    def over(cls, scores: list[QuestionScores], k: int) -> Aggregate:
        answerable = [s for s in scores if s.total_relevant > 0]
        return cls(
            k=k,
            n_questions=len(scores),
            n_scored=len(answerable),
            # Hit rate and MRR are computed over ANSWERABLE questions only:
            # "did we find something" is meaningless where there was nothing.
            hit_rate=_mean([1.0 if s.hit else 0.0 for s in answerable]),
            precision=_mean([s.precision for s in answerable]),
            recall=_mean([s.recall for s in answerable if s.recall is not None]),
            mrr=_mean([s.reciprocal_rank for s in answerable]),
            map=_mean([s.average_precision for s in answerable if s.average_precision is not None]),
            ndcg=_mean([s.ndcg for s in answerable if s.ndcg is not None]),
        )

"""Unit tests for the retrieval metrics.

Every expected value here is hand-computable, and the arithmetic is written out
in the assertion or its comment. That is deliberate: a test asserting
`recall == 0.5` with no visible derivation proves only that the code does what
it currently does.
"""

import math

import pytest

from app.services.evaluation import (
    Aggregate,
    QuestionScores,
    average_precision,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Rank:                1      2      3      4      5
RELEVANCE = [True, False, True, False, False]


class TestHitAtK:
    def test_hit_within_k(self):
        assert hit_at_k(RELEVANCE, k=1) is True

    def test_no_hit_when_k_excludes_every_relevant_result(self):
        # Rank 1 is relevant, so only k=0 can miss it.
        assert hit_at_k(RELEVANCE, k=0) is False

    def test_miss_when_nothing_relevant(self):
        assert hit_at_k([False, False], k=2) is False

    def test_empty_results(self):
        assert hit_at_k([], k=5) is False


class TestPrecisionAtK:
    def test_divides_by_k_not_by_results_returned(self):
        # THE important behaviour: 1 relevant result out of 5 slots asked for
        # is 0.2, even though only 1 result came back. Dividing by the number
        # returned would score a single perfect hit as 1.0 and let a system
        # game the metric by returning less.
        assert precision_at_k([True], k=5) == pytest.approx(1 / 5)

    def test_partial(self):
        # ranks 1 and 3 relevant, out of 3 slots
        assert precision_at_k(RELEVANCE, k=3) == pytest.approx(2 / 3)

    def test_all_relevant(self):
        assert precision_at_k([True, True], k=2) == pytest.approx(1.0)

    def test_k_zero_is_zero_not_a_crash(self):
        assert precision_at_k(RELEVANCE, k=0) == 0.0


class TestRecallAtK:
    def test_finds_some_of_the_relevant_set(self):
        # 2 of 4 known-relevant chunks reached the top 5
        assert recall_at_k(RELEVANCE, k=5, total_relevant=4) == pytest.approx(0.5)

    def test_monotonically_non_decreasing_in_k(self):
        # The property that makes quoting "recall" without a k meaningless.
        r1 = recall_at_k(RELEVANCE, k=1, total_relevant=4)
        r3 = recall_at_k(RELEVANCE, k=3, total_relevant=4)
        r5 = recall_at_k(RELEVANCE, k=5, total_relevant=4)
        assert r1 is not None and r3 is not None and r5 is not None
        assert r1 <= r3 <= r5

    def test_unanswerable_question_returns_none_not_zero(self):
        # An unanswerable question has no relevant chunks, so recall is 0/0.
        # Returning 0.0 would punish a system for correctly finding nothing.
        assert recall_at_k([False, False], k=2, total_relevant=0) is None

    def test_can_exceed_results_returned(self):
        # total_relevant may be larger than the result list -- that gap is
        # exactly what recall exists to expose.
        assert recall_at_k([True], k=5, total_relevant=10) == pytest.approx(0.1)


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank([True, False], k=2) == pytest.approx(1.0)

    def test_third_result_relevant(self):
        assert reciprocal_rank([False, False, True], k=3) == pytest.approx(1 / 3)

    def test_none_relevant_is_zero(self):
        assert reciprocal_rank([False, False], k=2) == 0.0

    def test_only_the_first_hit_counts(self):
        # Later relevant results do not improve RR -- which is why it is a poor
        # fit for RAG, where every retrieved chunk enters the prompt.
        assert reciprocal_rank([False, True, True], k=3) == pytest.approx(1 / 2)

    def test_truncates_at_k(self):
        """A relevant result BEYOND k must score 0 — the user never saw it.

        Without truncation this scans the whole list and returns the same value
        at every k, which makes a per-k report silently meaningless.
        """
        relevance = [False, False, False, False, False, False, False, True]
        assert reciprocal_rank(relevance, k=5) == 0.0
        assert reciprocal_rank(relevance, k=8) == pytest.approx(1 / 8)

    def test_varies_with_k(self):
        # The property whose absence was the bug: MRR must move as k moves.
        relevance = [False, False, True]
        assert reciprocal_rank(relevance, k=2) == 0.0
        assert reciprocal_rank(relevance, k=3) == pytest.approx(1 / 3)


class TestAveragePrecision:
    def test_hand_computed(self):
        # hits at ranks 1 and 3:
        #   precision@1 = 1/1 = 1.0
        #   precision@3 = 2/3 = 0.667
        #   AP = (1.0 + 0.667) / 4 relevant = 0.4167
        expected = (1 / 1 + 2 / 3) / 4
        assert average_precision(RELEVANCE, k=5, total_relevant=4) == pytest.approx(expected)

    def test_perfect_ordering_scores_one(self):
        assert average_precision([True, True], k=2, total_relevant=2) == pytest.approx(1.0)

    def test_unanswerable_returns_none(self):
        assert average_precision([False], k=1, total_relevant=0) is None

    def test_truncates_at_k(self):
        # Only the hit at rank 1 counts at k=2; the rank-3 hit is unseen.
        assert average_precision(RELEVANCE, k=2, total_relevant=4) == pytest.approx(1 / 4)


class TestDcg:
    def test_log_discount(self):
        # rank 1 -> 1/log2(2) = 1.0 ; rank 2 -> 1/log2(3) = 0.6309
        assert dcg_at_k([1.0, 1.0], k=2) == pytest.approx(1.0 + 1 / math.log2(3))

    def test_k_truncates(self):
        assert dcg_at_k([1.0, 1.0, 1.0], k=1) == pytest.approx(1.0)


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k([True, True], k=2, total_relevant=2) == pytest.approx(1.0)

    def test_relevant_result_lower_down_scores_less(self):
        good = ndcg_at_k([True, False], k=2, total_relevant=1)
        worse = ndcg_at_k([False, True], k=2, total_relevant=1)
        assert good is not None and worse is not None
        assert worse < good

    def test_hand_computed(self):
        # relevance ranks 1 and 3, total_relevant 4, k=5
        #   DCG   = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        #   IDCG  = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5)
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, 5))
        assert ndcg_at_k(RELEVANCE, k=5, total_relevant=4) == pytest.approx(dcg / idcg)

    def test_unanswerable_returns_none(self):
        assert ndcg_at_k([False], k=1, total_relevant=0) is None


class TestAggregate:
    def test_excludes_unanswerable_from_recall_mean(self):
        """The behaviour that stops a correct refusal looking like a failure."""
        scores = [
            QuestionScores.compute("answerable", [True, False], k=2, total_relevant=1),
            QuestionScores.compute("unanswerable", [False, False], k=2, total_relevant=0),
        ]
        agg = Aggregate.over(scores, k=2)

        assert agg.n_questions == 2
        # Only the answerable question is scored.
        assert agg.n_scored == 1
        # Recall is 1.0, not 0.5 -- the unanswerable row is skipped, not zeroed.
        assert agg.recall == pytest.approx(1.0)
        assert agg.hit_rate == pytest.approx(1.0)

    def test_all_unanswerable_yields_none_rather_than_zero(self):
        scores = [QuestionScores.compute("u", [False], k=1, total_relevant=0)]
        agg = Aggregate.over(scores, k=1)
        assert agg.n_scored == 0
        assert agg.recall is None
        assert agg.hit_rate is None

    def test_means_across_questions(self):
        scores = [
            QuestionScores.compute("a", [True], k=1, total_relevant=1),
            QuestionScores.compute("b", [False], k=1, total_relevant=1),
        ]
        agg = Aggregate.over(scores, k=1)
        assert agg.recall == pytest.approx(0.5)
        assert agg.precision == pytest.approx(0.5)
        assert agg.mrr == pytest.approx(0.5)

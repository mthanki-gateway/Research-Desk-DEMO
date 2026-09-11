"""Phase 1: BM25, the absolute floor, and the listwise reranker.

The BM25 tests assert PROPERTIES rather than numbers -- "a rarer term outweighs
a common one", "saturation is sub-linear" -- because the absolute scores are
meaningless on their own and pinning them would just freeze an implementation
detail. The properties are what make the algorithm the right one.
"""

import uuid

import pytest

from app.config import Settings
from app.services.lexical import BM25Index, tokenize
from app.services.retrieval import apply_floor
from app.services.vectorstore import SearchHit


def _hit(n: int, *, score: float = 0.5, text: str = "passage") -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(int=n),
        document_id=uuid.UUID(int=99),
        filename=f"doc{n}.md",
        page=1,
        chunk_index=n,
        heading="## Findings",
        text=text,
        score=score,
        meta={},
        found_by=["q"],
    )


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Revenue GROWTH") == ["revenue", "growth"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("4,182 employees", ["4,182", "employees"]),
            ("incident INC-2024-1183", ["incident", "inc-2024-1183"]),
            ("models/gemini-3.6-flash", ["models/gemini-3.6-flash"]),
        ],
    )
    def test_keeps_internal_punctuation(self, text, expected):
        """These ARE the queries BM25 exists to win.

        Splitting on every non-alphanumeric would shatter "4,182" into "4" and
        "182" and throw away exactly the rarity that makes an identifier
        findable -- handing the win back to dense retrieval, which is bad at
        precisely these.
        """
        assert tokenize(text) == expected

    def test_trailing_punctuation_is_dropped(self):
        """INTERNAL punctuation is kept, trailing punctuation is not, and the
        asymmetry is what makes matching work.

        "62.1%" indexes as "62.1", so a query asking about "62.1" finds it.
        Keeping the percent sign would make the token "62.1%", which a query
        writing the bare number would miss -- breaking the exact match the
        lexical half exists to provide.
        """
        assert tokenize("margin was 62.1%") == ["margin", "62.1"]
        assert tokenize("62.1") == ["62.1"]

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the revenue of the year")

    def test_empty_text(self):
        assert tokenize("") == []


# --------------------------------------------------------------------------
# BM25 properties
# --------------------------------------------------------------------------


class TestBM25:
    def test_finds_the_matching_document(self):
        index = BM25Index(
            [
                (uuid.UUID(int=1), "gross margin improved to 62.1 percent"),
                (uuid.UUID(int=2), "the cat sat on the mat"),
            ]
        )
        hits = index.search("gross margin", limit=5)
        assert [h.chunk_id for h in hits] == [uuid.UUID(int=1)]

    def test_a_rare_term_outweighs_a_common_one(self):
        """IDF is the whole point. A term in every document says nothing about
        which document you want; a term in one says almost everything."""
        docs = [(uuid.UUID(int=i), "revenue report") for i in range(1, 10)]
        docs.append((uuid.UUID(int=10), "revenue report INC-2024-1183"))
        index = BM25Index(docs)

        hits = index.search("revenue INC-2024-1183", limit=3)
        assert hits[0].chunk_id == uuid.UUID(int=10)

    def test_term_frequency_saturates(self):
        """k1 caps the value of repetition. Without saturation a chunk that
        says "margin" nine times beats one that actually answers the question.
        """
        index = BM25Index(
            [
                (uuid.UUID(int=1), "margin"),
                (uuid.UUID(int=2), "margin margin margin margin margin margin"),
            ]
        )
        by_id = {h.chunk_id: h.score for h in index.search("margin", limit=5)}
        once, six_times = by_id[uuid.UUID(int=1)], by_id[uuid.UUID(int=2)]
        assert six_times > once, "more occurrences should still score higher"
        assert six_times < once * 6, "but far less than linearly"

    def test_length_normalisation_prefers_the_focused_chunk(self):
        """b=0.75. One mention inside two words is stronger evidence than one
        mention buried in two hundred."""
        index = BM25Index(
            [
                (uuid.UUID(int=1), "capex guidance"),
                (uuid.UUID(int=2), "capex " + " ".join(f"filler{i}" for i in range(200))),
            ]
        )
        ranked = index.search("capex", limit=5)
        assert ranked[0].chunk_id == uuid.UUID(int=1)

    def test_idf_never_goes_negative(self):
        """The textbook IDF is negative for a term in more than half the
        corpus, so a chunk gets PUNISHED for containing a common word. On a
        small corpus that is most words. The `1 +` inside the log is the fix."""
        docs = [(uuid.UUID(int=i), "revenue") for i in range(1, 11)]
        index = BM25Index(docs)
        assert all(h.score > 0 for h in index.search("revenue", limit=10))

    def test_no_match_returns_nothing(self):
        """Sparse retrieval must be able to say "not here" -- it is half of what
        makes the absolute floor meaningful."""
        index = BM25Index([(uuid.UUID(int=1), "the cat sat on the mat")])
        assert index.search("quantum chromodynamics", limit=5) == []

    def test_empty_index_is_safe(self):
        assert BM25Index([]).search("anything", limit=5) == []

    def test_empty_query_is_safe(self):
        index = BM25Index([(uuid.UUID(int=1), "content")])
        assert index.search("   ", limit=5) == []

    def test_ties_break_deterministically(self):
        """Two chunks with equal scores swapping between runs would make the
        evaluation numbers wobble for reasons unrelated to any change."""
        docs = [(uuid.UUID(int=i), "identical text here") for i in range(1, 6)]
        first = [h.chunk_id for h in BM25Index(docs).search("identical", limit=5)]
        second = [h.chunk_id for h in BM25Index(docs).search("identical", limit=5)]
        assert first == second

    def test_respects_the_limit(self):
        docs = [(uuid.UUID(int=i), "revenue") for i in range(1, 20)]
        assert len(BM25Index(docs).search("revenue", limit=3)) == 3


# --------------------------------------------------------------------------
# the absolute floor
# --------------------------------------------------------------------------


class TestFloor:
    def test_disabled_by_default(self):
        """0.0 must be a true no-op: the floor ships off until calibrated, and
        a floor that quietly drops chunks would change every measured number."""
        hits = [_hit(1, score=0.1), _hit(2, score=0.9)]
        assert apply_floor(hits, 0.0) == hits

    def test_drops_everything_below_the_threshold(self):
        hits = [_hit(1, score=0.9), _hit(2, score=0.4), _hit(3, score=0.2)]
        kept = apply_floor(hits, 0.5)
        assert [h.chunk_index for h in kept] == [1]

    def test_can_return_nothing(self):
        """THE POINT. RRF reads rank and discards magnitude, so it always
        produces a confident top-k -- every candidate could be terrible and the
        output would look like a perfect run. This is what makes "the corpus
        cannot answer that" reachable at all."""
        assert apply_floor([_hit(1, score=0.1)], 0.5) == []

    def test_boundary_is_inclusive(self):
        assert len(apply_floor([_hit(1, score=0.5)], 0.5)) == 1

    def test_preserves_order(self):
        hits = [_hit(1, score=0.9), _hit(2, score=0.8), _hit(3, score=0.7)]
        assert [h.chunk_index for h in apply_floor(hits, 0.5)] == [1, 2, 3]


class TestPhase1DefaultsAreOn:
    """Every Phase 1 stage ships ENABLED, because each was measured first.

    They went in switched off and were turned on only once the golden set
    justified them (see the tables in config.py). Pinning the defaults here so
    that silently disabling one -- which would look like nothing at all, just
    slightly worse answers -- fails a test instead.
    """

    def test_hybrid_on(self):
        assert Settings(google_api_key="x").hybrid_search is True

    def test_rerank_on(self):
        assert Settings(google_api_key="x").rerank is True

    def test_floor_is_the_calibrated_value(self):
        """0.60 keeps 100% of relevant chunks and drops 40% of the noise.
        0.62 begins discarding real answers, which is the wrong trade: nothing
        downstream recovers from evidence that never arrived."""
        assert Settings(google_api_key="x").retrieval_score_floor == 0.60

    def test_the_floor_sits_below_the_weakest_relevant_chunk(self):
        """Measured minimum over the golden set was 0.619. The margin is thin
        on 27 samples, so this is the guard that stops someone nudging the
        floor up past it without re-running the calibration."""
        assert Settings(google_api_key="x").retrieval_score_floor < 0.619

    def test_candidate_pools_are_deeper_than_the_default_k(self):
        """A reranker can only reorder what it is handed, and a chunk one place
        past a list's end contributes nothing to RRF. Fetching exactly k and
        then 'reranking' is an expensive no-op."""
        s = Settings(google_api_key="x")
        assert s.rerank_candidates > s.retrieval_top_k
        assert s.hybrid_candidates > s.retrieval_top_k

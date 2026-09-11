"""Tier 2: the RAGAS judge and the harness that drives it.

None of these tests call RAGAS. They cannot -- it is an optional extra, and a
test suite that needed a 15 rpm judge model would take minutes and fail
offline. What they pin instead is everything AROUND the metric calls, which is
where this code's bugs have actually lived:

* the judge wrapper must be SHARED, because its rate limiter is per instance;
* the full quartet must be the default, because half a suite reported as
  "judged by RAGAS" is worse than an honest refusal;
* a missing score must stay distinguishable from a zero and from a failure;
* concurrent judging must not reorder the report.
"""

import asyncio

import pytest

from app.config import Settings
from app.services import judge as judge_mod
from app.services import tier2 as tier2_mod
from app.services.golden import GoldenQuestion
from app.services.judge import ALL_METRICS, CHEAP_METRICS, JudgeScores


def _settings(**overrides) -> Settings:
    return Settings(**{"google_api_key": "x", **overrides})


def _question(qid: str, *, reference: str = "the reference") -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=f"question {qid}?",
        expected_answer=reference,
        tags=("unit",),
    )


@pytest.fixture(autouse=True)
def _clean_judge_cache():
    """Every test starts with no cached wrapper.

    Without this the cache leaks between tests, and a test that asserts a
    wrapper was BUILT would pass or fail depending on execution order.
    """
    judge_mod.reset_judge_cache()
    yield
    judge_mod.reset_judge_cache()


# --------------------------------------------------------------------------
# the shared judge wrapper
# --------------------------------------------------------------------------


class TestJudgeIsShared:
    """The bug this cache exists to prevent.

    `_build_judge` constructs an InMemoryRateLimiter whose token bucket is per
    instance. Building it per question -- which `judge_answer` used to do --
    gave a 20-question run 20 independent limiters, each starting full, so it
    paced at 20x the intended rate while appearing correctly configured.
    """

    def test_built_once_and_reused(self, monkeypatch):
        builds = []

        def fake_build():
            builds.append(1)
            return ("llm", "embeddings")

        monkeypatch.setattr(judge_mod, "_build_judge", fake_build)
        monkeypatch.setattr(judge_mod, "get_settings", lambda: _settings())

        first = judge_mod.get_judge()
        second = judge_mod.get_judge()
        assert first is second, "a second call rebuilt the wrapper"
        assert len(builds) == 1

    def test_a_different_judge_model_rebuilds(self, monkeypatch):
        """Keyed on config, so changing the judge in a REPL or a test does not
        silently keep grading with the old one."""
        builds = []
        monkeypatch.setattr(
            judge_mod, "_build_judge", lambda: (builds.append(1), ("llm", "emb"))[1]
        )

        monkeypatch.setattr(judge_mod, "get_settings", lambda: _settings())
        judge_mod.get_judge()
        monkeypatch.setattr(
            judge_mod,
            "get_settings",
            lambda: _settings(judge_model="models/something-else"),
        )
        judge_mod.get_judge()
        assert len(builds) == 2

    def test_a_different_quota_rebuilds(self, monkeypatch):
        """The rpm is baked into the limiter, so it is part of the identity."""
        builds = []
        monkeypatch.setattr(
            judge_mod, "_build_judge", lambda: (builds.append(1), ("llm", "emb"))[1]
        )

        monkeypatch.setattr(judge_mod, "get_settings", lambda: _settings())
        judge_mod.get_judge()
        monkeypatch.setattr(
            judge_mod,
            "get_settings",
            lambda: _settings(judge_requests_per_minute=60),
        )
        judge_mod.get_judge()
        assert len(builds) == 2


# --------------------------------------------------------------------------
# what gets judged by default
# --------------------------------------------------------------------------


class TestFullQuartetIsTheDefault:
    def test_all_metrics_is_the_full_ragas_core(self):
        assert set(ALL_METRICS) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }

    def test_cheap_metrics_is_a_strict_subset(self):
        assert set(CHEAP_METRICS) < set(ALL_METRICS)

    def test_judge_answer_defaults_to_the_full_quartet(self):
        """The signature default, not just the CLI's. `judge_answer` is also
        the entry point for anything calling it directly."""
        import inspect

        default = inspect.signature(judge_mod.judge_answer).parameters["metrics"].default
        assert default == ALL_METRICS

    def test_run_tier2_defaults_to_the_full_quartet(self):
        import inspect

        default = inspect.signature(tier2_mod.run_tier2).parameters["metrics"].default
        assert default == ALL_METRICS


class TestMissingReferenceIsRecorded:
    """Two Nones with no explanation is the ambiguity JudgeScores exists to
    avoid: a reader cannot tell a judge failure from a question that had no
    ground truth to compare against.

    Tested through the pure `reference_gap` rather than `judge_answer`, because
    every other path through that function stops at the ragas import on a
    default install -- which is exactly why the check was extracted.
    """

    def test_no_reference_gives_a_reason(self):
        gap = judge_mod.reference_gap(ALL_METRICS, "   ")
        assert gap and "no reference" in gap

    def test_a_reference_means_no_gap(self):
        assert judge_mod.reference_gap(ALL_METRICS, "the reference") is None

    def test_no_gap_when_those_metrics_were_not_asked_for(self):
        """A --cheap-metrics run has nothing to explain: it never wanted the
        reference-based pair, so reporting a missing reference would be noise."""
        assert judge_mod.reference_gap(CHEAP_METRICS, "") is None


@pytest.mark.asyncio
class TestRagasAvailability:
    async def test_unavailable_ragas_fails_loudly_not_silently(self, monkeypatch):
        """The state this whole change was about. ragas was never installed in
        the image, so every score came back None with the reason buried in a
        field the table does not print."""
        monkeypatch.setattr(
            judge_mod, "ragas_available", lambda: (False, "ragas is not installed")
        )
        scores = await judge_mod.judge_answer(
            question="q", answer="a", contexts=["c"], reference="r"
        )
        assert scores.error == "ragas is not installed"
        assert scores.faithfulness is None


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """run_tier2 with generation stubbed and the cache pointed at tmp_path."""
    monkeypatch.setattr(tier2_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tier2_mod, "get_settings", lambda: _settings())
    monkeypatch.setattr(tier2_mod, "ragas_available", lambda: (True, ""))

    async def fake_generate(question, **kw):
        return tier2_mod.GeneratedAnswer(
            question_id=question.id,
            question=question.question,
            answer=f"answer for {question.id}",
            contexts=[f"context for {question.id}"],
        )

    monkeypatch.setattr(tier2_mod, "_generate", fake_generate)
    return monkeypatch


@pytest.mark.asyncio
class TestRunTier2:
    async def test_order_survives_concurrent_judging(self, harness):
        """gather preserves argument order regardless of completion order.

        It matters because reports are diffed between runs: rows landing in
        completion order would make every re-run look like a change.
        """

        async def slow_then_fast(*, question, answer, contexts, reference, metrics):
            # The LAST question finishes first, so a naive as-completed
            # collection would invert the report.
            await asyncio.sleep(0.03 if question.endswith("a?") else 0.001)
            return JudgeScores(
                faithfulness=1.0, answer_relevancy=1.0,
                context_precision=1.0, context_recall=1.0,
            )

        harness.setattr(tier2_mod, "judge_answer", slow_then_fast)

        report = await tier2_mod.run_tier2(
            [_question("a"), _question("b"), _question("c")]
        )
        assert [q.question_id for q in report.questions] == ["a", "b", "c"]

    async def test_concurrency_is_bounded(self, harness):
        """The semaphore is what keeps a 20-question run from opening 20
        simultaneous judge pipelines, each of which fans out ~12 calls."""
        in_flight = 0
        peak = 0

        async def counting_judge(**kw):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", counting_judge)

        await tier2_mod.run_tier2(
            [_question(str(i)) for i in range(10)], judge_concurrency=3
        )
        assert peak <= 3, f"ran {peak} judges at once"

    async def test_serialises_at_concurrency_one(self, harness):
        peak = 0
        in_flight = 0

        async def counting_judge(**kw):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", counting_judge)
        await tier2_mod.run_tier2(
            [_question(str(i)) for i in range(4)], judge_concurrency=1
        )
        assert peak == 1

    async def test_the_full_quartet_reaches_the_judge(self, harness):
        seen = {}

        async def capture(**kw):
            seen.update(kw)
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", capture)
        await tier2_mod.run_tier2([_question("a")])
        assert seen["metrics"] == ALL_METRICS
        assert seen["reference"] == "the reference"

    async def test_aggregates_skip_none_rather_than_averaging_zero(self, harness):
        """A failed judge call must not drag the mean down. Averaging None as
        zero makes a working system look broken -- the same trap Tier 1 avoids
        with unanswerable questions."""

        async def half_failing(*, question, **kw):
            if question.endswith("b?"):
                return JudgeScores.failed("boom")
            return JudgeScores(
                faithfulness=0.8, answer_relevancy=None,
                context_precision=None, context_recall=None,
            )

        harness.setattr(tier2_mod, "judge_answer", half_failing)

        report = await tier2_mod.run_tier2([_question("a"), _question("b")])
        agg = report.aggregates
        assert agg["faithfulness"] == pytest.approx(0.8)
        assert agg["faithfulness_n"] == 1
        assert agg["answer_relevancy"] is None
        assert agg["answer_relevancy_n"] == 0
        assert agg["n_judge_errors"] == 1

    async def test_judge_false_skips_judging_but_still_generates(self, harness):
        called = False

        async def should_not_run(**kw):
            nonlocal called
            called = True
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", should_not_run)

        report = await tier2_mod.run_tier2([_question("a")], judge=False)
        assert not called
        assert report.questions[0].answer == "answer for a"
        assert report.questions[0].scores["error"] == "judging skipped"

    async def test_an_answer_is_cached_and_reused(self, harness):
        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", ok)

        first = await tier2_mod.run_tier2([_question("a")])
        assert (first.n_generated, first.n_from_cache) == (1, 0)
        assert first.questions[0].from_cache is False

        second = await tier2_mod.run_tier2([_question("a")])
        assert (second.n_generated, second.n_from_cache) == (0, 1)
        assert second.questions[0].from_cache is True

    async def test_a_failed_generation_is_not_cached(self, harness):
        """Caching a failure would make every later run reuse it and hide a
        transient outage forever."""

        async def failing_generate(question, **kw):
            return tier2_mod.GeneratedAnswer(
                question_id=question.id, question=question.question,
                answer="", contexts=[], error="RuntimeError: quota",
            )

        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "_generate", failing_generate)
        harness.setattr(tier2_mod, "judge_answer", ok)

        await tier2_mod.run_tier2([_question("a")])
        second = await tier2_mod.run_tier2([_question("a")])
        assert second.n_generated == 1, "a failure was cached"
        assert second.questions[0].scores["error"] == "RuntimeError: quota"


class TestCacheKey:
    def test_config_is_part_of_the_key(self):
        """Omitting any input that changes an answer is the worst kind of
        evaluation bug, because the numbers stay plausible."""
        base = {"mode": "agent", "top_k": 5, "multi_query": False, "model": "m"}
        key = tier2_mod._cache_key("q1", base)
        for field, value in [
            ("mode", "baseline"), ("top_k", 10),
            ("multi_query", True), ("model", "other"),
        ]:
            assert tier2_mod._cache_key("q1", {**base, field: value}) != key, field

    def test_the_same_config_is_stable(self):
        base = {"mode": "agent", "top_k": 5}
        assert tier2_mod._cache_key("q1", base) == tier2_mod._cache_key("q1", base)

    def test_different_questions_differ(self):
        base = {"mode": "agent"}
        assert tier2_mod._cache_key("q1", base) != tier2_mod._cache_key("q2", base)


class TestGenConfigCoversTheAnswerModel:
    """The cache key must name the model that writes the answer.

    It recorded only `llm_model` (the planner). Both modes produce their answer
    text with `answer_model`, so swapping that left the key unchanged and the
    cache served answers written by the previous model -- the failure mode
    `_cache_key`'s own docstring warns about, and invisible because the numbers
    stay plausible.
    """

    def test_answer_model_changes_the_key(self):
        base = {"mode": "agent", "top_k": 5, "multi_query": False,
                "model": "planner", "answer_model": "answerer"}
        assert tier2_mod._cache_key("q", base) != tier2_mod._cache_key(
            "q", {**base, "answer_model": "a-different-answerer"}
        )

    @pytest.mark.asyncio
    async def test_the_retrieval_pipeline_is_in_the_key(self, harness):
        """Every retrieval stage changes which passages the answer is built
        from, so comparing "with reranking" against "without" would otherwise
        score one cached set of answers against itself. An evaluation cache
        that ignores the thing under test produces confident nonsense."""
        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", ok)
        report = await tier2_mod.run_tier2([_question("a")])
        for field in ("hybrid", "rerank", "floor", "parents"):
            assert field in report.config, field

    @pytest.mark.asyncio
    async def test_run_tier2_records_both_models(self, harness):
        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        harness.setattr(tier2_mod, "judge_answer", ok)
        report = await tier2_mod.run_tier2([_question("a")])

        settings = _settings()
        assert report.config["answer_model"] == settings.answer_model
        assert report.config["model"] == settings.llm_model

    @pytest.mark.asyncio
    async def test_changing_the_answer_model_misses_the_cache(self, monkeypatch, tmp_path):
        """End to end: the same question under a different answer model must
        regenerate rather than reuse."""
        monkeypatch.setattr(tier2_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(tier2_mod, "ragas_available", lambda: (True, ""))

        generated = []

        async def counting_generate(question, **kw):
            generated.append(question.id)
            return tier2_mod.GeneratedAnswer(
                question_id=question.id, question=question.question,
                answer="an answer", contexts=["a context"],
            )

        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        monkeypatch.setattr(tier2_mod, "_generate", counting_generate)
        monkeypatch.setattr(tier2_mod, "judge_answer", ok)

        monkeypatch.setattr(tier2_mod, "get_settings", lambda: _settings())
        await tier2_mod.run_tier2([_question("a")])
        assert generated == ["a"]

        # Same everything except the model that writes the answer.
        monkeypatch.setattr(
            tier2_mod,
            "get_settings",
            lambda: _settings(answer_model="models/some-other-answerer"),
        )
        await tier2_mod.run_tier2([_question("a")])
        assert generated == ["a", "a"], "reused an answer from a different model"


class TestTheHarnessRecordsWhatTheModelSaw:
    """The judge must be given the context the model was given.

    The harness recorded bare `hit.text`, omitting the
    `[n] document · filename › heading` label that `build_context` puts in
    front of every passage. Because the drafter is REQUIRED to name its source
    in the sentence, RAGAS then decomposed "According to acme-report.md, ..."
    into a claim about a filename that appeared nowhere in the context it was
    handed, and scored it unsupported.

    Measured on four questions, same answers and same retrieval, changing only
    what was recorded: faithfulness 0.125 -> 0.917.
    """

    def test_blocks_carry_the_source_label(self):
        from app.services.retrieval import context_blocks

        blocks = context_blocks([_search_hit()])
        assert "acme-report.md" in blocks[0]
        assert "Financial Summary" in blocks[0]

    def test_blocks_are_what_build_context_joins(self):
        """One renderer, so the two can never drift. If they did, this bug
        would come back in a form nothing detects."""
        from app.services.retrieval import build_context, context_blocks

        hits = [_search_hit(), _search_hit(2)]
        assert build_context(hits) == "\n\n".join(context_blocks(hits))

    def test_one_block_per_hit(self):
        from app.services.retrieval import context_blocks

        assert len(context_blocks([_search_hit(), _search_hit(2)])) == 2


@pytest.mark.asyncio
class TestAFailedGenerationIsNotAnAnswer:
    """`draft` turns a 503 or a 429 into readable text rather than raising --
    right for a chat window, wrong for a harness.

    `tier2` already refused to cache failures, but the guard checked for an
    EXCEPTION and this path returns normally. So "The answer could not be
    generated (503...)" was cached and scored for faithfulness, where no
    passage supports it, dragging the mean down on every later run.
    """

    async def test_a_flagged_failure_is_recorded_as_an_error(self, monkeypatch, tmp_path):
        import app.agent.graph as graph_mod

        monkeypatch.setattr(tier2_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(tier2_mod, "get_settings", lambda: _settings())

        class Failed:
            answer = "The answer could not be generated (503)."
            evidence: list = []
            iterations, sufficient = 0, True
            sub_questions: list = []
            citations: list = []
            failed = True

        async def fake_run_agent(question, **kw):
            return Failed()

        monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
        out = await tier2_mod._generate(
            _question("a"), mode="agent", top_k=5, multi_query=False, owner_id=None
        )
        assert out.error and "generation failed" in out.error

    async def test_it_is_not_cached(self, monkeypatch, tmp_path):
        """Caching it would make every later run reuse the outage forever."""
        import app.agent.graph as graph_mod

        cache = tmp_path / "cache"
        monkeypatch.setattr(tier2_mod, "CACHE_DIR", cache)
        monkeypatch.setattr(tier2_mod, "get_settings", lambda: _settings())
        monkeypatch.setattr(tier2_mod, "ragas_available", lambda: (True, ""))

        class Failed:
            answer = "The answer could not be generated (429)."
            evidence: list = []
            iterations, sufficient = 0, True
            sub_questions: list = []
            citations: list = []
            failed = True

        async def fake_run_agent(question, **kw):
            return Failed()

        async def ok(**kw):
            return JudgeScores(1.0, 1.0, 1.0, 1.0)

        monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
        monkeypatch.setattr(tier2_mod, "judge_answer", ok)

        await tier2_mod.run_tier2([_question("a")])
        second = await tier2_mod.run_tier2([_question("a")])
        assert second.n_generated == 1, "an outage was cached as an answer"

    async def test_a_real_answer_still_caches(self, monkeypatch, tmp_path):
        import app.agent.graph as graph_mod

        monkeypatch.setattr(tier2_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(tier2_mod, "get_settings", lambda: _settings())

        class Ok:
            answer = "A real answer [1]."
            evidence: list = []
            iterations, sufficient = 1, True
            sub_questions: list = []
            citations = [1]
            failed = False

        async def fake_run_agent(question, **kw):
            return Ok()

        monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
        out = await tier2_mod._generate(
            _question("a"), mode="agent", top_k=5, multi_query=False, owner_id=None
        )
        assert out.error is None


def _search_hit(n: int = 1):
    import uuid as _uuid

    from app.services.vectorstore import SearchHit

    return SearchHit(
        chunk_id=_uuid.UUID(int=n),
        document_id=_uuid.UUID(int=7),
        filename="acme-report.md",
        page=1,
        chunk_index=n,
        heading="## Financial Summary",
        text="Gross margin improved to 62.1% from 58.7%.",
        score=0.8,
        meta={},
        found_by=["q"],
    )

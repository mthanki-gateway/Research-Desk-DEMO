"""The model split, pinned as invariants.

Free-tier request quota is PER MODEL, so which model a call uses is a capacity
decision, not a preference. These tests guard the properties that make the
split work -- they would all have caught a real mistake made while building it.
"""

from app.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(**{"google_api_key": "x", **overrides})


class TestModelSplit:
    def test_nothing_a_user_reads_comes_from_the_rewriter(self):
        """Gemma is confined to query rewriting.

        The rewriter emits short search phrases that are never shown. If it
        ever becomes the workhorse or the answer model again, the swap has been
        undone.
        """
        s = _settings()
        assert "gemma" in s.rewriter_model.lower()
        assert "gemma" not in s.llm_model.lower()
        assert "gemma" not in s.answer_model.lower()
        assert "gemma" not in s.judge_model.lower()

    def test_judge_differs_from_the_answer_model(self):
        """THE Tier 2 invariant.

        Models show a documented self-preference bias grading their own
        output, so a judge pointed at the same id as the answer model measures
        self-preference rather than faithfulness. Both are Gemini now, which
        makes this collision much easier to cause than it used to be.
        """
        s = _settings()
        assert s.judge_model != s.answer_model

    def test_workhorse_has_more_request_budget_than_the_answer_model(self):
        """The reason for the split at all.

        The workhorse runs plan + clarify + critique -- three or more calls per
        turn. The answer model runs one, occasionally two. Reversing these
        would put a four-call sequence on a 5 rpm budget and stall every turn
        in backoff.
        """
        s = _settings()
        workhorse_rpm = s.limits_for(s.llm_model)[0]
        answer_rpm = s.limits_for(s.answer_model)[0]
        assert workhorse_rpm > answer_rpm

    def test_each_model_gets_its_own_limits(self):
        """A shared bucket would throttle everything at the strictest limit."""
        s = _settings()
        seen = {
            s.limits_for(m)
            for m in (s.llm_model, s.answer_model, s.rewriter_model)
        }
        assert len(seen) == 3, "two models resolved to identical limits"

    def test_unknown_model_falls_back_to_the_workhorse_limits(self):
        s = _settings()
        assert s.limits_for("models/something-new") == (
            s.llm_requests_per_minute,
            s.llm_tokens_per_minute,
        )

    def test_limits_are_matched_with_or_without_the_models_prefix(self):
        """`GenAIClient` strips `models/` before naming its limiter, so the
        lookup has to work either way or the client silently gets the default
        budget instead of its own."""
        s = _settings()
        assert s.limits_for(s.answer_model) == s.limits_for(
            s.answer_model.removeprefix("models/")
        )


class TestHistoryBudget:
    def test_whole_history_is_the_default(self):
        assert _settings().history_full is True

    def test_budget_leaves_room_for_retrieved_passages(self):
        """The budget must stay well under the 1,048,576-token window, or
        history could crowd out the passages the answer has to cite."""
        assert _settings().history_max_tokens < 1_000_000 // 2

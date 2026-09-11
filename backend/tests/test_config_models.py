"""The model split, pinned as invariants.

Free-tier request quota is PER MODEL, so which model a call uses is a capacity
decision, not a preference. These tests guard the properties that make the
split work -- they would all have caught a real mistake made while building it.

ASSERTED ON THE PROFILE, NOT ON A CONSTRUCTED `Settings()`.

`Settings` reads the process environment, so building one here tests whatever
the developer's .env happens to say. These tests used to do exactly that and
passed only because docker-compose pinned LLM_MODEL over a stale .env entry;
the moment the profile system replaced that pin, they failed -- not because the
split had changed, but because .env still carried `LLM_MODEL=...gemma...` plus
its 30 rpm / 16K budgets.

That misconfiguration is real and worth surfacing, but a unit test is the wrong
place for it: it makes the suite depend on an untracked local file. The startup
warning in main.py reports it instead (`model_profile_overridden`), and these
tests pin the INTENT, which lives in MODEL_PROFILES.
"""

import pytest

from app.config import MODEL_PROFILES, Settings


def _profile_settings(name: str) -> Settings:
    """A Settings carrying one profile's values, isolated from the environment.

    Every field the profile declares is passed explicitly, so nothing here can
    be moved by an env var.
    """
    return Settings(google_api_key="x", model_profile=name, **MODEL_PROFILES[name])


class TestModelSplit:
    def test_nothing_a_user_reads_comes_from_the_rewriter(self):
        """Gemma is confined to query rewriting ON THE GEMINI PROFILE.

        The rewriter emits short search phrases that are never shown. If it ever
        becomes the workhorse or the answer model there, the swap has been
        undone. The gemma profile is exempt by definition -- it is Gemma
        everywhere, deliberately, to buy the request budget.
        """
        s = _profile_settings("gemini")
        assert "gemma" in s.rewriter_model.lower()
        assert "gemma" not in s.llm_model.lower()
        assert "gemma" not in s.answer_model.lower()
        assert "gemma" not in s.judge_model.lower()

    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_judge_differs_from_the_answer_model(self, name):
        """THE Tier 2 invariant, and it must hold under EVERY profile.

        Models show a documented self-preference bias grading their own output,
        so a judge pointed at the same id as the answer model measures
        self-preference rather than faithfulness. The gemma profile makes this
        easy to break, since one model does every other job.
        """
        s = _profile_settings(name)
        assert s.judge_model != s.answer_model

    def test_workhorse_has_more_request_budget_than_the_answer_model(self):
        """The reason for the split at all.

        The workhorse runs plan + clarify + critique -- three or more calls per
        turn. The answer model runs one, occasionally two. Reversing these would
        put a four-call sequence on a 5 rpm budget and stall every turn in
        backoff.

        Gemini only: on the gemma profile they are the same model and therefore
        the same budget, which is the point of that profile.
        """
        s = _profile_settings("gemini")
        assert s.limits_for(s.llm_model)[0] > s.limits_for(s.answer_model)[0]

    def test_each_model_gets_its_own_limits(self):
        """A shared bucket would throttle everything at the strictest limit."""
        s = _profile_settings("gemini")
        seen = {s.limits_for(m) for m in (s.llm_model, s.answer_model, s.rewriter_model)}
        assert len(seen) == 3, "two models resolved to identical limits"

    def test_roles_sharing_a_model_share_one_budget(self):
        """The gemma profile points several roles at one model, and that is
        CORRECT: the quota is per model, so one limiter covering every role is
        what keeps the app inside it. Two limiters on one id would permit double
        the real quota."""
        s = _profile_settings("gemma")
        assert s.limits_for(s.llm_model) == s.limits_for(s.answer_model)

    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_unknown_model_falls_back_to_the_workhorse_limits(self, name):
        s = _profile_settings(name)
        assert s.limits_for("models/something-new") == (
            s.llm_requests_per_minute,
            s.llm_tokens_per_minute,
        )

    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_limits_are_matched_with_or_without_the_models_prefix(self, name):
        """`GenAIClient` strips `models/` before naming its limiter, so the
        lookup has to work either way or the client silently gets the default
        budget instead of its own."""
        s = _profile_settings(name)
        assert s.limits_for(s.answer_model) == s.limits_for(
            s.answer_model.removeprefix("models/")
        )


class TestHistoryBudget:
    def test_whole_history_is_the_default_on_gemini(self):
        assert _profile_settings("gemini").history_full is True

    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_budget_leaves_room_for_retrieved_passages(self, name):
        """History must never crowd out the passages the answer has to cite.

        On Gemini the ceiling is the 1M window; on Gemma it is the 16K/minute
        budget, which is far tighter. Both are expressed the same way: history
        takes at most half of what one turn can spend.
        """
        s = _profile_settings(name)
        assert s.history_max_tokens < 1_000_000 // 2
        if not s.history_full:
            assert s.history_max_tokens <= s.llm_tokens_per_minute // 2

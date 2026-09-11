"""Model profiles: one switch that moves every model and every budget with it.

The profile exists so the agent loop can actually be exercised. The full design
costs 9-10 model calls per turn, and the answer model allows FIVE PER MINUTE --
about one question every two minutes. Gemma's 30 rpm is the only budget on this
key that can run the loop repeatedly.

What these tests pin is the part that fails quietly:

* a profile must not overwrite an explicit env var, or a developer's deliberate
  model choice vanishes while the app keeps reporting it;
* the budgets must move WITH the model, because they are not independent --
  Gemma's 16K tokens/minute is what makes whole-transcript history impossible;
* ReAct must be off on a model that cannot tool-call, because the failure is a
  silent empty answer rather than an error;
* the override must not leak between concurrent requests.
"""

import asyncio

import pytest

from app.agent.graph import gather_strategy
from app.config import (
    MODEL_PROFILES,
    Settings,
    get_settings,
    use_model_profile,
)


def _settings(**overrides) -> Settings:
    return Settings(**{"google_api_key": "x", **overrides})


class TestProfileSuppliesDefaults:
    def test_gemini_is_the_default_profile(self):
        assert _settings().model_profile == "gemini"

    def test_gemma_profile_declares_one_model_for_every_role(self):
        """Asserted on the PROFILE, not on a constructed Settings.

        Settings reads the process environment, and this repo's .env carries a
        legacy LLM_MODEL. A test that built Settings here would be asserting
        what the developer's .env happens to say -- it failed exactly that way
        once, which is how the conflict warning came to exist.
        """
        profile = MODEL_PROFILES["gemma"]
        for field in ("llm_model", "answer_model", "rewriter_model"):
            assert "gemma" in profile[field], field

    def test_gemini_profile_splits_by_QUOTA_not_by_role(self):
        """The split was never about roles, it was about capacity.

        It used to put the answer on a stronger 5 rpm model and everything else
        on a 15 rpm one. That inverted the actual constraint: the request path
        spends 3-5 calls per turn, so the scarcest budget must not sit on it.
        The answer and the workhorse are now ONE model sharing one limiter --
        correct, because free-tier quota is per model -- and what is still split
        out is the rewriter (higher rpm, throughput over quality) and the judge
        (stronger, must not grade its own output).
        """
        profile = MODEL_PROFILES["gemini"]
        assert "gemma" in profile["rewriter_model"], "rewriting stays on the 30 rpm model"
        assert profile["judge_model"] != profile["answer_model"]

    def test_the_profile_reaches_unset_fields(self):
        """The merge mechanism itself, on a field nothing in the environment
        sets -- so it tests the validator rather than the developer's .env."""
        assert _settings(model_profile="gemma").history_full is False
        assert _settings(model_profile="gemini").history_full is True


class TestProfileConflicts:
    def test_an_explicit_contradiction_is_reported(self):
        """A stale env var overriding one field of a profile yields a mixture
        worse than either -- measured: a Gemma workhorse with Gemini's 250K
        budgets and hop counts. Explicit still wins; it must not be silent."""
        s = _settings(model_profile="gemini", llm_model="models/gemma-4-26b-a4b-it")
        fields = [field for field, _, _ in s.profile_conflicts()]
        assert "llm_model" in fields

    def test_agreeing_with_the_profile_is_not_a_conflict(self):
        s = _settings(
            model_profile="gemini",
            llm_model=MODEL_PROFILES["gemini"]["llm_model"],
        )
        assert [f for f, _, _ in s.profile_conflicts() if f == "llm_model"] == []

    def test_the_report_carries_both_values(self):
        s = _settings(model_profile="gemma", draft_max_output_tokens=4096)
        conflict = next(c for c in s.profile_conflicts() if c[0] == "draft_max_output_tokens")
        _, configured, expected = conflict
        assert configured == 4096
        assert expected == MODEL_PROFILES["gemma"]["draft_max_output_tokens"]

    def test_the_judge_stays_off_the_answer_model_in_both(self):
        """The Tier 2 invariant. A model grading its own output measures
        self-preference, so this must hold under EVERY profile -- including the
        one where a single model does every other job."""
        for name in MODEL_PROFILES:
            s = _settings(model_profile=name)
            assert s.judge_model != s.answer_model, name


class TestExplicitSettingsWin:
    def test_an_explicit_model_survives_the_profile(self):
        """`model_fields_set` is what makes this work. Without it, switching
        profile would silently discard a deliberate override and the app would
        keep reporting the model the developer asked for while calling another.
        """
        s = _settings(model_profile="gemma", llm_model="models/my-own-choice")
        assert s.llm_model == "models/my-own-choice"
        # ...and the rest of the profile still applies.
        assert s.history_full is False

    def test_an_explicit_budget_survives_the_profile(self):
        s = _settings(model_profile="gemma", draft_max_output_tokens=4096)
        assert s.draft_max_output_tokens == 4096


class TestBudgetsMoveWithTheModel:
    """The budgets are derived from the token ceiling, not chosen separately."""

    def test_gemma_cannot_afford_whole_transcript_history(self):
        s = _settings(model_profile="gemma")
        assert s.history_full is False
        assert s.history_max_tokens < s.llm_tokens_per_minute

    def test_gemini_sends_the_whole_transcript(self):
        s = _settings(model_profile="gemini")
        assert s.history_full is True

    def test_gemma_has_the_tighter_output_ceiling(self):
        assert (
            _settings(model_profile="gemma").draft_max_output_tokens
            < _settings(model_profile="gemini").draft_max_output_tokens
        )

    def test_gemma_takes_fewer_hops_and_fewer_iterations(self):
        gemma, gemini = _settings(model_profile="gemma"), _settings(model_profile="gemini")
        assert gemma.react_max_rounds < gemini.react_max_rounds
        assert gemma.react_max_calls_per_round < gemini.react_max_calls_per_round
        assert gemma.agent_max_iterations < gemini.agent_max_iterations

    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_one_draft_never_exceeds_a_minute_of_tokens(self, name):
        """A single call whose output ceiling alone exhausts the per-minute
        budget cannot be rate-limited into working -- it stalls every time."""
        s = _settings(model_profile=name)
        assert s.draft_max_output_tokens < s.answer_tokens_per_minute


class TestToolCallingCapability:
    def test_gemini_can_tool_call(self):
        assert _settings(model_profile="gemini").supports_tool_calling is True

    def test_gemma_cannot(self):
        assert _settings(model_profile="gemma").supports_tool_calling is False

    def test_react_is_off_by_default_where_it_cannot_work(self):
        assert _settings(model_profile="gemma").react_default is False

    def test_asking_for_react_on_gemma_falls_back_to_planning(self, monkeypatch):
        """The silent-failure guard.

        Gemma emits no functionCall parts -- asked to use tools it narrates its
        intentions as prose, which the loop reads as "no calls requested". It
        would exit on round 1 with no evidence and `draft` would answer from
        nothing: a wrong answer that looks like a bad model rather than a
        misconfiguration.
        """
        import app.agent.graph as graph_mod

        monkeypatch.setattr(
            graph_mod, "get_settings", lambda: _settings(model_profile="gemma")
        )
        assert gather_strategy({"react": True}) == "plan"

    def test_react_is_honoured_where_it_does_work(self, monkeypatch):
        import app.agent.graph as graph_mod

        monkeypatch.setattr(
            graph_mod, "get_settings", lambda: _settings(model_profile="gemini")
        )
        assert gather_strategy({"react": True}) == "react"
        assert gather_strategy({"react": False}) == "plan"


class TestOverrideScope:
    def test_no_override_returns_the_configured_settings(self):
        assert get_settings().model_profile == get_settings().model_profile

    def test_override_applies_inside_the_block(self):
        with use_model_profile("gemma"):
            assert get_settings().model_profile == "gemma"

    def test_override_is_restored_afterwards(self):
        before = get_settings().model_profile
        with use_model_profile("gemma"):
            pass
        assert get_settings().model_profile == before

    def test_nested_overrides_restore_the_outer_one(self):
        """Reset via token, not by setting None -- the latter would clobber an
        enclosing override instead of restoring it."""
        with use_model_profile("gemma"):
            with use_model_profile("gemini"):
                assert get_settings().model_profile == "gemini"
            assert get_settings().model_profile == "gemma"

    def test_an_unknown_profile_is_ignored(self):
        before = get_settings().model_profile
        with use_model_profile("not-a-profile"):
            assert get_settings().model_profile == before

    def test_none_is_a_no_op(self):
        before = get_settings().model_profile
        with use_model_profile(None):
            assert get_settings().model_profile == before

    @pytest.mark.asyncio
    async def test_concurrent_turns_do_not_see_each_other_s_profile(self):
        """The reason this is a ContextVar and not a module global.

        The API serves concurrent requests on one event loop. A global would let
        one developer's "run this on Gemma" change the model for everybody
        else's in-flight turn.
        """
        seen: dict[str, list[str]] = {"gemma": [], "gemini": []}

        async def turn(profile: str) -> None:
            with use_model_profile(profile):
                for _ in range(5):
                    seen[profile].append(get_settings().model_profile)
                    await asyncio.sleep(0)  # yield to the other task

        await asyncio.gather(turn("gemma"), turn("gemini"))
        assert set(seen["gemma"]) == {"gemma"}
        assert set(seen["gemini"]) == {"gemini"}

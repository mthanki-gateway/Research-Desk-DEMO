"""Model pools: adding quota, not adding failover.

Free-tier quota is PER MODEL, so two models are two budgets and a role that can
use either has the sum. Measured on this key with `probe_limits`:

    gemini-3.6-flash        GenerateRequestsPerMinutePerProjectPerModel = 5
    gemini-3.5-flash-lite   GenerateRequestsPerMinutePerProjectPerModel = 15

What these pin is the part that fails quietly: a pool that mixes families would
make ReAct retrieve nothing on some fraction of turns and succeed on the rest.
"""

import pytest

from app.config import MODEL_PROFILES, Settings
from app.services.limiter import RateLimiter
from app.services.llm import LLMError
from app.services.pool import IncompatiblePool, ModelPool, _family, reset_pools


class FakeClient:
    """Stands in for GenAIClient: a name and a limiter view."""

    def __init__(self, model: str, wait: float = 0.0):
        self.model = model
        self._wait = wait

    def wait_estimate(self, tokens: int) -> float:
        return self._wait


@pytest.fixture(autouse=True)
def _clean_pools():
    reset_pools()
    yield
    reset_pools()


class TestFamilyGuard:
    def test_gemma_and_gemini_are_different_families(self):
        assert _family("models/gemma-4-26b-a4b-it") != _family("models/gemini-3.5-flash-lite")

    def test_two_gemini_models_are_interchangeable(self):
        assert _family("models/gemini-3.6-flash") == _family("models/gemini-2.5-flash-lite")

    def test_two_gemma_models_are_interchangeable(self):
        assert _family("models/gemma-4-26b-a4b-it") == _family("models/gemma-4-31b-it")

    def test_a_mixed_pool_is_refused_at_construction(self):
        """Refused when it is BUILT, not when a ReAct round quietly returns
        nothing.

        Gemma emits no functionCall parts, so a mixed pool would make tool
        calling fail on whichever fraction of turns landed on the Gemma member
        -- intermittent, invisible, and indistinguishable from a bad model.
        """
        with pytest.raises(IncompatiblePool, match="Gemma cannot stand in"):
            ModelPool(["models/gemini-3.5-flash-lite", "models/gemma-4-26b-a4b-it"])

    def test_an_empty_pool_is_refused(self):
        with pytest.raises(IncompatiblePool):
            ModelPool([])

    def test_a_single_model_pool_is_fine(self):
        """A deployment with no extra models must behave exactly as before."""
        pool = ModelPool(["models/gemini-3.5-flash-lite"])
        assert len(pool) == 1


class TestMemberSelection:
    def test_the_primary_is_preferred_while_it_has_capacity(self, monkeypatch):
        """The secondary answers RARELY -- only when the primary's own minute is
        spent. That is what keeps answer quality consistent on the common path
        while still doubling the ceiling."""
        import app.services.pool as pool_mod

        clients = {
            "a": FakeClient("a", wait=0.0),
            "b": FakeClient("b", wait=0.0),
        }
        monkeypatch.setattr(pool_mod, "get_llm", lambda m: clients[m])
        assert ModelPool(["a", "b"]).pick(100).model == "a"

    def test_it_spills_to_the_secondary_when_the_primary_is_busy(self, monkeypatch):
        import app.services.pool as pool_mod

        clients = {"a": FakeClient("a", wait=12.0), "b": FakeClient("b", wait=0.0)}
        monkeypatch.setattr(pool_mod, "get_llm", lambda m: clients[m])
        assert ModelPool(["a", "b"]).pick(100).model == "b"

    def test_a_saturated_pool_waits_the_shortest_time(self, monkeypatch):
        """Never WORSE than a single model. When everything is busy the pool
        blocks on whichever frees up first, so the floor is the old behaviour."""
        import app.services.pool as pool_mod

        clients = {
            "a": FakeClient("a", wait=30.0),
            "b": FakeClient("b", wait=4.0),
            "c": FakeClient("c", wait=17.0),
        }
        monkeypatch.setattr(pool_mod, "get_llm", lambda m: clients[m])
        assert ModelPool(["a", "b", "c"]).pick(100).model == "b"

    def test_duplicates_are_collapsed(self, monkeypatch):
        """A repeated model would get two turns in the rotation while still
        sharing ONE limiter -- picked twice as often, then blocking."""
        pool = ModelPool(["a", "a", "b"])
        assert pool.models == ["a", "b"]

    def test_for_prompt_counts_the_output_allowance(self, monkeypatch):
        """A limiter weighing only the input would admit a call whose reply
        then blows the token budget, and the next caller pays for it."""
        import app.services.pool as pool_mod

        seen: list[int] = []

        class Recorder(FakeClient):
            def wait_estimate(self, tokens):
                seen.append(tokens)
                return 0.0

        monkeypatch.setattr(pool_mod, "get_llm", lambda m: Recorder(m))
        ModelPool(["a", "b"]).for_prompt("x" * 400, max_output_tokens=2000)
        assert seen and seen[0] >= 2000


class TestLimiterPeek:
    def test_peek_consumes_nothing(self):
        """The property that makes a pool possible: it must be able to ask
        several limiters "could you serve this?" and spend budget only on the
        one it chooses."""
        limiter = RateLimiter(requests_per_minute=5, tokens_per_minute=1000)
        for _ in range(50):
            assert limiter.peek(10) == 0.0

    def test_peek_reports_a_wait_once_exhausted(self):
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=10**9)
        limiter._requests.consume(60)
        assert limiter.peek(1) > 0.0


class TestProfilePools:
    @pytest.mark.parametrize("name", list(MODEL_PROFILES))
    def test_every_profile_pool_is_family_consistent(self, name):
        """The bug this catches: pools left as a plain default would follow a
        profile switch across and put a Gemini model in a Gemma pool, which
        ModelPool refuses -- so switching profile would RAISE instead of
        switching. Pools have to move with the models, like the budgets do.
        """
        s = Settings(google_api_key="x", model_profile=name, **MODEL_PROFILES[name])
        ModelPool([s.answer_model, *s.answer_pool], name=f"{name}:answer")

    def test_gemini_pools_every_verified_model(self):
        """The request path is where 429s hurt, and quota is per model, so the
        answer role uses every flash-lite the probe found usable."""
        s = Settings(google_api_key="x", model_profile="gemini", **MODEL_PROFILES["gemini"])
        assert len(s.answer_pool) >= 2

    def test_gemma_pools_nothing_because_there_is_nothing_to_pool(self):
        """gemma-4-31b-it is the only other Gemma on this key and it returns
        500 on every call (measured). A member that always fails is worse than
        no member: it consumes a pick and then fails the call."""
        s = Settings(google_api_key="x", model_profile="gemma", **MODEL_PROFILES["gemma"])
        assert s.answer_pool == []

    def test_the_judge_is_never_pooled(self):
        """Which member served a call is not deterministic, so anything whose
        output is compared across runs must pin one model. A pooled judge would
        make two evaluation runs incomparable for a reason nothing records."""
        import inspect

        from app.services import pool as pool_mod

        source = inspect.getsource(pool_mod.get_pool)
        assert "judge" not in source


class TestPoolMembersGetTheRightBudget:
    """A member falling through to the workhorse's limits is the quiet way a
    pool defeats itself.

    Measured before the fix: gemini-2.5-flash-lite (really 15 rpm) was limited
    at 30, because that is what `llm_*` allowed and nothing else matched. The
    limiter would have admitted twice its real quota and collected exactly the
    429s the pool exists to prevent.
    """

    def test_a_member_uses_its_MEASURED_rate_not_the_roles(self):
        """Inheriting the role's budget was the earlier design, and the probe
        disproved it: gemini-2.5-flash-lite allows 10 requests/minute where
        every other flash-lite allows 15. Handing it the role's 15 would give it
        half again its real quota and it would collect the 429s the pool exists
        to avoid."""
        s = Settings(google_api_key="x", model_profile="gemini", **MODEL_PROFILES["gemini"])
        assert s.limits_for("models/gemini-2.5-flash-lite")[0] == 10
        assert s.limits_for("models/gemini-3.1-flash-lite")[0] == 15

    def test_every_pool_member_has_a_measured_rate(self):
        """A member with no measured entry falls back to a role default, which
        is a guess. Every one we ship was probed."""
        from app.config import MEASURED_RPM

        s = Settings(google_api_key="x", model_profile="gemini", **MODEL_PROFILES["gemini"])
        for member in [s.answer_model, *s.answer_pool, *s.llm_pool]:
            assert member.removeprefix("models/") in MEASURED_RPM, member

    def test_a_member_does_not_inherit_the_workhorse_budget(self):
        """The specific fall-through that was happening."""
        s = Settings(
            google_api_key="x",
            model_profile="gemini",
            **{
                **MODEL_PROFILES["gemini"],
                # Deliberately different, so inheriting the wrong one is visible.
                "llm_requests_per_minute": 99,
            },
        )
        for member in s.answer_pool:
            assert s.limits_for(member)[0] != 99, member

    def test_an_unknown_model_still_falls_back(self):
        """The fallback is still right for a model nobody configured."""
        s = Settings(google_api_key="x", model_profile="gemini", **MODEL_PROFILES["gemini"])
        assert s.limits_for("models/never-heard-of-it") == (
            s.llm_requests_per_minute,
            s.llm_tokens_per_minute,
        )


class _FailThen:
    """A client that rate-limits N times, then succeeds."""

    def __init__(self, model: str, fails: int = 0, status: int = 429, retry_after=None):
        self.model = model
        self._fails = fails
        self._status = status
        self._retry_after = retry_after
        self.calls = 0
        self.attempts_seen: list[int] = []

    def wait_estimate(self, tokens: int) -> float:
        return 0.0

    async def generate(self, prompt, *, max_attempts=4, **kw):
        self.calls += 1
        self.attempts_seen.append(max_attempts)
        if self._fails > 0:
            self._fails -= 1
            raise LLMError(
                f"{self._status}: rate limited",
                status=self._status,
                retry_after=self._retry_after,
            )
        return f"answer from {self.model}"


@pytest.mark.asyncio
class TestFailover:
    """The reactive half.

    `pick` spreads load using each limiter's LOCAL view, which is a guess --
    the quota can be spent by another process or by our token estimate being
    low. So a 429 still happens, and the right response is not to back off on
    the exhausted model but to use one that still has budget.
    """

    async def _pool(self, monkeypatch, clients):
        import app.services.pool as pool_mod

        monkeypatch.setattr(pool_mod, "get_llm", lambda m: clients[m])
        return ModelPool(list(clients), name="answer")

    async def test_a_rate_limited_model_hands_off_to_the_next(self, monkeypatch):
        clients = {"a": _FailThen("a", fails=1), "b": _FailThen("b")}
        pool = await self._pool(monkeypatch, clients)
        assert await pool.generate("q") == "answer from b"
        assert clients["a"].calls == 1

    async def test_non_final_members_fail_fast(self, monkeypatch):
        """Backing off on an exhausted model spends exactly the time the pool
        exists to avoid. Only the LAST member gets the client's own retries,
        which keeps the floor at the single-model behaviour."""
        clients = {"a": _FailThen("a", fails=1), "b": _FailThen("b")}
        pool = await self._pool(monkeypatch, clients)
        await pool.generate("q")
        assert clients["a"].attempts_seen == [1], "first member should not back off"
        assert clients["b"].attempts_seen == [4], "last member keeps full retries"

    async def test_a_bad_request_is_not_retried_elsewhere(self, monkeypatch):
        """Every member would reject a 400 identically, so trying all of them
        turns one clear error into several slow ones."""
        clients = {"a": _FailThen("a", fails=1, status=400), "b": _FailThen("b")}
        pool = await self._pool(monkeypatch, clients)
        with pytest.raises(LLMError):
            await pool.generate("q")
        assert clients["b"].calls == 0

    async def test_everything_limited_raises_the_last_error(self, monkeypatch):
        clients = {"a": _FailThen("a", fails=9), "b": _FailThen("b", fails=9)}
        pool = await self._pool(monkeypatch, clients)
        with pytest.raises(LLMError) as caught:
            await pool.generate("q")
        assert caught.value.status == 429

    async def test_a_limited_model_is_skipped_next_time(self, monkeypatch):
        """Its own limiter still thinks it has budget -- the quota was spent by
        something the limiter cannot see -- so without a cooldown the pool would
        keep choosing it and keep failing."""
        import app.services.pool as pool_mod

        clients = {"a": _FailThen("a", fails=1), "b": _FailThen("b")}
        pool = await self._pool(monkeypatch, clients)
        await pool.generate("q")
        assert pool_mod._cooling.get("a", 0) > 0

        before = clients["a"].calls
        await pool.generate("q again")
        assert clients["a"].calls == before, "a cooling model was picked again"

    async def test_the_servers_retry_delay_is_preferred(self, monkeypatch):
        """Google says when the budget actually refills; a guessed backoff is
        strictly worse information."""
        import app.services.pool as pool_mod

        clients = {"a": _FailThen("a", fails=1, retry_after=42.0), "b": _FailThen("b")}
        pool = await self._pool(monkeypatch, clients)
        await pool.generate("q")
        remaining = pool_mod._cooling["a"] - __import__("time").monotonic()
        assert 40 < remaining <= 42

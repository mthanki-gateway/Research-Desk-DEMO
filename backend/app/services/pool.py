"""A pool of interchangeable models for one role.

WHY THIS WORKS AT ALL: FREE-TIER QUOTA IS PER MODEL.

Five requests per minute on `gemini-3.6-flash` does not mean five per minute on
the key -- `gemini-3.5-flash-lite` has its own separate allowance, and so does
every other model. Two models are therefore two budgets, and a role that can
use either has the sum of both. That is the entire idea: not failover for
reliability, but ADDITION for throughput.

HOW A MEMBER IS CHOSEN

Least-loaded-that-can-serve-now, not always-first. Each client owns a limiter
with a live view of its own budget, so the pool asks each one "could you take
this request without waiting?" and uses the first that says yes. Always
starting at the head of the list would saturate member one, spill to member
two only under pressure, and leave the rest idle -- which spreads nothing.

When every member would have to wait, the pool picks the one that would wait
LEAST and lets it block. That keeps the existing behaviour as the floor: a
fully saturated pool is exactly as slow as a single model, never slower.

WHAT MAY NOT GO IN A POOL TOGETHER

Members must be INTERCHANGEABLE, which is a stronger requirement than "both
work". Gemma and Gemini are not: Gemma emits no functionCall parts, so a ReAct
round that lands on a Gemma member silently retrieves nothing. Families are
checked rather than trusted -- see `_family`.

Reproducibility is the other constraint. Which member served a call is not
deterministic, so anything whose output is compared across runs -- the
evaluation harness, the judge -- must pin a single model rather than pool.
"""

from __future__ import annotations

import time

import structlog

from app.config import get_settings
from app.services.limiter import estimate_tokens
from app.services.llm import GenAIClient, LLMError, get_llm

log = structlog.get_logger()


class IncompatiblePool(ValueError):
    """Members that are not interchangeable were configured together."""


def _family(model: str) -> str:
    """Which models can stand in for each other.

    Deliberately coarse: the only distinction that has ever mattered here is
    Gemma versus Gemini, and it matters absolutely. Gemma cannot tool-call, so a
    mixed pool would make ReAct fail on some fraction of turns and succeed on
    the rest -- an intermittent, invisible failure that looks like a bad model
    rather than a bad configuration.
    """
    name = model.removeprefix("models/")
    return "gemma" if name.startswith("gemma") else "gemini"


class ModelPool:
    """Several models serving one role, sharing the load across their quotas."""

    __slots__ = ("_models", "_name")

    def __init__(self, models: list[str], *, name: str = "") -> None:
        if not models:
            raise IncompatiblePool("a pool needs at least one model")

        families = {_family(m) for m in models}
        if len(families) > 1:
            # Fails at construction, not at the first ReAct round that quietly
            # returns nothing.
            raise IncompatiblePool(
                f"pool {name!r} mixes {sorted(families)}: Gemma cannot stand in "
                "for Gemini (no native tool calling)"
            )

        # Order-preserving dedupe. A repeated model would get two entries and
        # therefore two turns in the rotation, quietly weighting it double while
        # still sharing ONE limiter -- so it would be picked more often and then
        # block.
        seen: set[str] = set()
        self._models = [
            m for m in models if not (m in seen or seen.add(m))
        ]
        self._name = name or "pool"

    def __len__(self) -> int:
        return len(self._models)

    @property
    def models(self) -> list[str]:
        return list(self._models)

    def _ordered(self, tokens: int) -> list[GenAIClient]:
        """Members best-first: ready ones in configured order, then by wait.

        The order the failover walks. Cooling members sink to the bottom rather
        than being removed, so an all-cooling pool still has something to try
        instead of raising.
        """
        clients = [get_llm(m) for m in self._models]
        now = time.monotonic()

        def rank(client: GenAIClient) -> tuple[float, float]:
            cooling = max(0.0, _cooling.get(client.model, 0.0) - now)
            return (cooling, client.wait_estimate(tokens))

        return sorted(clients, key=rank)

    def pick(self, tokens: int) -> GenAIClient:
        """The member that can serve `tokens` soonest.

        Returns a client rather than a name so the caller cannot accidentally
        use a different one than was scheduled.
        """
        clients = [get_llm(m) for m in self._models]
        if len(clients) == 1:
            return clients[0]

        now = time.monotonic()
        best: GenAIClient | None = None
        best_wait = float("inf")
        for client in clients:
            # A model that just returned 429 is skipped until its own
            # retryDelay has passed. Its local limiter thinks it has budget --
            # the quota was spent by something the limiter cannot see, another
            # process or another key user -- so without this the pool would keep
            # choosing it and keep failing.
            if _cooling.get(client.model, 0.0) > now:
                continue
            wait = client.wait_estimate(tokens)
            if wait <= 0.0:
                return client
            if wait < best_wait:
                best, best_wait = client, wait

        # Everything is saturated. Block on whichever frees up first, which is
        # never worse than a single model would have been.
        log.info(
            "pool_saturated",
            pool=self._name,
            members=len(clients),
            min_wait_s=round(best_wait, 1),
        )
        return best or clients[0]

    def for_prompt(
        self, *parts: str | None, max_output_tokens: int = 1024
    ) -> GenAIClient:
        """Pick a member sized for this prompt.

        The output allowance counts too. A limiter that only weighed the input
        would admit a call whose reply then blows the token budget, and the
        next caller pays for it -- which is the failure the dual bucket exists
        to prevent.
        """
        cost = sum(estimate_tokens(p) for p in parts if p) + max_output_tokens
        return self.pick(cost)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate, moving to another model the moment one is rate limited.

        THIS IS THE REACTIVE HALF. `pick` spreads load using each limiter's
        local view, which is a guess: the quota can be spent by another process,
        another key user, or simply by our token estimate being low. So a 429
        still happens, and when it does the right response is not to back off on
        the exhausted model -- it is to use one that still has budget.

        Each member is tried with max_attempts=1 so it fails FAST and the pool
        moves on. Only the last member falls back to the client's own backoff,
        which is the behaviour a single model always had: the floor is never
        worse than not pooling.

        Non-rate-limit failures are NOT retried elsewhere. A 400 is a bad
        request and every member would reject it identically, so trying four
        models would turn one clear error into four slow ones.
        """
        cost = (
            estimate_tokens(prompt)
            + estimate_tokens(kwargs.get("system") or "")
            + kwargs.get("max_output_tokens", 1024)
        )
        members = self._ordered(cost)

        last: LLMError | None = None
        for i, client in enumerate(members):
            final = i == len(members) - 1
            try:
                return await client.generate(
                    prompt, **kwargs, max_attempts=4 if final else 1
                )
            except LLMError as exc:
                if exc.status not in (429, 503):
                    raise
                last = exc
                _mark_cooling(client.model, exc.retry_after)
                if not final:
                    log.info(
                        "pool_failover",
                        pool=self._name,
                        from_model=client.model,
                        to_model=members[i + 1].model,
                        status=exc.status,
                    )
        raise last or LLMError(f"pool {self._name!r} had no usable member")


# model -> monotonic time it becomes usable again.
#
# Module level rather than per pool, because the quota is per MODEL: if the
# answer pool exhausts flash-lite, the workhorse pool must know too. Keying it
# on the pool would let each one rediscover the same exhaustion separately.
_cooling: dict[str, float] = {}

# Used when the 429 carries no RetryInfo. A per-minute bucket refills
# continuously, so a short pause is usually enough; too long would park a model
# that recovered seconds later.
_DEFAULT_COOLDOWN_S = 20.0


def _mark_cooling(model: str, retry_after: float | None) -> None:
    delay = retry_after if retry_after and retry_after > 0 else _DEFAULT_COOLDOWN_S
    _cooling[model] = time.monotonic() + delay
    log.info("model_cooling", model=model, seconds=round(delay, 1))


_pools: dict[str, ModelPool] = {}


def get_pool(role: str) -> ModelPool:
    """The pool for a role: "answer", "llm", "rewriter".

    Cached like the clients are, because the pool holds no state of its own but
    rebuilding it on every call would re-read settings for nothing.
    """
    settings = get_settings()
    primary = {
        "answer": settings.answer_model,
        "llm": settings.llm_model,
        "rewriter": settings.rewriter_model,
    }[role]
    extra = {
        "answer": settings.answer_pool,
        "llm": settings.llm_pool,
        "rewriter": [],
    }[role]

    key = f"{role}:{primary}:{','.join(extra)}"
    if key not in _pools:
        # The primary always leads, so a single-model deployment behaves
        # exactly as before and the "which model answered" question has an
        # obvious default answer.
        pool = ModelPool([primary, *extra], name=role)
        _pools[key] = pool
        if len(pool) > 1:
            log.info("model_pool_ready", role=role, models=pool.models)
    return _pools[key]


def reset_pools() -> None:
    """Drop cached pools and cooldowns. For tests and for a profile switch."""
    _pools.clear()
    _cooling.clear()

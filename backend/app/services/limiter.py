"""Dual token-bucket limiter: requests/minute AND tokens/minute.

Why both: for embeddings, batching to 100 items fixes RPM (one call instead of
100) but a full batch is ~22.5K tokens against a 30K TPM ceiling. So requests
stop being the constraint and tokens become it. A limiter that only counts
requests would sail past the token ceiling and collect 429s.

Buckets refill continuously (limit/60 per second) rather than resetting on a
minute boundary, which avoids the thundering-herd effect of a fixed window.
"""

import asyncio
import time

import structlog

log = structlog.get_logger()


class _Bucket:
    def __init__(self, capacity: int) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = capacity / 60.0
        self.updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec)
        self.updated = now

    def wait_time(self, amount: float) -> float:
        """Seconds until `amount` is available. 0 means go now."""
        self._refill()
        if self.tokens >= amount:
            return 0.0
        return (amount - self.tokens) / self.refill_per_sec

    def consume(self, amount: float) -> None:
        self._refill()
        self.tokens -= amount


class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int, name: str = "") -> None:
        self._requests = _Bucket(requests_per_minute)
        self._tokens = _Bucket(tokens_per_minute)
        self._lock = asyncio.Lock()
        self._name = name

    @property
    def token_capacity(self) -> int:
        return int(self._tokens.capacity)

    def peek(self, tokens: int) -> float:
        """Seconds this limiter would make a caller wait. 0 means go now.

        READ-ONLY: it refills the buckets (which is just advancing the clock)
        but consumes nothing. That is what lets a pool ask several limiters
        "could you serve this?" and use only the one it picks -- asking with
        `acquire` would spend a request on every member it merely considered.

        Taken without the lock, so the answer is a snapshot that another task
        can invalidate before the caller acts on it. That is acceptable here:
        the pool uses it to CHOOSE, and the chosen client still goes through
        `acquire`, which is authoritative. A stale peek costs a slightly worse
        choice, never an exceeded budget.
        """
        tokens = min(tokens, self.token_capacity)
        return max(self._requests.wait_time(1), self._tokens.wait_time(tokens))

    async def acquire(self, tokens: int) -> None:
        """Block until one request of `tokens` size fits in both budgets."""
        # Clamp rather than deadlock: a single item larger than the whole
        # per-minute budget can never fit, so let it through after a full
        # refill window and let the API reject it if it's genuinely too big.
        tokens = min(tokens, self.token_capacity)

        while True:
            async with self._lock:
                delay = max(self._requests.wait_time(1), self._tokens.wait_time(tokens))
                if delay <= 0:
                    self._requests.consume(1)
                    self._tokens.consume(tokens)
                    return
            # Sleep outside the lock so other callers can still be admitted.
            log.debug("rate_limited", limiter=self._name, sleep_s=round(delay, 2), tokens=tokens)
            await asyncio.sleep(delay)


def estimate_tokens(text: str) -> int:
    """~4 chars per token.

    Deliberately a heuristic: Gemma's real tokenizer isn't exposed locally, and
    Google's countTokens endpoint would spend a request from the very quota
    we're trying to protect. We round up, so we under-use the budget rather
    than overshoot it.
    """
    return max(1, -(-len(text) // 4))

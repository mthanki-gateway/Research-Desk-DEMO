"""Measure each model's real request-per-minute quota.

WHY THIS EXISTS: the API DOES NOT REPORT RATE LIMITS.

`GET /models/{name}` returns `inputTokenLimit` and `outputTokenLimit` -- token
sizes, not rates -- and nothing about requests per minute. `ListModels` is the
same metadata in bulk. There is no quota endpoint on an AI Studio key, so a
model's rpm cannot be looked up; it can only be OBSERVED.

The authoritative number is in the 429 body. Google returns a QuotaFailure
detail carrying `quotaId` and `quotaValue`, which is the actual configured
limit rather than an inference from how many calls succeeded. So this
deliberately trips the limit once and reads the answer out of the error.

    docker compose exec api python -m app.scripts.probe_limits
    docker compose exec api python -m app.scripts.probe_limits --model gemini-3.6-flash

COST, STATED PLAINLY: this SPENDS a minute of quota on every model it probes,
because tripping the limit is the measurement. Do not run it while something
else depends on that quota, and do not run it in a loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

from app.config import get_settings

BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# The families worth knowing about. Kept explicit rather than derived from
# ListModels: that returns image, TTS, transcription and robotics models too,
# and probing those costs quota to learn nothing.
DEFAULT_CANDIDATES = [
    # flash
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    # flash-lite
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    # pro -- included to MEASURE, not because they are likely to be used. Pro
    # tiers are typically a couple of requests per minute, which is worse than
    # the model this was moved off.
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    # aliases. Measured for completeness; never shipped as a pinned model,
    # because a moving target cannot be what an evaluation is reported against.
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    # gemma
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


async def _call(client: httpx.AsyncClient, model: str, key: str, schema: bool):
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with ok=true."}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16},
    }
    if schema:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = SCHEMA
    return await client.post(
        f"{BASE}/{model}:generateContent", params={"key": key}, json=body
    )


def _quota_from_429(response: httpx.Response) -> dict:
    """The configured limit, straight from the error rather than inferred."""
    found: dict = {}
    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return found
    for detail in details:
        for violation in detail.get("violations", []):
            qid = violation.get("quotaId") or "?"
            found[qid] = violation.get("quotaValue")
        if "RetryInfo" in detail.get("@type", ""):
            found["retryDelay"] = detail.get("retryDelay")
    return found


def _rpm_from_quota(quota: dict) -> int | None:
    """The requests-per-minute number out of a QuotaFailure, if present."""
    for key, value in quota.items():
        if "RequestsPerMinute" in key:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


async def probe(client: httpx.AsyncClient, model: str, key: str, burst: int) -> dict:
    out: dict = {"model": model}
    settings = get_settings()

    first = await _call(client, model, key, schema=True)
    out["status"] = first.status_code
    if first.status_code == 429:
        # Two very different situations arrive as the same status code, and
        # telling them apart is the point of reading quotaValue rather than
        # counting successes:
        #
        #   quota 0   -- the model is NOT AVAILABLE on this key at all. Some
        #                AI Studio models are published with a 0/0 free-tier
        #                limit. Putting one in a pool is worse than leaving it
        #                out: it consumes a pick and then fails the call.
        #   quota > 0 -- a real allowance that something has already spent.
        quota = _quota_from_429(first)
        out["quota"] = quota
        rpm = _rpm_from_quota(quota)
        out["rpm"] = rpm
        out["usable"] = bool(rpm)
        out["note"] = (
            "quota is ZERO -- not available on this key"
            if rpm == 0
            else "already at limit; quota read from the error"
        )
        return out
    if first.status_code != 200:
        # 503 is the other disqualifier: the model exists and is published but
        # is not serving. A pool member that 503s spends a pick and fails.
        out["error"] = first.text[:110].replace("\n", " ")
        out["usable"] = False
        return out

    out["schema_ok"] = '"ok"' in first.text or "ok" in first.text

    started = time.perf_counter()
    results = await asyncio.gather(
        *(_call(client, model, key, schema=False) for _ in range(burst)),
        return_exceptions=True,
    )
    ok = sum(1 for r in results if getattr(r, "status_code", None) == 200)
    limited = [r for r in results if getattr(r, "status_code", None) == 429]

    # +1 for the schema check above, which also spent a request.
    out["succeeded"] = ok + 1
    out["rate_limited"] = len(limited)
    out["seconds"] = round(time.perf_counter() - started, 1)
    if limited:
        out["quota"] = _quota_from_429(limited[0])
        out["rpm"] = _rpm_from_quota(out["quota"])
    else:
        # Never tripped, so the observed count is a LOWER BOUND rather than the
        # limit. Recorded as such -- reporting it as the quota would understate
        # a generous model.
        out["rpm"] = None
        out["note"] = f"no 429 within {burst + 1} requests; limit is at least that"

    # Usable means: reachable, honours responseSchema, and has a non-zero
    # allowance. All three are required for a pool member -- one that fails any
    # of them consumes a pick and then fails the call it was picked for.
    out["usable"] = bool(out.get("schema_ok")) and out.get("rpm") != 0
    out["configured_rpm"] = settings.limits_for(model)[0]
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", default=[], help="probe only these (repeatable)"
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=20,
        help="requests to fire at once. Must exceed the expected limit to trip it",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=62,
        help="seconds between models, so one probe does not poison the next",
    )
    args = parser.parse_args()

    key = get_settings().google_api_key
    if not key:
        print("GOOGLE_API_KEY is not set", file=sys.stderr)
        return 1

    models = args.model or DEFAULT_CANDIDATES
    print(f"Probing {len(models)} model(s). This SPENDS quota on each.\n", file=sys.stderr)

    async with httpx.AsyncClient(timeout=90) as client:
        for i, model in enumerate(models):
            result = await probe(client, model, key, args.burst)
            print(json.dumps(result))
            sys.stdout.flush()
            if i < len(models) - 1:
                await asyncio.sleep(args.cooldown)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""Gemma client.

Gemma 4 does NOT support native function-calling -- given tool declarations it
narrates what it would do in prose. It DOES honour generationConfig.
responseSchema, so every structured step in this app goes through a JSON schema
instead of a tool call. Never use responseMimeType without a schema: Gemma
emits its own reasoning trace instead of the object.

Shares no quota with embeddings. Gemma is 30 RPM / 16K TPM on its own budget,
so it gets its own limiter.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.services.limiter import RateLimiter, estimate_tokens

log = structlog.get_logger()

GENAI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    pass


class GemmaClient:
    def __init__(self) -> None:
        s = get_settings()
        if not s.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required")

        self._model = s.llm_model.removeprefix("models/")
        self._client = httpx.AsyncClient(
            base_url=GENAI_BASE,
            timeout=httpx.Timeout(180.0),
            headers={"x-goog-api-key": s.google_api_key},
        )
        self._limiter = RateLimiter(
            requests_per_minute=s.llm_requests_per_minute,
            tokens_per_minute=s.llm_tokens_per_minute,
            name="llm",
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> str:
        """One completion. Returns raw text (JSON text when `schema` is given)."""
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                # Gemma 4 is prone to repetition loops -- it once emitted
                # "way's actually" a dozen times until it hit the token cap,
                # producing truncated JSON. The API exposes no repetition
                # penalty, so nucleus sampling plus a low temperature is the
                # available defence.
                "topP": top_p,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = schema

        # Budget both directions: the prompt we send and the tokens we allow
        # back, since TPM counts both.
        cost = estimate_tokens(prompt) + estimate_tokens(system or "") + max_output_tokens

        last_error: Exception | None = None
        for attempt in range(4):
            await self._limiter.acquire(cost)
            try:
                resp = await self._client.post(
                    f"/models/{self._model}:generateContent", json=body
                )
                resp.raise_for_status()
                return _extract_text(resp.json())
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in (429, 500, 502, 503):
                    raise LLMError(
                        f"{exc.response.status_code}: {exc.response.text[:400]}"
                    ) from exc
                backoff = 2**attempt * 5
                log.warning(
                    "llm_retry",
                    status=exc.response.status_code,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)

        raise LLMError(f"LLM failed after retries: {last_error}")

    async def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> Any:
        """Structured output. This is our substitute for tool-calling."""
        raw = await self.generate(
            prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Even with a schema, a truncated response (hit maxOutputTokens
            # mid-object) is invalid JSON. Surface it clearly rather than
            # letting a caller crash on a missing key.
            raise LLMError(f"model returned invalid JSON: {raw[:400]}") from exc


def extract_string_list(raw: str, key: str) -> list[str]:
    """Pull a list of strings out of possibly-truncated JSON.

    Gemma sometimes produces a valid array and then degenerates in a LATER
    field, truncating the response and invalidating the whole document. The
    array itself is fine, so discarding everything would throw away good work
    -- which is exactly the bug this fixes.

    Tries a strict parse first; falls back to scanning the named array for
    complete "..." literals (the truncated final one has no closing quote and
    simply isn't matched).
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            items = data.get(key, [])
            return [s.strip() for s in items if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)(?:\]|$)', raw, re.DOTALL)
    if not match:
        return []

    found = [
        m.group(1).replace('\\"', '"').replace("\\'", "'").strip()
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    ]
    return [s for s in found if s and not is_repetitive(s)]


def is_repetitive(text: str) -> bool:
    """True if a few distinct words fill a long string -- a loop artefact."""
    words = text.lower().split()
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < 0.4


# Why a 200 can carry no usable text. The old error told the operator to raise
# max_output_tokens regardless of cause, which is only ever right for
# MAX_TOKENS -- and actively misleading for RECITATION, where more tokens
# cannot help.
_EMPTY_REASONS = {
    "MAX_TOKENS": (
        "the model hit its output limit before emitting anything usable. "
        "Raise max_output_tokens for this call."
    ),
    # Google stops generation when output starts reproducing memorised training
    # data. In this app it means the question was NOT answerable from the
    # retrieved passages, so the model fell back on world knowledge and began
    # reciting a remembered fact. The trigger is a grounding failure, not a
    # configuration problem.
    "RECITATION": (
        "the model began reproducing memorised training text and Google "
        "stopped it. This usually means the question is not answerable from "
        "the retrieved passages, so the model fell back on world knowledge. "
        "Raising max_output_tokens will not help."
    ),
    "SAFETY": "the response was blocked by a safety filter.",
    "PROHIBITED_CONTENT": "the response was blocked as prohibited content.",
    "OTHER": "generation stopped for an unspecified reason.",
}


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull text out of a generateContent response.

    Defensive because there are several ways to get a 200 with no usable text:
    a safety block, a recitation stop, or finishReason=MAX_TOKENS with an empty
    parts list. The finish reason decides what the operator should actually do,
    so it is named in the message rather than guessed at.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback", {})
        raise LLMError(f"no candidates returned (promptFeedback={feedback})")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)

    if not text.strip():
        reason = candidate.get("finishReason") or "UNKNOWN"
        explanation = _EMPTY_REASONS.get(
            reason, "no text was returned and the finish reason is unrecognised."
        )
        raise LLMError(f"empty response ({reason}): {explanation}")
    return text


_client: GemmaClient | None = None


def get_llm() -> GemmaClient:
    """Singleton, so the whole process shares one rate-limit budget."""
    global _client
    if _client is None:
        _client = GemmaClient()
        log.info("llm_ready", model=get_settings().llm_model)
    return _client


async def close_llm() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

"""An OpenAI-compatible chat client, pointed at GroqCloud.

WHAT THIS IS, PRECISELY

Not self-hosting. GroqCloud runs open-weights models (gpt-oss-120b and friends)
on their own LPU hardware and serves them over HTTP; the weights, the runtime,
the batching and the KV cache are all theirs. What is genuinely ours is the
CHOICE OF AN OPEN MODEL -- the same weights can be pulled from Hugging Face and
run under vLLM on a GPU we control, and because vLLM also speaks the OpenAI
chat API, this client keeps working against it with only `base_url` changed.

That is the whole reason the base URL is configuration rather than a constant.
Swapping GroqCloud for a local vLLM, an Ollama instance, or Together is a
one-line change here, which is exactly the property that makes an open model
worth using over a proprietary one.

WHY IT IS NOT PART OF `llm.py`

`GenAIClient` speaks Google's generateContent: different message shape,
different tool-call encoding, different usage accounting, and a responseSchema
mechanism that has no OpenAI equivalent. Folding two protocols into one class
means every call site carries a branch, and the two halves drift. This is
~100 lines and stays separate.

NO RATE LIMITER HERE, ON PURPOSE

The Gemini clients share a quota across the whole agent, so a limiter is what
stops one node starving another. This is a manual playground: one request per
button press, driven by a human. A limiter would add a moving part with nothing
to protect.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from app.config import get_settings

log = structlog.get_logger()


class GroqError(RuntimeError):
    """A Groq call failed. Carries the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _friendly(status: int, body: str) -> str:
    """Turn an HTTP failure into something that says what to DO.

    The raw bodies are accurate and useless to someone setting this up for the
    first time: a bare 401 does not say "your key is wrong", and a 404 on a
    model name reads as if the endpoint is missing rather than the model.
    """
    if status == 401:
        return (
            "Groq rejected the API key (401). Check GROQ_API_KEY in .env, then "
            "recreate the container -- `docker compose restart` does not "
            "re-read .env."
        )
    if status == 404:
        return (
            "Groq does not recognise that model (404). Model ids look like "
            "`openai/gpt-oss-120b`; the list endpoint shows what this key can "
            "actually reach."
        )
    if status == 429:
        return "Groq rate limit reached (429). Wait a moment and try again."
    if status >= 500:
        return f"Groq is having trouble ({status}). This is their side, not ours."
    return f"Groq returned {status}: {body[:300]}"


async def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """One chat completion. Returns the text plus what it cost and how long.

    The timings and token counts are returned rather than merely logged because
    they are the POINT of the playground: comparing an LPU-served open model
    against the Gemini path is only interesting if the numbers are on screen.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise GroqError(
            "GROQ_API_KEY is not set, so there is nothing to call. Add it to "
            ".env and recreate the container with "
            "`docker compose up -d --force-recreate api`."
        )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model or settings.groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            base_url=settings.groq_base_url,
            timeout=httpx.Timeout(settings.groq_timeout_seconds),
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        ) as client:
            resp = await client.post("/chat/completions", json=body)
    except httpx.HTTPError as exc:
        raise GroqError(f"Could not reach Groq: {type(exc).__name__}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)

    if resp.status_code >= 400:
        raise GroqError(_friendly(resp.status_code, resp.text), status=resp.status_code)

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)

    log.info(
        "groq_completed",
        model=body["model"],
        ms=elapsed_ms,
        completion_tokens=completion_tokens,
    )

    return {
        "text": (choice.get("message") or {}).get("content") or "",
        # Why the model stopped. "length" means max_tokens truncated it, which
        # otherwise looks like the model simply trailing off mid-sentence.
        "finish_reason": choice.get("finish_reason"),
        "model": data.get("model") or body["model"],
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "elapsed_ms": elapsed_ms,
        # The headline number for an LPU, and the reason anyone tries Groq.
        # Computed from wall-clock rather than read from the response, so it
        # includes the network round trip -- which is the speed a user
        # experiences, not the speed the provider advertises.
        "tokens_per_second": (
            round(completion_tokens / (elapsed_ms / 1000), 1)
            if completion_tokens and elapsed_ms
            else None
        ),
    }


# Groq's cap on the transcription prompt. It reports this as "characters" and
# MEASURES IT IN UTF-8 BYTES, which is the whole reason this constant and the
# function below exist. Verified against the live API:
#
#     ASCII      890 chars /  890 bytes -> 200
#     ASCII      900 chars /  900 bytes -> 400
#     Devanagari 300 chars /  900 bytes -> 400   <- only 300 "characters"
#
# So Hindi costs three bytes per character and a client counting `str.length`
# sends roughly a third more than it thinks. A Hinglish session hit exactly
# that: a 700-character tail measured 935 bytes and the turn failed.
_PROMPT_MAX_BYTES = 896


def _clip_prompt(prompt: str) -> str:
    """Trim a prompt to Groq's byte limit, keeping the END.

    The tail is what matters: it is the text nearest the audio being
    transcribed, so it carries the most useful context for continuing it.

    Cut on a CHARACTER boundary, never mid-sequence. Slicing encoded bytes
    directly would split a Devanagari character across the boundary and send
    invalid UTF-8 -- a failure that only appears in non-Latin scripts, which is
    to say only in the case this function exists for.
    """
    encoded = prompt.encode("utf-8")
    if len(encoded) <= _PROMPT_MAX_BYTES:
        return prompt
    # errors="ignore" drops the partial character the cut created.
    return encoded[-_PROMPT_MAX_BYTES:].decode("utf-8", errors="ignore").lstrip()


async def transcribe(
    audio: bytes,
    *,
    filename: str = "audio.webm",
    model: str = "whisper-large-v3-turbo",
    language: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Transcribe one audio clip. Returns the text plus timing.

    BATCH, NOT STREAMING, and the caller has to understand that. Whisper takes
    a complete file and returns a complete transcript; there is no partial
    result and no way to get one. "Live" transcription built on this is a
    sequence of short recordings, which is a real approximation with real edges
    -- see the segment length note in the UI.

    `language` is OPTIONAL and omitted by default so Whisper detects it. That
    is right for a one-off clip and wrong for a long bilingual recording: the
    detection runs per request, so consecutive segments of the same
    conversation can be detected differently and the transcript flips script
    mid-way. Passing an explicit language is what stops that -- at the cost of
    forcing the other language through the wrong one.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise GroqError(
            "GROQ_API_KEY is not set, so there is nothing to call. Add it to "
            ".env and recreate the container with "
            "`docker compose up -d --force-recreate api`."
        )

    # `verbose_json` rather than `json`: it carries the detected language and
    # the clip duration, and the detected language is the single most useful
    # diagnostic when a bilingual recording comes back in the wrong script.
    data: dict[str, str] = {"model": model, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    if prompt:
        # Whisper's prompt biases spelling, script and proper nouns. Worth
        # exposing because it is the only lever on how names are rendered --
        # and, for a bilingual speaker, on which script the answer comes back
        # in. Clipped here rather than trusted: the caller counts characters,
        # Groq counts bytes, and the two disagree by 3x in Devanagari.
        data["prompt"] = _clip_prompt(prompt)

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            base_url=settings.groq_base_url,
            timeout=httpx.Timeout(settings.groq_timeout_seconds),
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        ) as client:
            resp = await client.post(
                "/audio/transcriptions",
                files={"file": (filename, audio, "application/octet-stream")},
                data=data,
            )
    except httpx.HTTPError as exc:
        raise GroqError(f"Could not reach Groq: {type(exc).__name__}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if resp.status_code >= 400:
        raise GroqError(_friendly(resp.status_code, resp.text), status=resp.status_code)

    body = resp.json()
    text = (body.get("text") or "").strip()
    duration = float(body.get("duration") or 0.0)

    # Whisper's own confidence, surfaced but NOT acted on.
    #
    # It is tempting to drop a result whose `no_speech_prob` is high, and it
    # does not work. Measured against five seconds of DIGITAL SILENCE, Whisper
    # returned " Thank you." with no_speech_prob 0.000 and avg_logprob -0.293
    # -- maximum confidence, entirely fabricated. The model was trained on
    # subtitle tracks, so silence looks like an end-of-video card to it.
    #
    # Which is why silence is filtered at the MICROPHONE instead: by the time
    # audio reaches this function there is no signal left that distinguishes a
    # hallucination from a transcript. These are returned so that is visible
    # rather than a claim.
    segs = body.get("segments") or []
    no_speech = max((float(g.get("no_speech_prob") or 0.0) for g in segs), default=None)
    logprobs = [float(g["avg_logprob"]) for g in segs if g.get("avg_logprob") is not None]

    log.info(
        "groq_transcribed",
        model=model,
        ms=elapsed_ms,
        audio_seconds=round(duration, 2),
        detected=body.get("language"),
        chars=len(text),
    )

    return {
        "text": text,
        # What Whisper THOUGHT it heard, which is not always what was asked
        # for. Surfaced so a wrong-script transcript has a visible cause.
        "language": body.get("language"),
        "duration_seconds": duration,
        "elapsed_ms": elapsed_ms,
        # >1 means transcription is faster than the audio is long, which is the
        # condition for keeping up with a live speaker at all.
        "realtime_factor": (
            round(duration / (elapsed_ms / 1000), 1) if duration and elapsed_ms else None
        ),
        "no_speech_prob": round(no_speech, 3) if no_speech is not None else None,
        "avg_logprob": (
            round(sum(logprobs) / len(logprobs), 3) if logprobs else None
        ),
        "model": model,
    }


async def list_models() -> list[dict[str, Any]]:
    """Models this key can actually reach, newest-looking first.

    Live rather than a hardcoded list: what a key can reach changes as Groq
    adds and retires models, and a stale constant produces a 404 that looks
    like a bug in this app.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        return []

    try:
        async with httpx.AsyncClient(
            base_url=settings.groq_base_url,
            timeout=httpx.Timeout(30.0),
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        ) as client:
            resp = await client.get("/models")
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        # Best effort. A failed list must not block the playground, which can
        # still send to the configured default.
        log.warning("groq_list_models_failed", error=str(exc))
        return []

    out = [
        {
            "id": m.get("id", ""),
            "owned_by": m.get("owned_by"),
            "context_window": m.get("context_window"),
        }
        for m in (resp.json().get("data") or [])
        if m.get("id")
    ]
    return sorted(out, key=lambda m: m["id"])

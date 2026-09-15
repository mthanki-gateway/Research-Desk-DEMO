"""NVIDIA's hosted NeMo speech models, over NVIDIA Cloud Functions.

WHY A SECOND PROVIDER AT ALL

Groq serves Whisper and only Whisper, and Whisper has two structural limits
this project keeps running into: it is batch-only, and it cannot tell speakers
apart. Both are properties of the model, not of Groq -- Whisper is a
sequence-to-sequence transcriber trained to be speaker-INVARIANT, so there is
nothing in it to attach a speaker label to.

NVIDIA publishes the NeMo family, which has both of the missing pieces:
`nemotron-asr-streaming` is genuinely streaming, and the ASR NIM profiles carry
Sortformer speaker diarization. A free developer key reaches them.

WHERE THEY LIVE, AND WHERE THEY DO NOT

Not on `integrate.api.nvidia.com`. That host serves the LLM catalogue -- 81
models, none of them ASR -- and returns 404 for /audio/transcriptions. Reading
only that endpoint is how one concludes NVIDIA has no speech API, which is
wrong.

Speech runs on NVIDIA Cloud Functions. Measured without a key:

    https://integrate.api.nvidia.com/v1/audio/transcriptions   -> 404
    https://api.nvcf.nvidia.com/v2/nvcf/functions              -> 401
    https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/<uuid> -> 401

401 rather than 404 is the signal: those routes exist and want credentials.

WHY THIS STARTS WITH DISCOVERY RATHER THAN TRANSCRIPTION

A hosted function is addressed by UUID, and the UUIDs are per model and change
as previews are published and retired. Hardcoding one is a 404 waiting to
happen, and guessing the request body for a function that may want gRPC rather
than HTTP would be guessing twice. So the first thing this module does is ask
the account what it can actually reach -- the same way every other integration
in this codebase was built, by probing before writing.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Any

import httpx
import structlog

from app.config import get_settings

log = structlog.get_logger()

# Names that suggest speech rather than an LLM. Used only to sort the listing
# so the interesting ones surface first -- never to hide anything, because a
# filter that is slightly wrong looks exactly like an empty account.
_SPEECH_HINTS = (
    "asr",
    "parakeet",
    "canary",
    "whisper",
    "conformer",
    "sortformer",
    "diar",
    "speech",
    "riva",
    "transcri",
    "nemotron-asr",
)


class NvidiaError(RuntimeError):
    """An NVIDIA call failed. Carries the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _friendly(status: int, body: str) -> str:
    if status == 401:
        return (
            "NVIDIA rejected the API key (401). Keys look like `nvapi-...` and "
            "come from build.nvidia.com. Check NVIDIA_API_KEY in .env, then "
            "recreate the container -- `docker compose restart` does not "
            "re-read .env."
        )
    if status == 403:
        return (
            "NVIDIA accepted the key but refused this function (403). The "
            "account may not have access to that preview model."
        )
    if status == 404:
        return (
            "No such function (404). Function ids are per-model UUIDs and "
            "change as previews are retired -- list them rather than "
            "hardcoding one."
        )
    if status == 429:
        return "NVIDIA rate limit or free-tier credits exhausted (429)."
    return f"NVIDIA returned {status}: {body[:300]}"


def _client() -> httpx.AsyncClient:
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise NvidiaError(
            "NVIDIA_API_KEY is not set. Get a free key from build.nvidia.com, "
            "add it to .env, and recreate the container with "
            "`docker compose up -d --force-recreate api`."
        )
    return httpx.AsyncClient(
        base_url=settings.nvidia_nvcf_url,
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
    )


async def list_functions() -> list[dict[str, Any]]:
    """Every cloud function this key can invoke, speech ones first.

    This is the discovery step, and it is deliberately the first thing built:
    it turns "which model, at which id, over which protocol" from a guess into
    something the account answers directly.

    Returns the raw-ish shape rather than a tidy model, because the useful
    fields are the ones that decide the next step -- `id` to address it,
    `status` to know whether it is deployable, and whatever protocol hint the
    payload carries.
    """
    async with _client() as client:
        resp = await client.get("/functions")

    if resp.status_code >= 400:
        raise NvidiaError(
            _friendly(resp.status_code, resp.text), status=resp.status_code
        )

    raw = resp.json().get("functions") or []
    out: list[dict[str, Any]] = []
    for fn in raw:
        name = str(fn.get("name") or "")
        out.append(
            {
                "id": fn.get("id") or "",
                "name": name,
                "status": fn.get("status"),
                # Present on some functions and the thing that says whether an
                # HTTP body or a gRPC stream is expected. Passed through
                # verbatim because guessing it is exactly what this avoids.
                "protocol": fn.get("inferenceUrl") or fn.get("apiBodyFormat"),
                "speech": any(h in name.lower() for h in _SPEECH_HINTS),
            }
        )

    # Speech first, then alphabetical. A key with hundreds of LLM functions
    # would otherwise bury the three that matter.
    out.sort(key=lambda f: (not f["speech"], f["name"].lower()))
    log.info(
        "nvidia_functions_listed",
        total=len(out),
        speech=sum(1 for f in out if f["speech"]),
    )
    return out


async def invoke(function_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call one cloud function over HTTP.

    Kept generic on purpose. The ASR request body differs per model family and
    some speech functions want gRPC instead, so this stays a thin transport and
    the caller supplies the shape -- which can be written once the listing
    above has said what is actually there.
    """
    started = time.perf_counter()
    async with _client() as client:
        resp = await client.post(f"/pexec/functions/{function_id}", json=payload)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    if resp.status_code >= 400:
        raise NvidiaError(
            _friendly(resp.status_code, resp.text), status=resp.status_code
        )

    log.info("nvidia_invoked", function_id=function_id[:8], ms=elapsed_ms)
    return {"elapsed_ms": elapsed_ms, "body": resp.json()}


# --------------------------------------------------------------------------
# Transcription, over gRPC
#
# NOT HTTP, and this was measured rather than assumed. Posting audio to the
# NVCF HTTP invocation path returns, for every body shape tried:
#
#     500 "Inference connection error while making inference request"
#
# That is the gateway failing to reach a worker which does not speak HTTP --
# a schema mistake would have come back as a 4xx describing the field. Riva
# ASR is gRPC, so the gRPC client is a real dependency rather than a
# preference. See the `speech` extra in pyproject.toml.
# --------------------------------------------------------------------------

_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"

# name -> function id, resolved once per process.
#
# Resolved BY NAME rather than configured by UUID: the ids are per-model and
# change as previews are republished, so a pinned UUID is a 404 waiting to
# happen, while the name is stable and human-checkable.
_function_ids: dict[str, str] = {}


async def resolve_function(name: str) -> str:
    """The function id for a model name, cached for the process lifetime."""
    if name in _function_ids:
        return _function_ids[name]

    for fn in await list_functions():
        # ACTIVE only. An inactive function is listed and not deployable, so
        # picking one produces a timeout rather than a clear error.
        if fn["name"] == name and fn["status"] == "ACTIVE" and fn["id"]:
            _function_ids[name] = fn["id"]
            return fn["id"]

    raise NvidiaError(
        f"No ACTIVE function named {name!r} on this key. "
        "GET /playground/nvidia/functions lists what is reachable."
    )


def _unwrap_wav(audio: bytes) -> tuple[bytes, int, int]:
    """Split a WAV container into (raw frames, sample rate, channels).

    RIVA WANTS FRAMES, NOT A FILE. `offline_recognize` documents its argument
    as the output of `wave.readframes()`, and passing the whole container
    instead -- 44-byte RIFF header and all -- leaves the server to sniff a
    format nobody declared. It half worked: short clips came back fine and
    longer ones came back TRUNCATED, returning only the last few seconds of a
    twelve-second utterance, because the declared length and the real one
    disagreed. The earlier "Unavailable model" errors showed the same cause
    from the other side, reporting `sample_rate=0`.

    Falls back to treating the bytes as raw 16kHz PCM if they are not a WAV,
    which is what the browser would send if the recorder ever changed format.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            return (
                wav.readframes(wav.getnframes()),
                wav.getframerate(),
                wav.getnchannels(),
            )
    except (wave.Error, EOFError):
        return audio, 16_000, 1


def _recognise(audio: bytes, *, function_id: str, language: str, api_key: str):
    """The blocking gRPC call. Runs in a worker thread -- see `transcribe`."""
    import riva.client as rc

    frames, sample_rate, channels = _unwrap_wav(audio)

    auth = rc.Auth(
        None,
        True,  # SSL
        _GRPC_ENDPOINT,
        # Metadata, not headers: gRPC has no URL path per model, so the
        # function id is how the request is routed to one.
        [
            ("function-id", function_id),
            ("authorization", f"Bearer {api_key}"),
        ],
    )
    config = rc.RecognitionConfig(
        # DECLARED, not inferred. Leaving these unset made the server guess at
        # a container it had not been told about, and guess badly on anything
        # longer than a few seconds.
        encoding=rc.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        audio_channel_count=channels,
        language_code=language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    return rc.ASRService(auth).offline_recognize(frames, config)


async def transcribe(
    audio: bytes,
    *,
    model: str = "ai-parakeet-1_1b-rnnt-multilingual-asr",
    language: str = "en-US",
) -> dict[str, Any]:
    """Transcribe one clip with a hosted NeMo model.

    RETURNS EMPTY TEXT ON SILENCE, which is the reason this provider exists
    alongside Whisper. Measured on four seconds of digital silence:

        Parakeet -> ''
        Whisper  -> 'Thank you.'   (no_speech_prob 0.000)

    Whisper was trained on subtitle tracks and emits an end-of-video card when
    handed nothing; Parakeet declines. That difference is why the microphone
    gating in the frontend exists for one and not the other.
    """
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise NvidiaError(
            "NVIDIA_API_KEY is not set. Get a free key from build.nvidia.com, "
            "add it to .env, and recreate the container with "
            "`docker compose up -d --force-recreate api`."
        )

    function_id = await resolve_function(model)
    started = time.perf_counter()
    try:
        # to_thread, because riva.client is SYNCHRONOUS. Called directly it
        # blocks the event loop for the whole round trip -- about a second --
        # stalling every other request in the process, which is the kind of
        # bug that only shows up under concurrency.
        response = await asyncio.to_thread(
            _recognise,
            audio,
            function_id=function_id,
            language=language,
            api_key=settings.nvidia_api_key,
        )
    except Exception as exc:  # noqa: BLE001 - grpc raises its own error types
        raise NvidiaError(f"Transcription failed: {type(exc).__name__}: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)

    parts: list[str] = []
    confidences: list[float] = []
    for result in response.results:
        for alt in result.alternatives:
            if alt.transcript:
                parts.append(alt.transcript.strip())
                confidences.append(alt.confidence)

    text = " ".join(parts).strip()
    log.info(
        "nvidia_transcribed", model=model, ms=elapsed_ms, chars=len(text)
    )
    return {
        "text": text,
        "model": model,
        "language": language,
        "elapsed_ms": elapsed_ms,
        "confidence": (
            round(sum(confidences) / len(confidences), 3) if confidences else None
        ),
    }

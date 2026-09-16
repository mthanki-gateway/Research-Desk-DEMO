"""Speech in, speech out, through Gemini.

WHY GEMINI FOR BOTH HALVES

The Model Lab already talks to Groq's Whisper and NVIDIA's Parakeet, so the
obvious move was to reuse one of those for the ear and bolt a separate TTS
service onto the mouth. This uses Gemini for both instead, for a reason that is
about the SHAPE of the problem rather than about quality:

Gemini's ordinary `generateContent` accepts audio as an inline part. There is
no separate speech API to configure, no second key, no second rate limit to
reason about, and -- the part that matters -- transcription is not a fixed
pipeline stage but a prompt. It can be told what the audio is likely to contain,
which is how a question about "Cascade Ridge" or "Parakeet" survives contact
with a recogniser that has never heard either word.

The measured behaviour on this deployment (see `probe_voice.py`):

    gemini-3.5-flash-lite   audio in  -> accurate transcript, ~1s
    gemini-3.1-flash-tts    text in   -> audio/l16; rate=24000; channels=1

WHAT COMES BACK FROM TTS IS NOT A FILE

The TTS model returns RAW PCM -- signed 16-bit little-endian, 24kHz, mono --
with no container around it. Handing those bytes to an <audio> element produces
silence and no error, because nothing in the browser can know what they are.
`to_wav` prepends the 44-byte header that makes them playable. This is the same
class of mistake as the Riva truncation earlier in this project, in the opposite
direction: there, a container was sent where raw frames were wanted.
"""

from __future__ import annotations

import base64
import re
import struct
from typing import Any

import httpx
import structlog

from app.config import get_settings

log = structlog.get_logger()

GENAI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# What the TTS model emits. Not negotiable and not announced anywhere except
# the response's mimeType, which is checked at runtime rather than trusted.
TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS = 1
TTS_BITS = 16

# Prebuilt Gemini voices, curated.
#
# The full list is thirty-odd names from astronomy and myth, which is a poor
# menu: nobody can tell Sadaltager from Rasalgethi by reading it. These are the
# ones worth offering, each labelled by how it actually sounds.
VOICES: list[dict[str, str]] = [
    {"id": "Kore", "label": "Kore", "character": "Even, unhurried. The default."},
    {"id": "Puck", "label": "Puck", "character": "Brighter, quicker."},
    {"id": "Charon", "label": "Charon", "character": "Low and deliberate."},
    {"id": "Aoede", "label": "Aoede", "character": "Warm, conversational."},
    {"id": "Fenrir", "label": "Fenrir", "character": "Firm, declarative."},
    {"id": "Leda", "label": "Leda", "character": "Light, youthful."},
]
DEFAULT_VOICE = "Kore"


class VoiceError(RuntimeError):
    """A speech step failed in a way the caller should report, not swallow."""


def enabled() -> bool:
    return bool(get_settings().google_api_key)


def _client() -> httpx.AsyncClient:
    s = get_settings()
    if not s.google_api_key:
        raise VoiceError("GOOGLE_API_KEY is not set, so speech is unavailable.")
    return httpx.AsyncClient(
        base_url=GENAI_BASE,
        # Generous: a TTS call for a long answer is slower than a text
        # completion of the same length, and the failure mode of a short
        # timeout here is a turn that silently produces no audio.
        timeout=httpx.Timeout(180.0),
        headers={"x-goog-api-key": s.google_api_key},
    )


def to_wav(pcm: bytes, *, rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Wrap raw little-endian 16-bit PCM in a WAV container.

    Hand-written rather than pulled from a library for the same reason the
    browser-side encoder is: it is 44 bytes of header, and a dependency that
    exists to write 44 bytes is a dependency to keep updated for ever.
    """
    block_align = TTS_CHANNELS * TTS_BITS // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,  # fmt chunk size
            1,  # PCM
            TTS_CHANNELS,
            rate,
            rate * block_align,  # byte rate
            block_align,
            TTS_BITS,
        )
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def _rate_from_mime(mime: str) -> int:
    """Read the sample rate out of `audio/l16; rate=24000`.

    Parsed rather than assumed. If the model ever returns 16kHz, a hardcoded
    24000 in the header does not fail -- it plays the answer back at 1.5x
    speed, chipmunked, and nothing anywhere reports an error.
    """
    match = re.search(r"rate=(\d+)", mime or "")
    return int(match.group(1)) if match else TTS_SAMPLE_RATE


async def transcribe(audio: bytes, *, mime: str = "audio/wav", hint: str = "") -> str:
    """Speech to text, via ordinary multimodal generation.

    `hint` is prepended to the instruction, not to the transcript. Names that
    exist only in this user's corpus -- a document title, a project codename --
    are exactly the words a general recogniser mangles, and a recogniser that
    can be TOLD what to expect is the advantage of doing this with an LLM
    rather than a fixed ASR pipeline.
    """
    settings = get_settings()
    model = settings.voice_stt_model.removeprefix("models/")

    instruction = (
        "Transcribe the speech in this audio verbatim. Output ONLY the words "
        "spoken, with no preamble, no quotation marks, no description of the "
        "audio and no commentary.\n\n"
        "If the audio contains no intelligible speech, output exactly: "
        "(no speech)"
    )
    if hint:
        instruction += (
            "\n\nThese names may occur and are spelled as follows; prefer them "
            f"over similar-sounding common words: {hint}"
        )

    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {
                        "inlineData": {
                            "mimeType": mime,
                            "data": base64.b64encode(audio).decode(),
                        }
                    },
                ]
            }
        ],
        # Zero: a transcript has one correct answer, and sampling from it only
        # invents variation where none is wanted.
        "generationConfig": {"temperature": 0.0},
    }

    async with _client() as client:
        response = await client.post(f"/models/{model}:generateContent", json=payload)
    if response.status_code >= 400:
        raise VoiceError(_explain(response, "Transcription"))

    return heard(_first_text(response.json()))


def heard(raw: str) -> str:
    """The transcript, or "" when the model reports no intelligible speech.

    Separated from the network call so the guard can be tested without one,
    and because it is the guard that matters: passed through as a question,
    "(no speech)" is embedded, retrieved against, answered and spoken back --
    an entire turn, three model calls, spent on silence. A recogniser asked to
    transcribe an empty room will always return SOMETHING, so this has to
    recognise the something.
    """
    text = (raw or "").strip().strip('"').strip()
    if not text:
        return ""
    # Matched on the stem because the model is told to emit "(no speech)" and
    # obliges with "(no speech)", "(No speech)" and "(no speech detected)" --
    # but LENGTH-LIMITED, because "No speech was detected in the recording,
    # why?" is a real question a person might ask and an unbounded prefix match
    # throws it away. A marker is a marker; a sentence is a sentence.
    stripped = text.lower().strip("()").strip()
    if stripped.startswith("no speech") and len(stripped) <= 30:
        return ""
    return text


async def speak(text: str, *, voice: str = DEFAULT_VOICE) -> tuple[bytes, int]:
    """Text to speech, falling through the configured models. (wav, rate).

    WHY A FALLBACK CHAIN AND NOT ONE MODEL

    Free-tier TTS quota is per model and small. Measured mid-build, in one
    pass: 3.1-flash-tts returned 429, 2.5-flash-preview-tts answered in 3.1
    seconds, 2.5-pro-preview-tts returned 429. The newest is not the most
    available -- the same thing the answer pool exists to handle.

    It matters more here than anywhere else in the app. A degraded answer
    elsewhere is still an answer; a turn whose synthesis fails in an audio-only
    app produces nothing at all.

    Only RETRYABLE failures fall through. A 400 means this text or this voice
    is wrong and the next model will reject it identically, so trying three of
    them turns one clear error into a slow one.

    The rate is returned alongside rather than assumed, because it is read from
    the response -- see `_rate_from_mime`.
    """
    settings = get_settings()
    if voice not in {v["id"] for v in VOICES}:
        voice = DEFAULT_VOICE

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }

    attempts: list[str] = []
    async with _client() as client:
        for configured in settings.voice_tts_models:
            model = configured.removeprefix("models/")
            response = await client.post(
                f"/models/{model}:generateContent", json=payload
            )

            if response.status_code >= 400:
                attempts.append(f"{model} ({response.status_code})")
                if _retryable(response.status_code):
                    log.warning(
                        "tts_unavailable", model=model, status=response.status_code
                    )
                    continue
                raise VoiceError(_explain(response, "Speech synthesis"))

            body = response.json()
            try:
                inline = body["candidates"][0]["content"]["parts"][0]["inlineData"]
            except (KeyError, IndexError):
                # A TTS model that returns TEXT has usually refused. Its refusal
                # is more useful than "no audio part" -- but another model may
                # still oblige, so this falls through rather than stopping.
                attempts.append(f"{model} (no audio)")
                log.warning("tts_no_audio", model=model, said=_first_text(body)[:120])
                continue

            rate = _rate_from_mime(inline.get("mimeType", ""))
            pcm = base64.b64decode(inline["data"])
            log.info("tts_done", model=model, voice=voice, rate=rate, bytes=len(pcm))
            return to_wav(pcm, rate=rate), rate

    raise VoiceError(
        "Every speech model is currently unavailable or out of quota — tried "
        + ", ".join(attempts)
        + ". The answer is on screen; wait a minute and play it again."
    )


def _retryable(status: int) -> bool:
    """Worth trying the next model for.

    429 is quota, 5xx is the provider. A 400 means the request itself is wrong
    -- bad voice, bad text -- and every model will reject it the same way, so
    falling through would turn one clear error into three slow ones. 403 is the
    key, which does not improve by asking a different model either.
    """
    return status == 429 or status >= 500


def _first_text(body: dict) -> str:
    for part in (body.get("candidates") or [{}])[0].get("content", {}).get(
        "parts", []
    ) or []:
        if isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _explain(response: httpx.Response, what: str) -> str:
    """A provider error in words the UI can show without decoding JSON."""
    try:
        message = response.json().get("error", {}).get("message", "")
    except Exception:  # noqa: BLE001 - an error path must not raise
        message = response.text[:300]
    if response.status_code == 429:
        return f"{what} hit the model's rate limit. Wait a moment and try again."
    if response.status_code in (401, 403):
        return f"{what} was refused: the API key is missing or lacks access."
    return f"{what} failed ({response.status_code}). {message}"[:400]


# ---------------------------------------------------------------------------
# Making an answer speakable
# ---------------------------------------------------------------------------

# Bracketed citation markers: "[1]", "[2][5]", "[1, 3]".
_CITATION = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def speakable(text: str) -> str:
    """Strip everything that is punctuation to the eye and noise to the ear.

    A RAG answer is written to be READ: it carries "[3]" after a claim, "##"
    before a section, asterisks around emphasis and pipes around a table. Fed
    to a speech model, every one of those is either pronounced -- "bracket
    three" -- or silently distorts the phrasing.

    This does NOT try to make the answer shorter or more conversational. That
    is the model's job and it is asked for in the prompt, because rewriting
    prose with regular expressions produces exactly the mangled result you
    would expect. This only removes markup.
    """
    out = _CODE_FENCE.sub(" (code omitted) ", text)
    out = _TABLE_ROW.sub("", out)
    out = _LINK.sub(r"\1", out)
    out = _CITATION.sub("", out)
    out = _HEADING.sub("", out)
    out = _BOLD_ITALIC.sub(r"\2", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _BULLET.sub("", out)
    # Collapse the blank lines the strips leave behind. A speech model reads a
    # run of newlines as a long pause.
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()

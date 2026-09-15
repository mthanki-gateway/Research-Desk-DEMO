"""A bare endpoint for calling an open-weights model and seeing what happens.

Deliberately thin. This is not the RAG pipeline: no retrieval, no citations, no
critique loop. The point is to see one model call end to end -- what it costs,
how fast it is, what it returns -- so the difference between an LPU-served open
model and the Gemini agent path is visible rather than theoretical.

THE KEY NEVER REACHES THE BROWSER, which is the only non-obvious thing here.
Calling Groq directly from the frontend would be fewer moving parts and would
put a secret in a bundle anyone can read.
"""

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import User, current_user
from app.config import get_settings
from app.schemas.playground import (
    CompletionOut,
    CompletionRequest,
    ModelsOut,
    PlaygroundStatus,
    TranscriptionOut,
)
from app.services import groq_client, nvidia_client

log = structlog.get_logger()

router = APIRouter(prefix="/playground", tags=["playground"])


@router.get("/status", response_model=PlaygroundStatus)
async def status(_: User = Depends(current_user)) -> PlaygroundStatus:
    """Whether this is configured, without revealing anything about the key.

    Separate from the models list so the UI can explain "no key set" without a
    failed request -- an empty dropdown and an error look identical to someone
    who has not finished setting up.
    """
    settings = get_settings()
    return PlaygroundStatus(
        enabled=settings.groq_enabled,
        default_model=settings.groq_model,
        base_url=settings.groq_base_url,
    )


@router.get("/models", response_model=ModelsOut)
async def models(_: User = Depends(current_user)) -> ModelsOut:
    return ModelsOut(models=await groq_client.list_models())


@router.post("/complete", response_model=CompletionOut)
async def complete(
    req: CompletionRequest, user: User = Depends(current_user)
) -> CompletionOut:
    try:
        result = await groq_client.complete(
            req.prompt,
            model=req.model or None,
            system=req.system or None,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except groq_client.GroqError as exc:
        # 502, not 500: the failure is upstream, and the message from
        # `_friendly` says what to do about it. A 500 would read as a bug here.
        log.warning("playground_failed", error=str(exc), owner_id=user.owner_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return CompletionOut(**result)


# Groq caps upload size well above this; the limit here is about what a
# browser should be sending per segment. A 25MB blob from a "live" recorder
# means the segment length is wrong, not that the file is legitimately large.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


@router.post("/transcribe", response_model=TranscriptionOut)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    # Empty string, not None: an HTML form cannot send null, and "" is what
    # the browser actually posts when the language select is on "auto".
    language: str = Form(""),
    prompt: str = Form(""),
    # Which vendor transcribes. "groq" (Whisper) or "nvidia" (NeMo).
    provider: str = Form("groq"),
    user: User = Depends(current_user),
) -> TranscriptionOut:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="The audio file was empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is {len(audio) // 1024 // 1024}MB; the limit is 25MB.",
        )

    if provider == "nvidia":
        try:
            # NeMo takes no prompt: Whisper's `prompt` is a decoder-context
            # trick specific to its architecture, and Parakeet has no
            # equivalent. Silently dropping it is right -- the caller sends one
            # blindly for both providers, and erroring would make switching
            # provider fail for a parameter that simply does not apply.
            result = await nvidia_client.transcribe(
                audio,
                model=model,
                # Riva wants a full locale, not a bare language code.
                language=_locale(language),
            )
        except nvidia_client.NvidiaError as exc:
            log.warning("transcribe_failed", provider="nvidia", error=str(exc))
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return TranscriptionOut(**result)

    try:
        result = await groq_client.transcribe(
            audio,
            filename=file.filename or "audio.webm",
            model=model,
            language=language or None,
            prompt=prompt or None,
        )
    except groq_client.GroqError as exc:
        log.warning("transcribe_failed", error=str(exc), owner_id=user.owner_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TranscriptionOut(**result)


# ISO 639-1 -> the full locale Riva expects.
#
# Riva's RecognitionConfig wants `de-DE`, not `de`. The UI speaks Whisper's
# bare two-letter codes throughout, because that is what the other provider
# takes, so the translation happens here at the boundary rather than making
# every caller know which vendor it is talking to.
#
# Covers Parakeet's 25 European languages and Canary's four. A language not
# listed is passed THROUGH unchanged rather than defaulted: a new model may
# support something this table has not caught up with, and forcing it to
# English would silently transcribe the wrong language.
_LOCALES = {
    "bg": "bg-BG",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "en": "en-US",
    "es": "es-ES",
    "et": "et-EE",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "hr": "hr-HR",
    "hu": "hu-HU",
    "it": "it-IT",
    "lt": "lt-LT",
    "lv": "lv-LV",
    "mt": "mt-MT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sk": "sk-SK",
    "sl": "sl-SI",
    "sv": "sv-SE",
    "uk": "uk-UA",
}


def _locale(language: str) -> str:
    """Riva wants `en-US`; the UI sends Whisper's bare `en`.

    An empty value becomes English rather than being passed on, because Riva
    has no auto-detect and "" is not a language -- the UI does not offer the
    option for NeMo, so this only covers a caller that ignores that.

    An already-full locale (`pt-BR`) passes straight through.
    """
    if not language:
        return "en-US"
    return _LOCALES.get(language, language)


# --------------------------------------------------------------------------
# NVIDIA: hosted NeMo speech models
#
# Discovery first. A hosted function is addressed by a per-model UUID that
# changes as previews are published and retired, so the useful first endpoint
# is "what can this key reach", not a transcription call against an id someone
# pasted from a blog post.
# --------------------------------------------------------------------------


@router.get("/nvidia/status")
async def nvidia_status(_: User = Depends(current_user)) -> dict:
    settings = get_settings()
    return {
        "enabled": settings.nvidia_enabled,
        "base_url": settings.nvidia_nvcf_url,
        "function_id": settings.nvidia_asr_function_id,
    }


@router.get("/nvidia/functions")
async def nvidia_functions(user: User = Depends(current_user)) -> dict:
    try:
        functions = await nvidia_client.list_functions()
    except nvidia_client.NvidiaError as exc:
        log.warning("nvidia_list_failed", error=str(exc), owner_id=user.owner_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "functions": functions,
        "n_speech": sum(1 for f in functions if f["speech"]),
    }

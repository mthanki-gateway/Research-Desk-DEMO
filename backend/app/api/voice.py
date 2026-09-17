"""Parley: one spoken turn, end to end.

WHAT THIS IS

The same agent the Research Desk uses -- same corpus, same vector store, same
web search, same tools -- reached by voice instead of by typing. Not a
different assistant with its own index: a different DOOR onto the one that
already exists. Uploading a document in the Library makes it answerable here
with no further work.

WHY ONE ENDPOINT AND NOT THREE

Transcribe, answer and speak could each be their own route, and the browser
could orchestrate them. They are one route because they are one TURN: there is
no state worth exposing between the steps, an intermediate failure leaves the
user with nothing useful either way, and three round trips over a mobile
connection is three chances to lose the thread of a conversation.

WHY THE TURN IS PUSH-TO-TALK

The obvious design is continuous listening with voice activity detection, and
it is the wrong one here. VAD cannot distinguish a pause for thought from the
end of a question, so it either cuts people off mid-sentence or waits so long
that the conversation drags. Worse, an assistant that is always listening will
eventually interrupt, and being interrupted by a machine is a distinctly
unpleasant experience. A button means both parties know exactly whose turn it
is. The VAD recorder still exists in the Model Lab, where cutting on pauses is
the point.
"""

from __future__ import annotations

import base64
import uuid

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agent.graph import run_agent
from app.auth import User, current_user
from app.config import get_settings
from app.services import corpus, voice, websearch

log = structlog.get_logger()

router = APIRouter(prefix="/voice", tags=["voice"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024

# How the agent is told to write when it is going to be HEARD.
#
# Passed through `preferences`, which is the same channel the user's stored
# instructions travel on, so it reaches `draft` AND `resolve` without either
# node needing to know this app exists.
#
# The rules are not stylistic preferences. Each one fixes something that is
# actively broken in speech:
#
#   - Bracketed citations are read aloud as "bracket three".
#   - Markdown structure is inaudible; a heading and a sentence sound alike.
#   - A reader skims a long answer. A listener cannot skim at all, so length
#     costs them far more, and the useful part is usually at the end.
#   - "As mentioned above" refers to something that has already finished
#     playing and cannot be looked back at.
SPOKEN_STYLE = """

HOW THIS ANSWER WILL BE DELIVERED

This answer is going to be SPOKEN ALOUD to the user. They will hear it once and \
cannot re-read it, scroll back or skim.

Write it as speech:

- Plain sentences. No markdown at all -- no headings, no bullet points, no bold, \
no tables, no code blocks. They are inaudible and their punctuation is read out.
- Do NOT write bracketed citation numbers. Name the source in words instead, \
the way a person would: "your engineering handbook says" or "according to the \
November incident report".
- Be brief. Aim for under 120 words. Lead with the answer, then at most two \
supporting details. A listener who wants more will ask.
- Never say "above", "below", "as mentioned" or "see the table" -- there is no \
page and nothing to see.
- Numbers and dates should be written as they are to be SAID: "sixty four \
passages", "the eleventh of November".
- If you genuinely do not know, say so in one sentence. A spoken hedge is far \
more tiring than a written one.
"""


@router.get("/status")
async def status(user: User = Depends(current_user)) -> dict:
    """What the client needs before it offers a microphone button.

    Reported rather than assumed: a browser that asks for mic permission and
    then discovers the server cannot synthesise anything has spent the user's
    goodwill for nothing.
    """
    settings = get_settings()
    return {
        "enabled": voice.enabled(),
        "stt_model": settings.voice_stt_model,
        # The chain, not one name: the UI should say what may speak, and
        # the first entry is not always the one that does.
        "tts_models": settings.voice_tts_models,
        "voices": voice.VOICES,
        "default_voice": settings.voice_default,
        # Shown in the UI, because "it did not search the web" and "it cannot
        # search the web" are indistinguishable from a spoken answer.
        "web_search": websearch.enabled(),
    }


@router.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    voice_name: str = Form(""),
    top_k: int = Form(0),
    user: User = Depends(current_user),
) -> dict:
    """Audio in, audio out. One turn, three model calls.

    Returns the transcript and the answer text alongside the audio. They are
    not there to be read instead of listened to -- they are there because a
    voice interface that shows nothing is impossible to debug, for the user as
    much as for us: when the answer is wrong, the first question is always
    whether it heard the question correctly.
    """
    if not voice.enabled():
        raise HTTPException(
            status_code=503,
            detail="Speech is not configured on this deployment (no Google API key).",
        )

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That recording is {len(data) // 1024 // 1024}MB; the limit is "
            f"{MAX_AUDIO_BYTES // 1024 // 1024}MB. Ask a shorter question.",
        )

    settings = get_settings()

    # ---- 1. hear ---------------------------------------------------------
    # The corpus filenames go in as a spelling hint. Document titles are
    # exactly the words a recogniser mangles -- they are proper nouns it has
    # never seen -- and mangling them is what turns a good question into a
    # search for nothing.
    try:
        hint = await _name_hint(user.owner_id)
    except Exception:  # noqa: BLE001 - a hint is an optimisation, never required
        hint = ""

    try:
        transcript = await voice.transcribe(
            data, mime=audio.content_type or "audio/wav", hint=hint
        )
    except voice.VoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not transcript:
        # Deliberately answered with SPEECH rather than an error. The user is
        # not looking at the screen -- that is the whole premise of this app --
        # so a silent red banner is a dead end.
        spoken = "I did not catch that. Try again, a little closer to the microphone."
        return await _spoken_reply(
            transcript="",
            answer=spoken,
            voice_name=voice_name or settings.voice_default,
            sources=[],
            heard_nothing=True,
        )

    log.info("voice_heard", chars=len(transcript), owner=bool(user.owner_id))

    # ---- 2. think --------------------------------------------------------
    # The SAME agent as the Research Desk. No separate index, no separate
    # prompt stack -- only the delivery style differs, and that travels as a
    # preference so `draft` and `resolve` both receive it.
    #
    # `clarify` is forced OFF. The human-in-the-loop pause renders as a set of
    # buttons to click, and there is nothing to click here; a paused graph in
    # this app is a turn that silently produces no audio.
    # A thread id PER TURN, which is what the compiled graph requires.
    #
    # The running server installs a checkpointer during startup, and a
    # checkpointed graph refuses to run without one -- "Checkpointer requires
    # one or more of the following 'configurable' keys". Omitting it worked in
    # every test here precisely because a bare Python process never runs the
    # lifespan, so it got an UNCHECKPOINTED graph and no complaint. The one
    # configuration that mattered was the one not exercised.
    #
    # Per turn rather than per session, matching the chat path: the state
    # accumulators use append reducers, so a reused thread grows without bound
    # and leaks an earlier question's evidence into a later answer. Parley
    # keeps no conversation in graph state at all, so a fresh uuid is exactly
    # right -- the checkpointer's job here is durability within one run.
    result = await run_agent(
        transcript,
        owner_id=user.owner_id,
        top_k=top_k or None,
        preferences=SPOKEN_STYLE,
        clarify=False,
        thread_id=f"parley:{uuid.uuid4()}",
    )

    answer = (result.answer or "").strip()
    if not answer:
        answer = "I could not find an answer to that in your documents."

    # ---- 3. speak --------------------------------------------------------
    sources = _sources(result)
    return await _spoken_reply(
        transcript=transcript,
        answer=answer,
        voice_name=voice_name or settings.voice_default,
        sources=sources,
        iterations=result.iterations,
        partial=result.partial,
    )


async def _spoken_reply(
    *,
    transcript: str,
    answer: str,
    voice_name: str,
    sources: list[dict],
    heard_nothing: bool = False,
    iterations: int = 0,
    partial: bool = False,
) -> dict:
    """Synthesise and package one reply."""
    settings = get_settings()
    spoken = voice.speakable(answer)
    truncated = len(spoken) > settings.voice_answer_max_chars
    if truncated:
        # Cut at a sentence boundary, not mid-word. A spoken answer that stops
        # in the middle of a word sounds like a crash; one that stops after a
        # full stop merely sounds brief.
        clipped = spoken[: settings.voice_answer_max_chars]
        cut = max(clipped.rfind("."), clipped.rfind("?"), clipped.rfind("!"))
        spoken = clipped[: cut + 1] if cut > 200 else clipped
        spoken += " There is more detail than I can read out; the full answer is on screen."

    try:
        wav, rate = await voice.speak(spoken, voice=voice_name)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "transcript": transcript,
        "answer": answer,
        # What was actually SAID, which differs from `answer` once markup is
        # stripped and length is capped. Returned so the screen can show the
        # words being spoken rather than a different text that merely resembles
        # them.
        "spoken": spoken,
        # Base64 in the JSON body rather than a second request for the audio.
        # One turn, one round trip; the cost is ~33% over the raw bytes, which
        # for a 15-second answer is about 200KB.
        "audio": base64.b64encode(wav).decode(),
        "mime": "audio/wav",
        "sample_rate": rate,
        "sources": sources,
        "heard_nothing": heard_nothing,
        "truncated": truncated,
        "iterations": iterations,
        "partial": partial,
    }


def _sources(result) -> list[dict]:
    """The evidence behind the answer, deduplicated by document.

    Per DOCUMENT rather than per passage: a spoken answer cites in prose, so
    the screen's job is to say where this came from, and eight chunks of one
    handbook is one source to a listener.
    """
    seen: dict[str, dict] = {}
    for hit in result.evidence:
        key = hit.url if getattr(hit, "source", "document") == "web" else str(
            hit.document_id
        )
        if key in seen:
            seen[key]["passages"] += 1
            continue
        seen[key] = {
            "label": hit.filename,
            "kind": getattr(hit, "source", "document"),
            "url": getattr(hit, "url", None),
            "passages": 1,
        }
    return list(seen.values())


async def _name_hint(owner_id: str | None) -> str:
    """Document names, as a spelling aid for the recogniser."""
    rows = await corpus.documents(owner_id)
    names = [str(r.get("filename") or "") for r in rows][:25]
    # Stems only: ".md" read aloud is "dot em dee" and helps nothing.
    stems = [n.rsplit(".", 1)[0].replace("-", " ") for n in names if n]
    return ", ".join(stems)

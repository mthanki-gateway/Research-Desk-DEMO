"""The job that turns a finished interview's audio into emotion scores.

SEPARATE FROM BOTH SIDES ON PURPOSE. `jobs.py` knows how to queue and run work
without knowing what work is; `emotion.py` knows how to score a clip without
knowing where clips come from. This is the seam between them, and it is the
only module that knows an interview has turns, that turns have recordings, and
where those recordings live.

That matters for the thing this is built to allow: re-running the analysis when
a better model appears. Nothing above or below here has to change for that --
it is the same job, queued again, reading the same stored audio.
"""

from __future__ import annotations

import uuid

import structlog

from app.services import emotion
from app.services.storage import StorageError, get_storage

log = structlog.get_logger()

#: Queued against a conversation, so the Results tab can ask "is there an
#: emotion job for this interview" without knowing a payload's shape.
KIND = "emotion"
SUBJECT = "conversation"


async def run_emotion_job(payload: dict) -> dict:
    """Score every recorded turn of one conversation.

    Raises on a failure worth retrying -- a store that is briefly unreachable,
    a model that could not load -- because `jobs.run_one` turns that into
    another attempt. It returns normally, with a reason, when there is simply
    nothing to analyse: no recordings is a finished job, not a failed one.
    """
    # No guard on `emotion_analysis` here: the handler is only REGISTERED when
    # the feature is on (see `main.lifespan`), so a job reaching this function
    # is one something can actually run. Checking again and raising would fail
    # jobs that are merely waiting.
    session_id = uuid.UUID(str(payload["session_id"]))
    turns = await _recorded_turns(session_id)
    if not turns:
        log.info("emotion_no_audio", session=str(session_id))
        return {"turns": [], "moments": [], "reason": "no recordings"}

    store = get_storage()
    scored: list[dict] = []
    for turn in turns:
        try:
            audio = await store.get(turn["key"])
        except StorageError as exc:
            # One missing clip should not lose the other nine.
            log.warning("emotion_clip_missing", key=turn["key"], error=str(exc)[:120])
            continue
        samples, rate = _decode_wav(audio)
        scored.append({"turn": turn["turn"], "scores": await emotion.score_clip(samples, rate)})

    summary = emotion.summarise(scored)
    await _store_result(session_id, summary)
    log.info("emotion_scored", session=str(session_id), turns=len(scored))
    return summary


async def _recorded_turns(session_id: uuid.UUID) -> list[dict]:
    """Which turns of this conversation have audio, oldest first.

    EMPTY UNTIL RECORDINGS EXIST. Interview audio is not captured yet -- see
    docs/storage.md -- so this returns nothing and the job completes with
    "no recordings" rather than failing. That is deliberate: the queue, the
    handler and the status the UI reads are all exercised now, and the day
    recordings land this is the only function that changes.
    """
    from sqlalchemy import select

    from app.db.models import Message
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)
            )
        ).scalars().all()

    turns = []
    for index, row in enumerate(rows):
        key = (row.agent_meta or {}).get("audio_key")
        if key:
            turns.append({"turn": index, "key": key})
    return turns


def _decode_wav(data: bytes) -> tuple[list[float], int]:
    """A WAV's frames as floats in -1..1, with its sample rate.

    `wave` from the standard library rather than soundfile or librosa: the
    recordings this reads are written by us, as 16-bit mono PCM, so there is
    exactly one format to handle and no reason to add a dependency for it.
    """
    import array
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"Expected 16-bit PCM, got {width * 8}-bit")
    samples = array.array("h")
    samples.frombytes(frames)
    return [s / 32768.0 for s in samples], rate


async def _store_result(session_id: uuid.UUID, summary: dict) -> None:
    """Onto the conversation's profile, under `voice`.

    Beside `affect` rather than replacing it. They answer the same question
    from different evidence -- one is what the interviewer heard and can put
    into words, the other is what the waveform measures -- and a reader is
    better served by both than by whichever was written last.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        chat = await db.get(ChatSession, session_id)
        if chat is None:
            return
        merged = dict(chat.profile or {})
        merged["voice"] = summary
        chat.profile = merged
        await db.commit()

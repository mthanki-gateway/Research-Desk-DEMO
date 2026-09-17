"""Is what the screen shows actually what the model heard?

Observed in the browser: the participant said their name was Bill, the
interviewer replied "Thanks, Bill" -- and the transcript on screen never
contained the word. Another turn rendered as "When the maintenance yesterday",
which was not what was said at all.

So the model UNDERSTOOD correctly and the transcript did not. That is possible
because they are two different things: audio is tokenised straight into the
model's own sequence, and `input_transcription` is a SEPARATE, lossier pass
that exists only so a screen has something to show.

This logs every transcript fragment with its timing against `turn_complete`, to
find out whether we are also losing fragments by resetting too early -- a
fragment that arrives after the turn is closed would be attributed to the NEXT
question, which would truncate both.

Run:  docker compose exec api python -m app.probe_transcript
"""

from __future__ import annotations

import asyncio
import audioop
import io
import os
import pathlib
import time
import wave

from google import genai
from google.genai import types

from app.config import get_settings
from app.services import live

# A long utterance with natural pauses -- the case that misbehaves. 27 seconds
# of real synthesised speech, which is far longer than the probes that passed.
SOURCE = "/tmp/answer.wav"


def pcm16k(path: str) -> bytes:
    with wave.open(io.BytesIO(pathlib.Path(path).read_bytes())) as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
    out, _ = audioop.ratecv(raw, 2, 1, rate, 16_000, None)
    return out


async def main() -> None:
    settings = get_settings()
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"], http_options={"api_version": "v1beta"}
    )
    audio = pcm16k(SOURCE)
    print(f"utterance: {len(audio) / 2 / 16000:.1f}s")

    config = live.config("Kore", None, "speak")
    frame = 16_000 * 2 // 10

    async with client.aio.live.connect(
        model=settings.live_model.removeprefix("models/"), config=config
    ) as session:
        await session.send_realtime_input(activity_start=types.ActivityStart())
        for i in range(0, len(audio), frame):
            await session.send_realtime_input(
                audio=types.Blob(data=audio[i : i + frame], mime_type="audio/pcm;rate=16000")
            )
            await asyncio.sleep(0.01)
        # A LONG TRAILING PAUSE before ending the turn, which is what a person
        # does when they finish a thought and think about whether to add to it.
        await asyncio.sleep(2.0)
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        closed = time.time()
        print("activity_end sent\n")

        heard: list[str] = []
        said: list[str] = []
        turn_completes = 0

        async def consume() -> None:
            nonlocal turn_completes
            while True:
                empty = True
                async for message in session.receive():
                    empty = False
                    sc = message.server_content
                    if not sc:
                        continue
                    at = time.time() - closed
                    if sc.input_transcription and sc.input_transcription.text:
                        text = sc.input_transcription.text
                        heard.append(text)
                        marker = " <-- AFTER turn_complete" if turn_completes else ""
                        print(f"  {at:5.1f}s  heard  {text!r}{marker}")
                    if sc.output_transcription and sc.output_transcription.text:
                        said.append(sc.output_transcription.text)
                    if sc.turn_complete:
                        turn_completes += 1
                        print(f"  {at:5.1f}s  TURN_COMPLETE #{turn_completes}")
                        if said:
                            return
                if empty:
                    return

        try:
            await asyncio.wait_for(consume(), timeout=60)
        except TimeoutError:
            print("  (timed out)")

    print()
    print("TRANSCRIPT IT REPORTED:")
    print(" ", "".join(heard).strip() or "(none)")
    print()
    print("WHAT IT SAID BACK:")
    print(" ", "".join(said).strip()[:400] or "(none)")


if __name__ == "__main__":
    asyncio.run(main())

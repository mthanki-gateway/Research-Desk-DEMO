"""Does a native audio-to-audio model actually do the job? Measured, not assumed.

Three things have to be true before rebuilding Parley on the Live API:

  1. audio in, audio out, with no transcript in the middle
  2. it calls OUR tools while doing it
  3. it speaks the tool result back

Run:  docker compose exec api python -m app.probe_live
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

MODEL = os.environ.get("PROBE_LIVE_MODEL", "gemini-3.8-live")

SEARCH = types.FunctionDeclaration(
    name="search_documents",
    description=(
        "Search the user's uploaded documents. Call this before answering any "
        "factual question about their material."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING")},
        required=["query"],
    ),
)


def question_pcm() -> bytes:
    """A real spoken question at 16kHz mono, which is what Live wants in."""
    raw = pathlib.Path("/tmp/tts.wav").read_bytes()
    with wave.open(io.BytesIO(raw)) as w:
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    out, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
    # A SECOND OF SILENCE ON THE END, and this is not cosmetic.
    #
    # Live's automatic VAD decides a turn has ended by hearing the speaker
    # stop. Audio that stops at the last word gives it nothing to detect, and
    # `audio_stream_end` does NOT substitute: measured, the model accepted the
    # audio, reported no transcript and never replied at all. With one second
    # of trailing silence the same audio is transcribed correctly and answered.
    return out + bytes(16_000 * 2)


async def main() -> None:
    key = os.environ["GOOGLE_API_KEY"]
    client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[types.Tool(function_declarations=[SEARCH])],
        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=(
                        "You are a research assistant. You MUST call "
                        "search_documents before answering any factual "
                        "question, then answer from what it returns."
                    )
                )
            ]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    pcm = question_pcm()
    print(f"model    : {MODEL}")
    print(f"question : {len(pcm)} bytes of 16kHz PCM ({len(pcm) / 2 / 16000:.1f}s)")

    started = time.time()
    first_audio: float | None = None
    audio_bytes = 0
    heard: list[str] = []
    said: list[str] = []
    tool_calls: list[tuple[str, dict]] = []

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print(f"connected in {time.time() - started:.1f}s")

        # CHUNKED AND PACED, not one blob.
        #
        # Live's automatic VAD decides the turn ended by watching the audio
        # arrive in real time. A single 4-second blob dropped in at once gives
        # it no stream to watch -- measured: the model connected, accepted the
        # audio and never replied at all.
        #
        # 100ms frames at roughly wall-clock speed is what a microphone would
        # actually produce.
        frame = 16_000 * 2 // 10  # 100ms of 16kHz 16-bit mono
        for i in range(0, len(pcm), frame):
            await session.send_realtime_input(
                audio=types.Blob(
                    data=pcm[i : i + frame], mime_type="audio/pcm;rate=16000"
                )
            )
            await asyncio.sleep(0.02)
        await session.send_realtime_input(audio_stream_end=True)
        sent = time.time()
        print(f"sent {len(pcm) // frame} frames", flush=True)

        # Bounded, because `session.receive()` blocks for ever when nothing
        # arrives -- which is exactly the failure being investigated, and an
        # unbounded loop reports it as a hang rather than as a result.
        answered = False
        tool_rounds = 0

        async def consume() -> None:
            """Keep reading until the model has SPOKEN, not merely replied.

            `session.receive()` yields until the end of one exchange and then
            stops, so a tool call ends the generator -- the spoken answer
            arrives on the NEXT one. Treating the first end as the end of the
            turn reports a tool call and no audio, which is what happened here
            before this loop restarted.
            """
            nonlocal first_audio, audio_bytes, answered, tool_rounds
            while not answered:
                got_anything = False
                async for message in session.receive():
                    got_anything = True
                    if message.data:
                        if first_audio is None:
                            first_audio = time.time() - sent
                        audio_bytes += len(message.data)

                    sc = message.server_content
                    if sc:
                        if sc.input_transcription and sc.input_transcription.text:
                            heard.append(sc.input_transcription.text)
                        if sc.output_transcription and sc.output_transcription.text:
                            said.append(sc.output_transcription.text)
                        # Only an end WITH audio is the end. A turn_complete
                        # that carried no speech is the model finishing its
                        # tool-calling step.
                        if sc.turn_complete and audio_bytes:
                            answered = True

                    if message.tool_call:
                        tool_rounds += 1
                        responses = []
                        for fc in message.tool_call.function_calls:
                            args = dict(fc.args or {})
                            tool_calls.append((fc.name, args))
                            print(f"  TOOL CALL: {fc.name}({args})", flush=True)
                            responses.append(
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={
                                        "result": (
                                            "The engineering handbook states the "
                                            "on-call paging policy: an alert may "
                                            "only page a human if it is urgent, "
                                            "actionable and specific."
                                        )
                                    },
                                )
                            )
                        await session.send_tool_response(function_responses=responses)
                        print("  tool result sent back", flush=True)

                if not got_anything:
                    # A generator that yields nothing means the session is done
                    # with us; looping again would spin.
                    break

        try:
            await asyncio.wait_for(consume(), timeout=60)
        except TimeoutError:
            print("  (timed out waiting for the model)", flush=True)

    total = time.time() - sent
    print()
    print(f"it heard      : {''.join(heard).strip() or '(not reported)'}")
    print(f"tool calls    : {tool_calls or 'NONE'}")
    print(f"it said       : {''.join(said).strip()[:300] or '(not reported)'}")
    print(
        f"audio out     : {audio_bytes} bytes "
        f"= {audio_bytes / 2 / 24000:.1f}s of speech"
    )
    print(f"first audio in: {first_audio:.2f}s" if first_audio else "first audio: none")
    print(f"whole turn    : {total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

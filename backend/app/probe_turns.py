"""Why does only the FIRST turn report what it heard?

Observed in the browser: turn 1 shows a transcript, every turn after it shows
"nothing intelligible" -- while the ANSWERS are correct, so the model is
hearing perfectly well. Something about `input_transcription` is not surviving
the second turn.

This logs every event in order, across two turns of one session, so the
difference is visible rather than guessed at.

Run:  docker compose exec api python -m app.probe_turns
"""

from __future__ import annotations

import audioop
import io
import json
import pathlib
import time
import wave

from starlette.testclient import TestClient

from app.api import live as live_api
from app.auth import ANONYMOUS
from app.main import app

live_api._authenticate = lambda token: _anonymous()


async def _anonymous():
    return ANONYMOUS


def question_pcm() -> bytes:
    raw = pathlib.Path("/tmp/tts.wav").read_bytes()
    with wave.open(io.BytesIO(raw)) as w:
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    out, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
    return out


def turn(ws, pcm: bytes, label: str) -> None:
    frame = 16_000 * 2 // 10
    print(f"\n--- {label} ---", flush=True)
    started = time.time()
    ws.send_json({"type": "start"})
    for i in range(0, len(pcm), frame):
        ws.send_bytes(pcm[i : i + frame])
    ws.send_json({"type": "end"})

    audio = 0
    deadline = time.time() + 60
    while time.time() < deadline:
        message = ws.receive()
        if message.get("type") == "websocket.disconnect":
            print("  [disconnect]", flush=True)
            return
        if message.get("bytes") is not None:
            audio += len(message["bytes"])
            continue
        text = message.get("text")
        if not text:
            continue
        event = json.loads(text)
        kind = event["type"]
        at = time.time() - started
        if kind in ("heard", "said"):
            print(f"  {at:5.1f}s  {kind:10} {event['text']!r}", flush=True)
        elif kind == "turn":
            print(f"  {at:5.1f}s  TURN  Q={event['question']!r}", flush=True)
            print(f"           A={event['answer'][:90]!r}", flush=True)
        elif kind == "tool":
            print(f"  {at:5.1f}s  tool       {event['tool']}", flush=True)
        else:
            print(f"  {at:5.1f}s  {kind}", flush=True)
        if kind == "turn_end":
            print(f"  (audio out: {audio} bytes)", flush=True)
            return
        if kind == "error":
            return


def main() -> None:
    pcm = question_pcm()
    with TestClient(app) as http:
        import os
        mode = os.environ.get("PROBE_MODE", "speak")
        with http.websocket_connect(
            f"/live/ws?voice_name=Kore&mode={mode}"
        ) as ws:
            print("ready:", ws.receive_json())
            turn(ws, pcm, "TURN 1")
            turn(ws, pcm, "TURN 2")
            turn(ws, pcm, "TURN 3")


if __name__ == "__main__":
    main()

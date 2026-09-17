"""Does the live model remember the previous turn? And across sessions?

Two questions that matter for how Parley should be built:

  1. WITHIN one socket, does turn 2 know about turn 1?
  2. ACROSS sockets, does anything survive?

Run:  docker compose exec api python -m app.probe_memory
"""

from __future__ import annotations

import audioop
import io
import json
import time
import wave

from starlette.testclient import TestClient

from app.api import live as live_api
from app.auth import ANONYMOUS
from app.main import app
from app.services import voice

live_api._authenticate = lambda token: _anonymous()


async def _anonymous():
    return ANONYMOUS


def to_pcm16k(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav)) as w:
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    out, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
    return out


def ask(ws, pcm: bytes) -> tuple[str, str]:
    """One full turn. Returns (heard, said)."""
    frame = 16_000 * 2 // 10
    ws.send_json({"type": "start"})
    for i in range(0, len(pcm), frame):
        ws.send_bytes(pcm[i : i + frame])
    ws.send_json({"type": "end"})

    heard: list[str] = []
    said: list[str] = []
    deadline = time.time() + 90
    while time.time() < deadline:
        message = ws.receive()
        if message.get("type") == "websocket.disconnect":
            break
        if message.get("bytes") is not None:
            continue
        text = message.get("text")
        if not text:
            continue
        event = json.loads(text)
        if event["type"] == "heard":
            heard.append(event["text"])
        elif event["type"] == "said":
            said.append(event["text"])
        elif event["type"] == "turn_end":
            break
        elif event["type"] == "error":
            said.append(f"[ERROR {event['detail']}]")
            break
    return "".join(heard).strip(), "".join(said).strip()


async def synth(text: str) -> bytes:
    wav, _ = await voice.speak(text, voice="Puck")
    return to_pcm16k(wav)


def main() -> None:
    import asyncio

    first = asyncio.run(synth("My favourite pyramid is the Red Pyramid. Remember that."))
    second = asyncio.run(synth("Which pyramid did I just say was my favourite?"))

    print("=" * 66)
    print("WITHIN ONE SESSION")
    print("=" * 66)
    with TestClient(app) as http:
        with http.websocket_connect("/live/ws?voice_name=Kore") as ws:
            ws.receive_json()  # ready
            heard, said = ask(ws, first)
            print(f"  turn 1 heard: {heard}")
            print(f"  turn 1 said : {said[:180]}")
            heard, said = ask(ws, second)
            print(f"  turn 2 heard: {heard}")
            print(f"  turn 2 said : {said[:180]}")
            remembered = "red" in said.lower()
            print(f"  -> REMEMBERED WITHIN THE SESSION: {remembered}")

    print()
    print("=" * 66)
    print("ACROSS SESSIONS (a new socket, as a reconnect would be)")
    print("=" * 66)
    with TestClient(app) as http:
        with http.websocket_connect("/live/ws?voice_name=Kore") as ws:
            ws.receive_json()
            heard, said = ask(ws, second)
            print(f"  heard: {heard}")
            print(f"  said : {said[:180]}")
            print(f"  -> REMEMBERED ACROSS SESSIONS: {'red' in said.lower()}")


if __name__ == "__main__":
    main()

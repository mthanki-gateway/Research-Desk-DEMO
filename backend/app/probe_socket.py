"""The whole live path, exercised the way the browser will use it.

Not the SDK directly -- OUR WebSocket, OUR protocol, OUR tools against the real
Qdrant collection. The SDK probe proved the model can do this; this proves the
app can.

Run:  docker compose exec api python -m app.probe_socket
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

# IN-PROCESS, against the real app object.
#
# Auth is enabled on this deployment and a Supabase access token cannot be
# minted from here, so the token CHECK is stubbed -- and only that. The route,
# the protocol, the live session, the tool bridge and the Qdrant collection are
# all the real ones. The rejection path was verified separately over a real
# socket: a bad token closes with 4401.
live_api._authenticate = lambda token: _anonymous()


async def _anonymous():
    return ANONYMOUS


def question_pcm(path: str) -> bytes:
    raw = pathlib.Path(path).read_bytes()
    with wave.open(io.BytesIO(raw)) as w:
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    out, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
    return out


def main() -> None:
    pcm = question_pcm("/tmp/tts.wav")
    print(f"question: {len(pcm) / 2 / 16000:.1f}s of 16kHz PCM")

    frame = 16_000 * 2 // 10
    audio = 0
    first_audio = None
    heard: list[str] = []
    said: list[str] = []
    tools: list[dict] = []

    with TestClient(app) as http:
        with http.websocket_connect("/live/ws?voice_name=Kore") as ws:
            first = ws.receive_json()
            print("server says:", first)
            if first.get("type") != "ready":
                return

            ws.send_json({"type": "start"})

            # A THREE-SECOND PAUSE, DELIBERATELY, halfway through.
            #
            # This is the behaviour being tested. With automatic activity
            # detection the model would treat this silence as the end of the
            # question and answer half of it. With the button owning the turn,
            # it must wait.
            half = (len(pcm) // 2 // frame) * frame
            for i in range(0, half, frame):
                ws.send_bytes(pcm[i : i + frame])
            print("  ...pausing 3s mid-question...", flush=True)
            time.sleep(3)
            for i in range(half, len(pcm), frame):
                ws.send_bytes(pcm[i : i + frame])

            # The button was clicked. THIS is what ends the turn.
            ws.send_json({"type": "end"})
            sent = time.time()
            print("sent, waiting...", flush=True)

            while True:
                message = ws.receive()
                if message.get("type") == "websocket.disconnect":
                    print("  (server closed)")
                    break
                if message.get("bytes") is not None:
                    if first_audio is None:
                        first_audio = time.time() - sent
                    audio += len(message["bytes"])
                    continue
                text = message.get("text")
                if not text:
                    continue
                event = json.loads(text)
                kind = event.get("type")
                if kind == "heard":
                    heard.append(event["text"])
                elif kind == "said":
                    said.append(event["text"])
                elif kind == "tool":
                    tools.append(event)
                    print(
                        f"  TOOL {event['tool']}({event.get('args')}) "
                        f"-> {event.get('n')} hits, "
                        f"sources={[s['label'][:38] for s in event.get('sources', [])]}",
                        flush=True,
                    )
                elif kind == "turn_end":
                    break
                elif kind == "error":
                    print("  ERROR:", event.get("detail"))
                    break

    print()
    print(f"it heard      : {''.join(heard).strip() or '(none)'}")
    print(f"tool calls    : {len(tools)}")
    print(f"it said       : {''.join(said).strip()[:400] or '(none)'}")
    print(f"audio out     : {audio} bytes = {audio / 2 / 24000:.1f}s of speech")
    if first_audio:
        print(f"FIRST SOUND   : {first_audio:.2f}s after the button was released")


if __name__ == "__main__":
    main()

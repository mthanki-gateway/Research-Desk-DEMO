"""BigBlueButton without a browser: join, listen, record.

WHY THIS EXISTS

The Chromium bot costs about a gigabyte and a CPU core per meeting. That is
fine for one meeting and a wall at fifty. BigBlueButton is an open system, so
the client can be written down instead of run: about thirty megabytes per
meeting, and hundreds of bots in one process.

WHAT A BROWSER DOES THAT THIS REPRODUCES (read off a real room, BBB 2.7)

    1. GET the join link            -> 302 to /html5client/join?sessionToken=...
    2. GET /api/enter?sessionToken  -> who we are: meeting id, user id, auth token
    3. Meteor DDP over SockJS       -> `validateAuthToken` makes us a PARTICIPANT
       (without it we are a join that never completed, and the room expires us)
    4. wss /bbb-webrtc-sfu          -> listen-only audio. The SERVER offers; we
       answer with `subscriberAnswer`; `webRTCAudioSuccess` means RTP is flowing.

LISTENING, OR TALKING

Listen-only is `start` with role `recv`: the server offers, we answer.

Talking is the SAME socket with role `sendrecv`, and the direction of the offer
flips: WE offer (with an outgoing audio track) and the server answers. Read off
a real room whose microphone uses the SFU "fullaudio" bridge; the server bridges
it into the FreeSWITCH conference itself, so no SIP client is needed. The track
we receive is then the conference mix without our own voice, which is exactly
what the recorder and the listener want. (A server still configured for the
old SIP bridge would refuse `sendrecv`; the refusal is reported.)

THE ONE SURPRISE

The SFU is `ice-lite`: it only answers connectivity checks and never starts
them, so the side that is not the server MUST be the controlling ICE agent.
aiortc answers as the controlled side, and two controlled agents never agree on
a pair. The role is therefore forced (see `_force_controlling`).

Written against one server's behaviour. A different BBB version (3.x moved from
Meteor to GraphQL) needs its own session code; the audio half is shared.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
import string
import time
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlparse

import fractions

import av
import httpx
import numpy as np
import websockets
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp

import logging

# aioice says why it gave up on a connection ("Consent to send expired") only
# at INFO; that line is the whole diagnosis when the audio drops.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.INFO)

RATE = 16_000
VOICE_RATE = 48_000  # what goes out: opus at its native rate
FRAME = VOICE_RATE // 50  # 20 ms


def log(*a):
    print(time.strftime("%H:%M:%S"), "[native]", *a, flush=True)


class JoinFailed(RuntimeError):
    """The room refused us, with a reason worth showing."""


# --------------------------------------------------------------------------
# One 16 kHz mono opus/webm segment, built in memory.
# --------------------------------------------------------------------------


class Segment:
    def __init__(self, index: int, start: float) -> None:
        self.index = index
        self.start = start  # seconds since the meeting was joined
        self.t0 = time.monotonic()
        self.samples = 0
        self.buf = io.BytesIO()
        self.out = av.open(self.buf, mode="w", format="webm")
        self.stream = self.out.add_stream("libopus", rate=RATE)
        self.stream.layout = "mono"

    def write(self, pcm: np.ndarray) -> None:
        """int16 mono samples at 16 kHz."""
        for i in range(0, len(pcm), 320):  # 20 ms frames
            chunk = pcm[i : i + 320]
            if len(chunk) < 320:
                chunk = np.pad(chunk, (0, 320 - len(chunk)))
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = RATE
            frame.pts = self.samples
            frame.time_base = __import__("fractions").Fraction(1, RATE)
            self.samples += 320
            for packet in self.stream.encode(frame):
                self.out.mux(packet)

    def seconds(self) -> float:
        return self.samples / RATE

    def finish(self) -> bytes:
        for packet in self.stream.encode(None):
            self.out.mux(packet)
        self.out.close()
        return self.buf.getvalue()


# --------------------------------------------------------------------------
# The bot's own voice: an audio track we write speech into
# --------------------------------------------------------------------------


class Voice(MediaStreamTrack):
    """An outgoing track that plays what it is given, and silence otherwise.

    Paced by the clock (one 20 ms frame per 20 ms), so speech pushed in a burst
    is played at speaking speed rather than sent all at once.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()  # int16 mono at 48 kHz
        self._pts = 0
        self._t0: float | None = None
        self.speaking_until = 0.0

    def push(self, pcm: bytes, rate: int) -> None:
        """Queue int16 mono speech at any rate (resampled here)."""
        x = np.frombuffer(pcm, dtype=np.int16)
        if rate != VOICE_RATE and x.size:
            n = int(x.size * VOICE_RATE / rate)
            x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.int16)
        self._buf.extend(x.tobytes())
        self.speaking_until = time.monotonic() + len(self._buf) / (VOICE_RATE * 2)

    def hush(self) -> None:
        self._buf.clear()
        self.speaking_until = 0.0

    async def recv(self) -> av.AudioFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        wait = self._t0 + self._pts / VOICE_RATE - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        chunk = bytes(self._buf[: FRAME * 2])
        del self._buf[: FRAME * 2]
        if len(chunk) < FRAME * 2:
            chunk += bytes(FRAME * 2 - len(chunk))
        frame = av.AudioFrame.from_ndarray(
            np.frombuffer(chunk, dtype=np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = VOICE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, VOICE_RATE)
        self._pts += FRAME
        return frame


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------

SegmentSink = Callable[[int, float, float, bytes], Awaitable[None]]


class BBBNative:
    def __init__(self, url: str, segment_seconds: int = 120, talk: bool = False) -> None:
        self.url = url
        # Talking: a microphone (our Voice track) instead of listen-only, and
        # every 16 kHz frame of the meeting handed to `on_pcm` as it arrives.
        self.talk = talk
        self.voice: Voice | None = Voice() if talk else None
        self.on_pcm: Callable[[bytes], None] | None = None
        self.segment_seconds = segment_seconds
        u = urlparse(url)
        self.base = f"{u.scheme}://{u.netloc}"
        self.host = u.netloc
        self.http = httpx.AsyncClient(follow_redirects=False, timeout=20)
        self.token = ""
        self.enter: dict = {}
        self.ice: list[RTCIceServer] = []

        self.validated = asyncio.Event()
        self.audio_up = asyncio.Event()
        self.ended = asyncio.Event()
        self.end_reason = ""
        self.users: dict[str, str] = {}
        self.last_sound = time.monotonic()
        self.joined_at = time.monotonic()
        self.packets = 0
        self.segments_sent = 0

        self._ddp: websockets.ClientConnection | None = None
        self._sfu: websockets.ClientConnection | None = None
        self._pc: RTCPeerConnection | None = None
        self._tasks: list[asyncio.Task] = []
        self._seg: Segment | None = None
        self._sink: SegmentSink | None = None
        self._seg_index = 0
        self._audio_lost = asyncio.Event()
        self._ddp_id = 100

    # ---- step 1 and 2: who are we ------------------------------------------

    async def authenticate(self) -> None:
        q = parse_qs(urlparse(self.url).query)
        if "sessionToken" in q:  # already the client URL
            self.token = q["sessionToken"][0]
        else:
            r = await self.http.get(self.url)
            loc = r.headers.get("location", "")
            if r.status_code not in (301, 302, 303) or "sessionToken" not in loc:
                raise JoinFailed(
                    f"The join link was not accepted (HTTP {r.status_code}). "
                    "It may have expired, or the meeting has ended."
                )
            self.token = parse_qs(urlparse(loc).query)["sessionToken"][0]
        e = await self.http.get(f"{self.base}/bigbluebutton/api/enter", params={"sessionToken": self.token})
        body = (e.json().get("response") or {}) if e.status_code == 200 else {}
        if body.get("returncode") != "SUCCESS":
            raise JoinFailed(f"The room refused entry: {body.get('message') or e.status_code}")
        self.enter = body
        s = await self.http.get(f"{self.base}/bigbluebutton/api/stuns", params={"sessionToken": self.token})
        servers: list[RTCIceServer] = []
        if s.status_code == 200:
            data = s.json()
            for st in data.get("stunServers", []):
                servers.append(RTCIceServer(urls=st["url"]))
            for t in data.get("turnServers", []):
                servers.append(RTCIceServer(urls=t["url"], username=t["username"], credential=t["password"]))
        self.ice = servers
        log("entered as", body.get("fullname"), "role", body.get("role"))

    # ---- step 3: become a participant ---------------------------------------

    def _headers(self) -> dict[str, str]:
        """What the browser sends on every websocket: the session cookie and its origin.

        The audio SFU refuses a socket without them (HTTP 401), though the
        session token in its URL is what it then authenticates by.
        """
        h = {"Origin": self.base}
        cookies = "; ".join(f"{k}={v}" for k, v in self.http.cookies.items())
        if cookies:
            h["Cookie"] = cookies
        return h

    def _frame(self, msg: dict) -> str:
        return json.dumps([json.dumps(msg)])

    async def _ddp_send(self, msg: dict) -> None:
        if self._ddp is not None:
            await self._ddp.send(self._frame(msg))

    async def _ddp_run(self) -> None:
        sid = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        url = f"wss://{self.host}/html5client/sockjs/{random.randint(100, 999)}/{sid}/websocket"
        try:
            async with websockets.connect(url, additional_headers=self._headers(), max_size=None) as ws:
                self._ddp = ws
                async for raw in ws:
                    if raw == "o":
                        await self._ddp_send({"msg": "connect", "version": "1", "support": ["1", "pre2", "pre1"]})
                        continue
                    if raw == "h" or not raw.startswith("a"):
                        if raw.startswith("c"):
                            break
                        continue
                    for item in json.loads(raw[1:]):
                        await self._on_ddp(json.loads(item))
        except Exception as exc:  # noqa: BLE001
            log("ddp ended:", type(exc).__name__, str(exc)[:80])
        finally:
            self._ddp = None
            self._end("The connection to the room closed.")

    async def _on_ddp(self, m: dict) -> None:
        kind = m.get("msg")
        if kind == "connected":
            e = self.enter
            await self._ddp_send({"msg": "sub", "id": "cu1", "name": "current-user", "params": []})
            await self._ddp_send(
                {
                    "msg": "method",
                    "id": "v1",
                    "method": "validateAuthToken",
                    "params": [e["meetingID"], e["internalUserID"], e["authToken"], e.get("externUserID", e["internalUserID"])],
                }
            )
        elif kind == "ping":
            await self._ddp_send({"msg": "pong", **({"id": m["id"]} if "id" in m else {})})
        elif kind == "result" and m.get("id") == "v1":
            res = m.get("result") or {}
            if res.get("validationStatus") == 3 or res.get("userId"):
                self.validated.set()
                for i, name in enumerate(("users", "meetings")):
                    await self._ddp_send({"msg": "sub", "id": f"s{i}", "name": name, "params": []})
                log("validated: now a participant")
            else:
                self._end(f"The room did not accept the session ({res.get('reason') or 'unknown'}).")
        elif kind in ("added", "changed", "removed"):
            coll = m.get("collection")
            fields = m.get("fields") or {}
            if coll == "users":
                if kind == "removed":
                    self.users.pop(m.get("id", ""), None)
                else:
                    name = fields.get("name")
                    if name:
                        self.users[m["id"]] = name
                    if fields.get("ejected") or fields.get("left") is True and fields.get("userId") == self.enter.get("internalUserID"):
                        self._end("The bot was removed from the meeting.")
            elif coll == "meetings" and fields.get("meetingEnded"):
                self._end("The meeting ended.")

    # ---- step 4: the audio ----------------------------------------------------

    @staticmethod
    def _force_controlling(pc: RTCPeerConnection) -> None:
        """The SFU is ice-lite, so we must be the controlling ICE agent.

        aiortc sets the role from whether it made the offer, and here the server
        did. Marking ourselves the initiator makes it controlling; nothing else
        about answering changes.
        """
        pc._RTCPeerConnection__isInitiator = True  # type: ignore[attr-defined]

    async def _sfu_run(self) -> None:
        """The audio, reconnected if it drops while the meeting goes on.

        A dropped audio leg is not the end of the meeting (the room itself is
        the DDP connection), so up to a handful of reconnects are made before
        giving up. Recording carries on into the same segment.
        """
        for attempt in range(6):
            self._audio_lost = asyncio.Event()
            await self._sfu_once(attempt + 1)
            if self.ended.is_set():
                return
            log("audio dropped; reconnecting (attempt", attempt + 2, ")")
            await asyncio.sleep(1)
        self._end("The audio connection kept dropping.")

    async def _sfu_once(self, session_number: int) -> None:
        url = f"wss://{self.host}/bbb-webrtc-sfu?sessionToken={self.token}"
        pc = RTCPeerConnection(RTCConfiguration(iceServers=self.ice))
        self._pc = pc

        @pc.on("track")
        def on_track(track):  # noqa: ANN001
            if track.kind == "audio":
                self._tasks.append(asyncio.create_task(self._record(track)))

        @pc.on("iceconnectionstatechange")
        async def on_ice():  # noqa: ANN202
            log("ice:", pc.iceConnectionState)
            if pc.iceConnectionState in ("failed", "closed"):
                self._audio_lost.set()

        start: dict = {"id": "start", "type": "audio", "role": "recv", "clientSessionNumber": session_number,
                       "extension": None, "transparentListenOnly": False}
        if self.voice is not None:
            # A microphone: we offer, with our track, and the server answers.
            # A track can only be sent once, so a reconnect gets a fresh Voice
            # (anything still queued is dropped with the old one).
            if session_number > 1:
                self.voice = Voice()
            pc.addTransceiver(self.voice, direction="sendrecv")
            await pc.setLocalDescription(await pc.createOffer())
            start.update(role="sendrecv", sdpOffer=pc.localDescription.sdp)

        try:
            async with websockets.connect(url, additional_headers=self._headers(), max_size=None) as ws:
                self._sfu = ws
                await ws.send(json.dumps({"id": "ping"}))
                await ws.send(json.dumps(start))
                keep = asyncio.create_task(self._sfu_ping(ws))
                lost = asyncio.create_task(self._close_when_lost(ws))
                try:
                    async for raw in ws:
                        m = json.loads(raw)
                        mid = m.get("id")
                        if mid not in ("pong", "iceCandidate"):
                            log("sfu:", json.dumps({k: v for k, v in m.items() if k != "sdpAnswer"})[:200])
                        if mid == "startResponse":
                            if m.get("response") != "accepted":
                                self._end(f"The audio was refused: {m.get('message') or m.get('response')}")
                                return
                            if self.voice is not None:
                                # Our offer, their answer; as the offerer we
                                # are already the controlling ICE agent.
                                await pc.setRemoteDescription(RTCSessionDescription(sdp=m["sdpAnswer"], type="answer"))
                                continue
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=m["sdpAnswer"], type="offer"))
                            answer = await pc.createAnswer()
                            self._force_controlling(pc)
                            await pc.setLocalDescription(answer)
                            await ws.send(
                                json.dumps(
                                    {"id": "subscriberAnswer", "type": "audio", "role": "recv",
                                     "sdpOffer": pc.localDescription.sdp}
                                )
                            )
                        elif mid == "iceCandidate":
                            cand = m.get("candidate")
                            if cand and cand.get("candidate"):
                                c = candidate_from_sdp(cand["candidate"].split(":", 1)[1])
                                c.sdpMid, c.sdpMLineIndex = cand.get("sdpMid"), cand.get("sdpMLineIndex")
                                await pc.addIceCandidate(c)
                        elif mid == "webRTCAudioSuccess":
                            log("audio:", m.get("success"))
                            self.audio_up.set()
                        elif mid in ("error", "stopped") or m.get("type") == "audio" and m.get("id") == "close":
                            log("audio session stopped:", m.get("message") or mid)
                            return
                finally:
                    keep.cancel()
                    lost.cancel()
        except Exception as exc:  # noqa: BLE001
            log("sfu ended:", type(exc).__name__, str(exc)[:80])
        finally:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass

    async def _close_when_lost(self, ws) -> None:  # noqa: ANN001
        await self._audio_lost.wait()
        await ws.close()

    async def _sfu_ping(self, ws) -> None:  # noqa: ANN001
        while True:
            await asyncio.sleep(5)
            await ws.send(json.dumps({"id": "ping"}))

    # ---- recording ------------------------------------------------------------

    async def _flush(self, seg: Segment) -> None:
        data = seg.finish()
        if self._sink and seg.seconds() > 0.5:
            await self._sink(seg.index, seg.start, seg.seconds(), data)
            self.segments_sent += 1

    async def _record(self, track) -> None:  # noqa: ANN001
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        if self._seg is None:
            self._seg = Segment(self._seg_index, 0.0)
            self._seg_index += 1
        try:
            while not self.ended.is_set():
                try:
                    # Short, so the silence the server does not send (DTX) is
                    # filled in promptly: a listener needs it to hear a pause.
                    frame = await asyncio.wait_for(track.recv(), timeout=0.25)
                except asyncio.TimeoutError:
                    frame = None
                seg = self._seg
                assert seg is not None
                # Silence is not sent by the server (DTX), so fill the gap with
                # silence to keep the segment's clock honest: a transcript line
                # at 01:30 must be 01:30 into the meeting, not 01:12.
                elapsed = time.monotonic() - seg.t0
                missing = int(elapsed * RATE) - seg.samples
                if frame is None or missing > RATE // 2:
                    if missing > 320:
                        quiet = np.zeros(missing, dtype=np.int16)
                        seg.write(quiet)
                        self._tap(quiet)
                if frame is not None:
                    self.packets += 1
                    for f in resampler.resample(frame):
                        pcm = f.to_ndarray().reshape(-1).astype(np.int16)
                        if pcm.size:
                            if np.abs(pcm).max() > 130:  # ~ -48 dBFS
                                self.last_sound = time.monotonic()
                            seg.write(pcm)
                            self._tap(pcm)
                if time.monotonic() - seg.t0 >= self.segment_seconds:
                    nxt = Segment(self._seg_index, seg.start + seg.seconds())
                    self._seg_index += 1
                    self._seg = nxt
                    await self._flush(seg)
        except Exception as exc:  # noqa: BLE001
            # The track ended: the audio leg dropped. The SFU loop reconnects,
            # and the next track's recorder continues this segment.
            log("recorder stopped:", type(exc).__name__, str(exc)[:80])
            self._audio_lost.set()

    def _tap(self, pcm: np.ndarray) -> None:
        if self.on_pcm is not None:
            try:
                self.on_pcm(pcm.tobytes())
            except Exception as exc:  # noqa: BLE001 - listening must not stop recording
                log("listener failed:", str(exc)[:80])

    # ---- the whole thing ----------------------------------------------------------

    def _end(self, reason: str) -> None:
        if not self.ended.is_set():
            self.end_reason = reason
            self.ended.set()

    async def join(self, sink: SegmentSink, wait: float = 40.0) -> None:
        """Authenticate, become a participant, connect the audio. Raises JoinFailed."""
        self._sink = sink
        await self.authenticate()
        self._tasks.append(asyncio.create_task(self._ddp_run()))
        try:
            await asyncio.wait_for(self.validated.wait(), timeout=20)
        except asyncio.TimeoutError:
            await self.close()
            raise JoinFailed("The room did not accept the session in time.") from None
        self._tasks.append(asyncio.create_task(self._sfu_run()))
        try:
            await asyncio.wait_for(self.audio_up.wait(), timeout=wait)
        except asyncio.TimeoutError:
            reason = self.end_reason or "The audio did not connect (a firewall blocking UDP is the usual cause)."
            await self.close()
            raise JoinFailed(reason) from None
        self.joined_at = time.monotonic()
        self.last_sound = time.monotonic()

    async def close(self) -> None:
        """Leave properly (so the participant list updates), then tear down."""
        try:
            await self._ddp_send({"msg": "method", "id": "bye", "method": "userLeftMeeting", "params": []})
            await asyncio.sleep(0.3)
        except Exception:  # noqa: BLE001
            pass
        self.ended.set()
        # Whatever was recorded so far is uploaded before anything is torn down.
        seg, self._seg = self._seg, None
        if seg is not None:
            try:
                await self._flush(seg)
            except Exception as exc:  # noqa: BLE001
                log("final segment failed:", str(exc)[:80])
        for t in self._tasks:
            t.cancel()
        for closer in (self._pc.close() if self._pc else None,
                       self._sfu.close() if self._sfu else None,
                       self._ddp.close() if self._ddp else None,
                       self.http.aclose()):
            if closer is not None:
                try:
                    await closer
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------
# A standalone check:  python bbb_native.py <join-url> [seconds]
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    async def _main() -> None:
        seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        got: list[tuple[int, float, float, int]] = []

        async def sink(i: int, start: float, secs: float, data: bytes) -> None:
            got.append((i, start, secs, len(data)))
            open(f"/out/native-{i}.webm", "wb").write(data)

        bot = BBBNative(sys.argv[1], segment_seconds=10)
        await bot.join(sink)
        log("joined; listening for", seconds, "s; participants:", sorted(bot.users.values()))
        try:
            await asyncio.wait_for(bot.ended.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        log("participants now:", sorted(bot.users.values()), "| rtp packets:", bot.packets, "| ended:", bot.end_reason or "no")
        await bot.close()
        log("segments:", got)

    asyncio.run(_main())

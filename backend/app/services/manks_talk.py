"""Manks, talking: the Duplex pipeline pointed at a meeting instead of a person.

The bot in a meeting hears a MIX of everybody, so the single-speaker assumptions
Duplex makes in the browser do not hold: there is no headset to gate on, people
talk to each other, and the bot should stay quiet unless it is being spoken to.
So the front of the pipeline is different and the back is not.

    meeting audio (16 kHz PCM, from the bot)
      -> an energy detector cuts it into utterances
      -> Whisper writes each one down
      -> addressed to the bot? (its name, or a follow-up to its last answer)
           no  -> kept as context for later, nothing said
           yes -> handed to the SAME `duplex.Session` the browser uses:
                  speculative LLM with tools -> streaming Gemini Live voice
      -> 24 kHz PCM back to the bot, which plays it into the meeting

The utterance is transcribed once, before the Session sees it, so an answer
starts about a second later than in the browser; in exchange nothing is
answered that was not meant for the bot. Everything after that point
(drafts, tools, barge-in, history, the voice pool) is the code Duplex runs.
"""

from __future__ import annotations

import array
import asyncio
import json
import math
import re
import time
from collections import deque

import structlog

from app.config import get_settings
from app.services import duplex, groq_client, keys, voice

log = structlog.get_logger()

RATE = duplex.INPUT_RATE
FRAME = RATE // 20  # 50 ms
END_SILENCE_FRAMES = 14  # 700 ms of quiet ends an utterance
MIN_SPEECH_FRAMES = 8  # under 400 ms of voice is a cough, not a sentence
MAX_UTTERANCE_FRAMES = 20 * 20  # twenty seconds
FOLLOW_UP_SECONDS = 25  # after the bot speaks, a reply needs no name
CONTEXT_LINES = 14
# Join in on everything said, rather than wait to be named. The meeting's own
# people talk to each other, so this is chatty; the prompt asks for brevity.
PROACTIVE = True

# What Whisper writes when it hears "Manks". The vowel is anybody's guess.
_NAME_FORMS = ("manks", "manx", "monks", "mank", "banks", "menks", "munks")


def addressed(text: str, name: str) -> bool:
    """Is this utterance for the bot? Its name, spelled however Whisper spelled it."""
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    mine = name.lower().strip()
    forms = {mine, *_NAME_FORMS} if mine in _NAME_FORMS or mine == "manks" else {mine}
    return any(f and f in t.split() for f in forms)


class Utterances:
    """Cuts a stream of PCM into utterances, with an adaptive noise floor."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._floor = 200.0
        self._active = False
        self._voiced = 0
        self._quiet = 0
        self._frames: list[bytes] = []
        self._pre: deque[bytes] = deque(maxlen=6)  # 300 ms before the onset

    @staticmethod
    def _rms(frame: bytes) -> float:
        samples = array.array("h")
        samples.frombytes(frame)
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    def feed(self, chunk: bytes) -> list[bytes]:
        """Return every utterance this chunk completed."""
        done: list[bytes] = []
        self._pending.extend(chunk)
        step = FRAME * 2
        while len(self._pending) >= step:
            frame = bytes(self._pending[:step])
            del self._pending[:step]
            level = self._rms(frame)
            loud = level > max(350.0, self._floor * 3.2)
            if not self._active:
                # The floor follows the room while nobody is talking.
                self._floor = 0.97 * self._floor + 0.03 * min(level, 1500.0)
                self._pre.append(frame)
                self._voiced = self._voiced + 1 if loud else 0
                if self._voiced >= 2:
                    self._active = True
                    self._frames = list(self._pre)
                    self._quiet = 0
                continue
            self._frames.append(frame)
            self._quiet = 0 if loud else self._quiet + 1
            if self._quiet >= END_SILENCE_FRAMES or len(self._frames) >= MAX_UTTERANCE_FRAMES:
                voiced = len(self._frames) - self._quiet
                if voiced >= MIN_SPEECH_FRAMES:
                    done.append(b"".join(self._frames))
                self._active = False
                self._voiced = 0
                self._frames = []
                self._pre.clear()
        return done


class _Out:
    """Stands in for the browser's socket: Session writes here, the bot reads."""

    def __init__(self, driver: "Talker") -> None:
        self.driver = driver

    async def send_text(self, raw: str) -> None:
        await self.driver.on_event(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        await self.driver.send_audio(data)


class Talker:
    """One meeting's conversation. `ws` is the bot's socket."""

    def __init__(self, ws, owner_id: str | None, bot_name: str, title: str) -> None:
        from app.api import duplex as duplex_api  # late: that module imports services

        self.ws = ws
        self.name = bot_name
        self.title = title
        self.utterances = Utterances()
        self.context: deque[str] = deque(maxlen=CONTEXT_LINES)
        self.follow_until = 0.0
        self.busy = asyncio.Lock()
        self.session = duplex_api.Session(_Out(self), get_settings().voice_default, owner_id)
        self.session.system += system_for(bot_name, title)
        self._base_system = self.session.system
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> str | None:
        """Open the voice. Returns a reason it cannot talk, or None."""
        s = get_settings()
        if not duplex.usable(duplex.parse_chain(s.duplex_llm_chain)):
            return "Talking needs a Groq or Gemini key for the answers."
        if not keys.key_for("groq"):
            return "Talking needs a Groq key to write down what people say."
        if not keys.key_for("gemini"):
            return "Talking needs a Gemini key for the voice."
        self.session.live_voice = await duplex.start_voice(self.session.voice)
        return None

    # ---- bot -> us -----------------------------------------------------------

    def hear(self, chunk: bytes) -> None:
        for pcm in self.utterances.feed(chunk):
            task = asyncio.create_task(self._utterance(pcm))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _utterance(self, pcm: bytes) -> None:
        try:
            result = await groq_client.transcribe(
                voice.to_wav(pcm, rate=RATE),
                filename="utterance.wav",
                prompt=self.title or None,
                # Left to detect, a quiet patch comes back as Portuguese
                # "Obrigado" or a Russian subtitle credit.
                language="en",
            )
        except Exception as exc:  # noqa: BLE001 - one missed sentence is not fatal
            log.warning("manks_talk_whisper_failed", error=str(exc)[:120])
            return
        text = (result.get("text") or "").strip()
        seconds = len(pcm) / (RATE * 2)
        if not text or duplex.is_hallucination(text, seconds, result.get("no_speech_prob")):
            return
        await self.on_heard(text)

    async def on_heard(self, text: str) -> None:
        """Somebody said `text`. Answer it if it was for the bot."""
        s = self.session
        named = addressed(text, self.name)
        direct = named or PROACTIVE
        follow = time.monotonic() < self.follow_until
        log.info("manks_talk_heard", text=text[:100], direct=direct, follow=follow, state=s.state)
        if not (direct or follow):
            self.context.append(text)
            return
        async with self.busy:
            if s.state == "speaking":
                # Chatter does not cut the bot off; being called by name does.
                if not named:
                    self.context.append(text)
                    return
                await s.interrupt(reason="addressed")
            elif s.state != "idle":
                self.context.append(text)
                return
            # What was said around the bot goes in front of the question, so
            # "what do you think about that?" has a that.
            around = "\n".join(self.context)
            s.system = self._base_system + (
                f"\n<meeting_so_far>\n{around}\n</meeting_so_far>\n" if around else ""
            )
            self.context.clear()
            await s.speech_start()
            await s.on_text(text)
            await s.endpoint()

    async def joined(self) -> None:
        """The bot is in the room: introduce itself, out loud."""
        s = self.session
        if s.state != "idle":
            return
        async with self.busy:
            s.system = self._base_system
            await s.speech_start()
            await s.on_text(
                "(You have just joined the meeting. Say hello and what you are in one short "
                "sentence, and say you will chip in as people talk.)"
            )
            await s.endpoint()

    # ---- us -> bot -----------------------------------------------------------

    async def on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind in ("stop", "say", "error"):
            await self._send_text(event)
        elif kind == "turn_end":
            self.follow_until = time.monotonic() + FOLLOW_UP_SECONDS
            await self._send_text({"type": "turn_end"})

    async def _send_text(self, event: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(event))
        except Exception:  # noqa: BLE001 - the bot is going away
            pass

    async def send_audio(self, data: bytes) -> None:
        try:
            await self.ws.send_bytes(data)
        except Exception:  # noqa: BLE001
            pass

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self.session.close()


def system_for(name: str, title: str) -> str:
    """The standing instructions for a bot sitting in a meeting."""
    return (
        f"\nYou are {name}, an AI colleague who is sitting in a live meeting"
        + (f' titled "{title}"' if title else "")
        + " and can be spoken to out loud. Several people are in the call and "
        "you only hear them through a transcript, so you cannot tell who is "
        "speaking unless they say. You take an active part: answer questions, "
        "add a useful fact or a sharp question when you have one, and keep the "
        "conversation moving. Keep it to one or two short sentences: this is a "
        "meeting, not a lecture, and people are waiting to get on. Use the "
        "meeting so far when it helps, and use your tools when you need a fact "
        "you do not have.\n"
    )

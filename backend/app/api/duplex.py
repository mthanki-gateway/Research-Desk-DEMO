"""Parley > Duplex: the socket.

browser -> binary                16 kHz PCM, always (the STT hears everything)
        -> {"type":"speech_start"}   the browser's VAD heard speech begin
        -> {"type":"pause"}          ~250 ms of silence: the early signal
        -> {"type":"endpoint"}       Silero + Smart Turn say the speaker finished
        -> {"type":"interrupt"}      stop talking (button)
server  -> binary                24 kHz PCM, the spoken answer
        -> {"type":"partial","text"}         the transcript so far
        -> {"type":"draft","state",...}      an answer started / adopted / discarded
        -> {"type":"say","text"}             the sentence now being spoken
        -> {"type":"latency",...}            what this turn cost, in ms
        -> {"type":"turn_end"} / {"type":"stop"} / {"type":"error","detail"}

THE EARLY ANSWER. `pause` arrives ~250 ms after the last word; the turn
detector's `endpoint` ~350 ms later. On `pause` the transcript is brought up to
date and an answer is started for it. If the words are still the same at
`endpoint`, that answer is already under way and is simply adopted; if they
changed, it is discarded and a fresh one starts. See `services/duplex.py` for
what was measured and why it is shaped this way.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.agent import tools as agent_tools
from app.auth import User, _verify, current_user
from app.config import get_settings
from app.services import duplex, invites, keys, live, profile, voice

log = structlog.get_logger()

router = APIRouter(tags=["duplex"])

# Rough size of one answer's prompt and output, for the speculation budget.
_EST_TOKENS = 500


@router.get("/duplex/status")
async def status(user: User = Depends(current_user)) -> dict:
    """What this caller can actually run, so the page can say so up front."""
    s = get_settings()
    chain = duplex.parse_chain(s.duplex_llm_chain)
    ready = duplex.usable(chain)
    has_gemini = bool(keys.key_for("gemini"))
    has_groq = bool(keys.key_for("groq"))
    return {
        "enabled": has_gemini and bool(ready),
        "stt": "groq" if has_groq else ("gemini" if has_gemini else None),
        "tts": "live" if has_gemini and s.duplex_tts == "live" else ("batch" if has_gemini else None),
        "chain": [t.name for t in chain],
        "usable": [t.name for t in ready],
        "voices": voice.VOICES,
        "default_voice": s.voice_default,
        "max_speculations": s.duplex_max_speculations,
        "pause_ms": s.duplex_pause_ms,
        "tools": [d["name"] for d in available_tools()],
        "input_rate": duplex.INPUT_RATE,
        "output_rate": duplex.OUTPUT_RATE,
    }


# Tools that make no sense without a stored chat session behind them.
_NOT_HERE = {agent_tools.REMEMBER_PREFERENCE, agent_tools.CONVERSATION_STATS}
MAX_TOOL_CALLS_PER_TURN = 4


def available_tools() -> list[dict]:
    """The typed agent's own tools (web, documents), as plain declarations."""
    declared = agent_tools.tool_specs()[0]["functionDeclarations"]
    return [d for d in declared if d["name"] not in _NOT_HERE]


def interview_tools() -> list[dict]:
    """What the interviewer may call: the web for placing a name, and the end."""
    allowed = set(live.SPOKEN_TOOLS["howler"])
    declared = [d for d in available_tools() if d["name"] in allowed]
    return [*declared, live.END_ONLY]


INTERVIEW_ADDENDUM = """

<spoken_here>
Everything the interviewer writes is spoken aloud the instant it is written. One question at a time, at most two short sentences, no lists or markdown. When something needs placing, say a few words first ("One second.") and then use search_web. When the interview is over, say goodbye in one sentence and call end_interview.
</spoken_here>
"""


class Session:
    """One connected speaker, and every answer in flight for them."""

    def __init__(
        self,
        ws: WebSocket,
        voice_name: str,
        owner_id: str | None = None,
        interview: dict | None = None,
    ) -> None:
        self.ws = ws
        self.owner_id = owner_id
        # A Howler interview: somebody holding a link, with a stored
        # conversation, a schema and an interviewer prompt. None = plain Duplex.
        self.interview = interview
        self.chat_id: uuid.UUID | None = interview["chat_id"] if interview else None
        self.turns = 0
        self.finishing = False
        self._question = ""
        self._clip: bytes | None = None
        self._pre = bytearray()
        self._turn_pcm = bytearray()
        if interview:
            self.tools = interview_tools()
            self.system = interview["system"]
        else:
            self.tools = available_tools()
            self.system = duplex.SYSTEM + (
                duplex.TOOLS_PROMPT if self.tools else duplex.NO_TOOLS_PROMPT
            )
        # What a tool returned for these exact arguments this turn. Speculative
        # drafts repeat each other's searches, and each one is a metered call.
        self._tool_cache: dict[str, str] = {}
        self._tool_calls = 0
        s = get_settings()
        self.s = s
        self.voice = voice_name if voice_name in {v["id"] for v in voice.VOICES} else s.voice_default
        self.chain = duplex.parse_chain(s.duplex_llm_chain)
        self.history: list[dict] = []

        self.transcriber: duplex.Transcriber | None = None
        self.live_voice: duplex.LiveVoice | None = None
        self.transcript = ""
        self.transcript_at = 0.0

        self.state = "idle"  # idle | listening | settling | speaking
        self.draft: duplex.Draft | None = None
        self.speculations = 0
        self.t_endpoint = 0.0
        self.spoken: list[str] = []
        self._urgent = False

        self._debounce: asyncio.Task | None = None
        self._turn: asyncio.Task | None = None
        self._pass: asyncio.Task | None = None
        self._changed = asyncio.Event()
        self._send_lock = asyncio.Lock()

    # ---- the participant's own audio ------------------------------------------

    def hear(self, chunk: bytes) -> None:
        """Keep the turn's audio for the post-call pass (interviews only)."""
        if not self.interview:
            return
        if self.state in ("listening", "settling"):
            self._turn_pcm.extend(chunk)
        else:
            self._pre.extend(chunk)
            del self._pre[: max(0, len(self._pre) - duplex.WhisperTranscriber.PRE_ROLL_BYTES)]

    async def _save_turn(self, answer: str, tools: list[str] | None = None) -> None:
        """Store one exchange: what they said, what was spoken, their audio.

        The participant's words are the live transcript for now; the post-call
        pass re-transcribes the clip properly and replaces them, keeping this
        as `live_heard`. The interviewer's words are stored exactly as spoken.
        """
        if not self.chat_id:
            return
        clip, self._clip = self._clip, None
        question, self._question = self._question, ""
        if clip is not None and len(clip) < 12_800:  # under 0.4 s: not speech
            clip = None
        await live.save_turn(self.chat_id, question, answer, [], tools or [], clip=clip)
        if question:
            self.turns += 1

    # ---- outbound -----------------------------------------------------------

    async def send(self, **message) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(message))
            except Exception:  # noqa: BLE001 - the socket is going away
                pass

    async def send_audio(self, pcm: bytes) -> None:
        # Slices a few hundred ms long, so a barge-in is heard within one.
        step = duplex.OUTPUT_RATE * 2 // 4
        async with self._send_lock:
            for i in range(0, len(pcm), step):
                await self.ws.send_bytes(pcm[i : i + step])

    # ---- the transcript ------------------------------------------------------

    async def on_text(self, text: str) -> None:
        self.transcript = text
        self.transcript_at = time.monotonic()
        self._changed.set()
        await self.send(type="partial", text=text)
        if self.state == "listening":
            urgent, self._urgent = self._urgent, False
            self._schedule_speculation(urgent)

    def _schedule_speculation(self, urgent: bool) -> None:
        if self._debounce and not self._debounce.done():
            self._debounce.cancel()
        self._debounce = asyncio.create_task(self._after_quiet(urgent))

    async def _after_quiet(self, urgent: bool) -> None:
        # A pause already waited 250 ms of silence; waiting again would spend
        # the head start it exists to buy.
        delay = 0 if urgent else self.s.duplex_debounce_ms / 1000
        try:
            if delay:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self.state == "listening":
            await self.draft_for(self.transcript, final=False, urgent=urgent)

    # ---- tools ---------------------------------------------------------------

    async def run_tool(self, call: duplex.ToolCall) -> str:
        """Execute one tool for a draft. Never raises; cached per turn."""
        if call.name == profile.END_TOOL:
            return await self._end_interview()
        key = f"{call.name}:{json.dumps(call.args, sort_keys=True)}"
        if key in self._tool_cache:
            return self._tool_cache[key]
        if self._tool_calls >= MAX_TOOL_CALLS_PER_TURN:
            return "No more lookups are allowed this turn. Answer with what you have."
        self._tool_calls += 1
        await self.send(
            type="tool", name=call.name, query=str(call.args.get("query") or call.args)[:120]
        )
        try:
            hits, observation = await agent_tools.run_tool(
                call.name,
                dict(call.args),
                top_k=5,
                document_ids=None,
                owner_id=self.owner_id,
                session_id=None,
            )
        except Exception as exc:  # noqa: BLE001 - the model can read a failure
            log.warning("duplex_tool_failed", tool=call.name, error=str(exc)[:160])
            return f"The {call.name} tool failed ({type(exc).__name__}). Say so plainly."
        sources = []
        for h in hits[:5]:
            sources.append(
                {
                    "label": getattr(h, "filename", ""),
                    "kind": getattr(h, "source", "document"),
                    "url": getattr(h, "url", None),
                }
            )
        if sources:
            await self.send(type="sources", sources=sources)
        self._tool_cache[key] = observation
        return observation

    async def _end_interview(self) -> str:
        if not self.chat_id:
            return "There is no interview to end."
        # The same guard the audio model has: not on the turn the profile
        # completed, because the answer to "anything to add?" is the part the
        # participant chose to say.
        if not await live._may_end(self.chat_id, self.turns, self.interview["fields"]):
            return (
                "NOT YET. You have just asked whether they want to add anything "
                "and they have not answered. Wait for their reply first."
            )
        await live.store_summary(self.chat_id, "", None)
        self.finishing = True
        return "The interview is closed. Say a brief goodbye in one sentence and stop asking questions."

    async def greet(self) -> None:
        """The interviewer speaks first, as it does in the audio-to-audio mode."""
        if not self.interview or self.state != "idle" or (self._turn and not self._turn.done()):
            return
        cue = (
            "(The participant has rejoined after a dropped connection. Welcome "
            "them back in one sentence and carry on from where you left off.)"
            if self.history
            else "(The participant has just joined. Greet them and begin the interview.)"
        )
        self._turn = asyncio.create_task(self._greeting(cue))

    async def _greeting(self, cue: str) -> None:
        try:
            self.t_endpoint = time.monotonic()
            self.state = "speaking"
            self.spoken = []
            draft = duplex.Draft(for_text=cue)
            # The cue is the model's instruction for THIS turn only; it is not
            # kept in the history, which holds what was actually said.
            draft.task = asyncio.create_task(
                duplex.write_draft(
                    draft,
                    self.chain,
                    self.history,
                    tools=self.tools or None,
                    run_tool=self.run_tool,
                    system=self.system,
                )
            )
            self.draft = draft
            outcome = await self._speak(draft, ready_at_endpoint=False, synthetic=True)
            if outcome == "done":
                await self._save_turn(" ".join(self.spoken), draft.tools)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("duplex_greeting_failed")
            await self.send(type="error", detail=keys.explain(exc, "gemini"))
        finally:
            if self.state == "speaking":
                self.state = "idle"
            await self.send(type="turn_end")
            if self.finishing:
                await self.send(type="finished")

    async def finish(self) -> None:
        """The PARTICIPANT ended it: same bookkeeping as the audio mode."""
        if not self.chat_id:
            return
        await self.interrupt(reason="finish")
        await live.finish_interview(self.chat_id)
        await self.send(type="finished")

    # ---- drafts --------------------------------------------------------------

    async def draft_for(self, text: str, *, final: bool, urgent: bool = False) -> bool:
        """Start an answer for `text`, discarding the one for older words.

        Speculative drafts are OPTIONAL and budgeted; the final one is not.
        A draft started on a PAUSE is exempt from the per-turn cap (it is the
        one most likely to be adopted) but not from the rate budget.
        Returns whether a draft for these words now exists.
        """
        if len(duplex.words(text)) < 2:
            return False
        if self.draft and duplex.same_words(self.draft.for_text, text) and not self.draft.failed:
            return True

        if not final:
            if not urgent and self.speculations >= self.s.duplex_max_speculations:
                return False
            gov = duplex.governor()
            if gov.peek(_EST_TOKENS) > 0:
                await self.send(type="draft", state="skipped", reason="rate budget")
                return False
            await gov.acquire(_EST_TOKENS)
            self.speculations += 1
        else:
            try:
                await asyncio.wait_for(duplex.governor().acquire(_EST_TOKENS), 0.25)
            except asyncio.TimeoutError:
                pass  # the answer is owed regardless of the budget

        self.discard_draft("newer words")
        draft = duplex.Draft(for_text=text)
        draft.task = asyncio.create_task(
            duplex.write_draft(
                draft,
                self.chain,
                self.history,
                tools=self.tools or None,
                run_tool=self.run_tool if self.tools else None,
                system=self.system,
            )
        )
        self.draft = draft
        await self.send(type="draft", state="started", text=text, final=final)
        return True

    def discard_draft(self, why: str) -> None:
        if self.draft is None:
            return
        self.draft.cancel()
        log.info("duplex_draft_discarded", why=why, model=self.draft.model)
        asyncio.create_task(self.send(type="draft", state="discarded", text=self.draft.for_text))
        self.draft = None

    # ---- the browser's events ------------------------------------------------

    async def speech_start(self) -> None:
        if self.state == "listening":
            return
        if self.state == "settling":
            # They were not finished -- a long pause, not an end. Keep every
            # word so far and keep listening; nothing was spoken yet.
            if self._turn and not self._turn.done():
                self._turn.cancel()
            self.state = "listening"
            if self.transcriber:
                await self.transcriber.speech_resume()
            await self.send(type="resumed")
            return
        if self.state == "speaking":
            await self.interrupt(reason="barge-in")
        self.state = "listening"
        self.speculations = 0
        self._tool_cache = {}
        self._tool_calls = 0
        self.transcript = ""
        self.spoken = []
        self._urgent = False
        self._changed.clear()
        self.discard_draft("new turn")
        self._turn_pcm = bytearray(self._pre)
        self._pre.clear()
        if self.transcriber:
            await self.transcriber.speech_start()

    async def pause(self) -> None:
        """~250 ms of silence: bring the transcript up to date and start early."""
        if self.state != "listening" or not self.transcriber:
            return
        self._urgent = True
        # A task, so a slow pass never holds up the socket's receive loop.
        if self._pass and not self._pass.done():
            return
        self._pass = asyncio.create_task(self.transcriber.pass_now())

    async def endpoint(self) -> None:
        if self.state != "listening":
            return
        self.state = "settling"
        self.t_endpoint = time.monotonic()
        self._clip = bytes(self._turn_pcm)
        if self._debounce and not self._debounce.done():
            self._debounce.cancel()
        self._turn = asyncio.create_task(self._answer())

    async def interrupt(self, *, reason: str = "button") -> None:
        for task in (self._turn, self._debounce):
            if task and not task.done():
                task.cancel()
        self.discard_draft(reason)
        if self.state == "speaking":
            if self.interview and (self._question or self.spoken):
                asyncio.create_task(self._save_turn(" ".join(self.spoken)))
            if self.spoken:
                # Remember what was actually said, not what was planned.
                self.history.append({"role": "assistant", "content": " ".join(self.spoken)})
            elif self.history and self.history[-1]["role"] == "user":
                # Nothing was said: the question is still open, not answered.
                self.history.pop()
        self.state = "idle"
        await self.send(type="stop", reason=reason)

    # ---- answering -----------------------------------------------------------

    async def _answer(self) -> None:
        try:
            await self._answer_inner()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("duplex_turn_failed")
            await self.send(type="error", detail=keys.explain(exc, "gemini"))
            self.state = "idle"

    async def _settle(self) -> None:
        """Wait for the transcript's last words, but not longer than needed."""
        quiet = self.transcriber.quiet_s if self.transcriber else 0.0
        deadline = self.t_endpoint + self.s.duplex_settle_ms / 1000
        while time.monotonic() < deadline:
            if self.transcript and time.monotonic() - self.transcript_at >= quiet:
                return
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    async def _answer_inner(self) -> None:
        # The last transcription pass covers any audio since the pause pass.
        if self.transcriber:
            await self.transcriber.speech_end()
        await self._settle()
        text = self.transcript.strip()
        if len(duplex.words(text)) < 1:
            self.state = "idle"
            await self.send(type="turn_end", empty=True)
            return

        await self.send(type="final", text=text)
        self.history.append({"role": "user", "content": text})
        self._question = text

        for _attempt in range(3):
            had_draft = bool(self.draft and duplex.same_words(self.draft.for_text, text))
            ready = bool(had_draft and self.draft and self.draft.pieces.qsize() > 0)
            if not await self.draft_for(text, final=True):
                break
            draft = self.draft
            assert draft is not None
            await self.send(
                type="draft",
                state="adopted" if had_draft else "fresh",
                text=text,
                model=draft.model,
            )
            self.state = "speaking"
            outcome = await self._speak(draft, ready_at_endpoint=ready)
            if outcome != "restart":
                if self.interview:
                    await self._save_turn(" ".join(self.spoken), draft.tools)
                break
            text = self.transcript.strip()
            self.history[-1] = {"role": "user", "content": text}
            self.state = "settling"
            await self.send(type="final", text=text)

        if self.state == "speaking":
            self.state = "idle"
        await self.send(type="turn_end")
        if self.finishing:
            await self.send(type="finished")

    async def _pump(self, piece: str, out: asyncio.Queue, sem: asyncio.Semaphore) -> None:
        """Synthesise one piece into `out`: audio chunks, then None (or an error)."""
        try:
            got = False
            if self.live_voice:
                try:
                    async for chunk in self.live_voice.speak(piece):
                        got = True
                        out.put_nowait(chunk)
                except Exception as exc:  # noqa: BLE001
                    if got:
                        raise
                    log.warning("duplex_live_tts_failed", error=type(exc).__name__)
                if got:
                    out.put_nowait(None)
                    return
            async with sem:
                wav, _rate = await voice.speak(piece, voice=self.voice)
            out.put_nowait(duplex.pcm_of(wav))
            out.put_nowait(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            out.put_nowait(exc)

    async def _speak(
        self, draft: duplex.Draft, *, ready_at_endpoint: bool, synthetic: bool = False
    ) -> str:
        """Speak a draft's pieces as they arrive. "done" | "restart" | "failed"."""
        sem = asyncio.Semaphore(2)
        queue: asyncio.Queue = asyncio.Queue()

        async def produce() -> None:
            # Synthesis of piece N+1 starts the moment it is written, while N
            # is still playing.
            async for piece in duplex.pieces_of(draft):
                audio: asyncio.Queue = asyncio.Queue()
                queue.put_nowait((piece, audio, asyncio.create_task(self._pump(piece, audio, sem))))
            queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        first = True
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                piece, audio, _task = item
                started = False
                while True:
                    chunk = await audio.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        await self.send(type="error", detail=keys.explain(chunk, "gemini"))
                        return "failed"
                    if first:
                        first = False
                        # By the time the first sound exists the transcript has
                        # had time to finish. If it changed, this answer is for
                        # the wrong question -- say nothing, write the right one.
                        if not synthetic and not duplex.same_words(draft.for_text, self.transcript):
                            producer.cancel()
                            return "restart"
                        await self.send(
                            type="latency",
                            endpoint_to_audio_ms=round((time.monotonic() - self.t_endpoint) * 1000),
                            first_token_ms=draft.first_token_ms,
                            model=draft.model,
                            ready_at_endpoint=ready_at_endpoint,
                        )
                    if not started:
                        started = True
                        self.spoken.append(piece)
                        await self.send(type="say", text=piece)
                    await self.send_audio(chunk)
            if draft.failed and first:
                await self.send(type="error", detail=draft.failed)
                return "failed"
            if self.spoken:
                self.history.append({"role": "assistant", "content": " ".join(self.spoken)})
                self.history[:] = self.history[-12:]
            return "done"
        finally:
            producer.cancel()
            while not queue.empty():
                item = queue.get_nowait()
                if item:
                    item[2].cancel()

    async def close(self) -> None:
        for task in (self._turn, self._debounce, self._pass):
            if task and not task.done():
                task.cancel()
        self.discard_draft("closed")
        if self.transcriber:
            await self.transcriber.close()
        if self.live_voice:
            await self.live_voice.close()


async def live_history(chat_id: uuid.UUID) -> list[dict]:
    """The last few exchanges of a conversation, so a rejoined call continues."""
    from sqlalchemy import select

    from app.db.models import Message
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        rows = list(
            (
                await db.execute(
                    select(Message)
                    .where(Message.session_id == chat_id)
                    .order_by(Message.created_at.desc())
                    .limit(12)
                )
            ).scalars()
        )
    return [
        {"role": "user" if m.role.value == "user" else "assistant", "content": m.content}
        for m in reversed(rows)
        if (m.content or "").strip() and m.content != "(nothing intelligible)"
    ]


@router.websocket("/duplex/ws")
async def duplex_socket(
    ws: WebSocket,
    token: str = Query(default=""),
    voice_name: str = Query(default=""),
    invite: str = Query(default=""),
) -> None:
    await ws.accept()

    settings = get_settings()
    user: User | None
    interview: dict | None = None
    if invite:
        # A guest on a Howler link: resolved by the invite, never as a person,
        # exactly as in the audio-to-audio socket. The conversation, schema and
        # keys all come from the project behind it.
        try:
            row, project = await invites.claim(invite)
        except invites.InviteError as exc:
            await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
            await ws.close(code=4403)
            return
        from app.api.live import _GuestUser

        user = _GuestUser(project.owner_id)  # type: ignore[assignment]
        await keys.bind(project.owner_id)
        chat = await live.open_conversation(
            project.owner_id, str(row.session_id) if row.session_id else None, "howler"
        )
        fields = list(project.fields or [])
        vocabulary = list(project.vocabulary or [])
        participant = row.participant or project.participant or ""
        await live.adopt_project(
            chat.id,
            project.id,
            project.brief or "",
            participant,
            fields,
            row.id,
            vocabulary,
            (row.label or "").strip() or project.title or "",
        )
        await live.mark_interface(chat.id, "duplex")
        interview = {
            "chat_id": chat.id,
            "fields": fields,
            "system": live._system(
                "howler", project.brief or "", participant, vocabulary, fields
            )
            + INTERVIEW_ADDENDUM,
        }
    elif not settings.auth_enabled:
        from app.auth import ANONYMOUS

        user = ANONYMOUS
    else:
        try:
            user = await _verify(token) if token else None
        except Exception:  # noqa: BLE001
            user = None
    if user is None:
        await ws.send_text(json.dumps({"type": "error", "detail": "Not authenticated."}))
        await ws.close(code=4401)
        return

    if not interview:
        await keys.bind(user.owner_id)

    async def refuse(detail: str) -> None:
        await ws.send_text(json.dumps({"type": "error", "detail": detail}))
        await ws.close(code=1011)

    chain = duplex.parse_chain(settings.duplex_llm_chain)
    if not duplex.usable(chain):
        await refuse(
            "Duplex needs a Groq or Gemini key for the answers. "
            "Add one in Settings → API keys."
        )
        return
    if not keys.key_for("gemini"):
        await refuse(
            "Duplex speaks through Gemini's voices; add a Gemini key in Settings → API keys."
        )
        return

    session = Session(ws, voice_name, user.owner_id, interview)
    if interview:
        session.history = await live_history(interview["chat_id"])
    # Both open together: each takes about a second to connect.
    session.transcriber, session.live_voice = await asyncio.gather(
        duplex.start_transcriber(session.on_text), duplex.start_voice(session.voice)
    )
    if session.transcriber is None:
        await session.close()
        await refuse("No speech-to-text is available; add a Groq or Gemini key.")
        return
    await session.send(
        type="ready",
        stt=session.transcriber.name,
        tts="gemini-live" if session.live_voice else "gemini-tts",
        chain=[t.name for t in chain],
    )

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            chunk = message.get("bytes")
            if chunk:
                session.hear(chunk)
                t = session.transcriber
                await t.feed(chunk)
                # The live STT died mid-session: carry on with Whisper.
                if isinstance(t, duplex.GeminiTranscriber) and t.dead and keys.key_for("groq"):
                    await t.close()
                    session.transcriber = duplex.WhisperTranscriber(session.on_text)
                    await session.send(type="stt", name=session.transcriber.name)
                continue
            raw = message.get("text")
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "speech_start":
                await session.speech_start()
            elif kind == "pause":
                await session.pause()
            elif kind == "endpoint":
                await session.endpoint()
            elif kind == "interrupt":
                await session.interrupt()
            elif kind == "greet":
                await session.greet()
            elif kind == "finish":
                await session.finish()
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()

"""The browser's socket into a live audio session.

WHY A PROXY AND NOT A DIRECT CONNECTION

The browser could open a WebSocket to Gemini itself; the SDK supports it. It
does not, for two reasons that are not about convenience:

  * The API key would have to be in the browser. Every other provider in this
    project is reached server-side for exactly this reason, and speech is not
    the place to make an exception.
  * The model's tools are OUR tools. `search_documents` runs against a Qdrant
    collection scoped by `owner_id`; that scoping cannot live on the client,
    because a client that decides its own owner_id is not scoped at all.

So the audio passes through, and the tool calls stop here.

THE PROTOCOL

Browser to us:
    binary             raw 16kHz mono PCM, as captured
    {"type":"start"}   the speaker pressed the button
    {"type":"end"}     the speaker pressed it again; answer now

Us to browser:
    binary                      raw 24kHz mono PCM, to play
    {"type":"heard","text"}     what it understood, for the screen
    {"type":"said","text"}      what it is saying, for the screen
    {"type":"tool", ...}        which tool ran, and what it found
    {"type":"turn_end"}         the model has finished speaking
    {"type":"resume","handle"}  hold this; it restores the conversation
    {"type":"going_away","in"}  the server is about to drop us; reconnect
    {"type":"error","detail"}   something failed, in words

Text frames are JSON; audio frames are raw bytes. The split is deliberate:
base64 inside JSON would add a third to every audio frame, on the one path
where latency is the entire point.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from google.genai import types

from app.auth import ANONYMOUS, User, _verify
from app.config import get_settings
from app.services import live, voice, websearch

log = structlog.get_logger()

router = APIRouter(tags=["live"])


@router.get("/live/status")
async def status() -> dict:
    """What the client needs before offering a microphone.

    Unauthenticated on purpose -- it reports CONFIGURATION, not user data, the
    same way /health does. The socket itself still requires a token.
    """
    settings = get_settings()
    return {
        "enabled": live.enabled(),
        "model": settings.live_model,
        "voices": voice.VOICES,
        "default_voice": settings.voice_default,
        "web_search": websearch.enabled(),
        "input_rate": live.INPUT_RATE,
        "output_rate": live.OUTPUT_RATE,
    }


async def _authenticate(token: str) -> User | None:
    """Resolve the caller from a query-string token.

    A browser CANNOT set headers on a WebSocket -- the API has no equivalent of
    `fetch`'s `headers` -- so the bearer token arrives as a query parameter.
    That is the standard workaround and it is worth being explicit about the
    cost: query strings land in access logs where Authorization headers do not.
    Acceptable here because these are short-lived Supabase access tokens, and
    the alternative (a cookie) would need CSRF handling of its own.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return ANONYMOUS
    if not token:
        return None
    try:
        return await _verify(token)
    except Exception as exc:  # noqa: BLE001 - any verification failure is a 401
        log.info("live_auth_failed", error=type(exc).__name__)
        return None


@router.websocket("/live/ws")
async def live_socket(
    ws: WebSocket,
    token: str = Query(default=""),
    voice_name: str = Query(default=""),
    resume: str = Query(default=""),
) -> None:
    await ws.accept()

    user = await _authenticate(token)
    if user is None:
        await ws.send_text(json.dumps({"type": "error", "detail": "Not authenticated."}))
        await ws.close(code=4401)
        return

    if not live.enabled():
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": "Speech is not configured on this deployment.",
                }
            )
        )
        await ws.close(code=1011)
        return

    settings = get_settings()
    chosen = voice_name or settings.voice_default
    if chosen not in {v["id"] for v in voice.VOICES}:
        chosen = settings.voice_default

    try:
        client = live.client()
        async with client.aio.live.connect(
            model=settings.live_model, config=live.config(chosen, resume or None)
        ) as session:
            await ws.send_text(
                json.dumps(
                    {"type": "ready", "voice": chosen, "resumed": bool(resume)}
                )
            )
            log.info(
                "live_session_open",
                owner=bool(user.owner_id),
                voice=chosen,
                resumed=bool(resume),
            )

            # Two directions at once, which is the whole point of a live model:
            # the user can be speaking while it is still answering. Running
            # these sequentially would reintroduce the turn-taking the cascade
            # was stuck with.
            uplink = asyncio.create_task(_uplink(ws, session))
            downlink = asyncio.create_task(_downlink(ws, session, user))

            done, pending = await asyncio.wait(
                {uplink, downlink}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                # Surface a crash in either direction rather than closing
                # silently, which is indistinguishable from a clean hangup.
                if task.exception():
                    raise task.exception()  # type: ignore[misc]

    except WebSocketDisconnect:
        log.info("live_client_left")
    except Exception as exc:  # noqa: BLE001 - report, never 500 a socket
        if _idle_disconnect(exc):
            # Expected. Logged at info and reported as a plain close, so the
            # client reconnects on the next turn without showing a red banner
            # for something that is not a fault.
            log.info("live_session_idle_out", error=str(exc)[:160])
        else:
            log.warning("live_session_failed", error=str(exc)[:300])
            with contextlib.suppress(Exception):
                await ws.send_text(
                    json.dumps({"type": "error", "detail": _explain(exc)})
                )
    finally:
        with contextlib.suppress(Exception):
            await ws.close()
        log.info("live_session_closed")


async def _uplink(ws: WebSocket, session) -> None:
    """Browser audio into the model."""
    while True:
        message = await ws.receive()

        if message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))

        chunk = message.get("bytes")
        if chunk:
            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk, mime_type=f"audio/pcm;rate={live.INPUT_RATE}"
                )
            )
            continue

        text = message.get("text")
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue

        kind = event.get("type")

        # MANUAL TURN BOUNDARIES. The button, and nothing else, decides.
        #
        # Automatic activity detection is disabled in `live.config`, so the
        # model will not answer because it heard a pause -- and people pause
        # constantly while speaking. These two markers are now the only things
        # that open and close a turn.
        if kind == "start":
            await session.send_realtime_input(activity_start=types.ActivityStart())
        elif kind == "end":
            await session.send_realtime_input(activity_end=types.ActivityEnd())


async def _downlink(ws: WebSocket, session, user: User) -> None:
    """Model audio, transcripts and tool calls out to the browser."""
    settings = get_settings()
    spoke = False

    while True:
        received_anything = False

        async for message in session.receive():
            received_anything = True

            if message.data:
                spoke = True
                await ws.send_bytes(message.data)

            sc = message.server_content
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    await ws.send_text(
                        json.dumps(
                            {"type": "heard", "text": sc.input_transcription.text}
                        )
                    )
                if sc.output_transcription and sc.output_transcription.text:
                    await ws.send_text(
                        json.dumps(
                            {"type": "said", "text": sc.output_transcription.text}
                        )
                    )
                if sc.turn_complete and spoke:
                    # Only an end WITH audio ends the turn. A `turn_complete`
                    # carrying no speech is the model finishing its tool-calling
                    # step, and reporting that to the browser stops playback
                    # before a word has been said.
                    spoke = False
                    await ws.send_text(json.dumps({"type": "turn_end"}))

            # THE HANDLE THAT MAKES A RECONNECT INVISIBLE.
            #
            # Sent to the browser rather than stored here, deliberately: the
            # server holds no per-user state for this app, and a handle kept in
            # process memory would be lost on the next deploy -- which is
            # exactly when reconnects happen in bulk.
            update = message.session_resumption_update
            if update and update.resumable and update.new_handle:
                await ws.send_text(
                    json.dumps({"type": "resume", "handle": update.new_handle})
                )

            # The server announcing its own disconnection, with time to spare.
            # Live sessions have a hard lifetime; this is the warning, and
            # acting on it is the difference between a seamless reconnect and
            # a conversation that dies mid-sentence.
            if message.go_away:
                left = message.go_away.time_left
                log.info("live_going_away", time_left=str(left))
                await ws.send_text(
                    json.dumps({"type": "going_away", "in": str(left)})
                )

            if message.tool_call:
                responses = []
                for call in message.tool_call.function_calls:
                    response, report = await live.run_tool_call(
                        call,
                        owner_id=user.owner_id,
                        top_k=settings.retrieval_top_k,
                    )
                    responses.append(response)
                    # Reported as it happens, not at the end. A search takes a
                    # second or two and the model is silent through it; without
                    # this the app looks frozen at exactly the moment it is
                    # doing the most interesting thing.
                    await ws.send_text(json.dumps({"type": "tool", **report}))
                await session.send_tool_response(function_responses=responses)

        if not received_anything:
            # An empty generator means the session is finished with us. Looping
            # again would spin at full speed.
            return


def _idle_disconnect(exc: Exception) -> bool:
    """A session that timed out doing nothing, rather than a failure.

    Gemini drops a live socket that has been idle, and the browser holds one
    open between questions so the next one starts instantly. The result reached
    the user as a red banner reading "The live session failed:
    ConnectionClosedError" -- alarming, and describing nothing they did or need
    to do. The next turn simply opens a new session.
    """
    text = str(exc).lower()
    return "keepalive" in text or "1011" in text or "no close frame" in text


def _explain(exc: Exception) -> str:
    text = str(exc)
    if "quota" in text.lower() or "429" in text:
        return "The live model is out of quota for now. Try again in a minute."
    if "403" in text or "401" in text:
        return "The API key was refused for the live model."
    return f"The live session failed: {type(exc).__name__}."

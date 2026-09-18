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
    {"type":"greet"}   open the conversation; the model speaks first
    {"type":"start"}   the speaker pressed the button
    {"type":"end"}     the speaker pressed it again; answer now
    {"type":"finish"}  the PARTICIPANT ended the interview; close it

Us to browser:
    binary                      raw 24kHz mono PCM, to play
    {"type":"heard","text"}     what it understood, for the screen
    {"type":"said","text"}      what it is saying, for the screen
    {"type":"tool", ...}        which tool ran, and what it found
    {"type":"turn", ...}        the COMPLETED exchange, including what was
                                RECORDED from it -- which is accurate, where
                                the transcript is not
    {"type":"turn_end"}         the model has finished speaking
    {"type":"finished"}         the interview is closed, at the participant's
                                request rather than the interviewer's
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
import uuid
from typing import Any

import structlog
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from google.genai import types
from sqlalchemy import select

from app.auth import (
    ANONYMOUS,
    User,
    _verify,
    current_user,
    forbid_if_not_owner,
)
from app.config import get_settings
from app.db.models import ChatSession, Message, Role
from app.db.session import SessionLocal
from app.services import blueprint, invites, live, voice, websearch

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


@router.get("/live/conversations")
async def conversations(
    mode: str = Query(default="speak"), user: User = Depends(current_user)
) -> list[dict]:
    """Conversations for one mode, newest first.

    Filtered by `kind` rather than returning everything and letting the client
    sort it out: Speak and Interview are separate lists in separate places, and
    a drawer that briefly shows the other app's conversations before filtering
    them is worse than one that waits.
    """
    kind = live.kind_of(mode)
    from sqlalchemy import func as sql_func

    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(
                    ChatSession,
                    sql_func.count(Message.id).label("n"),
                )
                .outerjoin(Message, Message.session_id == ChatSession.id)
                .where(
                    ChatSession.kind == kind,
                    ChatSession.owner_id.is_(None)
                    if user.owner_id is None
                    else ChatSession.owner_id == user.owner_id,
                )
                .group_by(ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(30)
            )
        ).all()

    return [
        {
            "id": str(chat.id),
            "title": chat.title,
            "turns": n // 2,
            "updated_at": chat.updated_at.isoformat(),
            # Whether the live CONTEXT can be restored, as opposed to merely
            # the transcript being readable. Without a handle the conversation
            # can be reread but not continued, and those are different things.
            "resumable": bool(chat.live_handle),
            # So the list can mark a finished interview without opening it.
            "complete": live.profile.complete(
                chat.profile or {}, chat.fields or None
            ),
        }
        for chat, n in rows
        if n
    ]


@router.get("/live/conversations/{conversation_id}")
async def conversation(
    conversation_id: uuid.UUID, user: User = Depends(current_user)
) -> dict:
    """One spoken conversation, as alternating turns."""
    async with SessionLocal() as db:
        chat = await db.get(ChatSession, conversation_id)
        if chat is None or chat.kind not in live.CONVERSATION_KINDS:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        forbid_if_not_owner(chat.owner_id, user)

        rows = (
            await db.execute(
                select(Message)
                .where(Message.session_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()

    # Paired back into exchanges. They were written as two rows so the table
    # stays the same shape as the Research Desk's, and the UI wants them as
    # one card.
    turns: list[dict] = []
    for row in rows:
        if row.role is Role.user:
            turns.append(
                {"question": row.content, "answer": "", "sources": [], "tools": []}
            )
        elif turns:
            turns[-1]["answer"] = row.content
            turns[-1]["sources"] = row.sources or []
            turns[-1]["tools"] = (row.agent_meta or {}).get("tools", [])

    return {
        "id": str(chat.id),
        "title": chat.title,
        "resumable": bool(chat.live_handle),
        "turns": turns,
        # Interview only; an empty object everywhere else.
        "profile": chat.profile or {},
        "missing": live.profile.missing(chat.profile or {}, chat.fields or None),
        "complete": live.profile.complete(chat.profile or {}, chat.fields or None),
        # Howler's schema travels with the conversation; the card cannot render
        # fields it does not know about.
        "fields": chat.fields or [],
        "brief": chat.brief or "",
        "participant": chat.participant or "",
        # Which project this came from, so reading a result has a way BACK to
        # it. Without it the only route out of a conversation was the project
        # list, which loses the one thing you were looking at.
        "project_id": str(chat.project_id) if chat.project_id else None,
    }


@router.patch("/live/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: uuid.UUID,
    body: dict,
    user: User = Depends(current_user),
) -> dict:
    """Rename a conversation.

    The automatic title is the first thing said, which is a reasonable guess
    and frequently a bad one -- a misheard opening line becomes the permanent
    name of the conversation.
    """
    title = str(body.get("title") or "").strip()[:120]
    if not title:
        raise HTTPException(status_code=400, detail="A title is required.")

    async with SessionLocal() as db:
        chat = await db.get(ChatSession, conversation_id)
        if chat is None or chat.kind not in live.CONVERSATION_KINDS:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        forbid_if_not_owner(chat.owner_id, user)
        chat.title = title
        await db.commit()
    return {"id": str(conversation_id), "title": title}


@router.delete("/live/conversations/{conversation_id}", status_code=204)
async def remove_conversation(
    conversation_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    async with SessionLocal() as db:
        chat = await db.get(ChatSession, conversation_id)
        if chat is None or chat.kind not in live.CONVERSATION_KINDS:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        forbid_if_not_owner(chat.owner_id, user)
        await db.delete(chat)
        await db.commit()


@router.post("/live/howler")
async def create_howl(body: dict, user: User = Depends(current_user)) -> dict:
    """Turn a brief into a conversation with its own schema.

    The schema is generated HERE, once, and stored -- not at connect time and
    not per turn. See `blueprint.py` for the four things that depend on it
    being stable; the short version is that an interview whose target moves can
    never be finished.

    Returns the fields so they can be read before anyone speaks. A brief is a
    loose instruction and this is the point at which it becomes a specific
    list, which is exactly the moment worth checking.
    """
    brief = str(body.get("brief") or "").strip()
    participant = str(body.get("participant") or "").strip()
    if not brief:
        raise HTTPException(
            status_code=400,
            detail="Describe what you want to find out.",
        )

    try:
        fields = await blueprint.from_brief(brief)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a model failure is not a 500 here
        log.warning("blueprint_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=502,
            detail="The brief could not be turned into fields. Try again.",
        ) from exc

    async with SessionLocal() as db:
        chat = ChatSession(
            title=live.UNNAMED_INTERVIEW,
            owner_id=user.owner_id,
            kind="howler",
            brief=brief[: blueprint.MAX_BRIEF_CHARS],
            participant=participant[: blueprint.MAX_PARTICIPANT_CHARS],
            fields=fields,
        )
        db.add(chat)
        await db.commit()
        await db.refresh(chat)

    return {
        "id": str(chat.id),
        "brief": chat.brief,
        "participant": chat.participant,
        "fields": chat.fields,
    }


@router.post("/live/howler/preview")
async def preview_howl(body: dict, user: User = Depends(current_user)) -> dict:
    """The fields a brief would produce, without creating anything.

    So a brief can be adjusted and re-read before a conversation exists. Every
    attempt creating a session would leave a drawer full of abandoned ones.
    """
    try:
        return {"fields": await blueprint.from_brief(str(body.get("brief") or ""))}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.warning("blueprint_preview_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=502,
            detail="The brief could not be turned into fields. Try again.",
        ) from exc


@router.get("/live/profile-fields")
async def profile_fields() -> list[dict]:
    """The fields an interview is trying to fill, for rendering the card.

    Served rather than duplicated in the frontend: the schema, the tool
    declaration and the completeness check all come from one list, and a
    fourth copy in TypeScript is the one that would drift.
    """
    return [
        {
            "name": f["name"],
            "required": f["required"],
            "kind": f.get("type", "STRING"),
        }
        for f in live.profile.FIELDS
    ]


class _GuestUser:
    """Just enough of a User for a conversation row, and nothing more.

    It carries an `owner_id` so the interview lands in the right person's
    results, and no other capability. It is not a `User`, so it cannot be
    passed to anything expecting one without the type telling on it.
    """

    def __init__(self, owner_id: str | None) -> None:
        self.owner_id = owner_id
        self.id = "guest"
        self.anonymous = True


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
    session_id: str = Query(default=""),
    mode: str = Query(default="speak"),
    invite: str = Query(default=""),
) -> None:
    await ws.accept()

    # ---- a guest holding a link -----------------------------------------
    #
    # An invite is resolved SEPARATELY from `_authenticate`, and deliberately
    # does not produce a signed-in user. It grants one thing: the right to
    # speak into one conversation. If it resolved to a User it would inherit
    # every permission that user has, and the safety of the whole surface would
    # then rest on nobody ever handing it to the wrong dependency.
    #
    # The session, the mode and the schema all come from the invite. None of
    # them are taken from the query string, because a guest can write anything
    # there.
    guest: dict | None = None
    if invite:
        try:
            row, project = await invites.claim(invite)
        except invites.InviteError as exc:
            await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
            await ws.close(code=4403)
            return
        guest = {
            "invite_id": row.id,
            "session_id": str(row.session_id) if row.session_id else "",
            "project": project,
            "participant": row.participant or project.participant or "",
            # What to call the conversation. A link made for a named person
            # says who; an auto-made one falls back to the project.
            "label": (row.label or "").strip(),
        }
        mode = "howler"
        session_id = guest["session_id"]

    user = await _authenticate(token) if not guest else _GuestUser(
        guest["project"].owner_id
    )
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

    # The conversation this socket appends to. Found or created BEFORE the live
    # session opens, because its stored handle is what the live session needs
    # in order to resume rather than start blank.
    chosen_mode = live.mode_of(mode)
    chat = await live.open_conversation(
        user.owner_id, session_id or None, live.kind_of(chosen_mode)
    )
    resume = chat.live_handle

    # HOWLER READS ITS SCHEMA FROM THE ROW, and only from there.
    #
    # It was generated once from the brief and frozen. Regenerating it here
    # would produce a subtly different list on every reconnect, and a
    # half-filled profile would stop lining up with the fields it was filling.
    # The other modes pass None and get the built-in schema.
    fields = list(chat.fields or []) if chosen_mode == "howler" else None

    # A guest's conversation inherits the PROJECT's brief and schema, and is
    # stamped with the project so its results are findable. Done here rather
    # than at creation because `open_conversation` is shared with the other
    # modes and knows nothing about projects.
    if guest is not None:
        project = guest["project"]
        fields = list(project.fields or [])
        vocabulary = list(project.vocabulary or [])
        await live.adopt_project(
            chat.id,
            project.id,
            project.brief or "",
            guest["participant"],
            fields,
            guest["invite_id"],
            vocabulary,
            guest["label"] or project.title or "",
        )
        chat.brief = project.brief or ""
        chat.participant = guest["participant"]
        # In memory too, not only in the row: `config` is built from `chat` a
        # few lines below, and a guest's session is adopted in the same request
        # that opens it.
        chat.vocabulary = vocabulary

    try:
        client = live.client()
        async with client.aio.live.connect(
            model=settings.live_model,
            config=live.config(
                chosen,
                resume,
                chosen_mode,
                fields,
                chat.brief or "",
                chat.participant or "",
                chat.vocabulary or [],
            ),
        ) as session:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "ready",
                        "voice": chosen,
                        "resumed": bool(resume),
                        "session_id": str(chat.id),
                        "mode": chosen_mode,
                    }
                )
            )
            log.info(
                "live_session_open",
                owner=bool(user.owner_id),
                voice=chosen,
                resumed=bool(resume),
                session=str(chat.id),
                mode=chosen_mode,
            )

            # Two directions at once, which is the whole point of a live model:
            # the user can be speaking while it is still answering. Running
            # these sequentially would reintroduce the turn-taking the cascade
            # was stuck with.
            # Shared between the two directions.
            #
            # `turns` counts what the PARTICIPANT has said, and is the guard
            # against ending an interview before the closing question has been
            # answered -- see `run_tool_call`. The uplink knows when a turn is
            # sent; the tool layer decides whether the model may end.
            # `closed` is set by the downlink when `end_interview` comes back,
            # so a participant-initiated close can wait for the model's
            # account rather than polling for it.
            turn_state: dict[str, Any] = {"turns": 0, "closed": asyncio.Event()}

            uplink = asyncio.create_task(_uplink(ws, session, turn_state, chat.id))
            downlink = asyncio.create_task(
                _downlink(ws, session, user, chat.id, turn_state, fields)
            )

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


# How long to let the model write its closing account before the interview ends
# anyway. Long enough for one tool call, short enough that somebody who pressed
# stop does not sit watching a spinner.
CLOSING_SECONDS = 8.0

CLOSING = (
    "The participant has just ended the interview themselves. Do not ask "
    "anything further and do not say goodbye. Call end_interview now, with "
    "your one-sentence summary of who they are, `demeanour` describing how "
    "they came across over the whole conversation, and `notable_moments`. "
    "Base all of it only on what you actually heard -- an interview that was "
    "cut short has less to go on, and saying less is correct."
)


async def _uplink(ws: WebSocket, session, turn_state: dict, chat_id=None) -> None:
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

        # THE MODEL SPEAKS FIRST.
        #
        # Sent as a text turn rather than a system instruction, because the
        # instruction is already in place and describes what an opening should
        # be -- this is the cue to perform it now. It is `send_client_content`
        # rather than audio because there is nothing to say yet; the reply
        # comes back as speech like any other turn.
        if kind == "greet":
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Begin. Introduce yourself in one sentence and "
                                "ask your first question."
                            )
                        )
                    ],
                ),
                turn_complete=True,
            )
            continue

        # MANUAL TURN BOUNDARIES. The button, and nothing else, decides.
        #
        # Automatic activity detection is disabled in `live.config`, so the
        # model will not answer because it heard a pause -- and people pause
        # constantly while speaking. These two markers are now the only things
        # that open and close a turn.
        if kind == "start":
            await session.send_realtime_input(activity_start=types.ActivityStart())
        elif kind == "end":
            turn_state["turns"] += 1
            await session.send_realtime_input(activity_end=types.ActivityEnd())

        # THE PARTICIPANT CLOSING IT, which the model is not consulted about.
        #
        # Not a hint to the model to wrap up: it is over. Somebody who has
        # decided they are done does not want to be asked one more question,
        # and an interviewer that can talk them out of leaving is not a
        # feature. The conversation is marked ended, the browser is told, and
        # the socket closes -- and because the link checks that flag, it stops
        # working too, which is the point of pressing it.
        elif kind == "finish":
            # ASK IT TO CLOSE PROPERLY FIRST, then close regardless.
            #
            # `demeanour` and `notable_moments` only exist on `end_interview`,
            # so an interview the participant ended used to have no account of
            # how it sounded at all -- which is most of them, and exactly the
            # ones where knowing would help.
            #
            # The request goes to the LIVE model, not to a text pass over the
            # transcript afterwards. That transcript is a separate, lossier
            # recognition of the same audio; asking it to describe how somebody
            # sounded would be inventing from a bad reading, which is the one
            # failure this feature cannot afford.
            #
            # Bounded, and never blocking: if the model does not answer within
            # CLOSING_SECONDS the interview still ends. Somebody who pressed
            # stop is not waiting on a model's paperwork.
            turn_state["closing"] = True
            with contextlib.suppress(Exception):
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=CLOSING)]),
                    turn_complete=True,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        turn_state["closed"].wait(), timeout=CLOSING_SECONDS
                    )

            if chat_id is not None:
                await live.finish_interview(chat_id)
            await ws.send_text(json.dumps({"type": "finished"}))
            return


async def _downlink(
    ws: WebSocket,
    session,
    user: User,
    chat_id,
    turn_state: dict,
    fields: list[dict] | None = None,
) -> None:
    """Model audio, transcripts and tool calls out to the browser."""
    settings = get_settings()
    spoke = False
    # Accumulated per turn, so the exchange is written as ONE row pair rather
    # than a row per transcript fragment.
    heard: list[str] = []
    said: list[str] = []
    sources: list[dict] = []
    tools: list[str] = []
    # What the model RECORDED this turn, as it passed it. The accurate record
    # of what the participant said -- unlike `heard`, which is a separate and
    # demonstrably lossier transcription pass.
    recorded: list[dict] = []

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
                    heard.append(sc.input_transcription.text)
                    await ws.send_text(
                        json.dumps(
                            {"type": "heard", "text": sc.input_transcription.text}
                        )
                    )
                if sc.output_transcription and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
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
                    # WRITTEN BEFORE THE CLIENT IS TOLD, and shielded.
                    #
                    # Telling the browser first lost the last turn of every
                    # conversation: the client often closes the socket the
                    # instant it sees `turn_end`, which cancels this task --
                    # sometimes mid-write. Measured, three turns spoken and two
                    # stored, every time.
                    #
                    # `shield` covers the other half: a disconnect arriving
                    # while the INSERT is in flight must not abandon it
                    # half-done.
                    await asyncio.shield(
                        live.save_turn(
                            chat_id,
                            "".join(heard).strip(),
                            "".join(said).strip(),
                            sources,
                            tools,
                        )
                    )
                    # THE COMPLETED EXCHANGE, AS ONE EVENT.
                    #
                    # The browser used to assemble this itself from the
                    # streaming `heard` and `said` fragments, deciding for
                    # itself where one turn ended and the next began -- and it
                    # got that wrong repeatedly, because the only clue it had
                    # was the audio queue draining, which happens between
                    # chunks. Questions were split across exchanges, and a
                    # whole spoken sentence would land as a single stray word.
                    #
                    # The server already knows the boundary exactly: it is the
                    # `turn_complete` above, the same point at which the rows
                    # are written. Sending the assembled turn means the screen
                    # and the database cannot disagree, and the client has
                    # nothing left to infer.
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "turn",
                                "question": "".join(heard).strip(),
                                "answer": "".join(said).strip(),
                                "sources": sources,
                                "tools": tools,
                                "recorded": recorded,
                            }
                        )
                    )
                    heard, said, sources, tools, recorded = [], [], [], [], []
                    await ws.send_text(json.dumps({"type": "turn_end"}))

            # THE HANDLE THAT MAKES A RECONNECT INVISIBLE.
            #
            # Sent to the browser rather than stored here, deliberately: the
            # server holds no per-user state for this app, and a handle kept in
            # process memory would be lost on the next deploy -- which is
            # exactly when reconnects happen in bulk.
            update = message.session_resumption_update
            if update and update.resumable and update.new_handle:
                # Stored on the CONVERSATION, not handed to the browser to keep.
                # A handle in sessionStorage dies with the tab, which is
                # precisely when someone wants to pick a conversation up again.
                await live.store_handle(chat_id, update.new_handle)
                await ws.send_text(json.dumps({"type": "resume"}))

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
                        session_id=chat_id,
                        turns=turn_state["turns"],
                        fields=fields,
                    )
                    responses.append(response)
                    # Reported as it happens, not at the end. A search takes a
                    # second or two and the model is silent through it; without
                    # this the app looks frozen at exactly the moment it is
                    # doing the most interesting thing.
                    tools.append(report["tool"])
                    if report["tool"] == live.profile.TOOL_NAME and report["args"]:
                        recorded.append(report["args"])
                    for source in report["sources"]:
                        if source not in sources:
                            sources.append(source)
                    # So the `finish` handler knows the model has closed it
                    # and can stop waiting.
                    if report.get("ended"):
                        turn_state["closed"].set()
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

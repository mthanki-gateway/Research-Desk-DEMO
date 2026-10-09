"""Manks: send a bot to a meeting, read what it brought back.

Two audiences on one router, kept apart on purpose:

  /manks/meetings...   the signed-in person: create, list, read, leave, delete
  /manks/bot/...       the bot container, authenticated by a shared secret and
                       by nothing else. It can claim work, report status and
                       upload audio. It cannot read a transcript or list
                       anyone's meetings, so a leaked secret exposes a queue,
                       not a person's notes.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from datetime import UTC, datetime

import structlog
from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, or_, select

from app.auth import User, current_user
from app.config import get_settings
from app.db.models import Job, Meeting
from app.db.session import SessionLocal
from app.services import manks
from app.services.storage import get_storage

log = structlog.get_logger()

router = APIRouter(prefix="/manks", tags=["manks"])

# What the bot may report. Anything else is refused rather than stored, so a
# buggy bot cannot put the row in a state the page has no wording for.
BOT_STATES = {"joining", "waiting", "in_meeting", "ended", "failed"}
LIVE_STATES = ("queued", "joining", "waiting", "in_meeting")


def _owned(query, user: User):
    return query.where(
        Meeting.owner_id.is_(None) if user.owner_id is None else Meeting.owner_id == user.owner_id
    )


def _out(m: Meeting, *, full: bool = False) -> dict:
    segments = list(m.segments or [])
    out = {
        "id": str(m.id),
        "url": m.url,
        "platform": m.platform,
        "title": m.title,
        "bot_name": m.bot_name,
        "status": m.status,
        "detail": m.detail,
        "error": m.error,
        "leave_requested": m.leave_requested,
        "talk": bool(m.talk),
        "segments": len(segments),
        "segment_ids": [s.get("index") for s in segments],
        # What the player needs to lay the parts end to end: the recorded files
        # themselves do not carry a usable duration.
        "segment_info": [
            {"index": s.get("index"), "start": float(s.get("start", 0)), "seconds": float(s.get("seconds", 0))}
            for s in segments
        ],
        "recorded_seconds": round(sum(float(s.get("seconds", 0)) for s in segments)),
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "started_at": m.started_at.isoformat() if m.started_at else None,
        "ended_at": m.ended_at.isoformat() if m.ended_at else None,
        "has_insights": m.insights is not None,
    }
    if full:
        out["transcript"] = m.transcript
        out["insights"] = m.insights
    return out


async def _load(meeting_id: uuid.UUID, user: User) -> Meeting:
    async with SessionLocal() as db:
        m = (
            await db.execute(_owned(select(Meeting).where(Meeting.id == meeting_id), user))
        ).scalar_one_or_none()
    # 404 and not 403: a 403 confirms the id exists.
    if m is None:
        raise HTTPException(404, "Meeting not found.")
    return m


# ---- the person ------------------------------------------------------------


@router.get("/status")
async def status(user: User = Depends(current_user)) -> dict:
    """Whether a bot is wired up, so the page can say so before somebody waits."""
    s = get_settings()
    return {
        "enabled": bool(s.manks_bot_secret),
        "bot_name": s.manks_bot_name,
        "segment_seconds": s.manks_segment_seconds,
    }


@router.post("/meetings")
async def create_meeting(body: dict, user: User = Depends(current_user)) -> dict:
    s = get_settings()
    if not s.manks_bot_secret:
        raise HTTPException(503, "The Manks bot is not set up on this deployment.")
    try:
        platform, url = manks.parse_link(str(body.get("url") or ""))
    except manks.BadLink as exc:
        raise HTTPException(400, str(exc)) from exc
    # A BigBlueButton room can be joined in a browser (the default) or by the
    # lightweight client that speaks its protocols directly.
    if platform == "bbb" and body.get("method") == "native":
        platform = "bbb_native"

    async with SessionLocal() as db:
        # One live bot per link: pasting it twice must not send two bots into
        # the same call, each recording and each admitted by the host.
        existing = (
            await db.execute(
                _owned(
                    select(Meeting).where(Meeting.url == url, Meeting.status.in_(LIVE_STATES)),
                    user,
                )
            )
        ).scalars().first()
        if existing is not None:
            raise HTTPException(409, "A bot is already on its way to that meeting.")
        m = Meeting(
            owner_id=user.owner_id,
            url=url,
            platform=platform,
            title=str(body.get("title") or "").strip()[:200],
            talk=bool(body.get("talk")) and platform in ("bbb", "bbb_native"),
            bot_name=(str(body.get("bot_name") or "").strip() or s.manks_bot_name)[:80],
            status="queued",
            detail="Waiting for the bot",
        )
        db.add(m)
        await db.commit()
        await db.refresh(m)
    log.info("manks_meeting_created", meeting=str(m.id), platform=platform)
    return _out(m)


@router.get("/meetings")
async def list_meetings(
    q: str = "",
    show: str = "all",
    offset: int = 0,
    limit: int = 20,
    user: User = Depends(current_user),
) -> dict:
    """A page of meetings, newest first.

    `q` matches the title, the notes' title or the server a link points at
    (never the link's query string, which holds the room's password).
    `show` is all | active (anything not settled yet) | finished.
    `live` counts this owner's bots in a meeting right now, whatever the page.
    """
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    query = _owned(select(Meeting), user)
    term = q.strip()
    if term:
        like = f"%{term}%"
        query = query.where(
            or_(
                Meeting.title.ilike(like),
                Meeting.insights["title"].astext.ilike(like),
                func.split_part(Meeting.url, "?", 1).ilike(like),
            )
        )
    busy = (*LIVE_STATES, "transcribing")
    if show == "active":
        query = query.where(Meeting.status.in_(busy))
    elif show == "finished":
        query = query.where(Meeting.status.not_in(busy))
    async with SessionLocal() as db:
        total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
        rows = (
            await db.execute(query.order_by(Meeting.created_at.desc()).offset(offset).limit(limit))
        ).scalars().all()
        live = (
            await db.execute(
                select(func.count()).select_from(
                    _owned(select(Meeting), user).where(Meeting.status.in_(LIVE_STATES)).subquery()
                )
            )
        ).scalar_one()
        # Whether anything anywhere is still moving, so the page knows to poll
        # even when the moving one is on another page.
        busy_any = (
            await db.execute(
                select(func.count()).select_from(
                    _owned(select(Meeting), user).where(Meeting.status.in_(busy)).subquery()
                )
            )
        ).scalar_one()
    return {
        "items": [_out(m) for m in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
        "live": live,
        "busy": busy_any,
    }


@router.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: uuid.UUID, user: User = Depends(current_user)) -> dict:
    return _out(await _load(meeting_id, user), full=True)


@router.post("/meetings/{meeting_id}/leave")
async def leave(meeting_id: uuid.UUID, user: User = Depends(current_user)) -> dict:
    m = await _load(meeting_id, user)
    async with SessionLocal() as db:
        row = await db.get(Meeting, m.id)
        if row is None:
            raise HTTPException(404, "Meeting not found.")
        if row.status == "queued":
            # Not claimed yet: nobody to tell, just never send one.
            row.status = "cancelled"
            row.detail = ""
        elif row.status in LIVE_STATES:
            row.leave_requested = True
            row.detail = "Leaving the meeting"
        await db.commit()
        await db.refresh(row)
    return _out(row)


@router.post("/meetings/{meeting_id}/analyze")
async def analyze(meeting_id: uuid.UUID, user: User = Depends(current_user)) -> dict:
    """Run transcription and insights again over what was recorded."""
    m = await _load(meeting_id, user)
    if not m.segments:
        raise HTTPException(409, "Nothing was recorded for this meeting.")
    if m.status in LIVE_STATES:
        raise HTTPException(409, "The meeting is still going.")
    async with SessionLocal() as db:
        row = await db.get(Meeting, m.id)
        row.status = "transcribing"
        row.detail = "Queued"
        row.error = None
        await db.commit()
        await db.refresh(row)
    await manks.enqueue_analysis(row)
    return _out(row)


@router.delete("/meetings/{meeting_id}", status_code=204)
async def delete_meeting(meeting_id: uuid.UUID, user: User = Depends(current_user)) -> Response:
    """Everything about a meeting: recording, transcript, notes, queued work.

    A bot still in the meeting is not refused: once the row is gone its next
    control check reads "leave" (see `bot_control`), and anything it uploads
    after that is turned away as a meeting that does not exist.
    """
    m = await _load(meeting_id, user)
    storage = get_storage()
    for seg in m.segments or []:
        try:
            await storage.delete(seg["key"])
        except Exception as exc:  # noqa: BLE001 - a missing file must not block the delete
            log.warning("manks_segment_delete_failed", error=str(exc)[:120])
    async with SessionLocal() as db:
        # An analysis waiting in the queue would only fail on a missing row.
        await db.execute(
            delete(Job).where(
                Job.subject_type == manks.SUBJECT,
                Job.subject_id == m.id,
                Job.status.in_(("queued", "running")),
            )
        )
        row = await db.get(Meeting, m.id)
        if row is not None:
            await db.delete(row)
        await db.commit()
    log.info("manks_meeting_deleted", meeting=str(m.id), segments=len(m.segments or []))
    return Response(status_code=204)


@router.get("/meetings/{meeting_id}/audio/{index}", response_model=None)
async def audio(meeting_id: uuid.UUID, index: int, user: User = Depends(current_user)):
    """One recorded segment, for the player."""
    m = await _load(meeting_id, user)
    seg = next((s for s in m.segments or [] if s.get("index") == index), None)
    if seg is None:
        raise HTTPException(404, "No such segment.")
    data = await get_storage().get(seg["key"])

    async def body():
        yield data

    return StreamingResponse(body(), media_type="audio/webm")


# ---- the bot ---------------------------------------------------------------


def _bot(secret: str | None = Header(default=None, alias="X-Manks-Secret")) -> None:
    expected = get_settings().manks_bot_secret
    # Constant-time, and refused outright when no secret is configured: an
    # empty secret must never match an empty header.
    if not expected or not secret or not hmac.compare_digest(secret, expected):
        raise HTTPException(401, "Bad bot secret.")


@router.post("/bot/claim", dependencies=[Depends(_bot)], response_model=None)
async def bot_claim(body: dict | None = Body(default=None)) -> Response | dict:
    m = await manks.claim_next((body or {}).get("platforms"))
    if m is None:
        return Response(status_code=204)
    return {
        "id": str(m.id),
        "url": m.url,
        "name": m.bot_name,
        "platform": m.platform,
        "talk": bool(m.talk),
    }


@router.get("/bot/{meeting_id}/control", dependencies=[Depends(_bot)])
async def bot_control(meeting_id: uuid.UUID) -> dict:
    async with SessionLocal() as db:
        m = await db.get(Meeting, meeting_id)
    if m is None:
        return {"leave": True}  # deleted under it: get out
    return {"leave": bool(m.leave_requested) or m.status == "cancelled"}


@router.post("/bot/{meeting_id}/status", dependencies=[Depends(_bot)])
async def bot_status(meeting_id: uuid.UUID, body: dict) -> dict:
    state = str(body.get("status") or "")
    if state not in BOT_STATES:
        raise HTTPException(400, f"Unknown status {state!r}.")
    async with SessionLocal() as db:
        m = await db.get(Meeting, meeting_id)
        if m is None:
            raise HTTPException(404, "No such meeting.")
        detail = str(body.get("detail") or "")[:400]
        if state == "failed":
            m.status = "failed"
            m.error = detail or "The bot could not join."
            m.detail = ""
            m.ended_at = datetime.now(UTC)
        elif state == "ended":
            m.ended_at = datetime.now(UTC)
            m.detail = ""
            # Whatever was recorded is analysed; an empty recording is a failure
            # the page can explain, not a spinner that never stops.
            m.status = "transcribing" if m.segments else "failed"
            if not m.segments:
                m.error = detail or "The bot left without recording anything."
        else:
            m.status = state
            m.detail = detail
            if state == "in_meeting" and m.started_at is None:
                m.started_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(m)
    if state == "ended" and m.segments:
        await manks.enqueue_analysis(m)
    return {"ok": True}


@router.post("/bot/{meeting_id}/segment", dependencies=[Depends(_bot)])
async def bot_segment(
    meeting_id: uuid.UUID, request: Request, index: int, start: float = 0.0, seconds: float = 0.0
) -> dict:
    """One recorded piece of the meeting, as the raw request body."""
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty segment.")
    async with SessionLocal() as db:
        m = await db.get(Meeting, meeting_id)
        if m is None:
            raise HTTPException(404, "No such meeting.")
        key = manks.segment_key(m.owner_id, m.id, index)
        await get_storage().put(key, data, "audio/webm")
        # Replaced, not appended, so a retried upload does not duplicate it.
        kept = [s for s in (m.segments or []) if s.get("index") != index]
        kept.append(
            {"index": index, "key": key, "start": start, "seconds": seconds, "bytes": len(data)}
        )
        m.segments = sorted(kept, key=lambda s: s["index"])
        await db.commit()
    return {"ok": True, "bytes": len(data)}


@router.websocket("/bot/{meeting_id}/talk")
async def bot_talk(ws: WebSocket, meeting_id: uuid.UUID) -> None:
    """The bot's voice line: meeting audio in, the bot's spoken answers out.

    The same shared-secret door as the other bot routes (as a header, since a
    socket cannot carry a body). The bot still holds no provider keys: the
    transcription, the model and the voice all run here, on the owner's.
    """
    expected = get_settings().manks_bot_secret
    secret = ws.headers.get("x-manks-secret", "")
    if not expected or not secret or not hmac.compare_digest(secret, expected):
        await ws.close(code=4401)
        return
    async with SessionLocal() as db:
        m = await db.get(Meeting, meeting_id)
    if m is None or not m.talk:
        await ws.close(code=4404)
        return
    await ws.accept()

    from app.services import keys
    from app.services.manks_talk import Talker

    await keys.bind(m.owner_id)
    talker = Talker(ws, m.owner_id, m.bot_name, m.title)
    refusal = await talker.start()
    if refusal:
        await ws.send_text(json.dumps({"type": "error", "detail": refusal}))
        await ws.close(code=1011)
        return
    await ws.send_text(json.dumps({"type": "ready"}))
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                talker.hear(message["bytes"])
            elif message.get("text") and '"joined"' in message["text"]:
                asyncio.create_task(talker.joined())
    except WebSocketDisconnect:
        pass
    finally:
        await talker.close()

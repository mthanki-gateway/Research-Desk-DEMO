"""What the app remembers about you, and how to make it forget.

Memory that cannot be inspected is memory the user has to reverse-engineer
from behaviour. A preference captured from a throwaway remark ("just give me
the short version") then quietly shapes every later answer, and without a view
like this the only symptom is that the assistant "changed" for no reason.

So this endpoint is not a settings page -- it is the audit trail. It shows both
kinds of memory side by side, with the message each preference came from, and
offers exactly one action: forget.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.auth import User, current_user
from app.db.models import ChatSession, Preference
from app.db.session import SessionLocal
from app.schemas.profile import ConversationMemoryOut, MemoryOut, ProfileMemoryOut
from app.services import preferences as prefs_service

log = structlog.get_logger()

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/memory", response_model=ProfileMemoryOut)
async def get_memory(user: User = Depends(current_user)) -> ProfileMemoryOut:
    """Everything remembered and IN FORCE, grouped by scope.

    Forgotten preferences are left out. They used to be listed struck through
    as an audit trail, and the page became a growing list of things that no
    longer apply; `forget` is still a soft delete underneath.
    """
    async with SessionLocal() as db:
        stmt = select(Preference).where(Preference.active.is_(True))
        if user.owner_id is not None:
            stmt = stmt.where(Preference.owner_id == user.owner_id)
        # Newest first: the thing most likely to explain today's behaviour is
        # the thing captured most recently.
        rows = list(
            (await db.execute(stmt.order_by(Preference.created_at.desc()))).scalars()
        )

        # Sessions are fetched even when they carry no preferences, because a
        # conversation's summary is memory too and is the more common of the
        # two -- a chat with a summary and no preferences still belongs here.
        s_stmt = select(ChatSession)
        if user.owner_id is not None:
            s_stmt = s_stmt.where(ChatSession.owner_id == user.owner_id)
        sessions = list(
            (
                await db.execute(s_stmt.order_by(ChatSession.updated_at.desc()))
            ).scalars()
        )

    by_session: dict[uuid.UUID, list[MemoryOut]] = {}
    user_level: list[MemoryOut] = []
    for row in rows:
        out = MemoryOut.model_validate(row)
        if row.session_id is None:
            user_level.append(out)
        else:
            by_session.setdefault(row.session_id, []).append(out)

    conversations = [
        ConversationMemoryOut(
            session_id=s.id,
            title=s.title,
            summary=s.summary,
            preferences=by_session.get(s.id, []),
        )
        for s in sessions
        # A conversation with neither a summary nor a preference has nothing
        # remembered about it, and listing it would pad this view with every
        # chat the user ever opened.
        if s.summary or by_session.get(s.id)
    ]

    return ProfileMemoryOut(user_preferences=user_level, conversations=conversations)


class NewPreference(BaseModel):
    text: str = Field(min_length=2, max_length=500)


@router.post("/memory", response_model=MemoryOut, status_code=201)
async def add_memory(body: NewPreference, user: User = Depends(current_user)) -> MemoryOut:
    """A preference the person writes themselves, applying everywhere.

    Stored verbatim and tagged `user`, so it is never mistaken for one the
    assistant inferred.
    """
    async with SessionLocal() as db:
        row = Preference(
            owner_id=user.owner_id,
            text=body.text.strip(),
            origin="user",
            active=True,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return MemoryOut.model_validate(row)


@router.delete("/account")
async def delete_account(user: User = Depends(current_user)) -> dict:
    """Delete the CALLER's account data, everywhere.

    No id in the path or body: the owner comes only from the verified token,
    so there is nothing a request can change to aim this at someone else.
    """
    from app.services import account

    try:
        counts = await account.delete_account(user.owner_id)
    except account.NoAccount as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("account_delete_requested", deleted=counts)
    return {"deleted": counts}


@router.delete("/memory/{preference_id}", status_code=204)
async def forget_memory(
    preference_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    """Stop applying one preference.

    404 rather than 204 when it does not exist OR is not yours: `forget` scopes
    its own lookup by owner, so a miss covers both cases and neither one should
    report success.
    """
    if not await prefs_service.forget(preference_id, owner_id=user.owner_id):
        raise HTTPException(status_code=404, detail="Preference not found")

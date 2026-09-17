"""Howler projects: a brief, its links, and what came back.

WHY A PROJECT AND NOT A CONVERSATION

One brief is worth interviewing several people against -- that is the reason
for writing it down. So the brief, the schema it produced, the links sent out
and the results they returned all outlive any single session, and a project is
what holds them together.

WHAT A GUEST CAN SEE

Every route here requires the owner. A participant holding a link never touches
this module: they reach one WebSocket, with a token that grants exactly one
conversation. The split is deliberate and structural -- there is no code path
where an invite token can reach a project listing, rather than a check that
must be remembered on each one.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func as sql_func
from sqlalchemy import select

from app.auth import User, current_user, forbid_if_not_owner
from app.db.models import ChatSession, HowlerInvite, HowlerProject, Message
from app.db.session import SessionLocal
from app.services import blueprint, invites, profile

log = structlog.get_logger()

router = APIRouter(prefix="/howler", tags=["howler"])


def _owned(query, user: User, model):
    """Scoped by owner, with NULL handled explicitly.

    `x = NULL` is never true, so with auth off -- every row carrying a NULL
    owner -- a plain equality matches nothing at all.
    """
    return query.where(
        model.owner_id.is_(None)
        if user.owner_id is None
        else model.owner_id == user.owner_id
    )


async def _load(project_id: uuid.UUID, user: User) -> HowlerProject:
    async with SessionLocal() as db:
        project = await db.get(HowlerProject, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    forbid_if_not_owner(project.owner_id, user)
    return project


@router.post("/projects")
async def create_project(body: dict, user: User = Depends(current_user)) -> dict:
    """Turn a brief into a project with a frozen schema."""
    brief = str(body.get("brief") or "").strip()
    if not brief:
        raise HTTPException(status_code=400, detail="Describe what you want to find out.")

    try:
        fields = await blueprint.from_brief(brief)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a model failure is not our 500
        log.warning("blueprint_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=502,
            detail="The brief could not be turned into fields. Try again.",
        ) from exc

    async with SessionLocal() as db:
        project = HowlerProject(
            title=str(body.get("title") or "").strip()[:256]
            or brief.split(".")[0][:80],
            owner_id=user.owner_id,
            brief=brief[: blueprint.MAX_BRIEF_CHARS],
            participant=str(body.get("participant") or "").strip()[
                : blueprint.MAX_PARTICIPANT_CHARS
            ]
            or None,
            fields=fields,
        )
        db.add(project)
        await db.commit()
        await db.refresh(project)

    return _project_out(project)


@router.get("/projects")
async def list_projects(user: User = Depends(current_user)) -> list[dict]:
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                _owned(select(HowlerProject), user, HowlerProject).order_by(
                    HowlerProject.updated_at.desc()
                )
            )
        ).scalars().all()
        counts = dict(
            (
                await db.execute(
                    select(HowlerInvite.project_id, sql_func.count(HowlerInvite.id))
                    .where(HowlerInvite.project_id.in_([r.id for r in rows] or [None]))
                    .group_by(HowlerInvite.project_id)
                )
            ).all()
        )
    return [{**_project_out(r), "invites": counts.get(r.id, 0)} for r in rows]


@router.get("/projects/{project_id}")
async def get_project(
    project_id: uuid.UUID, user: User = Depends(current_user)
) -> dict:
    project = await _load(project_id, user)
    return _project_out(project)


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    await _load(project_id, user)
    async with SessionLocal() as db:
        project = await db.get(HowlerProject, project_id)
        if project is not None:
            await db.delete(project)
            await db.commit()


@router.post("/projects/{project_id}/invites")
async def create_invite(
    project_id: uuid.UUID, body: dict, user: User = Depends(current_user)
) -> dict:
    await _load(project_id, user)
    invite = await invites.create(
        project_id,
        label=str(body.get("label") or ""),
        participant=str(body.get("participant") or ""),
    )
    return await _invite_out(invite)


@router.get("/projects/{project_id}/invites")
async def list_invites(
    project_id: uuid.UUID, user: User = Depends(current_user)
) -> list[dict]:
    """Every link for this project, and what came back through it."""
    await _load(project_id, user)
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(HowlerInvite)
                .where(HowlerInvite.project_id == project_id)
                .order_by(HowlerInvite.created_at.desc())
            )
        ).scalars().all()
    return [await _invite_out(r) for r in rows]


@router.delete("/invites/{invite_id}", status_code=204)
async def revoke_invite(
    invite_id: uuid.UUID, user: User = Depends(current_user)
) -> None:
    """Withdraw a link. The conversation it opened is KEPT.

    Revoking is about the link, not the interview. Deleting the results
    alongside it would make "stop this going any further" and "throw away what
    we already learned" the same button, and they are not remotely the same
    decision.
    """
    async with SessionLocal() as db:
        invite = await db.get(HowlerInvite, invite_id)
        if invite is None:
            raise HTTPException(status_code=404, detail="Link not found.")
        project = await db.get(HowlerProject, invite.project_id)
    forbid_if_not_owner(project.owner_id if project else None, user)
    await invites.revoke(invite_id)


def _project_out(p: HowlerProject) -> dict:
    return {
        "id": str(p.id),
        "title": p.title,
        "brief": p.brief,
        "participant": p.participant or "",
        "fields": p.fields or [],
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


async def _invite_out(invite: HowlerInvite) -> dict:
    """A link, with the state of the interview behind it.

    The status is computed rather than stored, because it is derived from the
    conversation and storing it would be a second source of truth that drifts
    the moment an interview ends.
    """
    result = None
    status = "unopened"
    if invite.revoked_at is not None:
        status = "revoked"

    if invite.session_id is not None:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, invite.session_id)
            turns = (
                await db.execute(
                    select(sql_func.count(Message.id)).where(
                        Message.session_id == invite.session_id
                    )
                )
            ).scalar() or 0
        if chat is not None:
            data = chat.profile or {}
            fields = chat.fields or None
            if invite.revoked_at is None:
                status = "complete" if data.get("ended") else "in_progress"
            result = {
                "session_id": str(chat.id),
                "title": chat.title,
                "turns": turns // 2,
                "summary": data.get("summary") or "",
                "complete": profile.complete(data, fields),
                "missing": profile.missing(data, fields),
            }

    return {
        "id": str(invite.id),
        "token": invite.token,
        "label": invite.label,
        "participant": invite.participant or "",
        "opens": invite.opens,
        "last_opened_at": invite.last_opened_at.isoformat()
        if invite.last_opened_at
        else None,
        "revoked": invite.revoked_at is not None,
        "status": status,
        "result": result,
    }

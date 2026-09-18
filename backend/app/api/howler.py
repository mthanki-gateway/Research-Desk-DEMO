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
from app.services import blueprint, designer, invites, jobs, live, profile

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
    """Start a project.

    A brief is OPTIONAL. The designer conversation is where one gets written,
    and requiring it up front would be the form this replaced -- somebody who
    knows exactly what they want can still paste it in and skip ahead.
    """
    brief = str(body.get("brief") or "").strip()
    fields: list = []

    if brief:
        try:
            fields = await blueprint.from_brief(brief)
        except ValueError:
            # Not fatal here. The designer will ask about it, which is a better
            # conversation than an error under a text box.
            fields = []
        except Exception as exc:  # noqa: BLE001
            log.warning("blueprint_failed", error=str(exc)[:300])
            fields = []

    async with SessionLocal() as db:
        project = HowlerProject(
            title=str(body.get("title") or "").strip()[:256]
            or (brief.split(".")[0][:80] if brief else UNTITLED),
            owner_id=user.owner_id,
            brief=brief[: blueprint.MAX_BRIEF_CHARS],
            participant=str(body.get("participant") or "").strip()[
                : blueprint.MAX_PARTICIPANT_CHARS
            ]
            or None,
            fields=fields,
            design=[],
        )
        db.add(project)
        await db.commit()
        await db.refresh(project)

    return _project_out(project)


@router.post("/projects/{project_id}/design")
async def design(
    project_id: uuid.UUID, body: dict, user: User = Depends(current_user)
) -> dict:
    """One turn of the conversation that designs the interview.

    The reply and the revised artefacts come back together, so the panel beside
    the chat updates in the same round trip that answers you -- watching the
    data points change as you talk is the point of designing this way rather
    than filling in a form.
    """
    project = await _load(project_id, user)
    message = str(body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Say something.")

    history = list(project.design or [])
    try:
        result = await designer.respond(project, history, message)
    except Exception as exc:  # noqa: BLE001 - never lose the conversation
        log.warning("designer_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=502, detail="That did not go through. Try again."
        ) from exc

    row = await _apply(project_id, history, message, result)

    # IT MAY FINISH THE JOB ITSELF. The designer can decide the data points
    # will stand up to a real interview, and then there is nothing left for a
    # button to confirm -- so the link is generated on the same turn that says
    # so, and arrives beside the reply that announced it.
    invite = None
    if result.get("synthesise"):
        invite = await _invite_out(await _live_invite(project_id))

    return {
        "reply": result["reply"],
        "ready": bool(result.get("ready")),
        "project": _project_out(row),
        "invite": invite,
    }


@router.post("/projects/{project_id}/synthesise")
async def synthesise(
    project_id: uuid.UUID, body: dict, user: User = Depends(current_user)
) -> dict:
    """"That is enough talking" -- commit to a schema, and hand back a link.

    Three things in one press, because they are one decision. The data points,
    the description of who is being interviewed, and the link that will gather
    them are useless separately, and asking for them in three steps is the form
    this replaced.

    The link is REUSED if one is already live. Pressing this again after more
    conversation must not scatter a trail of links, because the one already
    sent out is the one that matters -- and it will gather the revised data
    points anyway, since a conversation takes its schema when it STARTS.
    """
    project = await _load(project_id, user)
    history = list(project.design or [])
    # Recorded in the transcript, so reading it back later shows where the
    # schema was settled rather than a reply to nothing.
    message = str(body.get("message") or "").strip() or "Synthesise now."

    try:
        result = await designer.respond(project, history, message, force=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("designer_failed", error=str(exc)[:300])
        result = {"reply": "", "ready": False}

    # The designer is one model call and can come back empty-handed. The brief
    # it has already written is enough to generate from, so a failure here
    # costs a worse schema rather than the whole press.
    if not result.get("fields") and not (project.fields or []):
        brief = result.get("brief") or project.brief
        try:
            result["fields"] = await blueprint.from_brief(brief)
            result.setdefault("reply", "")
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesise_fallback_failed", error=str(exc)[:300])
            raise HTTPException(
                status_code=422,
                detail="There is not enough here yet. Tell me what you want to "
                "find out and who you are interviewing.",
            ) from exc

    if not result.get("reply"):
        result["reply"] = (
            "Here are the data points, and a link you can send to whoever you "
            "are interviewing. Keep talking if you want to change them — the "
            "same link will gather whatever we end up with."
        )

    row = await _apply(project_id, history, message, result)
    invite = await _live_invite(project_id)

    return {
        "reply": result["reply"],
        "ready": True,
        "project": _project_out(row),
        "invite": await _invite_out(invite),
    }


UNTITLED = "Untitled project"


async def _apply(
    project_id: uuid.UUID, history: list[dict], message: str, result: dict
) -> HowlerProject:
    """Store one designer turn: the two lines said, and whatever it revised.

    The project is named here, ONCE, on the first turn that produces a brief --
    not every turn. A name that changes under you while you are still talking
    is worse than a late one, and the drawer entry moving every time you speak
    was the version of this that had to be taken out again.
    """
    async with SessionLocal() as db:
        row = await db.get(HowlerProject, project_id)
        row.design = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["reply"]},
        ][-designer.MAX_TURNS * 2 :]
        for key in ("brief", "participant"):
            if result.get(key):
                setattr(row, key, result[key])
        if result.get("fields"):
            row.fields = result["fields"]
        if result.get("vocabulary"):
            row.vocabulary = result["vocabulary"]

        if row.brief and row.title in ("", UNTITLED):
            row.title = await designer.title_for(row.brief) or row.brief[:60]

        await db.commit()
        await db.refresh(row)
    return row


async def _live_invite(project_id: uuid.UUID) -> HowlerInvite:
    """The link for this project, made if there is not one already.

    "Live" means not withdrawn and not already finished. A completed interview
    leaves a link that can never be used again, so synthesising after one has
    come back gives the next person a new link rather than a dead one.

    IT CARRIES NO PARTICIPANT OF ITS OWN, deliberately. An invite's own
    `participant` OVERRIDES the project's at connect time, which is right for a
    link made for a named person and wrong here: this link is made the moment
    the schema settles -- often on the first turn -- so copying the paragraph
    across would freeze an early guess and later refinement would never reach
    the guest. Left empty, the interviewer reads whatever the project says
    when the call actually starts.
    """
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(HowlerInvite)
                .where(
                    HowlerInvite.project_id == project_id,
                    HowlerInvite.revoked_at.is_(None),
                )
                .order_by(HowlerInvite.created_at.desc())
            )
        ).scalars().all()
        for row in rows:
            if row.session_id is None:
                return row
            chat = await db.get(ChatSession, row.session_id)
            if chat is None or not (chat.profile or {}).get("ended"):
                return row

    return await invites.create(project_id)


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


def _derived_label(result: dict | None) -> str:
    """The participant's name, if the interview learned one."""
    if not result:
        return ""
    title = str(result.get("title") or "").strip()
    # The placeholder is not a name. An interview that never learned one is
    # still called "Interview", and copying that across would label every
    # anonymous link identically -- which is worse than "Unnamed", because it
    # looks deliberate.
    return "" if title == live.UNNAMED_INTERVIEW else title


def _project_out(p: HowlerProject) -> dict:
    return {
        "id": str(p.id),
        "title": p.title,
        "brief": p.brief,
        "participant": p.participant or "",
        "fields": p.fields or [],
        "vocabulary": p.vocabulary or [],
        "design": p.design or [],
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
                # The gathered answers, with the schema THAT CONVERSATION was
                # given rather than the project's current one. A link opened
                # last week gathered last week's data points, and rendering it
                # against today's would invent empty rows for fields nobody was
                # ever asked about -- and hide the answers to ones since
                # dropped.
                "profile": data,
                "fields": fields or [],
                # How the whole conversation sounded, from the one thing that
                # heard it. Separate from the field values because it is about
                # delivery rather than content.
                "affect": data.get("affect") or None,
                # Measured from the audio, as opposed to `affect` which is what
                # the interviewer heard and put into words. Both, not either.
                "voice": data.get("voice") or None,
                # So the tab can say "queued" or "failed" rather than showing
                # nothing and looking broken while the work is still pending.
                "analysis": await jobs.status_for(chat.id, "emotion"),
            }

    return {
        "id": str(invite.id),
        "token": invite.token,
        # A LINK NAMES ITSELF once the interview has happened. Most links are
        # made without a label -- the designer mints one the moment the schema
        # settles, before anybody has said who it is for -- so the list read
        # "Unnamed" beside a finished profile that plainly knew the person's
        # name.
        #
        # Derived here rather than written to the row, for the same reason
        # `status` is: it comes from the conversation, and a stored copy is a
        # second source of truth that drifts the moment the name is corrected.
        # A label somebody typed always wins.
        "label": invite.label or _derived_label(result),
        "participant": invite.participant or "",
        "opens": invite.opens,
        "last_opened_at": invite.last_opened_at.isoformat()
        if invite.last_opened_at
        else None,
        "revoked": invite.revoked_at is not None,
        "status": status,
        "result": result,
    }

"""Links that let someone be interviewed without an account.

WHAT A TOKEN IS, AND IS NOT

It authenticates TO ONE CONVERSATION. Never as a person. A guest holding a link
can speak into that one interview and do nothing else -- not list projects, not
search the corpus, not open a different session, not see anything belonging to
the owner. That is why it is resolved here, by its own function, rather than
being fed through `current_user`: an invite that resolved to a User would
inherit every permission that user has, and the whole surface would then depend
on nobody ever passing it to the wrong dependency.

WHEN A LINK WORKS

Reusable until the interview is over, which needs no expiry of its own. The
conversation behind the link already knows whether it finished, and that is the
condition:

    revoked        -> dead, permanently
    ended          -> dead, because there is nothing left to say
    anything else  -> live

Which means a dropped call, a closed tab, or coming back after lunch all work,
and that is precisely who needs it: somebody halfway through an interview whose
connection died. An expiry measured in hours would kill exactly those and
nothing else.

The link is NOT single-use. A single-use link is one refresh away from being a
support request.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from app.db.models import ChatSession, HowlerInvite, HowlerProject
from app.db.session import SessionLocal

log = structlog.get_logger()

# 32 bytes of randomness, base64url. Long enough that guessing is not a threat
# model, short enough to survive being pasted into a chat window.
TOKEN_BYTES = 32


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


class InviteError(RuntimeError):
    """A link that cannot be used, with a reason a guest should be shown."""


async def create(
    project_id: Any, label: str = "", participant: str = ""
) -> HowlerInvite:
    async with SessionLocal() as db:
        invite = HowlerInvite(
            project_id=project_id,
            token=new_token(),
            label=(label or "").strip()[:160],
            participant=(participant or "").strip() or None,
        )
        db.add(invite)
        await db.commit()
        await db.refresh(invite)
    log.info("invite_created", project=str(project_id), label=label[:40])
    return invite


async def resolve(token: str) -> tuple[HowlerInvite, HowlerProject]:
    """The invite and its project, or a reason it cannot be used.

    Raises rather than returning None, because every refusal has a DIFFERENT
    thing the guest should be told -- "this link was withdrawn" and "this
    interview is already finished" are not the same message, and a bare 404
    makes both look like a broken link.
    """
    if not token:
        raise InviteError("This link is missing its code.")

    async with SessionLocal() as db:
        invite = (
            await db.execute(select(HowlerInvite).where(HowlerInvite.token == token))
        ).scalar_one_or_none()
        if invite is None:
            raise InviteError("This link is not valid.")
        if invite.revoked_at is not None:
            raise InviteError("This link has been withdrawn.")

        project = await db.get(HowlerProject, invite.project_id)
        if project is None:
            raise InviteError("The project behind this link no longer exists.")

        if invite.session_id is not None:
            chat = await db.get(ChatSession, invite.session_id)
            if chat is not None and (chat.profile or {}).get("ended"):
                raise InviteError("This interview is already complete. Thank you.")

        return invite, project


async def claim(token: str) -> tuple[HowlerInvite, HowlerProject]:
    """Resolve and count the open.

    The count is what makes a link auditable -- whether it was ever followed,
    and how often. Recorded on OPEN rather than on completion, because "opened
    twice and never finished" is the interesting case and completion alone
    cannot show it.
    """
    invite, project = await resolve(token)
    async with SessionLocal() as db:
        row = await db.get(HowlerInvite, invite.id)
        if row is not None:
            row.opens += 1
            row.last_opened_at = datetime.now(UTC)
            await db.commit()
    return invite, project


async def attach_session(invite_id: Any, session_id: Any) -> None:
    """Remember which conversation a link opened. Never raises."""
    try:
        async with SessionLocal() as db:
            invite = await db.get(HowlerInvite, invite_id)
            if invite is not None and invite.session_id is None:
                invite.session_id = session_id
                await db.commit()
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not end a call
        log.warning("invite_attach_failed", error=str(exc)[:200])


async def revoke(invite_id: Any) -> None:
    async with SessionLocal() as db:
        invite = await db.get(HowlerInvite, invite_id)
        if invite is not None and invite.revoked_at is None:
            invite.revoked_at = datetime.now(UTC)
            await db.commit()
            log.info("invite_revoked", invite=str(invite_id))

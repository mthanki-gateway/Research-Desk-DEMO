"""The corpus as geometry, for the Atlas view.

One endpoint rather than two, because both views come from the same pass over
the vectors: fetching them twice to serve a projection and a matrix separately
would double the work for a page that always shows both.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.auth import User, current_user, forbid_if_not_owner
from app.db.models import ChatSession, Message, Role
from app.db.session import SessionLocal
from app.services import atlas

log = structlog.get_logger()

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/atlas")
async def get_atlas(user: User = Depends(current_user)) -> dict:
    return await atlas.build(user.owner_id)


@router.get("/atlas/ray/{message_id}")
async def get_query_ray(
    message_id: uuid.UUID, user: User = Depends(current_user)
) -> dict:
    """Where this question landed in the corpus, and what it pulled in.

    Addressed by the USER message, because that is what the reader clicks --
    the question is the thing they are asking about. The retrieved set lives on
    the assistant message that answered it, so this walks one step forward.
    """
    async with SessionLocal() as db:
        message = await db.get(Message, message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Message not found.")

        chat = await db.get(ChatSession, message.session_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        forbid_if_not_owner(chat.owner_id, user)

        if message.role is not Role.user:
            raise HTTPException(
                status_code=400,
                detail="A ray is drawn for a question, not for an answer.",
            )

        # The next assistant turn in the same session. Ordered by created_at
        # rather than assumed adjacent: a turn that was interrupted or clarified
        # can leave more than one message between the two.
        answer = (
            await db.execute(
                select(Message)
                .where(
                    Message.session_id == message.session_id,
                    Message.role == Role.assistant,
                    Message.created_at >= message.created_at,
                )
                .order_by(Message.created_at.asc())
                .limit(1)
            )
        ).scalar_one_or_none()

    if answer is None:
        raise HTTPException(
            status_code=409,
            detail="That question has not been answered yet.",
        )

    sources = list(answer.sources or [])
    if not sources:
        # Small talk, a clarification, or an answer the agent gave without
        # retrieving. Not an error -- there is genuinely no ray to draw, and
        # saying which is more useful than an empty plot.
        raise HTTPException(
            status_code=409,
            detail="Nothing was retrieved for this question, so there is no ray to draw.",
        )

    return await atlas.build_ray(user.owner_id, message.content, sources)

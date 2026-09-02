import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, update

from app.auth import User, current_user
from app.config import get_settings
from app.db.models import ChatSession, Document
from app.db.session import SessionLocal
from app.services.vectorstore import get_vector_store

log = structlog.get_logger()

router = APIRouter(prefix="/auth", tags=["auth"])


class WhoAmI(BaseModel):
    authenticated: bool
    auth_enabled: bool
    user_id: str | None
    email: str | None
    n_documents: int
    n_sessions: int
    # Rows predating auth, which belong to nobody. Non-zero means the client
    # should offer to claim them.
    unclaimed_documents: int
    unclaimed_sessions: int


class ClaimResult(BaseModel):
    documents: int
    sessions: int
    vectors: int


@router.get("/me", response_model=WhoAmI)
async def me(user: User = Depends(current_user)) -> WhoAmI:
    settings = get_settings()

    async with SessionLocal() as db:
        if user.owner_id is None:
            # Auth disabled: everything is "mine", nothing is unclaimed.
            n_docs = await db.scalar(select(func.count()).select_from(Document)) or 0
            n_sessions = (
                await db.scalar(select(func.count()).select_from(ChatSession)) or 0
            )
            unclaimed_docs = unclaimed_sessions = 0
        else:
            n_docs = (
                await db.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.owner_id == user.owner_id)
                )
                or 0
            )
            n_sessions = (
                await db.scalar(
                    select(func.count())
                    .select_from(ChatSession)
                    .where(ChatSession.owner_id == user.owner_id)
                )
                or 0
            )
            unclaimed_docs = (
                await db.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.owner_id.is_(None))
                )
                or 0
            )
            unclaimed_sessions = (
                await db.scalar(
                    select(func.count())
                    .select_from(ChatSession)
                    .where(ChatSession.owner_id.is_(None))
                )
                or 0
            )

    return WhoAmI(
        authenticated=not user.anonymous,
        auth_enabled=settings.auth_enabled,
        user_id=user.owner_id,
        email=user.email,
        n_documents=n_docs,
        n_sessions=n_sessions,
        unclaimed_documents=unclaimed_docs,
        unclaimed_sessions=unclaimed_sessions,
    )


@router.post("/claim", response_model=ClaimResult)
async def claim_unowned(user: User = Depends(current_user)) -> ClaimResult:
    """Take ownership of everything created before auth was switched on.

    Idempotent -- claims only rows where owner_id IS NULL, so a second call is
    a no-op. Needed because otherwise the existing corpus becomes invisible the
    moment filtering starts, while still occupying the database.

    Updates BOTH stores. Missing the Qdrant half would leave rows visible in
    the library but unsearchable.
    """
    settings = get_settings()

    if not settings.allow_claim_unowned:
        # Disabled by default. The pre-auth corpus has already been adopted, and
        # leaving this open would let any new account claim ownerless rows that
        # appeared later. 403 rather than 404: the endpoint exists, it is
        # switched off.
        raise HTTPException(
            status_code=403,
            detail=(
                "Claiming is disabled. Set ALLOW_CLAIM_UNOWNED=true to enable "
                "it for a one-time migration."
            ),
        )

    if user.owner_id is None:
        # Auth disabled: nothing is owned, so nothing to claim.
        return ClaimResult(documents=0, sessions=0, vectors=0)

    async with SessionLocal() as db:
        docs = await db.execute(
            update(Document)
            .where(Document.owner_id.is_(None))
            .values(owner_id=user.owner_id)
        )
        sessions = await db.execute(
            update(ChatSession)
            .where(ChatSession.owner_id.is_(None))
            .values(owner_id=user.owner_id)
        )
        await db.commit()

    vectors = await get_vector_store().claim_unowned(user.owner_id)

    log.info(
        "claimed",
        owner_id=user.owner_id,
        documents=docs.rowcount,
        sessions=sessions.rowcount,
        vectors=vectors,
    )
    return ClaimResult(
        documents=docs.rowcount or 0,
        sessions=sessions.rowcount or 0,
        vectors=vectors,
    )

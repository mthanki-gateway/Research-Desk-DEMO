"""The corpus as geometry, for the Atlas view.

One endpoint rather than two, because both views come from the same pass over
the vectors: fetching them twice to serve a projection and a matrix separately
would double the work for a page that always shows both.
"""

import structlog
from fastapi import APIRouter, Depends

from app.auth import User, current_user
from app.services import atlas

log = structlog.get_logger()

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/atlas")
async def get_atlas(user: User = Depends(current_user)) -> dict:
    return await atlas.build(user.owner_id)

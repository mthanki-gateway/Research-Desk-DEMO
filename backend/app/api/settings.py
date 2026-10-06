"""Settings: the API keys a person brings, and what they unlock."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import User, current_user
from app.services import keys

router = APIRouter(prefix="/settings", tags=["settings"])


class KeyIn(BaseModel):
    key: str = Field(min_length=8, max_length=512)


@router.get("/keys")
async def get_keys(user: User = Depends(current_user)) -> dict:
    return await keys.status(user.owner_id)


@router.put("/keys/{provider}")
async def put_key(provider: str, body: KeyIn, user: User = Depends(current_user)) -> dict:
    # Checked BEFORE it is stored: a key the provider refuses would otherwise
    # be saved, unlock the feature, and fail on every request with an error
    # nobody can read.
    try:
        await keys.validate(provider, body.key.strip())
    except keys.KeyInvalid as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        await keys.save(user.owner_id, provider, body.key)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Unknown provider.") from exc
    except keys.KeysUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return await keys.status(user.owner_id)


@router.delete("/keys/{provider}")
async def delete_key(provider: str, user: User = Depends(current_user)) -> dict:
    await keys.remove(user.owner_id, provider)
    return await keys.status(user.owner_id)

"""Supabase authentication.

Supabase is used ONLY as an identity provider. Application data stays in our
own Postgres and Qdrant, so Row Level Security is irrelevant here -- RLS
protects tables inside Supabase's own database. Authorization is enforced by
this module plus the Qdrant payload filter.

Verification is local. The project signs tokens with an asymmetric key
(measured: ES256), and the matching public key comes from the JWKS endpoint.
We fetch that once, cache it, and verify every request offline -- so there is
no per-request network call to Supabase and no shared secret in our config.
Keys can be rotated on Supabase's side without redeploying us.

Auth is OPTIONAL by configuration. With SUPABASE_URL unset the app behaves
exactly as it did before auth existed: one anonymous user, owner_id None, no
filtering. That makes enabling auth a config change rather than a migration,
and keeps the app runnable before keys are in place.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import jwt
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

from app.config import get_settings

log = structlog.get_logger()

# Supabase may issue ES256 (default for new projects), RS256, or EdDSA.
# Listing all three means key rotation to a different algorithm needs no
# redeploy. HS256 is handled separately -- see _decode_symmetric.
ASYMMETRIC_ALGORITHMS = ["ES256", "RS256", "EdDSA"]

# auto_error=False so a missing header reaches our own handler and can be
# treated as "anonymous" when auth is disabled, rather than becoming a 403.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class User:
    """The authenticated caller.

    `id` is the Supabase `sub` claim, a UUID string. `owner_id` is what gets
    written to rows and vector payloads -- None for the anonymous user, which
    is what makes every query fall back to unfiltered behaviour.
    """

    id: str
    email: str | None
    anonymous: bool = False

    @property
    def owner_id(self) -> str | None:
        return None if self.anonymous else self.id


ANONYMOUS = User(id="local", email=None, anonymous=True)


@lru_cache
def _jwks_client() -> PyJWKClient:
    """One client per process. It caches keys in memory and looks them up by
    the token's `kid`, so rotation is handled without a restart."""
    settings = get_settings()
    return PyJWKClient(settings.supabase_jwks_url, cache_keys=True, lifespan=600)


def _decode_asymmetric(token: str) -> dict:
    settings = get_settings()
    signing_key = _jwks_client().get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=ASYMMETRIC_ALGORITHMS,
        audience=settings.supabase_jwt_audience,
    )


def _decode_symmetric(token: str) -> dict:
    """Legacy path: projects created before asymmetric keys became the default
    sign with a shared HS256 secret."""
    settings = get_settings()
    return jwt.decode(
        token,
        settings.supabase_jwt_secret,
        algorithms=["HS256"],
        audience=settings.supabase_jwt_audience,
    )


def _decode(token: str) -> dict:
    settings = get_settings()
    if settings.supabase_jwt_secret:
        return _decode_symmetric(token)
    return _decode_asymmetric(token)


async def _verify(token: str) -> User:
    """Verify a bearer token and map its claims onto a User.

    PyJWKClient does blocking I/O on a cache miss (the first request, and after
    the cache lifespan), so it runs in a thread rather than stalling the event
    loop for every other in-flight request.
    """
    try:
        claims = await asyncio.to_thread(_decode, token)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except PyJWKClientConnectionError as exc:
        # We could not reach the JWKS endpoint. That is OUR failure, not the
        # caller's, so it must not be a 401 -- a 401 would make the client
        # sign the user out over a transient network blip.
        log.error("jwks_unreachable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot verify credentials right now. Try again shortly.",
        ) from exc
    # PyJWKClientError is NOT a subclass of InvalidTokenError, so without this
    # a token carrying an unknown or missing `kid` escaped both handlers and
    # surfaced as a 500 instead of a 401. Measured, not theorised.
    except (jwt.InvalidTokenError, PyJWKClientError) as exc:
        # Deliberately vague to the caller, specific in the log: the reason a
        # token is invalid is useful to us and useful to an attacker.
        log.warning("token_rejected", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    subject = claims.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has no subject."
        )

    user = User(id=str(subject), email=claims.get("email"))

    # Bind the identity into the logging context so EVERY log line emitted
    # while serving this request carries it, without threading a user object
    # through every function signature.
    #
    # This is why the dependency does it and not middleware: middleware runs
    # before dependency resolution, so it cannot see a decoded token. A
    # contextvar set here is visible to everything downstream, and asyncio
    # gives each request its own context so there is no cross-talk.
    structlog.contextvars.bind_contextvars(owner_id=user.id)
    return user


async def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    """Require a caller. With auth disabled, that caller is ANONYMOUS.

    Endpoints therefore never branch on whether auth is on -- they just use
    `user.owner_id`, which is None in the anonymous case and so leaves queries
    unfiltered.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        return ANONYMOUS

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await _verify(credentials.credentials)


async def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User | None:
    """For endpoints that may serve anonymous callers even when auth is on.

    Currently used by /health and /stats, which report configuration rather
    than user data.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return ANONYMOUS
    if credentials is None or not credentials.credentials:
        return None
    return await _verify(credentials.credentials)


def forbid_if_not_owner(row_owner: str | None, user: User) -> None:
    """Guard for a fetched row.

    Raises 404, not 403: a 403 confirms the id exists, which turns any
    endpoint taking a UUID into an existence oracle. 404 leaks nothing.
    """
    if user.owner_id is None:
        return  # auth disabled — single-tenant behaviour
    if row_owner != user.owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

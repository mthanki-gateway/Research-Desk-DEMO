from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base

settings = get_settings()

# libpq spells SSL options one way and asyncpg spells them another. Managed
# Postgres providers hand out libpq-style URLs -- Neon's ends in
# `?sslmode=require&channel_binding=require` -- and asyncpg rejects BOTH of
# those as unknown connect() keyword arguments.
#
# The app cannot simply rewrite DATABASE_URL to asyncpg's spelling either,
# because the LangGraph checkpointer talks to the SAME url through psycopg,
# which is libpq and understands only `sslmode`. One url, two dialects.
#
# So: DATABASE_URL stays in the canonical libpq form that providers actually
# give you (paste it unchanged apart from the +asyncpg driver prefix), and the
# translation happens here, for asyncpg only. checkpointer.psycopg_url() keeps
# passing the libpq form straight through.
#
# Invisible locally: plain Postgres over the compose network carries no SSL
# parameters at all, so nothing exercises this until the first managed host.
_LIBPQ_ONLY_PARAMS = ("sslmode", "channel_binding")


def _split_asyncpg_url(url: str) -> tuple[str, dict[str, object]]:
    """Strip libpq-only query params, returning them as asyncpg connect_args."""
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query))
    connect_args: dict[str, object] = {}

    sslmode = params.pop("sslmode", None)
    if sslmode:
        # asyncpg >= 0.30 accepts libpq's mode NAMES for `ssl`, so the exact
        # semantic survives -- `require` still means encrypt without verifying
        # the certificate chain, which is what Neon's default url asks for.
        connect_args["ssl"] = sslmode

    # channel_binding has no asyncpg equivalent. Dropping it loses nothing that
    # matters here: it hardens SCRAM against MITM, which TLS already covers.
    for name in _LIBPQ_ONLY_PARAMS:
        params.pop(name, None)

    cleaned = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment)
    )
    return cleaned, connect_args


_url, _connect_args = _split_asyncpg_url(settings.database_url)

# One engine per process. It owns the connection pool, so creating engines
# per-request would exhaust Postgres connections almost immediately.
# max_overflow is set EXPLICITLY because its default is 10, not 0: `pool_size`
# alone is a soft target, not a limit, so the old `pool_size=10` permitted up
# to 20 connections. pool_pre_ping costs a round-trip per checkout but is worth
# it against a managed Postgres that closes idle connections -- without it the
# first query after an idle period fails instead of transparently reconnecting.
engine = create_async_engine(
    _url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    connect_args=_connect_args,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, always closed."""
    async with SessionLocal() as session:
        yield session


async def create_tables() -> None:
    """Good enough for a learning demo.

    A production app would use Alembic migrations instead -- create_all cannot
    alter an existing table, so any model change needs `docker compose down -v`.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import text
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


# Additive DDL that `create_all` cannot do, run on every boot.
#
# `create_all` only ever CREATEs missing tables; it will not ALTER one that
# already exists, which is why a model change has historically meant
# `docker compose down -v` and re-ingesting the whole corpus. These statements
# are all `IF NOT EXISTS`, so running them every boot is a no-op once applied
# and a migration on the boot after a model change.
#
# This is not a substitute for Alembic. It is the narrow subset that is safe to
# make idempotent: adding a nullable column and adding an index. Anything that
# rewrites or drops data belongs in a real migration tool.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)",
    # Content identity, scoped per owner.
    #
    # COALESCE(owner_id, '') rather than a plain two-column index, because in
    # SQL NULLs are DISTINCT from each other: with auth off every row has
    # owner_id NULL, so a plain UNIQUE (content_sha256, owner_id) would never
    # once fire. Exactly the case this is meant to catch.
    #
    # Partial on `content_sha256 IS NOT NULL` so rows ingested before the
    # column existed -- whose bytes are gone and cannot be hashed -- do not all
    # collide with each other on NULL.
    "CREATE UNIQUE INDEX IF NOT EXISTS documents_content_owner_uniq "
    "ON documents (content_sha256, COALESCE(owner_id, '')) "
    "WHERE content_sha256 IS NOT NULL",
    # Parley shares the conversation tables with the Research Desk. `kind`
    # separates them; `live_handle` is what lets a spoken conversation be
    # picked up again later, since its history lives inside the Live session
    # rather than in anything we send.
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS kind VARCHAR(16) "
    "NOT NULL DEFAULT 'chat'",
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS live_handle TEXT",
    "CREATE INDEX IF NOT EXISTS chat_sessions_kind_idx ON chat_sessions (kind)",
    # The interview's structured output, filled in by the model as it goes.
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS profile JSONB "
    "NOT NULL DEFAULT '{}'::jsonb",
    # Howler: the brief, the schema it produced, and who is being interviewed.
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS brief TEXT",
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS participant TEXT",
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS fields JSONB "
    "NOT NULL DEFAULT '[]'::jsonb",
    # A conversation run against a Howler project. The project and invite
    # TABLES are created by create_all; only this column is an alteration.
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS project_id UUID",
    "CREATE INDEX IF NOT EXISTS chat_sessions_project_idx "
    "ON chat_sessions (project_id)",
    # The designer transcript. The table itself is created by create_all; this
    # is only needed for projects that predate the conversation.
    "ALTER TABLE howler_projects ADD COLUMN IF NOT EXISTS design JSONB "
    "NOT NULL DEFAULT '[]'::jsonb",
    # Spellings for the microphone. On both tables, because a conversation
    # snapshots the project's list when it starts.
    "ALTER TABLE howler_projects ADD COLUMN IF NOT EXISTS vocabulary JSONB "
    "NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS vocabulary JSONB "
    "NOT NULL DEFAULT '[]'::jsonb",
    # Where the original upload is kept. Nullable, so every document that
    # predates object storage stays exactly as valid as it was.
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key VARCHAR(512)",
    # The job queue's claim query orders by created_at within a kind, and
    # sweeps stale `running` rows by started_at. Without this it is a sequential
    # scan on every poll -- cheap at this size and not worth leaving to grow.
    "CREATE INDEX IF NOT EXISTS jobs_claim_idx "
    "ON jobs (kind, status, created_at)",
)


async def create_tables() -> None:
    """Create anything missing, then apply the additive DDL above."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for statement in _MIGRATIONS:
            await conn.execute(text(statement))

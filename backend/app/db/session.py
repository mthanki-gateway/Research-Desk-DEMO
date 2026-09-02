from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base

settings = get_settings()

# One engine per process. It owns the connection pool, so creating engines
# per-request would exhaust Postgres connections almost immediately.
# max_overflow is set EXPLICITLY because its default is 10, not 0: `pool_size`
# alone is a soft target, not a limit, so the old `pool_size=10` permitted up
# to 20 connections. pool_pre_ping costs a round-trip per checkout but is worth
# it against a managed Postgres that closes idle connections -- without it the
# first query after an idle period fails instead of transparently reconnecting.
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
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

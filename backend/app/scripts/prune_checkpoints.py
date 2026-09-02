"""Remove orphaned LangGraph checkpoints.

Completed turns clean up after themselves (see `discard_thread`), so this is
only needed for rows left behind by a crash, or by app versions from before
that cleanup existed.

Safe to run any time: the transcript, citations and agent trace all live in
`messages`, and nothing reads a checkpoint after its turn finishes. The only
threads worth keeping are ones currently mid-run.

    docker compose exec api python -m app.scripts.prune_checkpoints
"""

import asyncio

from sqlalchemy import text

from app.db.session import engine

TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


async def main() -> None:
    async with engine.begin() as conn:
        for table in TABLES:
            before = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
            await conn.execute(text(f"TRUNCATE TABLE {table}"))
            print(f"{table}: removed {before} rows")

    # TRUNCATE reclaims space immediately, but the indexes benefit from this.
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        for table in TABLES:
            await conn.execute(text(f"VACUUM ANALYZE {table}"))
    print("vacuumed")


if __name__ == "__main__":
    asyncio.run(main())

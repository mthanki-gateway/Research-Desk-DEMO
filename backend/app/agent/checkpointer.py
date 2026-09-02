"""LangGraph's Postgres checkpointer.

This is the single line that turns a stateless graph into resumable sessions:
state is snapshotted after every node, keyed by thread_id. Hand-rolling durable
mid-execution state is most of why LangGraph is here at all.

Note the driver split -- the checkpointer speaks psycopg3 while SQLAlchemy uses
asyncpg. Same database, two pools, two URL formats.
"""

from __future__ import annotations

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

log = structlog.get_logger()

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None


def psycopg_url() -> str:
    """Convert the SQLAlchemy URL to a plain libpq one."""
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


async def init_checkpointer() -> AsyncPostgresSaver | None:
    """Open the pool and create the checkpoint tables. None on failure.

    Returning None rather than raising is deliberate: without a checkpointer the
    agent still answers questions, it just cannot resume mid-run. Losing that is
    much better than refusing to start.
    """
    global _pool, _saver
    if _saver is not None:
        return _saver

    try:
        _pool = AsyncConnectionPool(
            conninfo=psycopg_url(),
            # min_size MUST be set alongside max_size. psycopg defaults
            # min_size=4, so lowering max_size below 4 alone raises
            # "max_size must be greater or equal than min_size" -- and because
            # this whole block is wrapped in try/except, that surfaced as the
            # app booting fine but silently WITHOUT resume. Measured, not
            # theorised. 1 is right here: the pool grows on demand, and holding
            # idle connections open against a managed Postgres wastes a scarce
            # connection budget for a checkpointer that is idle between turns.
            min_size=1,
            max_size=get_settings().checkpointer_pool_size,
            # The checkpointer issues multi-statement DDL during setup(), which
            # psycopg refuses inside an implicit transaction.
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await _pool.open(wait=True, timeout=15)
        _saver = AsyncPostgresSaver(_pool)
        # Creates checkpoints / checkpoint_writes / checkpoint_blobs. Idempotent.
        await _saver.setup()
        log.info("checkpointer_ready")
        return _saver
    except Exception:
        log.exception("checkpointer_init_failed")
        _saver = None
        if _pool is not None:
            await _pool.close()
            _pool = None
        return None


async def discard_thread(thread_id: str) -> None:
    """Delete a thread's checkpoints once its result is safely persisted.

    Two problems this solves.

    1. Storage. LangGraph snapshots full state after every node and never
       prunes: ~25 test turns left 920KB across the three checkpoint tables,
       about 4x all real application data. Since threads are per-attempt,
       a finished thread is unreachable dead weight.
    2. Correctness. The append reducers on `evidence`, `sub_questions`,
       `tried_queries` and `trace` merge with whatever is already on the
       thread. Reusing a thread duplicated sub_questions and trace, and
       polluted `tried_queries` -- which makes `critique` refuse to re-run a
       query it believes was already tried.

    Safe because the transcript, citations and the agent trace are all stored
    in `messages`; nothing reads a checkpoint after a turn completes.
    """
    if _saver is None:
        return
    try:
        await _saver.adelete_thread(thread_id)
    except Exception:
        # Never fail a user's turn over cleanup. Worst case the rows linger.
        log.warning("checkpoint_cleanup_failed", thread_id=thread_id)


async def close_checkpointer() -> None:
    global _pool, _saver
    _saver = None
    if _pool is not None:
        await _pool.close()
        _pool = None

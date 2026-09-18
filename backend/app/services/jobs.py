"""Work that happens after the request is over.

WHY A QUEUE AT ALL

Analysing an interview's audio takes seconds to minutes and nobody is waiting
on it. `BackgroundTasks` -- which is what document ingest uses -- is the wrong
tool for that: it dies with the process, has no record that it ever ran, cannot
be retried, and cannot tell a Results tab whether the work is queued, running
or failed. Every one of those is something this needs.

WHY POSTGRES AND NOT CELERY / RQ / ARQ

They are all good, and all of them cost two services this deployment does not
have. Celery, RQ and arq each need a BROKER (Redis or RabbitMQ) plus a separate
WORKER process. The target is Render's free tier, which gives one web service
and 512MB; Redis would be a second hosted service and the worker a third.

Postgres is already here, already the source of truth, and already has a
migration path. So the queue is a table:

  * SURVIVES RESTARTS, which is the one thing BackgroundTasks does not. A job
    interrupted mid-flight is simply still `running` with a stale heartbeat,
    and gets picked up again.
  * STATUS IS A ROW. The Results tab asks the same database it already asks
    for everything else -- no result backend, no second store to keep in step.
  * `FOR UPDATE SKIP LOCKED` is the standard claim pattern and is safe with
    several workers, so moving the worker into its own process later changes
    where `work()` runs and nothing else.

What it gives up, honestly: polling rather than push, so a job starts within a
few seconds rather than instantly, and it would not suit thousands per second.
Neither matters for a handful of interviews a day. If that ever changes, a real
broker slots in underneath and this table stays as the record.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import text

from app.db.session import SessionLocal

log = structlog.get_logger()

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# How often an idle worker asks for work. Long enough to be invisible on the
# database, short enough that nobody notices the wait after an interview ends.
POLL_SECONDS = 3.0

# A job that has been `running` for longer than this is assumed dead -- the
# process was restarted or killed mid-flight -- and becomes claimable again.
# Generously longer than any handler should take, because reclaiming a job that
# is merely slow means running it twice.
STALE_MINUTES = 30

Handler = Callable[[dict], Awaitable[dict]]

#: Job kind -> the coroutine that does it. Registered by the module that owns
#: the work, so this one stays ignorant of what a job actually is.
HANDLERS: dict[str, Handler] = {}


def register(kind: str, handler: Handler) -> None:
    HANDLERS[kind] = handler


async def enqueue(
    kind: str,
    payload: dict,
    *,
    subject_type: str = "",
    subject_id: uuid.UUID | None = None,
    owner_id: str | None = None,
    max_attempts: int = 3,
) -> uuid.UUID | None:
    """Add a job. Returns its id, or None if it could not be recorded.

    NEVER RAISES. This is called at the end of things that have already
    succeeded -- an interview that finished, a recording that was stored -- and
    failing to queue the follow-up work must not undo any of it.

    `subject_type`/`subject_id` are how the UI finds a job without knowing what
    a payload looks like: "the emotion job for this conversation" is a query,
    not a scan.
    """
    from app.db.models import Job

    try:
        async with SessionLocal() as db:
            job = Job(
                kind=kind,
                status=QUEUED,
                payload=payload,
                subject_type=subject_type,
                subject_id=subject_id,
                owner_id=owner_id,
                max_attempts=max_attempts,
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
        log.info("job_enqueued", kind=kind, job=str(job.id), subject=str(subject_id))
        return job.id
    except Exception as exc:  # noqa: BLE001
        log.warning("job_enqueue_failed", kind=kind, error=str(exc)[:200])
        return None


async def claim() -> dict | None:
    """Take the oldest queued job, atomically. None when there is nothing to do.

    `FOR UPDATE SKIP LOCKED` is what makes this safe to run in more than one
    process: each worker locks a different row rather than queueing behind the
    same one, and a row already being claimed is skipped rather than waited on.

    Stale `running` rows are eligible too. A process killed mid-job leaves one
    behind for ever otherwise, and "it never finished and nothing retried it"
    is the failure mode a queue exists to prevent.
    """
    kinds = list(HANDLERS)
    if not kinds:
        return None

    sql = text(
        """
        UPDATE jobs
           SET status = :running,
               attempts = attempts + 1,
               started_at = now(),
               error = NULL
         WHERE id = (
               SELECT id FROM jobs
                WHERE kind = ANY(:kinds)
                  AND attempts < max_attempts
                  AND (
                        status = :queued
                     OR (status = :running
                         AND started_at < now() - make_interval(mins => :stale))
                  )
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
         )
        RETURNING id, kind, payload, attempts, max_attempts
        """
    )
    try:
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    sql,
                    {
                        "running": RUNNING,
                        "queued": QUEUED,
                        "kinds": kinds,
                        "stale": STALE_MINUTES,
                    },
                )
            ).mappings().first()
            await db.commit()
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        log.warning("job_claim_failed", error=str(exc)[:200])
        return None


async def _finish(job_id, status: str, result: dict | None, error: str | None) -> None:
    from app.db.models import Job

    try:
        async with SessionLocal() as db:
            job = await db.get(Job, job_id)
            if job is None:
                return
            job.status = status
            job.result = result or {}
            job.error = error
            job.finished_at = datetime.now(UTC)
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("job_finish_failed", job=str(job_id), error=str(exc)[:200])


async def run_one() -> bool:
    """Claim and run a single job. True if one was found.

    A handler that raises is recorded as a failure with its message, and the
    job goes back to `queued` if it has attempts left -- so a transient outage
    retries and a genuine bug does not spin forever.
    """
    job = await claim()
    if job is None:
        return False

    handler = HANDLERS.get(job["kind"])
    bound = log.bind(job=str(job["id"]), kind=job["kind"], attempt=job["attempts"])
    if handler is None:
        # Registered when it was queued, gone now -- a rename or a half-applied
        # deploy. Failing it is better than leaving it to be claimed for ever.
        await _finish(job["id"], FAILED, None, f"No handler for {job['kind']!r}")
        return True

    try:
        result = await handler(dict(job["payload"] or {}))
        await _finish(job["id"], DONE, result, None)
        bound.info("job_done")
    except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
        exhausted = job["attempts"] >= job["max_attempts"]
        await _finish(
            job["id"],
            FAILED if exhausted else QUEUED,
            None,
            str(exc)[:500],
        )
        bound.warning("job_failed", exhausted=exhausted, error=str(exc)[:200])
    return True


async def work(poll_seconds: float = POLL_SECONDS) -> None:
    """The loop. Runs until cancelled.

    Drains greedily: as long as jobs keep coming it keeps taking them, and only
    sleeps once the queue is empty. Sleeping between every job would make a
    backlog take `poll_seconds` per item to clear.
    """
    log.info("job_worker_started", kinds=sorted(HANDLERS), poll_s=poll_seconds)
    try:
        while True:
            try:
                if await run_one():
                    continue
            except Exception as exc:  # noqa: BLE001 - the loop outlives anything
                log.warning("job_worker_error", error=str(exc)[:200])
            await asyncio.sleep(poll_seconds)
    except asyncio.CancelledError:
        log.info("job_worker_stopped")
        raise


async def status_for(subject_id: uuid.UUID, kind: str = "") -> dict[str, Any] | None:
    """The latest job about `subject_id`, for the UI. None when there is none."""
    from sqlalchemy import select

    from app.db.models import Job

    try:
        async with SessionLocal() as db:
            query = (
                select(Job)
                .where(Job.subject_id == subject_id)
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            if kind:
                query = query.where(Job.kind == kind)
            job = (await db.execute(query)).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        log.warning("job_status_failed", error=str(exc)[:200])
        return None

    if job is None:
        return None
    return {
        "id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "attempts": job.attempts,
        "error": job.error or "",
        "result": job.result or {},
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }

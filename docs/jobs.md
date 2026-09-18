# Background jobs

## The idea

Some work should not happen while somebody waits. Analysing an interview's
audio takes seconds to minutes, nobody is watching, and it must survive a
restart and be able to say whether it is queued, running or failed.

## Notable decisions

**The queue is a Postgres table, not Celery, RQ or arq.**

Those are all good and all cost two services this deployment does not have.
Each needs a **broker** (Redis or RabbitMQ) plus a separate **worker** process.
The target is Render's free tier: one web service, 512 MB. Redis would be a
second hosted service and the worker a third.

Postgres is already here, already the source of truth, already has a migration
path. So:

- **Survives restarts**, which is the one thing `BackgroundTasks` — what
  document ingest uses — does not. A job interrupted mid-flight is `running`
  with a stale timestamp and becomes claimable again after 30 minutes.
- **Status is a row.** The Results tab asks the same database it already asks
  for everything else. No result backend, no second store to keep in step.
- **`FOR UPDATE SKIP LOCKED`** is the standard claim pattern and is safe across
  processes, so moving the worker out later changes *where* `work()` runs and
  nothing else.

Given up, honestly: polling rather than push, so a job starts within a few
seconds rather than instantly; and it would not suit thousands per second.
Neither matters for a handful of interviews a day. If that changes, a real
broker slots in underneath and this table stays as the record.

**Jobs are queued even when nothing can run them.** The row is the record that
an interview is *waiting* for analysis. Only queueing when the feature happens
to be switched on would silently lose every interview held before it was.

**Retries are bounded and visible.** A handler that raises goes back to
`queued` with its error recorded, until `max_attempts`. So a transient outage
retries and a genuine bug does not spin forever — and the error is on the row
rather than only in a log.

**The worker runs in the API process**, as an asyncio task. Every handler is
async and shares the same database pool. Handlers that block — model inference
does — use a thread themselves; `emotion.score_clip` goes through
`asyncio.to_thread`, because CPU-bound work on the event loop would stall every
open WebSocket, including live conversations.

## The tech

| | |
|---|---|
| Queue | `jobs` table in Postgres |
| Claim | `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED)` |
| Worker | asyncio task in the API lifespan, 3 s poll |
| Status | `jobs.status_for(subject_id, kind)` |

## How it works

```python
from app.services import jobs

jobs.register("emotion", run_emotion_job)          # at startup
await jobs.enqueue("emotion", {"session_id": ...},  # when work appears
                   subject_type="conversation", subject_id=chat.id)
await jobs.status_for(chat.id, "emotion")           # for the UI
```

`subject_type` / `subject_id` are how a UI finds a job without knowing the
shape of a payload: "the emotion job for this conversation" is a query rather
than a scan through JSON.

The worker drains greedily — as long as jobs keep coming it keeps taking them,
and only sleeps once the queue is empty. Sleeping between every job would make
a backlog take three seconds per item to clear.

`JOBS_WORKER_ENABLED=false` stops the worker without stopping the queue. Jobs
still accumulate and are picked up by whatever runs next, which is what makes
moving the worker to its own process a deployment change rather than a code
one.

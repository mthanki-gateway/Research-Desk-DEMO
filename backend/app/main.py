import asyncio
import contextlib
import logging
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph, set_graph
from app.api import (
    atlas,
    chat,
    documents,
    evaluation,
    howler,
    live,
    playground,
    profile,
    sessions,
    voice,
)
from app.api import auth as auth_api
from app.config import get_settings
from app.db.session import create_tables
from app.services import jobs, tracing
from app.services.analysis import run_emotion_job
from app.services.embeddings import close_embeddings, get_embeddings
from app.services.llm import close_llm
from app.services.vectorstore import close_vector_store, get_vector_store
from app.services.websearch import close_web_search

settings = get_settings()

logging.basicConfig(level=settings.log_level)
structlog.configure(
    # merge_contextvars must come FIRST so anything bound during the request
    # (currently owner_id, set by the auth dependency) appears on every line.
    # Prepending to the existing defaults keeps the console format unchanged.
    processors=[
        structlog.contextvars.merge_contextvars,
        *structlog.get_config()["processors"],
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelNamesMapping()[settings.log_level]
    ),
)
log = structlog.get_logger()


# What failed during startup, for /health to report.
#
# A dict rather than a flag: "the database is unreachable" and "Qdrant is
# unreachable" are different problems with different fixes, and a single
# `degraded: true` makes whoever is on call go looking for which.
BOOT_FAILURES: dict[str, str] = {}

# How long any one startup step may take before the process gives up on it.
#
# THE PORT MUST OPEN. Render scans for a listening socket and kills the deploy
# if it does not appear -- reported as "Port scan timeout reached, no open
# ports detected", which says nothing about the cause. Startup used to `await
# create_tables()` and `await init_checkpointer()` unguarded, so an unreachable
# Postgres meant the process hung before uvicorn ever bound, and the only
# evidence was that opaque line.
#
# Binding and reporting a specific failure beats never binding at all: the log
# then names the thing that is broken, and /health says so on every request
# instead of the service simply not existing.
BOOT_TIMEOUT_SECONDS = 20


async def _boot_step(name: str, coro):
    """Run one startup step. Never raises, never hangs for ever."""
    try:
        return await asyncio.wait_for(coro, timeout=BOOT_TIMEOUT_SECONDS)
    except TimeoutError:
        BOOT_FAILURES[name] = f"timed out after {BOOT_TIMEOUT_SECONDS}s"
        log.error("boot_step_timeout", step=name, seconds=BOOT_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - report every failure the same way
        BOOT_FAILURES[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
        log.exception("boot_step_failed", step=name)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "starting",
        env=settings.app_env,
        docs=settings.docs_enabled,
        profile=settings.model_profile,
        llm=settings.llm_model,
        answer_model=settings.answer_model,
        rewriter=settings.rewriter_model,
        embeddings=f"{settings.embedding_provider}:{settings.embedding_model}",
        qdrant=settings.qdrant_url,
    )

    # A profile's fields are a coherent SET -- models, token ceiling, history
    # strategy and hop budgets together. One stale environment variable
    # overriding a single field produces a mixture that is worse than either
    # profile, and it is invisible: every value looks plausible on its own.
    # Explicit settings still win; this is what stops that being silent.
    for field, actual, expected in settings.profile_conflicts():
        log.warning(
            "model_profile_overridden",
            field=field,
            configured=actual,
            profile_expects=expected,
            profile=settings.model_profile,
            hint=f"unset {field.upper()} in .env to use the profile's value",
        )

    await _boot_step("database", create_tables())

    # Build the embedding provider and Qdrant collection up front, so a
    # misconfigured key or a dimension mismatch fails loudly at boot rather
    # than halfway through a user's first upload.
    try:
        provider = get_embeddings()
        await get_vector_store().ensure_collection()
        log.info("vector_store_ready", dim=provider.dim)
    except Exception:
        log.exception("vector_store_init_failed")

    # Compile the graph WITH the checkpointer, so agent state is snapshotted to
    # Postgres after every node and a session can resume. If the checkpointer
    # fails we still serve, just without resume.
    saver = await _boot_step("checkpointer", init_checkpointer())
    set_graph(build_graph(checkpointer=saver))
    log.info("agent_graph_compiled", checkpointer=saver is not None)

    # Initialised at boot rather than on the first request, so a wrong host or
    # a bad key shows up in the startup log instead of adding latency to
    # someone's first question. Returns None and logs when unconfigured.
    tracing.get_tracer()

    # THE JOB WORKER, in this process. See `services/jobs.py` for why the queue
    # is a Postgres table rather than Celery: a broker plus a worker is two
    # more services than the target deployment has.
    #
    # A task rather than a thread, because every handler is async and shares
    # this loop's database pool. Handlers that block -- model inference does --
    # are responsible for using a thread themselves, which `emotion.score_clip`
    # does.
    worker: asyncio.Task | None = None
    if settings.jobs_worker_enabled:
        # REGISTERED ONLY WHEN IT CAN ACTUALLY RUN.
        #
        # `claim` only takes kinds that have a handler, so leaving this
        # unregistered means emotion jobs sit QUEUED rather than being claimed
        # and failed. That is the honest state: the interview is waiting for
        # analysis, and switching the model on later picks up everything that
        # accumulated in the meantime.
        #
        # Registering it regardless and raising "EMOTION_ANALYSIS is off" was
        # worse than useless -- it reported a failure for a deployment working
        # exactly as configured, and burned the retry budget doing it, so the
        # day somebody enabled the model every earlier interview was already
        # permanently failed.
        if settings.emotion_analysis:
            jobs.register("emotion", run_emotion_job)
        worker = asyncio.create_task(jobs.work())

    yield

    # Stopped FIRST, before the clients it uses are closed underneath it.
    if worker is not None:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    # Flush BEFORE anything else closes. The SDK batches in a background
    # thread, so a process that exits promptly drops its last traces -- and the
    # traces most worth having are the ones from just before a shutdown.
    tracing.flush()
    await close_checkpointer()
    await close_web_search()
    await close_vector_store()
    await close_embeddings()
    await close_llm()
    log.info("shutting down")


# In prod all three of these are None, which means the routes are NOT
# REGISTERED AT ALL -- not 404'd by a guard, simply absent from the router.
# There is no code path left to misconfigure.
#
# openapi_url has to go too, not just docs_url. /docs is only a JavaScript shell
# that fetches the schema from /openapi.json, so disabling the viewer while
# leaving the schema served would hide the front door and leave the entire API
# surface readable to anyone who guesses the URL. That is the mistake this
# comment exists to prevent.
_docs = settings.docs_enabled
app = FastAPI(
    title="Research Desk API",
    version="0.4.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_context(request: Request, call_next):
    """Give each request a clean logging context, and tag it with a request id.

    This is the one job middleware is genuinely better at than a dependency:
    it is mechanical, applies to every request without exception, and must run
    *before* anything else. Identity, by contrast, belongs in a dependency --
    middleware runs before dependency resolution and so cannot decode a token.

    Clearing matters because contextvars can survive into a later request when
    the event loop reuses a task, which would attribute one user's log lines to
    another.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex[:8])
    response = await call_next(request)
    # Handed back so a user can quote it when reporting a problem, and it can
    # be grepped straight out of the logs.
    response.headers["X-Request-ID"] = (
        structlog.contextvars.get_contextvars().get("request_id", "")
    )
    return response

app.include_router(auth_api.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(evaluation.router)
app.include_router(profile.router)
app.include_router(playground.router)
app.include_router(atlas.router)
app.include_router(voice.router)
app.include_router(live.router)
app.include_router(howler.router)


class Health(BaseModel):
    status: str
    llm_model: str
    embedding_provider: str
    google_api_key_present: bool
    # Lets the frontend decide whether to show a login screen without needing
    # its own copy of the configuration.
    auth_enabled: bool
    # WHICH STORE IS ACTUALLY LIVE -- "local" or "s3" -- and the bucket, when
    # there is one.
    #
    # Here because the failure it catches is silent. Editing `.env` and
    # restarting does NOT change a container's environment: `docker compose
    # restart`, and Docker Desktop's restart button, both reuse the existing
    # one. So the app carries on with the old configuration, and with storage
    # that looks like nothing at all -- uploads succeed, text indexes, answers
    # cite correctly, and the original is quietly going somewhere else.
    #
    # The same reading catches the deploy version of it: Render's free tier has
    # no persistent disk, so `local` there loses every file on the next
    # restart. One GET now answers "where are my files actually going".
    storage_backend: str
    storage_bucket: str = ""
    # Which startup steps failed, by name. Empty when everything came up.
    #
    # NAMED, not a boolean: "the database is unreachable" and "Qdrant is
    # unreachable" are different problems with different fixes, and a bare
    # `degraded: true` makes whoever is looking go and find out which.
    boot_failures: dict[str, str] = {}


@app.get("/health", response_model=Health, tags=["meta"])
async def health() -> Health:
    """Liveness probe. Also the endpoint a free-tier keep-alive pinger would hit."""
    return Health(
        status="degraded" if BOOT_FAILURES else "ok",
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        google_api_key_present=bool(settings.google_api_key),
        auth_enabled=settings.auth_enabled,
        storage_backend="s3" if settings.storage_backend in ("s3", "r2") else "local",
        storage_bucket=settings.s3_bucket,
        boot_failures=dict(BOOT_FAILURES),
    )

import logging
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph, set_graph
from app.api import auth as auth_api
from app.api import chat, documents, evaluation, sessions
from app.config import get_settings
from app.db.session import create_tables
from app.services.embeddings import close_embeddings, get_embeddings
from app.services.llm import close_llm
from app.services.vectorstore import close_vector_store, get_vector_store

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "starting",
        env=settings.app_env,
        docs=settings.docs_enabled,
        llm=settings.llm_model,
        embeddings=f"{settings.embedding_provider}:{settings.embedding_model}",
        qdrant=settings.qdrant_url,
    )

    await create_tables()

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
    saver = await init_checkpointer()
    set_graph(build_graph(checkpointer=saver))
    log.info("agent_graph_compiled", checkpointer=saver is not None)

    yield

    await close_checkpointer()
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


class Health(BaseModel):
    status: str
    llm_model: str
    embedding_provider: str
    google_api_key_present: bool
    # Lets the frontend decide whether to show a login screen without needing
    # its own copy of the configuration.
    auth_enabled: bool


@app.get("/health", response_model=Health, tags=["meta"])
async def health() -> Health:
    """Liveness probe. Also the endpoint a free-tier keep-alive pinger would hit."""
    return Health(
        status="ok",
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        google_api_key_present=bool(settings.google_api_key),
        auth_enabled=settings.auth_enabled,
    )

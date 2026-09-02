# syntax=docker/dockerfile:1

# uv's official image ships the resolver + a pinned CPython, so no pip bootstrap.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies resolve in their own layer so app edits don't bust the cache.
#
# uv.lock is COPYed in and `--frozen` forbids uv from updating or ignoring it.
# Without the lock committed, every build re-resolved the `>=` ranges in
# pyproject.toml from scratch, so local, CI and Render could each end up on
# different transitive versions -- and a dependency publishing a breaking
# release would break production with no commit to blame it on.
#
# `--frozen` also fails loudly when pyproject.toml and uv.lock disagree, which
# turns "someone added a dependency without relocking" into a build error
# instead of a silent divergence. Regenerate with:
#   docker compose exec api uv lock
#   docker compose cp api:/app/uv.lock ./backend/uv.lock
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- dev: source is bind-mounted by compose, so don't COPY it in -------------
FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group dev
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- prod: self-contained image for Render ----------------------------------
FROM base AS prod
COPY app ./app
EXPOSE 8000
# Render injects $PORT and it is not always 8000, so honour it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

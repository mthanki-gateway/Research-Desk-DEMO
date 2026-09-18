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
# `--extra storage` is on the BASE sync, not just dev, because `prod` inherits
# from base -- and prod is the stage that actually talks to object storage.
# boto3 is ~15MB; the alternative is a deploy where STORAGE_BACKEND=s3 raises
# on the first upload and nothing in the build log hints at why.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra storage

# --- dev: source is bind-mounted by compose, so don't COPY it in -------------
FROM base AS dev
# `--extra eval` installs RAGAS, and its absence here was a silent hole.
#
# Tier 2 imports ragas function-locally and degrades to JudgeScores.failed()
# when it is missing, which is the right behaviour for an optional extra --
# except that it meant the whole judge reported "ragas is not installed" rather
# than a score, in the one image where evaluation actually runs. Every Tier 2
# number was None and the harness said so quietly, in a field nobody reads.
#
# DEV ONLY, deliberately. ragas pulls pandas, datasets and pyarrow; the prod
# stage below stays at 177MB and cannot judge anything, which is correct --
# evaluation is a development activity and serving a request never needs it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group dev --extra eval --extra speech         --extra storage
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- prod: self-contained image for Render ----------------------------------
#
# KEEP THIS STAGE LAST. Render's UI has no "target stage" setting -- it builds
# the FINAL stage in the Dockerfile. So `prod` being last is what makes the
# deploy correct, and it is load-bearing: move `dev` below this and Render
# silently ships a container running --reload with pytest and ruff installed,
# with nothing in its build log to suggest anything is wrong.
#
# docker-compose.yml is unaffected either way -- it names its target explicitly.
FROM base AS prod
COPY app ./app
EXPOSE 8000
# Render injects $PORT and it is not always 8000, so honour it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

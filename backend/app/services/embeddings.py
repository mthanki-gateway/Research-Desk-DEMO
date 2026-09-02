"""Embedding providers behind one interface.

Two implementations, chosen by EMBEDDING_PROVIDER:
  gemini    -- Gemini Embedding over the AI Studio key (free tier, quota'd)
  fastembed -- local ONNX bge-small, no network, no quota, lower recall

The interface exists because the free tier can run dry mid-session; swapping
providers must not require touching ingestion or retrieval code.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import Literal

import httpx
import structlog

from app.config import get_settings
from app.services.limiter import RateLimiter, estimate_tokens

log = structlog.get_logger()

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

GENAI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None:  # pragma: no cover - trivial
        return None


class GeminiEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        s = get_settings()
        if not s.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for EMBEDDING_PROVIDER=gemini")

        self.dim = s.embedding_dim
        self._model = s.embedding_model.removeprefix("models/")
        self._batch_size = min(s.embedding_batch_size, 100)  # API hard cap
        self._client = httpx.AsyncClient(
            base_url=GENAI_BASE,
            timeout=httpx.Timeout(120.0),
            headers={"x-goog-api-key": s.google_api_key},
        )
        self._limiter = RateLimiter(
            requests_per_minute=s.embedding_requests_per_minute,
            tokens_per_minute=s.embedding_tokens_per_minute,
            name="embeddings",
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _split_batches(self, texts: list[str]) -> list[list[str]]:
        """Split on BOTH limits: item count (API cap 100) and token budget.

        A batch bigger than the whole per-minute token budget could never be
        admitted by the limiter, so cap batches there too.
        """
        budget = self._limiter.token_capacity
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            tokens = estimate_tokens(text)
            too_many = len(current) >= self._batch_size
            too_big = current and (current_tokens + tokens) > budget
            if too_many or too_big:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    async def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        tokens = sum(estimate_tokens(t) for t in texts)

        # Up to 4 attempts: the limiter prevents most 429s, but a shared key or
        # a bad token estimate can still trip one.
        last_error: Exception | None = None
        for attempt in range(4):
            await self._limiter.acquire(tokens)
            try:
                resp = await self._client.post(
                    f"/models/{self._model}:batchEmbedContents", json=payload
                )
                resp.raise_for_status()
                embeddings = resp.json()["embeddings"]
                # Google only guarantees unit length at the native 3072 dims;
                # a truncated vector must be re-normalised or cosine distance
                # is subtly wrong. Normalising an already-unit vector is a
                # no-op, so this is unconditionally safe.
                return [l2_normalize(e["values"]) for e in embeddings]
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in (429, 500, 503):
                    raise
                backoff = 2**attempt * 5
                log.warning(
                    "embed_retry",
                    status=exc.response.status_code,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError(f"embedding failed after retries: {last_error}")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in self._split_batches(texts):
            out.extend(await self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return out

    async def embed_query(self, text: str) -> list[float]:
        # RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT is not cosmetic: the model
        # projects questions and passages differently, and mismatching them
        # measurably degrades recall.
        result = await self._embed_batch([text], "RETRIEVAL_QUERY")
        return result[0]


class FastEmbedEmbeddings(EmbeddingProvider):
    """Local ONNX fallback. Offline, unlimited, 384-dim, weaker recall."""

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fastembed is not installed. Rebuild with: "
                "uv sync --extra local-embeddings"
            ) from exc

        s = get_settings()
        self._model = TextEmbedding(model_name=s.fastembed_model)
        self.dim = 384  # bge-small-en-v1.5 is fixed at 384

    async def _run(self, texts: list[str]) -> list[list[float]]:
        # fastembed is synchronous CPU work; offload so it can't block the
        # event loop while a request is being served.
        def _work() -> list[list[float]]:
            return [l2_normalize(v.tolist()) for v in self._model.embed(texts)]

        return await asyncio.to_thread(_work)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._run(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._run([text]))[0]


_provider: EmbeddingProvider | None = None


def get_embeddings() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        s = get_settings()
        _provider = (
            FastEmbedEmbeddings() if s.embedding_provider == "fastembed" else GeminiEmbeddings()
        )
        log.info("embeddings_ready", provider=s.embedding_provider, dim=_provider.dim)
    return _provider


async def close_embeddings() -> None:
    global _provider
    if _provider is not None:
        await _provider.aclose()
        _provider = None

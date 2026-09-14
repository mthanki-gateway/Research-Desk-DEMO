"""Web search via Serper, for questions the documents cannot answer.

WHY THIS EXISTS

A document-grounded agent has one honest failure mode: the corpus does not
contain the answer. Until now the only response was to say so. That is correct
behaviour and often useless -- "what is HNSW?" is not in anyone's quarterly
report, but it is a reasonable thing to ask a research assistant.

WHAT IT DOES NOT CHANGE

Grounding. A web result is still a SOURCE that must be cited, not licence to
answer from the model's own memory. The whole citation contract survives: every
claim points at something, and a reader can follow it. The difference is only
that some sources are now URLs rather than chunks.

Inert without `SERPER_API_KEY`, the same way tracing is inert without Langfuse
keys -- so the app behaves exactly as it did before this file existed.
"""

from __future__ import annotations

import uuid

import httpx
import structlog

from app.config import get_settings
from app.services import progress, tracing
from app.services.limiter import RateLimiter
from app.services.vectorstore import SearchHit

log = structlog.get_logger()

SERPER_URL = "https://google.serper.dev/search"

# Stable namespace for the synthetic chunk ids web results carry.
#
# uuid5 rather than uuid4, so the SAME url produces the SAME id on every
# search. That is what makes dedupe work: `merge_evidence` keys on chunk_id,
# and a random id per call would let one page accumulate several times across
# ReAct iterations, each copy eating a citation slot.
_WEB_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


class WebSearchError(RuntimeError):
    pass


_client: httpx.AsyncClient | None = None
_limiter: RateLimiter | None = None


def enabled() -> bool:
    return bool(get_settings().serper_api_key)


def _get_client() -> httpx.AsyncClient:
    global _client, _limiter
    settings = get_settings()
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
            headers={
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json",
            },
        )
        _limiter = RateLimiter(
            requests_per_minute=settings.serper_requests_per_minute,
            # Serper bills per search, not per token, so the token bucket is
            # effectively unused. It is set high rather than removed because
            # RateLimiter budgets both dimensions.
            tokens_per_minute=10**9,
            name="web:serper",
        )
    return _client


async def close_web_search() -> None:
    global _client, _limiter
    if _client is not None:
        await _client.aclose()
    _client = None
    _limiter = None


def _to_hit(item: dict, rank: int, query: str) -> SearchHit | None:
    """One organic result as a SearchHit. None if it carries no usable text."""
    url = (item.get("link") or "").strip()
    title = (item.get("title") or "").strip()
    snippet = (item.get("snippet") or "").strip()
    if not url or not snippet:
        # A result with no snippet has nothing to ground a claim in, and an
        # uncitable source is worse than a missing one.
        return None

    return SearchHit(
        chunk_id=uuid.uuid5(_WEB_NAMESPACE, url),
        # Web results have no document row. A nil UUID is the honest value --
        # it is not a document, and inventing one would break the "open this
        # document" path in the UI.
        document_id=uuid.UUID(int=0),
        filename=title or url,
        page=None,
        chunk_index=rank,
        heading=None,
        text=snippet,
        # Serper returns rank order, not scores. Deriving a pseudo-score from
        # rank keeps the field meaningful for display, and RRF ignores it
        # anyway -- fusion works on position.
        score=round(1.0 / (1 + rank), 4),
        meta={"date": item.get("date"), "position": item.get("position")},
        found_by=[query],
        source="web",
        url=url,
    )


async def search_web(query: str, *, limit: int | None = None) -> list[SearchHit]:
    """Search the web. Returns [] rather than raising when unavailable.

    Never load-bearing: the agent must still answer from documents if the web
    is unreachable or unconfigured, so every failure here degrades to "no web
    results" rather than failing the turn.
    """
    if not enabled():
        return []

    settings = get_settings()
    n = limit or settings.web_search_results

    with tracing.observe(
        "web.search",
        as_type="retriever",
        input=query,
        metadata={"limit": n, "provider": "serper"},
    ) as span:
        progress.searching("web", query)
        try:
            client = _get_client()
            if _limiter is not None:
                await _limiter.acquire(1)
            resp = await client.post(SERPER_URL, json={"q": query, "num": n})
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            log.warning("web_search_failed", query=query[:60], error=str(exc))
            tracing.update(span, level="ERROR", status_message=str(exc)[:200])
            return []

        organic = payload.get("organic") or []
        hits = [h for i, item in enumerate(organic[:n]) if (h := _to_hit(item, i, query))]

        # The "answer box" / knowledge panel, when Google produced one. Put
        # FIRST because it is a direct answer rather than a page that might
        # contain one -- and it is exactly what a factual lookup wants.
        box = payload.get("answerBox") or {}
        direct = (box.get("answer") or box.get("snippet") or "").strip()
        if direct:
            hits.insert(
                0,
                SearchHit(
                    chunk_id=uuid.uuid5(_WEB_NAMESPACE, f"answerbox:{query}"),
                    document_id=uuid.UUID(int=0),
                    filename=box.get("title") or "Web answer",
                    page=None,
                    chunk_index=-1,
                    heading=None,
                    text=direct,
                    score=1.0,
                    meta={"answer_box": True},
                    found_by=[query],
                    source="web",
                    url=(box.get("link") or "").strip() or None,
                ),
            )

        tracing.update(
            span,
            output=[{"title": h.filename, "url": h.url} for h in hits],
            metadata={"n_hits": len(hits), "answer_box": bool(direct)},
        )
        progress.searched("web", query, len(hits))
        log.info("web_searched", query=query[:60], n=len(hits), answer_box=bool(direct))
        return hits

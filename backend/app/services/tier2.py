"""Tier 2 evaluation: generation quality, judged by RAGAS.

Tier 1 asks "did retrieval return the right chunks". Tier 2 asks "given those
chunks, was the answer any good".

**Answers are cached to disk, and that is the design decision that makes this
usable.** Generating answers is the expensive half: each question is a full
agent run -- plan, retrieve, draft, critique, sometimes a retry loop -- so 3 to
5 Gemma calls against a 16K tokens/minute budget, or roughly 7 minutes for 20
questions. Judging is comparatively cheap AND runs on a different model with a
different quota (Flash Lite, 250K tokens/minute).

So the two phases are separated by a cache. Change a judge prompt or a metric
and re-judging costs about a minute instead of eight, because the agent never
runs again. Without that separation nobody iterates on the judge.

The cache key includes every input that changes an answer -- mode, top_k,
multi_query, model -- so a config change correctly misses rather than silently
scoring stale answers against new settings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

from app.config import get_settings
from app.services.golden import GoldenQuestion
from app.services.judge import (
    CHEAP_METRICS,
    JudgeScores,
    judge_answer,
    ragas_available,
)

log = structlog.get_logger()

# Gitignored: these are generated artifacts, and they contain model output that
# would otherwise churn the diff on every run.
CACHE_DIR = Path("/app/.eval-cache")


@dataclass(slots=True)
class GeneratedAnswer:
    question_id: str
    question: str
    answer: str
    contexts: list[str]
    # Agent telemetry, kept so a bad score can be traced to how it was produced.
    iterations: int = 0
    sufficient: bool = True
    sub_questions: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class Tier2QuestionResult:
    question_id: str
    question: str
    tags: list[str]
    answerable: bool
    answer: str
    reference: str
    n_contexts: int
    from_cache: bool
    scores: dict
    generation_error: str | None = None


@dataclass(slots=True)
class Tier2Report:
    config: dict
    questions: list[Tier2QuestionResult]
    aggregates: dict
    elapsed_seconds: float
    n_generated: int
    n_from_cache: int

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "n_generated": self.n_generated,
            "n_from_cache": self.n_from_cache,
            "aggregates": self.aggregates,
            "questions": [asdict(q) for q in self.questions],
        }


def _cache_key(question_id: str, config: dict) -> str:
    """Hash of the question plus every input that changes its answer.

    Omitting any of these would let a config change reuse an answer produced
    under different settings -- the worst kind of evaluation bug, because the
    numbers stay plausible.
    """
    payload = json.dumps({"id": question_id, **config}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{question_id}.{digest}.json"


def _read_cache(path: Path) -> GeneratedAnswer | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GeneratedAnswer(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        # A corrupt or stale-shaped cache entry is not worth failing over --
        # regenerate instead.
        return None


# Cache I/O goes through asyncio.to_thread rather than being called directly.
# Path.read_text and .write_text are BLOCKING, and blocking the event loop
# inside an async function is what ruff's ASYNC rules exist to catch. It is
# harmless in a CLI run, but `run_tier2` is also callable from the API, where
# a blocked loop stalls every other in-flight request.
async def _load_cached(path: Path) -> GeneratedAnswer | None:
    return await asyncio.to_thread(_read_cache, path)


async def _save_cached(path: Path, answer: GeneratedAnswer) -> None:
    await asyncio.to_thread(
        path.write_text, json.dumps(asdict(answer), indent=2), encoding="utf-8"
    )


async def _generate(
    question: GoldenQuestion,
    *,
    mode: str,
    top_k: int,
    multi_query: bool,
    owner_id: str | None,
) -> GeneratedAnswer:
    """Produce an answer for one question. Never raises."""
    # Imported here rather than at module scope: `graph` pulls in the compiled
    # LangGraph and the checkpointer, which a caching-only run does not need.
    from app.agent.graph import run_agent
    from app.services.rag import answer_question

    try:
        if mode == "baseline":
            # Plain RAG: one retrieval, one generation. Kept as the comparison
            # point -- without it "the agent is better" is an assertion.
            result = await answer_question(
                question.question,
                top_k=top_k,
                owner_id=owner_id,
                multi_query=multi_query,
            )
            return GeneratedAnswer(
                question_id=question.id,
                question=question.question,
                answer=result.answer,
                contexts=[h.text for h in result.hits],
                citations=list(result.sources_used),
            )

        result = await run_agent(
            question.question,
            top_k=top_k,
            owner_id=owner_id,
            multi_query=multi_query,
        )
        return GeneratedAnswer(
            question_id=question.id,
            question=question.question,
            answer=result.answer,
            contexts=[h.text for h in result.evidence],
            iterations=result.iterations,
            sufficient=result.sufficient,
            sub_questions=list(result.sub_questions),
            citations=list(result.citations),
        )
    except Exception as exc:  # noqa: BLE001 - one failure must not end the run
        log.warning("tier2_generate_failed", question_id=question.id, error=str(exc))
        return GeneratedAnswer(
            question_id=question.id,
            question=question.question,
            answer="",
            contexts=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(results: list[Tier2QuestionResult]) -> dict:
    """Mean of each metric, skipping None.

    None means "not measured" -- a failed judge call, or a reference-based
    metric on a question with no reference. Averaging those in as zero would
    make a working system look broken, which is the same trap recall avoids
    with unanswerable questions.
    """
    out: dict = {"n_questions": len(results)}
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        values = [
            r.scores[metric]
            for r in results
            if r.scores.get(metric) is not None
        ]
        out[metric] = _mean(values)
        out[f"{metric}_n"] = len(values)
    out["n_judge_errors"] = sum(1 for r in results if r.scores.get("error"))
    return out


async def run_tier2(
    questions: list[GoldenQuestion],
    *,
    mode: str = "agent",
    top_k: int | None = None,
    multi_query: bool = False,
    owner_id: str | None = None,
    use_cache: bool = True,
    judge: bool = True,
    metrics: tuple[str, ...] = CHEAP_METRICS,
    filters: dict | None = None,
) -> Tier2Report:
    """Generate answers (cached) then judge them with RAGAS.

    `judge=False` generates and caches only -- useful for paying the expensive
    phase once, in the background, before iterating on the judge.
    """
    settings = get_settings()
    fetch = top_k or settings.retrieval_top_k
    gen_config = {
        "mode": mode,
        "top_k": fetch,
        "multi_query": multi_query,
        "model": settings.llm_model,
    }

    await asyncio.to_thread(CACHE_DIR.mkdir, parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[Tier2QuestionResult] = []
    n_generated = n_cached = 0

    for question in questions:
        path = CACHE_DIR / _cache_key(question.id, gen_config)
        generated = await _load_cached(path) if use_cache else None
        # Captured per question. Deriving this from the running counters was a
        # bug: once anything had been cached, every later row claimed to be.
        came_from_cache = generated is not None

        if generated is None:
            generated = await _generate(
                question,
                mode=mode,
                top_k=fetch,
                multi_query=multi_query,
                owner_id=owner_id,
            )
            # Only cache a real answer. Caching a failure would make the next
            # run silently reuse it and hide a transient outage forever.
            if not generated.error:
                await _save_cached(path, generated)
            n_generated += 1
        else:
            n_cached += 1

        if judge and generated.answer and not generated.error:
            scores = await judge_answer(
                question=question.question,
                answer=generated.answer,
                contexts=generated.contexts,
                reference=question.expected_answer,
                metrics=metrics,
            )
        else:
            scores = JudgeScores.failed(
                generated.error or ("judging skipped" if not judge else "empty answer")
            )

        results.append(
            Tier2QuestionResult(
                question_id=question.id,
                question=question.question,
                tags=list(question.tags),
                answerable=question.answerable,
                answer=generated.answer,
                reference=question.expected_answer,
                n_contexts=len(generated.contexts),
                from_cache=came_from_cache,
                scores=scores.to_dict(),
                generation_error=generated.error,
            )
        )

    elapsed = time.perf_counter() - started
    available, why = ragas_available()
    report = Tier2Report(
        config={
            **gen_config,
            "judge_model": settings.judge_model,
            "judged": judge,
            "metrics": list(metrics),
            "ragas_available": available,
            "ragas_note": why,
            "n_questions": len(questions),
            "filters": filters or {},
            "question_ids": [q.id for q in questions],
        },
        questions=results,
        aggregates=_aggregate(results),
        elapsed_seconds=elapsed,
        n_generated=n_generated,
        n_from_cache=n_cached,
    )
    log.info(
        "tier2_complete",
        n=len(questions),
        generated=n_generated,
        cached=n_cached,
        seconds=round(elapsed, 1),
        faithfulness=report.aggregates.get("faithfulness"),
    )
    return report

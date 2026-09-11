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
    ALL_METRICS,
    JudgeScores,
    judge_answer,
    ragas_available,
)
from app.services.retrieval import context_blocks

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
                contexts=context_blocks(result.hits),
                citations=list(result.sources_used),
            )

        result = await run_agent(
            question.question,
            top_k=top_k,
            owner_id=owner_id,
            multi_query=multi_query,
            # PINNED, not left to REACT_DEFAULT -- and this is load-bearing.
            #
            # "agent" in this harness means the plan/retrieve/draft/critique
            # graph. That is what every recorded recall, faithfulness and
            # iteration number measures. When REACT_DEFAULT flipped to true,
            # omitting this silently changed what "agent" meant, so old and
            # new runs would sit in the same table measuring different
            # systems. A config default must never be able to redefine the
            # thing under test.
            react=False,
            # Likewise: the harness answers questions, it does not converse,
            # so a paused turn would score as an empty answer.
            clarify=False,
        )
        return GeneratedAnswer(
            question_id=question.id,
            question=question.question,
            answer=result.answer,
            contexts=context_blocks(result.evidence),
            iterations=result.iterations,
            sufficient=result.sufficient,
            sub_questions=list(result.sub_questions),
            citations=list(result.citations),
            # `draft` turns a quota error or a 503 into readable text instead
            # of raising -- right for a chat window, wrong here. Without this
            # the harness cannot tell an outage from an answer: it cached
            # "The answer could not be generated (503...)" and scored it for
            # faithfulness, where no passage supports it, dragging the mean
            # down on every later run.
            error=(
                f"generation failed: {result.answer[:200]}" if result.failed else None
            ),
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
    metrics: tuple[str, ...] = ALL_METRICS,
    judge_concurrency: int = 4,
    filters: dict | None = None,
) -> Tier2Report:
    """Generate answers (cached) then judge them with RAGAS.

    TWO PHASES, and they are separate loops rather than one, because they have
    different bottlenecks:

    * GENERATION is answer-model bound (5 rpm) and mostly served from cache, so
      it stays sequential -- simple, and the cache makes it free on re-runs.
    * JUDGING is judge-model bound and latency-heavy: the full quartet is
      ~10-15 calls per question, each a round-trip. Run sequentially, the run
      spends most of its wall clock waiting rather than being throttled.

    So judging is issued CONCURRENTLY, bounded by `judge_concurrency`. This is
    only safe because the judge wrapper -- and therefore its rate limiter -- is
    shared via `get_judge()`; with a limiter per question, concurrency would
    multiply the request rate instead of packing it into the same budget.

    `judge=False` generates and caches only -- useful for paying the expensive
    phase once, in the background, before iterating on the judge.
    """
    settings = get_settings()
    fetch = top_k or settings.retrieval_top_k
    gen_config = {
        "mode": mode,
        "top_k": fetch,
        "multi_query": multi_query,
        # BOTH models, and `answer_model` is the one that was missing.
        #
        # The answer text comes from `answer_model` in both modes -- `draft`
        # and baseline `answer_question` each call get_llm(answer_model). The
        # key recorded only `llm_model`, so swapping the answer model left the
        # key unchanged and the cache happily served answers written by the
        # previous one. That is the exact failure this key's docstring claims
        # to prevent, and it is invisible: the numbers stay plausible.
        #
        # `llm_model` still belongs here too -- it plans, critiques and rewrites
        # in agent mode, which changes what evidence the answer is built from.
        "answer_model": settings.answer_model,
        "model": settings.llm_model,
        # THE RETRIEVAL PIPELINE, because every one of these changes which
        # passages the answer is built from.
        #
        # Added after the same omission bit twice. The key started with only
        # `model`, missed `answer_model` -- the model that actually writes the
        # answer -- and would have missed all of these, so comparing "with
        # reranking" against "without" would have silently scored one cached
        # set of answers against itself. An evaluation cache that ignores the
        # thing under test produces confident, meaningless numbers.
        "hybrid": settings.hybrid_search,
        "rerank": settings.rerank,
        "floor": settings.retrieval_score_floor,
        "parents": settings.parent_retrieval,
    }

    await asyncio.to_thread(CACHE_DIR.mkdir, parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[Tier2QuestionResult] = []
    n_generated = n_cached = 0

    # --- phase 1: answers (sequential, cached) ------------------------------
    answers: list[tuple[GoldenQuestion, GeneratedAnswer, bool]] = []
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

        answers.append((question, generated, came_from_cache))

    # --- phase 2: judging (concurrent, bounded) -----------------------------
    gate = asyncio.Semaphore(max(1, judge_concurrency))

    async def score_one(
        question: GoldenQuestion, generated: GeneratedAnswer
    ) -> JudgeScores:
        if not judge:
            return JudgeScores.failed("judging skipped")
        if generated.error or not generated.answer:
            return JudgeScores.failed(generated.error or "empty answer")
        async with gate:
            return await judge_answer(
                question=question.question,
                answer=generated.answer,
                contexts=generated.contexts,
                reference=question.expected_answer,
                metrics=metrics,
            )

    # gather preserves ORDER regardless of completion order, so rows still line
    # up with the golden set -- which matters because the report is diffed
    # between runs. judge_answer never raises, so no return_exceptions here.
    scored = await asyncio.gather(
        *(score_one(q, g) for q, g, _ in answers)
    )

    for (question, generated, came_from_cache), scores in zip(
        answers, scored, strict=True
    ):
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
            "judge_concurrency": judge_concurrency,
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

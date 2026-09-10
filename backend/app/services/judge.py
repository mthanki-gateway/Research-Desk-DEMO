"""RAGAS judge for generation quality — evaluation Tier 2.

Tier 1 (`eval_runner.py`) measures RETRIEVAL: did the right chunks come back.
This measures GENERATION: given those chunks, was the answer any good.

RAGAS rather than hand-written judge prompts, for one reason above the others:
**its definitions are the ones everyone else reports against.** A hand-rolled
`faithfulness = 0.82` means whatever the prompt happens to do; RAGAS's 0.82 is
comparable to published numbers and to other teams. Faithfulness in particular
is not one prompt but a pipeline -- decompose the answer into atomic claims,
then verify each against the context -- and getting the decomposition right is
where the fiddly work lives.

The judge is a DIFFERENT model from the one under test
(`settings.judge_model`, default gemini-3.5-flash-lite):

* models show a documented self-preference bias when grading their own output;
* the model under test must not grade itself. The ANSWER comes from
  `settings.answer_model` (gemini-3.6-flash); this judge is a different model,
  and that separation is the invariant -- if the two are ever pointed at the
  same id, Tier 2 measures self-preference rather than faithfulness;
* the quota shapes suit the split. Judging sends the answer plus every
  retrieved chunk, which is large, and Flash Lite allows 250K tokens/minute
  against Gemma's 16K.

RAGAS lives in the optional `eval` extra, so every import here is
function-local. A module-level import would make this file -- and the whole
evaluation package with it -- unimportable on a default install.

The DEV image installs the extra (`uv sync --group dev --extra eval`, see
backend/Dockerfile); prod does not, and cannot judge anything. That is correct:
evaluation is a development activity, ragas pulls pandas/datasets/pyarrow, and
serving a request never needs any of it. Outside Docker:

    uv sync --extra eval
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import structlog

from app.config import get_settings

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class JudgeScores:
    """The four generation-side metrics for one answer.

    Every field is None-able. A judge call can fail, and a missing score must
    stay distinguishable from a zero -- aggregation skips None rather than
    averaging it in, for the same reason recall skips unanswerable questions.
    """

    # Are the answer's claims supported by the retrieved context.
    # THE hallucination metric.
    faithfulness: float | None
    # Does the answer address the question actually asked.
    answer_relevancy: float | None
    # Were the retrieved chunks relevant, and ranked well. Needs a reference.
    context_precision: float | None
    # Did retrieval supply everything the reference answer needs. Needs a
    # reference. This measures RETRIEVAL, not the answer.
    context_recall: float | None

    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def failed(cls, why: str) -> JudgeScores:
        return cls(
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
            error=why,
        )


def ragas_available() -> tuple[bool, str]:
    """Is the optional `eval` extra installed?

    Returned rather than raised so the CLI can print something actionable
    instead of a traceback -- RAGAS is deliberately not in the default
    dependency set, so its absence is a normal state, not an error.
    """
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        return False, f"ragas is not installed ({exc}). Run: uv sync --extra eval"
    return True, ""


def _build_judge():
    """Wrap the judge model in the LangChain interface RAGAS expects.

    RAGAS takes its LLM and embeddings through LangChain wrappers.
    `langchain-google-genai` was already a declared dependency in this project
    and never imported; this is the first thing that actually uses it.

    Call `get_judge()` rather than this -- the wrapper must be shared across
    questions or its rate limiter is meaningless. See the note there.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()

    # RAGAS calls the model through LangChain, NOT through llm.py -- so this
    # project's own RateLimiter never sees these requests. Without a limiter
    # here, RAGAS issues its calls as fast as it can, hits Flash Lite's 15
    # requests/minute ceiling, collects 429s, and LangChain retries with
    # exponential backoff. Measured: two questions took over ten minutes,
    # almost all of it sleeping in backoff.
    #
    # RAGAS is call-hungry per answer -- faithfulness decomposes then verifies,
    # and context precision costs ONE CALL PER CONTEXT CHUNK -- so roughly
    # 10-15 calls per question. Pacing at the quota is both faster overall and
    # predictable, because backoff is strictly worse than not being throttled.
    rpm = settings.judge_requests_per_minute
    limiter = InMemoryRateLimiter(
        requests_per_second=rpm / 60,
        check_every_n_seconds=0.1,
        # A small burst absorbs RAGAS's tendency to fire a metric's calls
        # together without exceeding the per-minute budget.
        max_bucket_size=max(1, rpm // 3),
    )
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.judge_model.removeprefix("models/"),
            google_api_key=settings.google_api_key,
            # Requested for reproducibility -- grading should not vary between
            # runs, or a real regression is indistinguishable from variance.
            #
            # NOTE: gemini-3.5-flash-lite IGNORES it. The provider warns
            # "uses fixed sampling defaults; the sampling parameter(s)
            # temperature will be ignored". So Tier 2 scores carry some
            # run-to-run noise that cannot be turned off, and small differences
            # between runs should not be read as signal.
            temperature=0.0,
            rate_limiter=limiter,
            # Fail fast rather than retrying into a wall: the limiter above is
            # what should be preventing 429s, so a 429 getting through means
            # the pacing is wrong and hiding it in retries wastes minutes.
            max_retries=1,
        )
    )
    # ResponseRelevancy needs embeddings, not just an LLM: it generates
    # questions FROM the answer and compares them to the original question, so
    # it is a similarity measure rather than a judgement.
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    )
    return llm, embeddings


# Cached, and this is load-bearing rather than an optimisation.
#
# `_build_judge` constructs an InMemoryRateLimiter, whose token bucket is
# PER INSTANCE. Calling it once per question -- which is what `judge_answer`
# used to do -- gave every question a fresh, full bucket, so a 20-question run
# created 20 independent limiters and paced at 20x the intended rate. The
# limiter existed, was configured correctly, and throttled nothing across the
# run: exactly the trap `get_llm()` in llm.py documents for its own clients.
#
# Keyed on the settings that shape it, so changing the judge model or its quota
# in a test or a REPL rebuilds rather than silently reusing the old one.
_judge_cache: dict[tuple[str, int], tuple] = {}


def get_judge():
    """The shared judge wrapper: (llm, embeddings). Built once per config."""
    settings = get_settings()
    key = (settings.judge_model, settings.judge_requests_per_minute)
    if key not in _judge_cache:
        _judge_cache[key] = _build_judge()
        log.info(
            "judge_ready", model=settings.judge_model,
            rpm=settings.judge_requests_per_minute,
        )
    return _judge_cache[key]


def reset_judge_cache() -> None:
    """Drop the cached wrapper. For tests, which must not share a limiter."""
    _judge_cache.clear()


REFERENCE_METRICS = ("context_precision", "context_recall")


def reference_gap(metrics: tuple[str, ...], reference: str) -> str | None:
    """Why the reference-based metrics cannot be scored, or None if they can.

    Pure, and extracted for that reason: it is a precondition on the INPUTS,
    with no dependency on ragas being installed, so it is testable on a default
    install where every other path through `judge_answer` stops at the import.
    """
    if not set(REFERENCE_METRICS) & set(metrics):
        return None  # not asked for; nothing to explain
    if reference.strip():
        return None
    return "context_precision/recall: no reference answer"


# Metrics needing no reference answer. Faithfulness IS the hallucination
# metric, and relevancy catches the faithful-but-useless answer, so this pair
# is what `--cheap-metrics` falls back to when a full run is too slow.
CHEAP_METRICS = ("faithfulness", "answer_relevancy")
# The DEFAULT, and the full RAGAS core quartet. The two reference-based metrics
# were opt-in behind --all-metrics while ragas was not installed at all, which
# meant the advertised "judged by RAGAS" was measuring half the suite at best
# and nothing at worst. Every golden question carries an `expected_answer`, so
# there is a reference for all of them and no reason to default to half.
ALL_METRICS = (*CHEAP_METRICS, "context_precision", "context_recall")


async def judge_answer(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str = "",
    metrics: tuple[str, ...] = ALL_METRICS,
) -> JudgeScores:
    """Score one answer with RAGAS. Never raises.

    A judge failure must not end a 20-question run, so every error is captured
    into `JudgeScores.error` and the remaining questions still get scored.

    Context precision and recall are skipped when the golden set has no
    reference answer for the question -- both are reference-based, and RAGAS
    cannot compute them without one.
    """
    ok, why = ragas_available()
    if not ok:
        return JudgeScores.failed(why)

    try:
        # DEPRECATION, known and deliberate. The lock pins ragas 0.4.3, where
        # every name below resolves through a `__getattr__` shim that warns
        # "Importing X from 'ragas.metrics' is deprecated and will be removed
        # in v1.0. Please use 'ragas.metrics.collections' instead." The same
        # applies to LangchainLLMWrapper in _build_judge, which points at
        # `llm_factory`.
        #
        # NOT migrated yet, for one concrete reason: 0.4's replacement path is
        # built around an OpenAI client, and this judge is Gemini behind
        # LangChain wrappers. Moving to it without being able to run the suite
        # would trade a working deprecated API for an unverified one.
        #
        # What this needs is an upper bound (`ragas>=0.2,<1.0`) so a future
        # relock cannot silently pull v1.0 and delete the judge. That is a
        # pyproject change and therefore a uv.lock regeneration -- see the
        # Dockerfile for why those two must move together.
        from ragas import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        llm, embeddings = get_judge()
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ragas_setup_failed", error=str(exc))
        return JudgeScores.failed(f"{type(exc).__name__}: {exc}")

    async def score(name: str, metric) -> float | None:
        """Score one metric in isolation.

        Each metric gets its own try/except because they fail independently and
        for different reasons. A single wrapping try meant one metric raising
        discarded the other three -- measured: ResponseRelevancy failed on
        multi-candidate support and took faithfulness down with it, reporting
        "0 scored" when faithfulness had been computed fine.
        """
        try:
            value = await metric.single_turn_ascore(sample)
            return None if value is None else float(value)
        except Exception as exc:  # noqa: BLE001
            log.warning("ragas_metric_failed", metric=name, error=str(exc)[:200])
            errors.append(f"{name}: {type(exc).__name__}")
            return None

    errors: list[str] = []
    faithfulness = relevancy = precision = recall = None

    if "faithfulness" in metrics:
        faithfulness = await score("faithfulness", Faithfulness(llm=llm))

    if "answer_relevancy" in metrics:
        # strictness=1 generates ONE question from the answer instead of the
        # default 3. The default asks the provider for multiple candidates in a
        # single call, and gemini-3.5-flash-lite rejects that outright:
        #   400 INVALID_ARGUMENT: Multiple candidates is not enabled for this model
        # One sample is noisier, which is a fair trade for the metric existing.
        relevancy = await score(
            "answer_relevancy",
            ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        )

    # Both reference-based metrics need a ground-truth answer, and RAGAS cannot
    # compute them without one.
    #
    # They are also the EXPENSIVE pair. LLMContextPrecisionWithReference makes
    # roughly one call per context chunk, so its cost scales with top_k.
    # Measured on a 15 requests/minute judge, sequentially: all four metrics
    # took ~9 minutes for a single question. `run_tier2` now issues questions
    # concurrently against this shared limiter, which is what makes the full
    # quartet a practical default rather than an overnight job.
    # Recorded, not silent. Two Nones with no explanation is the same ambiguity
    # JudgeScores exists to avoid -- a reader cannot tell a judge failure from a
    # question that had no ground truth to compare against.
    gap = reference_gap(metrics, reference)
    if gap:
        errors.append(gap)
    elif reference.strip():
        if "context_precision" in metrics:
            precision = await score(
                "context_precision", LLMContextPrecisionWithReference(llm=llm)
            )
        if "context_recall" in metrics:
            recall = await score("context_recall", LLMContextRecall(llm=llm))

    return JudgeScores(
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        context_precision=precision,
        context_recall=recall,
        error="; ".join(errors) or None,
    )

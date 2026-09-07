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


# Metrics that need no reference answer, and are the cheapest per question.
# The default set, because they are also the two that matter most:
# faithfulness IS the hallucination metric, and relevancy catches the
# faithful-but-useless answer.
CHEAP_METRICS = ("faithfulness", "answer_relevancy")
ALL_METRICS = (*CHEAP_METRICS, "context_precision", "context_recall")


async def judge_answer(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str = "",
    metrics: tuple[str, ...] = CHEAP_METRICS,
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
        from ragas import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        llm, embeddings = _build_judge()
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
    # roughly one call per context chunk, so its cost scales with top_k -- which
    # is why they are opt-in. Measured on a 15 requests/minute judge: all four
    # metrics took ~9 minutes for a single question, so a 20-question suite is
    # a multi-hour job rather than something to run between edits.
    if reference.strip():
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

"""Run the golden evaluation set from the command line.

    # Tier 1, dense only — the default. ~10s, 20 embedding calls, no LLM.
    docker compose exec api python -m app.scripts.evaluate --owner <uuid>

    # With multi-query expansion (adds one Gemma call per question)
    docker compose exec api python -m app.scripts.evaluate --owner <uuid> --multi-query

    # Wider candidate pool, to see whether a reranker would help
    docker compose exec api python -m app.scripts.evaluate --owner <uuid> --top-k 30

    # Only some questions
    docker compose exec api python -m app.scripts.evaluate --tag bm25
    docker compose exec api python -m app.scripts.evaluate --id rollback-threshold

    # Machine-readable, for diffing two runs
    docker compose exec api python -m app.scripts.evaluate --json > before.json

**`--owner` matters.** With auth enabled, retrieval filters on `owner_id`, so
omitting it evaluates against unowned documents only — which looks exactly like
retrieval failing. Use the Supabase user id that owns the fixtures.

Reading the output: compare **recall at a small k against recall at a large k.**
If recall@10 is high while recall@3 is low, retrieval is finding the right
chunks and RANKING is burying them — a reranker would help. If both are low,
retrieval itself is failing and a reranker would change nothing; fix chunking or
add hybrid search instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import structlog

from app.services.eval_runner import DEFAULT_K_VALUES, run_tier1
from app.services.golden import load_golden_set
from app.services.judge import ALL_METRICS, CHEAP_METRICS, ragas_available
from app.services.tier2 import run_tier2


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  --  "


def _print_report(report) -> None:
    cfg = report.config
    print()
    print("=" * 72)
    print(
        f"TIER 1 — retrieval only    "
        f"top_k={cfg['top_k']}  multi_query={cfg['multi_query']}"
    )
    print(
        f"{cfg['n_questions']} questions · {report.n_embedding_calls} embedding calls · "
        f"{report.elapsed_seconds:.1f}s"
    )
    if cfg.get("filters"):
        bits = ", ".join(f"{k}={'|'.join(v)}" for k, v in cfg["filters"].items())
        print()
        print(f"!! SUBSET RUN — filtered by {bits}")
        print("!! These metrics cover a subset. Do not compare them to a full-suite run.")
    print("=" * 72)
    print()
    print(f"{'k':>4}  {'recall':>8} {'prec':>8} {'hit':>8} {'mrr':>8} {'map':>8} {'ndcg':>8}")
    print(f"{'':>4}  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for k, agg in report.aggregates.items():
        print(
            f"{k:>4}  {_fmt(agg['recall']):>8} {_fmt(agg['precision']):>8} "
            f"{_fmt(agg['hit_rate']):>8} {_fmt(agg['mrr']):>8} "
            f"{_fmt(agg['map']):>8} {_fmt(agg['ndcg']):>8}"
        )

    largest = str(max(cfg["k_values"]))
    scored = report.aggregates[largest]["n_scored"]
    print()
    print(
        f"scored {scored} of {cfg['n_questions']} questions "
        f"({cfg['n_questions'] - scored} unanswerable, excluded from recall)"
    )

    # Failures, ordered worst first. `at_rank` is the diagnosis: a fact found at
    # rank 12 with top_k=5 is a ranking problem, not a retrieval one.
    print()
    print("-" * 72)
    print("FAILURES  (facts not found within k=5)")
    print("-" * 72)
    any_failure = False
    for q in report.questions:
        if not q.answerable:
            continue
        recall = q.scores["5"]["recall"]
        if recall is not None and recall >= 1.0:
            continue
        any_failure = True
        found = ", ".join(f"spec{i}@rank{r}" for i, r in sorted(q.satisfied_at.items()))
        print(f"  {q.id}")
        print(f"      tags     {', '.join(q.tags) or '-'}")
        print(f"      recall@5 {recall:.2f}   facts {q.specs_satisfied}/{q.total_specs}")
        print(f"      found    {found or 'nothing'}")
        if q.unsatisfied_specs:
            print(f"      missing  spec indices {q.unsatisfied_specs} (not in top {cfg['top_k']})")
    if not any_failure:
        print("  none")
    print()


def _print_tier2(report) -> None:
    cfg = report.config
    agg = report.aggregates
    print()
    print("=" * 72)
    print(f"TIER 2 — generation, judged by RAGAS    mode={cfg['mode']} top_k={cfg['top_k']}")
    # The ANSWER model is what wrote the text being judged; the workhorse only
    # plans and critiques. Printing the workhorse under "answering" made the
    # report look like the judge grading its own output, because the workhorse
    # and the judge share a model id -- a self-preference alarm on a run that
    # was not self-grading at all.
    print(f"answering:  {cfg.get('answer_model', '?')}")
    print(f"planning:   {cfg['model']}")
    print(f"judge:      {cfg['judge_model']}")
    if cfg.get("answer_model") == cfg.get("judge_model"):
        print()
        print(
            "!! SELF-GRADING — the answering and judging models are the same id. "
            "These scores measure self-preference, not faithfulness."
        )
    print(
        f"{cfg['n_questions']} questions · {report.n_generated} generated · "
        f"{report.n_from_cache} from cache · {report.elapsed_seconds:.1f}s"
    )
    if cfg.get("filters"):
        bits = ", ".join(f"{k}={'|'.join(v)}" for k, v in cfg["filters"].items())
        print()
        print(f"!! SUBSET RUN — filtered by {bits}. Not comparable to a full-suite run.")
    # Same warning, for metric coverage rather than question coverage. A
    # --cheap-metrics run reports two of the four, and its table looks exactly
    # like a full one except for two dashes -- easy to mistake for "those
    # metrics failed" and easier still to paste into a comparison.
    if set(cfg.get("metrics", [])) != set(ALL_METRICS):
        print()
        print(
            f"!! PARTIAL METRICS — judged {', '.join(cfg['metrics'])}. "
            "Not comparable to a full-quartet run."
        )
    print("=" * 72)
    print()

    rows = [
        ("Faithfulness", "faithfulness", "claims supported by the retrieved context"),
        ("Answer relevancy", "answer_relevancy", "does it address the question asked"),
        ("Context precision", "context_precision", "were the retrieved chunks needed"),
        ("Context recall", "context_recall", "did retrieval get everything needed"),
    ]
    print(f"{'metric':20s} {'score':>7}  {'n':>4}   what it measures")
    print(f"{'-' * 20} {'-' * 7}  {'-' * 4}   {'-' * 40}")
    for label, key, blurb in rows:
        value = agg.get(key)
        shown = f"{value:.3f}" if value is not None else "  --  "
        print(f"{label:20s} {shown:>7}  {agg.get(f'{key}_n', 0):>4}   {blurb}")

    if agg.get("n_judge_errors"):
        print()
        print(f"!! {agg['n_judge_errors']} judge call(s) failed — see --json for details")

    # Worst faithfulness first: an unsupported claim is a hallucination, which
    # matters more than a merely irrelevant answer.
    scored = [q for q in report.questions if q.scores.get("faithfulness") is not None]
    scored.sort(key=lambda q: q.scores["faithfulness"])
    print()
    print("-" * 72)
    print("LOWEST FAITHFULNESS")
    print("-" * 72)
    for q in scored[:5]:
        s = q.scores
        print(f"  {q.question_id}   faithfulness {s['faithfulness']:.2f}")
        print(f"      {q.question}")
        print(f"      answer: {q.answer[:110]}")
    print()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--owner", default=None, help="owner_id the fixtures belong to")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="candidates to retrieve (default: max of --k). Raise to test reranking headroom",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help=f"k values to report (default: {' '.join(map(str, DEFAULT_K_VALUES))})",
    )
    parser.add_argument("--multi-query", action="store_true", help="enable query expansion")
    parser.add_argument("--tag", action="append", default=[], help="only questions with this tag")
    parser.add_argument("--id", action="append", default=[], help="only these question ids")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=1,
        help="1 = retrieval metrics (fast, no LLM). 2 = generation metrics via RAGAS",
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "baseline"],
        default="agent",
        help="tier 2 only: run the LangGraph agent, or plain single-pass RAG",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="tier 2 only: regenerate answers instead of reusing the cache",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="tier 2 only: produce and cache answers without judging them",
    )
    parser.add_argument(
        "--cheap-metrics",
        action="store_true",
        help=(
            "tier 2 only: judge faithfulness and answer relevancy ONLY, skipping "
            "the two reference-based metrics. The full quartet is the default; "
            "this trades coverage for time when iterating"
        ),
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=4,
        help=(
            "tier 2 only: how many questions to judge at once. The judge's rate "
            "limiter is shared, so this packs requests into the same per-minute "
            "budget rather than exceeding it; raise it if the run looks "
            "latency-bound, lower it to 1 to serialise"
        ),
    )
    args = parser.parse_args()

    if args.json:
        # Logs to STDERR, so `--json > report.json` yields a parseable file.
        #
        # structlog's default writes to stdout, so the report came out
        # interleaved with progress lines and every attempt to parse it failed
        # at "Extra data: line 1 column 5". The docstring below already promised
        # "stdout only, so `> before.json` captures a clean document" -- it was
        # true of the print and untrue of the process.
        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)
        )

    questions = load_golden_set()
    if args.tag:
        wanted = set(args.tag)
        questions = [q for q in questions if wanted & set(q.tags)]
    if args.id:
        wanted_ids = set(args.id)
        questions = [q for q in questions if q.id in wanted_ids]

    if not questions:
        print("no questions matched the filters", file=sys.stderr)
        return 1

    filters = {
        **({"tags": args.tag} if args.tag else {}),
        **({"ids": args.id} if args.id else {}),
    }

    if args.tier == 2:
        ok, why = ragas_available()
        if not ok and not args.generate_only:
            print(f"Tier 2 needs RAGAS: {why}", file=sys.stderr)
            return 1
        report2 = await run_tier2(
            questions,
            mode=args.mode,
            top_k=args.top_k,
            multi_query=args.multi_query,
            owner_id=args.owner,
            use_cache=not args.no_cache,
            judge=not args.generate_only,
            metrics=CHEAP_METRICS if args.cheap_metrics else ALL_METRICS,
            judge_concurrency=args.judge_concurrency,
            filters=filters,
        )
        if args.json:
            print(json.dumps(report2.to_dict(), indent=2))
        else:
            _print_tier2(report2)
        return 0

    report = await run_tier1(
        questions,
        top_k=args.top_k,
        k_values=tuple(sorted(args.k)),
        multi_query=args.multi_query,
        owner_id=args.owner,
        filters=filters,
    )

    if args.json:
        # stdout only, so `> before.json` captures a clean document.
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

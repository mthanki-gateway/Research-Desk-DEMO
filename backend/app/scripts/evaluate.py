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

from app.services.eval_runner import DEFAULT_K_VALUES, run_tier1
from app.services.golden import load_golden_set


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
    args = parser.parse_args()

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

    report = await run_tier1(
        questions,
        top_k=args.top_k,
        k_values=tuple(sorted(args.k)),
        multi_query=args.multi_query,
        owner_id=args.owner,
        filters={
            **({"tags": args.tag} if args.tag else {}),
            **({"ids": args.id} if args.id else {}),
        },
    )

    if args.json:
        # stdout only, so `> before.json` captures a clean document.
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

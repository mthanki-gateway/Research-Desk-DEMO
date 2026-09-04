"use client";

import { useCallback, useEffect, useState } from "react";
import {
  type EvalCorpus,
  type EvalQuestionResult,
  type EvalReport,
  type GoldenSet,
  getEvalCorpus,
  getGoldenSet,
  runTier1,
} from "@/lib/api";
import { useApp } from "../providers";
import { Button, Chip, LinearProgress, Switch } from "../md";
import { IconSpinner } from "../icons";

/**
 * Tier 1 retrieval benchmark.
 *
 * Runs the golden set through retrieval and reports Recall@k, Precision@k,
 * hit rate, MRR, MAP and NDCG@k. No LLM is involved unless multi-query is on,
 * which is what makes it fast enough to sit behind a button.
 *
 * The number to read first is **recall at a small k against recall at a large
 * k**. If recall@10 is high while recall@3 is low, retrieval is finding the
 * right chunks and RANKING is burying them — a reranker would help. If both
 * are low, retrieval itself is failing and a reranker would change nothing.
 * The panel says this out loud rather than leaving it to be inferred.
 */
export default function Benchmark() {
  const [golden, setGolden] = useState<GoldenSet | null>(null);
  const [corpus, setCorpus] = useState<EvalCorpus | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [topK, setTopK] = useState(10);
  const [multiQuery, setMultiQuery] = useState(false);
  const [tags, setTags] = useState<string[]>([]);

  useEffect(() => {
    void getGoldenSet().then(setGolden).catch(() => setGolden(null));
    void getEvalCorpus().then(setCorpus).catch(() => setCorpus(null));
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      // k values are capped at topK: reporting recall@10 when only 5
      // candidates were retrieved would be identical to recall@5 and imply a
      // measurement that never happened.
      const kValues = [1, 3, 5, 10, 20].filter((k) => k <= topK);
      setReport(await runTier1({ topK, kValues, multiQuery, tags }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Benchmark failed");
      setReport(null);
    } finally {
      setBusy(false);
    }
  }, [topK, multiQuery, tags]);

  const toggleTag = (tag: string) =>
    setTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
    );

  const selected = golden
    ? tags.length === 0
      ? golden.n_questions
      : golden.questions.filter((q) => q.tags.some((t) => tags.includes(t))).length
    : 0;

  return (
    <section className="space-y-5">
      <div>
        <h2 className="md-title-large">Retrieval benchmark</h2>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Runs the golden set through retrieval only — no answer generation, no
          LLM unless multi-query is on. One embedding call per question.
        </p>
      </div>

      {/* What is actually being measured. A low score has two very different
          causes — poor retrieval, or a corpus that was never ingested — and
          showing the corpus makes them distinguishable before anyone draws a
          conclusion. */}
      {corpus && (
        <div
          className="md-body-small rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-surface-container-highest)",
            color: "var(--md-on-surface-variant)",
          }}
        >
          Measuring against{" "}
          <strong>
            {corpus.n_documents} document{corpus.n_documents === 1 ? "" : "s"},{" "}
            {corpus.n_chunks} chunks
          </strong>
          {golden && (
            <>
              {" · "}
              {golden.n_questions} questions ({golden.n_unanswerable} unanswerable)
            </>
          )}
          {corpus.n_chunks > 0 && corpus.n_chunks < 60 && (
            <>
              {" — "}
              <span style={{ color: "var(--md-tertiary)" }}>
                small corpus: top-{topK} is{" "}
                {Math.round((topK / corpus.n_chunks) * 100)}% of everything, so
                differences will be hard to see
              </span>
            </>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <label className="flex items-center gap-2">
          <span className="md-label-large">Retrieve</span>
          <span className="md-segmented">
            {[5, 10, 20, 30].map((k) => (
              <button
                key={k}
                type="button"
                data-active={topK === k}
                onClick={() => setTopK(k)}
                className="tabular-nums"
              >
                {k}
              </button>
            ))}
          </span>
        </label>

        <label className="flex items-center gap-2">
          <span className="md-label-large">Multi-query</span>
          <Switch on={multiQuery} onChange={setMultiQuery} />
        </label>

        <Button onClick={() => void run()} disabled={busy || selected === 0}>
          {busy ? <IconSpinner className="h-4 w-4" /> : null}
          {busy ? "Running" : `Run ${selected} question${selected === 1 ? "" : "s"}`}
        </Button>
      </div>

      {/* Tag filter */}
      {golden && golden.tags.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="md-label-medium mr-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Filter
          </span>
          {golden.tags.map((tag) => (
            <Chip
              key={tag}
              size="sm"
              selected={tags.includes(tag)}
              onClick={() => toggleTag(tag)}
            >
              {tag}
            </Chip>
          ))}
          {tags.length > 0 && (
            <Button variant="text" size="sm" onClick={() => setTags([])}>
              Clear
            </Button>
          )}
        </div>
      )}

      {busy && (
        <div className="space-y-2">
          <LinearProgress />
          <p
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {multiQuery
              ? "Expanding each question, then retrieving — this adds one model call per question."
              : "Embedding each question and searching."}
          </p>
        </div>
      )}

      {error && (
        <p
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      {report && !busy && <Results report={report} />}
    </section>
  );
}

function fmt(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(3);
}

/**
 * What each metric means, in the order the table shows them.
 *
 * Kept next to the table rather than in documentation because the numbers are
 * meaningless without it — and two of them are actively misleading if you do
 * not know their shape. Precision necessarily FALLS as k grows, and recall
 * necessarily RISES, so neither direction is a quality signal on its own.
 */
const METRICS: { key: string; label: string; blurb: string }[] = [
  {
    key: "recall",
    label: "Recall",
    blurb:
      "Of all the facts a question needed, what fraction appeared in the top k. THE metric that matters most — a chunk that is not retrieved cannot be cited, reranked or reasoned over. Rises as k grows, so always read it with its k.",
  },
  {
    key: "precision",
    label: "Precision",
    blurb:
      "Of the k chunks returned, what fraction were relevant. Necessarily FALLS as k grows (1 relevant chunk in 5 slots is 0.2 by arithmetic), so a drop is not a regression. Matters for token cost, not correctness.",
  },
  {
    key: "hit_rate",
    label: "Hit rate",
    blurb:
      "The fraction of questions where AT LEAST ONE needed fact was retrieved. Blunt and binary — 0.882 means 88% of questions got something usable, regardless of whether they got everything.",
  },
  {
    key: "mrr",
    label: "MRR",
    blurb:
      "Mean Reciprocal Rank: the average of 1/(rank of the first relevant result). Rewards putting a good result first. A weak fit for RAG, since all k chunks go into the prompt anyway — it is a search metric, reported because interviews ask for it.",
  },
  {
    key: "map",
    label: "MAP",
    blurb:
      "Mean Average Precision: averages precision at each rank where a relevant chunk appears. Rank-aware, so it rewards clustering the good results early rather than scattering them.",
  },
  {
    key: "ndcg",
    label: "NDCG",
    blurb:
      "Normalised Discounted Cumulative Gain. Like MAP but with a logarithmic position discount, normalised so questions needing different numbers of facts are comparable. Its real strength is graded relevance, which this suite does not use — labels here are binary.",
  },
];

function Results({ report }: { report: EvalReport }) {
  const ks = report.config.k_values.map(String);
  const smallest = report.aggregates[ks[0]];
  const largest = report.aggregates[ks[ks.length - 1]];

  // The diagnosis. Recall climbing sharply with k means the chunks are being
  // found but ranked too low, which is the specific condition a reranker fixes.
  const rSmall = smallest?.recall ?? null;
  const rLarge = largest?.recall ?? null;
  const rankingProblem =
    rSmall !== null && rLarge !== null && rLarge - rSmall > 0.1;
  const retrievalProblem = rLarge !== null && rLarge < 0.8;

  const maxK = ks[ks.length - 1];
  const answerable = report.questions.filter((q) => q.answerable);
  const failures = answerable.filter((q) => {
    const s = q.scores[maxK];
    return s && s.recall !== null && s.recall < 1;
  });
  const passed = answerable.filter((q) => {
    const s = q.scores[maxK];
    return s && s.recall !== null && s.recall >= 1;
  });
  const unanswerable = report.questions.filter((q) => !q.answerable);

  const filteredTags = report.config.filters?.tags ?? [];

  return (
    <div className="space-y-5">
      {/* A subset run's metrics are NOT the system's metrics — they cover
          whichever hard questions were selected. Saying so here is the only
          thing stopping 0.500 on two bm25 questions being read as "recall
          dropped from 0.89". */}
      {filteredTags.length > 0 && (
        <p
          className="md-body-small rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-tertiary-container)",
            color: "var(--md-on-tertiary-container)",
          }}
        >
          <strong>Subset run</strong> — {report.config.n_questions} question
          {report.config.n_questions === 1 ? "" : "s"} matching{" "}
          {filteredTags.map((t) => `“${t}”`).join(", ")}. These numbers describe
          that subset, not the whole suite — don’t compare them to an unfiltered
          run.
        </p>
      )}

      {/* Metrics */}
      <div className="md-card md-card-outlined overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr
              className="md-label-medium"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              <th className="px-4 py-2.5">k</th>
              {METRICS.map((m) => (
                <th key={m.key} className="px-4 py-2.5">
                  {/* `title` gives a native tooltip with no library and no
                      focus-trap concerns; the full glossary sits below for
                      anyone who does not hover. */}
                  <span title={m.blurb} className="cursor-help border-b border-dotted">
                    {m.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="md-body-medium tabular-nums">
            {ks.map((k) => {
              const a = report.aggregates[k];
              return (
                <tr key={k} className="border-t">
                  <th className="px-4 py-2.5 font-semibold">{k}</th>
                  {METRICS.map((m) => (
                    <td
                      key={m.key}
                      className={`px-4 py-2.5 ${m.key === "recall" ? "font-semibold" : ""}`}
                    >
                      {fmt(a?.[m.key as keyof typeof a] as number | null)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <details className="md-card md-card-outlined px-4 py-3">
        <summary className="md-label-large cursor-pointer">
          What these metrics mean
        </summary>
        <dl className="mt-3 space-y-3">
          {METRICS.map((m) => (
            <div key={m.key}>
              <dt className="md-body-medium font-semibold">{m.label}</dt>
              <dd
                className="md-body-small"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {m.blurb}
              </dd>
            </div>
          ))}
          <div>
            <dt className="md-body-medium font-semibold">
              Reading recall across k
            </dt>
            <dd
              className="md-body-small"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Recall <strong>rising</strong> steeply with k means the right
              chunks are in the candidate pool but ranked too low — a reranker
              reorders the pool, so it fixes this. Recall <strong>flat</strong>{" "}
              and low means the chunk is not in the pool at any depth, and a
              reranker cannot reorder what was never retrieved — that needs
              better chunking or hybrid (BM25) search.
            </dd>
          </div>
        </dl>
      </details>

      <p
        className="md-body-small"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {largest?.n_scored} of {report.config.n_questions} questions scored
        {report.config.n_questions - (largest?.n_scored ?? 0) > 0 && (
          <>
            {" "}
            ({report.config.n_questions - (largest?.n_scored ?? 0)} unanswerable,
            excluded from recall rather than counted as zero)
          </>
        )}
        {" · "}
        {report.n_embedding_calls} embedding calls ·{" "}
        {report.elapsed_seconds.toFixed(1)}s
      </p>

      {/* Diagnosis, stated rather than implied. */}
      {(rankingProblem || retrievalProblem) && (
        <div
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-secondary-container)",
            color: "var(--md-on-secondary-container)",
          }}
        >
          {rankingProblem && (
            <p>
              <strong>Ranking problem.</strong> Recall rises from{" "}
              {fmt(rSmall)} at k={ks[0]} to {fmt(rLarge)} at k=
              {ks[ks.length - 1]} — the right chunks are being retrieved but
              ranked too low. A cross-encoder reranker would help here.
            </p>
          )}
          {retrievalProblem && (
            <p className={rankingProblem ? "mt-2" : ""}>
              <strong>Retrieval problem.</strong> Recall is still{" "}
              {fmt(rLarge)} at k={ks[ks.length - 1]}, so some facts are not
              being found at all. Reranking cannot fix this — it needs better
              chunking or hybrid (BM25) search.
            </p>
          )}
        </div>
      )}

      {/* Every question, in a consistent frame, ordered by how much attention
          it needs: failures first, then the honesty checks, then the passes
          last. Showing only failures hid half the picture — you could not tell
          whether a pass was comfortable or a near-miss (fact found at rank 5
          of 5 passes, but barely), nor sanity-check the labels themselves. */}
      <div className="space-y-4">
        <QuestionGroup
          title="Missing facts"
          tone="error"
          questions={failures}
          maxK={maxK}
          topK={report.config.top_k}
        />
        <QuestionGroup
          title="Unanswerable — should retrieve nothing useful"
          tone="neutral"
          questions={unanswerable}
          maxK={maxK}
          topK={report.config.top_k}
        />
        <QuestionGroup
          title="All facts found"
          tone="ok"
          questions={passed}
          maxK={maxK}
          topK={report.config.top_k}
        />
      </div>
    </div>
  );
}

const TONES = {
  error: "var(--md-error)",
  ok: "var(--md-primary)",
  neutral: "var(--md-on-surface-variant)",
} as const;

function QuestionGroup({
  title,
  tone,
  questions,
  maxK,
  topK,
}: {
  title: string;
  tone: keyof typeof TONES;
  questions: EvalQuestionResult[];
  maxK: string;
  topK: number;
}) {
  if (questions.length === 0) return null;
  return (
    <div>
      <h3 className="md-title-medium mb-2">
        {title}{" "}
        <span
          className="md-body-medium tabular-nums"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          ({questions.length})
        </span>
      </h3>
      <div className="space-y-2">
        {questions.map((q) => (
          <QuestionDetail
            key={q.id}
            q={q}
            tone={tone}
            maxK={maxK}
            topK={topK}
          />
        ))}
      </div>
    </div>
  );
}

function QuestionDetail({
  q,
  tone,
  maxK,
  topK,
}: {
  q: EvalQuestionResult;
  tone: keyof typeof TONES;
  maxK: string;
  topK: number;
}) {
  // From context rather than four levels of prop drilling. `showChunk` already
  // owns the chunk cache and loading state.
  const { showChunk } = useApp();
  const onCite = showChunk;
  const s = q.scores[maxK];
  // The deepest rank any needed fact was found at. A pass at rank 5 of 5 is a
  // near-miss that one chunking change away from becoming a failure, and the
  // summary should say so rather than showing a flat "recall 1.000".
  const deepest = Math.max(
    0,
    ...q.expected
      .map((e) => e.found_at_rank)
      .filter((r): r is number => r !== null),
  );
  return (
    <details className="md-card md-card-outlined px-4 py-3">
      <summary className="cursor-pointer">
        <span className="md-body-medium font-medium">{q.question}</span>
        <span
          className="md-body-small ml-2 tabular-nums"
          style={{ color: TONES[tone] }}
        >
          {q.answerable ? (
            <>
              {q.specs_satisfied}/{q.total_specs} of {q.total_specs} expected
              chunk{q.total_specs === 1 ? "" : "s"} · recall {fmt(s?.recall)}
              {q.specs_satisfied === q.total_specs && deepest > 0 && (
                <span style={{ color: "var(--md-on-surface-variant)" }}>
                  {" "}
                  · deepest rank {deepest}
                  {deepest >= topK && " (only just)"}
                </span>
              )}
            </>
          ) : (
            <>0 expected chunks — should find nothing</>
          )}
        </span>
      </summary>

      <div className="mt-3 space-y-3">
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          <code>{q.id}</code> · {q.tags.join(", ") || "no tags"}
        </p>

        {/* WHAT SHOULD HAVE BEEN RETRIEVED. Without this a red row says
            "fact 1 missing" and you cannot tell whether the miss was obvious
            or obscure. */}
        {q.expected.length > 0 && (
          <div>
            <p className="md-label-medium mb-1">Expected</p>
            <ul className="md-body-small space-y-1">
              {q.expected.map((e) => {
                const broken = e.matching_chunk_ids.length === 0;
                const target = e.matching_chunk_ids[0];
                const body = (
                  <>
                    <span
                      style={{
                        color: broken
                          ? "var(--md-tertiary)"
                          : e.found_at_rank !== null
                            ? "var(--md-primary)"
                            : "var(--md-error)",
                      }}
                    >
                      {broken ? "!" : e.found_at_rank !== null ? "✓" : "✗"}
                    </span>
                    <span className="min-w-0 flex-1">
                      <strong>{e.file}</strong>
                      {e.must_contain.length > 0 && (
                        <>
                          {" "}
                          containing {e.must_contain.map((c) => `“${c}”`).join(" + ")}
                        </>
                      )}
                      {broken ? (
                        /* The label matches no chunk at all, so the question
                           CANNOT pass. Distinguishing this from a retrieval
                           miss matters: otherwise you tune retrieval against
                           an impossible target. */
                        <span style={{ color: "var(--md-tertiary)" }}>
                          {" "}
                          — no chunk in the corpus matches this label. Either the
                          fixture is not ingested, or the label is wrong.
                        </span>
                      ) : e.found_at_rank !== null ? (
                        <span style={{ color: "var(--md-on-surface-variant)" }}>
                          {" "}
                          — found at rank {e.found_at_rank}
                          {e.found_at_rank > topK && " (beyond top-k)"}
                        </span>
                      ) : (
                        <span style={{ color: "var(--md-error)" }}>
                          {" "}
                          — not retrieved in top {topK}
                          {e.matching_chunk_ids.length > 1 &&
                            ` · ${e.matching_chunk_ids.length} chunks would satisfy it`}
                        </span>
                      )}
                    </span>
                  </>
                );

                // Clickable whenever the label resolves — most valuable on a
                // MISS, since reading the chunk that should have won is how
                // you judge whether the miss was reasonable.
                return (
                  <li key={e.index}>
                    {target ? (
                      <button
                        type="button"
                        onClick={() => void onCite(target)}
                        title="Open the chunk this label points at"
                        className="flex w-full items-start gap-2 rounded-[var(--md-shape-xs)] px-1 py-0.5 text-left hover:bg-[color-mix(in_srgb,var(--md-on-surface)_8%,transparent)]"
                      >
                        {body}
                      </button>
                    ) : (
                      <span className="flex items-start gap-2 px-1 py-0.5">{body}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {q.expected_answer && (
          <div>
            <p className="md-label-medium mb-1">
              Reference answer{q.expect_refusal ? " (should refuse)" : ""}
            </p>
            <p
              className="md-body-small"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {q.expected_answer}
            </p>
          </div>
        )}

        {/* WHAT WAS ACTUALLY RETRIEVED, in rank order. */}
        <div>
          <p className="md-label-medium mb-1">
            Retrieved ({q.hits.length}, best first)
          </p>
          <ol className="md-body-small space-y-1">
            {q.hits.map((h) => (
              <li key={h.rank}>
                {/* Clickable: the preview is 140 chars, so the full chunk opens
                    in the same side panel citations use. Reusing showChunk
                    means the chunk cache and loading state come for free. */}
                <button
                  type="button"
                  onClick={() => void onCite(h.chunk_id)}
                  title={`Open the full ${h.n_chars}-character chunk`}
                  className="flex w-full gap-2 rounded-[var(--md-shape-xs)] px-1 py-0.5 text-left hover:bg-[color-mix(in_srgb,var(--md-on-surface)_8%,transparent)]"
                  style={{
                    color: h.relevant
                      ? "var(--md-on-surface)"
                      : "var(--md-on-surface-variant)",
                  }}
                >
                  <span className="w-4 shrink-0 text-right tabular-nums opacity-60">
                    {h.rank}
                  </span>
                  <span
                    className="w-3 shrink-0"
                    style={{ color: h.relevant ? "var(--md-primary)" : undefined }}
                  >
                    {h.relevant ? "✓" : "·"}
                  </span>
                  <span className="min-w-0 flex-1 truncate">
                    <strong>{h.filename}</strong>
                    {h.heading ? ` › ${h.heading}` : ""} — {h.preview}
                  </span>
                  <span
                    className="shrink-0 tabular-nums opacity-60"
                    title="cosine similarity — only comparable within this one query"
                  >
                    {h.score.toFixed(3)}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </details>
  );
}

"use client";

import { useEffect, useState } from "react";
import {
  type AskResult,
  type ResearchResult,
  type SearchHit,
  type Stats,
  ask,
  getStats,
  research,
  search,
} from "@/lib/api";
import { useApp } from "../providers";
import { Button, Ripplable, Switch, TextField } from "../md";
import { IconSearch, IconSpinner } from "../icons";
import Benchmark from "./benchmark";

/**
 * Developer surface, kept separate from Chat.
 *
 * These controls are how you tell whether a bad answer came from retrieval or
 * generation, and how you compare the agent against plain RAG. Useful, but not
 * something a user of the product should have to look at.
 */
export default function Lab() {
  const { readyDocuments, showChunk } = useApp();
  const [stats, setStats] = useState<Stats | null>(null);
  // Three states, not two. `stats === null` cannot distinguish "still
  // loading" from "the request failed", and rendering nothing for both meant
  // the badges popped in with no warning that anything was coming — the
  // route-level loading.tsx does not help here, because this fetch starts
  // AFTER the page has mounted.
  const [statsState, setStatsState] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    void getStats()
      .then((s) => {
        setStats(s);
        setStatsState("ready");
      })
      .catch(() => {
        setStats(null);
        setStatsState("error");
      });
  }, []);

  const enabled = readyDocuments.length > 0;

  return (
    <div className="mx-auto max-w-3xl space-y-8 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Lab</h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Single-shot tools for diagnosing retrieval and comparing the agent
          against the plain-RAG baseline.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4" data-stats>
        {statsState === "loading" &&
          /* Same card, same height, skeleton contents — so the grid does not
             reflow when the real numbers land. A spinner here would shift the
             whole page down on arrival. */
          ["Documents", "Chunks", "Vectors", "Embeddings"].map((label) => (
            <StatSkeleton key={label} label={label} />
          ))}

        {statsState === "ready" && stats && (
          <>
            <Stat label="Documents" value={stats.documents} />
            <Stat label="Chunks" value={stats.chunks_in_postgres} />
            <Stat
              label="Vectors"
              value={stats.vectors_in_qdrant}
              warn={stats.chunks_in_postgres !== stats.vectors_in_qdrant}
            />
            <Stat
              label="Embeddings"
              value={`${stats.embedding_dim}d`}
              sub={stats.embedding_provider}
            />
          </>
        )}

        {statsState === "error" && (
          /* Says so, rather than silently showing nothing. On the deployed
             backend the first request after 15 minutes idle can take 50s, and
             an empty area is indistinguishable from a broken page. */
          <p
            className="md-body-small col-span-2 sm:col-span-4"
            style={{ color: "var(--md-tertiary)" }}
          >
            Could not load index stats. The backend may be waking up — reload
            in a moment.
          </p>
        )}
      </div>

      {stats && stats.chunks_in_postgres !== stats.vectors_in_qdrant && (
        <p className="md-body-small" style={{ color: "var(--md-tertiary)" }}>
          Chunk and vector counts disagree — an ingest probably failed partway.
        </p>
      )}

      {!enabled && (
        <p
          className="md-card md-card-outlined md-body-medium px-4 py-8 text-center"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Upload a document in the Library first.
        </p>
      )}

      <AskPanel enabled={enabled} onCite={showChunk} />
      <SearchPanel enabled={enabled} onCite={showChunk} />

      {/* Benchmark last: the single-shot tools above are for diagnosing ONE
          question, this measures the whole suite. Separate file because it is
          substantial enough to own its state. */}
      <div
        className="border-t pt-8"
        style={{ borderColor: "var(--md-outline-variant)" }}
      >
        <Benchmark />
      </div>
    </div>
  );
}

/** Placeholder matching Stat's exact box, so nothing reflows on arrival. */
function StatSkeleton({ label }: { label: string }) {
  return (
    <div className="md-card md-card-filled px-4 py-3" aria-busy="true">
      <p
        className="md-label-medium"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {label}
      </p>
      {/* Height matches md-title-large's line box, so the card keeps its size. */}
      <div className="mt-1.5 md-skeleton h-5 w-12 rounded-[var(--md-shape-xs)]" />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  warn,
}: {
  label: string;
  value: string | number;
  sub?: string;
  warn?: boolean;
}) {
  return (
    <div className="md-card md-card-filled px-4 py-3">
      <p
        className="md-label-medium"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {label}
      </p>
      <p
        className="md-title-large mt-0.5 tabular-nums"
        style={{ color: warn ? "var(--md-error)" : "var(--md-on-surface)" }}
      >
        {value}
      </p>
      {sub && (
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {sub}
        </p>
      )}
    </div>
  );
}

function Stepper({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <span className="md-segmented">
      <Ripplable
        as="button"
        onClick={() => onChange(Math.max(1, value - 1))}
        aria-label="Decrease"
      >
        &minus;
      </Ripplable>
      <button
        type="button"
        data-active="true"
        className="tabular-nums"
        style={{ pointerEvents: "none", minWidth: "2.5rem" }}
      >
        {value}
      </button>
      <Ripplable
        as="button"
        onClick={() => onChange(Math.min(20, value + 1))}
        aria-label="Increase"
      >
        +
      </Ripplable>
    </span>
  );
}

function AskPanel({
  enabled,
  onCite,
}: {
  enabled: boolean;
  onCite: (id: string) => Promise<void>;
}) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [multiQuery, setMultiQuery] = useState(false);
  const [useAgent, setUseAgent] = useState(true);
  const [result, setResult] = useState<AskResult | ResearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const agentResult =
    result && "trace" in result ? (result as ResearchResult) : null;

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const opts = { topK, multiQuery };
      setResult(
        useAgent ? await research(question, opts) : await ask(question, opts),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="md-title-medium">Single-shot ask</h2>
        <div className="md-segmented">
          <Ripplable
            as="button"
            data-active={useAgent}
            onClick={() => setUseAgent(true)}
          >
            Agent
          </Ripplable>
          <Ripplable
            as="button"
            data-active={!useAgent}
            onClick={() => setUseAgent(false)}
          >
            Baseline
          </Ripplable>
        </div>
      </div>

      <p
        className="md-body-small"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        No session, no history. Set top-k to 1 and ask a two-part question — the
        baseline misses a fact the agent finds.
      </p>

      <form onSubmit={run} className="space-y-4">
        <div className="flex items-start gap-3">
          <TextField
            label="Ask a question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={!enabled}
            className="flex-1"
          />
          <Button
            type="submit"
            disabled={!enabled || busy || !question.trim()}
            className="mt-2"
          >
            {busy ? <IconSpinner /> : "Run"}
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <span className="flex items-center gap-3">
            <span
              className="md-label-large"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Top-k
            </span>
            <Stepper value={topK} onChange={setTopK} />
          </span>
          <span className="flex items-center gap-3">
            <span
              className="md-label-large"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Multi-query
            </span>
            <Switch
              on={multiQuery}
              onChange={setMultiQuery}
              aria-label="Multi-query"
              title="Rewrite each query into 3 variations and fuse with RRF. Costs one extra Gemma call per query."
            />
          </span>
        </div>
      </form>

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

      {result && (
        <div className="space-y-4">
          <div className="md-card md-card-elevated p-4">
            <p className="md-body-large whitespace-pre-wrap">{result.answer}</p>
          </div>

          {agentResult && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="md-badge">
                {agentResult.iterations} iteration
                {agentResult.iterations === 1 ? "" : "s"}
              </span>
              <span className="md-badge">
                {agentResult.sub_questions.length} sub-questions
              </span>
              <span
                className={`md-badge ${
                  agentResult.sufficient
                    ? "md-badge-primary"
                    : "md-badge-tertiary"
                }`}
              >
                {agentResult.sufficient ? "Critic satisfied" : "Gaps remain"}
              </span>
            </div>
          )}

          {agentResult && (
            <details className="md-card md-card-outlined p-4">
              <summary
                className="md-label-large cursor-pointer"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Graph path · {agentResult.trace.length} steps
              </summary>
              <ol className="mt-3 space-y-2">
                {agentResult.trace.map((step, i) => (
                  <li
                    key={i}
                    className="md-body-small rounded-[var(--md-shape-md)] p-3"
                    style={{ background: "var(--md-surface-container-high)" }}
                  >
                    <span
                      className="md-label-small rounded-[var(--md-shape-xs)] px-1.5 font-mono"
                      style={{
                        background: "var(--md-primary-container)",
                        color: "var(--md-on-primary-container)",
                      }}
                    >
                      {step.node}
                    </span>
                    {step.sub_questions && (
                      <ul
                        className="mt-2 ml-1 space-y-0.5"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        {step.sub_questions.map((s) => (
                          <li key={s}>— {s}</li>
                        ))}
                      </ul>
                    )}
                    {step.queries && (
                      <ul
                        className="mt-2 ml-1 space-y-0.5"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        {step.queries.map((q) => (
                          <li key={q.query}>
                            — {q.query}{" "}
                            <span className="tabular-nums opacity-70">
                              → {q.n} chunk{q.n === 1 ? "" : "s"}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                    {step.unanswered && step.unanswered.length > 0 && (
                      <p
                        className="mt-2 ml-1"
                        style={{ color: "var(--md-tertiary)" }}
                      >
                        unanswered: {step.unanswered.join("; ")}
                      </p>
                    )}
                    {step.node === "critique" && (
                      <p className="mt-2 ml-1">
                        <span
                          style={{
                            color: step.sufficient
                              ? "var(--md-primary)"
                              : "var(--md-tertiary)",
                          }}
                        >
                          sufficient = {String(step.sufficient)}
                        </span>
                        {step.missing && step.missing.length > 0 && (
                          <span
                            style={{ color: "var(--md-on-surface-variant)" }}
                          >
                            {" "}
                            → retrying: {step.missing.join("; ")}
                          </span>
                        )}
                      </p>
                    )}
                  </li>
                ))}
              </ol>
            </details>
          )}

          <Hits
            hits={result.sources}
            used={result.sources_used}
            onCite={onCite}
          />
        </div>
      )}
    </section>
  );
}

function SearchPanel({
  enabled,
  onCite,
}: {
  enabled: boolean;
  onCite: (id: string) => Promise<void>;
}) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setHits(await search(query, 5));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section
      className="space-y-4 border-t pt-8"
      style={{ borderColor: "var(--md-outline-variant)" }}
    >
      <h2 className="md-title-medium">Retrieval check</h2>
      <p
        className="md-body-small"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        Raw vector search, no model involved. Scores are meaningful only relative
        to each other — an irrelevant match can still score 0.57.
      </p>

      <form onSubmit={run} className="flex items-start gap-3">
        <TextField
          label="Search the chunks"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={!enabled}
          className="flex-1"
        />
        <Button
          type="submit"
          variant="tonal"
          disabled={!enabled || busy || !query.trim()}
          className="mt-2"
        >
          {busy ? <IconSpinner /> : <IconSearch />}
          Search
        </Button>
      </form>

      {error && (
        <p className="md-body-medium" style={{ color: "var(--md-error)" }}>
          {error}
        </p>
      )}
      {hits && hits.length === 0 && (
        <p
          className="md-body-medium"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          No matches.
        </p>
      )}
      {hits && hits.length > 0 && (
        <Hits hits={hits} used={[]} onCite={onCite} />
      )}
    </section>
  );
}

function Hits({
  hits,
  used,
  onCite,
}: {
  hits: SearchHit[];
  used: number[];
  onCite: (id: string) => Promise<void>;
}) {
  const cited = new Set(used);
  return (
    <ul className="space-y-2">
      {hits.map((hit, i) => {
        const isCited = cited.has(i + 1);
        return (
          <li key={hit.chunk_id}>
            <Ripplable
              as="div"
              onClick={() => void onCite(hit.chunk_id)}
              role="button"
              tabIndex={0}
              className="flex w-full cursor-pointer items-center gap-3 rounded-[var(--md-shape-md)] px-3 py-2.5"
              style={{
                background: isCited
                  ? "var(--md-secondary-container)"
                  : "var(--md-surface-container-high)",
                color: isCited
                  ? "var(--md-on-secondary-container)"
                  : "var(--md-on-surface)",
              }}
            >
              <span
                className="md-label-small grid h-7 w-7 shrink-0 place-items-center rounded-[var(--md-shape-full)] tabular-nums"
                style={{
                  background: isCited
                    ? "var(--md-primary)"
                    : "var(--md-surface-container-highest)",
                  color: isCited
                    ? "var(--md-on-primary)"
                    : "var(--md-on-surface-variant)",
                }}
              >
                {i + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className="md-body-medium block truncate">
                  {hit.heading
                    ? hit.heading.replace(/^#+\s*/, "")
                    : hit.filename}
                </span>
                <span
                  className="md-body-small block truncate"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {hit.filename}
                  {hit.page !== null && ` · p.${hit.page}`}
                </span>
              </span>
              <span className="md-body-small shrink-0 tabular-nums">
                {hit.score.toFixed(3)}
                {hit.rrf_score != null && (
                  <span style={{ color: "var(--md-on-surface-variant)" }}>
                    {" "}
                    / {hit.rrf_score.toFixed(4)}
                  </span>
                )}
              </span>
            </Ripplable>
          </li>
        );
      })}
    </ul>
  );
}

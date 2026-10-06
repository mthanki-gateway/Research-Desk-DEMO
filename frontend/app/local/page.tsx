"use client";

import { useEffect, useMemo, useState } from "react";
import {
  localClassify,
  localEntities,
  localStatus,
  type LocalEntities,
  type LocalScores,
  type LocalStatus,
} from "@/lib/api";
import { Button, Chip, TextArea, TextField } from "../md";
import { IconGrid, IconSpinner } from "../icons";

/**
 * Small encoders running on the API's own CPU -- no key, no upstream.
 *
 * Three use cases, two models. GLiNER pulls out entities for labels typed at
 * request time. GLiClass scores every candidate label on its own, 0..1 --
 * shown raw for JEV-style scoring, and read as a ROUTE for intent routing,
 * where the top label wins only if it clears a cutoff and beats the runner-up.
 */

type CaseId = "entities" | "routing" | "jev";

type Case = {
  id: CaseId;
  label: string;
  blurb: string;
  text: string;
  labels: string[];
};

const CASES: Case[] = [
  {
    id: "routing",
    label: "Intent routing",
    blurb:
      "The top label is where the message goes — if it clears the cutoff and clearly beats the runner-up. Scores are independent, so two intents can both be near 1; a router that silently picks one of a tie is wrong half the time with full confidence. Try “I was charged twice and want one refunded” (billing and cancel tie), or “hey how's it going” with no small-talk label.",
    text: "The app crashes every time I try to upload a PDF larger than 10MB.",
    labels: [
      "billing question",
      "technical support",
      "cancel subscription",
      "sales enquiry",
      "small talk",
    ],
  },
  {
    id: "jev",
    label: "JEV scoring",
    blurb:
      "Each statement is scored on its own, 0 to 1. Adding a label does not lower the others and the scores do not sum to 1 — this is “how far does each of these hold”, not “which one is it”.",
    text: "The quarterly report shows revenue up 12% but the team warns supply delays could hurt Q3 margins.",
    labels: [
      "positive financial news",
      "risk warning",
      "mentions revenue growth",
      "about hiring",
      "forward-looking statement",
    ],
  },
  {
    id: "entities",
    label: "Entities",
    blurb:
      "Zero-shot NER: the entity types are whatever you type, not a tag set fixed at training. Each span carries its own confidence.",
    text: "Dr. Priya Raman from Novartis met the Basel city council on 14 March 2025 to discuss a CHF 4.2 million grant.",
    labels: ["person", "organization", "location", "date", "money"],
  },
];

// A route needs a top score at least this high...
const ROUTE_CUTOFF = 0.5;
// ...and at least this far ahead of the runner-up, or it is a tie.
const ROUTE_MARGIN = 0.15;

export default function LocalModels() {
  const [status, setStatus] = useState<LocalStatus | null>(null);
  const [caseId, setCaseId] = useState<CaseId>("routing");
  const current = CASES.find((c) => c.id === caseId)!;
  const [text, setText] = useState(current.text);
  const [labels, setLabels] = useState(current.labels.join(", "));
  const [threshold, setThreshold] = useState(0.4);
  const [busy, setBusy] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scores, setScores] = useState<LocalScores | null>(null);
  const [ents, setEnts] = useState<LocalEntities | null>(null);
  // Where entities run. Browser by default: it works on the hosted site,
  // where the API has no torch. Server is offered only when it can answer.
  const [runtime, setRuntime] = useState<"browser" | "server">("browser");
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);

  useEffect(() => {
    localStatus()
      .then(setStatus)
      .catch((e) => setError(e instanceof Error ? e.message : "Could not reach the API"));
  }, []);

  function pick(id: CaseId) {
    const c = CASES.find((x) => x.id === id)!;
    setCaseId(id);
    setText(c.text);
    setLabels(c.labels.join(", "));
    setScores(null);
    setEnts(null);
    setHasRun(false);
    setError(null);
  }

  function clearOutput() {
    setScores(null);
    setEnts(null);
  }

  const labelList = useMemo(
    () =>
      labels
        .split(",")
        .map((l) => l.trim())
        .filter(Boolean),
    [labels],
  );

  // Whether the model for this case still has to load. The first call then
  // includes a download and a load, and its latency is not the model's.
  const inBrowser = caseId === "entities" && runtime === "browser";
  const cold =
    status &&
    !inBrowser &&
    (caseId === "entities" ? !status.loaded.gliner : !status.loaded.gliclass);
  // Routing and JEV have no browser model (see lib/browser-nlp.ts), so they
  // need the server's; entities in the browser need nothing from it.
  const runnable = inBrowser || !!status?.installed;

  async function run() {
    if (busy || !text.trim() || labelList.length === 0) return;
    setBusy(true);
    setError(null);
    setScores(null);
    setEnts(null);
    try {
      if (inBrowser) {
        const nlp = await import("@/lib/browser-nlp");
        setEnts(
          await nlp.entities(text, labelList, threshold, (p) =>
            setProgress({ loaded: p.loaded, total: p.total }),
          ),
        );
        setHasRun(true);
        setProgress(null);
        return;
      }
      if (caseId === "entities") {
        setEnts(await localEntities({ text, labels: labelList, threshold }));
      } else {
        setScores(
          await localClassify({ text, labels: labelList }),
        );
      }
      setHasRun(true);
      setStatus(await localStatus());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Local models</h1>
        <p className="md-body-medium mt-1" style={{ color: "var(--md-on-surface-variant)" }}>
          GLiNER and GLiClass — small encoders answering one narrow question in
          milliseconds, with no API key. Entities run right here in your browser;
          routing and JEV scoring run on the API&apos;s own CPU in the local stack.
        </p>
      </header>

      {status && !status.installed && !inBrowser && (
        <div
          className="md-card md-card-filled p-4"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {status.hosted ? (
            <>
              <p className="md-title-small">Runs in the local stack only</p>
              <p className="md-body-medium mt-1">
                Routing and JEV scoring use GLiClass on the API&apos;s own CPU, and
                the hosted API is deliberately built without torch — it does not
                fit the free tier, and GLiClass has no browser build yet. Run the
                app locally with <code>docker compose up</code> to try them, or
                use Entities, which runs in your browser.
              </p>
            </>
          ) : (
            <>
              <p className="md-title-small">Local models are not installed</p>
              <p className="md-body-medium mt-1">
                They are in the <code>local-nlp</code> extra, which the dev image
                installs. Rebuild it: <code>docker compose build api</code>, then{" "}
                <code>docker compose up -d --force-recreate api</code>.
              </p>
            </>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-2" role="tablist">
        {CASES.map((c) => (
          <Chip
            key={c.id}
            role="tab"
            aria-selected={c.id === caseId}
            selected={c.id === caseId}
            onClick={() => pick(c.id)}
          >
            {c.label}
          </Chip>
        ))}
      </div>
      <p className="md-body-medium" style={{ color: "var(--md-on-surface-variant)" }}>
        {current.blurb}
      </p>

      <TextArea
        label="Text"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          clearOutput();
        }}
        surface="var(--md-surface)"
        disabled={busy}
      />
      <TextField
        label={caseId === "entities" ? "Entity types, comma-separated" : "Labels, comma-separated"}
        value={labels}
        onChange={(e) => {
          setLabels(e.target.value);
          clearOutput();
        }}
        disabled={busy}
      />
      {caseId === "entities" && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="md-body-medium">Run on</span>
          <Chip selected={runtime === "browser"} onClick={() => setRuntime("browser")}>
            This browser
          </Chip>
          {status?.installed && (
            <Chip selected={runtime === "server"} onClick={() => setRuntime("server")}>
              Server
            </Chip>
          )}
          {runtime === "browser" && (
            <span className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
              Downloads GLiNER (183MB) once, then it is cached. The text never leaves
              this device.
            </span>
          )}
        </div>
      )}
      {caseId === "entities" && (
        <label className="md-body-medium flex items-center gap-3">
          Threshold
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={threshold}
            onChange={(e) => {
              setThreshold(Number(e.target.value));
              clearOutput();
            }}
            className="flex-1"
          />
          <span className="tabular-nums">{threshold.toFixed(2)}</span>
        </label>
      )}

      <div className="flex items-center gap-3">
        <Button
          onClick={() => void run()}
          disabled={busy || !text.trim() || labelList.length === 0 || !runnable}
        >
          {busy ? <IconSpinner /> : <IconGrid />}
          {busy ? (progress ? "Downloading…" : "Running…") : hasRun ? "Run again" : "Run"}
        </Button>
        {progress && progress.total > 0 && (
          <span className="md-body-small tabular-nums" style={{ color: "var(--md-on-surface-variant)" }}>
            {Math.round(progress.loaded / 1e6)} / {Math.round(progress.total / 1e6)} MB
          </span>
        )}
        {cold && !busy && (
          <span className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
            First call downloads and loads the model — expect seconds, not milliseconds.
          </span>
        )}
      </div>

      {error && (
        <div
          className="md-card md-card-filled p-4"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          <p className="md-body-medium">{error}</p>
        </div>
      )}

      {scores && <ScoreView data={scores} routing={caseId === "routing"} />}
      {ents && <EntityView data={ents} text={text} />}
    </div>
  );
}

function ScoreView({ data, routing }: { data: LocalScores; routing: boolean }) {
  const [top, second] = data.scores;
  const margin = top ? top.score - (second?.score ?? 0) : 0;
  const routed = routing && top && top.score >= ROUTE_CUTOFF && margin >= ROUTE_MARGIN;
  return (
    <section className="md-card md-card-outlined space-y-4 p-4">
      {routing && (
        <p className="md-title-medium">
          {routed ? (
            <>
              Route → <strong>{top.label}</strong>
            </>
          ) : top && top.score >= ROUTE_CUTOFF ? (
            <>
              Ambiguous — {top.label} vs {second?.label} → ask, or fall back
            </>
          ) : (
            <>No confident intent (top below {ROUTE_CUTOFF}) → fallback</>
          )}
        </p>
      )}
      <ul className="space-y-2">
        {data.scores.map((s) => (
          <li key={s.label} className="space-y-1">
            <div className="md-body-medium flex justify-between gap-3">
              <span>{s.label}</span>
              <span className="tabular-nums">{s.score.toFixed(3)}</span>
            </div>
            <div
              className="h-2 overflow-hidden rounded-[var(--md-shape-full)]"
              style={{ background: "var(--md-surface-container-high)" }}
            >
              <div
                className="h-full"
                style={{
                  width: `${Math.round(s.score * 100)}%`,
                  background:
                    routing && s === top && routed
                      ? "var(--md-primary)"
                      : !routing && s.score >= 0.5
                        ? "var(--md-primary)"
                        : "var(--md-outline)",
                }}
              />
            </div>
          </li>
        ))}
      </ul>
      <Footer
        items={[
          ["scores", "independent 0–1 · do not sum to 1"],
          ...(routing ? ([["margin", margin.toFixed(2)]] as [string, string][]) : []),
          ["latency", `${data.elapsed_ms} ms`],
          ["model", data.model],
        ]}
      />
    </section>
  );
}

function EntityView({ data, text }: { data: LocalEntities; text: string }) {
  // The text with each span marked in place, so a wrong boundary is visible
  // rather than hidden in a list of strings.
  const parts: React.ReactNode[] = [];
  let at = 0;
  [...data.entities]
    .sort((a, b) => a.start - b.start)
    .forEach((e, i) => {
      if (e.start < at) return;
      parts.push(text.slice(at, e.start));
      parts.push(
        <mark
          key={i}
          className="rounded-[var(--md-shape-xs)] px-1"
          style={{
            background: "var(--md-primary-container)",
            color: "var(--md-on-primary-container)",
          }}
          title={`${e.label} · ${e.score.toFixed(3)}`}
        >
          {text.slice(e.start, e.end)}
          <span className="md-label-small ml-1 opacity-75">{e.label}</span>
        </mark>,
      );
      at = e.end;
    });
  parts.push(text.slice(at));

  return (
    <section className="md-card md-card-outlined space-y-4 p-4">
      <p className="md-body-large leading-8">{parts}</p>
      {data.entities.length === 0 ? (
        <p className="md-body-medium" style={{ color: "var(--md-on-surface-variant)" }}>
          Nothing above the threshold.
        </p>
      ) : (
        <table className="md-body-medium w-full">
          <tbody>
            {data.entities.map((e, i) => (
              <tr key={i}>
                <td className="py-1 pr-3">{e.text}</td>
                <td className="py-1 pr-3" style={{ color: "var(--md-on-surface-variant)" }}>
                  {e.label}
                </td>
                <td className="py-1 text-right tabular-nums">{e.score.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <Footer
        items={[
          ["latency", `${data.elapsed_ms} ms`],
          ["model", data.model],
        ]}
      />
    </section>
  );
}

function Footer({ items }: { items: [string, string][] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {items.map(([k, v]) => (
        <span
          key={k}
          className="md-label-medium rounded-[var(--md-shape-full)] px-3 py-1.5"
          style={{
            background: "var(--md-surface-container-high)",
            color: "var(--md-on-surface-variant)",
          }}
        >
          {k} <strong>{v}</strong>
        </span>
      ))}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import {
  playgroundComplete,
  playgroundModels,
  playgroundStatus,
  type Completion,
  type GroqModel,
  type PlaygroundStatus,
} from "@/lib/api";
import { Button, TextArea } from "../md";
import { IconChip, IconSpinner } from "../icons";

/**
 * One model call, with the numbers on screen.
 *
 * Deliberately not a chat. There is no history, no retrieval, no citations and
 * no streaming: every one of those is a thing to build and debug that teaches
 * nothing about the model behind the API. What it does show is the part worth
 * seeing -- latency, tokens in and out, and tokens per second -- because the
 * reason to try an LPU-served open model at all is that those numbers differ
 * from the Gemini path this app already uses.
 */
export default function Playground() {
  const [status, setStatus] = useState<PlaygroundStatus | null>(null);
  const [models, setModels] = useState<GroqModel[]>([]);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState(
    "Explain in three sentences why an LPU serves tokens faster than a GPU.",
  );
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Completion | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const s = await playgroundStatus();
      setStatus(s);
      setModel((m) => m || s.default_model);
      // Only if configured: an unconfigured key makes this a guaranteed empty
      // list, and a failed request behind a "not configured" banner is noise.
      if (s.enabled) setModels(await playgroundModels());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the API");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function send() {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    // Cleared, not left in place: a stale answer sitting under a spinner
    // reads as the new one having arrived instantly.
    setResult(null);
    try {
      setResult(await playgroundComplete({ prompt, model }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Playground</h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          One call to an open-weights model over an OpenAI-compatible API. No
          retrieval, no history — just the request, the reply, and what it cost.
        </p>
      </header>

      {status && !status.enabled && (
        <div
          className="md-card md-card-filled p-4"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          <p className="md-title-small">No API key configured</p>
          <p className="md-body-medium mt-1">
            Set <code>GROQ_API_KEY</code> in <code>.env</code>, then recreate
            the container:{" "}
            <code>docker compose up -d --force-recreate api</code>. A plain{" "}
            <code>restart</code> does not re-read <code>.env</code>.
          </p>
        </div>
      )}

      {status && (
        <div className="md-card md-card-outlined space-y-4 p-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="md-title-small shrink-0">Model</span>
            {models.length > 0 ? (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2"
                style={{
                  background: "var(--md-surface)",
                  color: "var(--md-on-surface)",
                  border: "1px solid var(--md-outline)",
                }}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                    {m.context_window
                      ? ` — ${(m.context_window / 1000).toFixed(0)}k ctx`
                      : ""}
                  </option>
                ))}
              </select>
            ) : (
              <code className="md-body-small">{model || "—"}</code>
            )}
          </div>
          {/* The base URL is on screen because it is the ONE line that changes
              to move off GroqCloud. Seeing it makes "swap to a self-hosted
              vLLM" concrete rather than a claim in a comment. */}
          <p
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Endpoint <code>{status.base_url}</code> — swapping this for a local
            vLLM or Ollama is the whole difference between hosted and
            self-hosted.
          </p>
        </div>
      )}

      <TextArea
        label="Prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        surface="var(--md-surface)"
        disabled={busy}
      />

      <div className="flex items-center gap-3">
        <Button
          onClick={() => void send()}
          disabled={busy || !prompt.trim() || (status ? !status.enabled : true)}
        >
          {busy ? <IconSpinner /> : <IconChip />}
          {busy ? "Calling…" : "Send"}
        </Button>
        {busy && (
          <span
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Waiting for the model…
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

      {result && (
        <section className="space-y-4">
          <div className="md-card md-card-outlined p-4">
            <p className="md-body-large whitespace-pre-wrap">{result.text}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Stat label="latency" value={`${result.elapsed_ms} ms`} />
            {result.tokens_per_second !== null && (
              <Stat
                label="throughput"
                value={`${result.tokens_per_second} tok/s`}
                strong
              />
            )}
            <Stat label="prompt" value={`${result.prompt_tokens} tok`} />
            <Stat label="completion" value={`${result.completion_tokens} tok`} />
            <Stat label="model" value={result.model} />
            {/* "length" means max_tokens cut it off, which otherwise looks
                like the model simply trailing off mid-sentence. */}
            {result.finish_reason && result.finish_reason !== "stop" && (
              <Stat label="stopped" value={result.finish_reason} warn />
            )}
          </div>
        </section>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  strong,
  warn,
}: {
  label: string;
  value: string;
  strong?: boolean;
  warn?: boolean;
}) {
  const background = warn
    ? "var(--md-error-container)"
    : strong
      ? "var(--md-primary-container)"
      : "var(--md-surface-container-high)";
  const color = warn
    ? "var(--md-on-error-container)"
    : strong
      ? "var(--md-on-primary-container)"
      : "var(--md-on-surface-variant)";
  return (
    <span
      className="md-label-medium rounded-[var(--md-shape-full)] px-3 py-1.5"
      style={{ background, color }}
    >
      {label} <strong className="tabular-nums">{value}</strong>
    </span>
  );
}

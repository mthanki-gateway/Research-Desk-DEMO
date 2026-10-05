"use client";

import { useEffect, useState } from "react";
import { deleteKey, getKeys, saveKey, type KeyStatus, type ProviderKey } from "@/lib/keys";
import { Button } from "../md";
import { IconSpinner } from "../icons";

/**
 * API keys: what each one powers, how to get it, and whether one is in place.
 *
 * The server's keys stand in for now, so every feature works out of the box.
 * When bring-your-own-keys is switched on (REQUIRE_USER_KEYS), the server's
 * stop counting and anything without a key here is greyed out in the app --
 * this page is then the one place to unlock it.
 */

const FEATURE_LABELS: Record<string, string> = {
  chat: "Chat",
  library: "Library",
  atlas: "Atlas",
  lab: "Lab",
  parley: "Parley voice",
  web_search: "Web search",
  hydrate: "Hydrate from web",
  playground: "Playground",
  transcribe: "Transcribe",
  interview_transcription: "Interview transcription",
  nvidia_speech: "NVIDIA speech",
};

export default function SettingsPage() {
  const [status, setStatus] = useState<KeyStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getKeys()
      .then(setStatus)
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load settings"));
  }, []);

  return (
    <div className="mx-auto max-w-3xl space-y-8 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Settings</h1>
        <p className="md-body-medium mt-1" style={{ color: "var(--md-on-surface-variant)" }}>
          Bring your own API keys. Keys are encrypted on the server and never shown
          again in full; only the last four characters appear here.
        </p>
      </header>

      {error && (
        <p className="md-body-medium" style={{ color: "var(--md-error)" }}>
          {error}
        </p>
      )}
      {!status && !error && <div className="md-skeleton h-64" aria-hidden />}

      {status && (
        <>
          <section id="features" className="space-y-3">
            <h2 className="md-title-medium">Feature access</h2>
            <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
              {status.require_user_keys
                ? "Each feature needs its key added below."
                : "Everything works today on the shared server keys. Adding your own moves usage onto your account; in a future release your own keys will be required."}
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(status.features).map(([f, on]) => (
                <span
                  key={f}
                  className="md-label-medium flex items-center gap-1.5 rounded-[var(--md-shape-full)] px-3 py-1.5"
                  style={{
                    background: on ? "var(--md-secondary-container)" : "var(--md-surface-container-high)",
                    color: on ? "var(--md-on-secondary-container)" : "var(--md-on-surface-variant)",
                    opacity: on ? 1 : 0.6,
                  }}
                >
                  <span aria-hidden>{on ? "●" : "○"}</span>
                  {FEATURE_LABELS[f] ?? f}
                </span>
              ))}
            </div>
          </section>

          <section id="keys" className="space-y-3">
            <h2 className="md-title-medium">API keys</h2>
            {status.providers.map((p, i) => (
              <ProviderCard key={p.id} p={p} tone={i % 4} onChange={setStatus} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}

function ProviderCard({
  p,
  tone,
  onChange,
}: {
  p: ProviderKey;
  tone: number;
  onChange: (s: KeyStatus) => void;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [help, setHelp] = useState(p.source === "none");

  async function run(fn: () => Promise<KeyStatus>) {
    setBusy(true);
    setErr(null);
    try {
      onChange(await fn());
      setValue("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  const badge =
    p.source === "yours"
      ? { text: `Your key ••••${p.yours?.last4}`, bg: "var(--md-primary-container)", fg: "var(--md-on-primary-container)" }
      : p.source === "server"
        ? { text: "Using the shared server key", bg: "var(--md-surface-container-highest)", fg: "var(--md-on-surface-variant)" }
        : { text: "Not set — features are off", bg: "var(--md-error-container)", fg: "var(--md-on-error-container)" };

  return (
    <div className="group md-card md-card-outlined space-y-3 p-4">
      <div className="flex items-start gap-3">
        <span className={`md-morph-tile md-morph-${tone}`} aria-hidden>
          <KeyGlyph />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="md-title-small">{p.label}</h3>
            <span
              className="md-label-small rounded-[var(--md-shape-full)] px-2 py-0.5"
              style={{ background: badge.bg, color: badge.fg }}
            >
              {badge.text}
            </span>
          </div>
          <p className="md-body-small mt-0.5" style={{ color: "var(--md-on-surface-variant)" }}>
            {p.powers}
          </p>
        </div>
      </div>

      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (value.trim()) void run(() => saveKey(p.id, value));
        }}
      >
        <input
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={p.yours ? "Paste a new key to replace it" : "Paste your key"}
          aria-label={`${p.label} API key`}
          className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2 outline-none"
          style={{
            background: "var(--md-surface)",
            color: "var(--md-on-surface)",
            border: "1px solid var(--md-outline)",
          }}
        />
        <Button type="submit" disabled={busy || !value.trim()}>
          {busy && <IconSpinner />}
          Save
        </Button>
        {p.yours && (
          <Button variant="text" type="button" disabled={busy} onClick={() => void run(() => deleteKey(p.id))}>
            Remove
          </Button>
        )}
      </form>
      {err && (
        <p className="md-body-small" style={{ color: "var(--md-error)" }}>
          {err}
        </p>
      )}

      <div>
        <button
          type="button"
          onClick={() => setHelp((v) => !v)}
          className="md-label-large"
          style={{ color: "var(--md-primary)" }}
          aria-expanded={help}
        >
          {help ? "Hide" : "How to get a key"}
        </button>
        {help && (
          <ol className="md-body-small mt-2 list-decimal space-y-1 pl-5" style={{ color: "var(--md-on-surface-variant)" }}>
            {p.steps.map((s, k) => (
              <li key={k}>{s}</li>
            ))}
            <li>
              <a href={p.url} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--md-primary)" }}>
                Open {new URL(p.url).hostname}
              </a>
            </li>
          </ol>
        )}
      </div>
    </div>
  );
}

function KeyGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
      <circle cx="8" cy="15" r="4" />
      <path d="M11 12l9-9M17 6l3 3M15 8l2 2" />
    </svg>
  );
}

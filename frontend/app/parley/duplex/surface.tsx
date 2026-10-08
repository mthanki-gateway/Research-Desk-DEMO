"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type DuplexStatus, getDuplexStatus } from "@/lib/api";
import { Button, Dialog, useAutoHideScroll } from "../../md";
import { IconDuplex, IconMic } from "../../icons";
import { type DuplexEvent, type DuplexSession, openDuplexSession } from "../duplexSession";
import { PATIENCE, type Patience, disableTurnDetection, enableTurnDetection } from "../turnDetector";

/**
 * Duplex: talk to it the way you would talk to a person.
 *
 * The microphone stays open while it speaks (barge in whenever you like), and
 * it starts writing its answer while you are still finishing the sentence, so
 * the first sound arrives about two seconds after you stop rather than five or
 * six. The panel on the right shows that happening: every answer it starts,
 * the ones it throws away because you kept talking, and the one it adopts.
 */

type Turn = {
  user: string;
  assistant: string;
  ms?: number;
  early?: boolean;
  model?: string;
};

type DraftLog = { id: number; state: string; text: string; model?: string; reason?: string };

/** Labels that usually mean a headset, and ones that mean the laptop itself. */
const HEADSET = /headset|headphone|airpods|buds|bluetooth|jabra|bose|sony|usb|logi|hands-free|earphone/i;
const BUILT_IN = /array|built-?in|internal|realtek|laptop/i;

type Phase = "idle" | "connecting" | "listening" | "thinking" | "speaking";

const PHASE_LABEL: Record<Phase, string> = {
  idle: "Not listening",
  connecting: "Connecting…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

function Banner({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
      style={{ background: "var(--md-error-container)", color: "var(--md-on-error-container)" }}
    >
      {children}
    </p>
  );
}

export default function DuplexSurface() {
  const [status, setStatus] = useState<DuplexStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [partial, setPartial] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [speaking, setSpeaking] = useState("");
  const [drafts, setDrafts] = useState<DraftLog[]>([]);
  const [stt, setStt] = useState("");
  const [tts, setTts] = useState("");
  const [detector, setDetector] = useState<"loading" | "ready" | "error">("loading");
  const [level, setLevel] = useState(0);
  const [vad, setVad] = useState<{ p: number; speech: boolean; paused: boolean; silentMs: number } | null>(null);
  const [ended, setEnded] = useState("");
  const [voiceName, setVoiceName] = useState("Kore");
  const [patience, setPatience] = useState<Patience>("eager");
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [micId, setMicId] = useState("");
  const [micInUse, setMicInUse] = useState("");
  const [openDraft, setOpenDraft] = useState<DraftLog | null>(null);
  const draftsScroll = useAutoHideScroll<HTMLUListElement>();

  const session = useRef<DuplexSession | null>(null);
  const draftId = useRef(0);
  // The turn being spoken right now, kept in a ref so event handlers (which
  // are created once) always append to the latest one.
  const pending = useRef<Turn | null>(null);

  useEffect(() => {
    getDuplexStatus()
      .then((s) => {
        setStatus(s);
        setVoiceName(s.default_voice);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not reach the server."));
    const refreshMics = () =>
      navigator.mediaDevices
        ?.enumerateDevices()
        .then((all) => setMics(all.filter((d) => d.kind === "audioinput")))
        .catch(() => {});
    void refreshMics();
    // Plugging in a headset should appear without a reload.
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshMics);
    try {
      setMicId(localStorage.getItem("duplex.mic") ?? "");
    } catch {
      /* private window */
    }
    try {
      const saved = localStorage.getItem("duplex.patience") as Patience | null;
      if (saved && PATIENCE.some((p) => p.id === saved)) setPatience(saved);
    } catch {
      /* private window */
    }
    // Fetch the ~11MB turn models now, not on Start: speech before they are
    // ready is not heard by the detector, so the first sentence would never end.
    enableTurnDetection(() => {}, "eager");
    disableTurnDetection();
    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", refreshMics);
      session.current?.close();
    };
  }, []);

  const log = useCallback((entry: Omit<DraftLog, "id">) => {
    setDrafts((d) => [{ id: ++draftId.current, ...entry }, ...d].slice(0, 14));
  }, []);

  const onEvent = useCallback(
    (e: DuplexEvent) => {
      switch (e.type) {
        case "ready":
          setStt(e.stt);
          setTts(e.tts);
          setPhase("listening");
          break;
        case "stt":
          setStt(e.name);
          break;
        case "partial":
          setPartial(e.text);
          break;
        case "final":
          setPartial("");
          pending.current = { user: e.text, assistant: "" };
          setSpeaking("");
          setPhase("thinking");
          break;
        case "draft":
          log({
            state: e.state,
            text: e.text ?? "",
            model: e.model,
            reason: e.reason,
          });
          break;
        case "say":
          setPhase("speaking");
          setSpeaking((s) => (s ? `${s} ${e.text}` : e.text));
          if (pending.current) {
            pending.current.assistant = pending.current.assistant
              ? `${pending.current.assistant} ${e.text}`
              : e.text;
          }
          break;
        case "latency":
          if (pending.current) {
            pending.current.ms = e.endpoint_to_audio_ms;
            pending.current.early = e.ready_at_endpoint;
            pending.current.model = e.model;
          }
          break;
        case "resumed":
          setPhase("listening");
          pending.current = null;
          break;
        case "stop":
        case "turn_end": {
          const t = pending.current;
          if (t && (t.assistant || e.type === "turn_end")) {
            setTurns((all) => [...all, t]);
          }
          pending.current = null;
          setSpeaking("");
          setPhase("listening");
          break;
        }
        case "error":
          setError(e.detail);
          break;
        case "level":
          setLevel(e.level);
          break;
        case "vad":
          setVad({ p: e.p, speech: e.speech, paused: e.paused, silentMs: e.silentMs });
          break;
        case "endpoint":
          setEnded(e.reason);
          break;
        case "mic":
          setMicInUse(e.label);
          break;
        case "detector":
          setDetector(e.state);
          if (e.state === "error") setError(e.detail ?? "The turn detector could not start.");
          break;
        case "closed":
          setPhase("idle");
          setLevel(0);
          session.current = null;
          break;
      }
    },
    [log],
  );

  const start = useCallback(async () => {
    if (!status) return;
    // With no choice made, prefer a headset over the laptop's own array when
    // the browser can name them: the OS "default" is often the laptop.
    const headset = mics.find(
      (d) => d.label && HEADSET.test(d.label) && !BUILT_IN.test(d.label) && d.deviceId !== "default",
    );
    const chosen = micId || headset?.deviceId || "";
    setError(null);
    setPhase("connecting");
    try {
      session.current = await openDuplexSession(
        voiceName,
        patience,
        status.pause_ms,
        status.output_rate,
        onEvent,
        chosen,
      );
      // Device names are hidden until the browser has granted the microphone.
      void navigator.mediaDevices
        .enumerateDevices()
        .then((all) => setMics(all.filter((d) => d.kind === "audioinput")));
    } catch (err) {
      setPhase("idle");
      setError(
        err instanceof Error && err.name === "NotAllowedError"
          ? "The microphone was refused. Allow it in the address bar and try again."
          : err instanceof Error
            ? err.message
            : "Could not start.",
      );
    }
  }, [status, voiceName, patience, micId, mics, onEvent]);

  const stop = useCallback(() => {
    session.current?.close();
    session.current = null;
    setPhase("idle");
    setPartial("");
    setLevel(0);
  }, []);

  const live = phase !== "idle";
  const counts = drafts.reduce(
    (acc, d) => ({ ...acc, [d.state]: (acc[d.state as keyof typeof acc] ?? 0) + 1 }),
    { started: 0, discarded: 0, adopted: 0, fresh: 0, skipped: 0 } as Record<string, number>,
  );
  const timed = turns.filter((t) => t.ms != null);
  const average = timed.length
    ? Math.round(timed.reduce((s, t) => s + (t.ms ?? 0), 0) / timed.length)
    : null;
  const last = [...turns].reverse().find((t) => t.ms != null);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small flex items-center gap-2">
          <IconDuplex className="h-6 w-6" />
          Duplex
        </h1>
        <p className="md-body-medium mt-1" style={{ color: "var(--md-on-surface-variant)" }}>
          Talk naturally. It starts answering while you finish the sentence, and you can cut in at any time.
        </p>
      </header>

      {status && !status.enabled && (
        <Banner>
          Duplex needs a Gemini key (for the voice) and a Groq or Gemini key (for the answers). Add them in
          Settings → API keys.
        </Banner>
      )}
      {status && status.enabled && status.stt === "gemini" && (
        <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
          No Groq key: transcription arrives only after you stop talking, so answers cannot start early.
          Add a Groq key for the full effect.
        </p>
      )}
      {error && <Banner>{error}</Banner>}

      <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
        <section className="space-y-4">
          <div className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-8">
            <button
              type="button"
              onClick={live ? stop : () => void start()}
              disabled={!status?.enabled || phase === "connecting"}
              aria-label={live ? "Stop" : "Start"}
              className="relative grid h-24 w-24 place-items-center rounded-full disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                background: live ? "var(--md-error-container)" : "var(--md-primary)",
                color: live ? "var(--md-on-error-container)" : "var(--md-on-primary)",
                boxShadow: live
                  ? `0 0 0 ${Math.round(Math.min(1, level * 4) * 14)}px color-mix(in srgb, var(--md-primary) 22%, transparent)`
                  : undefined,
                transition: "box-shadow 80ms linear",
              }}
            >
              <IconMic className="h-9 w-9" />
            </button>
            <div className="text-center">
              <p className="md-title-medium">{live ? PHASE_LABEL[phase] : "Start talking"}</p>
              <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
                {live
                  ? "The microphone stays open while it speaks."
                  : "One click, then just speak. Headphones keep it from hearing itself."}
              </p>
            </div>
            {live && micInUse && (
              <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
                Listening on: {micInUse}
              </p>
            )}
            {live && (
              <p className="md-label-small tabular-nums" style={{ color: "var(--md-on-surface-variant)" }}>
                {detector !== "ready"
                  ? detector === "error"
                    ? "Turn detector failed to load"
                    : "Loading the turn detector… (speak after it says ready)"
                  : vad
                    ? `voice ${vad.p.toFixed(2)} · ${vad.speech ? (vad.paused ? "paused" : "hearing you") : "waiting for you"}${
                        vad.speech ? ` · silent ${vad.silentMs}ms` : ""
                      }${ended ? ` · last end: ${ended}` : ""}`
                    : "waiting for audio…"}
              </p>
            )}
            <div
              className="md-body-large min-h-[2.5rem] w-full text-center"
              style={{ color: partial ? "var(--md-on-surface)" : "var(--md-on-surface-variant)" }}
              aria-live="polite"
            >
              {partial || (speaking ? speaking : live ? "…" : "")}
            </div>
          </div>

          <div className="space-y-3">
            {turns.length === 0 && (
              <p className="md-body-medium" style={{ color: "var(--md-on-surface-variant)" }}>
                Nothing said yet. Try “what does a hash map do, and when would you not use one?”
              </p>
            )}
            {turns.map((t, i) => (
              <div key={i} className="space-y-2">
                <p
                  className="md-body-medium ml-auto w-fit max-w-[85%] rounded-[var(--md-shape-lg)] px-4 py-2"
                  style={{
                    background: "var(--md-secondary-container)",
                    color: "var(--md-on-secondary-container)",
                  }}
                >
                  {t.user}
                </p>
                {t.assistant && (
                  <p
                    className="md-body-medium w-fit max-w-[85%] rounded-[var(--md-shape-lg)] px-4 py-2"
                    style={{ background: "var(--md-surface-container-high)" }}
                  >
                    {t.assistant}
                    {t.ms != null && (
                      <span
                        className="md-label-small ml-2 whitespace-nowrap"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        {(t.ms / 1000).toFixed(1)}s{t.early ? " · answered early" : ""}
                      </span>
                    )}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>

        <aside className="space-y-4">
          <div className="md-card md-card-outlined space-y-3 p-4">
            <h2 className="md-title-small">Latency</h2>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="md-headline-small">{last?.ms != null ? (last.ms / 1000).toFixed(1) : "–"}s</p>
                <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
                  you stopped → first sound
                </p>
              </div>
              <div>
                <p className="md-headline-small">{average != null ? (average / 1000).toFixed(1) : "–"}s</p>
                <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
                  average of {timed.length}
                </p>
              </div>
            </div>
          </div>

          <div className="md-card md-card-outlined space-y-3 p-4">
            <h2 className="md-title-small">Drafts</h2>
            <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
              {counts.started} started · {counts.discarded} discarded · {counts.adopted} adopted
              {counts.skipped ? ` · ${counts.skipped} skipped (budget)` : ""}
            </p>
            <ul ref={draftsScroll} className="md-scroll max-h-64 space-y-1.5 overflow-y-auto pr-1">
              {drafts.length === 0 && (
                <li className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
                  Answers it writes ahead of you will show here.
                </li>
              )}
              {drafts.map((d) => (
                <li key={d.id}>
                  <button
                    type="button"
                    onClick={() => setOpenDraft(d)}
                    title="View in full"
                    className="md-body-small md-state flex w-full gap-2 rounded-[var(--md-shape-sm)] px-1 py-0.5 text-left"
                  >
                  <span
                    className="md-label-small mt-0.5 h-fit shrink-0 rounded-[var(--md-shape-full)] px-2 py-0.5"
                    style={{
                      background:
                        d.state === "adopted" || d.state === "fresh"
                          ? "var(--md-primary-container)"
                          : "var(--md-surface-container-high)",
                      color:
                        d.state === "adopted" || d.state === "fresh"
                          ? "var(--md-on-primary-container)"
                          : "var(--md-on-surface-variant)",
                      textDecoration: d.state === "discarded" ? "line-through" : undefined,
                    }}
                  >
                    {d.state}
                  </span>
                  <span className="min-w-0 truncate">{d.text || d.reason}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="md-card md-card-outlined space-y-3 p-4">
            <h2 className="md-title-small">Pipeline</h2>
            <dl className="md-body-small grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
              <dt style={{ color: "var(--md-on-surface-variant)" }}>Hearing</dt>
              <dd>{stt || (status?.stt === "groq" ? "Groq Whisper" : status?.stt ? "Gemini live" : "–")}</dd>
              <dt style={{ color: "var(--md-on-surface-variant)" }}>Turn end</dt>
              <dd>
                Silero + Smart Turn{" "}
                {detector === "loading" && live ? "(loading…)" : detector === "error" ? "(failed)" : ""}
              </dd>
              <dt style={{ color: "var(--md-on-surface-variant)" }}>Thinking</dt>
              <dd className="break-words">{(status?.usable ?? []).join(" → ") || "–"}</dd>
              <dt style={{ color: "var(--md-on-surface-variant)" }}>Voice</dt>
              <dd>{tts || (status?.tts === "live" ? "Gemini Live" : status?.tts ? "Gemini TTS" : "–")}</dd>
            </dl>
          </div>

          <div className="md-card md-card-outlined space-y-3 p-4">
            <h2 className="md-title-small">Settings</h2>
            <label className="md-label-medium block">
              Microphone
              <select
                value={micId}
                disabled={live}
                onChange={(e) => {
                  setMicId(e.target.value);
                  try {
                    localStorage.setItem("duplex.mic", e.target.value);
                  } catch {
                    /* private window */
                  }
                }}
                className="md-body-medium mt-1 w-full rounded-[var(--md-shape-sm)] px-2 py-2"
                style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface)" }}
              >
                <option value="">Automatic (prefers a headset)</option>
                {mics
                  .filter((d) => d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications")
                  .map((d, i) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </option>
                  ))}
              </select>
            </label>
            <label className="md-label-medium block">
              Voice
              <select
                value={voiceName}
                disabled={live}
                onChange={(e) => setVoiceName(e.target.value)}
                className="md-body-medium mt-1 w-full rounded-[var(--md-shape-sm)] px-2 py-2"
                style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface)" }}
              >
                {(status?.voices ?? []).map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label} — {v.character}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <p className="md-label-medium">How long a pause means you are done</p>
              <div
                role="radiogroup"
                className="mt-1 flex gap-1 rounded-[var(--md-shape-full)] p-1"
                style={{ background: "var(--md-surface-container-high)" }}
              >
                {PATIENCE.map((step) => {
                  const on = step.id === patience;
                  return (
                    <button
                      key={step.id}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      disabled={live}
                      title={step.detail}
                      onClick={() => {
                        setPatience(step.id);
                        try {
                          localStorage.setItem("duplex.patience", step.id);
                        } catch {
                          /* private window */
                        }
                      }}
                      className="md-label-small md-state flex-1 rounded-[var(--md-shape-full)] px-1 py-1.5 disabled:cursor-not-allowed"
                      style={{
                        background: on ? "var(--md-secondary-container)" : "transparent",
                        color: on ? "var(--md-on-secondary-container)" : "var(--md-on-surface-variant)",
                      }}
                    >
                      {step.label}
                    </button>
                  );
                })}
              </div>
            </div>
            {live && (
              <Button variant="tonal" size="sm" onClick={() => session.current?.interrupt()}>
                Stop talking
              </Button>
            )}
          </div>
        </aside>
      </div>
      <Dialog
        open={!!openDraft}
        onClose={() => setOpenDraft(null)}
        title={openDraft ? `Draft · ${openDraft.state}` : ""}
        wide
        contentClassName="mt-4"
      >
        <p className="md-body-medium whitespace-pre-wrap break-words">{openDraft?.text || openDraft?.reason}</p>
        {openDraft?.model && (
          <p className="md-label-small mt-3" style={{ color: "var(--md-on-surface-variant)" }}>
            {openDraft.model}
          </p>
        )}
      </Dialog>
    </div>
  );
}

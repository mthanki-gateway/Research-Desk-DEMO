"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type LiveStatus, getLiveStatus } from "@/lib/api";
import { Switch } from "../md";
import {
  IconEarshot,
  IconExternal,
  IconMic,
  IconSearch,
  IconSpinner,
  IconStop,
  IconWave,
} from "../icons";
import { type LiveEvent, type LiveSession, openLiveSession } from "./liveSession";

/**
 * Earshot — audio to audio, natively.
 *
 * WHAT THIS IS NOT ANY MORE
 *
 * It began as a cascade: record a question, transcribe it, answer it with the
 * LangGraph agent, synthesise the answer, play it back. Three models with text
 * in the middle, measured at 12 to 53 seconds a turn.
 *
 * It now talks to a native audio model over one socket. The audio is tokenised
 * into the same sequence the model generates from — there is no transcript in
 * the middle — and speech comes back as it is produced rather than after a
 * complete answer exists. Measured end to end through our own socket, with our
 * own tools against the real corpus: FIRST SOUND AT 2.98 SECONDS.
 *
 * SAME CORPUS, SAME TOOLS. `search_documents`, `search_web`, `list_documents`
 * and `corpus_stats` run server-side through the identical code path the typed
 * agent uses, against the identical Qdrant collection.
 *
 * TURNS ARE STILL TAKEN BY BUTTON. The model can interrupt and be interrupted;
 * it is not asked to. Voice activity detection cannot tell a pause for thought
 * from the end of a question, and an assistant that decides for itself when you
 * have finished will eventually cut you off. The button means both parties
 * always know whose turn it is.
 *
 * WHY A VOICE APP HAS A SCREEN
 *
 * Audio-first, not audio-only on principle. The transcript is shown because
 * when an answer is wrong the first question is always whether it heard the
 * question right. The tool line is shown because the seconds spent searching
 * are silent, and silence is indistinguishable from a crash. The sources are
 * shown because a spoken citation cannot be clicked.
 */

type Phase = "idle" | "connecting" | "listening" | "thinking" | "speaking";

type Source = { label: string; kind: "document" | "web"; url: string | null };

type Exchange = {
  id: string;
  question: string;
  answer: string;
  tools: { tool: string; n: number }[];
  sources: Source[];
};

const EMPTY = { question: "", answer: "", tools: [], sources: [] };

export default function Earshot() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [voiceName, setVoiceName] = useState("");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  /** Keep the socket open between questions — reconnecting costs a second. */
  const [keepOpen, setKeepOpen] = useState(true);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);

  /**
   * The turn in flight.
   *
   * A ref, not state, because transcript fragments arrive many times a second
   * and each one would otherwise be a render scheduled from inside a socket
   * callback. `tick` repaints deliberately instead.
   */
  const live = useRef<Omit<Exchange, "id">>({ ...EMPTY });
  const [tick, setTick] = useState(0);

  const session = useRef<LiveSession | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    getLiveStatus()
      .then((s) => {
        setStatus(s);
        setVoiceName(s.default_voice);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not reach the API"),
      );
  }, []);

  useEffect(
    () => () => {
      session.current?.close();
      if (timer.current) window.clearInterval(timer.current);
    },
    [],
  );

  const commit = useCallback(() => {
    const turn = live.current;
    if (!turn.question && !turn.answer) return;
    setExchanges((prev) => [{ id: `${Date.now()}`, ...turn }, ...prev]);
    live.current = { ...EMPTY, tools: [], sources: [] };
    setTick((n) => n + 1);
  }, []);

  const onEvent = useCallback(
    (event: LiveEvent) => {
      switch (event.type) {
        case "level":
          setLevel(event.level);
          break;
        case "heard":
          // APPENDED, not replaced: each frame carries the next few words, not
          // the whole transcript so far.
          live.current.question += event.text;
          setTick((n) => n + 1);
          break;
        case "said":
          live.current.answer += event.text;
          setPhase("speaking");
          setTick((n) => n + 1);
          break;
        case "tool": {
          live.current.tools.push({ tool: event.tool, n: event.n });
          for (const s of event.sources) {
            const key = `${s.kind}:${s.url ?? s.label}`;
            const known = live.current.sources.some(
              (x) => `${x.kind}:${x.url ?? x.label}` === key,
            );
            if (!known) live.current.sources.push(s);
          }
          setTick((n) => n + 1);
          break;
        }
        case "playback_end":
          setPhase("idle");
          commit();
          break;
        case "turn_end":
          // The model has finished GENERATING. Playback is still draining, so
          // the phase is left alone — `playback_end` ends the turn for the
          // user, and ending it here cuts off the last words.
          break;
        case "error":
          setError(event.detail);
          setPhase("idle");
          break;
        case "closed":
          session.current = null;
          setPhase("idle");
          break;
        default:
          break;
      }
    },
    [commit],
  );

  const start = useCallback(async () => {
    setError(null);
    try {
      if (!session.current) {
        setPhase("connecting");
        session.current = await openLiveSession(voiceName, onEvent);
      }
      await session.current.beginTurn();
      setPhase("listening");
      setElapsed(0);
      // CLEAR BEFORE SETTING. Without this every start left its interval
      // running, so the second turn counted two seconds per second and the
      // third counted three -- which is exactly what it looked like.
      if (timer.current) window.clearInterval(timer.current);
      timer.current = window.setInterval(
        () => setElapsed((n) => n + 1),
        1000,
      ) as unknown as number;
    } catch (e) {
      setPhase("idle");
      session.current = null;
      setError(
        e instanceof Error
          ? e.message
          : "The microphone was refused, or the session could not open.",
      );
    }
  }, [voiceName, onEvent]);

  const stop = useCallback(() => {
    if (timer.current) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setLevel(0);
    session.current?.endTurn();
    setPhase("thinking");
  }, []);

  const hangUp = useCallback(() => {
    session.current?.close();
    session.current = null;
    setPhase("idle");
    commit();
  }, [commit]);

  const busy = phase === "thinking" || phase === "connecting";
  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const inFlight = live.current;

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small flex items-center gap-2">
          <IconEarshot className="h-6 w-6" />
          Earshot
        </h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Speak to a native audio model. Your documents and the web, answered
          out loud — with no transcript in the middle.
        </p>
      </header>

      {status && !status.enabled && (
        <Banner>
          Speech is not configured on this deployment — no Google API key.
        </Banner>
      )}
      {error && <Banner>{error}</Banner>}

      <section className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-10">
        <MicButton
          phase={phase}
          level={level}
          disabled={!status?.enabled}
          onStart={() => void start()}
          onStop={stop}
          onStopSpeaking={() => {
            session.current?.stopSpeaking();
            setPhase("idle");
            commit();
          }}
        />

        <p className="md-title-small text-center">
          {phase === "connecting"
            ? "Opening the session"
            : listening
              ? `Listening — ${elapsed}s. Pause as long as you like; click when you're done.`
              : phase === "thinking"
                ? "Thinking"
                : speaking
                  ? "Speaking — click to stop"
                  : "Click to speak"}
        </p>

        <p
          className="md-body-small text-center"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {listening
            ? "It will not answer until you click. Pausing mid-sentence is fine."
            : status?.web_search
              ? "Your documents and the web."
              : "Your documents. Web search is not configured."}
        </p>

        {(inFlight.question || inFlight.tools.length > 0 || inFlight.answer) && (
          <div className="w-full space-y-2 pt-2" key={tick}>
            {inFlight.question && (
              <p className="md-body-medium text-center font-medium">
                &ldquo;{inFlight.question}&rdquo;
              </p>
            )}
            {inFlight.tools.map((t, i) => (
              <p
                key={`${t.tool}-${i}`}
                className="md-body-small flex items-center justify-center gap-2"
                style={{ color: "var(--md-primary)" }}
              >
                <IconSearch className="h-3.5 w-3.5" />
                {toolLabel(t.tool)}
                {t.n > 0 && ` — ${t.n} result${t.n === 1 ? "" : "s"}`}
              </p>
            ))}
            {inFlight.answer && (
              <p
                className="md-body-medium text-center"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {inFlight.answer}
              </p>
            )}
          </div>
        )}
      </section>

      {status?.enabled && (
        <section
          className="md-card md-card-outlined divide-y"
          style={{ borderColor: "var(--md-outline-variant)" }}
        >
          <Row
            title="Voice"
            detail={
              status.voices.find((v) => v.id === voiceName)?.character ??
              "How the answer sounds"
            }
          >
            <select
              value={voiceName}
              onChange={(e) => {
                setVoiceName(e.target.value);
                // The voice is fixed when the session opens, so a change only
                // takes effect on the next one. Closing here makes that
                // visible rather than silently ignoring the choice.
                session.current?.close();
                session.current = null;
              }}
              disabled={listening || busy}
              aria-label="Voice"
              className="md-body-medium shrink-0 rounded-[var(--md-shape-sm)] px-3 py-2"
              style={{
                background: "var(--md-surface-container-high)",
                color: "var(--md-on-surface)",
                border: "1px solid var(--md-outline-variant)",
              }}
            >
              {status.voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
          </Row>

          <Row
            title="Stay connected between questions"
            detail="Holds the socket open so the next question starts instantly"
          >
            <Switch
              on={keepOpen}
              onChange={(v) => {
                setKeepOpen(v);
                if (!v) hangUp();
              }}
              aria-label="Stay connected between questions"
            />
          </Row>

          <div
            className="md-body-small flex items-center justify-between gap-4 px-4 py-3"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            <span>Model</span>
            <span className="truncate text-right">
              {status.model.replace("models/", "")} · audio in, audio out
            </span>
          </div>
        </section>
      )}

      {exchanges.length === 0 && !inFlight.question ? (
        <p
          className="md-body-medium"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Nothing asked yet. Try &ldquo;what does the engineering handbook say
          about on-call paging?&rdquo;
        </p>
      ) : (
        <ol className="space-y-4">
          {exchanges.map((x) => (
            <Turn key={x.id} exchange={x} />
          ))}
        </ol>
      )}
    </div>
  );
}

function Banner({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
      style={{
        background: "var(--md-error-container)",
        color: "var(--md-on-error-container)",
      }}
    >
      {children}
    </p>
  );
}

function Row({
  title,
  detail,
  children,
}: {
  title: string;
  detail: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 p-4">
      <span className="md-body-medium">
        {title}
        <span
          className="md-body-small mt-0.5 block"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {detail}
        </span>
      </span>
      {children}
    </div>
  );
}

/**
 * The microphone button, which is the entire interface.
 *
 * Large on purpose: used at arm's length, where the standard 40px target is
 * sized for a mouse on a desk. It carries the live input level as a ring,
 * because a recording UI that does not visibly react to sound is
 * indistinguishable from a broken one — and the user finds out only after
 * speaking a whole question into nothing.
 */
function MicButton({
  phase,
  level,
  disabled,
  onStart,
  onStop,
  onStopSpeaking,
}: {
  phase: Phase;
  level: number;
  disabled?: boolean;
  onStart: () => void;
  onStop: () => void;
  onStopSpeaking: () => void;
}) {
  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const busy = phase === "thinking" || phase === "connecting";

  const ring = listening ? Math.min(26, Math.round(level * 90)) : 0;
  const onClick = listening ? onStop : speaking ? onStopSpeaking : onStart;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      aria-label={
        listening
          ? "Stop and send"
          : speaking
            ? "Stop speaking"
            : "Start speaking"
      }
      className="relative grid h-32 w-32 place-items-center rounded-[var(--md-shape-full)] transition-transform active:scale-95 disabled:opacity-60"
      style={{
        background: listening
          ? "var(--md-error)"
          : speaking
            ? "var(--md-tertiary)"
            : "var(--md-primary)",
        color: listening
          ? "var(--md-on-error)"
          : speaking
            ? "var(--md-on-tertiary)"
            : "var(--md-on-primary)",
        // A box-shadow rather than a scaled element, so the ring cannot reflow
        // anything around it as the level moves.
        boxShadow: ring
          ? `0 0 0 ${ring}px color-mix(in srgb, var(--md-error) 22%, transparent)`
          : "var(--md-elev-2)",
      }}
    >
      {busy ? (
        <IconSpinner className="h-12 w-12" />
      ) : listening ? (
        <IconStop className="h-12 w-12" />
      ) : speaking ? (
        <IconWave className="h-12 w-12" />
      ) : (
        <IconMic className="h-12 w-12" />
      )}
    </button>
  );
}

function Turn({ exchange }: { exchange: Exchange }) {
  return (
    <li className="md-card md-card-elevated space-y-3 p-5">
      <p className="md-body-medium flex items-start gap-2">
        <span
          className="mt-0.5 shrink-0"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          <IconMic className="h-4 w-4" />
        </span>
        <span className="font-medium">
          {exchange.question || <em>nothing intelligible</em>}
        </span>
      </p>

      <div
        className="border-t pt-3"
        style={{ borderColor: "var(--md-outline-variant)" }}
      >
        <p className="md-body-medium whitespace-pre-wrap">
          {exchange.answer || <em>no answer</em>}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {exchange.tools.map((t, i) => (
          <span key={`${t.tool}-${i}`} className="md-badge">
            {toolLabel(t.tool)}
          </span>
        ))}
        {exchange.sources.map((s) =>
          s.kind === "web" && s.url ? (
            <a
              key={`web:${s.url}`}
              className="md-badge"
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              title={s.url}
            >
              {s.label}
              <IconExternal className="h-3 w-3 shrink-0 opacity-60" />
            </a>
          ) : (
            <span key={`doc:${s.label}`} className="md-badge">
              {s.label}
            </span>
          ),
        )}
      </div>
    </li>
  );
}

/** Plain English for tools that are named for the code. */
function toolLabel(tool: string): string {
  if (tool === "search_documents") return "searched your documents";
  if (tool === "search_web") return "searched the web";
  if (tool === "list_documents") return "checked your document list";
  if (tool === "corpus_stats") return "checked collection statistics";
  return tool;
}

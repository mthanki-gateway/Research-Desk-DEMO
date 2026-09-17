"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  type LiveStatus,
  type Mode,
  type Profile,
  type ProfileField,
  type ProfileNotes,
  getLiveStatus,
  getParleyConversation,
  getProfileFields,
} from "@/lib/api";
import { useApp } from "../providers";
import { Switch } from "../md";
import {
  IconInterview,
  IconParley,
  IconExternal,
  IconMic,
  IconSearch,
  IconSpinner,
  IconStop,
  IconWave,
} from "../icons";
import { type LiveEvent, type LiveSession, openLiveSession } from "./liveSession";

/**
 * Parley — audio to audio, natively.
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

type Phase =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  /** The model has finished; the microphone reopens shortly unless stopped. */
  | "counting";

/**
 * How long after the answer before listening resumes.
 *
 * A COUNTDOWN RATHER THAN EITHER EXTREME. Reopening the microphone instantly
 * is how the model ends up hearing the room, a cough, or the tail of its own
 * answer through the speakers. Never reopening it makes every follow-up cost a
 * deliberate click, which is exactly the friction a voice interface exists to
 * remove.
 *
 * Five seconds is long enough to read the answer on screen and decide, short
 * enough that a natural follow-up does not need a click at all. It is
 * cancellable, and clicking through it starts listening immediately.
 */
const RESTART_SECONDS = 5;

type Source = { label: string; kind: "document" | "web"; url: string | null };

type Exchange = {
  id: string;
  question: string;
  answer: string;
  tools: { tool: string; n: number }[];
  sources: Source[];
};

const EMPTY = { question: "", answer: "", tools: [], sources: [] };

/**
 * THE SURFACE BOTH MODES SHARE.
 *
 * Speak and Interview are the same app pointed at a different system prompt.
 * Everything else -- the socket, the audio handling, the manual turn
 * boundaries, the tools, the persistence, the resumption, the countdown -- is
 * identical, so it is written once and told which mode it is in.
 *
 * `useSearchParams` opts the tree into client rendering, and Next requires a
 * Suspense boundary around that so the rest of the page can still be
 * prerendered. Without it the build fails outright.
 */
export default function ParleySurface({ mode }: { mode: Mode }) {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-3xl space-y-6 px-6 py-9" aria-hidden>
          <div className="md-skeleton h-8 w-40" />
          <div className="md-skeleton h-[22rem]" />
        </div>
      }
    >
      <Parley mode={mode} />
    </Suspense>
  );
}

/** What each mode calls itself, and what it is for. */
const COPY: Record<Mode, { title: string; blurb: string; hint: string }> = {
  speak: {
    title: "Parley",
    blurb:
      "Speak to a native audio model. Your documents and the web, answered out loud — with no transcript in the middle.",
    hint: "Nothing asked yet. Try “what does the engineering handbook say about on-call paging?”",
  },
  interview: {
    title: "Interview",
    blurb:
      "A spoken interview that builds a profile of the participant. One question at a time, and it follows up on vague answers.",
    hint: "Nothing recorded yet. Press the microphone and it will introduce itself.",
  },
};

function Parley({ mode }: { mode: Mode }) {
  // The drawer owns the conversation LIST; this page owns the conversation.
  // They meet at `?c=<id>`, which is a real URL -- so a conversation can be
  // linked to, reloaded, and reached with the back button.
  const params = useSearchParams();
  const wanted = params.get("c");
  const { refreshParleyConversations } = useApp();

  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [voiceName, setVoiceName] = useState("");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  /** Keep the socket open between questions — reconnecting costs a second. */
  const [keepOpen, setKeepOpen] = useState(true);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  /** Which stored conversation this socket appends to. */
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  /** Interview only: what has been gathered, and what is still missing. */
  const [fields, setFields] = useState<ProfileField[]>([]);
  const [profileData, setProfileData] = useState<Profile>({});
  const [missing, setMissing] = useState<string[]>([]);
  const [profileDone, setProfileDone] = useState(false);

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
  /** The auto-restart countdown, separate from the listening clock. */
  const countdown = useRef<number | null>(null);
  const [remaining, setRemaining] = useState(0);
  /** The conversation survived a reconnect — worth saying once. */
  const [resumed, setResumed] = useState(false);

  useEffect(() => {
    // The field list comes from the server, where the tool schema and the
    // completeness check already live. A fourth copy in TypeScript is the one
    // that would drift.
    if (mode === "interview") getProfileFields().then(setFields).catch(() => {});
  }, [mode]);

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

  // Follow the URL. `openConversation` is defined below and captured through
  // a ref, so the two do not have to be declared in a particular order.
  const openRef = useRef<(id: string) => void>(() => {});
  useEffect(() => {
    if (wanted && wanted !== conversationId) openRef.current(wanted);
  }, [wanted, conversationId]);

  useEffect(
    () => () => {
      session.current?.close();
      if (timer.current) window.clearInterval(timer.current);
      if (countdown.current) window.clearInterval(countdown.current);
    },
    [],
  );

  /**
   * Clear the in-flight display.
   *
   * It no longer COMMITS anything: the server sends the completed exchange as
   * a single `turn` event, at the same moment it writes the rows. The browser
   * used to assemble exchanges itself from streaming fragments, deciding where
   * one turn ended by watching the audio queue drain -- which happens between
   * chunks, so questions were split across cards and a whole spoken sentence
   * could land as one stray word.
   */
  const clearInFlight = useCallback(() => {
    live.current = { question: "", answer: "", tools: [], sources: [] };
    setTick((n) => n + 1);
  }, []);

  const stopCountdown = useCallback(() => {
    if (countdown.current) {
      window.clearInterval(countdown.current);
      countdown.current = null;
    }
    setRemaining(0);
  }, []);

  /**
   * Hand the microphone back after the answer, with a visible delay.
   *
   * Declared as a ref the countdown calls, because `start` is defined below
   * and both refer to each other — the countdown starts listening, and
   * listening cancels any countdown.
   */
  const startRef = useRef<() => void>(() => {});

  const beginCountdown = useCallback(() => {
    stopCountdown();
    setPhase("counting");
    setRemaining(RESTART_SECONDS);
    countdown.current = window.setInterval(() => {
      setRemaining((n) => {
        if (n <= 1) {
          stopCountdown();
          startRef.current();
          return 0;
        }
        return n - 1;
      });
    }, 1000) as unknown as number;
  }, [stopCountdown]);

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
          if (event.profile) {
            setProfileData(event.profile);
            setMissing(event.missing ?? []);
            setProfileDone(Boolean(event.complete));
          }
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
        case "turn":
          // The record, from the one place that knows the boundary.
          if (event.question || event.answer) {
            setExchanges((prev) => [
              {
                id: `${Date.now()}`,
                question: event.question,
                answer: event.answer,
                sources: event.sources,
                tools: event.tools.map((tool) => ({ tool, n: 0 })),
              },
              ...prev,
            ]);
          }
          clearInFlight();
          // The rows are written, so the list's titles and counts are stale.
          void refreshParleyConversations(mode);
          break;
        case "playback_end":
          // Only the countdown. The exchange was stored when `turn` arrived;
          // this is purely "the speakers have gone quiet".
          beginCountdown();
          break;
        case "turn_end":
          // The model has finished GENERATING. Playback is still draining, so
          // the phase is left alone — `playback_end` ends the turn for the
          // user, and ending it here cuts off the last words.
          break;
        case "ready":
          setResumed(event.resumed);
          // The server decides which conversation this is -- it may have
          // created one. Adopting its answer keeps the two in step.
          setConversationId(event.session_id);
          break;
        case "going_away":
          // The server has announced its own disconnection. Dropping the
          // socket NOW, while a resume handle is held, turns a dying session
          // into an invisible reconnect — the alternative is losing the
          // conversation mid-sentence with no warning to the user.
          session.current?.close();
          session.current = null;
          break;
        case "error":
          setError(event.detail);
          stopCountdown();
          setPhase("idle");
          break;
        case "closed":
          session.current = null;
          // Only fall back to idle if nothing is in flight. A close that
          // arrives while the countdown is running is the idle timeout doing
          // its job, and interrupting the countdown for it would be wrong.
          setPhase((p) => (p === "counting" ? p : "idle"));
          break;
        default:
          break;
      }
    },
    [clearInFlight, beginCountdown, stopCountdown, refreshParleyConversations],
  );

  const start = useCallback(async () => {
    setError(null);
    // Clicking through the countdown starts listening NOW. The countdown is a
    // convenience, never a thing to wait out.
    stopCountdown();
    try {
      if (!session.current) {
        setPhase("connecting");
        session.current = await openLiveSession(
          voiceName,
          conversationId,
          mode,
          onEvent,
        );
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
  }, [voiceName, conversationId, mode, onEvent, stopCountdown]);

  // The countdown calls whatever `start` currently is, without either of them
  // having to be declared before the other.
  useEffect(() => {
    startRef.current = () => void start();
  }, [start]);

  useEffect(() => {
    openRef.current = (id: string) => void openConversation(id);
  });

  const stop = useCallback(() => {
    if (timer.current) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setLevel(0);
    session.current?.endTurn();
    setPhase("thinking");
  }, []);

  /** Leave this conversation intact and begin a new one. */
  const startFresh = useCallback(() => {
    stopCountdown();
    clearInFlight();
    session.current?.close();
    session.current = null;
    setConversationId(null);
    setExchanges([]);
    setProfileData({});
    setMissing([]);
    setProfileDone(false);
    setResumed(false);
    setPhase("idle");
  }, [clearInFlight, stopCountdown]);

  /** Open a stored conversation and continue it. */
  const openConversation = useCallback(
    async (id: string) => {
      stopCountdown();
      session.current?.close();
      session.current = null;
      setPhase("idle");
      setLoadingHistory(true);
      setError(null);
      try {
        const detail = await getParleyConversation(id);
        setConversationId(id);
        setProfileData(detail.profile ?? {});
        setMissing(detail.missing ?? []);
        setProfileDone(Boolean(detail.complete));
        // Newest first, matching the live view -- the turn just spoken should
        // be the one under the button, not buried at the bottom.
        setExchanges(
          detail.turns
            .map((t, i) => ({
              id: `${id}:${i}`,
              question: t.question,
              answer: t.answer,
              sources: t.sources,
              // Restored turns carry tool NAMES only; the hit counts were live
              // telemetry and are not stored. 0 renders as a bare label.
              tools: t.tools.map((tool) => ({ tool, n: 0 })),
            }))
            .reverse(),
        );
        setResumed(false);
        if (!detail.resumable) {
          setError(
            "This conversation can be read, but its live context has expired — " +
              "the assistant will not remember what was said before.",
          );
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not open that conversation");
      } finally {
        setLoadingHistory(false);
      }
    },
    [stopCountdown],
  );

  const busy = phase === "thinking" || phase === "connecting";
  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const counting = phase === "counting";
  const inFlight = live.current;

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small flex items-center gap-2">
          {mode === "interview" ? (
            <IconInterview className="h-6 w-6" />
          ) : (
            <IconParley className="h-6 w-6" />
          )}
          {COPY[mode].title}
        </h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {COPY[mode].blurb}
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
          remaining={remaining}
          disabled={!status?.enabled}
          onStart={() => void start()}
          onStop={stop}
          onStopSpeaking={() => {
            // The exchange is already stored; stopping playback only ends the
            // sound, so nothing is lost by cutting it off.
            session.current?.stopSpeaking();
            beginCountdown();
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
                  : counting
                    ? `Listening again in ${remaining}…`
                    : "Click to speak"}
        </p>

        <p
          className="md-body-small text-center"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {listening
            ? "It will not answer until you click. Pausing mid-sentence is fine."
            : speaking
              ? "Nothing is being sent while it speaks."
            : counting
              ? "Click to start now, or stay quiet to cancel."
              : status?.web_search
                ? "Your documents and the web."
                : "Your documents. Web search is not configured."}
        </p>

        {counting && (
          <button
            type="button"
            onClick={() => {
              stopCountdown();
              setPhase("idle");
            }}
            className="md-label-large rounded-[var(--md-shape-full)] px-4 py-2"
            style={{
              background: "var(--md-surface-container-high)",
              color: "var(--md-on-surface)",
            }}
          >
            Stay quiet
          </button>
        )}

        {resumed && (
          <p
            className="md-body-small text-center"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Reconnected — the earlier conversation was restored.
          </p>
        )}

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

      {/* The settings pane only exists once /live/status has answered, so the
          page previously grew by a whole card a moment after it painted —
          pushing the transcript down and making the layout jump under the
          reader. A skeleton of the same height holds the space. */}
      {!status && !error && (
        <section
          className="md-card md-card-outlined divide-y"
          style={{ borderColor: "var(--md-outline-variant)" }}
          aria-hidden
        >
          <div className="flex items-center justify-between gap-4 p-4">
            <span className="w-full space-y-2">
              <span className="md-skeleton block h-4 w-24" />
              <span className="md-skeleton block h-3 w-52" />
            </span>
            <span className="md-skeleton block h-9 w-24 shrink-0" />
          </div>
          <div className="flex items-center justify-between gap-4 p-4">
            <span className="w-full space-y-2">
              <span className="md-skeleton block h-4 w-56" />
              <span className="md-skeleton block h-3 w-72" />
            </span>
            <span className="md-skeleton block h-7 w-12 shrink-0" />
          </div>
          <div className="flex items-center justify-between gap-4 px-4 py-3">
            <span className="md-skeleton block h-3 w-16" />
            <span className="md-skeleton block h-3 w-48" />
          </div>
        </section>
      )}

      {mode === "interview" && fields.length > 0 && (
        <ProfileCard
          fields={fields}
          profile={profileData}
          missing={missing}
          complete={profileDone}
        />
      )}

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
                if (!v) {
                  session.current?.close();
                  session.current = null;
                }
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

      {/* ---- the conversation, and the ones before it -------------------- */}
      {loadingHistory ? (
        <div className="space-y-3" aria-hidden>
          {[0, 1].map((i) => (
            <div key={i} className="md-skeleton h-[104px]" />
          ))}
        </div>
      ) : !status && !error ? (
        <div className="space-y-2" aria-hidden>
          <span className="md-skeleton block h-4 w-80" />
        </div>
      ) : exchanges.length === 0 && !inFlight.question ? (
        <p
          className="md-body-medium"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {COPY[mode].hint}
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
  remaining,
  disabled,
  onStart,
  onStop,
  onStopSpeaking,
}: {
  phase: Phase;
  level: number;
  remaining: number;
  disabled?: boolean;
  onStart: () => void;
  onStop: () => void;
  onStopSpeaking: () => void;
}) {
  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const counting = phase === "counting";
  const busy = phase === "thinking" || phase === "connecting";

  const ring = listening ? Math.min(26, Math.round(level * 90)) : 0;
  // During the countdown the button is the SAME button and does the same
  // thing it does from idle — start listening. Clicking through is the fast
  // path, not a special case.
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
            : counting
              ? `Start listening now (otherwise in ${remaining} seconds)`
              : "Start speaking"
      }
      className="relative grid h-32 w-32 place-items-center rounded-[var(--md-shape-full)] transition-transform active:scale-95 disabled:opacity-60"
      style={{
        background: listening
          ? "var(--md-error)"
          : speaking
            ? "var(--md-tertiary)"
            : counting
              ? "var(--md-secondary-container)"
              : "var(--md-primary)",
        color: listening
          ? "var(--md-on-error)"
          : speaking
            ? "var(--md-on-tertiary)"
            : counting
              ? "var(--md-on-secondary-container)"
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
      ) : counting ? (
        <span className="md-headline-small tabular-nums">{remaining}</span>
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


/**
 * What the interview has gathered, filling in as it goes.
 *
 * Shown because a conversation whose PRODUCT is a profile should show the
 * profile. The alternative is a transcript and a promise, where the only way
 * to know whether anything was captured is to finish and go looking.
 *
 * Required fields are listed even when empty, so the remaining work is
 * visible; optional ones appear only once they have something in them, because
 * a permanent row of blanks reads as a form that was abandoned.
 */
function ProfileCard({
  fields,
  profile,
  missing,
  complete,
}: {
  fields: ProfileField[];
  profile: Profile;
  missing: string[];
  complete: boolean;
}) {
  const shown = fields.filter(
    (f) => f.name !== "notes" && (f.required || profile[f.name] !== undefined),
  );
  const filled = fields.filter((f) => f.required && !missing.includes(f.name));
  // What was notable about HOW each answer was given. Keyed by field, with
  // `general` for anything about the person rather than one answer.
  const notes = (profile.notes ?? {}) as ProfileNotes;
  const general = notes.general ?? [];

  return (
    <section className="md-card md-card-outlined p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="md-title-medium">Profile</h2>
        <span
          className="md-label-medium rounded-[var(--md-shape-full)] px-2.5 py-1"
          style={{
            background: complete
              ? "var(--md-secondary-container)"
              : "var(--md-surface-container-high)",
            color: complete
              ? "var(--md-on-secondary-container)"
              : "var(--md-on-surface-variant)",
          }}
        >
          {complete
            ? "Complete"
            : `${filled.length} of ${fields.filter((f) => f.required).length}`}
        </span>
      </div>

      <dl className="space-y-2">
        {shown.map((f) => {
          const value = profile[f.name] as string | number | string[] | undefined;
          const empty = value === undefined || value === "";
          return (
            <div key={f.name} className="flex gap-3">
              <dt
                className="md-body-small w-40 shrink-0"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {label(f.name)}
              </dt>
              <dd className="md-body-medium min-w-0 flex-1">
                {empty ? (
                  <span style={{ color: "var(--md-on-surface-variant)" }}>
                    &mdash;
                  </span>
                ) : Array.isArray(value) ? (
                  <span className="flex flex-wrap gap-1.5">
                    {value.map((v) => (
                      <span key={v} className="md-badge">
                        {v}
                      </span>
                    ))}
                  </span>
                ) : (
                  String(value)
                )}
                {/* The note sits UNDER its answer, not in a separate block.
                    "hybrid" and "firm about it, mentioned a long commute" are
                    one fact; separating them leaves a table of values and a
                    pile of orphaned observations. */}
                {(notes[f.name] ?? []).map((note) => (
                  <span
                    key={note}
                    className="md-body-small mt-1 block italic"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    {note}
                  </span>
                ))}
              </dd>
            </div>
          );
        })}
      </dl>

      {general.length > 0 && (
        <div
          className="mt-4 border-t pt-3"
          style={{ borderColor: "var(--md-outline-variant)" }}
        >
          <p
            className="md-label-medium mb-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Impressions
          </p>
          {general.map((note) => (
            <p
              key={note}
              className="md-body-small italic"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {note}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

/** `years_experience` -> "Years experience". The API names fields for code. */
function label(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

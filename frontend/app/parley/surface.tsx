"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
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
import { ConfirmButton, Switch } from "../md";
import {
  DEFAULT_PATIENCE,
  PATIENCE,
  type Patience,
  disableTurnDetection,
  enableTurnDetection,
} from "./turnDetector";
import {
  IconHowler,
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

export type Phase =
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
 * Two seconds. Five felt like waiting for permission, and three still did once
 * the round trip is counted -- the gap the user experiences is always longer
 * than the number, because the countdown starts after playback drains rather
 * than when the model stops generating.
 * Still cancellable, and clicking through it starts listening immediately.
 */
const RESTART_SECONDS = 2;

type Source = { label: string; kind: "document" | "web"; url: string | null };

type Exchange = {
  id: string;
  answer: string;
  tools: { tool: string; n: number }[];
  sources: Source[];
  /** What was captured from the participant's answer. */
  recorded: Record<string, unknown>[];
};

const EMPTY = { answer: "", tools: [], sources: [], recorded: [] };

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

/**
 * What each mode calls itself, and what it is for.
 *
 * EVERY user-facing string that differs by mode lives here, and that is the
 * point of the shape. The version before this branched inline on
 * `mode === "interview"`, so Howler silently inherited Parley's half of every
 * ternary -- it told people it would "answer from your documents and the web",
 * which is neither what it does nor something it has the tools for.
 */
const COPY: Record<
  Mode,
  {
    title: string;
    blurb: string;
    hint: string;
    /** The button that opens a session, before anything has been said. */
    start: string;
    /** The button that abandons this one and begins another. */
    fresh: string;
    /** Under the button, at rest and once under way. */
    idle: string;
    idleHint: string;
    running: string;
  }
> = {
  speak: {
    title: "Parley",
    blurb:
      "Speak to a native audio model. Your documents and the web, answered out loud — with no transcript in the middle.",
    hint: "Nothing asked yet. Try “what does the engineering handbook say about on-call paging?”",
    start: "Start session",
    fresh: "New conversation",
    idle: "Start a conversation",
    idleHint: "It will listen, then answer from your documents and the web.",
    running: "Your documents and the web.",
  },
  howler: {
    title: "Howler",
    blurb:
      "An interview against the data points you asked for. It stops when it has them.",
    hint: "Ready when you are. It will introduce itself and ask the first question.",
    start: "Start interview",
    // Not "new interview": Howler cannot start blank, because a session with
    // no schema has nothing to fill. `startFresh` sends you back to the
    // project, and the label says so rather than implying a blank one.
    fresh: "Back to project",
    idle: "Ready when you are",
    idleHint: "It will introduce itself and ask the first question.",
    running: "It asks; you answer. It stops once it has the data points.",
  },
  interview: {
    title: "Interview",
    blurb:
      "A spoken interview that builds a profile of the participant. One question at a time, and it follows up on vague answers.",
    hint: "Nothing recorded yet. Press the microphone and it will introduce itself.",
    start: "Start interview",
    fresh: "New interview",
    idle: "Ready when you are",
    idleHint: "It will introduce itself and ask the first question.",
    running: "It will ask; you answer. It stops when it has what it needs.",
  },
};

function Parley({ mode }: { mode: Mode }) {
  // The drawer owns the conversation LIST; this page owns the conversation.
  // They meet at `?c=<id>`, which is a real URL -- so a conversation can be
  // linked to, reloaded, and reached with the back button.
  const params = useSearchParams();
  const wanted = params.get("c");
  const router = useRouter();
  const pathname = usePathname();
  const { refreshParleyConversations } = useApp();

  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [voiceName, setVoiceName] = useState("");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  /** Keep the socket open between questions — reconnecting costs a second. */
  const [keepOpen, setKeepOpen] = useState(true);
  /**
   * Let the local models decide when a turn is over.
   *
   * OFF BY DEFAULT, and Speak only. The button owning turns is the behaviour
   * this app deliberately chose -- people pause mid-sentence and energy VAD
   * cuts them off -- so this is an opt-in on top of it, not a replacement for
   * it. Remembered per browser, because it is a preference about how somebody
   * likes to talk rather than anything about the conversation.
   */
  const [autoTurns, setAutoTurns] = useState(false);
  const [detector, setDetector] = useState<"off" | "loading" | "ready" | "failed">(
    "off",
  );
  /** How long to wait before deciding somebody has finished. */
  const [patience, setPatienceStep] = useState<Patience>(DEFAULT_PATIENCE);
  /** Why the last turn ended itself, shown once so it is not a mystery. */
  const [endedBy, setEndedBy] = useState<string | null>(null);
  /**
   * The conversation is held: no turn will open by itself.
   *
   * Distinct from `ended`, which is the interviewer closing an interview for
   * good. This is "stop for a moment" -- the socket stays open, the
   * conversation keeps its context, and pressing the microphone carries on
   * where it left off. With auto-turns running there was otherwise no way out
   * of the loop at all: every answer reopened the microphone, and the only
   * exit was leaving the page.
   */
  const [held, setHeld] = useState(false);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  /** Which stored conversation this socket appends to. */
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  /** Interview only: what has been gathered, and what is still missing. */
  const [fields, setFields] = useState<ProfileField[]>([]);
  const [profileData, setProfileData] = useState<Profile>({});
  const [missing, setMissing] = useState<string[]>([]);
  const [profileDone, setProfileDone] = useState(false);
  /** The model has called `end_interview`. The microphone stops reopening. */
  const [ended, setEnded] = useState(false);
  /** Howler only: the project this conversation came from, so reading a result
   *  has a way back to the tabs it was opened from. */
  const [projectId, setProjectId] = useState<string | null>(null);
  /** Ended by the PARTICIPANT rather than the interviewer -- two different
   *  things to say afterwards, and two different findings. */
  const [endedByUser, setEndedByUser] = useState(false);
  /**
   * Nothing has happened in this conversation yet.
   *
   * The entry point differs by mode. Speak is a tool: you arrive with a
   * question and press the microphone. An interview is a conversation someone
   * is being taken through, and being handed a live microphone with no idea
   * what is wanted is the wrong way into one -- so it opens with the model
   * introducing itself.
   */
  const [started, setStarted] = useState(false);

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
  const endedRef = useRef(false);
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
    // Interview's fields are fixed and served from the backend. HOWLER'S ARE
    // NOT: they were generated from a brief and stored on the conversation, so
    // they arrive with it in `openConversation` below. Fetching a global list
    // for Howler would render the wrong schema entirely.
    if (mode === "interview") getProfileFields().then(setFields).catch(() => {});
  }, [mode]);

  // Read after mount, not in the initialiser: Next renders this on the server
  // for the first HTML, where `localStorage` does not exist.
  useEffect(() => {
    try {
      setAutoTurns(localStorage.getItem("parley.autoTurns") === "1");
      const saved = localStorage.getItem("parley.patience");
      if (PATIENCE.some((p) => p.id === saved)) {
        setPatienceStep(saved as Patience);
      }
    } catch {
      /* private window; the defaults stand */
    }
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
    live.current = { answer: "", tools: [], sources: [], recorded: [] };
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
  /** `stop` is defined below this callback; the ref lets them refer to each
   *  other without either having to be declared first. */
  const stopRef = useRef<() => void>(() => {});
  /** Read inside the socket callback, which was captured when the socket
   *  opened -- a plain `autoTurns` there is whatever it was at that moment. */
  const autoRef = useRef(false);
  /** Read in the socket callback, for the same reason as `autoRef`. */
  const heldRef = useRef(false);

  useEffect(() => {
    endedRef.current = ended;
    // A finished interview is not listening to anything. The socket is left
    // open -- the transcript and profile are still being written -- but the
    // microphone goes back, or the recording indicator stays lit over a
    // conversation that is plainly over.
    if (ended) session.current?.releaseMic();
  }, [ended]);

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
          // DELIBERATELY IGNORED.
          //
          // `input_transcription` is a separate, lossier pass than the model's
          // own understanding -- it rendered a participant saying their name
          // was John as "madre es un", in a turn the model answered with
          // "Thanks, John". Showing it next to a correct answer does not
          // merely look wrong, it looks authoritative and wrong, and invites
          // the reader to doubt an answer that was right.
          //
          // What the participant said is captured accurately elsewhere: by
          // the model itself, as quotes and notes on the profile.
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
          if (event.ended) {
            // The model decided the conversation is over. Distinct from "the
            // required fields are filled" -- it fills the last one, then asks
            // whether there is anything to add, and that answer is often the
            // most useful thing in the profile.
            setEnded(true);
            stopCountdown();
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
          if (event.answer) {
            setExchanges((prev) => [
              {
                id: `${Date.now()}`,
                answer: event.answer,
                sources: event.sources,
                tools: event.tools.map((tool) => ({ tool, n: 0 })),
                recorded: event.recorded ?? [],
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
          //
          // Read through a ref rather than the state value: this callback is
          // captured when the socket opens, so a plain `ended` here would be
          // whatever it was at that moment -- always false.
          // NO COUNTDOWN WHEN THE DETECTOR IS RUNNING. The countdown exists
          // to give somebody a moment before the microphone reopens, because
          // reopening it starts a turn that only the button can end. With
          // auto-turns on that is no longer true -- the turn does not end
          // until they have actually spoken and then paused -- so the delay
          // is dead time, and a counter ticking down next to a microphone
          // that is about to decide for itself reads as two things competing.
          if (endedRef.current) break;
          // HELD BEATS EVERYTHING. Somebody who asked for a pause gets one,
          // whether the next turn would have come from the detector or from
          // the countdown.
          if (heldRef.current) {
            setPhase("idle");
            break;
          }
          if (autoRef.current) startRef.current();
          else beginCountdown();
          break;
        case "finished":
          // The PARTICIPANT ended it. Same closed state the model's
          // `end_interview` produces, reached by a different decision -- and
          // the server records which, because a half-filled profile somebody
          // walked out of is a different finding from one the interviewer
          // could not get answers to.
          setEnded(true);
          setEndedByUser(true);
          stopCountdown();
          setPhase("idle");
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

  /**
   * Open the conversation with the model speaking.
   *
   * It connects and asks the model to introduce itself; the microphone opens
   * afterwards through the ordinary countdown, once it has stopped talking.
   * Nothing special happens at the end of that first turn -- it is a turn like
   * any other, which is why there is no separate state for "greeting".
   */
  const startSession = useCallback(async () => {
    setError(null);
    stopCountdown();
    try {
      setPhase("connecting");
      if (!session.current) {
        session.current = await openLiveSession(
          voiceName,
          conversationId,
          mode,
          onEvent,
        );
      }
      setStarted(true);
      session.current.greet();
      setPhase("thinking");
    } catch (e) {
      setPhase("idle");
      session.current = null;
      setError(
        e instanceof Error ? e.message : "The session could not be opened.",
      );
    }
  }, [voiceName, conversationId, mode, onEvent, stopCountdown]);

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
      setStarted(true);
      setEndedBy(null);
      // Pressing the microphone IS the resume. A separate Resume button next
      // to a microphone that already means "talk to me now" is two controls
      // for one intention.
      setHeld(false);
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

  /**
   * Own the detector directly, not through the socket.
   *
   * It loads when the SETTING is switched on, which is nearly always before
   * anybody presses Start -- so anything that waited for a live session sat on
   * "fetching the models" for ever, having never begun.
   */
  useEffect(() => {
    if (!autoTurns) {
      disableTurnDetection();
      return;
    }
    enableTurnDetection((event) => {
      if (event.type === "loading") return setDetector("loading");
      if (event.type === "ready") return setDetector("ready");
      if (event.type === "error") {
        setDetector("failed");
        setError(event.detail);
        return;
      }
      // THE SURFACE DECIDES, not the detector. It reports that somebody sounds
      // finished; ending the turn is the same action the button takes, routed
      // through the same place, so there is one answer to "what closed this
      // turn".
      //
      // Only while LISTENING. A report arriving during playback, or after the
      // button was already pressed, is about audio that is no longer a turn.
      setPhase((p) => {
        if (p !== "listening") return p;
        setEndedBy(
          event.reason === "timeout"
            ? "Ended after a long pause"
            : `Ended — sounded finished (${Math.round(event.probability * 100)}%)`,
        );
        stopRef.current();
        return p;
      });
    }, patience);
  }, [autoTurns, mode, patience]);

  const stop = useCallback(() => {
    if (timer.current) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setLevel(0);
    session.current?.endTurn();
    setPhase("thinking");
  }, []);

  useEffect(() => {
    stopRef.current = () => stop();
  }, [stop]);

  useEffect(() => {
    autoRef.current = autoTurns && detector === "ready";
  }, [mode, autoTurns, detector]);

  useEffect(() => {
    heldRef.current = held;
  }, [held]);

  /** Leave this conversation intact and begin a new one. */
  /**
   * Stop for now, without ending anything.
   *
   * What happens depends on where the turn is, because "pause" means different
   * things mid-sentence and mid-answer:
   *
   *   listening  commit what was said and let it answer -- discarding audio
   *              somebody has already spoken is the one outcome nobody wants
   *   speaking   cut the answer off, the same as clicking the mic would
   *   otherwise  simply stop the clock
   *
   * In every case the microphone does not reopen afterwards.
   */
  const hold = useCallback(() => {
    setHeld(true);
    heldRef.current = true;
    stopCountdown();
    if (phase === "listening") {
      stopRef.current();
      return;
    }
    if (phase === "speaking") session.current?.stopSpeaking();
    // Give the device back. The socket stays open so resuming is instant, but
    // a paused conversation must not leave the browser's recording indicator
    // lit -- a tab that looks like it is still listening, while the user
    // believes they paused it, is the worst version of this.
    session.current?.releaseMic();
    setPhase("idle");
  }, [phase, stopCountdown]);

  const startFresh = useCallback(() => {
    // Howler cannot start blank -- a session without a schema has nothing to
    // fill -- so "new" means going back to the project that defined one, and
    // to its Results tab, because reading a result is why you are here.
    //
    // Falls back to the project LIST only when the conversation has no project
    // -- which is a session created before projects existed.
    if (mode === "howler") {
      router.push(
        projectId
          ? `/parley/howler?p=${projectId}&tab=results`
          : "/parley/howler",
      );
      return;
    }
    // CLEAR THE URL FIRST, and this is the whole bug it fixes.
    //
    // Which conversation is open lives in `?c=<id>`, and an effect below
    // follows it. Resetting the state without clearing the parameter left the
    // id in the URL, so that effect immediately reopened the very conversation
    // that had just been closed -- "New interview" appeared to do nothing at
    // all, every time, once one had been opened from the drawer.
    if (wanted) router.replace(pathname);

    stopCountdown();
    clearInFlight();
    session.current?.close();
    session.current = null;
    setConversationId(null);
    setExchanges([]);
    setProfileData({});
    setMissing([]);
    setProfileDone(false);
    setEnded(false);
    setProjectId(null);
    setEndedByUser(false);
    setStarted(false);
    setResumed(false);
    setHeld(false);
    setPhase("idle");
  }, [clearInFlight, stopCountdown, wanted, router, pathname, mode, projectId]);

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
        if (mode === "howler") {
          setFields(
            (detail.fields ?? []).map((f) => ({
              name: f.name,
              required: f.required,
              kind: f.type,
            })),
          );
        }
        setProfileData(detail.profile ?? {});
        setMissing(detail.missing ?? []);
        setProfileDone(Boolean(detail.complete));
        setEnded(Boolean((detail.profile ?? {}).ended));
        setProjectId(detail.project_id ?? null);
        setEndedByUser(
          (detail.profile ?? {}).ended_by === "participant",
        );
        // Reopening a conversation that already has turns: it has plainly
        // started, and greeting again would have it introduce itself to
        // someone it has been talking to for ten minutes.
        setStarted(detail.turns.length > 0);
        // Newest first, matching the live view -- the turn just spoken should
        // be the one under the button, not buried at the bottom.
        setExchanges(
          detail.turns
            .filter((t) => t.answer)
            .map((t, i) => ({
              id: `${id}:${i}`,
              answer: t.answer,
              sources: t.sources,
              // Not stored per turn; the profile card carries the whole of it.
              recorded: [],
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
  // Auto-turns is on but its models have not arrived yet.
  const loadingModels = autoTurns && detector === "loading";
  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const counting = phase === "counting";
  const inFlight = live.current;

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header className="flex items-start justify-between gap-4">
        <div>
        <h1 className="md-headline-small flex items-center gap-2">
          {mode === "interview" ? (
            <IconInterview className="h-6 w-6" />
          ) : mode === "howler" ? (
            <IconHowler className="h-6 w-6" />
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
        </div>
        {/* Starting a fresh one is the second most common action here, after
            speaking. It is in the drawer too, but the drawer is collapsible
            and can be closed. */}
        <button
          type="button"
          onClick={startFresh}
          className="md-label-large shrink-0 rounded-[var(--md-shape-full)] px-4 py-2"
          style={{
            background: "var(--md-secondary-container)",
            color: "var(--md-on-secondary-container)",
          }}
        >
          {COPY[mode].fresh}
        </button>
      </header>

      {status && !status.enabled && (
        <Banner>
          Speech is not configured on this deployment — no Google API key.
        </Banner>
      )}
      {error && <Banner>{error}</Banner>}

      <section className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-10">
        {!started && phase !== "connecting" && phase !== "thinking" ? (
          <button
            type="button"
            onClick={() => void startSession()}
            // HELD WHILE THE MODELS ARE STILL ARRIVING. Starting now would open
            // the microphone with auto-turns switched on and nothing able to
            // detect a turn, so the first thing somebody said would be
            // answered only when they gave up and pressed the button -- which
            // reads as the setting not working.
            disabled={!status?.enabled || loadingModels}
            className="md-label-large flex items-center gap-2 rounded-[var(--md-shape-full)] px-8 py-4 disabled:opacity-60"
            style={{
              background: "var(--md-primary)",
              color: "var(--md-on-primary)",
              boxShadow: "var(--md-elev-2)",
            }}
          >
            {loadingModels && <IconSpinner className="h-4 w-4" />}
            {loadingModels ? "Loading turn detection" : COPY[mode].start}
          </button>
        ) : (
        <MicButton
          phase={phase}
          level={level}
          remaining={remaining}
          // Once the model has closed the interview there is nothing to say to
          // it. Leaving the button live invites a question that reopens a
          // conversation the model has already finished.
          disabled={!status?.enabled || ended || loadingModels}
          onStart={() => void start()}
          onStop={stop}
          onStopSpeaking={() => {
            // The exchange is already stored; stopping playback only ends the
            // sound, so nothing is lost by cutting it off.
            session.current?.stopSpeaking();
            if (autoTurns && detector === "ready") void start();
            else beginCountdown();
          }}
        />
        )}

        <p className="md-title-small text-center">
          {held && phase === "idle"
            ? "Paused"
            : !started && phase === "idle"
            ? COPY[mode].idle
            : ended
            ? "Interview complete"
            : phase === "connecting"
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
          {held && phase === "idle"
            ? "Nothing is being recorded. Press the microphone to carry on where you left off."
            : !started && phase === "idle"
            ? COPY[mode].idleHint
            : ended
            ? endedByUser
              ? "You ended it. Everything said so far is kept."
              : "The interviewer has everything it needs. Start a new one to go again."
            : endedBy && phase === "thinking"
              ? endedBy
            : listening
              ? autoTurns && detector === "ready"
                ? "Click when you are done, or pause and it will work it out."
                : "It will not answer until you click. Pausing mid-sentence is fine."
            : speaking
              ? "Nothing is being sent while it speaks."
              : counting
                ? "Click to start now, or stay quiet to cancel."
                : mode === "speak" && !status?.web_search
                  ? "Your documents. Web search is not configured."
                  : COPY[mode].running}
        </p>

        {/* THE WAY OUT. Shown while a turn is in flight, because that is
            exactly when there is something to stop -- and with auto-turns on,
            every answer reopened the microphone with no exit but leaving the
            page. Quiet, and no confirmation: a pause is undone by pressing
            the microphone.

            Not during the countdown: "Stay quiet" below already cancels that,
            and the countdown only runs when auto-turns is OFF, so there is no
            loop to escape from there. Two buttons for one intention is
            clutter. */}
        {started && !ended && phase !== "idle" && phase !== "counting" && (
          <button
            type="button"
            onClick={hold}
            className="md-label-large rounded-[var(--md-shape-full)] px-4 py-2"
            style={{
              background: "var(--md-surface-container-high)",
              color: "var(--md-on-surface)",
            }}
          >
            {/* Named for what it DOES to the words already spoken. "Stop"
                over a live microphone reads as "discard this", and throwing
                away a sentence somebody just finished is the one outcome
                nobody wants. */}
            {listening ? "Send and stop" : "Stop for now"}
          </button>
        )}

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

        {/* ENDING IT YOURSELF.
            Offered once the conversation is under way and not yet closed. It
            is deliberately quiet -- a text button, below everything -- because
            it is a way out rather than a thing to do, and a prominent one next
            to the microphone invites a mis-click that cannot be undone. The
            confirmation is what makes it safe to have at all. */}
        {started && !ended && mode !== "speak" && (
          <ConfirmButton
            label="End the interview"
            title="End the interview?"
            // Says what actually happens, including the part people will not
            // guess: it is over for good, and anything already gathered is
            // kept rather than thrown away.
            body="It closes now, and whoever is being interviewed cannot reopen it. Everything said so far is kept."
            confirmLabel="End it"
            onConfirm={() => {
              session.current?.finish();
              stopCountdown();
              setEnded(true);
              setPhase("idle");
            }}
          />
        )}

        {resumed && (
          <p
            className="md-body-small text-center"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Reconnected — the earlier conversation was restored.
          </p>
        )}

        {(inFlight.tools.length > 0 || inFlight.answer) && (
          <div className="w-full space-y-2 pt-2" key={tick}>
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
              <div className="text-center">
                <p
                  className="md-label-medium"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  AI Assistant
                </p>
                <p
                  className="md-body-medium"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {inFlight.answer}
                </p>
              </div>
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

      {mode !== "speak" && fields.length > 0 && (
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

          {/* SPEAK ONLY, for now. Interview and Howler are conversations
              somebody is being taken through, and the cost of cutting a
              participant off mid-answer is higher there than the convenience
              is worth -- so this gets proven here first. */}
          {/* ONE CHILD OF THE SECTION, not two. The section draws its
              dividers with `divide-y`, so a sibling block would get a rule
              above it -- and the negative margin that pulled the stepper up
              under the switch dragged its label onto that rule, which is the
              line running through the text in the screenshot. Nested, there
              is no divider to collide with and the pair reads as one setting,
              which is what they are. */}
          {/* EVERY MODE THIS SURFACE RUNS.
              It began as Speak-only, because Interview and Howler take
              somebody THROUGH a conversation, and cutting a participant off
              mid-answer costs more there than the convenience is worth. Speak
              was simply where that risk was cheapest to carry while the
              thresholds were wrong -- and they were, twice.

              The GUEST page is still excluded, and that is a separate decision
              rather than an oversight: somebody on a magic link has no
              settings to read, so enabling it for them would be the operator
              choosing on their behalf. That belongs on the project, not on a
              toggle they never see. */}
          <div className="[&>div:first-child]:pb-3">
              <Row
                title="End my turn automatically"
                detail={
                  detector === "failed"
                    ? "The detector could not load — the button still works"
                    : detector === "loading" && autoTurns
                      ? "Fetching the models (~11MB, once)"
                      : autoTurns
                        ? "Two local models listen for the end of a sentence. Nothing is uploaded"
                        : mode === "speak"
                          ? "Off — the button decides when you have finished"
                          : "Off — the button decides when an answer is finished"
                }
              >
                <Switch
                  on={autoTurns}
                  onChange={(v) => {
                    setAutoTurns(v);
                    try {
                      localStorage.setItem("parley.autoTurns", v ? "1" : "0");
                    } catch {
                      /* private window; it just will not be remembered */
                    }
                  }}
                  aria-label="End my turn automatically"
                />
              </Row>

              {/* DISABLED RATHER THAN HIDDEN. A control that appears and
                  disappears makes the panel jump and hides what the setting
                  can even do; greyed out, it still says there is a choice
                  here and what it would be. */}
              <div className="px-4 pb-5 pl-8" aria-hidden={!autoTurns}>
                <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3">
                  <span
                    className="md-label-medium"
                    style={{
                      color: "var(--md-on-surface-variant)",
                      opacity: autoTurns ? 1 : 0.5,
                    }}
                  >
                    How long to wait
                  </span>
                  <span
                    className="md-body-small"
                    style={{
                      color: "var(--md-on-surface-variant)",
                      opacity: autoTurns ? 1 : 0.5,
                    }}
                  >
                    {PATIENCE.find((p) => p.id === patience)?.detail}
                  </span>
                </div>

                {/* Four steps, not a slider. Each one moves a pause length AND
                    a confidence bar together, which a continuous track cannot
                    honestly represent -- and nobody can tell 640ms from 700ms
                    by dragging. */}
                <div
                  role="radiogroup"
                  aria-label="How long to wait before answering"
                  className="flex gap-1 rounded-[var(--md-shape-full)] p-1"
                  style={{
                    background: "var(--md-surface-container-high)",
                    opacity: autoTurns ? 1 : 0.5,
                  }}
                >
                  {PATIENCE.map((step) => {
                    const on = step.id === patience;
                    return (
                      <button
                        key={step.id}
                        type="button"
                        role="radio"
                        aria-checked={on}
                        disabled={!autoTurns}
                        title={step.detail}
                        onClick={() => {
                          setPatienceStep(step.id);
                          try {
                            localStorage.setItem("parley.patience", step.id);
                          } catch {
                            /* private window; it will not be remembered */
                          }
                        }}
                        className="md-label-medium md-state flex-1 rounded-[var(--md-shape-full)] px-2 py-1.5 disabled:cursor-not-allowed"
                        style={{
                          background: on
                            ? "var(--md-secondary-container)"
                            : "transparent",
                          color: on
                            ? "var(--md-on-secondary-container)"
                            : "var(--md-on-surface-variant)",
                        }}
                      >
                        {step.label}
                      </button>
                    );
                  })}
                </div>
              </div>
          </div>

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
      ) : exchanges.length === 0 && !inFlight.answer ? (
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
    // `py-5`, not `p-4`. These are separate settings that happen to share a
    // card, and at 16px the rules between them did more separating than the
    // space did -- which reads as a table of rows rather than a handful of
    // independent choices.
    <div className="flex items-center justify-between gap-4 px-4 py-5">
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
export function MicButton({
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
    <li className="md-card md-card-elevated space-y-1.5 p-5">
      {/* THE ANSWER ONLY. What the participant said used to sit above it, taken
          from `input_transcription` -- a separate, lossier pass that rendered
          someone saying their name was John as "madre es un", directly above
          the model's correct reply of "Thanks, John". A wrong transcript beside
          a right answer is worse than no transcript: it reads as authoritative
          and invites doubt about the half that was correct.

          The reply carries the question anyway. "Thanks, John, how many years
          of experience do you have?" tells you what was asked and confirms what
          was heard, in the words of the thing that actually heard it. */}
      {/* LABELLED, because the card now holds only one side of the exchange.
          With the participant's line removed there is nothing to contrast
          against, and an unattributed paragraph reads as the app talking
          rather than the assistant. */}
      <p
        className="md-label-medium"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        AI Assistant
      </p>
      <p className="md-body-medium whitespace-pre-wrap">
        {exchange.answer || <em>no answer</em>}
      </p>

      {exchange.recorded.length > 0 && (
        <div
          className="mt-3 border-t pt-3"
          style={{ borderColor: "var(--md-outline-variant)" }}
        >
          <p
            className="md-label-medium mb-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Taken from your answer
          </p>
          {exchange.recorded.map((entry, i) => (
            <Recorded key={i} entry={entry} />
          ))}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
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
export function ProfileCard({
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
  // EVERY declared field, filled or not.
  //
  // Optional ones used to appear only once they had a value, on the reasoning
  // that a row of blanks reads as an abandoned form. That was wrong in a way
  // the card could not show: it hid what this interview is CAPABLE of
  // capturing, so an empty "Location" looked like a field that did not exist
  // rather than one nobody had asked about.
  const shown = fields.filter((f) => f.name !== "notes" && f.name !== "quotes");
  const required = fields.filter((f) => f.required);

  // COUNTED FROM THE PROFILE, not from `missing`.
  //
  // `missing` starts as an empty array -- nothing has been recorded, so no
  // tool call has reported anything -- and "not in missing" then read as
  // "filled". An untouched profile displayed "6 of 6" above six empty rows.
  const filled = required.filter((f) => !isEmpty(profile[f.name]));
  // What was notable about HOW each answer was given. Keyed by field, with
  // `general` for anything about the person rather than one answer.
  const notes = (profile.notes ?? {}) as ProfileNotes;
  const quotes = (profile.quotes ?? {}) as ProfileNotes;
  const buckets = new Set([
    ...Object.keys(notes),
    ...Object.keys(quotes),
  ]);
  const density =
    Object.values(notes).reduce((n, v) => n + v.length, 0) +
    Object.values(quotes).reduce((n, v) => n + v.length, 0);

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
            : `${filled.length} of ${required.length}`}
        </span>
      </div>

      {typeof profile.summary === "string" && profile.summary && (
        <p
          className="md-body-medium mb-3 italic"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {profile.summary}
        </p>
      )}

      <dl className="space-y-2">
        {shown.map((f) => {
          const value = profile[f.name] as string | number | string[] | undefined;
          const empty = isEmpty(value);
          return (
            <div key={f.name} className="flex gap-3">
              <dt
                className="md-body-small w-40 shrink-0"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {label(f.name)}
                {/* Marked, so an empty optional row reads as "not asked"
                    rather than "missing". */}
                {!f.required && (
                  <span className="ml-1 opacity-60">optional</span>
                )}
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
                <Detail notes={notes[f.name]} quotes={quotes[f.name]} />
              </dd>
            </div>
          );
        })}
      </dl>

      {/* The buckets that belong to no field. `other` is expected to be the
          largest: the fields were chosen in advance and the person was not, so
          the most interesting thing they say usually belongs nowhere. */}
      {(["general", "other"] as const).map((bucket) =>
        // `other` is shown even when EMPTY, because it is the one section
        // whose emptiness is a finding: if nothing landed there, the
        // interviewer summarised instead of organising.
        buckets.has(bucket) || bucket === "other" ? (
          <div
            key={bucket}
            className="mt-4 border-t pt-3"
            style={{ borderColor: "var(--md-outline-variant)" }}
          >
            <p
              className="md-label-medium mb-1"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {bucket === "general" ? "Impressions" : "Other notes"}
            </p>
            {buckets.has(bucket) ? (
              <Detail notes={notes[bucket]} quotes={quotes[bucket]} />
            ) : (
              <p
                className="md-body-small"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Nothing yet. Everything that fits no field belongs here.
              </p>
            )}
          </div>
        ) : null,
      )}

      {complete && density < 6 && (
        <p
          className="md-body-small mt-3"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Thin on detail — {density} note{density === 1 ? "" : "s"} and quotes.
          The fields are the least valuable part of this.
        </p>
      )}
    </section>
  );
}

/**
 * The observations and the actual words, under whatever they belong to.
 *
 * Quotes are set apart from notes because they are a different KIND of thing:
 * a note is our reading of the person, a quote is the person. A reader trusts
 * the second in a way they never quite trust the first, so it should not be
 * possible to mistake one for the other at a glance.
 */
function Detail({
  notes,
  quotes,
}: {
  notes?: string[];
  quotes?: string[];
}) {
  if (!notes?.length && !quotes?.length) return null;
  return (
    <>
      {(notes ?? []).map((note) => (
        <span
          key={note}
          className="md-body-small mt-1 block"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {note}
        </span>
      ))}
      {(quotes ?? []).map((quote) => (
        <span
          key={quote}
          className="md-body-small mt-1 block border-l-2 pl-2 italic"
          style={{
            color: "var(--md-on-surface)",
            borderColor: "var(--md-primary)",
          }}
        >
          &ldquo;{quote}&rdquo;
        </span>
      ))}
    </>
  );
}

/** `years_experience` -> "Years experience". The API names fields for code. */
function label(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}


/**
 * What one `record_profile` call captured, in plain English.
 *
 * This is the honest version of "what you said": it comes from the model,
 * which heard the audio, rather than from the transcription pass, which
 * frequently did not. It is also more useful -- a participant can see that
 * their answer was understood as "six years" without reading a transcript of
 * themselves.
 */
function Recorded({ entry }: { entry: Record<string, unknown> }) {
  const lines: string[] = [];
  for (const [key, value] of Object.entries(entry)) {
    if (value === null || value === undefined || value === "") continue;
    if (key === "notes" || key === "quotes") {
      const n = Array.isArray(value) ? value.length : 1;
      lines.push(`${n} ${key === "notes" ? "note" : "quote"}${n === 1 ? "" : "s"}`);
      continue;
    }
    const shown = Array.isArray(value) ? value.join(", ") : String(value);
    lines.push(`${label(key)}: ${shown}`);
  }
  if (!lines.length) return null;
  return (
    <p
      className="md-body-small"
      style={{ color: "var(--md-on-surface-variant)" }}
    >
      {lines.join(" · ")}
    </p>
  );
}


/**
 * Nothing was recorded for this field.
 *
 * `0` is a value: a graduate with no professional experience has answered the
 * question, and a plain falsy check would render that as a dash and count it
 * as missing.
 */
function isEmpty(value: unknown): boolean {
  if (value === 0) return false;
  if (Array.isArray(value)) return value.length === 0;
  return value === undefined || value === null || value === "";
}

"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import {
  type LiveEvent,
  type LiveSession,
  openLiveSession,
} from "../../parley/liveSession";
import { MicButton, type Phase } from "../../parley/surface";
import { ConfirmButton } from "../../md";
import { IconHowler, IconSpinner } from "../../icons";

/**
 * The magic link: being interviewed, with no account.
 *
 * WHAT THIS PAGE DELIBERATELY IS NOT
 *
 * It is not Parley with the drawer hidden. A participant is not a user of this
 * app -- they were sent a link by somebody who wants to talk to them, and they
 * have one thing to do. So there is no navigation, no conversation list, no
 * settings, no mention of documents or modes or any of the rest of it. The
 * shell skips /howl/* entirely (see `isAuthRoute`), which also keeps the
 * unauthenticated redirect off a page whose whole point is having no account.
 *
 * THE TOKEN IS THE ONLY CREDENTIAL, and it buys exactly one conversation. It
 * goes straight to the socket; there is no REST call a guest can make, because
 * there is no endpoint that would take it.
 *
 * RETURNING IS NORMAL. A dropped call, a closed tab, a phone that rang: the
 * link reopens the SAME conversation with what was already said, because the
 * server resolves it to the session it started. So this page never warns about
 * leaving, and reloading it is not an error state.
 */
export default function HowlPage({
  params,
}: {
  params: Promise<{ token: string }>;
}) {
  const { token } = use(params);
  return <Guest token={token} />;
}

/** Seconds between the interviewer finishing and the microphone reopening. */
const RESTART_SECONDS = 2;

type Said = { id: string; text: string };

function Guest({ token }: { token: string }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [started, setStarted] = useState(false);
  const [ended, setEnded] = useState(false);
  /** They ended it rather than the interviewer -- a different thing to say. */
  const [endedByUser, setEndedByUser] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [remaining, setRemaining] = useState(0);
  const [said, setSaid] = useState<Said[]>([]);
  /** The reply being spoken right now, before the turn closes. */
  const [saying, setSaying] = useState("");

  const session = useRef<LiveSession | null>(null);
  const timer = useRef<number | null>(null);
  const countdown = useRef<number | null>(null);
  const endedRef = useRef(false);
  const startRef = useRef<() => void>(() => {});

  useEffect(() => {
    endedRef.current = ended;
  }, [ended]);

  useEffect(
    () => () => {
      session.current?.close();
      if (timer.current) window.clearInterval(timer.current);
      if (countdown.current) window.clearInterval(countdown.current);
    },
    [],
  );

  const stopCountdown = useCallback(() => {
    if (countdown.current) {
      window.clearInterval(countdown.current);
      countdown.current = null;
    }
    setRemaining(0);
  }, []);

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
        case "said":
          // Accumulated locally only so there is something on screen WHILE it
          // talks. The `turn` event replaces it with the assembled version,
          // which is the one that matches what was stored.
          setSaying((prev) => prev + event.text);
          setPhase("speaking");
          break;
        case "turn":
          if (event.answer) {
            setSaid((prev) => [{ id: `${Date.now()}`, text: event.answer }, ...prev]);
          }
          setSaying("");
          break;
        case "tool":
          // The only one that changes anything here. A participant has no
          // business seeing which fields were filled -- that is the operator's
          // view, and showing a progress bar over someone's own answers turns
          // a conversation into a form being watched.
          if (event.ended) {
            setEnded(true);
            stopCountdown();
            // The interview is over; hand the microphone back. The socket
            // stays open so the closing words still play, but a finished
            // page must not leave the recording indicator lit.
            session.current?.releaseMic();
          }
          break;
        case "finished":
          // Their own decision, acknowledged by the server. Recorded there as
          // ended by the participant, which is what tells a half-finished
          // interview apart from one the interviewer could not complete.
          setEnded(true);
          setEndedByUser(true);
          stopCountdown();
          session.current?.releaseMic();
          setPhase("idle");
          break;
        case "playback_end":
          // Read through the ref: this callback was captured when the socket
          // opened, so a plain `ended` here is whatever it was at that moment.
          if (!endedRef.current) beginCountdown();
          break;
        case "going_away":
          // The server is about to drop us; closing now, while a resume handle
          // is held, turns a dying session into an invisible reconnect.
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
          setPhase((p) => (p === "counting" ? p : "idle"));
          break;
        default:
          break;
      }
    },
    [beginCountdown, stopCountdown],
  );

  /** Open the conversation with the interviewer introducing itself. */
  const begin = useCallback(async () => {
    setError(null);
    stopCountdown();
    try {
      setPhase("connecting");
      if (!session.current) {
        // The empty voice and null conversation are IGNORED for an invite:
        // the server takes the session, the mode and the schema from the
        // project behind the token.
        session.current = await openLiveSession("", null, "howler", onEvent, token);
      }
      setStarted(true);
      session.current.greet();
      setPhase("thinking");
    } catch (e) {
      setPhase("idle");
      session.current = null;
      setError(
        e instanceof Error ? e.message : "The conversation could not be opened.",
      );
    }
  }, [onEvent, token, stopCountdown]);

  const listen = useCallback(async () => {
    setError(null);
    stopCountdown();
    try {
      if (!session.current) {
        setPhase("connecting");
        session.current = await openLiveSession("", null, "howler", onEvent, token);
      }
      await session.current.beginTurn();
      setStarted(true);
      setPhase("listening");
      setElapsed(0);
      // Cleared before setting, or every turn leaves its interval running and
      // the clock counts two seconds per second, then three.
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
          : "The microphone was refused, or the conversation could not open.",
      );
    }
  }, [onEvent, token, stopCountdown]);

  useEffect(() => {
    startRef.current = () => void listen();
  }, [listen]);

  const stop = useCallback(() => {
    if (timer.current) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setLevel(0);
    session.current?.endTurn();
    setPhase("thinking");
  }, []);

  const listening = phase === "listening";
  const speaking = phase === "speaking";
  const counting = phase === "counting";
  // A link that is withdrawn, finished or simply wrong fails at the socket,
  // before anything starts. There is nothing to press afterwards, so the
  // button is not offered.
  const dead = Boolean(error) && !started;

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-6 py-10">
      <header className="mb-8 text-center">
        <span
          className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
          style={{
            background: "var(--md-primary-container)",
            color: "var(--md-on-primary-container)",
          }}
        >
          <IconHowler className="h-7 w-7" />
        </span>
        <h1 className="md-headline-small">
          {ended ? "All done — thank you" : "You have been invited to talk"}
        </h1>
        <p
          className="md-body-medium mx-auto mt-2 max-w-md"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {ended
            ? endedByUser
              ? "You ended the interview. Everything you said has been passed on, and you can close this page."
              : "Your answers have been passed on. You can close this page."
            : started
              ? "Take as long as you like over an answer. Nothing is sent until you press the button."
              : "A short spoken conversation. It will introduce itself, then ask you a few questions — you answer out loud, and press the button when you have finished each answer."}
        </p>
      </header>

      {error && (
        <p
          className="md-body-medium mb-6 rounded-[var(--md-shape-md)] px-4 py-3 text-center"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      {!dead && (
        <section className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-10">
          {!started && phase !== "connecting" && phase !== "thinking" ? (
            <button
              type="button"
              onClick={() => void begin()}
              className="md-label-large rounded-[var(--md-shape-full)] px-8 py-4"
              style={{
                background: "var(--md-primary)",
                color: "var(--md-on-primary)",
                boxShadow: "var(--md-elev-2)",
              }}
            >
              Start
            </button>
          ) : (
            <MicButton
              phase={phase}
              level={level}
              remaining={remaining}
              disabled={ended}
              onStart={() => void listen()}
              onStop={stop}
              onStopSpeaking={() => {
                session.current?.stopSpeaking();
                beginCountdown();
              }}
            />
          )}

          <p className="md-title-small text-center">
            {ended
              ? "The conversation is finished"
              : !started && phase === "idle"
                ? "Ready when you are"
                : phase === "connecting"
                  ? "Connecting"
                  : listening
                    ? `Listening — ${elapsed}s. Pause as long as you like; press when you're done.`
                    : phase === "thinking"
                      ? "Thinking"
                      : speaking
                        ? "Speaking — press to interrupt"
                        : counting
                          ? `Listening again in ${remaining}…`
                          : "Press to answer"}
          </p>

          <p
            className="md-body-small text-center"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {ended
              ? "Nothing further is needed from you."
              : !started && phase === "idle"
                ? "Your browser will ask for the microphone."
                : listening
                  ? "It will not reply until you press. Pausing mid-sentence is fine."
                  : speaking
                    ? "Nothing is being sent while it speaks."
                    : counting
                      ? "Press to start now."
                      : "One question at a time."}
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
              Wait a moment
            </button>
          )}

          {/* A WAY OUT, and one that does not have to be negotiated with the
              interviewer. Somebody who has decided they are done should not
              have to talk their way past one more question -- so this closes
              it outright rather than asking the model to wrap up. Quiet, below
              everything, and behind a confirmation, because it cannot be
              undone. */}
          {started && !ended && (
            <ConfirmButton
              label="End the interview"
              title="End the interview?"
              body="It closes now and this link stops working, so you will not be able to come back to it. Everything you have said so far is kept and passed on."
              confirmLabel="End it"
              onConfirm={() => {
                session.current?.finish();
                stopCountdown();
                setEnded(true);
                setEndedByUser(true);
                setPhase("idle");
              }}
            />
          )}
        </section>
      )}

      {(saying || said.length > 0) && (
        <ol className="mt-6 space-y-3">
          {saying && (
            <li className="md-card md-card-elevated space-y-1.5 p-5">
              <p
                className="md-label-medium flex items-center gap-2"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                <IconSpinner className="h-3.5 w-3.5" />
                Interviewer
              </p>
              <p className="md-body-medium whitespace-pre-wrap">{saying}</p>
            </li>
          )}
          {/* What the INTERVIEWER said, and only that. The transcript of the
              participant's own speech is a separate, lossier pass -- it
              rendered someone saying their name was John as "madre es un" --
              and showing somebody a bad transcript of themselves reads as
              being misheard, right at the moment they need to trust it. */}
          {said.map((s) => (
            <li key={s.id} className="md-card md-card-elevated space-y-1.5 p-5">
              <p
                className="md-label-medium"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Interviewer
              </p>
              <p className="md-body-medium whitespace-pre-wrap">{s.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

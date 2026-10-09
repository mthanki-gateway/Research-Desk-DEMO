"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import { type DuplexEvent, type DuplexSession, openDuplexSession } from "../../../parley/duplexSession";
import { ConfirmButton } from "../../../md";
import { IconHowler, IconMic, IconSpinner } from "../../../icons";

/**
 * The magic link, Duplex edition: the same interview as `/howl/<token>`, held
 * through the streaming pipeline (speech-to-text, a fast model, a streaming
 * voice) instead of the audio-to-audio model. Both doors resolve to the same
 * invite, so this is the A/B: one brief, two interfaces, and the result card
 * says which one held it.
 *
 * Hands-free: no button per answer. The microphone stays open, the turn
 * detector decides when they have finished, and they can talk over the
 * interviewer. As on the other page, the participant is never shown a
 * transcript of themselves -- only what the interviewer said.
 */
export default function HowlDuplexPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = use(params);
  return <Guest token={token} />;
}

type Said = { id: number; text: string };

function Guest({ token }: { token: string }) {
  const [started, setStarted] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [ended, setEnded] = useState(false);
  const [endedByUser, setEndedByUser] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [talking, setTalking] = useState(false);
  const [saying, setSaying] = useState("");
  const [said, setSaid] = useState<Said[]>([]);
  const [heard, setHeard] = useState(false);
  const [detector, setDetector] = useState<"loading" | "ready" | "error">("loading");

  const session = useRef<DuplexSession | null>(null);
  const finishing = useRef(false);
  const counter = useRef(0);
  const current = useRef("");

  const end = useCallback((byUser: boolean) => {
    session.current?.close();
    session.current = null;
    setEnded(true);
    setEndedByUser(byUser);
    setLevel(0);
  }, []);

  useEffect(() => {
    // The turn models take a while to arrive; fetch them now.
    void import("../../../parley/turnDetector").then((m) => {
      m.enableTurnDetection(() => {}, "balanced");
      m.disableTurnDetection();
    });
    return () => session.current?.close();
  }, []);

  const onEvent = useCallback(
    (e: DuplexEvent) => {
      switch (e.type) {
        case "ready":
          session.current?.greet();
          break;
        case "say":
          current.current = current.current ? `${current.current} ${e.text}` : e.text;
          setSaying(current.current);
          break;
        case "partial":
          setHeard(true);
          break;
        case "turn_end":
        case "stop":
          if (current.current) {
            const text = current.current;
            setSaid((all) => [...all, { id: ++counter.current, text }]);
          }
          current.current = "";
          setSaying("");
          break;
        case "speaking":
          setTalking(e.on);
          // The interviewer closed it: let the goodbye finish, then stop.
          if (!e.on && finishing.current) end(false);
          break;
        case "finished":
          finishing.current = true;
          break;
        case "level":
          setLevel(e.level);
          break;
        case "detector":
          setDetector(e.state);
          break;
        case "error":
          setError(e.detail);
          break;
        case "closed":
          setConnecting(false);
          break;
      }
    },
    [end],
  );

  const begin = useCallback(async () => {
    setError(null);
    setConnecting(true);
    try {
      session.current = await openDuplexSession("Kore", "balanced", 250, 24_000, onEvent, "", token);
      setStarted(true);
    } catch (err) {
      setError(
        err instanceof Error && err.name === "NotAllowedError"
          ? "The microphone was refused. Allow it in the address bar and try again."
          : err instanceof Error
            ? err.message
            : "Could not start.",
      );
    } finally {
      setConnecting(false);
    }
  }, [onEvent, token]);

  // A link that is withdrawn, finished or wrong fails at the socket, before
  // anything starts; there is nothing to press afterwards.
  const dead = Boolean(error) && !started;

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-6 py-10">
      <header className="mb-8 text-center">
        <span
          className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
          style={{ background: "var(--md-primary-container)", color: "var(--md-on-primary-container)" }}
        >
          <IconHowler className="h-7 w-7" />
        </span>
        <h1 className="md-headline-small">{ended ? "All done — thank you" : "You have been invited to talk"}</h1>
        <p className="md-body-medium mx-auto mt-2 max-w-md" style={{ color: "var(--md-on-surface-variant)" }}>
          {ended
            ? endedByUser
              ? "You ended the interview. Everything you said has been passed on, and you can close this page."
              : "Your answers have been passed on. You can close this page."
            : started
              ? "Just talk. It answers when you pause, and you can interrupt it whenever you like."
              : "A short spoken conversation. It will introduce itself and ask you a few questions. Answer out loud — there is no button to press between answers."}
        </p>
      </header>

      {error && (
        <p
          className="md-body-medium mb-6 rounded-[var(--md-shape-md)] px-4 py-3 text-center"
          style={{ background: "var(--md-error-container)", color: "var(--md-on-error-container)" }}
        >
          {error}
        </p>
      )}

      {!dead && (
        <section className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-10">
          {!started ? (
            <button
              type="button"
              disabled={connecting}
              onClick={() => void begin()}
              className="md-label-large rounded-[var(--md-shape-full)] px-8 py-4 disabled:cursor-not-allowed disabled:opacity-60"
              style={{ background: "var(--md-primary)", color: "var(--md-on-primary)", boxShadow: "var(--md-elev-2)" }}
            >
              {connecting ? "Connecting…" : "Start"}
            </button>
          ) : (
            <span
              className="grid h-20 w-20 place-items-center rounded-full"
              style={{
                background: talking ? "var(--md-secondary-container)" : "var(--md-primary-container)",
                color: talking ? "var(--md-on-secondary-container)" : "var(--md-on-primary-container)",
                boxShadow: `0 0 0 ${Math.round(Math.min(1, level * 4) * 12)}px color-mix(in srgb, var(--md-primary) 22%, transparent)`,
                transition: "box-shadow 80ms linear",
              }}
            >
              <IconMic className="h-8 w-8" />
            </span>
          )}

          <p className="md-title-small text-center">
            {ended
              ? "The conversation is finished"
              : !started
                ? "Ready when you are"
                : talking
                  ? "Speaking — talk to interrupt"
                  : "Listening"}
          </p>
          <p className="md-body-small text-center" style={{ color: "var(--md-on-surface-variant)" }}>
            {ended
              ? "Nothing further is needed from you."
              : !started
                ? "Your browser will ask for the microphone. Headphones work best."
                : detector === "loading"
                  ? "Getting ready… speak in a moment."
                  : heard
                    ? "Pausing mid-sentence is fine."
                    : "Say something whenever you are ready."}
          </p>

          {started && !ended && (
            <ConfirmButton
              label="End the interview"
              title="End the interview?"
              body="It closes now and this link stops working, so you will not be able to come back to it. Everything you have said so far is kept and passed on."
              confirmLabel="End it"
              onConfirm={() => {
                session.current?.finish();
                end(true);
              }}
            />
          )}
        </section>
      )}

      {(saying || said.length > 0) && (
        <ol className="mt-6 space-y-3">
          {saying && (
            <li className="md-card md-card-elevated space-y-1.5 p-5">
              <p className="md-label-medium flex items-center gap-2" style={{ color: "var(--md-on-surface-variant)" }}>
                <IconSpinner className="h-3.5 w-3.5" />
                Interviewer
              </p>
              <p className="md-body-medium whitespace-pre-wrap">{saying}</p>
            </li>
          )}
          {[...said].reverse().map((s) => (
            <li key={s.id} className="md-card md-card-elevated space-y-1.5 p-5">
              <p className="md-label-medium" style={{ color: "var(--md-on-surface-variant)" }}>
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

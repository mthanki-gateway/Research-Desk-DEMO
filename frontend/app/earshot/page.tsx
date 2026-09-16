"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type VoiceSource,
  type VoiceStatus,
  type VoiceTurn,
  askByVoice,
  audioUrl,
  getVoiceStatus,
} from "@/lib/api";
import { Button, Switch } from "../md";
import {
  IconEarshot,
  IconExternal,
  IconMic,
  IconPlay,
  IconSpinner,
  IconStop,
  IconWave,
} from "../icons";
import { type PushToTalk, openMicrophone, pushToTalk } from "./pushToTalk";

/**
 * Earshot — the Research Desk, answered out loud.
 *
 * SAME CORPUS, DIFFERENT DOOR. Every document uploaded in the Library is
 * answerable here the moment it finishes indexing, and the agent that answers
 * is the same graph with the same tools and the same web search. Nothing is
 * indexed twice and nothing is configured twice; what differs is that the
 * question arrives as sound and the answer leaves as sound.
 *
 * TURNS ARE TAKEN BY BUTTON, NOT BY SILENCE.
 *
 * Voice activity detection cannot tell a pause for thought from the end of a
 * question. Tuned tight it cuts people off mid-sentence; tuned loose it leaves
 * a gap long enough that people start speaking again. And an assistant that
 * listens continuously will, sooner or later, interrupt — which is a far worse
 * experience than pressing a button. Start and Stop mean both parties know
 * whose turn it is, always.
 *
 * WHY A VOICE APP HAS A SCREEN AT ALL
 *
 * It is audio-first, not audio-only-on-principle. The transcript is shown
 * because when an answer is wrong the first question is always whether it
 * heard the question right, and without the transcript that is unanswerable.
 * The sources are shown because a spoken citation is unverifiable — you cannot
 * click a sentence you just heard.
 */

type Phase = "idle" | "listening" | "thinking" | "speaking";

type Exchange = {
  id: string;
  turn: VoiceTurn;
  url: string;
};

export default function Earshot() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [voiceName, setVoiceName] = useState("");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  /** Replay the reply automatically. Off for anyone who wants it quiet. */
  const [autoplay, setAutoplay] = useState(true);

  const recorder = useRef<PushToTalk | null>(null);
  const player = useRef<HTMLAudioElement | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    getVoiceStatus()
      .then((s) => {
        setStatus(s);
        setVoiceName(s.default_voice);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not reach the API"),
      );
  }, []);

  // Release the microphone and revoke the blob URLs on unmount. Object URLs
  // are not garbage collected — the browser holds the blob until told not to,
  // and a long session leaves megabytes of spoken answers resident.
  useEffect(
    () => () => {
      recorder.current?.cancel();
      if (timer.current) window.clearInterval(timer.current);
    },
    [],
  );

  const startListening = useCallback(async () => {
    setError(null);
    try {
      const stream = await openMicrophone();
      recorder.current = pushToTalk(stream, (l) => setLevel(l.level));
      setPhase("listening");
      setElapsed(0);
      timer.current = window.setInterval(
        () => setElapsed((n) => n + 1),
        1000,
      ) as unknown as number;
    } catch {
      setError(
        "The microphone was refused. Allow it for this site — browsers only " +
          "offer the mic on https or localhost.",
      );
    }
  }, []);

  const stopAndAsk = useCallback(async () => {
    const rec = recorder.current;
    recorder.current = null;
    if (timer.current) window.clearInterval(timer.current);
    setLevel(0);
    if (!rec) return;

    const take = await rec.stop();
    if (!take) {
      setPhase("idle");
      setError("That take was silent — nothing was recorded.");
      return;
    }

    setPhase("thinking");
    try {
      const turn = await askByVoice(take.wav, voiceName);
      const url = audioUrl(turn);
      const id = `${Date.now()}`;
      // Newest FIRST. A conversation you cannot see the top of is a
      // conversation where the thing you just asked about has scrolled away.
      setExchanges((prev) => [{ id, turn, url }, ...prev]);
      if (autoplay) {
        setPhase("speaking");
        play(url);
      } else {
        setPhase("idle");
      }
    } catch (e) {
      setPhase("idle");
      setError(e instanceof Error ? e.message : "The turn failed");
    }
  }, [voiceName, autoplay]);

  function play(url: string) {
    player.current?.pause();
    const audio = new Audio(url);
    player.current = audio;
    audio.onended = () => setPhase("idle");
    // A rejected play() is the browser's autoplay policy, not a broken file.
    // Falling back to idle leaves the manual replay button as the way through.
    audio.play().catch(() => setPhase("idle"));
  }

  function stopPlayback() {
    player.current?.pause();
    player.current = null;
    setPhase("idle");
  }

  const busy = phase === "thinking";
  const listening = phase === "listening";
  const speaking = phase === "speaking";

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
          Ask out loud. Same documents as the Research Desk, same web search —
          answered in speech.
        </p>
      </header>

      {status && !status.enabled && (
        <p
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          Speech is not configured on this deployment — no Google API key, so
          nothing can be transcribed or spoken.
        </p>
      )}

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

      {/* ---- the one control that matters --------------------------------- */}
      <section className="md-card md-card-outlined flex flex-col items-center gap-4 px-6 py-10">
        <MicButton
          phase={phase}
          level={level}
          disabled={!status?.enabled}
          onStart={() => void startListening()}
          onStop={() => void stopAndAsk()}
          onCancelSpeech={stopPlayback}
        />

        <p className="md-title-small text-center">
          {listening
            ? `Listening — ${elapsed}s. Click to send.`
            : busy
              ? "Searching and composing the answer"
              : speaking
                ? "Speaking — click to stop"
                : "Click to speak"}
        </p>

        <p
          className="md-body-small text-center"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {listening
            ? "Take as long as you need; nothing is sent until you click."
            : busy
              ? "Transcribing, then searching your documents, then speaking."
              : status?.web_search
                ? "Your documents and the web."
                : "Your documents. Web search is not configured."}
        </p>
      </section>

      {/* ---- settings ------------------------------------------------------
          A card with M3 rows, matching the chat rail. A bare checkbox and a
          bare <select> inherit the browser's own widgets, which is the one
          place in this app that looked like an unstyled form. */}
      {status?.enabled && (
        <section className="md-card md-card-outlined divide-y" style={{ borderColor: "var(--md-outline-variant)" }}>
          <div className="flex items-center justify-between gap-4 p-4">
            <span className="md-body-medium">
              Voice
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {status.voices.find((v) => v.id === voiceName)?.character ??
                  "How the answer sounds"}
              </span>
            </span>
            <select
              value={voiceName}
              onChange={(e) => setVoiceName(e.target.value)}
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
          </div>

          <div
            className="flex items-center justify-between gap-4 p-4"
            style={{ borderColor: "var(--md-outline-variant)" }}
          >
            <span className="md-body-medium">
              Play the answer automatically
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Off if you would rather read first and listen on demand
              </span>
            </span>
            <Switch
              on={autoplay}
              onChange={setAutoplay}
              aria-label="Play the answer automatically"
            />
          </div>

          <div
            className="md-body-small flex items-center justify-between gap-4 px-4 py-3"
            style={{
              color: "var(--md-on-surface-variant)",
              borderColor: "var(--md-outline-variant)",
            }}
          >
            <span>Models</span>
            <span className="truncate text-right">
              {status.stt_model.replace("models/", "")} hears ·{" "}
              {status.tts_models[0]?.replace("models/", "")} speaks
              {status.tts_models.length > 1 &&
                `, +${status.tts_models.length - 1} spare`}
            </span>
          </div>
        </section>
      )}

      {/* ---- the transcript ---------------------------------------------- */}
      {exchanges.length === 0 ? (
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
            <Turn key={x.id} exchange={x} onPlay={() => play(x.url)} />
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * The microphone button, which is the entire interface.
 *
 * Large on purpose. This is a hands-free app used at arm's length, and the
 * standard 40px target is a target for a mouse on a desk. It also carries the
 * live level as a ring, because a recording UI that does not visibly react to
 * sound is indistinguishable from a broken one — and the user finds out only
 * after speaking a whole question into nothing.
 */
function MicButton({
  phase,
  level,
  disabled,
  onStart,
  onStop,
  onCancelSpeech,
}: {
  phase: Phase;
  level: number;
  disabled?: boolean;
  onStart: () => void;
  onStop: () => void;
  onCancelSpeech: () => void;
}) {
  const listening = phase === "listening";
  const thinking = phase === "thinking";
  const speaking = phase === "speaking";

  // Raw RMS on speech sits around 0.02–0.15, so a linear map would barely
  // move. Scaled and clamped to a ring of a few sensible pixels.
  const ring = listening ? Math.min(26, Math.round(level * 260)) : 0;

  const onClick = listening ? onStop : speaking ? onCancelSpeech : onStart;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || thinking}
      aria-label={listening ? "Stop and send" : speaking ? "Stop speaking" : "Start speaking"}
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
        // The ring is a box-shadow rather than a scaled element so it cannot
        // reflow anything around it as the level moves.
        boxShadow: ring
          ? `0 0 0 ${ring}px color-mix(in srgb, var(--md-error) 22%, transparent)`
          : "var(--md-elev-2)",
      }}
    >
      {thinking ? (
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

function Turn({
  exchange,
  onPlay,
}: {
  exchange: Exchange;
  onPlay: () => void;
}) {
  const { turn } = exchange;
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
          {turn.transcript || <em>nothing intelligible</em>}
        </span>
      </p>

      <div
        className="border-t pt-3"
        style={{ borderColor: "var(--md-outline-variant)" }}
      >
        {/* `spoken`, not `answer`. What is on screen should be what was said —
            showing the written form next to different spoken words is worse
            than showing nothing, because it looks like a transcription error. */}
        <p className="md-body-medium whitespace-pre-wrap">{turn.spoken}</p>

        {turn.truncated && (
          <p
            className="md-body-small mt-2"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            The spoken answer was shortened. The full text:
            <span className="mt-1 block whitespace-pre-wrap">{turn.answer}</span>
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="tonal" onClick={onPlay}>
          <IconPlay className="h-4 w-4" />
          Play again
        </Button>
        {turn.sources.map((s) => (
          <SourceChip key={`${s.kind}:${s.label}:${s.url ?? ""}`} source={s} />
        ))}
        {turn.sources.length === 0 && !turn.heard_nothing && (
          <span
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            No passages cited
          </span>
        )}
      </div>
    </li>
  );
}

/** Where it came from. A spoken citation cannot be clicked, so this can. */
function SourceChip({ source }: { source: VoiceSource }) {
  const label = `${source.label}${source.passages > 1 ? ` ·${source.passages}` : ""}`;
  if (source.kind === "web" && source.url) {
    return (
      <a
        className="md-badge"
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        title={source.url}
      >
        {label}
        <IconExternal className="h-3 w-3 shrink-0 opacity-60" />
      </a>
    );
  }
  return <span className="md-badge">{label}</span>;
}

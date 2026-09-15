"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  nvidiaStatus,
  playgroundStatus,
  transcribeAudio,
  type Transcription,
} from "@/lib/api";
import { Button, Chip, TextArea } from "../md";
import { IconMic, IconSpinner, IconStop } from "../icons";
import { VAD, record, type Meter, type Recorder } from "./recorder";
import { languagesFor } from "./languages";

/**
 * One vendor, end to end: its own microphone, settings and transcript.
 *
 * FULLY INDEPENDENT, not two skins over shared state. Each panel holds its own
 * recorder, so both can run at once -- speak once and watch Whisper and NeMo
 * transcribe the same words side by side, which is the only honest way to
 * compare them. A shared recorder would have forced a choice before speaking
 * and made every comparison a comparison of two different utterances.
 *
 * The cost is two `getUserMedia` streams and two AudioContexts on the same
 * microphone. Browsers allow that, and the duplication is the price of the
 * comparison.
 */

export type Provider = "groq" | "nvidia";

type Line = {
  id: number;
  text: string;
  language: string | null;
  ms: number;
  voicedMs: number;
};

type Config = {
  title: string;
  blurb: string;
  models: { id: string; label: string }[];
  /**
   * Said once per vendor. The LANGUAGE LIST is not here -- it belongs to the
   * model, because "NVIDIA supports Hindi" has no truth value: Parakeet's 25
   * are European and its English-only variants accept one. See ./languages.ts.
   */
  languageNote: string;
};

/**
 * What each vendor can run.
 *
 * Two of them because their failure modes differ in a way worth switching
 * between mid-session. Measured on four seconds of digital silence:
 *
 *     Parakeet -> ""
 *     Whisper  -> "Thank you."   (no_speech_prob 0.000, its top confidence)
 */
export const PROVIDERS: Record<Provider, Config> = {
  groq: {
    title: "Groq — Whisper",
    blurb:
      "OpenAI's Whisper on Groq's LPUs. Very fast, and invents end-credit phrases when handed silence.",
    models: [
      { id: "whisper-large-v3-turbo", label: "whisper-large-v3-turbo — fastest" },
      { id: "whisper-large-v3", label: "whisper-large-v3 — more accurate" },
    ],
    languageNote:
      "Whisper covers 99 languages and can auto-detect. Detection runs per segment, so a bilingual speaker can flip script part way through; pinning stops that, at the cost of forcing the other language into this one's script.",
  },
  nvidia: {
    title: "NVIDIA — NeMo",
    blurb:
      "Parakeet and Canary on NVIDIA Cloud Functions. Returns nothing on silence rather than guessing.",
    models: [
      {
        id: "ai-parakeet-1_1b-rnnt-multilingual-asr",
        label: "parakeet-1.1b multilingual — 13 languages (hosted)",
      },
      { id: "ai-parakeet-ctc-1_1b-asr", label: "parakeet-ctc-1.1b — English" },
      { id: "ai-parakeet-tdt-0_6b-v2", label: "parakeet-tdt-0.6b-v2 — timestamps" },
      { id: "ai-canary-1b-asr", label: "canary-1b — 26 languages, incl. Hindi" },
      { id: "ai-whisper-large-v3", label: "whisper-large-v3 — on NVIDIA" },
    ],
    // NO AUTO-DETECT OPTION, because Riva has none. Offering one would produce
    // a setting that silently means "English" -- worse than not offering it.
    languageNote:
      "Riva has no auto-detection, so a language must be chosen; the backend maps it to a full locale. The list changes with the model, and comes from probing the live endpoints rather than the model cards — the hosted Parakeet serves 12 of its documented 25 and no Hindi, while Canary serves 25 including Hindi.",
  },
};

/**
 * Transcript tail handed to Whisper as `prompt`.
 *
 * MEASURED IN BYTES. Groq rejects a prompt over 896 "characters" and counts
 * them as UTF-8 bytes -- verified: 890 ASCII characters pass, 300 Devanagari
 * characters fail at 900 bytes. Hindi costs three bytes a character, so a tail
 * measured with `str.length` sends about a third more than it thinks.
 */
const PROMPT_TAIL_BYTES = 700;

/**
 * A ready-made style example for Hindi-English speech.
 *
 * WRITTEN THE WAY THE OUTPUT SHOULD LOOK -- Hindi in Devanagari, English in
 * Latin, one sentence -- because that is the signal Whisper picks up.
 * Describing the intent in English does nothing: the prompt is a continuation
 * seed, not an instruction.
 */
const HINGLISH_STYLE =
  "हाँ, मैंने उस meeting में बोला था कि deployment अगले sprint में होगा। Team ने कहा requirements क्लियर हैं।";

const utf8Length = (text: string) => new TextEncoder().encode(text).length;

function clipToBytes(text: string, maxBytes: number): string {
  if (utf8Length(text) <= maxBytes) return text;
  let out = text;
  while (out.length > 0 && utf8Length(out) > maxBytes) {
    out = out.slice(Math.ceil(out.length / 20) || 1);
  }
  return out.replace(/^\S*\s/, "").trim();
}

export default function ProviderPanel({ provider }: { provider: Provider }) {
  const config = PROVIDERS[provider];
  const isGroq = provider === "groq";

  /**
   * Three outcomes, not two.
   *
   * This was a boolean, and a FAILED REQUEST rendered as "the key is not
   * set" -- which is what it looked like when the status route was
   * accidentally deleted and started 404ing: the panel confidently blamed a
   * key that was correctly configured. A check that cannot distinguish "the
   * answer is no" from "there was no answer" will eventually tell that lie.
   */
  const [ready, setReady] = useState<"checking" | "yes" | "nokey" | "error">(
    "checking",
  );
  const [readyError, setReadyError] = useState<string | null>(null);
  const [model, setModel] = useState(config.models[0].id);
  const languages = languagesFor(model);
  const [language, setLanguage] = useState(languages[0].value);
  const [style, setStyle] = useState("");
  /**
   * Three states, because "starting" is a real one the user can see.
   *
   * Between the click and the first usable frame there is a permission
   * prompt, a `getUserMedia` round trip and 600ms of noise-floor calibration.
   * As a boolean that whole window looked like a button that had not
   * responded, and the natural reaction is to click it again.
   */
  const [phase, setPhase] = useState<"idle" | "starting" | "recording">("idle");
  const recording = phase === "recording";
  const [pending, setPending] = useState(0);
  const [lines, setLines] = useState<Line[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [meter, setMeter] = useState<Meter>({ level: 0, speaking: false });
  const [calibrated, setCalibrated] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const recorder = useRef<Recorder | null>(null);
  const nextId = useRef(0);
  const transcriptTail = useRef("");
  // Read inside callbacks created once, so they cannot close over stale state.
  const opts = useRef({ model, language, style });
  useEffect(() => {
    opts.current = { model, language, style };
  }, [model, language, style]);

  // A language the NEW model cannot do is reset to its first option.
  //
  // Switching from Whisper to Parakeet with Hindi selected would otherwise
  // keep sending `hi` to a model whose 25 languages are all European -- a
  // request that fails, or worse, quietly returns something plausible in the
  // wrong language.
  useEffect(() => {
    if (!languages.some((l) => l.value === language)) {
      setLanguage(languages[0].value);
    }
  }, [languages, language]);

  // Each panel checks ITS OWN vendor. One key can be set and the other not,
  // and a single shared "enabled" flag would disable a working provider
  // because the other one is unconfigured.
  useEffect(() => {
    const check = isGroq
      ? playgroundStatus().then((s) => s.enabled)
      : nvidiaStatus().then((s) => s.enabled);
    void check
      .then((enabled) => setReady(enabled ? "yes" : "nokey"))
      .catch((e) => {
        setReady("error");
        setReadyError(e instanceof Error ? e.message : "status check failed");
      });
  }, [isGroq]);

  const send = useCallback(
    async (wav: Blob, voicedMs: number) => {
      const { model: m, language: lang, style: st } = opts.current;
      setPending((n) => n + 1);
      try {
        const r: Transcription = await transcribeAudio(wav, {
          filename: "audio.wav",
          provider,
          model: m,
          language: lang,
          // Whisper only. NeMo has no equivalent decoder-context parameter,
          // and the backend drops it rather than erroring.
          prompt: isGroq
            ? [st, transcriptTail.current].filter(Boolean).join(" ").trim()
            : undefined,
        });
        if (r.text) {
          transcriptTail.current = clipToBytes(
            `${transcriptTail.current} ${r.text}`.trim(),
            PROMPT_TAIL_BYTES,
          );
          setLines((prev) => [
            ...prev,
            {
              id: nextId.current++,
              text: r.text,
              language: r.language,
              ms: r.elapsed_ms,
              voicedMs,
            },
          ]);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Transcription failed");
      } finally {
        setPending((n) => n - 1);
      }
    },
    [provider, isGroq],
  );

  async function start() {
    setError(null);
    setCalibrated(false);
    setPhase("starting");
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          // OFF, deliberately. AGC hunts for a target level, so during a pause
          // it winds the gain up until room noise reaches speaking volume --
          // manufacturing the input that makes Whisper hallucinate, and
          // defeating the noise floor the recorder calibrates against.
          autoGainControl: false,
        },
      });
    } catch {
      setError(
        "Microphone access was refused. Allow it for this site; browsers only " +
          "expose the mic on https or localhost.",
      );
      // Back to idle, or the button stays stuck on "Starting" after a refusal.
      setPhase("idle");
      return;
    }
    // "recording" only once the graph is live. Calibration is still running
    // for another 600ms, which the meter reports separately.
    setPhase("recording");
    recorder.current = record(streamRef.current, {
      onSegment: (wav, voicedMs) => void send(wav, voicedMs),
      onMeter: setMeter,
      onReady: () => setCalibrated(true),
    });
  }

  const stop = useCallback(() => {
    setPhase("idle");
    // FLUSH BEFORE STOPPING, so the half-finished sentence in the buffer is
    // transcribed rather than discarded. The escape hatch for a noisy room,
    // where a pause may never be detected at all.
    recorder.current?.flush();
    recorder.current?.stop();
    recorder.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setMeter({ level: 0, speaking: false });
  }, []);

  // Releases this panel's microphone if the page is navigated away from.
  useEffect(
    () => () => {
      recorder.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    },
    [],
  );

  const transcript = lines.map((l) => l.text).join(" ");
  // Raw RMS on speech sits around 0.02-0.15, so a linear 0..1 bar would never
  // visibly move.
  const meterPct = Math.min(100, Math.round(meter.level * 600));

  return (
    <section
      className="md-card md-card-outlined space-y-4 p-5"
      style={
        recording
          ? {
              borderColor: "var(--md-primary)",
              // An inset ring rather than a thicker border: border-width would
              // reflow the card by a pixel when recording starts.
              boxShadow: "inset 0 0 0 1px var(--md-primary)",
            }
          : undefined
      }
    >
      <div>
        <h2 className="md-title-medium">{config.title}</h2>
        <p
          className="md-body-small mt-0.5"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {config.blurb}
        </p>
      </div>

      {ready === "error" && (
        <div
          className="md-card md-card-filled p-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          <p className="md-body-small">
            Could not check this provider: {readyError}. The key may well be
            set — this is the API not answering, not a missing key.
          </p>
        </div>
      )}

      {ready === "nokey" && (
        <div
          className="md-card md-card-filled p-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          <p className="md-body-small">
            {isGroq ? (
              <>
                <code>GROQ_API_KEY</code> is not set.
              </>
            ) : (
              <>
                <code>NVIDIA_API_KEY</code> is not set — get a free key from
                build.nvidia.com.
              </>
            )}{" "}
            Add it to <code>.env</code> and run{" "}
            <code>docker compose up -d --force-recreate api</code>.
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <span className="md-title-small w-20 shrink-0">Model</span>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={recording}
          className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2"
          style={{
            background: "var(--md-surface)",
            color: "var(--md-on-surface)",
            border: "1px solid var(--md-outline)",
          }}
        >
          {config.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-3">
          <span className="md-title-small w-20 shrink-0">Language</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2"
            style={{
              background: "var(--md-surface)",
              color: "var(--md-on-surface)",
              border: "1px solid var(--md-outline)",
            }}
          >
            {languages.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {config.languageNote}
        </p>
      </div>

      {isGroq && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="md-title-small">Style example</span>
            <Chip
              selected={style === HINGLISH_STYLE}
              onClick={() =>
                setStyle(style === HINGLISH_STYLE ? "" : HINGLISH_STYLE)
              }
            >
              Hinglish
            </Chip>
            {style && (
              <Button variant="text" onClick={() => setStyle("")}>
                Clear
              </Button>
            )}
          </div>
          <TextArea
            label="Write one line the way you want the output to look"
            value={style}
            onChange={(e) => setStyle(e.target.value)}
            surface="var(--md-surface)"
          />
          <p
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Whisper continues the style of its prompt, so an example mixing
            scripts is what keeps Hindi in Devanagari and English in Latin.
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        {recording ? (
          <Button variant="error" onClick={stop}>
            <IconStop />
            Stop
          </Button>
        ) : (
          <Button
            onClick={() => void start()}
            disabled={ready !== "yes" || phase === "starting"}
          >
            {phase === "starting" ? <IconSpinner /> : <IconMic />}
            {phase === "starting" ? "Starting…" : "Start speaking"}
          </Button>
        )}

        {recording && (
          <span className="flex min-w-[9rem] flex-1 items-center gap-2">
            {/* Not decoration: it is how "the room is below threshold" is told
                apart from "the request is failing", which otherwise look
                identical -- nothing appears either way. */}
            <span
              className="h-1.5 flex-1 overflow-hidden rounded-[var(--md-shape-full)]"
              style={{ background: "var(--md-surface-container-high)" }}
            >
              <span
                className="block h-full transition-[width]"
                style={{
                  width: `${meterPct}%`,
                  background: meter.speaking
                    ? "var(--md-primary)"
                    : "var(--md-outline)",
                  transitionDuration: "80ms",
                }}
              />
            </span>
            <span
              className="md-body-small w-20 shrink-0"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {!calibrated ? "calibrating" : meter.speaking ? "speaking" : "silent"}
            </span>
          </span>
        )}

        {pending > 0 && (
          <span
            className="md-body-small flex items-center gap-2"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            <IconSpinner className="h-3.5 w-3.5" />
            {pending} in flight
          </span>
        )}

        {lines.length > 0 && !recording && (
          <Button
            variant="text"
            onClick={() => {
              setLines([]);
              transcriptTail.current = "";
              setError(null);
            }}
          >
            Clear
          </Button>
        )}
      </div>

      {error && (
        <div
          className="md-card md-card-filled p-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          <p className="md-body-small">{error}</p>
        </div>
      )}

      {lines.length > 0 && (
        <div className="space-y-2">
          <div
            className="md-card p-4"
            style={{ background: "var(--md-surface-container-high)" }}
          >
            <p className="md-body-large whitespace-pre-wrap">{transcript}</p>
          </div>

          {/* Per-segment detail, because the joined transcript hides what goes
              wrong: which segment was slow, and which was decoded as a
              different language from its neighbours. */}
          <details>
            <summary className="md-label-large">
              {lines.length} segment{lines.length === 1 ? "" : "s"}
            </summary>
            <ul className="mt-2 space-y-1">
              {lines.map((l) => (
                <li
                  key={l.id}
                  className="md-body-small flex gap-3"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  <span className="w-14 shrink-0 tabular-nums">{l.ms}ms</span>
                  <span className="w-12 shrink-0 tabular-nums">
                    {(l.voicedMs / 1000).toFixed(1)}s
                  </span>
                  <span className="w-16 shrink-0">{l.language ?? "—"}</span>
                  <span className="min-w-0">{l.text}</span>
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </section>
  );
}

export { VAD };

/// <reference lib="webworker" />

/**
 * Turn detection, off the main thread.
 *
 * TWO MODELS, AND THEY ANSWER DIFFERENT QUESTIONS
 *
 *   Silero VAD (2MB)      "is anyone speaking right now?"
 *   Smart Turn v3 (8MB)   "did that sound like a finished thought?"
 *
 * They are not alternatives, they are a pipeline. Silero is cheap enough to
 * run on every 32ms frame and is what NOTICES a pause. Smart Turn is far too
 * expensive for that -- tens to hundreds of milliseconds per call -- but it is
 * the only one that can tell "I went to the shop and..." from "I went to the
 * shop." Prosody, not words: it reads the Whisper encoder's view of the last
 * eight seconds, so it hears trailing intonation and final lengthening.
 *
 * Energy VAD alone is exactly the thing this app already refused to use,
 * because people pause mid-sentence and it cuts them off. Smart Turn alone
 * would mean running an 8MB model continuously. Together: the cheap one waits
 * for a gap, the expensive one decides whether the gap was the end.
 *
 * WHY A WORKER
 *
 * A decision costs 60ms of spectrogram plus 150-600ms of inference, measured
 * under WASM without SIMD. On the main thread that is a visible freeze of the
 * level ring, in the exact moment the user is watching it to see whether they
 * were heard. Here it is invisible, and the audio path never waits on it.
 */

// `/wasm`, NOT the default entry.
//
// The default bundle is the jsep build, which carries WebGPU and WebNN and is
// a 27MB download; it also asks for `ort-wasm-simd-threaded.jsep.mjs`, which
// is what 404'd when only the plain pair had been copied. This entry uses the
// CPU build at 13.6MB. These are an 8M and a 1M parameter model -- WebGPU
// setup would cost more than it saves, and halving what the browser fetches
// matters far more.
import * as ort from "onnxruntime-web/wasm";
import {
  N_FRAMES,
  N_MELS,
  SAMPLE_RATE,
  WINDOW_SAMPLES,
  logMelSpectrogram,
} from "./mel";

/** Silero v5 takes exactly this many samples per call at 16kHz. */
const VAD_FRAME = 512;
/** Above this, Silero thinks it is hearing speech. Its own documented default. */
const SPEECH = 0.5;

export type ToWorker =
  | { type: "init"; vadUrl: string; turnUrl: string; wasmBase: string }
  | { type: "audio"; pcm: Float32Array }
  /** A new turn: forget the previous one's audio and silence. */
  | { type: "reset" }
  | { type: "config"; silenceMs: number; threshold: number; maxSilenceMs: number };

export type FromWorker =
  | { type: "ready" }
  | { type: "error"; detail: string }
  /** Per frame, so the UI can show that it is hearing something. */
  | { type: "speech"; probability: number }
  /** The speaker appears to have finished. */
  | { type: "endpoint"; probability: number; reason: "semantic" | "timeout"; ms: number };

let vad: ort.InferenceSession | null = null;
let turn: ort.InferenceSession | null = null;

/** Silero is recurrent: this is the conversation state it carries between frames. */
let state: ort.Tensor;
const SR = new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);

function freshState() {
  state = new ort.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
}

/* ------------------------------------------------------------- the settings */

let silenceMs = 600;
let threshold = 0.7;
let maxSilenceMs = 3000;

/**
 * How much ACTUAL SPEECH has to be heard before a turn can end at all.
 *
 * A structural guard, not a tuning knob. `voiced` used to latch on a single
 * frame above 0.5 -- a cough, a chair, a breath -- and 600ms later the model
 * was asked about a window with nothing in it. Silence reads as 0.99
 * "complete", so the turn ended before anybody had spoken. No amount of
 * threshold tuning fixes that, because the model is answering correctly: the
 * question was wrong.
 */
const MIN_SPEECH_MS = 700;

/* ---------------------------------------------------------------- the audio */

/**
 * The last 8 seconds, as a ring. Smart Turn always looks at 8 seconds, so
 * there is no reason to keep more, and keeping it in one preallocated buffer
 * means no allocation on the audio path.
 */
const ring = new Float32Array(WINDOW_SAMPLES);
let ringAt = 0;
let ringFilled = 0;

/** Left over from the last message, because frames rarely divide by 512. */
let pending = new Float32Array(0);

let voiced = false;
/** Total voiced samples this turn, which is what MIN_SPEECH_MS is measured in. */
let voicedSamples = 0;
let silentSamples = 0;
/** Samples of silence at the last Smart Turn call, so it is not asked twice. */
let askedAt = -1;
let busy = false;

function push(pcm: Float32Array) {
  for (let i = 0; i < pcm.length; i++) {
    ring[ringAt] = pcm[i];
    ringAt = (ringAt + 1) % WINDOW_SAMPLES;
  }
  ringFilled = Math.min(ringFilled + pcm.length, WINDOW_SAMPLES);
}

/** The ring, unwrapped oldest-to-newest. */
function window(): Float32Array {
  const out = new Float32Array(ringFilled);
  const start = (ringAt - ringFilled + WINDOW_SAMPLES) % WINDOW_SAMPLES;
  for (let i = 0; i < ringFilled; i++) {
    out[i] = ring[(start + i) % WINDOW_SAMPLES];
  }
  return out;
}

function reset() {
  ringAt = 0;
  ringFilled = 0;
  pending = new Float32Array(0);
  voiced = false;
  voicedSamples = 0;
  silentSamples = 0;
  askedAt = -1;
  freshState();
}

/* ------------------------------------------------------------- the decision */

const post = (m: FromWorker) => (self as DedicatedWorkerGlobalScope).postMessage(m);

async function askSmartTurn(reason: "semantic" | "timeout") {
  // One at a time. A second call while the first is running would queue behind
  // it and answer a question about audio that is already stale.
  if (busy || !turn) return;
  busy = true;
  const started = performance.now();
  try {
    const features = logMelSpectrogram(window());
    const out = await turn.run({
      input_features: new ort.Tensor("float32", features, [1, N_MELS, N_FRAMES]),
    });
    // Named `logits`, but the graph has the sigmoid in it -- the values come
    // back in [0, 1]. Measured: silence 0.99, a steady tone 0.03.
    const probability = Number(out.logits.data[0]);
    const ms = performance.now() - started;

    if (reason === "timeout" || probability >= threshold) {
      post({ type: "endpoint", probability, reason, ms });
      // Whatever happens next is a new turn as far as this is concerned.
      voiced = false;
      voicedSamples = 0;
      silentSamples = 0;
    }
  } catch (e) {
    post({ type: "error", detail: e instanceof Error ? e.message : String(e) });
  } finally {
    busy = false;
  }
}

async function consume(pcm: Float32Array) {
  if (!vad) return;
  push(pcm);

  // Silero wants exactly 512 samples, and the capture callback does not hand
  // over multiples of 512, so the remainder is carried to the next message.
  const joined = new Float32Array(pending.length + pcm.length);
  joined.set(pending);
  joined.set(pcm, pending.length);

  let offset = 0;
  while (offset + VAD_FRAME <= joined.length) {
    const frame = joined.subarray(offset, offset + VAD_FRAME);
    offset += VAD_FRAME;

    let probability = 0;
    try {
      const out = await vad.run({
        input: new ort.Tensor("float32", frame.slice(), [1, VAD_FRAME]),
        state,
        sr: SR,
      });
      state = out.stateN as ort.Tensor;
      probability = Number(out.output.data[0]);
    } catch (e) {
      post({ type: "error", detail: e instanceof Error ? e.message : String(e) });
      return;
    }
    post({ type: "speech", probability });

    if (probability >= SPEECH) {
      voiced = true;
      voicedSamples += VAD_FRAME;
      silentSamples = 0;
      askedAt = -1;
    } else if (voiced) {
      silentSamples += VAD_FRAME;
    }
  }
  pending = joined.slice(offset);

  // NOTHING HAPPENS UNTIL SOMEBODY HAS ACTUALLY SPOKEN, and "actually" means
  // MIN_SPEECH_MS of it rather than one frame that crossed a threshold. The
  // pause before the first word is otherwise a pause like any other, and the
  // turn ends before it has begun.
  if (!voiced || voicedSamples < (MIN_SPEECH_MS * SAMPLE_RATE) / 1000) return;

  const quietMs = (silentSamples / SAMPLE_RATE) * 1000;

  // The long stop. Smart Turn can be genuinely unsure -- a trailing "so..."
  // that never resolves -- and without this the microphone stays open on
  // somebody who has plainly finished.
  if (quietMs >= maxSilenceMs) {
    await askSmartTurn("timeout");
    return;
  }

  // ASKED ONCE PER PAUSE, then again only if the pause grows. Re-running on
  // every frame would be ten inferences a second on the most expensive thing
  // here, all of them answering the same question.
  if (quietMs >= silenceMs && silentSamples !== askedAt) {
    askedAt = silentSamples;
    await askSmartTurn("semantic");
  }
}

/* ------------------------------------------------------------------ the wire */

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      // Served from our own origin rather than a CDN, so the feature works
      // offline and cannot break because somebody else's host changed.
      ort.env.wasm.wasmPaths = message.wasmBase;
      ort.env.logLevel = "error";
      freshState();
      [vad, turn] = await Promise.all([
        ort.InferenceSession.create(message.vadUrl),
        ort.InferenceSession.create(message.turnUrl),
      ]);
      post({ type: "ready" });
    } catch (e) {
      post({
        type: "error",
        detail: e instanceof Error ? e.message : "The models could not be loaded.",
      });
    }
    return;
  }

  if (message.type === "config") {
    silenceMs = message.silenceMs;
    threshold = message.threshold;
    maxSilenceMs = message.maxSilenceMs;
    return;
  }

  if (message.type === "reset") {
    reset();
    return;
  }

  if (message.type === "audio") await consume(message.pcm);
};

import type { FromWorker, ToWorker } from "./turnWorker";

let worker: Worker | null = null;
let ready = false;
let enabled = false;
let listener: ((event: TurnEvent) => void) | null = null;
/** Held here because `attach` runs after an await and needs the value the
 *  caller asked for, not whatever the default happens to be. */
let chosen: Patience = "balanced";
/** The dynamic import is in flight. Without this a second call -- an effect
 *  re-running, a toggle flicked twice -- starts a SECOND worker, and two of
 *  them would both fetch 11MB and both report endpoints. */
let starting = false;

function send(message: ToWorker, transfer?: Transferable[]) {
  worker?.postMessage(message, transfer ?? []);
}

/**
 * The main thread's handle on turn detection.
 *
 * All this does is own the worker and hand it audio. The models, the ring
 * buffer and the decision live in `turnWorker.ts`; see its header for why
 * there are two of them and what each is for.
 *
 * A MODULE SINGLETON, NOT PART OF THE LIVE SESSION. It was a method on the
 * socket wrapper first, which was wrong in a way that showed up immediately:
 * the toggle is switched on BEFORE anybody presses Start, so there was no
 * session to call, nothing began loading, and the setting sat on "fetching the
 * models" for ever. The models belong to the browser, not to a socket -- they
 * outlive any one conversation and switching the socket should not re-fetch
 * 11MB.
 *
 * NOTHING HERE ENDS A TURN. It reports that the speaker sounds finished and
 * stops. Whether that closes the turn is the surface's decision, because the
 * button is still the authority and auto-turns is a setting layered over it.
 */

export type TurnEvent =
  | { type: "loading" }
  | { type: "ready" }
  | { type: "error"; detail: string }
  | {
      type: "endpoint";
      probability: number;
      /** `semantic` = it sounded finished. `timeout` = it gave up waiting. */
      reason: "semantic" | "timeout";
      /** How long the decision took, which is worth knowing on a slow machine. */
      ms: number;
    };

/**
 * How long a pause has to last before it is worth asking the expensive model,
 * how sure that model has to be, and how long to wait before giving up on it.
 *
 * 600ms is long enough to survive the gap between two words and the pause
 * somebody takes to find one, and short enough to still feel like a
 * conversation. It started at 280ms, which was the same length as thinking
 * about what to say next.
 *
 * The 3s stop is what rescues the opposite failure: Smart Turn can be
 * genuinely unsure -- a trailing "so..." that never resolves -- and without it
 * somebody is left looking at a microphone that will not answer. A false
 * "finished" costs an interruption, which is recoverable; a false "still
 * talking" costs the whole turn.
 */
/**
 * How patient to be, as four steps rather than three sliders.
 *
 * The two knobs move TOGETHER or not at all. A long pause with a low
 * confidence bar is not "in between", it is the worst of both -- it waits, and
 * then ends the turn anyway on weak evidence. So each step raises both: wait
 * longer AND be surer before acting.
 *
 * The backstop grows with them, because it has to stay comfortably beyond the
 * pause it is backing up; at `Very patient` a 3s stop would fire before the
 * detector had finished being patient.
 */
export const PATIENCE = [
  {
    id: "eager",
    label: "Eager",
    detail: "Answers quickly. Cuts in if you think mid-sentence",
    silenceMs: 380,
    threshold: 0.6,
    maxSilenceMs: 2200,
  },
  {
    id: "balanced",
    label: "Balanced",
    detail: "Waits out a short pause",
    silenceMs: 600,
    threshold: 0.7,
    maxSilenceMs: 3000,
  },
  {
    id: "patient",
    label: "Patient",
    detail: "Room to think, at the cost of a beat before each answer",
    silenceMs: 900,
    threshold: 0.78,
    maxSilenceMs: 4000,
  },
  {
    id: "very-patient",
    label: "Very patient",
    detail: "Waits for a clear finish. Use the button when you want it sooner",
    silenceMs: 1300,
    threshold: 0.85,
    maxSilenceMs: 5200,
  },
] as const;

export type Patience = (typeof PATIENCE)[number]["id"];

export const DEFAULT_PATIENCE: Patience = "balanced";

/** Retune a detector that is already running. Safe before it has loaded. */
export function setPatience(id: Patience): void {
  const step = PATIENCE.find((p) => p.id === id) ?? PATIENCE[1];
  send({
    type: "config",
    silenceMs: step.silenceMs,
    threshold: step.threshold,
    maxSilenceMs: step.maxSilenceMs,
  });
}


/**
 * Switch it on, loading the models the first time.
 *
 * Safe to call repeatedly. The worker and its ~11MB of weights are kept once
 * loaded, because turning the setting off and on again is a toggle, not a
 * reason to download them twice.
 */
export function enableTurnDetection(
  onEvent: (event: TurnEvent) => void,
  patience: Patience = DEFAULT_PATIENCE,
): void {
  listener = onEvent;
  enabled = true;
  chosen = patience;

  if (worker || starting) {
    // Already here, or on its way. Say so, so a UI that just mounted is not
    // left waiting for an event that fired before it was listening.
    setPatience(patience);
    onEvent(ready ? { type: "ready" } : { type: "loading" });
    return;
  }

  starting = true;
  onEvent({ type: "loading" });
  // DYNAMICALLY IMPORTED, so onnxruntime-web is not in this route's compile.
  // Webpack resolves `new Worker(new URL(...))` statically wherever it appears,
  // which pulled ~10MB of ORT into /parley whether or not anybody switched
  // this on -- a 32-second first compile in dev, long enough for the browser
  // to give up. Behind `import()` it is a lazy chunk built on first use.
  void (async () => {
    try {
      const { createTurnWorker } = await import("./turnWorkerHost");
      worker = createTurnWorker();
    } catch (e) {
      onEvent({
        type: "error",
        detail:
          e instanceof Error ? e.message : "The turn detector could not start.",
      });
      starting = false;
      return;
    }
    starting = false;
    attach(onEvent);
  })();
}

/** Wire the worker up and start it loading. Split out so `enable` can await
 *  the import without the rest of it becoming async. */
function attach(onEvent: (event: TurnEvent) => void): void {
  if (!worker) return;

  worker.onmessage = (event: MessageEvent<FromWorker>) => {
    const message = event.data;
    if (message.type === "ready") {
      ready = true;
      if (enabled) listener?.({ type: "ready" });
      return;
    }
    // `speech` is deliberately dropped. The button's ring already shows input
    // level, and a second, differently-behaved liveness indicator beside it
    // explains nothing.
    if (message.type === "speech") return;
    if (!enabled) return;
    listener?.(message);
  };

  worker.onerror = (event) => {
    listener?.({
      type: "error",
      detail: event.message || "The turn detector stopped.",
    });
  };

  send({
    type: "init",
    // Our own origin: the models are committed under `public/models`, and
    // ORT's runtime is copied into `public/ort` at build time by
    // scripts/copyOrt.mjs. Nothing here reaches a third party.
    vadUrl: "/models/silero_vad.onnx",
    turnUrl: "/models/smart_turn_v3.onnx",
    wasmBase: "/ort/",
  });
  setPatience(chosen);
}

/** Stop acting on it. The worker and its weights are KEPT. */
export function disableTurnDetection(): void {
  enabled = false;
}

export function turnDetectionReady(): boolean {
  return ready;
}

/**
 * 16kHz mono, exactly as it is sent to the model.
 *
 * A no-op while disabled, so the audio path can call it unconditionally rather
 * than knowing anything about whether detection is on.
 */
export function feedTurnDetector(pcm: Int16Array): void {
  if (!enabled || !worker || !ready) return;
  // Int16 in, float out. The capture path already resampled to 16kHz for the
  // server, and re-deriving floats costs one divide per sample -- far less
  // than running the resampler twice, and 16-bit is well beyond what a VAD
  // can tell apart.
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) floats[i] = pcm[i] / 0x8000;
  // TRANSFERRED, not copied: this runs on every audio callback, and a
  // structured clone of every frame is garbage the audio thread pays for.
  send({ type: "audio", pcm: floats }, [floats.buffer]);
}

/** A new turn begins: forget the previous one's audio. */
export function resetTurnDetector(): void {
  if (!worker || !ready) return;
  send({ type: "reset" });
}

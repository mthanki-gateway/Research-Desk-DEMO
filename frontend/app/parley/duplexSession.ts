import { browserBase } from "@/lib/api";
import { getAccessToken } from "@/lib/supabase";
import {
  PATIENCE,
  type Patience,
  type TurnEvent,
  disableTurnDetection,
  enableTurnDetection,
  feedTurnDetector,
  onSpeechProbability,
  resetTurnDetector,
} from "./turnDetector";

/**
 * Duplex: the browser's half.
 *
 * THIS SIDE DECIDES NOTHING ABOUT ANSWERS. It streams the microphone up,
 * plays whatever audio comes down, and reports three facts about the speaker
 * from the models it already runs (Silero VAD and Smart Turn, in a worker):
 *
 *   speech_start   somebody began talking (or kept talking after a false end)
 *   pause          ~250 ms of silence -- the EARLY signal. The server brings
 *                  its transcript up to date and starts writing an answer.
 *   endpoint       Smart Turn says the sentence is finished. The server adopts
 *                  the answer it already started, or discards it if the
 *                  words changed.
 *
 * The microphone is open for the whole session, including while the model is
 * speaking: that is what makes it duplex, and what makes barge-in possible.
 */

const INPUT_RATE = 16_000;

/** Silero's own default. */
const SPEECH_THRESHOLD = 0.5;
/**
 * Stricter while the assistant is talking. The speaker's own voice reaches the
 * microphone even with echo cancellation, and without a higher bar it would
 * interrupt itself. Headphones remove the problem entirely.
 */
const BARGE_THRESHOLD = 0.85;
const BARGE_FRAMES = 4; // ~130 ms of it
/**
 * Voiced frames needed before a turn OPENS (~380 ms). A cough, a breath or a
 * chair is a frame or two; Whisper turns those into "Thank you." and "I'm
 * sorry." with full confidence, and the answer then follows. The server keeps
 * the second before the opening, so the first word is not lost to the wait.
 */
const ONSET_FRAMES = 12;

export type DuplexEvent =
  | { type: "ready"; stt: string; tts: string; chain: string[] }
  | { type: "partial"; text: string }
  | { type: "final"; text: string }
  | { type: "draft"; state: "started" | "discarded" | "adopted" | "fresh" | "skipped"; text?: string; model?: string; final?: boolean; reason?: string }
  | { type: "say"; text: string }
  | {
      type: "latency";
      endpoint_to_audio_ms: number;
      first_token_ms: number | null;
      model: string;
      ready_at_endpoint: boolean;
    }
  | { type: "resumed" }
  | { type: "stop"; reason: string }
  | { type: "stt"; name: string }
  | { type: "turn_end"; empty?: boolean }
  | { type: "error"; detail: string }
  // Local, not from the server:
  | { type: "level"; level: number }
  /** Silero's view, ~10 times a second, so a stuck turn can be diagnosed. */
  | { type: "vad"; p: number; speech: boolean; paused: boolean; silentMs: number }
  | { type: "endpoint"; reason: string }
  /** The device the browser actually opened, so a wrong one is visible. */
  | { type: "mic"; label: string }
  | { type: "speaking"; on: boolean }
  | { type: "detector"; state: "loading" | "ready" | "error"; detail?: string }
  | { type: "closed" };

export type DuplexSession = {
  interrupt: () => void;
  close: () => void;
};

export async function openDuplexSession(
  voiceName: string,
  patience: Patience,
  pauseMs: number,
  outputRate: number,
  onEvent: (event: DuplexEvent) => void,
  /** Which microphone. Empty = the system default. */
  deviceId = "",
): Promise<DuplexSession> {
  const token = await getAccessToken();
  const url = new URL(browserBase.replace(/^http/, "ws") + "/duplex/ws");
  if (token) url.searchParams.set("token", token);
  url.searchParams.set("voice_name", voiceName);

  const ws = new WebSocket(url.toString());
  ws.binaryType = "arraybuffer";

  // ---- playback: a queue on the audio clock --------------------------------
  const out = new AudioContext({ sampleRate: outputRate });
  let playHead = 0;
  let scheduled: AudioBufferSourceNode[] = [];
  // Audio arriving after a stop belongs to the cancelled answer; the server
  // may have a few chunks in flight. Dropped until the next `say`.
  let muted = false;
  let speakingTimer: number | null = null;

  const isSpeaking = () => scheduled.length > 0 || out.currentTime < playHead;

  function setSpeaking(on: boolean) {
    onEvent({ type: "speaking", on });
  }

  function enqueue(pcm: ArrayBuffer) {
    if (muted) return;
    const samples = new Int16Array(pcm);
    if (!samples.length) return;
    const buffer = out.createBuffer(1, samples.length, outputRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
      channel[i] = samples[i] / (samples[i] < 0 ? 0x8000 : 0x7fff);
    }
    const source = out.createBufferSource();
    source.buffer = buffer;
    source.connect(out.destination);
    const now = out.currentTime;
    // A little lead: the live voice generates at about real time, so a chunk
    // that arrives a hair late would otherwise land as a gap mid-word.
    if (playHead < now) {
      playHead = now + 0.12;
      setSpeaking(true);
    }
    source.start(playHead);
    scheduled.push(source);
    source.onended = () => {
      scheduled = scheduled.filter((n) => n !== source);
    };
    playHead += buffer.duration;
    if (speakingTimer) window.clearTimeout(speakingTimer);
    speakingTimer = window.setTimeout(
      () => {
        speakingTimer = null;
        if (!isSpeaking()) setSpeaking(false);
      },
      Math.max(0, playHead - out.currentTime) * 1000 + 80,
    ) as unknown as number;
  }

  function stopPlayback() {
    muted = true;
    for (const s of scheduled) {
      try {
        s.stop();
      } catch {
        // already finished
      }
    }
    scheduled = [];
    playHead = 0;
    if (speakingTimer) {
      window.clearTimeout(speakingTimer);
      speakingTimer = null;
    }
    setSpeaking(false);
  }

  // ---- capture --------------------------------------------------------------
  let stream: MediaStream | null = null;
  let input: AudioContext | null = null;
  let processor: ScriptProcessorNode | null = null;
  let acc = 0;
  let accN = 0;
  let debt = 0;

  const clamp = (x: number) => {
    const s = Math.max(-1, Math.min(1, x));
    return s < 0 ? s * 0x8000 : s * 0x7fff;
  };

  /** Device rate to 16 kHz Int16 with a box filter (see liveSession.ts). */
  function toModelRate(floats: Float32Array, rate: number): Int16Array<ArrayBuffer> {
    if (rate === INPUT_RATE) {
      const same = new Int16Array(floats.length);
      for (let i = 0; i < floats.length; i++) same[i] = clamp(floats[i]);
      return same;
    }
    const per = INPUT_RATE / rate;
    const buf = new Int16Array(Math.ceil(floats.length * per) + 1);
    let n = 0;
    for (let i = 0; i < floats.length; i++) {
      acc += floats[i];
      accN++;
      debt += per;
      if (debt >= 1) {
        buf[n++] = clamp(acc / accN);
        acc = 0;
        accN = 0;
        debt -= 1;
      }
    }
    return buf.slice(0, n);
  }

  function send(message: Record<string, unknown>) {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
  }

  // ---- what the VAD says ----------------------------------------------------
  let inSpeech = false;
  let paused = false;
  let bargeRun = 0;
  // SILENCE IS COUNTED IN AUDIO, NOT READ OFF A CLOCK. Each probability stands
  // for one 32 ms frame. The detector runs in a worker and its results can
  // arrive in bursts when it falls behind; timing the pause by when they
  // ARRIVED made a backlog look like continuous speech, and the turn never
  // ended.
  const FRAME_MS = 32;
  let silentMs = 0;
  let frames = 0;
  const fallbackMs = (PATIENCE.find((s) => s.id === patience) ?? PATIENCE[0]).silenceMs + 500;

  function endTurn(reason: string) {
    inSpeech = false;
    paused = false;
    silentMs = 0;
    send({ type: "endpoint" });
    onEvent({ type: "endpoint", reason });
    // The next turn starts from an empty window, not this one's tail.
    resetTurnDetector();
  }

  function onProbability(p: number) {
    const playing = isSpeaking();
    if (++frames % 3 === 0) {
      onEvent({ type: "vad", p, speech: inSpeech, paused, silentMs });
    }

    if (p >= (playing ? BARGE_THRESHOLD : SPEECH_THRESHOLD)) {
      silentMs = 0;
      if (!inSpeech) {
        // A run of speech, not one frame: shorter to cut in on the assistant,
        // longer to open a turn from silence.
        if (++bargeRun < (playing ? BARGE_FRAMES : ONSET_FRAMES)) return;
      }
      bargeRun = 0;
      if (!inSpeech) {
        inSpeech = true;
        paused = false;
        if (playing) stopPlayback();
        send({ type: "speech_start" });
      } else if (paused) {
        // Spoke again after a pause: the early answer may be for half a thought.
        paused = false;
      }
      return;
    }
    if (!inSpeech) {
      // Decays rather than resets: speech with a brief dip still counts.
      bargeRun = Math.max(0, bargeRun - 1);
      return;
    }
    silentMs += FRAME_MS;
    if (!paused && silentMs >= pauseMs) {
      paused = true;
      send({ type: "pause" });
    }
    // BACKSTOP. Smart Turn can decline to call a pause finished, and a turn
    // that never ends is the worst failure this mode has: it just listens. If
    // the speaker has been silent this long, the turn is over.
    if (paused && silentMs >= fallbackMs) endTurn("silence");
  }

  function onTurn(event: TurnEvent) {
    if (event.type === "loading") onEvent({ type: "detector", state: "loading" });
    else if (event.type === "ready") onEvent({ type: "detector", state: "ready" });
    else if (event.type === "error") onEvent({ type: "detector", state: "error", detail: event.detail });
    else if (event.type === "endpoint") {
      // As in Speak: the detector's word is final, whatever our own frame
      // counting thought. Gating this on `inSpeech` was how a turn could
      // never end. The server needs a turn open to close it, so open one.
      if (!inSpeech) send({ type: "speech_start" });
      endTurn(event.reason);
    }
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      enqueue(event.data);
      return;
    }
    try {
      const parsed = JSON.parse(event.data) as DuplexEvent;
      if (parsed.type === "say") muted = false;
      if (parsed.type === "stop") stopPlayback();
      onEvent(parsed);
    } catch {
      // not worth killing the session over
    }
  };
  ws.onerror = () => onEvent({ type: "error", detail: "The connection to the server failed." });

  function teardown() {
    onSpeechProbability(null);
    disableTurnDetection();
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
      processor = null;
    }
    void input?.close().catch(() => {});
    input = null;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    stopPlayback();
    void out.close().catch(() => {});
  }

  ws.onclose = () => {
    teardown();
    onEvent({ type: "closed" });
  };

  await new Promise<void>((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) return resolve();
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener("error", () => reject(new Error("Could not open the duplex session.")), {
      once: true,
    });
  });

  // The microphone, once, for the life of the session.
  const constraints = {
    // ON, and not optional here: this mode listens WHILE it speaks, so
    // without echo cancellation it hears itself.
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,
  };
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? { ...constraints, deviceId: { exact: deviceId } } : constraints,
    });
  } catch (err) {
    // The chosen device was unplugged since it was saved: use the default
    // rather than failing to start.
    if (deviceId && err instanceof DOMException && err.name === "OverconstrainedError") {
      stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
    } else {
      throw err;
    }
  }
  onEvent({ type: "mic", label: stream.getAudioTracks()[0]?.label || "unknown device" });
  input = new AudioContext();
  const source = input.createMediaStreamSource(stream);
  processor = input.createScriptProcessor(2048, 1, 1);
  source.connect(processor);
  processor.connect(input.destination);

  onSpeechProbability(onProbability);
  enableTurnDetection(onTurn, patience);

  processor.onaudioprocess = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const floats = e.inputBuffer.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < floats.length; i++) peak = Math.max(peak, Math.abs(floats[i]));
    const pcm = toModelRate(floats, input!.sampleRate);
    if (pcm.length) {
      ws.send(pcm);
      feedTurnDetector(pcm);
    }
    onEvent({ type: "level", level: peak });
  };

  return {
    interrupt() {
      stopPlayback();
      send({ type: "interrupt" });
    },
    close() {
      teardown();
      ws.close();
    },
  };
}

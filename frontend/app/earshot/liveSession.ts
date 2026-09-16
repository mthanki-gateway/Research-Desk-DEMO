import { browserBase } from "@/lib/api";
import { getAccessToken } from "@/lib/supabase";

/**
 * A live audio session: microphone out, speech back, over one socket.
 *
 * THE SHAPE OF THIS IS DIFFERENT FROM THE CASCADE IT REPLACES.
 *
 * The first Earshot recorded a whole question, POSTed a WAV, waited, and
 * played a WAV back. Request and response. This streams: audio leaves as it is
 * captured and arrives as it is generated, so the first sound comes back about
 * three seconds after the button rather than twelve to fifty.
 *
 * That forces two things the old design never had to solve:
 *
 *   1. PLAYBACK IS A QUEUE, not a file. Audio arrives in fragments with no
 *      length known in advance. Each one is scheduled to start exactly where
 *      the previous ended, on the AudioContext's own clock -- `play()` per
 *      chunk would leave audible seams, because setTimeout is not a clock.
 *
 *   2. CAPTURE AND PLAYBACK RUN AT DIFFERENT RATES. 16kHz up, 24kHz down,
 *      because that is what the model wants and what it produces. Two
 *      AudioContexts, one each, rather than resampling in JS.
 */

/** What the model wants in, and what it sends back. Neither is negotiable. */
const INPUT_RATE = 16_000;
const OUTPUT_RATE = 24_000;

export type LiveEvent =
  | { type: "ready"; voice: string }
  | { type: "heard"; text: string }
  | { type: "said"; text: string }
  | {
      type: "tool";
      tool: string;
      args: Record<string, unknown>;
      n: number;
      sources: { label: string; kind: "document" | "web"; url: string | null }[];
    }
  | { type: "turn_end" }
  | { type: "error"; detail: string }
  | { type: "closed" }
  /** Not from the server: emitted locally when the queue drains. */
  | { type: "playback_end" }
  /** Microphone level, for the button's ring. */
  | { type: "level"; level: number };

export type LiveSession = {
  /** Stop capturing and tell the server the turn is over. */
  endTurn: () => void;
  /** Start capturing again for the next question. */
  beginTurn: () => Promise<void>;
  /** Tear everything down: socket, microphone, playback. */
  close: () => void;
  /** Cut off whatever is being spoken right now. */
  stopSpeaking: () => void;
};

export async function openLiveSession(
  voiceName: string,
  onEvent: (event: LiveEvent) => void,
): Promise<LiveSession> {
  // The token travels in the query string because a browser CANNOT set headers
  // on a WebSocket -- there is no equivalent of fetch's `headers`. The server
  // note explains the trade.
  const token = await getAccessToken();
  const url = new URL(browserBase.replace(/^http/, "ws") + "/live/ws");
  if (token) url.searchParams.set("token", token);
  url.searchParams.set("voice_name", voiceName);

  const ws = new WebSocket(url.toString());
  ws.binaryType = "arraybuffer";

  // ---- playback ---------------------------------------------------------
  const out = new AudioContext({ sampleRate: OUTPUT_RATE });
  // When the next chunk should START. Kept on the audio clock, not wall time:
  // an AudioContext runs on its own high-resolution timeline and scheduling
  // against `Date.now()` drifts audibly within a couple of seconds.
  let playHead = 0;
  let playing = 0;

  function enqueue(pcm: ArrayBuffer) {
    const samples = new Int16Array(pcm);
    if (!samples.length) return;
    const buffer = out.createBuffer(1, samples.length, OUTPUT_RATE);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
      // Int16 to float. 0x8000 for negatives and 0x7fff for positives, which
      // is the asymmetry of two's complement -- dividing both by the same
      // number clips one end.
      channel[i] = samples[i] / (samples[i] < 0 ? 0x8000 : 0x7fff);
    }
    const source = out.createBufferSource();
    source.buffer = buffer;
    source.connect(out.destination);

    const now = out.currentTime;
    // A small lead on the first chunk. Scheduling at exactly `now` means any
    // hesitation in the next frame arriving lands as a gap mid-word.
    if (playHead < now) playHead = now + 0.08;
    source.start(playHead);
    playHead += buffer.duration;

    playing += 1;
    source.onended = () => {
      playing -= 1;
      if (playing === 0) onEvent({ type: "playback_end" });
    };
  }

  function stopSpeaking() {
    // Rebuilding the head is what actually stops queued audio: every scheduled
    // source is already committed to the graph, so the context is suspended
    // and resumed to flush them.
    void out.suspend().then(() => {
      playHead = 0;
      playing = 0;
      void out.resume();
    });
  }

  // ---- capture ----------------------------------------------------------
  let stream: MediaStream | null = null;
  let input: AudioContext | null = null;
  let processor: ScriptProcessorNode | null = null;
  let capturing = false;

  async function beginTurn() {
    if (capturing) return;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        // OFF: AGC hunts for a target level, so during a pause it winds the
        // gain up until room noise reaches speaking volume — which is the
        // input that makes a recogniser invent words.
        autoGainControl: false,
      },
    });
    input = new AudioContext({ sampleRate: INPUT_RATE });
    const source = input.createMediaStreamSource(stream);
    processor = input.createScriptProcessor(2048, 1, 1);
    source.connect(processor);
    // Connected to a sink with no audible effect: in some browsers a
    // ScriptProcessor that reaches no destination is never pulled by the graph
    // and its callback simply never fires.
    processor.connect(input.destination);

    processor.onaudioprocess = (e) => {
      if (!capturing || ws.readyState !== WebSocket.OPEN) return;
      const floats = e.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(floats.length);
      let peak = 0;
      for (let i = 0; i < floats.length; i++) {
        const s = Math.max(-1, Math.min(1, floats[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        peak = Math.max(peak, Math.abs(s));
      }
      ws.send(pcm.buffer);
      onEvent({ type: "level", level: peak });
    };
    capturing = true;
  }

  function releaseMicrophone() {
    capturing = false;
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
      processor = null;
    }
    // Closing the context is what turns the browser's recording indicator off.
    // Disconnecting the nodes alone leaves it lit, and a tab that looks like it
    // is still listening is alarming.
    void input?.close().catch(() => {});
    input = null;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  function endTurn() {
    releaseMicrophone();
    if (ws.readyState === WebSocket.OPEN) {
      // The SERVER appends the trailing silence the model needs to notice the
      // speaker stopped. Doing it here would send a second of silence over the
      // wire for no reason.
      ws.send(JSON.stringify({ type: "end" }));
    }
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      enqueue(event.data);
      return;
    }
    try {
      onEvent(JSON.parse(event.data) as LiveEvent);
    } catch {
      // A frame we cannot parse is not worth killing the session over.
    }
  };
  ws.onerror = () =>
    onEvent({ type: "error", detail: "The connection to the server failed." });
  ws.onclose = () => {
    releaseMicrophone();
    onEvent({ type: "closed" });
  };

  await new Promise<void>((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) return resolve();
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener(
      "error",
      () => reject(new Error("Could not open the live session.")),
      { once: true },
    );
  });

  return {
    beginTurn,
    endTurn,
    stopSpeaking,
    close() {
      releaseMicrophone();
      void out.close().catch(() => {});
      ws.close();
    },
  };
}

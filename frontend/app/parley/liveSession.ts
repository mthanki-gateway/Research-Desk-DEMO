import { type Mode, browserBase } from "@/lib/api";
import { getAccessToken } from "@/lib/supabase";

/**
 * A live audio session: microphone out, speech back, over one socket.
 *
 * THE SHAPE OF THIS IS DIFFERENT FROM THE CASCADE IT REPLACES.
 *
 * The first Parley recorded a whole question, POSTed a WAV, waited, and
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

/**
 * The conversation the socket appends to.
 *
 * The RESUME HANDLE IS NOT HELD HERE. It lives on the conversation row in
 * Postgres, keyed by the session id, because a handle in sessionStorage dies
 * with the tab -- which is exactly when somebody wants to pick a conversation
 * up again. The browser only has to remember WHICH conversation; the server
 * knows how to restore it.
 */
export type LiveEvent =
  | {
      type: "ready";
      voice: string;
      resumed: boolean;
      session_id: string;
      mode: Mode;
    }
  | { type: "heard"; text: string }
  | { type: "said"; text: string }
  | {
      type: "tool";
      tool: string;
      args: Record<string, unknown>;
      n: number;
      sources: { label: string; kind: "document" | "web"; url: string | null }[];
      /** `record_profile` only: the merged profile and what is still missing,
       *  so the card fills in as the interview happens rather than at the end. */
      profile?: Record<string, string | number | string[]>;
      missing?: string[];
      complete?: boolean;
    }
  /** A COMPLETED exchange, assembled server-side. The client stores this
   *  rather than reconstructing boundaries from streaming fragments, which it
   *  could only guess at and repeatedly got wrong. */
  | {
      type: "turn";
      question: string;
      answer: string;
      sources: { label: string; kind: "document" | "web"; url: string | null }[];
      tools: string[];
    }
  | { type: "turn_end" }
  /** The server has stored a resumption handle; the conversation is safe. */
  | { type: "resume" }
  /** The server is about to drop us. Reconnecting now keeps the context. */
  | { type: "going_away"; in: string }
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
  conversationId: string | null,
  mode: Mode,
  onEvent: (event: LiveEvent) => void,
): Promise<LiveSession> {
  // The token travels in the query string because a browser CANNOT set headers
  // on a WebSocket -- there is no equivalent of fetch's `headers`. The server
  // note explains the trade.
  const token = await getAccessToken();
  const url = new URL(browserBase.replace(/^http/, "ws") + "/live/ws");
  if (token) url.searchParams.set("token", token);
  url.searchParams.set("voice_name", voiceName);
  if (conversationId) url.searchParams.set("session_id", conversationId);
  // Picks the system prompt, server-side. Everything else is identical.
  url.searchParams.set("mode", mode);

  const ws = new WebSocket(url.toString());
  ws.binaryType = "arraybuffer";

  // ---- playback ---------------------------------------------------------
  const out = new AudioContext({ sampleRate: OUTPUT_RATE });
  // When the next chunk should START. Kept on the audio clock, not wall time:
  // an AudioContext runs on its own high-resolution timeline and scheduling
  // against `Date.now()` drifts audibly within a couple of seconds.
  let playHead = 0;
  /**
   * The model has finished GENERATING, per the server.
   *
   * Separate from "the speakers have gone quiet", and conflating the two was
   * a real bug: playback was declared over whenever the scheduled queue
   * momentarily drained, which happens constantly between chunks arriving over
   * a network. Every one of those gaps ended the turn, so a single long answer
   * was committed to the transcript as three or four separate exchanges -- and
   * only the first of them kept the question, because the rest began on a
   * freshly reset turn. On screen that read as "nothing intelligible".
   */
  let generated = false;
  let drainTimer: number | null = null;

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
    // More audio after the server called the turn complete simply pushes the
    // finish out; without this the tail of a long answer is cut off.
    scheduleFinish();

  }

  /**
   * Announce the end of the turn once the audio has actually finished.
   *
   * `playHead` is the exact moment the last scheduled chunk ends, on the
   * AudioContext's own clock -- so the remaining time is known rather than
   * guessed at, and no per-source bookkeeping is needed. Rescheduled on every
   * new chunk, because more audio can still arrive after the server says the
   * model has stopped generating.
   */
  function scheduleFinish() {
    if (!generated) return;
    if (drainTimer) window.clearTimeout(drainTimer);
    const remaining = Math.max(0, playHead - out.currentTime);
    drainTimer = window.setTimeout(
      () => {
        drainTimer = null;
        onEvent({ type: "playback_end" });
      },
      remaining * 1000 + 60,
    ) as unknown as number;
  }

  function stopSpeaking() {
    // Rebuilding the head is what actually stops queued audio: every scheduled
    // source is already committed to the graph, so the context is suspended
    // and resumed to flush them.
    if (drainTimer) {
      window.clearTimeout(drainTimer);
      drainTimer = null;
    }
    generated = false;
    void out.suspend().then(() => {
      playHead = 0;
      void out.resume();
    });
  }

  // ---- capture ----------------------------------------------------------
  let stream: MediaStream | null = null;
  let input: AudioContext | null = null;
  let processor: ScriptProcessorNode | null = null;
  let capturing = false;

  /**
   * Build the capture graph ONCE, and keep it for the whole conversation.
   *
   * THE BUG THIS FIXES. Every turn used to call `getUserMedia` and build a
   * fresh AudioContext and ScriptProcessor. Opening a capture device takes
   * anywhere from 100 to 500 milliseconds, and the countdown starts the next
   * turn automatically -- so the speaker is very often already talking while
   * the microphone is still opening, and those opening words are never
   * captured at all.
   *
   * The first turn always looked fine because it is the one the user starts by
   * clicking, and then speaks. Every turn after it lost its beginning, which
   * reached the screen as a mangled transcript or a single stray word: a whole
   * spoken sentence arriving as "7".
   *
   * So the graph is built on the first turn and reused. `capturing` gates
   * whether frames are SENT, which is a boolean rather than a device.
   */
  async function ensureGraph(): Promise<void> {
    if (input) return;

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
  }

  async function beginTurn() {
    if (capturing) return;
    // Cancel any pending end-of-playback from the PREVIOUS turn.
    if (drainTimer) {
      window.clearTimeout(drainTimer);
      drainTimer = null;
    }
    generated = false;

    await ensureGraph();

    // THE TURN OPENS HERE, not when audio starts arriving.
    //
    // Automatic activity detection is disabled server-side, so the model is
    // not listening for speech to begin -- it is waiting to be told. Without
    // this marker the audio is accepted and nothing is ever treated as a turn.
    //
    // Sent BEFORE `capturing` is set, so no frame can reach the model outside
    // an open turn.
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "start" }));
    }
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
    // The graph STAYS UP. Tearing it down here is what made the next turn miss
    // its opening words; `capturing` alone decides whether frames are sent.
    capturing = false;
    if (ws.readyState === WebSocket.OPEN) {
      // THE ONLY THING THAT ENDS A TURN. The model does not decide; a pause
      // does not decide; this click decides.
      ws.send(JSON.stringify({ type: "end" }));
    }
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      enqueue(event.data);
      return;
    }
    try {
      const parsed = JSON.parse(event.data) as LiveEvent;
      if (parsed.type === "turn_end") {
        // The server says the model has stopped generating. The turn is over
        // when the audio ALREADY QUEUED has finished playing, which may be
        // several seconds later.
        generated = true;
        scheduleFinish();
      }
      onEvent(parsed);
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

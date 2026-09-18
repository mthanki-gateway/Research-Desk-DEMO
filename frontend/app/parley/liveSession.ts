import { type Mode, browserBase } from "@/lib/api";
import { feedTurnDetector, resetTurnDetector } from "./turnDetector";
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
      /** `end_interview` only: the model has closed the conversation. */
      ended?: boolean;
      summary?: string;
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
      /** What the model RECORDED from this turn. Accurate, where the
       *  transcript is not — it comes from the thing that heard the audio. */
      recorded?: Record<string, unknown>[];
    }
  | { type: "turn_end" }
  /** The interview is closed because the PARTICIPANT ended it. */
  | { type: "finished" }
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
  /** Ask the model to open the conversation. It speaks first. */
  greet: () => void;
  /** Stop capturing and tell the server the turn is over. */
  endTurn: () => void;
  /** Start capturing again for the next question. */
  beginTurn: () => Promise<void>;
  /** Tear everything down: socket, microphone, playback. */
  close: () => void;
  /** Cut off whatever is being spoken right now. */
  stopSpeaking: () => void;
  /** The PARTICIPANT ending the interview, not the interviewer. Closes the
   *  conversation server-side and stops the link working again. */
  finish: () => void;
  /**
   * Hand the microphone back WITHOUT closing the conversation.
   *
   * The capture graph is built once and kept between turns on purpose --
   * rebuilding it costs 100-500ms and swallows the first words of every turn
   * after the first. The cost of keeping it is that the browser's recording
   * indicator stays lit, which is correct between two turns and alarming once
   * a conversation has finished or been paused: the tab looks like it is
   * still listening, because it is.
   *
   * So the two are separated. This releases the device and leaves the socket
   * and the conversation intact; the next `beginTurn` rebuilds the graph, and
   * pays the opening-words cost exactly where it never mattered -- a turn the
   * user starts by pressing a button and then speaking.
   */
  releaseMic: () => void;
};

export async function openLiveSession(
  voiceName: string,
  conversationId: string | null,
  mode: Mode,
  onEvent: (event: LiveEvent) => void,
  /**
   * A Howler magic link, for somebody with no account.
   *
   * MUTUALLY EXCLUSIVE with signing in, and deliberately so. An invite grants
   * exactly one conversation -- not a session, not an identity -- and the
   * server resolves it down its own path rather than through the bearer
   * token, so that it can never widen into one. Given an invite, the mode,
   * the conversation and the schema all come from the project behind it, and
   * everything passed here is ignored.
   */
  invite?: string,
): Promise<LiveSession> {
  // The token travels in the query string because a browser CANNOT set headers
  // on a WebSocket -- there is no equivalent of fetch's `headers`. The server
  // note explains the trade.
  const token = invite ? null : await getAccessToken();
  const url = new URL(browserBase.replace(/^http/, "ws") + "/live/ws");
  if (invite) url.searchParams.set("invite", invite);
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
  /**
   * Every chunk scheduled for the current turn.
   *
   * Kept so they can actually be STOPPED. Suspending the context only defers
   * them -- resuming plays the whole backlog -- so "stop speaking" left the
   * model talking a moment later, with the chunks still arriving over the
   * socket queued on top.
   */
  let scheduled: AudioBufferSourceNode[] = [];
  /**
   * The user cut this turn off.
   *
   * Audio still arriving is DISCARDED until the next turn begins. The model
   * does not know it was interrupted and keeps sending, and enqueuing that is
   * the other half of why the voice came back.
   */
  let muted = false;

  function enqueue(pcm: ArrayBuffer) {
    if (muted) return;
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
    scheduled.push(source);
    // Dropped once played, or the array grows for the life of the session and
    // holds every buffer in it.
    source.onended = () => {
      scheduled = scheduled.filter((node) => node !== source);
    };
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
    // A muted turn has already ended as far as the user is concerned. Letting
    // its real ending through fired `playback_end` seconds later and RESTARTED
    // the countdown the stop had just begun.
    if (!generated || muted) return;
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
    if (drainTimer) {
      window.clearTimeout(drainTimer);
      drainTimer = null;
    }
    generated = false;
    // MUTED, NOT PAUSED. Suspending the context only defers the backlog, and
    // resuming played all of it -- so the model carried on talking a second
    // after being told to stop. Every scheduled chunk is stopped outright, and
    // anything still arriving for this turn is dropped.
    muted = true;
    for (const source of scheduled) {
      try {
        source.stop();
      } catch {
        // Already finished. Nothing to stop.
      }
    }
    scheduled = [];
    playHead = 0;
  }

  // ---- capture ----------------------------------------------------------
  let stream: MediaStream | null = null;
  let input: AudioContext | null = null;
  let processor: ScriptProcessorNode | null = null;
  let capturing = false;

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
    // THE CONTEXT TAKES THE DEVICE'S OWN RATE, and we resample ourselves.
    //
    // `new AudioContext({ sampleRate: 16000 })` followed by
    // `createMediaStreamSource` is a CHROME-ONLY pattern. Chrome and Safari
    // quietly resample the microphone into the context; FIREFOX THROWS --
    // "Connecting AudioNodes from AudioContexts with different sample-rate is
    // currently not supported" (Bugzilla 1725336, still open). The throw came
    // out of `ensureGraph`, so every turn failed as "the microphone was
    // refused" on Firefox while working perfectly in Chrome.
    //
    // Asking for the rate as a getUserMedia constraint is not the fix either:
    // Firefox does not implement the sampleRate constraint (Bugzilla 1388586).
    // So the only portable answer is to take whatever the device gives --
    // 48000 on most Linux machines -- and do the conversion in JS.
    input = new AudioContext();
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
      let peak = 0;
      for (let i = 0; i < floats.length; i++) {
        peak = Math.max(peak, Math.abs(floats[i]));
      }
      const pcm = toModelRate(floats, input!.sampleRate);
      if (pcm.length) {
        ws.send(pcm);
        // The SAME samples the model gets, so the detector is judging exactly
        // what was sent rather than a parallel reading of the microphone.
        feedTurnDetector(pcm);
      }
      onEvent({ type: "level", level: peak });
    };
  }

  /**
   * Device rate to the model's 16kHz, as signed 16-bit PCM.
   *
   * A BOX FILTER, not sample-dropping: averaging the samples that collapse
   * into one output sample low-passes as it decimates, and skipping two of
   * every three instead folds everything above 8kHz back down into the speech
   * band as aliasing -- which a recogniser hears as a lisp.
   *
   * The accumulator is deliberately OUTSIDE the function. A ScriptProcessor
   * hands over 2048 samples at a time and 2048 does not divide evenly by the
   * ratio, so resetting per callback would round the boundary off every 43
   * milliseconds and leave a periodic tick through the whole recording.
   */
  let acc = 0;
  let accN = 0;
  let debt = 0;

  // `Int16Array<ArrayBuffer>`, spelled out: the bare `Int16Array` resolves to
  // `Int16Array<ArrayBufferLike>`, and `WebSocket.send` will not take one --
  // it might be backed by a SharedArrayBuffer, which is not transferable.
  function toModelRate(
    floats: Float32Array,
    rate: number,
  ): Int16Array<ArrayBuffer> {
    // Already there, or close enough that resampling would only add error.
    if (rate === INPUT_RATE) {
      const same = new Int16Array(floats.length);
      for (let i = 0; i < floats.length; i++) same[i] = clamp(floats[i]);
      return same;
    }

    const per = INPUT_RATE / rate; // output samples produced per input sample
    const out = new Int16Array(Math.ceil(floats.length * per) + 1);
    let n = 0;

    for (let i = 0; i < floats.length; i++) {
      acc += floats[i];
      accN++;
      debt += per;
      if (debt >= 1) {
        out[n++] = clamp(acc / accN);
        acc = 0;
        accN = 0;
        debt -= 1;
      }
    }
    // `slice`, not `subarray`: a copy with its own exactly-sized buffer. A
    // view would carry the over-allocated tail along with it, and anything
    // reading `.buffer` downstream would find however many unused zeros were
    // left at the end of it.
    return out.slice(0, n);
  }

  function clamp(sample: number): number {
    const s = Math.max(-1, Math.min(1, sample));
    return s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  async function beginTurn() {
    if (capturing) return;
    // Cancel any pending end-of-playback from the PREVIOUS turn.
    if (drainTimer) {
      window.clearTimeout(drainTimer);
      drainTimer = null;
    }
    generated = false;
    // A new turn: audio is wanted again.
    muted = false;

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
    // A new turn starts with an empty ring. Otherwise the tail of the previous
    // answer is still in the 8-second window and the detector is asked whether
    // a sentence nobody is speaking has finished.
    resetTurnDetector();
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
    greet() {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "greet" }));
      }
    },
    beginTurn,
    endTurn,
    stopSpeaking,
    releaseMic: releaseMicrophone,
    finish() {
      // Microphone first. Whatever is captured after this decision is not
      // wanted, and leaving it live records somebody's reaction to having
      // pressed the button.
      releaseMicrophone();
      stopSpeaking();
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "finish" }));
      }
    },
    close() {
      releaseMicrophone();
      void out.close().catch(() => {});
      ws.close();
    },
  };
}

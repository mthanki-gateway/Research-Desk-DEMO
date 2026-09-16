import { TARGET_RATE, downsample, toWav } from "../transcribe/recorder";

/**
 * Record from a button press to a button release. No gating, no cutting.
 *
 * DELIBERATELY NOT THE VAD RECORDER NEXT DOOR.
 *
 * `record()` in the Model Lab exists to slice continuous dictation into
 * segments at natural pauses. That is the right behaviour for a transcript and
 * exactly the wrong behaviour for a conversation: a pause mid-question would
 * end the turn and send half a sentence to be answered. Every piece of
 * machinery that makes the other recorder good -- the noise-floor calibration,
 * the silence timer, the minimum segment length -- is machinery for deciding
 * when the speaker has finished, and here the speaker says so by clicking.
 *
 * What IS shared is the encoding: downsampling and the WAV header live in one
 * place, because two implementations of a 44-byte header is two chances to get
 * the byte order wrong in only one of them.
 *
 * The meter is kept, for a different reason than the other recorder needs it:
 * not to detect speech, but so the user can see the microphone is live. A
 * recording UI with no visible response to sound is indistinguishable from a
 * broken one, and the user finds out only after speaking a whole question.
 */

export type Level = { level: number; peak: number };

export type PushToTalk = {
  /** Stop capturing and hand back everything recorded. Null if silent. */
  stop: () => Promise<{ wav: Blob; seconds: number } | null>;
  /** Abandon the recording and release the microphone. */
  cancel: () => void;
};

/** Below this, the take is treated as silence and not sent. */
const SILENT_PEAK = 0.008;

export function pushToTalk(
  stream: MediaStream,
  onLevel?: (level: Level) => void,
): PushToTalk {
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;

  // ScriptProcessorNode is deprecated in favour of AudioWorklet and is used
  // anyway, for the reason given in the other recorder: a worklet needs a
  // separate module fetched at runtime, which is more moving parts than the
  // capture itself. Supported everywhere, with no removal date announced.
  const processor = ctx.createScriptProcessor(4096, 1, 1);

  source.connect(analyser);
  source.connect(processor);
  // Connected to a sink with no audible effect: in some browsers a
  // ScriptProcessor that reaches no destination is not pulled by the graph,
  // and its callback simply never fires.
  processor.connect(ctx.destination);

  let chunks: Float32Array[] = [];
  let peak = 0;
  let stopped = false;

  processor.onaudioprocess = (e) => {
    if (stopped) return;
    // Copied, not referenced: the node reuses one buffer for every callback,
    // so keeping the reference leaves every chunk pointing at the newest audio.
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  const levels = new Float32Array(analyser.fftSize);
  const meter = window.setInterval(() => {
    if (stopped || !onLevel) return;
    analyser.getFloatTimeDomainData(levels);
    let sum = 0;
    let frame = 0;
    for (let i = 0; i < levels.length; i++) {
      sum += levels[i] * levels[i];
      frame = Math.max(frame, Math.abs(levels[i]));
    }
    peak = Math.max(peak, frame);
    onLevel({ level: Math.sqrt(sum / levels.length), peak });
  }, 50);

  const release = () => {
    stopped = true;
    window.clearInterval(meter);
    processor.onaudioprocess = null;
    processor.disconnect();
    source.disconnect();
    // Closing the context is what actually turns the microphone indicator off
    // in the browser chrome. Disconnecting the nodes alone leaves it lit, and
    // a tab that appears to still be listening is alarming.
    void ctx.close().catch(() => {});
    stream.getTracks().forEach((t) => t.stop());
  };

  return {
    async stop() {
      const rate = ctx.sampleRate;
      release();

      const total = chunks.reduce((n, c) => n + c.length, 0);
      if (!total) return null;
      const joined = new Float32Array(total);
      let at = 0;
      for (const c of chunks) {
        joined.set(c, at);
        at += c.length;
      }
      chunks = [];

      if (peak < SILENT_PEAK) return null;

      // Downsampled to 16kHz before encoding. Speech carries nothing above
      // 8kHz, and the browser captures at 44.1 or 48kHz, so sending the native
      // rate triples the upload for no gain in what the model hears.
      const samples = downsample(joined, rate, TARGET_RATE);
      return {
        wav: toWav(samples, TARGET_RATE),
        seconds: samples.length / TARGET_RATE,
      };
    },
    cancel: release,
  };
}

/**
 * Ask for the microphone.
 *
 * `autoGainControl` is OFF for the same reason as in the Model Lab: AGC hunts
 * for a target level, so during a pause it winds the gain up until room noise
 * reaches speaking volume. Here that costs accuracy rather than triggering
 * false speech detection, but it is the same bad input either way.
 */
export async function openMicrophone(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
    },
  });
}

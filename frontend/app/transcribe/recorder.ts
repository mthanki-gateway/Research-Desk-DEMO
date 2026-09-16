/**
 * Microphone capture that produces WAV, with voice detection on the same graph.
 *
 * WHY NOT MediaRecorder
 *
 * MediaRecorder gives WebM/Opus, and NVIDIA Riva cannot decode it:
 *
 *     Audio decoder exception: Request config encoding not specified and
 *     could not detect encoding from audio content
 *
 * Whisper on Groq accepts WebM, so the first version worked and the second
 * provider broke. Transcoding server-side would mean ffmpeg in the image and a
 * subprocess per segment; asking the browser for WAV is not possible, because
 * MediaRecorder has no WAV mode.
 *
 * So this reads raw PCM off the audio graph and writes the WAV header itself.
 * WAV is what BOTH providers accept natively, which is the point.
 *
 * THREE OTHER PROBLEMS DISAPPEAR WITH IT
 *
 *   - No container-fragment hack. MediaRecorder's `timeslice` mode emits
 *     fragments where only the first carries the WebM header, so segments had
 *     to be produced by stopping and restarting the recorder. Here a segment
 *     is just a slice of a sample buffer.
 *   - Sample-accurate boundaries. The restart cycle dropped a few milliseconds
 *     at every cut; this drops none.
 *   - One audio graph instead of two. The detector and the recorder were
 *     reading the same microphone through separate objects.
 *
 * The cost is bandwidth: WAV is uncompressed, so 10 seconds of 16kHz mono is
 * ~320KB where Opus would be ~10KB. On localhost against a batch API that is
 * not a real constraint.
 */

/** Tunables, in one place because they are all about the same trade. */
export const VAD = {
  /** How often the level is sampled. 50ms is well under a syllable. */
  FRAME_MS: 50,
  /**
   * How far above the measured noise floor counts as speech.
   *
   * A MULTIPLE of the floor, not an absolute level: a fixed threshold that
   * works in a quiet room treats a noisy one as continuous speech, and one
   * tuned for the noisy room goes deaf in the quiet one.
   */
  SPEECH_MULTIPLIER: 2.2,
  /** Absolute floor, so a silent mic cannot make the threshold ~0. */
  MIN_THRESHOLD: 0.006,
  /** Ambient measured for this long before recording starts. */
  CALIBRATION_MS: 600,
  /**
   * Least audio in a segment, regardless of pauses.
   *
   * THE DETECTOR GATES, IT DOES NOT CUT SHORT. Ending a segment at the first
   * 700ms pause produced two-second clips from anyone who pauses mid-sentence,
   * and short clips are what both models are worst on -- Whisper was trained
   * on 30-second windows, so thin context is where it guesses.
   */
  MIN_SEGMENT_MS: 10_000,
  /** Silence that ends a segment, once it is already long enough. */
  SILENCE_HOLD_MS: 700,
  /** Least SPEECH worth sending. Below this it is a cough or a chair. */
  MIN_SPEECH_MS: 400,
  /** Hard cap, for someone who does not pause. Whisper truncates past 30s. */
  MAX_SEGMENT_MS: 25_000,
} as const;

/**
 * Sample rate sent to the providers.
 *
 * Both ASR families are trained on 16kHz speech, so sending the hardware's
 * usual 48kHz wastes three times the bandwidth to be downsampled at the other
 * end anyway.
 */
export const TARGET_RATE = 16_000;

export type Meter = { level: number; speaking: boolean };

export type Recorder = {
  stop: () => void;
  /** Voiced ms in the segment running RIGHT NOW, for a manual flush. */
  voicedMs: () => number;
  /** Loudest frame of the current segment, the fallback for a manual flush. */
  peak: () => number;
  /** Cut the current segment now and hand it over. */
  flush: () => void;
};

/** Average `factor` input samples into one output sample. */
export function downsample(input: Float32Array, from: number, to: number): Float32Array {
  if (to >= from) return input;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    // AVERAGED, not picked. Taking every Nth sample aliases high frequencies
    // down into the speech band as a metallic buzz, which is audible to a
    // person and measurably worse for a recogniser.
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

/**
 * Trim silence from both ends and lift the level to a consistent peak.
 *
 * WHY, given the segment is already gated on speech: the gate says a segment
 * CONTAINS speech, not that it is mostly speech. A 10-second segment holding
 * two seconds of quiet talking and eight of room tone is a plausible thing to
 * hand an ASR model, and the NeMo models are measurably less forgiving of it
 * than Whisper -- Whisper was trained on 680k hours of messy web audio and
 * shrugs; Parakeet returns an empty string.
 *
 * Normalisation is peak-based rather than RMS: it is the cheap version, it
 * cannot clip, and what matters here is getting a quiet microphone into the
 * range these models were trained on rather than matching a loudness target.
 *
 * Leaves a short margin at each end. Cutting exactly at the threshold clips
 * the onset of the first word and the tail of the last, which are the
 * consonants a recogniser needs most.
 */
function clean(samples: Float32Array, threshold: number): Float32Array {
  const margin = Math.floor(TARGET_RATE * 0.15); // 150ms
  let first = 0;
  let last = samples.length - 1;
  while (first < samples.length && Math.abs(samples[first]) < threshold) first++;
  while (last > first && Math.abs(samples[last]) < threshold) last--;
  if (first >= last) return samples; // nothing above threshold; send as-is

  const start = Math.max(0, first - margin);
  const end = Math.min(samples.length, last + margin);
  const out = samples.slice(start, end);

  let peak = 0;
  for (let i = 0; i < out.length; i++) peak = Math.max(peak, Math.abs(out[i]));
  // 0.9 rather than 1.0, leaving headroom so rounding into int16 cannot wrap.
  // Capped at 8x so a segment of near-silence is not amplified into noise
  // loud enough to be mistaken for speech.
  if (peak > 0.001) {
    const gain = Math.min(0.9 / peak, 8);
    if (gain > 1) for (let i = 0; i < out.length; i++) out[i] *= gain;
  }
  return out;
}

/** Wrap float samples in a 16-bit PCM WAV container. */
export function toWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    // Clamped before scaling: a sample slightly outside [-1, 1] would wrap to
    // the opposite extreme as an int16, which is heard as a loud click.
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function record(
  stream: MediaStream,
  handlers: {
    /** A finished segment: WAV audio plus how much of it was speech. */
    onSegment: (wav: Blob, voicedMs: number) => void;
    onMeter?: (meter: Meter) => void;
    onReady?: (threshold: number) => void;
  },
): Recorder {
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;

  // ScriptProcessorNode is deprecated in favour of AudioWorklet, and is used
  // anyway: an AudioWorklet needs a separate module file fetched at runtime,
  // which in this bundler setup is more moving parts than the capture itself.
  // It is supported everywhere today, and the deprecation has no removal date.
  const processor = ctx.createScriptProcessor(4096, 1, 1);

  source.connect(analyser);
  source.connect(processor);
  // Connected to the destination with no audible effect -- a ScriptProcessor
  // that reaches no sink is not pulled by the graph in some browsers, so its
  // callback silently never fires.
  processor.connect(ctx.destination);

  const levelBuf = new Float32Array(analyser.fftSize);
  let threshold: number = VAD.MIN_THRESHOLD;
  let calibrating = true;
  const ambient: number[] = [];

  let chunks: Float32Array[] = [];
  let voicedMs = 0;
  let silenceMs = 0;
  let segmentMs = 0;
  let sawSpeech = false;
  let peak = 0;
  const started = performance.now();

  processor.onaudioprocess = (e) => {
    if (calibrating) return; // discard the ambient-measurement audio
    // Copied, not referenced: the node reuses its buffer every callback, so
    // keeping the reference would leave every chunk holding the newest audio.
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  const cut = (): { wav: Blob; voiced: number } | null => {
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const joined = new Float32Array(total);
    let at = 0;
    for (const c of chunks) {
      joined.set(c, at);
      at += c.length;
    }
    const captured = voicedMs;
    chunks = [];
    voicedMs = 0;
    silenceMs = 0;
    segmentMs = 0;
    sawSpeech = false;
    peak = 0;
    if (!total) return null;
    return {
      wav: toWav(
        clean(downsample(joined, ctx.sampleRate, TARGET_RATE), threshold),
        TARGET_RATE,
      ),
      voiced: captured,
    };
  };

  const timer = window.setInterval(() => {
    analyser.getFloatTimeDomainData(levelBuf);
    let sum = 0;
    for (let i = 0; i < levelBuf.length; i++) sum += levelBuf[i] * levelBuf[i];
    const rms = Math.sqrt(sum / levelBuf.length);

    if (calibrating) {
      ambient.push(rms);
      if (performance.now() - started >= VAD.CALIBRATION_MS) {
        // MEDIAN, not mean: one cough during calibration would drag a mean up
        // and leave the threshold deaf for the whole session.
        const sorted = [...ambient].sort((a, b) => a - b);
        const floor = sorted[Math.floor(sorted.length / 2)] || 0;
        threshold = Math.max(floor * VAD.SPEECH_MULTIPLIER, VAD.MIN_THRESHOLD);
        calibrating = false;
        handlers.onReady?.(threshold);
      }
      handlers.onMeter?.({ level: rms, speaking: false });
      return;
    }

    const speaking = rms > threshold;
    if (rms > peak) peak = rms;
    handlers.onMeter?.({ level: rms, speaking });

    segmentMs += VAD.FRAME_MS;
    if (speaking) {
      voicedMs += VAD.FRAME_MS;
      silenceMs = 0;
      sawSpeech = true;
    } else if (sawSpeech) {
      silenceMs += VAD.FRAME_MS;
    }

    const longEnough = segmentMs >= VAD.MIN_SEGMENT_MS;
    const pauseEnded = longEnough && sawSpeech && silenceMs >= VAD.SILENCE_HOLD_MS;
    const tooLong = segmentMs >= VAD.MAX_SEGMENT_MS;
    if (!pauseEnded && !tooLong) return;

    const segment = cut();
    // The speech gate lives here rather than in the caller so an automatic cut
    // with nothing in it never becomes a request. Ten seconds of empty room is
    // precisely the input that makes Whisper answer "Thank you."
    if (segment && segment.voiced >= VAD.MIN_SPEECH_MS) {
      handlers.onSegment(segment.wav, segment.voiced);
    }
  }, VAD.FRAME_MS);

  return {
    stop: () => {
      window.clearInterval(timer);
      processor.onaudioprocess = null;
      processor.disconnect();
      analyser.disconnect();
      source.disconnect();
      void ctx.close();
    },
    voicedMs: () => voicedMs,
    peak: () => peak,
    flush: () => {
      // READ peak BEFORE cutting: `cut` resets it, so testing afterwards would
      // always compare against zero and reject every manual flush in a room
      // too quiet to cross the speech threshold -- the exact case this branch
      // exists for.
      const loudest = peak;
      const segment = cut();
      if (!segment) return;
      // A MANUAL cut carries the user's intent -- they spoke, then reached for
      // Stop -- so the bar drops to "was there audible sound at all". In a
      // noisy room the threshold can sit above a quiet speaker and leave
      // `voiced` at zero for speech that plainly happened.
      if (segment.voiced > 0 || loudest >= VAD.MIN_THRESHOLD) {
        handlers.onSegment(segment.wav, segment.voiced);
      }
    },
  };
}

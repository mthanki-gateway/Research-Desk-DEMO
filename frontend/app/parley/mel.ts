/**
 * Whisper's log-mel spectrogram, in the browser.
 *
 * WHAT A "MEL" IS
 *
 * A mel spectrogram is a picture of sound shaped the way hearing works. Split
 * the audio into 25ms slices, take the frequency content of each, then squash
 * those frequencies onto the MEL SCALE -- fine detail low down, where humans
 * tell 200Hz from 300Hz easily, and progressively coarser higher up, where
 * 5000Hz and 5100Hz are indistinguishable. Eighty bands of that, eight hundred
 * slices deep, is the [80, 800] image this file produces.
 *
 * WHY THIS EXISTS AT ALL
 *
 * Smart Turn v3 is a Whisper Tiny encoder with a classifier head, so its ONNX
 * graph takes `input_features` of shape [1, 80, 800] -- NOT a waveform. The
 * Python reference gets those from `WhisperFeatureExtractor(chunk_length=8)`.
 * There is no equivalent in the browser, so it is reimplemented here.
 *
 * IT FAILS SILENTLY WHEN IT IS WRONG, which is why this file is so heavily
 * commented. A filterbank off by a scale factor, a periodic window where a
 * symmetric one was meant, padding on the wrong side -- none of them throw.
 * They produce a correctly shaped tensor of plausible numbers, the model
 * returns confident probabilities, and turn detection becomes a coin flip.
 * Every constant below is Whisper's, and every one of them matters.
 *
 * Checked against a numpy implementation of the same spec in
 * `backend/app/scripts/mel_reference.py`, via `scripts/verifyMel.mjs`.
 */

/** Whisper's, and not negotiable -- the encoder was trained on these. */
export const SAMPLE_RATE = 16_000;
export const N_FFT = 400;
export const HOP = 160;
export const N_MELS = 80;

/** Smart Turn looks at the last 8 seconds, so 8 * 16000 samples. */
export const WINDOW_SAMPLES = 8 * SAMPLE_RATE;
/** 128000 / 160 = 800 frames, which is the 800 in [1, 80, 800]. */
export const N_FRAMES = WINDOW_SAMPLES / HOP;

/* --------------------------------------------------------------- mel filters */

/** Hz to mel, on the Slaney scale Whisper uses (NOT the HTK one).
 *
 * Linear below 1kHz, logarithmic above. Getting this wrong is the classic
 * silent failure: HTK's formula is a plausible-looking one-liner that shifts
 * every filter and leaves the model reading a spectrogram of the wrong shape.
 */
function hzToMel(hz: number): number {
  const minLogHz = 1000;
  const minLogMel = 15;
  const step = Math.log(6.4) / 27;
  return hz >= minLogHz
    ? minLogMel + Math.log(hz / minLogHz) / step
    : (3 * hz) / 200;
}

function melToHz(mel: number): number {
  const minLogMel = 15;
  const minLogHz = 1000;
  const step = Math.log(6.4) / 27;
  return mel >= minLogMel
    ? minLogHz * Math.exp(step * (mel - minLogMel))
    : (200 * mel) / 3;
}

type MelFilter = {
  /** First FFT bin with a non-zero weight. */
  start: number;
  /** The non-zero weights, from `start` onwards. */
  weights: Float32Array;
};

/**
 * The 80 triangular filters, built once.
 *
 * Area-normalised ("slaney" norm) -- each is scaled by 2/(hz[i+2] - hz[i]) so
 * a flat spectrum gives a flat mel response rather than one rising with
 * frequency.
 */
function melFilters(): MelFilter[] {
  const nBins = N_FFT / 2 + 1; // 201
  // n_mels + 2 points, evenly spaced in MEL space, converted back to Hz.
  const melMin = hzToMel(0);
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const points = new Float64Array(N_MELS + 2);
  for (let i = 0; i < points.length; i++) {
    points[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }

  // The centre frequency of every FFT bin.
  const binHz = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) binHz[i] = (i * SAMPLE_RATE) / N_FFT;

  const filters: MelFilter[] = [];
  for (let m = 0; m < N_MELS; m++) {
    const left = points[m];
    const centre = points[m + 1];
    const right = points[m + 2];
    // Area normalisation, applied per filter rather than to the whole bank.
    const norm = 2 / (right - left);

    // SPARSE, and this is where the time went. A triangular filter touches a
    // handful of the 201 bins -- the lowest ones span two or three -- so the
    // dense form spent 800 x 80 x 201 = 12.8 million multiply-adds per
    // spectrogram, almost all of them by zero. Keeping only the non-zero span
    // is the difference between this being the dominant cost and being
    // background noise next to the FFT.
    const dense = new Float64Array(nBins);
    let start = -1;
    let end = -1;
    for (let i = 0; i < nBins; i++) {
      const hz = binHz[i];
      const rising = (hz - left) / (centre - left);
      const falling = (right - hz) / (right - centre);
      const weight = Math.max(0, Math.min(rising, falling));
      dense[i] = weight * norm;
      if (weight > 0) {
        if (start < 0) start = i;
        end = i;
      }
    }
    if (start < 0) {
      // A filter narrower than the bin spacing. Legal at the very bottom of
      // the scale, and it contributes nothing.
      filters.push({ start: 0, weights: new Float32Array(0) });
      continue;
    }
    const weights = new Float32Array(end - start + 1);
    for (let i = start; i <= end; i++) weights[i - start] = dense[i];
    filters.push({ start, weights });
  }
  return filters;
}

/**
 * The SYMMETRIC Hann window, matching numpy's `hanning(N)` with
 * `periodic=False`.
 *
 * torch.hann_window defaults to PERIODIC (divides by N, not N-1), and Whisper
 * passes `periodic=True`. One sample of difference at each end, which is
 * exactly the kind of thing that never shows up as an error.
 */
function hannWindow(): Float32Array {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N_FFT));
  }
  return w;
}

/* ---------------------------------------------------------------------- FFT */

/**
 * In-place iterative radix-2 FFT. 400 is not a power of two, so the frame is
 * zero-padded to 512 and only the first 201 bins are read -- which is what
 * `n_fft=400` with a 512-point transform would give, and matches the
 * reference's bin centres because those depend on N_FFT, not on the padding.
 */
const FFT_SIZE = 512;

/**
 * Precomputed twiddle factors and bit-reversal indices.
 *
 * Built once, because this FFT runs 800 times for every decision -- once per
 * frame of the 8-second window. Recomputing `cos`/`sin` inside the butterfly
 * loop cost more than the transform itself: 119ms per spectrogram before this,
 * and a spectrogram is on the path between somebody finishing a sentence and
 * the model answering.
 */
const REV = (() => {
  const rev = new Uint16Array(FFT_SIZE);
  for (let i = 1, j = 0; i < FFT_SIZE; i++) {
    let bit = FFT_SIZE >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    rev[i] = j;
  }
  return rev;
})();

const TWIDDLE = (() => {
  // One entry per (stage, k), flattened. The stages need 1 + 2 + 4 + ... +
  // FFT_SIZE/2 angles, which is FFT_SIZE - 1 in total -- NOT FFT_SIZE/2. Sized
  // at half that, the later stages read past the end, every lookup came back
  // undefined and the whole spectrogram went NaN.
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  let offset = 0;
  for (let len = 2; len <= FFT_SIZE; len <<= 1) {
    const half = len / 2;
    for (let k = 0; k < half; k++) {
      const ang = (-2 * Math.PI * k) / len;
      re[offset + k] = Math.cos(ang);
      im[offset + k] = Math.sin(ang);
    }
    offset += half;
  }
  return { re, im };
})();

/** In-place iterative radix-2 FFT over `FFT_SIZE` points.
 *
 * 400 is not a power of two, so each frame is zero-padded to 512 and only the
 * first 201 bins are read. The bin CENTRES still come from N_FFT rather than
 * the padding, which is why they line up with the filterbank.
 */
function fft(re: Float64Array, im: Float64Array): void {
  for (let i = 1; i < FFT_SIZE; i++) {
    const j = REV[i];
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  let offset = 0;
  for (let len = 2; len <= FFT_SIZE; len <<= 1) {
    const half = len / 2;
    for (let i = 0; i < FFT_SIZE; i += len) {
      for (let k = 0; k < half; k++) {
        const wRe = TWIDDLE.re[offset + k];
        const wIm = TWIDDLE.im[offset + k];
        const a = i + k;
        const b = a + half;
        const vRe = re[b] * wRe - im[b] * wIm;
        const vIm = re[b] * wIm + im[b] * wRe;
        re[b] = re[a] - vRe;
        im[b] = im[a] - vIm;
        re[a] += vRe;
        im[a] += vIm;
      }
    }
    offset += half;
  }
}

/* ------------------------------------------------------------------ features */

let FILTERS: MelFilter[] | null = null;
let WINDOW: Float32Array | null = null;

/**
 * 8 seconds of 16kHz mono audio to the [80, 800] tensor Smart Turn expects,
 * flattened row-major (all 800 frames of mel 0, then mel 1, ...).
 *
 * `audio` is taken from the END: the question is whether the speaker has just
 * finished, so the most recent 8 seconds is the evidence.
 */
export function logMelSpectrogram(audio: Float32Array): Float32Array {
  FILTERS ??= melFilters();
  WINDOW ??= hannWindow();

  // The last 8 seconds, RIGHT-padded -- real audio first, silence after.
  //
  // Left-padding was my own reasoning ("do not put silence where the model is
  // looking for the end of an utterance") and it was WRONG, because it is not
  // how the model was trained: the reference truncates to the last 8 seconds
  // and then pads with `padding="max_length"`, which pads on the right.
  const samples = new Float32Array(WINDOW_SAMPLES);
  const take = Math.min(audio.length, WINDOW_SAMPLES);
  samples.set(audio.subarray(audio.length - take), 0);

  // `do_normalize=True`, over the REAL SAMPLES ONLY, with the padding left at
  // zero -- `zero_mean_unit_var_norm` normalises `vector[:length]` and then
  // restores `padding_value` over the rest.
  //
  // Including the padding in the statistics makes a short clip look quieter
  // and flatter than it is, and since silence reads as 0.99 "turn complete",
  // that biases towards ending a turn exactly when the buffer is mostly
  // padding -- which is the first seconds of every turn. Together with the
  // padding side, that is why turn detection fired almost immediately.
  let mean = 0;
  for (let i = 0; i < take; i++) mean += samples[i];
  mean /= take || 1;
  let variance = 0;
  for (let i = 0; i < take; i++) {
    const d = samples[i] - mean;
    variance += d * d;
  }
  const std = Math.sqrt(variance / (take || 1) + 1e-7);
  for (let i = 0; i < take; i++) samples[i] = (samples[i] - mean) / std;

  const nBins = N_FFT / 2 + 1;
  const out = new Float32Array(N_MELS * N_FRAMES);
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  const power = new Float64Array(nBins);

  let maxLog = -Infinity;

  for (let frame = 0; frame < N_FRAMES; frame++) {
    const start = frame * HOP;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      // REFLECT padding at the edges, which is what the reference's centred
      // STFT does. Zero padding here darkens the first and last few frames.
      let idx = start + i - N_FFT / 2;
      if (idx < 0) idx = -idx;
      if (idx >= WINDOW_SAMPLES) idx = 2 * (WINDOW_SAMPLES - 1) - idx;
      re[i] = samples[idx] * WINDOW[i];
    }
    fft(re, im);
    for (let b = 0; b < nBins; b++) power[b] = re[b] * re[b] + im[b] * im[b];

    for (let m = 0; m < N_MELS; m++) {
      const { start: from, weights } = FILTERS[m];
      let sum = 0;
      for (let b = 0; b < weights.length; b++) sum += weights[b] * power[from + b];
      // log10 with a floor, exactly as in the reference.
      const value = Math.log10(Math.max(sum, 1e-10));
      out[m * N_FRAMES + frame] = value;
      if (value > maxLog) maxLog = value;
    }
  }

  // The two lines that make it Whisper's rather than any log-mel: clamp to
  // 8 decades below the loudest bin, then map roughly onto [-1, 1].
  const floor = maxLog - 8;
  for (let i = 0; i < out.length; i++) {
    out[i] = (Math.max(out[i], floor) + 4) / 4;
  }
  return out;
}

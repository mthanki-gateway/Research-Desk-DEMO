/**
 * Does the browser's log-mel agree with the numpy one?
 *
 * `app/parley/mel.ts` reimplements Whisper's feature extractor so Smart Turn
 * v3 can run in a tab. It is the kind of code that fails SILENTLY: the wrong
 * mel scale, a periodic window, a missing area normalisation -- none of them
 * throw, they just hand the model a plausible tensor of the wrong numbers.
 *
 * So the same spec is implemented twice, once here in TypeScript and once in
 * numpy (`backend/app/scripts/mel_reference.py`), and this compares them on
 * identical input. Agreement does not prove either matches PyTorch -- both are
 * ours -- but it catches coding mistakes, which is the failure that actually
 * happens.
 *
 * Run:
 *   docker compose exec web npx tsc app/parley/mel.ts --outDir /tmp/mel \
 *       --target es2022 --module esnext --moduleResolution bundler
 *   docker compose exec web node scripts/verifyMel.mjs
 */

import { logMelSpectrogram, N_FRAMES, N_MELS } from "../.mel-check/mel.js";

const SAMPLE_RATE = 16_000;

/** Byte-for-byte the reference's `test_audio()`.
 *
 * The filler is trigonometric rather than random ON PURPOSE. numpy's PCG64
 * cannot be reproduced here, and a different noise realisation is not
 * cosmetic: the tensor is clamped 8 decades below its loudest bin, so a
 * quieter filler pushes more cells onto the floor and moves the mean of the
 * whole thing. That looked like a mel bug and was not one.
 */
function testAudio() {
  const n = 3 * SAMPLE_RATE;
  const audio = new Float32Array(n);
  const last = (n - 1) / SAMPLE_RATE;
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const freq = 80 + ((3800 - 80) * t) / last;
    const sweep = Math.sin(2 * Math.PI * freq * t);
    const envelope = Math.sin((Math.PI * t) / last) ** 2;
    const filler = 0.02 * Math.sin(i * 0.9137) * Math.cos(i * 1.7231);
    audio[i] = Math.fround(sweep * envelope + filler);
  }
  return audio;
}

const REFERENCE = {
  shape: [80, 800],
  mean: -0.28363463282585144,
  std: 0.22207999229431152,
  min: -0.3271459937095642,
  max: 1.6728540658950806,
  row_means: {
    0: -0.326822966337204,
    20: -0.3096446096897125,
    40: -0.2967405617237091,
    60: -0.2782483994960785,
    79: -0.2747458815574646,
  },
  // Frames past ~520 are the right-hand padding, so they sit on the clamp
  // floor. That they ARE the floor is itself the check that the audio was
  // placed at the start rather than the end.
  col_means: {
    0: 0.03339647501707077,
    520: -0.3271460235118866,
    600: -0.3271460235118866,
    700: -0.3271460235118866,
    799: -0.3271460235118866,
  },
  argmax: [64, 133],
};

const mel = logMelSpectrogram(testAudio());

let mean = 0;
let min = Infinity;
let max = -Infinity;
for (const v of mel) {
  mean += v;
  if (v < min) min = v;
  if (v > max) max = v;
}
mean /= mel.length;
let variance = 0;
for (const v of mel) variance += (v - mean) ** 2;
const std = Math.sqrt(variance / mel.length);

const at = (m, f) => mel[m * N_FRAMES + f];

// Tight, because the two now share their input exactly. Only double-precision
// and Float32 rounding separate them; a real mistake -- wrong mel scale, wrong
// window, transposed output -- moves these by whole units.
const TOLERANCE = 1e-4;
const rows = [];
let failed = 0;

function check(name, got, want) {
  const delta = Math.abs(got - want);
  const ok = delta <= TOLERANCE;
  if (!ok) failed++;
  rows.push(
    `${ok ? "ok  " : "FAIL"} ${name.padEnd(14)} js=${got.toFixed(7).padStart(11)}  py=${want
      .toFixed(7)
      .padStart(11)}  d=${delta.toExponential(1)}`,
  );
}

const shapeOk = mel.length === N_MELS * N_FRAMES;
rows.push(
  `${shapeOk ? "ok  " : "FAIL"} shape          js=[${N_MELS}, ${N_FRAMES}]  py=[${REFERENCE.shape}]`,
);
if (!shapeOk) failed++;

check("mean", mean, REFERENCE.mean);
check("std", std, REFERENCE.std);
check("min", min, REFERENCE.min);
check("max", max, REFERENCE.max);

// A per-row mean catches a wrong mel scale or filterbank; a per-frame mean
// catches a wrong window or hop; the argmax catches a transpose. Single cells
// mostly sit on the clamp floor and would agree however wrong the code was.
for (const [m, want] of Object.entries(REFERENCE.row_means)) {
  let sum = 0;
  for (let f = 0; f < N_FRAMES; f++) sum += at(Number(m), f);
  check(`row ${m}`, sum / N_FRAMES, want);
}
for (const [f, want] of Object.entries(REFERENCE.col_means)) {
  let sum = 0;
  for (let m = 0; m < N_MELS; m++) sum += at(m, Number(f));
  check(`frame ${f}`, sum / N_MELS, want);
}

let bestIdx = 0;
for (let i = 1; i < mel.length; i++) if (mel[i] > mel[bestIdx]) bestIdx = i;
const argmax = [Math.floor(bestIdx / N_FRAMES), bestIdx % N_FRAMES];
const argmaxOk =
  argmax[0] === REFERENCE.argmax[0] && argmax[1] === REFERENCE.argmax[1];
if (!argmaxOk) failed++;
rows.push(
  `${argmaxOk ? "ok  " : "FAIL"} argmax         js=[${argmax}]  py=[${REFERENCE.argmax}]`,
);

console.log(rows.join("\n"));
console.log(`\n${failed === 0 ? "AGREES" : `${failed} MISMATCH(ES)`}`);
process.exit(failed === 0 ? 0 : 1);

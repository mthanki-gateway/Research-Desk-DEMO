/**
 * Does the pair actually run, and what do they return?
 *
 * The mel is verified numerically against numpy in `verifyMel.mjs`. This is
 * the other half: that both graphs load, accept the tensors we build, come
 * back with finite numbers, and do it fast enough to sit in a turn loop.
 *
 * It does NOT prove the semantic judgement is any good -- that needs a real
 * voice, and synthesised audio tells you nothing about whether a sentence
 * sounded finished. What it does prove is that nothing is silently mis-shaped.
 */
import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";
import { logMelSpectrogram, N_FRAMES, N_MELS, SAMPLE_RATE } from "../.mel-check/mel.js";

ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const vad = await ort.InferenceSession.create(readFileSync("public/models/silero_vad.onnx"));
const turn = await ort.InferenceSession.create(readFileSync("public/models/smart_turn_v3.onnx"));

/* ---- Silero, frame by frame ------------------------------------------- */

const FRAME = 512; // Silero v5 wants exactly 512 samples at 16kHz
let state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

async function speechProb(frame) {
  const out = await vad.run({
    input: new ort.Tensor("float32", frame, [1, frame.length]),
    state,
    sr: new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []),
  });
  state = out.stateN;
  return out.output.data[0];
}

function tone(n, hz, amp = 0.3) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = amp * Math.sin((2 * Math.PI * hz * i) / SAMPLE_RATE);
  return a;
}

const silence = new Float32Array(FRAME);
let quiet = 0;
for (let i = 0; i < 20; i++) quiet = await speechProb(silence);

state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
let loud = 0;
for (let i = 0; i < 20; i++) {
  // Vaguely voice-shaped: a 140Hz buzz with formant-ish harmonics.
  const f = new Float32Array(FRAME);
  for (const [hz, amp] of [[140, 0.4], [700, 0.25], [1220, 0.15], [2600, 0.05]]) {
    const t = tone(FRAME, hz, amp);
    for (let j = 0; j < FRAME; j++) f[j] += t[j];
  }
  loud = await speechProb(f);
}

console.log("Silero VAD");
console.log(`  silence      p(speech) = ${quiet.toFixed(4)}`);
console.log(`  voiced buzz  p(speech) = ${loud.toFixed(4)}`);
console.log(`  separates    ${loud > quiet ? "yes" : "NO"}`);

/* ---- Smart Turn, on 8 seconds ----------------------------------------- */

async function endpoint(audio) {
  const features = logMelSpectrogram(audio);
  const out = await turn.run({
    input_features: new ort.Tensor("float32", features, [1, N_MELS, N_FRAMES]),
  });
  return out.logits.data;
}

const clips = {
  silence: new Float32Array(2 * SAMPLE_RATE),
  "steady tone": tone(2 * SAMPLE_RATE, 220),
  "tone, cut off": (() => {
    const a = tone(2 * SAMPLE_RATE, 220);
    return a.subarray(0, a.length); // ends abruptly at full amplitude
  })(),
  "tone, faded out": (() => {
    const a = tone(2 * SAMPLE_RATE, 220);
    const tail = SAMPLE_RATE / 2;
    for (let i = 0; i < tail; i++) {
      a[a.length - tail + i] *= 1 - i / tail;
    }
    return a;
  })(),
};

console.log("\nSmart Turn v3");
let shape = null;
for (const [name, clip] of Object.entries(clips)) {
  const t0 = performance.now();
  const data = await endpoint(clip);
  const ms = performance.now() - t0;
  shape ??= data.length;
  console.log(
    `  ${name.padEnd(16)} logits=[${[...data].map((v) => v.toFixed(4)).join(", ")}]  ${ms.toFixed(0)}ms`,
  );
}

const t0 = performance.now();
const ROUNDS = 10;
for (let i = 0; i < ROUNDS; i++) await endpoint(clips["steady tone"]);
console.log(`\n  mel + inference: ${((performance.now() - t0) / ROUNDS).toFixed(1)}ms per call`);
console.log(`  output width   : ${shape}`);

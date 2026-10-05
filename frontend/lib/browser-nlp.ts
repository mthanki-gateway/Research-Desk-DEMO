/**
 * Zero-shot NLP that runs in the visitor's browser, with no server involved.
 *
 * WHY IN THE BROWSER. The hosted API runs on Render's free tier, which cannot
 * hold torch -- so the server-side GLiNER/GLiClass path exists only in the
 * local Docker stack. Here the model runs on the visitor's own device through
 * ONNX Runtime Web (WebAssembly),
 * which fits any hosting tier because the host does nothing, and makes
 * "nothing leaves the machine" literally true.
 *
 * Loaded ONLY by the Local page, via dynamic import: transformers.js and the
 * ONNX runtime are a large bundle, and no other page should pay for them.
 *
 * entities   onnx-community/gliner_small-v2.1 (183MB int8) -- the SAME
 *            model the server runs. transformers.js has no GLiNER pipeline,
 *            so the pre- and post-processing below is a port of GLiNER.js's
 *            span-mode code (v0.0.19). That package itself is not used: it
 *            has not been released since March 2025 and pins an older
 *            transformers. Checked against the server's Python GLiNER: same
 *            spans, same labels, scores a few points lower from int8.
 *
 * ROUTING AND JEV ARE NOT HERE, deliberately. GLiClass has no browser
 * export, and the NLI zero-shot models that do (deberta-v3-xsmall and
 * ModernBERT-base zeroshot) scored "the app crashes on upload" as technical
 * support at 0.17 and 0.10 -- a router that falls back on the easiest case
 * is not worth shipping. A self-exported GLiClass matched exactly at fp32
 * (747MB) and collapsed to ~0.8 on every label at int8.
 *
 * Downloaded once, then kept in the Cache API.
 */

import type * as OrtNs from "onnxruntime-web";

export type Progress = { file: string; loaded: number; total: number };
type OnProgress = (p: Progress) => void;

const GLINER = "onnx-community/gliner_small-v2.1";
const GLINER_ONNX = `https://huggingface.co/${GLINER}/resolve/main/onnx/model_int8.onnx`;
// From the model's gliner_config.json.
const MAX_WIDTH = 12;
const MAX_WORDS = 384;
const CACHE = "research-desk-models-v1";

export const MODELS = { gliner: GLINER };

// ---- entities: GLiNER, ported -------------------------------------------

type Tokenizer = {
  encode(text: string): number[];
  sep_token_id: number;
};

let gliner: Promise<{ ort: typeof OrtNs; session: OrtNs.InferenceSession; tokenizer: Tokenizer }> | null =
  null;

async function fetchCached(url: string, onProgress?: OnProgress): Promise<Uint8Array> {
  // The Cache API, not the HTTP cache: a 183MB response is routinely evicted
  // from the HTTP cache, and re-downloading it on every visit would make the
  // page unusable on anything but fast wifi.
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Model download failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onProgress?.({ file: "model_int8.onnx", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) {
    bytes.set(p, at);
    at += p.length;
  }
  try {
    await cache?.put(url, new Response(bytes));
  } catch {
    // Quota exceeded or private mode: it works, just without the cache.
  }
  return bytes;
}

function loadGliner(onProgress?: OnProgress) {
  gliner ??= (async () => {
    const [ort, { AutoTokenizer }] = await Promise.all([
      import("onnxruntime-web"),
      import("@huggingface/transformers"),
    ]);
    // The same runtime files transformers.js uses, from the same CDN, so the
    // WebAssembly binary is fetched once for both models.
    if (typeof window !== "undefined")
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
    const [tokenizer, bytes] = await Promise.all([
      AutoTokenizer.from_pretrained(GLINER),
      fetchCached(GLINER_ONNX, onProgress),
    ]);
    // WASM, not WebGPU, for this one: int8 matmuls are unsupported on most
    // WebGPU backends and silently fall back op by op, which is slower than
    // just running on the CPU.
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return { ort, session, tokenizer: tokenizer as unknown as Tokenizer };
  })();
  gliner.catch(() => (gliner = null));
  return gliner;
}

const WORDS = /\w+(?:[-_]\w+)*|\S/g;

export async function entities(
  text: string,
  labels: string[],
  threshold: number,
  onProgress?: OnProgress,
): Promise<{
  entities: { text: string; label: string; score: number; start: number; end: number }[];
  elapsed_ms: number;
  model: string;
}> {
  const { ort, session, tokenizer } = await loadGliner(onProgress);
  const start = performance.now();

  // 1. Words, with character offsets so spans map back onto the text.
  const words: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  for (const m of text.matchAll(WORDS)) {
    if (words.length >= MAX_WORDS) break;
    words.push(m[0]);
    starts.push(m.index!);
    ends.push(m.index! + m[0].length);
  }
  const n = words.length;
  if (n === 0) return { entities: [], elapsed_ms: 0, model: GLINER };

  // 2. Prompt: <<ENT>> label <<ENT>> label ... <<SEP>> word word ...
  const prompt: string[] = [];
  for (const l of labels) prompt.push("<<ENT>>", l);
  prompt.push("<<SEP>>");
  const sequence = [...prompt, ...words];

  // 3. Subword ids, and a words_mask marking the FIRST subtoken of each text
  //    word with its 1-based index -- that is how the model pools subtokens
  //    back into words. Prompt tokens and later subtokens get 0.
  const ids: number[] = [1];
  const wordsMask: number[] = [0];
  let w = 1;
  sequence.forEach((word, i) => {
    const toks = tokenizer.encode(word).slice(1, -1);
    toks.forEach((t, j) => {
      ids.push(t);
      wordsMask.push(i >= prompt.length && j === 0 ? w++ : 0);
    });
  });
  ids.push(tokenizer.sep_token_id);
  wordsMask.push(0);

  // 4. Every candidate span up to MAX_WIDTH words, starting at every word.
  const spanIdx: number[] = [];
  const spanMask: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < MAX_WIDTH; j++) {
      const e = Math.min(i + j, n - 1);
      spanIdx.push(i, e);
      spanMask.push(i + j < n ? 1 : 0);
    }
  }

  const i64 = (a: number[]) => BigInt64Array.from(a, (x) => BigInt(x));
  const T = ids.length;
  const S = n * MAX_WIDTH;
  const out = await session.run({
    input_ids: new ort.Tensor("int64", i64(ids), [1, T]),
    attention_mask: new ort.Tensor("int64", i64(ids.map(() => 1)), [1, T]),
    words_mask: new ort.Tensor("int64", i64(wordsMask), [1, T]),
    text_lengths: new ort.Tensor("int64", i64([n]), [1, 1]),
    span_idx: new ort.Tensor("int64", i64(spanIdx), [1, S, 2]),
    span_mask: new ort.Tensor("bool", Uint8Array.from(spanMask), [1, S]),
  });
  const logits = out.logits.data as Float32Array;

  // 5. Decode. Layout is [1, n, MAX_WIDTH, labels]; sigmoid each logit.
  const k = labels.length;
  type Span = { s: number; e: number; label: string; p: number };
  const found: Span[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < MAX_WIDTH; j++) {
      const e = i + j;
      if (e >= n) continue;
      for (let c = 0; c < k; c++) {
        const p = 1 / (1 + Math.exp(-logits[(i * MAX_WIDTH + j) * k + c]));
        if (p >= threshold) found.push({ s: i, e, label: labels[c], p });
      }
    }
  }

  // 6. Greedy flat NER: highest score first, drop anything overlapping a
  //    span already kept. Same rule as GLiNER's own decoder.
  found.sort((a, b) => b.p - a.p);
  const kept: Span[] = [];
  for (const sp of found) {
    if (kept.some((q) => !(sp.s > q.e || q.s > sp.e))) continue;
    kept.push(sp);
  }
  kept.sort((a, b) => a.s - b.s);

  const elapsed = performance.now() - start;
  return {
    entities: kept.map((sp) => ({
      text: text.slice(starts[sp.s], ends[sp.e]),
      label: sp.label,
      score: Math.round(sp.p * 1e4) / 1e4,
      start: starts[sp.s],
      end: ends[sp.e],
    })),
    elapsed_ms: Math.round(elapsed * 10) / 10,
    model: GLINER,
  };
}

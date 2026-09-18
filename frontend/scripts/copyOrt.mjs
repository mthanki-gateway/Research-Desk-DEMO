/**
 * Put onnxruntime-web's WASM where the browser can fetch it.
 *
 * ORT loads its runtime as a separate .wasm at execution time, and neither
 * webpack nor Turbopack emits it for us. The options were to point
 * `wasmPaths` at a CDN -- a third-party host on the critical path of a feature
 * that is meant to run locally -- or to serve it ourselves. This copies it out
 * of node_modules into `public/ort`, which Next serves at /ort.
 *
 * GENERATED, NOT COMMITTED. It is 14MB and it already exists in the lockfile;
 * committing it would put a second copy of a dependency in git and let the two
 * drift on the next `npm update`. `public/models` IS committed, because those
 * are not a dependency of anything and a build that has to reach Hugging Face
 * is a build that fails when Hugging Face does.
 *
 * Runs from `predev` and `prebuild`, so it happens in the container, on the
 * host and on a deploy without anybody remembering it.
 */
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, "..", "node_modules", "onnxruntime-web", "dist");
const to = join(here, "..", "public", "ort");

// The plain SIMD+threaded pair, which is what the `onnxruntime-web/wasm` entry
// asks for. The default entry would want the `.jsep` pair instead -- 27MB
// rather than 13.6MB, for WebGPU and WebNN that two tiny models have no use
// for. If the worker's import ever changes, this list has to change with it:
// ORT fetches its runtime by name at execution time, so a mismatch is not a
// build error, it is a 404 at the moment somebody switches the feature on.
//
// Threads additionally need cross-origin isolation, which we do not set --
// without it ORT falls back to a single thread using this same file.
const FILES = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

if (!existsSync(from)) {
  console.warn("[ort] onnxruntime-web is not installed; skipping");
  process.exit(0);
}

mkdirSync(to, { recursive: true });
for (const file of FILES) {
  copyFileSync(join(from, file), join(to, file));
}
console.log(`[ort] copied ${FILES.length} files to public/ort`);

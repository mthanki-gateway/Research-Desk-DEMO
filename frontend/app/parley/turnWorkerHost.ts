/**
 * Constructing the worker, in a module of its own.
 *
 * WHY THIS IS NOT INLINE IN `turnDetector.ts`
 *
 * Webpack resolves `new Worker(new URL(...))` STATICALLY. It does not care
 * that the call sits behind a setting nobody has switched on -- seeing it
 * anywhere in a route's module graph is enough to pull the worker, and
 * therefore onnxruntime-web, into that route's compile.
 *
 * ORT is about 10MB of JavaScript, and the result was a 32-second first
 * compile of /parley in dev, long enough that the browser gave up before the
 * page arrived. Behind a dynamic `import()` this becomes a lazy chunk that is
 * only built when somebody actually turns auto-turns on.
 */
export function createTurnWorker(): Worker {
  return new Worker(new URL("./turnWorker.ts", import.meta.url));
}

/**
 * Node ESM loader hook that retries a failed import with `.js` appended.
 *
 * `@material/material-color-utilities` is TypeScript compiled without
 * extension rewriting, so its internal imports look like
 * `./dynamiccolor/dynamic_scheme` — legal for bundlers, rejected by Node's ESM
 * resolver. `--experimental-specifier-resolution=node` used to paper over this
 * but was removed in Node 20.
 *
 * Twelve lines here is cheaper than adding esbuild or tsx just to run one
 * build-time script.
 */

export async function resolve(specifier, context, next) {
  try {
    return await next(specifier, context);
  } catch (error) {
    if (
      error?.code === "ERR_MODULE_NOT_FOUND" &&
      !specifier.endsWith(".js") &&
      !specifier.endsWith(".json")
    ) {
      return next(`${specifier}.js`, context);
    }
    throw error;
  }
}

"use client";

import { ACCENTS } from "@/lib/accents";
import { useApp } from "./providers";
import { Ripplable } from "./md";

/**
 * Accent swatches. **Development only.**
 *
 * A shipped product has one accent, chosen by whoever designed it — a colour
 * picker in the navigation drawer says "we could not decide", and it is one
 * more control between the user and the thing they came to do. This is a tool
 * for picking the default, not a feature.
 *
 * Gated on NODE_ENV rather than a new env var: Next substitutes it at build
 * time, so the whole component and its swatch list are dead-code-eliminated
 * from the production bundle rather than merely hidden. Same instinct as the
 * backend's `docs_enabled` — dev affordances should not exist in prod, not
 * just be invisible there.
 *
 * Each swatch is painted with its own seed as a literal hex rather than
 * `var(--md-primary)`, because every swatch has to show a DIFFERENT palette
 * while only one is active — a token would resolve them all to the current
 * accent and render six identical dots.
 *
 * Switching writes one attribute on <html>; the whole palette is already in
 * the stylesheet under `[data-accent="…"]`, so there is no re-render, no
 * refetch and no colour maths at runtime.
 */
export default function AccentPicker() {
  const { accent, setAccent } = useApp();

  if (process.env.NODE_ENV !== "development") return null;

  return (
    <div className="px-4 py-2">
      <span
        className="md-label-medium mb-2 block"
        style={{ color: "var(--md-nav-on-surface-variant)" }}
      >
        Accent
      </span>
      <div
        className="flex flex-wrap gap-1.5"
        role="radiogroup"
        aria-label="Accent colour"
      >
        {ACCENTS.map(({ id, label, seed }) => {
          const active = accent === id;
          return (
            <Ripplable
              as="button"
              type="button"
              key={id}
              onClick={() => setAccent(id)}
              role="radio"
              aria-checked={active}
              aria-label={label}
              title={label}
              className="md-state grid h-7 w-7 place-items-center rounded-full"
              style={{
                background: seed,
                // The selected mark is a ring OUTSIDE the swatch, so it never
                // covers the colour being chosen.
                boxShadow: active
                  ? "0 0 0 2px var(--md-surface), 0 0 0 4px var(--md-on-surface)"
                  : "none",
              }}
            >
              {active && (
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#fff"
                  strokeWidth={3.2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-3.5 w-3.5"
                  aria-hidden="true"
                >
                  <path d="m5 13 4 4L19 7" />
                </svg>
              )}
            </Ripplable>
          );
        })}
      </div>
    </div>
  );
}

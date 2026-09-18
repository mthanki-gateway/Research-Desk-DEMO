/**
 * Generate Material 3 colour role tokens from a single source colour.
 *
 *   node scripts/gen-theme.mjs
 *
 * M3 does not use hand-picked hex values. A source colour is converted to HCT
 * (hue / chroma / tone), five tonal palettes are derived from it, and 13 tones
 * of each are mapped onto semantic ROLES (primary, on-primary, surface-
 * container-high, ...). Light and dark schemes come from the same palettes at
 * different tones, which is why M3 dark mode looks deliberate rather than
 * inverted.
 *
 * Run at build time, output committed, so the app ships static CSS with no
 * runtime colour maths and no dependency in the container.
 */

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  argbFromHex,
  hexFromArgb,
  themeFromSourceColor,
} from "@material/material-color-utilities";

/**
 * Every accent the app ships, each derived from ONE seed.
 *
 * The first entry is the default and lands on bare `:root`; the rest are
 * scoped to `[data-accent="<id>"]` on <html>, so switching is a single
 * attribute write with no re-render and no runtime colour maths.
 *
 * Adding one is a line here plus a regeneration -- the 70-odd role tokens,
 * light and dark, are all computed. That is the whole reason this file exists
 * rather than a hand-written palette.
 *
 *   node --import ./scripts/register-loader.mjs scripts/gen-theme.mjs
 */
const ACCENTS = [
  // Material 3's own baseline seed, and the app's default. Note this is the
  // one seed where the generated primary equals the seed exactly -- M3's
  // reference palette is built around it.
  { id: "purple", label: "Purple", seed: "#6750a4" },
  { id: "teal", label: "Teal", seed: "#0f766e" },
  { id: "violet", label: "Violet", seed: "#7c4dff" },
  { id: "amber", label: "Amber", seed: "#b45309" },
  { id: "olive", label: "Olive", seed: "#3f6212" },
  { id: "blue", label: "Blue", seed: "#0284c7" }, // what the app shipped with
  // Material's own Brown 500. Parley wears this: the three apps share one
  // shell, and a warm neutral reads as a different PLACE at a glance without
  // competing with Research Desk's purple the way another saturated hue would.
  { id: "brown", label: "Brown", seed: "#795548" },
];

// The navigation drawer is a LIGHT surface, which is what stock M3 specifies.
//
// It used to be a fixed navy slab, kept as brand identity. That was reverted:
// a dark chrome column beside a flat white workspace was the single loudest
// thing on screen, and it was loud about navigation -- the part of the app the
// user looks at least. The aliases are kept rather than deleted so the nav can
// diverge again without touching every call site.
//
// Now derived, so they follow the light/dark scheme instead of being frozen.
const NAV_LIGHT = {
  "nav-surface": "var(--md-surface)",
  "nav-surface-container": "var(--md-surface-container-high)",
  "nav-on-surface": "var(--md-on-surface)",
  "nav-on-surface-variant": "var(--md-on-surface-variant)",
  "nav-outline": "var(--md-outline-variant)",
};


/** The role tokens we actually consume. M3 defines more; unused ones are noise. */
const ROLES = [
  "primary",
  "onPrimary",
  "primaryContainer",
  "onPrimaryContainer",
  "secondary",
  "onSecondary",
  "secondaryContainer",
  "onSecondaryContainer",
  "tertiary",
  "onTertiary",
  "tertiaryContainer",
  "onTertiaryContainer",
  "error",
  "onError",
  "errorContainer",
  "onErrorContainer",
  "background",
  "onBackground",
  "surface",
  "onSurface",
  "surfaceVariant",
  "onSurfaceVariant",
  "outline",
  "outlineVariant",
  "shadow",
  "scrim",
  "inverseSurface",
  "inverseOnSurface",
];

const kebab = (s) => s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

/**
 * Surface containers are the five elevation-ish surface tones M3 uses instead
 * of stacking shadows. material-color-utilities exposes the neutral palette,
 * so we read the tones the spec assigns to each container level.
 */
function surfaceContainers(palettes, dark) {
  const n = palettes.neutral;
  const tone = (t) => hexFromArgb(n.tone(t));
  return dark
    ? {
        "surface-dim": tone(6),
        "surface-bright": tone(24),
        "surface-container-lowest": tone(4),
        "surface-container-low": tone(10),
        "surface-container": tone(12),
        "surface-container-high": tone(17),
        "surface-container-highest": tone(22),
      }
    : {
        "surface-dim": tone(87),
        "surface-bright": tone(98),
        "surface-container-lowest": tone(100),
        "surface-container-low": tone(96),
        "surface-container": tone(94),
        "surface-container-high": tone(92),
        "surface-container-highest": tone(90),
      };
}

function block(scheme, palettes, dark) {
  const lines = [];
  for (const role of ROLES) {
    const argb = scheme[role];
    if (argb === undefined) continue;
    lines.push(`  --md-${kebab(role)}: ${hexFromArgb(argb)};`);
  }
  for (const [name, value] of Object.entries(surfaceContainers(palettes, dark))) {
    lines.push(`  --md-${name}: ${value};`);
  }
  return lines.join("\n");
}

/** One accent: a light block, a dark block, both under the same selector. */
function accentBlocks({ id, seed }, isDefault) {
  const theme = themeFromSourceColor(argbFromHex(seed));
  const light = block(theme.schemes.light.toJSON(), theme.palettes, false);
  const dark = block(theme.schemes.dark.toJSON(), theme.palettes, true);
  // The default lands on bare `:root` so the app has colours before any
  // attribute is set -- no flash of unstyled palette, and it still works with
  // JavaScript disabled.
  const sel = isDefault ? ":root" : `[data-accent="${id}"]`;

  return `/* ---- ${id} (${seed}) ---- */
${sel} {
${light}
}

@media (prefers-color-scheme: dark) {
  ${sel} {
${dark}
  }
}
`;
}

// Aliases, so they resolve per-scheme automatically -- a custom property
// holding `var(...)` is substituted at use time, not at definition time, which
// means one declaration covers every accent AND both schemes.
const nav = Object.entries(NAV_LIGHT)
  .map(([k, v]) => `  --md-${k}: ${v};`)
  .join("\n");

const palettes = ACCENTS.map((a, i) => accentBlocks(a, i === 0)).join("\n");

// ASCII only in generated output: this file gets read by tools with varying
// encoding defaults, and a stray em-dash is not worth the ambiguity.
const css = `/* GENERATED by scripts/gen-theme.mjs - do not edit by hand.
 * Accents: ${ACCENTS.map((a) => `${a.id} ${a.seed}`).join(", ")}
 * Default: ${ACCENTS[0].id}
 * Regenerate: node --import ./scripts/register-loader.mjs scripts/gen-theme.mjs
 *
 * Switching accent = setting data-accent on <html>. Every rule below is just
 * custom properties, so the swap costs one attribute write and a repaint.
 */

:root {
  /* Navigation chrome. Aliases onto the ordinary surface roles, so the drawer
     follows the scheme instead of being a fixed slab. Declared once, outside
     the accent blocks, because a var() alias resolves at USE time. */
${nav}
}

${palettes}`;

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "app", "theme.css");
writeFileSync(out, css, "utf8");
console.log(`wrote ${out}`);
for (const a of ACCENTS) {
  const t = themeFromSourceColor(argbFromHex(a.seed));
  console.log(
    `  ${a.id.padEnd(8)} ${a.seed} -> primary ${hexFromArgb(t.schemes.light.primary)}`,
  );
}

// The picker needs the same list. Emitting it rather than duplicating it by
// hand is what stops the two drifting -- add an accent above and the UI knows.
const tsOut = join(here, "..", "lib", "accents.ts");
writeFileSync(
  tsOut,
  `// GENERATED by scripts/gen-theme.mjs - do not edit by hand.
// Mirrors the ACCENTS list there, so the picker cannot drift from the CSS.

export type Accent = (typeof ACCENTS)[number]["id"];

export const ACCENTS = [
${ACCENTS.map(
  (a) => `  { id: "${a.id}", label: "${a.label}", seed: "${a.seed}" },`,
).join("\n")}
] as const;

/** The one on bare \`:root\`, i.e. what renders with no data-accent set. */
export const DEFAULT_ACCENT: Accent = "${ACCENTS[0].id}";
`,
  "utf8",
);
console.log(`wrote ${tsOut}`);

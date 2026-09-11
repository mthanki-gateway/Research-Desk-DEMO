/**
 * Turn settings, remembered between visits.
 *
 * WHY localStorage AND NOT THE BACKEND
 *
 * These are UI knobs, not data: how many passages to retrieve, whether to
 * expand the query, whether the agent may ask a question. Nothing else reads
 * them, they carry no tenancy, and losing them costs four clicks. Server-side
 * preferences would mean a new table and a hand-written migration (the project
 * has no Alembic yet) to store something the browser is already good at.
 *
 * The trade is real and worth stating: prefs are PER BROWSER. A second device
 * starts from the defaults. If they ever need to follow the user across
 * devices, this module is the seam -- swap the two functions for API calls and
 * nothing else changes.
 *
 * WHAT THIS FILE IS CAREFUL ABOUT
 *
 * Stored data is untrusted input. It was written by an older version of this
 * app, possibly a much older one, and can be hand-edited in devtools. So every
 * field is validated on the way in and anything unusable falls back to its
 * default, one field at a time -- a single bad value must not discard the rest.
 */

/**
 * Which model family answers. DEVELOPMENT ONLY — the server ignores it unless
 * APP_ENV=dev, and the control is not rendered in a production build.
 *
 * "gemma" trades quality for a 30 rpm / 14,400-per-day budget. The answer
 * model allows FIVE requests per minute, which is about one question every two
 * minutes once the agent loop is doing real work — unusable for development.
 */
export type ModelProfile = "gemini" | "gemma";

export type TurnSettings = {
  topK: number;
  multiQuery: boolean;
  /** Let the agent ask what a vague question means before searching. */
  clarify: boolean;
  /** Gather evidence with the tool-calling loop instead of a fixed plan. */
  react: boolean;
  /** Dev-only model override. null = use whatever the server is configured for. */
  modelProfile: ModelProfile | null;
};

/**
 * The starting point for someone who has never changed anything.
 *
 * `react: true` and `clarify: true` mirror REACT_DEFAULT and AGENT_CLARIFY on
 * the server. They are duplicated rather than fetched because the UI has to
 * render before any request completes, and a toggle that flips under the user
 * a second after load is worse than one that starts in the documented state.
 */
export const DEFAULT_TURN_SETTINGS: TurnSettings = {
  topK: 5,
  multiQuery: false,
  clarify: true,
  react: true,
  // null, not "gemini": the UI should not assert which model the server runs.
  // Sending an explicit profile on every turn would override a deployment's own
  // configuration from the browser, which is exactly backwards.
  modelProfile: null,
};

const KEY = "research-desk:turn-settings";

// Matches the rail's stepper bounds and the backend's Field(ge=1, le=20). A
// stored 500 would be rejected by the API on every turn, so it is clamped here
// rather than sent.
const MIN_TOP_K = 1;
const MAX_TOP_K = 20;

function readBool(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readProfile(value: unknown): ModelProfile | null {
  // Validated against the known set rather than cast. A stored profile from an
  // older build (or a hand-edited one) would otherwise be sent to the server
  // and rejected on every turn.
  return value === "gemini" || value === "gemma" ? value : null;
}

function readTopK(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  // Rounded as well as clamped: the stepper only produces integers, but a
  // hand-edited 7.5 would reach the API and be rejected.
  return Math.min(MAX_TOP_K, Math.max(MIN_TOP_K, Math.round(value)));
}

/**
 * Load saved settings, falling back to the defaults field by field.
 *
 * MUST NOT be called during render. On the server `localStorage` does not
 * exist, and reading it in a `useState` initialiser would either throw during
 * SSR or make the server and client render different toggle states, which
 * React reports as a hydration mismatch. Call it from an effect.
 */
export function loadTurnSettings(): TurnSettings {
  // Every access is guarded. localStorage throws outright -- not returns
  // null -- when a browser is set to block site data, and in that case the app
  // must still work with defaults rather than fail to render the chat.
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_TURN_SETTINGS;

    const stored: unknown = JSON.parse(raw);
    if (!stored || typeof stored !== "object") return DEFAULT_TURN_SETTINGS;
    const s = stored as Record<string, unknown>;

    // Merged over the defaults, which is what makes adding a setting later
    // safe: a payload written before `react` existed yields the default for
    // `react` instead of `undefined` leaking into a request body.
    return {
      topK: readTopK(s.topK, DEFAULT_TURN_SETTINGS.topK),
      multiQuery: readBool(s.multiQuery, DEFAULT_TURN_SETTINGS.multiQuery),
      clarify: readBool(s.clarify, DEFAULT_TURN_SETTINGS.clarify),
      react: readBool(s.react, DEFAULT_TURN_SETTINGS.react),
      modelProfile: readProfile(s.modelProfile),
    };
  } catch {
    return DEFAULT_TURN_SETTINGS;
  }
}

/** Persist settings. Silent on failure -- a full quota must not break a turn. */
export function saveTurnSettings(settings: TurnSettings): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // Private windows, blocked site data, quota exceeded. The settings still
    // work for this session; only the memory is lost, and warning about it
    // would be noise the user cannot act on.
  }
}

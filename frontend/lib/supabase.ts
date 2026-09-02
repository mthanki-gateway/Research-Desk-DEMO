import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * Browser-side Supabase client, used only for authentication.
 *
 * We never query Supabase's database — application data lives in our own
 * Postgres and Qdrant. This client exists to sign in, hold the session, and
 * hand us an access token for the `Authorization` header.
 *
 * `createBrowserClient` from @supabase/ssr (not the plain `createClient`)
 * stores the session in cookies rather than localStorage, which is what lets
 * Next middleware and server components see it too.
 */
export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

/**
 * True when both values are configured. With auth off the app runs exactly as
 * it did before — no login screen, one anonymous user. Every auth-aware branch
 * in the UI checks this rather than assuming a client exists.
 */
export const authEnabled = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

let client: SupabaseClient | null = null;

/** Null when auth is not configured, so callers must handle that case. */
export function getSupabase(): SupabaseClient | null {
  if (!authEnabled) return null;
  // One instance per tab. A second client would keep its own session state and
  // the two would race on token refresh.
  client ??= createBrowserClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return client;
}

/**
 * The current access token, or null.
 *
 * `getSession()` reads from storage and refreshes if the token is close to
 * expiry, so this is safe to call before every request rather than caching a
 * token that might already be stale.
 */
export async function getAccessToken(): Promise<string | null> {
  const supabase = getSupabase();
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

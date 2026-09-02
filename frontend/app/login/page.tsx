"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { authEnabled, getSupabase } from "@/lib/supabase";
import { Button, TextField } from "../md";
import { IconChat, IconSpinner } from "../icons";

/**
 * Sign-in. Two methods, deliberately:
 *
 *  - **Magic link** works with zero external setup, so auth is testable
 *    without the Google Cloud OAuth client existing.
 *  - **Google** is the SSO path, and is only offered when configured.
 *
 * Supabase handles the OAuth redirect itself; the browser goes
 * app -> Google -> Supabase -> back here, which is why the only redirect this
 * app declares is its own origin.
 */
export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState<null | "google" | "email">(null);
  const [error, setError] = useState<string | null>(null);

  // NOTE: useSearchParams() is deliberately NOT called here. It is called in
  // <OAuthErrorReporter/> below, inside a Suspense boundary. Reading it in this
  // component makes the whole page uncacheable and, more concretely, breaks
  // `next build`: prerendering bails out with "useSearchParams() should be
  // wrapped in a suspense boundary". `next dev` renders it happily, so this
  // fails only in a production build -- which is exactly why CI builds.

  // Already signed in? Don't show a login form.
  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) return;
    void supabase.auth.getSession().then(({ data }) => {
      if (data.session) router.replace("/chat");
    });
  }, [router]);

  if (!authEnabled) {
    return (
      <Centered>
        <h1 className="md-title-large">Auth is not configured</h1>
        <p
          className="md-body-medium mt-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Set <code>SUPABASE_URL</code> and <code>SUPABASE_ANON_KEY</code> in{" "}
          <code>.env</code> to enable sign-in. Until then the app runs
          single-user with no login.
        </p>
        <Button className="mt-6" onClick={() => router.push("/chat")}>
          Continue
        </Button>
      </Centered>
    );
  }

  async function withGoogle() {
    const supabase = getSupabase();
    if (!supabase) return;
    setBusy("google");
    setError(null);
    const { error: err } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
    if (err) {
      setError(err.message);
      setBusy(null);
    }
    // On success the browser navigates away, so nothing to do here.
  }

  async function withMagicLink(e: React.FormEvent) {
    e.preventDefault();
    const supabase = getSupabase();
    if (!supabase || !email.trim()) return;
    setBusy("email");
    setError(null);
    const { error: err } = await supabase.auth.signInWithOtp({
      email: email.trim(),
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    });
    if (err) setError(err.message);
    else setSent(true);
    setBusy(null);
  }

  return (
    <Centered>
      {/* Renders nothing; exists only to isolate the useSearchParams() read so
          the rest of the page can still be prerendered at build time. */}
      <Suspense fallback={null}>
        <OAuthErrorReporter onError={setError} />
      </Suspense>

      <span
        className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-[var(--md-shape-full)]"
        style={{
          background: "var(--md-primary-container)",
          color: "var(--md-on-primary-container)",
        }}
      >
        <IconChat className="h-8 w-8" />
      </span>

      <h1 className="md-headline-small">Research Desk</h1>
      <p
        className="md-body-medium mt-2"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        Sign in to keep your documents and conversations private to you.
      </p>

      {error && (
        <p
          className="md-body-medium mt-5 rounded-[var(--md-shape-md)] px-4 py-3 text-left"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      {sent ? (
        <div
          className="md-body-medium mt-6 rounded-[var(--md-shape-md)] p-4 text-left"
          style={{
            background: "var(--md-secondary-container)",
            color: "var(--md-on-secondary-container)",
          }}
        >
          <p className="font-medium">Check your email</p>
          <p className="mt-1">
            A sign-in link is on its way to {email}. It opens this app directly,
            so no password is needed.
          </p>
        </div>
      ) : (
        <>
          <Button
            className="mt-7 w-full"
            onClick={() => void withGoogle()}
            disabled={busy !== null}
          >
            {busy === "google" ? <IconSpinner /> : <GoogleMark />}
            Continue with Google
          </Button>

          <div className="my-5 flex items-center gap-3">
            <hr className="md-divider flex-1" />
            <span
              className="md-label-medium"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              OR
            </span>
            <hr className="md-divider flex-1" />
          </div>

          <form onSubmit={withMagicLink} className="space-y-3 text-left">
            <TextField
              label="Email address"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={busy !== null}
              surface="var(--md-surface-container-low)"
            />
            <Button
              type="submit"
              variant="tonal"
              className="w-full"
              disabled={busy !== null || !email.trim()}
            >
              {busy === "email" ? <IconSpinner /> : null}
              Email me a sign-in link
            </Button>
          </form>
        </>
      )}
    </Centered>
  );
}

/**
 * Supabase reports OAuth failures as query params on the redirect target
 * (`?error=...&error_description=...`), so the login page has to read them to
 * explain why a sign-in bounced.
 *
 * It lives in its own component because useSearchParams() forces client-side
 * rendering of whatever component calls it. Confined here and wrapped in
 * Suspense, the cost is one null-rendering leaf instead of the entire page.
 *
 * `onError` is a useState setter, which React guarantees is stable, so it is a
 * safe effect dependency -- an unstable callback here would loop.
 */
function OAuthErrorReporter({ onError }: { onError: (message: string) => void }) {
  const params = useSearchParams();
  useEffect(() => {
    const description = params.get("error_description") ?? params.get("error");
    if (description) onError(description);
  }, [params, onError]);
  return null;
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid min-h-screen place-items-center px-6 py-12">
      <div className="md-card md-card-elevated w-full max-w-sm p-8 text-center">
        {children}
      </div>
    </div>
  );
}

/** Google's mark, inline. Their brand guidelines require the multi-colour G. */
function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-[1.15rem] w-[1.15rem]" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.4a5.5 5.5 0 0 1-2.4 3.6v3h3.9c2.3-2.1 3.6-5.2 3.6-8.8Z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.2 0 5.9-1.1 7.9-2.9l-3.9-3a7.2 7.2 0 0 1-10.7-3.8H1.3v3.1A12 12 0 0 0 12 24Z"
      />
      <path
        fill="#FBBC05"
        d="M5.3 14.3a7.2 7.2 0 0 1 0-4.6V6.6H1.3a12 12 0 0 0 0 10.8l4-3.1Z"
      />
      <path
        fill="#EA4335"
        d="M12 4.8c1.8 0 3.400.6 4.7 1.8l3.4-3.4A12 12 0 0 0 1.3 6.6l4 3.1A7.2 7.2 0 0 1 12 4.8Z"
      />
    </svg>
  );
}

"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { claimUnowned, whoAmI } from "@/lib/api";
import { getSupabase } from "@/lib/supabase";
import { IconSpinner } from "../../icons";

/**
 * Where Supabase lands after Google OAuth or a magic link.
 *
 * `createBrowserClient` from @supabase/ssr detects the session in the URL and
 * stores it, so there is nothing to exchange by hand — we just wait for the
 * session to appear, then do the one-time housekeeping.
 *
 * That housekeeping is claiming pre-auth rows. Documents and sessions created
 * before auth was enabled have owner_id NULL; without this they would still
 * exist but be invisible to every filtered query. Claiming is idempotent, so
 * running it on every sign-in is harmless.
 */
export default function AuthCallback() {
  const router = useRouter();
  const [message, setMessage] = useState("Completing sign-in");

  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) {
      router.replace("/chat");
      return;
    }

    let cancelled = false;

    // The session may not be written at first paint, so wait for the auth
    // event rather than polling getSession().
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session && !cancelled) void finish();
    });

    async function finish() {
      try {
        const me = await whoAmI();
        if (me.unclaimed_documents > 0 || me.unclaimed_sessions > 0) {
          setMessage("Linking your existing documents");
          const claimed = await claimUnowned();
          // Not fatal if it claimed nothing — the counts are advisory.
          console.info("claimed", claimed);
        }
      } catch (e) {
        // Never block sign-in on housekeeping. Worst case the user sees an
        // empty library and can re-run this by signing in again.
        console.warn("post-signin setup failed", e);
      }
      if (!cancelled) router.replace("/chat");
    }

    // Handle the case where the session already existed before we subscribed.
    void supabase.auth.getSession().then(({ data }) => {
      if (data.session && !cancelled) void finish();
    });

    return () => {
      cancelled = true;
      sub.subscription.unsubscribe();
    };
  }, [router]);

  return (
    <div className="grid min-h-screen place-items-center px-6">
      <div className="text-center">
        <IconSpinner className="mx-auto h-6 w-6" />
        <p
          className="md-body-medium mt-4"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {message}…
        </p>
      </div>
    </div>
  );
}

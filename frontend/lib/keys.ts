"use client";

import { useEffect, useState } from "react";
import { authedFetch, detail } from "./api";

/** One provider card on the Settings page, as the API describes it. */
export type ProviderKey = {
  id: string;
  label: string;
  powers: string;
  url: string;
  steps: string[];
  features: string[];
  /** The person's own key, if they added one. Only the last four characters. */
  yours: { last4: string; updated_at: string } | null;
  /** Whether the server's key is standing in for theirs. */
  server_fallback: boolean;
  source: "yours" | "server" | "none";
};

export type KeyStatus = {
  providers: ProviderKey[];
  /** Feature id -> usable right now (their key, or the server's fallback). */
  features: Record<string, boolean>;
  require_user_keys: boolean;
};

export async function getKeys(): Promise<KeyStatus> {
  const res = await authedFetch("/settings/keys");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function saveKey(provider: string, key: string): Promise<KeyStatus> {
  const res = await authedFetch(`/settings/keys/${provider}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  if (!res.ok) throw new Error(await detail(res));
  return publish(await res.json());
}

export async function deleteKey(provider: string): Promise<KeyStatus> {
  const res = await authedFetch(`/settings/keys/${provider}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await detail(res));
  return publish(await res.json());
}

// One shared copy, so the drawer greys items out the moment a key is saved
// on the Settings page instead of on the next reload.
let cached: KeyStatus | null = null;
const listeners = new Set<(s: KeyStatus) => void>();
function publish(s: KeyStatus): KeyStatus {
  cached = s;
  listeners.forEach((l) => l(s));
  return s;
}

/**
 * Which features are usable. `null` until loaded -- callers treat that as
 * "available", so nothing flashes grey on every page load.
 */
export function useFeatures(): Record<string, boolean> | null {
  const [s, setS] = useState<KeyStatus | null>(cached);
  useEffect(() => {
    listeners.add(setS);
    if (!cached) getKeys().then(publish).catch(() => {});
    return () => {
      listeners.delete(setS);
    };
  }, []);
  return s?.features ?? null;
}

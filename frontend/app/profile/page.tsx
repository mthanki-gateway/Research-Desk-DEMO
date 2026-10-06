"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  addMemory,
  deleteAccount,
  forgetMemory,
  getMemory,
  type Memory,
  type ProfileMemory,
} from "@/lib/api";
import { useApp } from "../providers";
import { Button, ConfirmButton } from "../md";
import { IconChat, IconPlus, IconSpinner, IconTrash } from "../icons";

/**
 * What the app remembers about how you like to be answered, and your account.
 *
 * Preferences come from two places, and each row says which: the assistant
 * captures them from things said in a chat, or you add them here. Forgotten
 * ones are not listed -- they no longer apply, and a page of struck-through
 * history buried the ones that do.
 *
 * Lists scroll inside their own panels, so the page stays one screen tall
 * however much is remembered.
 */

type Filter = "all" | "user" | "assistant";

export default function ProfilePage() {
  const [data, setData] = useState<ProfileMemory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [draft, setDraft] = useState("");
  const [adding, setAdding] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await getMemory());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load your profile");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function forget(id: string) {
    // Removed from the list at once; the reload keeps the server the source
    // of truth if the call failed.
    setData((prev) =>
      prev
        ? {
            user_preferences: prev.user_preferences.filter((p) => p.id !== id),
            conversations: prev.conversations.map((c) => ({
              ...c,
              preferences: c.preferences.filter((p) => p.id !== id),
            })),
          }
        : prev,
    );
    try {
      await forgetMemory(id);
    } finally {
      await load();
    }
  }

  async function add() {
    const text = draft.trim();
    if (text.length < 2 || adding) return;
    setAdding(true);
    try {
      await addMemory(text);
      setDraft("");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not add that");
    } finally {
      setAdding(false);
    }
  }

  const prefs = useMemo(
    () =>
      (data?.user_preferences ?? []).filter((p) => filter === "all" || p.origin === filter),
    [data, filter],
  );
  const counts = useMemo(() => {
    const all = data?.user_preferences ?? [];
    return {
      all: all.length,
      user: all.filter((p) => p.origin === "user").length,
      assistant: all.filter((p) => p.origin !== "user").length,
    };
  }, [data]);

  if (error && !data) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-9">
        <p className="md-body-medium" style={{ color: "var(--md-error)" }}>
          {error}
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mx-auto max-w-3xl space-y-3 px-6 py-9" aria-hidden>
        <div className="md-skeleton h-8 w-40" />
        <div className="md-skeleton h-4 w-64" />
        <div className="md-skeleton mt-6 h-[300px]" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Profile</h1>
        <p className="md-body-medium mt-1" style={{ color: "var(--md-on-surface-variant)" }}>
          How you like to be answered, and your account.
        </p>
      </header>

      {/* ---- Preferences ---------------------------------------------- */}
      <section className="md-card md-card-outlined overflow-hidden">
        <div className="space-y-3 p-4 pb-3">
          <div>
            <h2 className="md-title-medium">Preferences</h2>
            <p className="md-body-small mt-0.5" style={{ color: "var(--md-on-surface-variant)" }}>
              Applied in every conversation, including new ones.
            </p>
          </div>

          <form
            className="flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void add();
            }}
          >
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={500}
              placeholder="Add one, e.g. “Use metric units” or “Keep answers short”"
              aria-label="New preference"
              className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-full)] px-4 py-2 outline-none"
              style={{
                background: "var(--md-surface-container-high)",
                color: "var(--md-on-surface)",
              }}
            />
            <Button type="submit" disabled={adding || draft.trim().length < 2}>
              {adding ? <IconSpinner /> : <IconPlus />}
              Add
            </Button>
          </form>

          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Filter preferences">
            {(
              [
                ["all", "All"],
                ["user", "Added by you"],
                ["assistant", "Learned by the assistant"],
              ] as [Filter, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={filter === id}
                onClick={() => setFilter(id)}
                className={`md-chip md-state md-chip-sm ${filter === id ? "md-chip-selected" : ""}`}
              >
                {label}
                <span style={{ opacity: 0.6 }}>{counts[id]}</span>
              </button>
            ))}
          </div>
        </div>

        <div
          className="max-h-[22rem] overflow-y-auto border-t"
          style={{ borderColor: "var(--md-outline-variant)" }}
        >
          {prefs.length === 0 ? (
            <p className="md-body-medium p-4" style={{ color: "var(--md-on-surface-variant)" }}>
              {counts.all === 0
                ? "Nothing yet. Add one above, or tell the assistant in a chat — “always search the web too” — and it will appear here."
                : "None in this view."}
            </p>
          ) : (
            <ul>
              {prefs.map((p) => (
                <MemoryRow key={p.id} memory={p} onForget={forget} />
              ))}
            </ul>
          )}
        </div>
      </section>

      {/* ---- Per conversation ----------------------------------------- */}
      {data.conversations.length > 0 && (
        <section className="md-card md-card-outlined overflow-hidden">
          <div className="p-4 pb-3">
            <h2 className="md-title-medium">Per conversation</h2>
            <p className="md-body-small mt-0.5" style={{ color: "var(--md-on-surface-variant)" }}>
              What each chat is about, plus any instruction that applies to it alone.
            </p>
          </div>
          <div
            className="max-h-[22rem] overflow-y-auto border-t"
            style={{ borderColor: "var(--md-outline-variant)" }}
          >
            {data.conversations.map((c) => (
              <article
                key={c.session_id}
                className="space-y-2 border-b p-4 last:border-b-0"
                style={{ borderColor: "var(--md-outline-variant)" }}
              >
                <Link href={`/chat/${c.session_id}`} className="flex items-center gap-2 hover:underline">
                  <IconChat className="h-4 w-4 shrink-0" />
                  <span className="md-title-small truncate">{c.title}</span>
                </Link>
                {c.summary && (
                  <p className="md-body-medium" style={{ color: "var(--md-on-surface-variant)" }}>
                    {c.summary}
                  </p>
                )}
                {c.preferences.length > 0 && (
                  <ul>
                    {c.preferences.map((p) => (
                      <MemoryRow key={p.id} memory={p} onForget={forget} />
                    ))}
                  </ul>
                )}
              </article>
            ))}
          </div>
        </section>
      )}

      <DangerZone />
    </div>
  );
}

function MemoryRow({ memory, onForget }: { memory: Memory; onForget: (id: string) => void }) {
  const mine = memory.origin === "user";
  return (
    <li
      className="flex items-start gap-3 border-b px-4 py-3 last:border-b-0"
      style={{ borderColor: "var(--md-outline-variant)" }}
    >
      <div className="min-w-0 flex-1">
        <p className="md-body-medium">{memory.text}</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <span
            className="md-label-small rounded-[var(--md-shape-full)] px-2 py-0.5"
            style={{
              background: mine ? "var(--md-primary-container)" : "var(--md-tertiary-container)",
              color: mine ? "var(--md-on-primary-container)" : "var(--md-on-tertiary-container)",
            }}
          >
            {mine ? "Added by you" : "Learned by the assistant"}
          </span>
          <span className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
            {new Date(memory.created_at).toLocaleDateString(undefined, {
              day: "numeric",
              month: "short",
              year: "numeric",
            })}
          </span>
        </div>
        {!mine && memory.source_message && (
          <p
            className="md-body-small mt-1.5 truncate border-l-2 pl-2.5 italic"
            style={{ color: "var(--md-on-surface-variant)", borderColor: "var(--md-outline-variant)" }}
            title={memory.source_message}
          >
            From: {memory.source_message}
          </p>
        )}
      </div>
      <ConfirmButton
        label="Forget"
        icon={<IconTrash />}
        title="Forget this preference?"
        body="It stops applying to future answers. Conversations already answered are unchanged."
        confirmLabel="Forget"
        onConfirm={() => onForget(memory.id)}
      />
    </li>
  );
}

/**
 * Account deletion. Typed confirmation, because it cannot be undone and
 * removes everything: documents and their files, chats, memories, the
 * knowledge graph, Howler projects and recordings, and saved API keys.
 *
 * The request carries no id -- the server deletes the account the login
 * token belongs to, and nothing else.
 */
function DangerZone() {
  const { signOut } = useApp();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      await deleteAccount();
      await signOut();
      router.replace("/login");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not delete the account");
      setBusy(false);
    }
  }

  return (
    <section
      className="md-card md-card-outlined space-y-3 p-4"
      style={{ borderColor: "var(--md-error)" }}
    >
      <div>
        <h2 className="md-title-medium" style={{ color: "var(--md-error)" }}>
          Delete account
        </h2>
        <p className="md-body-small mt-0.5" style={{ color: "var(--md-on-surface-variant)" }}>
          Permanently removes everything you own across every app — documents and their
          files, chats, memories, the knowledge graph, Howler projects and recordings, and
          your saved API keys. This cannot be undone.
        </p>
      </div>
      {!open ? (
        <Button variant="outlined" onClick={() => setOpen(true)} style={{ color: "var(--md-error)" }}>
          <IconTrash />
          Delete my account…
        </Button>
      ) : (
        <div className="space-y-2">
          <label className="md-body-small block" style={{ color: "var(--md-on-surface-variant)" }}>
            Type <strong>DELETE</strong> to confirm.
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoFocus
              aria-label="Type DELETE to confirm"
              className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2 outline-none"
              style={{ background: "var(--md-surface)", border: "1px solid var(--md-error)" }}
            />
            <Button
              onClick={() => void run()}
              disabled={typed !== "DELETE" || busy}
              style={typed === "DELETE" ? { background: "var(--md-error)", color: "var(--md-on-error)" } : undefined}
            >
              {busy && <IconSpinner />}
              Delete everything
            </Button>
            <Button
              variant="text"
              disabled={busy}
              onClick={() => {
                setOpen(false);
                setTyped("");
              }}
            >
              Cancel
            </Button>
          </div>
          {err && (
            <p className="md-body-small" style={{ color: "var(--md-error)" }}>
              {err}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

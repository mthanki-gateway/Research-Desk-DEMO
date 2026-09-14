"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { forgetMemory, getMemory, type Memory, type ProfileMemory } from "@/lib/api";
import { ConfirmButton } from "../md";
import { IconChat, IconProfile, IconTrash } from "../icons";

/**
 * What the app remembers, and the one control that matters: forget.
 *
 * Deliberately NOT a settings page. Preferences here were not typed into a
 * form -- they were captured from things the user said in passing, which
 * means the first question about any of them is "where did that come from?".
 * So every entry shows the message it was captured from, and the only action
 * is removal. Editing a remembered instruction in place would produce a
 * preference the user never actually stated, with a provenance line claiming
 * they did.
 */
export default function ProfilePage() {
  const [data, setData] = useState<ProfileMemory | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await getMemory());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load memory");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function forget(id: string) {
    // Optimistic: flip to inactive locally so the row visibly changes on the
    // click rather than after a round trip. Reload afterwards so the server
    // remains the source of truth if the call failed.
    setData((prev) =>
      prev
        ? {
            user_preferences: prev.user_preferences.map((p) =>
              p.id === id ? { ...p, active: false } : p,
            ),
            conversations: prev.conversations.map((c) => ({
              ...c,
              preferences: c.preferences.map((p) =>
                p.id === id ? { ...p, active: false } : p,
              ),
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

  if (error) {
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
        <div className="md-skeleton mt-6 h-[96px]" />
        <div className="md-skeleton h-[96px]" />
      </div>
    );
  }

  const nothing =
    !data.user_preferences.length && !data.conversations.length;

  return (
    <div className="mx-auto max-w-3xl space-y-8 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Profile</h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Everything Research Desk remembers about how you like to be answered,
          and what each conversation is about. Forgetting an instruction stops
          it applying to future turns.
        </p>
      </header>

      {nothing ? (
        <div className="md-card md-card-filled px-6 py-12 text-center">
          <span
            className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
            style={{
              background: "var(--md-primary-container)",
              color: "var(--md-on-primary-container)",
            }}
          >
            <IconProfile className="h-7 w-7" />
          </span>
          <h2 className="md-title-medium">Nothing remembered yet</h2>
          <p
            className="md-body-medium mx-auto mt-2 max-w-md"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Tell the assistant how you want answers — &ldquo;always search the
            web too&rdquo;, &ldquo;keep it brief&rdquo; — and it will be kept
            here and applied from then on.
          </p>
        </div>
      ) : (
        <>
          <section className="space-y-3">
            <div>
              <h2 className="md-title-medium">Applies everywhere</h2>
              <p
                className="md-body-small mt-0.5"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                In force in every conversation, including new ones.
              </p>
            </div>
            {data.user_preferences.length ? (
              data.user_preferences.map((p) => (
                <MemoryRow key={p.id} memory={p} onForget={forget} />
              ))
            ) : (
              <p
                className="md-body-medium"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                None yet.
              </p>
            )}
          </section>

          {data.conversations.length > 0 && (
            <section className="space-y-3">
              <div>
                <h2 className="md-title-medium">Per conversation</h2>
                <p
                  className="md-body-small mt-0.5"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  What each chat is about, plus any instruction scoped to it
                  alone.
                </p>
              </div>
              {data.conversations.map((c) => (
                <article
                  key={c.session_id}
                  className="md-card md-card-outlined space-y-3 p-4"
                >
                  <Link
                    href={`/chat/${c.session_id}`}
                    className="flex items-center gap-2 hover:underline"
                  >
                    <IconChat className="h-4 w-4 shrink-0" />
                    <span className="md-title-small truncate">{c.title}</span>
                  </Link>
                  {c.summary && (
                    <p
                      className="md-body-medium"
                      style={{ color: "var(--md-on-surface-variant)" }}
                    >
                      {c.summary}
                    </p>
                  )}
                  {c.preferences.map((p) => (
                    <MemoryRow key={p.id} memory={p} onForget={forget} />
                  ))}
                </article>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}

function MemoryRow({
  memory,
  onForget,
}: {
  memory: Memory;
  onForget: (id: string) => void;
}) {
  return (
    <div
      className="md-card md-card-outlined flex items-start gap-4 p-4"
      // Forgotten entries stay listed but recede. They are kept because a
      // cancelled instruction explains a change in behaviour just as much as
      // a live one does -- removing the row entirely would leave the user
      // wondering whether it was ever captured.
      style={memory.active ? undefined : { opacity: 0.55 }}
    >
      <div className="min-w-0 flex-1">
        <p
          className="md-body-large"
          style={
            memory.active
              ? undefined
              : { textDecoration: "line-through" }
          }
        >
          {memory.text}
        </p>
        {memory.source_message && (
          <p
            className="md-body-small mt-1.5 border-l-2 pl-3 italic"
            style={{
              color: "var(--md-on-surface-variant)",
              borderColor: "var(--md-outline-variant)",
            }}
          >
            {memory.source_message}
          </p>
        )}
        <p
          className="md-label-small mt-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {memory.active ? "Remembered" : "Forgotten"} ·{" "}
          {new Date(memory.created_at).toLocaleDateString(undefined, {
            day: "numeric",
            month: "short",
            year: "numeric",
          })}
        </p>
      </div>
      {memory.active && (
        <ConfirmButton
          label="Forget"
          icon={<IconTrash />}
          title="Forget this instruction?"
          body="It stops applying to future answers. Conversations already answered are unchanged."
          confirmLabel="Forget"
          onConfirm={() => onForget(memory.id)}
        />
      )}
    </div>
  );
}

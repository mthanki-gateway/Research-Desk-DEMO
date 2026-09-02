"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { createSession } from "@/lib/api";
import { useApp } from "../providers";
import { Button, Ripplable } from "../md";
import { IconChat, IconLibrary, IconPlus, IconSpinner } from "../icons";

export default function ChatIndex() {
  const { sessions, readyDocuments, loading, refreshSessions } = useApp();
  const router = useRouter();
  const [creating, setCreating] = useState(false);

  async function start() {
    setCreating(true);
    try {
      const s = await createSession();
      await refreshSessions();
      router.push(`/chat/${s.id}`);
    } finally {
      setCreating(false);
    }
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-3xl space-y-3 px-6 py-9" aria-hidden>
        <div className="md-skeleton h-8 w-40" />
        <div className="md-skeleton h-4 w-64" />
        <div className="md-skeleton mt-6 h-[72px]" />
        <div className="md-skeleton h-[72px]" />
      </div>
    );
  }

  // Empty states chain: no documents → library; documents but no sessions → chat.
  if (!readyDocuments.length) {
    return (
      <Empty
        icon={<IconLibrary className="h-8 w-8" />}
        title="Add a document to begin"
        body="Research Desk answers questions from documents you provide, with citations you can open and verify. Nothing is indexed yet."
        action={
          <Button onClick={() => router.push("/library")}>
            <IconPlus />
            Go to Library
          </Button>
        }
      />
    );
  }

  if (!sessions.length) {
    return (
      <Empty
        icon={<IconChat className="h-8 w-8" />}
        title="Start your first conversation"
        body={`${readyDocuments.length} document${
          readyDocuments.length === 1 ? "" : "s"
        } indexed and ready. Follow-up questions resolve against the conversation, so you can ask "and the prior year?" and it will understand.`}
        action={
          <Button onClick={() => void start()} disabled={creating}>
            {creating ? <IconSpinner /> : <IconPlus />}
            New chat
          </Button>
        }
      />
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="md-headline-small">Chat</h1>
          <p
            className="md-body-medium mt-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Pick up where you left off, or start something new.
          </p>
        </div>
        <Button
          variant="tonal"
          onClick={() => void start()}
          disabled={creating}
          className="shrink-0"
        >
          {creating ? <IconSpinner /> : <IconPlus />}
          New
        </Button>
      </header>

      <ul className="space-y-3">
        {sessions.map((s) => (
          <li key={s.id}>
            <Ripplable
              as="div"
              className="md-card md-card-outlined md-card-interactive flex items-center gap-4 p-4"
              onClick={() => router.push(`/chat/${s.id}`)}
              role="link"
              tabIndex={0}
            >
              <span
                className="grid h-10 w-10 shrink-0 place-items-center rounded-[var(--md-shape-full)]"
                style={{
                  background: "var(--md-primary-container)",
                  color: "var(--md-on-primary-container)",
                }}
              >
                <IconChat className="h-5 w-5" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="md-title-small block truncate">{s.title}</span>
                <span
                  className="md-body-small mt-0.5 block"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {s.n_messages} messages ·{" "}
                  {new Date(s.updated_at).toLocaleDateString(undefined, {
                    day: "numeric",
                    month: "short",
                  })}
                  {s.document_ids.length > 0 &&
                    ` · ${s.document_ids.length} doc${
                      s.document_ids.length === 1 ? "" : "s"
                    } in scope`}
                </span>
              </span>
            </Ripplable>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Empty({
  icon,
  title,
  body,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  action: React.ReactNode;
}) {
  return (
    <div className="md-card md-card-filled mx-auto mt-9 max-w-2xl px-6 py-14 text-center">
      <span
        className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-[var(--md-shape-full)]"
        style={{
          background: "var(--md-primary-container)",
          color: "var(--md-on-primary-container)",
        }}
      >
        {icon}
      </span>
      <h1 className="md-title-large">{title}</h1>
      <p
        className="md-body-medium mx-auto mt-2 max-w-md"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {body}
      </p>
      <div className="mt-6 flex justify-center">{action}</div>
    </div>
  );
}

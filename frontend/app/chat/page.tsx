"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { type ChatSession, createSession, deleteSession, listSessions } from "@/lib/api";
import { useApp } from "../providers";
import { Button, Checkbox, ConfirmButton, Ripplable } from "../md";
import {
  IconChat,
  IconClose,
  IconLibrary,
  IconPlus,
  IconSpinner,
  IconTrash,
} from "../icons";

/** Page sizes offered. Small enough to scan, and 100 is the server's cap. */
const PAGE_SIZES = [10, 25, 50, 100] as const;

export default function ChatIndex() {
  const { sessions: recent, readyDocuments, loading, refreshSessions } = useApp();
  const router = useRouter();
  const [creating, setCreating] = useState(false);
  /**
   * THIS PAGE PAGES ITSELF, rather than rendering the drawer's copy.
   *
   * The drawer holds a short "jump back into something recent" list; this is
   * the archive, and the two want different lengths. Sharing one list meant
   * either the drawer was enormous or this page could not reach anything old.
   */
  const [page, setPage] = useState<ChatSession[]>([]);
  const [total, setTotal] = useState(0);
  const [size, setSize] = useState<number>(25);
  const [offset, setOffset] = useState(0);
  const [paging, setPaging] = useState(false);

  useEffect(() => {
    let live = true;
    setPaging(true);
    listSessions(size, offset)
      .then(({ sessions: rows, total: count }) => {
        if (!live) return;
        setPage(rows);
        setTotal(count);
      })
      .catch(() => {})
      .finally(() => live && setPaging(false));
    return () => {
      live = false;
    };
  }, [size, offset, recent.length]);

  const sessions = page;
  /**
   * Ids ticked for deletion. An empty set means selection mode is OFF and
   * a row click navigates as usual.
   *
   * Selection mode is entered explicitly rather than by long-press or by
   * the first checkbox click doubling as a mode switch: on a list whose
   * rows are links, a click that sometimes navigates and sometimes selects
   * is the kind of ambiguity that deletes the wrong conversation.
   */
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);

  const toggle = (id: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const exitSelect = () => {
    setSelecting(false);
    setPicked(new Set());
  };

  const allPicked = sessions.length > 0 && picked.size === sessions.length;

  // One control that toggles both ways rather than separate "Select all" and
  // "Clear" buttons. Once everything is ticked the only thing anyone wants
  // from that spot is to untick it, and a dead "Select all" sitting there is
  // a button that stopped meaning anything.
  const toggleAll = () =>
    setPicked(allPicked ? new Set() : new Set(sessions.map((s) => s.id)));

  async function removePicked() {
    setDeleting(true);
    try {
      // Concurrent, not sequential: these are independent DELETEs and
      // twelve round trips in series is a visible wait for no reason.
      //
      // allSettled, not all: one failure must not abandon the rest, and
      // the list refresh below shows exactly what survived -- which is a
      // truer report than any message this could write.
      await Promise.allSettled([...picked].map((id) => deleteSession(id)));
      exitSelect();
      await refreshSessions();
    } finally {
      setDeleting(false);
    }
  }

  async function start() {
    setCreating(true);
    try {
      const s = await createSession();
      // Navigate first; the list refresh is not needed to render the new chat
      // and awaiting it just delayed the navigation. See shell.tsx.
      router.push(`/chat/${s.id}`);
      void refreshSessions();
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
        <div className="flex shrink-0 items-center gap-2">
          {selecting ? (
            <>
              <Button variant="text" onClick={exitSelect} disabled={deleting}>
                <IconClose />
                Cancel
              </Button>
              <Button variant="text" onClick={toggleAll} disabled={deleting}>
                <Checkbox on={allPicked} />
                {allPicked ? "Clear" : "All"}
              </Button>
              {/* Rendered only when something is ticked, rather than shown
                  disabled. A disabled destructive button invites clicking it
                  to find out why it is dead; absent, it simply is not an
                  option yet. */}
              {picked.size > 0 && (
                <ConfirmButton
                  label={deleting ? "Deleting…" : `Delete (${picked.size})`}
                  icon={deleting ? <IconSpinner /> : <IconTrash />}
                  title={`Delete ${picked.size} conversation${
                    picked.size === 1 ? "" : "s"
                  }?`}
                  // Names the consequence rather than asking "are you sure".
                  // Deleting a chat also deletes its messages, and that is the
                  // part someone would not think of.
                  body="Their questions, answers and citations are deleted too. This cannot be undone."
                  confirmLabel={`Delete ${picked.size}`}
                  onConfirm={() => void removePicked()}
                />
              )}
            </>
          ) : (
            <>
              <Button variant="text" onClick={() => setSelecting(true)}>
                <IconTrash />
                Select
              </Button>
              <Button variant="tonal" onClick={() => void start()} disabled={creating}>
                {creating ? <IconSpinner /> : <IconPlus />}
                New
              </Button>
            </>
          )}
        </div>
      </header>

      <ul className="space-y-3">
        {sessions.map((s) => (
          <li key={s.id}>
            <Ripplable
              as="div"
              className="md-card md-card-outlined md-card-interactive flex items-center gap-4 p-4"
              // In selection mode the WHOLE ROW toggles rather than only the
              // checkbox. A 20px target inside a 72px row that is otherwise
              // clickable is the classic way to make people miss and open the
              // chat they were trying to delete.
              onClick={() =>
                selecting ? toggle(s.id) : router.push(`/chat/${s.id}`)
              }
              role={selecting ? "checkbox" : "link"}
              aria-checked={selecting ? picked.has(s.id) : undefined}
              tabIndex={0}
            >
              {/* Presentational only -- it is aria-hidden and has no handler.
                  The ROW carries role="checkbox" and aria-checked, so a screen
                  reader announces one control rather than a checkbox sitting
                  inside a separate clickable thing. */}
              {selecting && <Checkbox on={picked.has(s.id)} />}
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

      {/* Only when there is more than one page. A pager over eight rows is
          furniture. */}
      {total > size && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {offset + 1}–{Math.min(offset + sessions.length, total)} of {total}
          </span>

          <div className="flex items-center gap-2">
            <label
              className="md-body-small flex items-center gap-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Per page
              <select
                value={size}
                onChange={(e) => {
                  // Back to the first page. Staying at an offset that no
                  // longer exists in the new page size shows an empty list
                  // and looks like the chats are gone.
                  setOffset(0);
                  setSize(Number(e.target.value));
                }}
                className="md-body-small rounded-[var(--md-shape-sm)] px-2 py-1"
                style={{
                  background: "var(--md-surface-container-high)",
                  color: "var(--md-on-surface)",
                  border: "1px solid var(--md-outline-variant)",
                }}
              >
                {PAGE_SIZES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>

            <Button
              variant="text"
              size="sm"
              disabled={offset === 0 || paging}
              onClick={() => setOffset(Math.max(0, offset - size))}
            >
              Newer
            </Button>
            <Button
              variant="text"
              size="sm"
              disabled={offset + size >= total || paging}
              onClick={() => setOffset(offset + size)}
            >
              Older
            </Button>
          </div>
        </div>
      )}
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

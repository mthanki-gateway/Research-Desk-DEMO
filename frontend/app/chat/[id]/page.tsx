"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ChatMessage,
  type SessionDetail,
  deleteSession,
  getSession,
  streamTurn,
  updateSession,
} from "@/lib/api";
import { useApp } from "../../providers";
import { Button, Chip, Fab, TextField } from "../../md";
import { IconQuote, IconSpinner } from "../../icons";
import Rail, { RAIL_WIDTH, RAIL_WIDTH_COLLAPSED, type TurnSettings } from "./rail";

/**
 * Session details, kept across navigations.
 *
 * Without this, switching chats blanked the view and showed a full-page
 * skeleton for the duration of a round-trip. Now a session already opened
 * renders instantly from cache and refreshes in the background.
 *
 * Module scope rather than context: it is a cache, not state, and nothing
 * should re-render because it changed.
 */
const detailCache = new Map<string, SessionDetail>();

export default function ConversationPage() {
  const { id } = useParams<{ id: string }>();
  // Keyed so switching sessions gets clean local state — no leaking of the
  // previous conversation's draft, progress or error into the next one.
  return <Conversation key={id} id={id} />;
}

function Conversation({ id }: { id: string }) {
  const router = useRouter();
  const {
    sessions,
    readyDocuments,
    refreshSessions,
    showChunk,
    openChunk,
    closeChunk,
    setRailActive,
    railCollapsed,
    setRailCollapsed,
    railReady,
  } = useApp();

  const [session, setSession] = useState<SessionDetail | null>(
    () => detailCache.get(id) ?? null,
  );
  const [question, setQuestion] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [railOpen, setRailOpen] = useState(false);
  const [settings, setSettings] = useState<TurnSettings>({
    topK: 5,
    multiQuery: false,
  });
  const bottom = useRef<HTMLDivElement>(null);
  // The first scroll should jump, not glide. A smooth scroll on open read as
  // jank when moving between chats.
  const hasPainted = useRef(false);

  useEffect(() => {
    setRailActive(true);
    return () => {
      setRailActive(false);
      closeChunk();
    };
  }, [setRailActive, closeChunk]);

  const load = useCallback(async () => {
    try {
      const fresh = await getSession(id);
      detailCache.set(id, fresh);
      setSession(fresh);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session");
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!session) return;
    bottom.current?.scrollIntoView({
      behavior: hasPainted.current ? "smooth" : "auto",
      block: "end",
    });
    hasPainted.current = true;
  }, [session, pendingQuestion, progress]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    setQuestion("");
    setPendingQuestion(q); // optimistic: show it before the round-trip
    setBusy(true);
    setProgress("Thinking");
    setError(null);

    try {
      await streamTurn(
        id,
        q,
        { topK: settings.topK, multiQuery: settings.multiQuery },
        (_node, detail) => setProgress(detail),
      );
      await load();
      await refreshSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setQuestion(q); // never lose what they typed
    } finally {
      setBusy(false);
      setPendingQuestion(null);
      setProgress(null);
    }
  }

  async function toggleDoc(docId: string) {
    if (!session) return;
    const next = session.document_ids.includes(docId)
      ? session.document_ids.filter((d) => d !== docId)
      : [...session.document_ids, docId];
    // Optimistic, so the checkbox responds immediately.
    setSession({ ...session, document_ids: next });
    await updateSession(id, { document_ids: next });
    await load();
  }

  async function remove() {
    detailCache.delete(id);
    await deleteSession(id);
    await refreshSessions();
    router.push("/chat");
  }

  // Title is known from the sidebar list before the detail arrives, so the
  // header renders immediately rather than as a placeholder.
  const title =
    session?.title ?? sessions.find((s) => s.id === id)?.title ?? "Loading";

  return (
    <div
      className={`lg:pr-[var(--rail-pad)] ${
        railReady ? "transition-[padding]" : ""
      }`}
      style={
        {
          "--rail-pad": railCollapsed ? RAIL_WIDTH_COLLAPSED : RAIL_WIDTH,
          transitionDuration: "var(--md-dur-medium)",
          transitionTimingFunction: "var(--md-ease-emphasized)",
        } as React.CSSProperties
      }
    >
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-3xl flex-col px-6 py-6">
        {/* Top app bar, small */}
        <header
          className="sticky top-0 z-20 mb-5 flex h-16 items-center gap-3"
          style={{ background: "var(--md-surface-container-low)" }}
        >
          <h1 className="md-title-large min-w-0 flex-1 truncate">{title}</h1>
          {!session && <IconSpinner className="h-4 w-4 opacity-40" />}
          <Button
            variant="tonal"
            size="sm"
            onClick={() => setRailOpen(true)}
            className="shrink-0 lg:hidden"
          >
            Controls
          </Button>
        </header>

        <ol className="flex-1 space-y-6">
          {session?.messages.map((m) => (
            <Turn
              key={m.id}
              message={m}
              activeChunkId={openChunk?.id ?? null}
              onCite={showChunk}
            />
          ))}

          {session && session.messages.length === 0 && !pendingQuestion && (
            <li
              className="md-body-medium py-10 text-center"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Ask a question to begin. Answers cite the passages they came from.
            </li>
          )}

          {pendingQuestion && (
            <li className="flex justify-end">
              <p
                className="md-body-medium max-w-[80%] rounded-[var(--md-shape-lg)] px-4 py-3 opacity-60"
                style={{
                  background: "var(--md-primary-container)",
                  color: "var(--md-on-primary-container)",
                }}
              >
                {pendingQuestion}
              </p>
            </li>
          )}
        </ol>

        {error && (
          <p
            className="md-body-medium mt-4 rounded-[var(--md-shape-md)] px-4 py-3"
            style={{
              background: "var(--md-error-container)",
              color: "var(--md-on-error-container)",
            }}
          >
            {error}
          </p>
        )}

        <div ref={bottom} />

        <form
          onSubmit={send}
          className="sticky bottom-0 mt-6 pb-5 pt-3"
          style={{
            background:
              "linear-gradient(to top, var(--md-surface-container-low) 65%, transparent)",
          }}
        >
          <div className="flex items-end gap-3">
            <TextField
              label="Ask about your documents"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={busy || !session}
              surface="var(--md-surface-container-low)"
              className="flex-1"
            />
            <Fab
              type="submit"
              disabled={busy || !session || !question.trim()}
              aria-label="Send"
            >
              {busy ? (
                <IconSpinner className="h-6 w-6" />
              ) : (
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-6 w-6"
                  aria-hidden="true"
                >
                  <path d="M4 12h15M13 6l6 6-6 6" />
                </svg>
              )}
            </Fab>
          </div>
          <p
            className="md-body-small mt-2 flex h-4 items-center gap-1.5 px-1"
            style={{
              color: progress
                ? "var(--md-primary)"
                : "var(--md-on-surface-variant)",
            }}
          >
            {progress ? (
              <>
                <IconSpinner className="h-3.5 w-3.5" />
                {progress}
              </>
            ) : (
              'Follow-ups resolve against history — try "and the prior year?"'
            )}
          </p>
        </form>
      </div>

      {session && (
        <Rail
          session={session}
          documents={readyDocuments}
          settings={settings}
          chunk={openChunk}
          open={railOpen}
          collapsed={railCollapsed}
          animate={railReady}
          onClose={() => setRailOpen(false)}
          onCollapse={setRailCollapsed}
          onToggleDoc={toggleDoc}
          onSettings={setSettings}
          onClearChunk={closeChunk}
          onDeleteSession={remove}
        />
      )}
    </div>
  );
}

function Turn({
  message,
  activeChunkId,
  onCite,
}: {
  message: ChatMessage;
  activeChunkId: string | null;
  onCite: (chunkId: string) => Promise<void>;
}) {
  // User turns are M3-style sent bubbles: primary container, right-aligned.
  if (message.role === "user") {
    return (
      <li className="flex justify-end">
        <p
          className="md-body-medium max-w-[80%] rounded-[var(--md-shape-lg)] px-4 py-3"
          style={{
            background: "var(--md-primary-container)",
            color: "var(--md-on-primary-container)",
          }}
        >
          {message.content}
        </p>
      </li>
    );
  }

  const meta = message.agent_meta ?? {};
  const used = new Set(meta.sources_used ?? []);
  const cited = message.sources.filter((s) => used.has(s.n));
  const uncited = cited.length === 0;

  return (
    <li className="space-y-2">
      <div className="md-card md-card-elevated p-4">
        <p className="md-body-large whitespace-pre-wrap">{message.content}</p>

        {cited.length > 0 && (
          <div
            className="mt-4 flex flex-wrap items-center gap-2 border-t pt-3"
            style={{ borderColor: "var(--md-outline-variant)" }}
          >
            <IconQuote className="h-4 w-4 shrink-0 opacity-40" />
            {cited.map((s) => (
              <Chip
                key={s.chunk_id}
                size="sm"
                selected={s.chunk_id === activeChunkId}
                onClick={() => void onCite(s.chunk_id)}
                title="Read the source passage"
              >
                <span className="font-semibold tabular-nums">{s.n}</span>
                <span className="max-w-[15rem] truncate">
                  {s.heading ? s.heading.replace(/^#+\s*/, "") : s.filename}
                </span>
              </Chip>
            ))}
          </div>
        )}
      </div>

      <div
        className="md-body-small flex flex-wrap items-center gap-x-3 gap-y-1 px-1"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {meta.iterations != null && (
          <span className="tabular-nums">
            {meta.iterations} iteration{meta.iterations === 1 ? "" : "s"}
          </span>
        )}
        {meta.sub_questions && meta.sub_questions.length > 1 && (
          <span className="tabular-nums">
            {meta.sub_questions.length} sub-questions
          </span>
        )}
        {meta.sufficient === false && (
          <span style={{ color: "var(--md-tertiary)" }}>
            critic flagged gaps
          </span>
        )}
        {/* Zero citations means nothing in the library supported the answer —
            the shape a hallucination would take, so it gets the error role. */}
        {uncited && <span className="md-badge md-badge-error">no sources cited</span>}
      </div>
    </li>
  );
}

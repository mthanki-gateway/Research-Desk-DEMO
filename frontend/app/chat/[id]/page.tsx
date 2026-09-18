"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type Activity,
  type ChatMessage,
  type InterruptEvent,
  type ClarifyDecision,
  type QueryRay,
  type SessionDetail,
  type TurnOutcome,
  deleteSession,
  getQueryRay,
  getSession,
  sendFeedback,
  resumeTurn,
  streamTurn,
  updateSession,
} from "@/lib/api";
import { useApp } from "../../providers";
import { Answer } from "../../answer";
import { Button, Chip, Dialog, Fab, IconButton, LinkChip, TextArea } from "../../md";
import {
  IconAtlas,
  IconCheck,
  IconChevron,
  IconExternal,
  IconQuote,
  IconSpinner,
  IconThumbDown,
  IconThumbUp,
} from "../../icons";
import Clarify from "./clarify";
import RayView from "../rayview";
import type { PlotTheme } from "../../atlas/scatter";
import Rail from "./rail";
import {
  DEFAULT_TURN_SETTINGS,
  type TurnSettings,
  loadTurnSettings,
  saveTurnSettings,
} from "@/lib/prefs";

/**
 * "https://www.reuters.com/x/y?q=1" -> "reuters.com".
 *
 * Wrapped in try/catch because the URL comes from a search API, not from us:
 * a malformed one must degrade to showing the raw string, never throw during
 * render and blank the whole message.
 */
function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

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
  /**
   * Searches happening right now, newest first.
   *
   * A LIST rather than one line, because the agent runs independent lookups
   * CONCURRENTLY -- three searches can be one wall-clock step -- and
   * collapsing them into a single line would show one query while three were
   * in flight, making parallel work look sequential.
   *
   * Keyed by source+query so a finished search is marked done IN PLACE rather
   * than appended again, which would otherwise make the list jump.
   */
  const [activity, setActivity] = useState<
    {
      key: string;
      /** "search" | "lookup" | "remember" — decides the verb, not the styling. */
      kind: "search" | "lookup" | "remember";
      source: string;
      /** The query, the tool name, or the stored text, per `kind`. */
      label: string;
      done: boolean;
      n?: number;
    }[]
  >([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /**
   * The composer, so focus can be put back after a turn.
   *
   * It is `disabled` while a turn runs, and a disabled element cannot hold
   * focus -- the browser drops it the moment the attribute is set. So every
   * question cost a click to get back into the box, which is the opposite of
   * how a chat should feel.
   */
  const composer = useRef<HTMLTextAreaElement | null>(null);
  /**
   * Whether a turn was running on the previous render.
   *
   * Without it the effect below cannot tell "a turn just finished" from "this
   * page mounted", since both see `busy === false`. Focusing on mount would
   * steal focus on every navigation into a chat and yank the viewport down to
   * the composer.
   */
  const wasBusy = useRef(false);
  const [railOpen, setRailOpen] = useState(false);
  // Starts at the defaults, then the saved values arrive in an effect below.
  //
  // NOT `useState(loadTurnSettings)`: this is a client component but Next still
  // renders it on the server for the initial HTML, where `localStorage` does
  // not exist. Reading it in the initialiser either throws during SSR or makes
  // the server and client disagree about every toggle, which React reports as
  // a hydration mismatch.
  const [settings, setSettings] = useState<TurnSettings>(DEFAULT_TURN_SETTINGS);
  /**
   * The graph paused and is waiting on the human-in-the-loop prompt.
   *
   * Held here rather than in the message list because the turn does not exist
   * server-side yet: nothing is persisted while paused, so there is no message
   * to attach it to. It carries the `thread_id`, which is the only route back
   * to the checkpoint.
   */
  const [pendingClarify, setPendingClarify] = useState<InterruptEvent | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  // The first scroll should jump, not glide. A smooth scroll on open read as
  // jank when moving between chats.
  const hasPainted = useRef(false);

  // Restore saved settings once, after mount, for the SSR reason above.
  useEffect(() => {
    setSettings(loadTurnSettings());
  }, []);

  /**
   * Change settings and remember them.
   *
   * Writes on the user's action rather than in an effect watching `settings`.
   * Such an effect would also fire on the mount that still holds the defaults,
   * overwriting the stored values a moment before the restore effect above
   * replaced them -- correct in the end, but it puts the wrong thing in
   * storage in between, and a tab closed in that window would lose the prefs.
   */
  const changeSettings = useCallback((next: TurnSettings) => {
    setSettings(next);
    saveTurnSettings(next);
  }, []);

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

  /**
   * Keep the newest turn in view.
   *
   * Keyed on the message COUNT, not the `session` object: `load()` replaces
   * that object on every background refresh, so depending on it re-scrolled on
   * refreshes that changed nothing visible -- and fought the user if they had
   * scrolled up to read.
   *
   * `progress` is deliberately not a dependency either. It updates on every
   * graph node, and scrolling on each one yanked the view mid-read.
   *
   * The sentinel carries `scroll-mb-40`, which is what actually makes this
   * land: the composer is `sticky bottom-0` and overlays the end of the
   * document, so a plain scrollIntoView puts the newest turn *underneath* it.
   * That looked like the scroll had not happened at all.
   */
  const messageCount = session?.messages.length ?? 0;
  useEffect(() => {
    if (!session) return;
    // rAF: the optimistic bubble is added in the same commit, so without
    // waiting a frame the scroll measures the layout from before it existed
    // and stops one bubble short.
    const id = requestAnimationFrame(() => {
      bottom.current?.scrollIntoView({
        behavior: hasPainted.current ? "smooth" : "auto",
        block: "end",
      });
      hasPainted.current = true;
    });
    return () => cancelAnimationFrame(id);
    // `Boolean(pendingClarify)`, not the object: the review card appearing is
    // worth scrolling to, but editing inside it must not yank the view.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // activity.length too: a new search line appears at the bottom of the
    // list, and without it the newest one lands below the fold.
    //
    // `busy` is the definitive end-of-turn signal, and the layout changes
    // again at that moment for reasons no other dependency here sees: the
    // progress line and the activity list are both removed, so the document
    // gets SHORTER after the answer was already scrolled to. Without this the
    // view ends up short of the bottom by exactly the height of the progress
    // block that just disappeared.
  }, [
    messageCount,
    pendingQuestion,
    activity.length,
    busy,
    Boolean(pendingClarify),
    Boolean(session),
  ]);

  /**
   * Apply the end of a stream, which lands one of two ways.
   *
   * A pause is NOT an ending: nothing was persisted, so the optimistic question
   * bubble must stay on screen and the turn stays open until the human answers.
   * Reloading the session here would make the question vanish.
   */
  async function settle(outcome: TurnOutcome, q: string) {
    if (outcome.status === "paused") {
      setPendingClarify(outcome.interrupt);
      setPendingQuestion(q);
      return;
    }
    setPendingClarify(null);
    setPendingQuestion(null);
    await load();
    await refreshSessions();
  }

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    setQuestion("");
    setPendingQuestion(q); // optimistic: show it before the round-trip
    setPendingClarify(null);
    setBusy(true);
    setProgress("Thinking");
    setActivity([]);
    setError(null);

    try {
      const outcome = await streamTurn(
        id,
        q,
        {
          topK: settings.topK,
          multiQuery: settings.multiQuery,
          clarify: settings.clarify,
          react: settings.react,
          modelProfile: settings.modelProfile,
        },
        (_node, detail) => setProgress(detail),
        (a) => onActivity(a, setActivity),
      );
      await settle(outcome, q);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setQuestion(q); // never lose what they typed
      setPendingQuestion(null);
    } finally {
      setBusy(false);
      setProgress(null);
      setActivity([]);
    }
  }

  // Put the caret back in the composer once a turn finishes.
  useEffect(() => {
    if (busy) {
      wasBusy.current = true;
      return;
    }
    if (!wasBusy.current) return; // a mount, not a completion
    wasBusy.current = false;

    // Not while a clarifying question is open: focus belongs to its options,
    // and the composer is still disabled anyway.
    if (pendingClarify) return;

    // Pointer-based devices only. On a touch device, focusing a text field
    // throws the on-screen keyboard up over half the screen -- so the answer
    // that just arrived would be hidden by the act of being ready for the next
    // question.
    if (!window.matchMedia("(pointer: fine)").matches) return;

    // preventScroll, because FOCUS MUST NOT MOVE THE VIEWPORT.
    //
    // `focus()` scrolls its element into view by default, and the composer is
    // `sticky bottom-0` -- so the browser scrolls to where it sits in the
    // document flow, which is not the same place as the sentinel the scroll
    // effect above aims for (that one clears the composer via `scroll-mb-40`).
    // Two scrolls raced on every completed turn and the browser's landed last,
    // stopping short of the bottom: focus worked, the view did not follow.
    //
    // Scrolling is the other effect's job. This one only moves the caret.
    composer.current?.focus({ preventScroll: true });
  }, [busy, pendingClarify]);

  /** Answer the clarifying question: narrow the search, skip, or cancel. */
  async function answerClarify(decision: ClarifyDecision) {
    if (!pendingClarify || busy) return;
    // `original`, NOT `question`. Since clarification landed, `question` is the
    // model's clarifying question -- persisting the turn against that would put
    // the agent's words in the transcript where the user's belong.
    const q = pendingClarify.original;
    const thread = pendingClarify.thread_id;

    setBusy(true);
    setError(null);
    setProgress(decision.action === "cancel" ? "Stopping" : "Searching");

    try {
      const outcome = await resumeTurn(id, thread, q, decision, (_n, detail) =>
        setProgress(detail),
        (a) => onActivity(a, setActivity),
      );
      await settle(outcome, q);
    } catch (err) {
      // The pause is left in place on failure. The checkpoint still exists
      // server-side, so the decision can simply be retried -- clearing it here
      // would strand the thread with no way to reach it.
      setError(err instanceof Error ? err.message : "Could not resume the turn");
    } finally {
      setBusy(false);
      setProgress(null);
      setActivity([]);
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
    // The rail's width is reserved by <main>'s right MARGIN in shell.tsx, not
    // by padding here. As padding, main still ran under the fixed rail to the
    // window edge and its scrollbar went with it -- which is the whole reason
    // the chat scrollbar looked like a browser scrollbar.
    <div>
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-3xl flex-col px-6 py-6">
        {/* Top app bar, small. Sticky only works because html/body use
            `overflow-x: clip` rather than `hidden` -- `hidden` makes body a
            scroll container, which silently disables sticky in descendants.
            The negative margin plus padding lets the opaque background bleed
            to the column edges so text scrolling underneath is covered. */}
        <header
          className="sticky top-0 z-20 mb-5 -mx-6 flex h-16 items-center gap-3 px-6"
          style={{
            background: "var(--md-surface-container-low)",
            boxShadow: "0 1px 0 0 var(--md-outline-variant)",
          }}
        >
          {/* Back to the list, as every other detail page in this app has.
              The drawer holds only the most recent chats now that it is
              paginated, so an older conversation opened from the archive had
              no way back to it at all. */}
          <IconButton
            onClick={() => router.push("/chat")}
            aria-label="All chats"
            className="shrink-0"
          >
            <IconChevron className="h-5 w-5 rotate-180" />
          </IconButton>
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
              sessionId={id}
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
                className={`md-body-medium max-w-[80%] rounded-[var(--md-shape-lg)] px-4 py-3 ${
                  // Full opacity once paused: the turn is no longer in flight,
                  // it is waiting on the reader, and a faded bubble reads as
                  // "still sending".
                  pendingClarify ? "" : "opacity-60"
                }`}
                style={{
                  background: "var(--md-secondary-container)",
                  color: "var(--md-on-secondary-container)",
                }}
              >
                {pendingQuestion}
              </p>
            </li>
          )}

          {/* Live searches, in the position the ANSWER will occupy.
              Not above the composer: that form is `sticky bottom-0`, so
              anything growing inside it makes the form taller and -- anchored
              at the bottom -- pushes its top edge upward. A third search
              appearing visibly shoved the input up mid-turn.

              Here it also reads correctly: the work shows up where its result
              will, and is replaced by the answer rather than vanishing from a
              different part of the screen. */}
          {activity.length > 0 && (
            <li className="md-body-small space-y-1 px-1">
              {activity.map((a) => (
                <div
                  key={a.key}
                  className="flex items-center gap-2"
                  style={{
                    color: a.done
                      ? "var(--md-on-surface-variant)"
                      : "var(--md-primary)",
                  }}
                >
                  {a.done ? (
                    <IconCheck className="h-3.5 w-3.5 shrink-0" />
                  ) : (
                    <IconSpinner className="h-3.5 w-3.5 shrink-0" />
                  )}
                  <span className="truncate">
                    {a.kind === "lookup" ? (
                      <>
                        {a.done ? "Checked" : "Checking"} {toolLabel(a.label)}
                      </>
                    ) : a.kind === "remember" ? (
                      <>Remembered “{a.label}”</>
                    ) : (
                      <>
                        {a.done ? "Searched" : "Searching"}{" "}
                        {a.source === "web" ? "the web" : "your documents"} for
                        “{a.label}”
                      </>
                    )}
                  </span>
                  {a.done && a.n != null && (
                    <span className="shrink-0 tabular-nums opacity-70">
                      {a.n} result{a.n === 1 ? "" : "s"}
                    </span>
                  )}
                </div>
              ))}
            </li>
          )}

          {pendingClarify && (
            <Clarify
              interrupt={pendingClarify}
              busy={busy}
              onDecide={(d) => void answerClarify(d)}
            />
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

        {/* Scroll sentinel. `scroll-mb-40` (10rem) reserves room for the
            sticky composer, which otherwise covers the very thing we just
            scrolled to. */}
        <div ref={bottom} className="scroll-mb-40" />

        <form
          onSubmit={send}
          className="sticky bottom-0 -mx-6 mt-6 px-6 pb-5 pt-3"
          style={{
            background:
              "linear-gradient(to top, var(--md-surface-container-low) 65%, transparent)",
          }}
        >
          <div className="flex items-end gap-3">
            <TextArea
              ref={composer}
              label={
                pendingClarify
                  ? "Answer the question above to continue"
                  : "Ask about your documents"
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== "Enter" || e.shiftKey) return;
                // IME GUARD. While composing (Japanese, Chinese, Korean),
                // Enter CONFIRMS the candidate word -- it is not a submit.
                // Without this check the message sends mid-word every time
                // someone picks a character, which is invisible to anyone
                // testing in English.
                if (e.nativeEvent.isComposing) return;
                e.preventDefault();
                void send(e);
              }}
              // Locked while paused. A second question would start a second
              // turn and orphan the paused checkpoint, leaving a thread nothing
              // can ever reach.
              disabled={busy || !session || Boolean(pendingClarify)}
              // `--md-surface`, not the page's container-low: the composer
              // should read as a distinct input sitting ON the page rather
              // than a cut-out of it. Drives BOTH the input fill and the
              // floating label's background, so the notch matches whatever
              // the field is filled with.
              surface="var(--md-surface)"
              className="flex-1"
              // Pill composer. `shape` moves the floating label's inset in
              // step with the radius; see TextField.
              shape="var(--md-shape-xl)"
              // Chrome was offering previously-typed questions as autofill
              // history, dropping a suggestion list over the answer above the
              // composer. `autoComplete="off"` alone is unreliable here --
              // Chrome ignores it on fields it has already learned -- so the
              // field also carries no `name`, which is what its autofill
              // heuristics key on. spellCheck stays ON: this is prose.
              autoComplete="off"
              autoCorrect="off"
            />
            <Fab
              type="submit"
              disabled={busy || !session || !question.trim() || Boolean(pendingClarify)}
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
            // FIXED height, and it has to fit the tallest thing that can go in
            // here. At h-4 the key caps (border + line-height ~20px) overflowed
            // upward into the composer, so starting a turn appeared to nudge
            // the input. The line swaps between three different contents on
            // every turn; reserving the space means none of them move anything.
            className="md-body-small mt-2 flex h-5 items-center gap-1.5 px-1"
            style={{
              color: progress
                ? "var(--md-primary)"
                : pendingClarify
                  ? "var(--md-tertiary)"
                  : "var(--md-on-surface-variant)",
            }}
          >
            {progress ? (
              <>
                <IconSpinner className="h-3.5 w-3.5" />
                {progress}
              </>
            ) : pendingClarify ? (
              "Waiting on your answer — nothing has been searched yet"
            ) : (
              // Shift+Enter is invisible unless it is stated. A textarea that
              // submits on Enter looks identical to one that does not, so
              // without this the only way to discover the newline is to try it
              // and risk sending a half-written question.
              <>
                <kbd className="md-kbd">Shift</kbd>
                <span aria-hidden="true"> + </span>
                <kbd className="md-kbd">Enter</kbd> for a new line
              </>
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
          onSettings={changeSettings}
          onClearChunk={closeChunk}
          onDeleteSession={remove}
        />
      )}
    </div>
  );
}

function Turn({
  sessionId,
  message,
  activeChunkId,
  onCite,
}: {
  sessionId: string;
  message: ChatMessage;
  activeChunkId: string | null;
  onCite: (chunkId: string) => Promise<void>;
}) {
  // User turns are M3-style sent bubbles: primary container, right-aligned.
  if (message.role === "user") {
    return (
      <li className="flex flex-col items-end gap-1">
        <p
          className="md-body-medium max-w-[80%] rounded-[var(--md-shape-lg)] px-4 py-3"
          style={{
            background: "var(--md-secondary-container)",
            color: "var(--md-on-secondary-container)",
          }}
        >
          {message.content}
        </p>
        <QueryRayButton messageId={message.id} />
      </li>
    );
  }

  const meta = message.agent_meta ?? {};
  const used = new Set(meta.sources_used ?? []);
  const cited = message.sources.filter((s) => used.has(s.n));
  // A memory turn cites nothing BY DESIGN -- it stored an instruction and
  // never searched -- so the warning would be accusing it of the thing it was
  // supposed to do.
  const remembered = meta.intent === "remember";
  const uncited = cited.length === 0 && !remembered;

  return (
    <li className="space-y-2">
      {/* `.md-answer`, not `.md-card md-card-elevated`: the elevated card's
          background is surface-container-low, which is also the page
          background, so answers were a shadow around nothing. */}
      <div className="md-answer p-5">
        <Answer
          content={message.content}
          sources={message.sources}
          activeChunkId={activeChunkId}
          onCite={(chunkId) => void onCite(chunkId)}
        />

        {cited.length > 0 && (
          <div
            className="mt-4 flex flex-wrap items-center gap-2 border-t pt-3"
            style={{ borderColor: "var(--md-outline-variant)" }}
          >
            <IconQuote className="h-4 w-4 shrink-0 opacity-40" />
            {cited.map((s) =>
              // A web source has no chunk to open, so its chip links out
              // instead. It shows the HOSTNAME rather than the page title:
              // the title is already the citation's label, and what the
              // reader needs before clicking is who published it.
              s.source === "web" && s.url ? (
                <LinkChip
                  key={s.chunk_id}
                  size="sm"
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={s.url}
                >
                  <span className="font-semibold tabular-nums">{s.n}</span>
                  <span className="max-w-[15rem] truncate">
                    {hostname(s.url)}
                  </span>
                  <IconExternal className="h-3 w-3 shrink-0 opacity-60" />
                </LinkChip>
              ) : (
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
              ),
            )}
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
        {/* Only present when the user was asked and answered, so its absence
            means the question was clear rather than that it was confirmed. */}
        {meta.clarification && (
          <span
            className="md-badge md-badge-tertiary max-w-[18rem] truncate"
            title={`You narrowed this to: ${meta.clarification}`}
          >
            narrowed: {meta.clarification}
          </span>
        )}
        {/* Only when the turn was traced -- otherwise the score would have
            nowhere to attach and the buttons would silently do nothing. */}
        {meta.trace_id && <Feedback sessionId={sessionId} messageId={message.id} />}
        {/* Zero citations means nothing in the library supported the answer —
            the shape a hallucination would take, so it gets the error role. */}
        {uncited && <span className="md-badge md-badge-error">no sources cited</span>}
        {/* Deliberately NOT an error. The agent found part of the answer,
            cited it, and said what was missing — which is the intended
            outcome for a question the documents only partly cover. Marking
            it red would train the reader to distrust the honest case. */}
        {meta.partial && (
          <span className="md-badge md-badge-tertiary" title="Some of the question could not be answered from the sources">
            partial answer
          </span>
        )}
      </div>
    </li>
  );
}

/**
 * Thumbs up/down on one answer.
 *
 * The single highest-value quality signal in the whole system, and it costs a
 * button: it is the only measurement that reflects what the USER thought.
 * Every model-based metric — faithfulness, relevancy — is a proxy for this.
 *
 * Optimistic and irreversible by design. The choice renders immediately, and
 * there is no undo: a rating is an observation about a moment, and letting
 * people toggle it back and forth produces noise rather than data. A failed
 * request is swallowed for the same reason it is fire-and-forget on the API
 * side — there is nothing useful to tell someone whose feedback did not send,
 * and interrupting their reading to say so would be worse than losing it.
 */
function Feedback({
  sessionId,
  messageId,
}: {
  sessionId: string;
  messageId: string;
}) {
  const [sent, setSent] = useState<boolean | null>(null);

  function rate(helpful: boolean) {
    if (sent !== null) return;
    setSent(helpful);
    void sendFeedback(sessionId, messageId, helpful).catch(() => {
      /* ignore — see the note above */
    });
  }

  if (sent !== null) {
    return (
      <span className="flex items-center gap-1">
        {sent ? (
          <IconThumbUp className="h-3.5 w-3.5" />
        ) : (
          <IconThumbDown className="h-3.5 w-3.5" />
        )}
        Thanks
      </span>
    );
  }

  return (
    <span className="flex items-center gap-0.5">
      <button
        onClick={() => rate(true)}
        title="This answer was helpful"
        aria-label="This answer was helpful"
        className="md-icon-btn md-icon-btn-sm md-state"
      >
        <IconThumbUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => rate(false)}
        title="This answer was not helpful"
        aria-label="This answer was not helpful"
        className="md-icon-btn md-icon-btn-sm md-state"
      >
        <IconThumbDown className="h-3.5 w-3.5" />
      </button>
    </span>
  );
}


/**
 * "Where did this question land?" — the query ray, in a dialog.
 *
 * Attached to the QUESTION rather than to the answer, because the thing being
 * explained is the question: which region of the corpus it fell into, and
 * whether the passages it pulled back were neighbours of one another or three
 * unrelated stragglers. A score list cannot tell those apart.
 *
 * Fetched on FIRST OPEN, not with the transcript. It costs an embedding call
 * and a projection of the whole corpus, and the overwhelming majority of turns
 * are never asked about. Cached afterwards so reopening is instant.
 */
function QueryRayButton({ messageId }: { messageId: string }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<QueryRay | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The plot carries its own ground, independent of the app theme, exactly as
  // the Atlas does — a scatter needs a dark field to have contrast against,
  // whatever the surrounding page is doing.
  const [theme, setTheme] = useState<PlotTheme>("dark");

  async function show() {
    setOpen(true);
    if (data || loading) return;
    setLoading(true);
    setError(null);
    try {
      setData(await getQueryRay(messageId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not build the view");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => void show()}
        title="Where did this question land in the corpus?"
        className="md-label-small flex items-center gap-1 rounded-[var(--md-shape-full)] px-2 py-1 opacity-55 transition-opacity hover:opacity-100"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        <IconAtlas className="h-3.5 w-3.5" />
        Where did this land?
      </button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="Where this question landed"
        body={
          data
            ? `${data.rays.length} passage(s) retrieved from ${data.n_documents} document(s). The three axes account for ${Math.round(
                data.explained_variance.reduce((a, b) => a + b, 0) * 100,
              )}% of the real structure — a flat picture of a 768-dimensional space, so read it for shape, not for exact distance.`
            : undefined
        }
        wide="xl"
        contentClassName="mt-4"
      >
        <div className="h-[60vh] min-h-[22rem] w-full">
          {loading && (
            <p
              className="md-body-medium flex h-full items-center justify-center gap-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              <IconSpinner className="h-4 w-4" />
              Embedding the question and projecting the corpus
            </p>
          )}
          {error && (
            <p
              className="md-body-medium flex h-full items-center justify-center px-6 text-center"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {error}
            </p>
          )}
          {data && !loading && !error && <RayView data={data} theme={theme} />}
        </div>
        <div className="mt-3 flex items-center justify-between gap-3">
          <Button
            variant="text"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "Light field" : "Dark field"}
          </Button>
          <Button variant="tonal" onClick={() => setOpen(false)}>
            Close
          </Button>
        </div>
      </Dialog>
    </>
  );
}


/** Plain English for the metadata tools, which are named for the code. */
function toolLabel(tool: string): string {
  if (tool === "list_documents") return "your document list";
  if (tool === "corpus_stats") return "collection statistics";
  if (tool === "conversation_stats") return "your conversation history";
  return tool;
}

/**
 * Fold one progress event into the activity list.
 *
 * Extracted because the streaming call site appears twice -- a fresh turn and a
 * resumed one -- and the two copies had to agree. They did, until this grew a
 * second and third event kind.
 *
 * Keyed so a finished step is marked done IN PLACE rather than appended again,
 * which would make the list jump while the user is reading it.
 */
function onActivity(
  a: Activity,
  set: React.Dispatch<React.SetStateAction<ActivityRow[]>>,
) {
  let row: ActivityRow | null = null;
  if (a.kind === "search" || a.kind === "search_done") {
    if (!a.query) return;
    row = {
      key: `s:${a.source}:${a.query}`,
      kind: "search",
      source: a.source ?? "documents",
      label: a.query,
      done: a.kind === "search_done",
      n: a.n,
    };
  } else if (a.kind === "lookup" || a.kind === "lookup_done") {
    if (!a.tool) return;
    row = {
      key: `l:${a.tool}`,
      kind: "lookup",
      source: "documents",
      label: a.tool,
      done: a.kind === "lookup_done",
    };
  } else if (a.kind === "remember") {
    if (!a.text) return;
    // No paired done event: the write is over by the time it is announced.
    row = {
      key: `m:${a.text}`,
      kind: "remember",
      source: "documents",
      label: a.text,
      done: true,
    };
  } else {
    // "rerank" and anything added later: no query to show, and a line the
    // user cannot act on is worse than no line.
    return;
  }

  const next = row;
  set((prev) => {
    const at = prev.findIndex((x) => x.key === next.key);
    if (at >= 0) {
      const merged = [...prev];
      merged[at] = { ...merged[at], done: next.done, n: next.n ?? merged[at].n };
      return merged;
    }
    return [next, ...prev].slice(0, 6);
  });
}

type ActivityRow = {
  key: string;
  kind: "search" | "lookup" | "remember";
  source: string;
  label: string;
  done: boolean;
  n?: number;
};

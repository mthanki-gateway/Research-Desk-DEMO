"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  type BlueprintField,
  type DesignMessage,
  type HowlerInvite,
  type HowlerProject,
  createHowlerInvite,
  createHowlerProject,
  deleteHowlerProject,
  designHowlerProject,
  getHowlerProject,
  listHowlerInvites,
  listHowlerProjects,
  revokeHowlerInvite,
  synthesiseHowlerProject,
} from "@/lib/api";
import {
  Button,
  ConfirmButton,
  Fab,
  IconButton,
  Ripplable,
  TextArea,
} from "../../md";
import {
  IconCheck,
  IconChevron,
  IconCopy,
  IconHowler,
  IconLink,
  IconPlus,
  IconProfile,
  IconRefresh,
  IconSpinner,
  IconTrash,
} from "../../icons";
import ParleySurface, { ProfileCard } from "../surface";
import Transcript from "./transcript";
import VoiceTimeline from "./voiceTimeline";
import ConversationPanel from "./conversationPanel";

/**
 * Howler — design an interview by talking about it, then send someone a link.
 *
 * WHY THIS IS A CHAT AND NOT A FORM
 *
 * The first version was two text boxes and a Preview button. It worked, and it
 * was wrong: writing a brief is not data entry, it is the part where you work
 * out what you actually want to know. A form asks you to arrive already
 * knowing. A conversation gets there with you, and the questions it asks are
 * exactly the ones that turn a vague intention into a schema worth
 * interviewing against.
 *
 * THE SAME CHAT AS THE RESEARCH DESK, deliberately down to the details: a
 * sticky app bar, a column that fills the viewport so the composer sits at the
 * bottom of the SCREEN rather than wherever the last message happened to end,
 * Enter to send with the IME guard, a reserved-height status line, and the
 * caret returned to the composer when a turn finishes. Two chat surfaces in
 * one product that behave differently is two things to learn instead of one.
 *
 * THREE TABS, BECAUSE A PROJECT HAS THREE LIVES
 *
 *   Design   the conversation, with what it is producing beside it
 *   Links    who it has been sent to, and what state each one is in
 *   Results  what came back
 *
 * They are separated because they are used at different TIMES -- you design
 * today, send tomorrow, read next week -- and stacking all three into one
 * scroll meant the part you wanted was always the part below the fold. The
 * chat keeps the data points and the participant beside it, because those
 * change as you talk and watching them change is the point; links get a
 * summary there and their detail in their own tab.
 *
 * The tab lives in the URL, so a particular view is linkable and survives a
 * reload -- and so "open the results" is a link somebody can be sent.
 *
 * ROUTES
 *
 *   /parley/howler                  the projects
 *   /parley/howler?p=<id>&tab=…     designing, sending, reading
 *   /parley/howler?c=<id>           a live interview, run by the owner
 */
export default function HowlerPage() {
  return (
    <Suspense fallback={<Skeleton />}>
      <Howler />
    </Suspense>
  );
}

function Howler() {
  const params = useSearchParams();
  const conversation = params.get("c");
  const project = params.get("p");

  // READING a conversation and HOLDING one are different pages.
  //
  // `?c=` used to hand over to the live surface, so opening a finished result
  // came with a microphone, a voice picker and a keep-the-socket-open toggle
  // -- controls for running an interview, shown to somebody who came to read
  // one. `?live=1` is the owner running one themselves; anything else reads.
  if (conversation) {
    return params.get("live") ? (
      <ParleySurface mode="howler" />
    ) : (
      <Transcript id={conversation} projectId={project} />
    );
  }
  return project ? <Project id={project} /> : <Projects />;
}

function Skeleton() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9" aria-hidden>
      <div className="md-skeleton h-8 w-40" />
      <div className="md-skeleton h-[22rem]" />
    </div>
  );
}

/* ------------------------------------------------------------- the projects */

function Projects() {
  const router = useRouter();
  const [projects, setProjects] = useState<HowlerProject[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listHowlerProjects()
      .then(setProjects)
      .catch((e) => {
        setProjects([]);
        setError(e instanceof Error ? e.message : "Could not load projects");
      });
  }, []);

  useEffect(refresh, [refresh]);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const made = await createHowlerProject();
      router.push(`/parley/howler?p=${made.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start a project");
      setBusy(false);
    }
  }

  async function remove(id: string) {
    try {
      await deleteHowlerProject(id);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete that");
    }
  }

  if (projects === null) return <Skeleton />;

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="md-headline-small flex items-center gap-2">
            <IconHowler className="h-6 w-6" />
            Howler
          </h1>
          <p
            className="md-body-medium mt-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Talk through what you want to find out. It works out the data points
            with you, then gives you a link to send to whoever you are
            interviewing.
          </p>
        </div>
        <Button
          variant="tonal"
          onClick={() => void start()}
          disabled={busy}
          className="shrink-0"
        >
          {busy ? <IconSpinner /> : <IconPlus />}
          New
        </Button>
      </header>

      {error && <Banner>{error}</Banner>}

      {projects.length === 0 ? (
        <div className="md-card md-card-filled px-6 py-14 text-center">
          <span
            className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-[var(--md-shape-full)]"
            style={{
              background: "var(--md-primary-container)",
              color: "var(--md-on-primary-container)",
            }}
          >
            <IconHowler className="h-8 w-8" />
          </span>
          <h2 className="md-title-large">Nothing designed yet</h2>
          <p
            className="md-body-medium mx-auto mt-2 max-w-md"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Start with a sentence about what you want to know — &ldquo;I want to
            interview people who just cancelled&rdquo; is enough. The rest is a
            conversation.
          </p>
          <div className="mt-6 flex justify-center">
            <Button onClick={() => void start()} disabled={busy}>
              {busy ? <IconSpinner /> : <IconPlus />}
              New project
            </Button>
          </div>
        </div>
      ) : (
        <ul className="space-y-3">
          {projects.map((p) => (
            <li key={p.id}>
              <Ripplable
                as="div"
                className="md-card md-card-outlined md-card-interactive flex items-center gap-4 p-4"
                onClick={() => router.push(`/parley/howler?p=${p.id}`)}
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
                  <IconHowler className="h-5 w-5" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="md-title-small block truncate">
                    {p.title}
                  </span>
                  <span
                    className="md-body-small mt-0.5 block"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    {p.fields.length} data point
                    {p.fields.length === 1 ? "" : "s"}
                    {p.invites
                      ? ` · ${p.invites} link${p.invites === 1 ? "" : "s"}`
                      : " · no links yet"}
                  </span>
                </span>
                {/* `stopPropagation` on the wrapper, not the button: the card
                    navigates, and a confirm dialog opening inside it would
                    otherwise take the click through to the router. */}
                <span onClick={(e) => e.stopPropagation()}>
                  <ConfirmButton
                    label="Delete"
                    icon={<IconTrash />}
                    title={`Delete ${p.title}?`}
                    body="Its data points, its links and the interviews they gathered are deleted too. This cannot be undone."
                    confirmLabel="Delete"
                    onConfirm={() => void remove(p.id)}
                  />
                </span>
              </Ripplable>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------- one project */

type Tab = "design" | "links" | "results";
type Line = DesignMessage;

function Project({ id }: { id: string }) {
  const router = useRouter();
  const params = useSearchParams();
  const tab = (params.get("tab") as Tab) || "design";

  const [project, setProject] = useState<HowlerProject | null>(null);
  const [invites, setInvites] = useState<HowlerInvite[]>([]);
  const [lines, setLines] = useState<Line[]>([]);
  const [busy, setBusy] = useState<"say" | "synthesise" | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * A link appeared while you were not looking at the links.
   *
   * Synthesising is the moment the project becomes sendable, and it happens in
   * the middle of a conversation about something else -- so the one thing
   * produced by pressing that button is on a tab you are not on. The dot says
   * so, and it clears the moment you look.
   */
  const [fresh, setFresh] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const go = useCallback(
    (next: Tab) => router.replace(`/parley/howler?p=${id}&tab=${next}`),
    [router, id],
  );

  useEffect(() => {
    if (tab === "links") setFresh(false);
  }, [tab]);

  useEffect(() => {
    getHowlerProject(id)
      .then((p) => {
        setProject(p);
        setLines(p.design);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not open that"),
      );
    listHowlerInvites(id)
      .then(setInvites)
      .catch(() => {});
  }, [id]);

  /** One designer turn, whichever button started it. */
  const turn = useCallback(
    async (
      text: string,
      run: () => Promise<{
        reply: string;
        project: HowlerProject;
        invite?: HowlerInvite | null;
      }>,
      kind: "say" | "synthesise",
    ): Promise<boolean> => {
      setBusy(kind);
      setError(null);
      setLines((prev) => [...prev, { role: "user", content: text }]);
      try {
        const out = await run();
        setProject(out.project);
        // The STORED transcript replaces the optimistic one rather than being
        // appended to. It is the same lines plus the reply, and taking the
        // server's copy is what keeps a reload showing what the screen showed.
        setLines(out.project.design);
        if (out.invite) {
          const made = out.invite;
          setInvites((prev) => [made, ...prev.filter((i) => i.id !== made.id)]);
          setFresh(true);
        }
        return true;
      } catch (e) {
        // Take the optimistic line back out -- exactly the one just added.
        // Leaving it there reads as sent, and the designer never saw it.
        setLines((prev) => prev.slice(0, -1));
        setError(e instanceof Error ? e.message : "That did not go through");
        return false;
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  const synthesise = useCallback(async () => {
    if (busy) return;
    await turn("Synthesise now.", () => synthesiseHowlerProject(id), "synthesise");
  }, [busy, turn, id]);

  const addLink = useCallback(
    async (label: string) => {
      try {
        const made = await createHowlerInvite(
          id,
          label,
          project?.participant ?? "",
        );
        setInvites((prev) => [made, ...prev]);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not create a link");
      }
    },
    [id, project],
  );

  /** Re-read the links and what has come back through them.
   *
   * Results arrive from SOMEBODY ELSE'S browser -- a participant talking into
   * their own link, minutes or days after this page was opened -- so there is
   * nothing here that could know to update itself. Reloading the whole page
   * worked and threw away the conversation view to do it.
   */
  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [rows, latest] = await Promise.all([
        listHowlerInvites(id),
        getHowlerProject(id),
      ]);
      setInvites(rows);
      setProject(latest);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not refresh");
    } finally {
      setRefreshing(false);
    }
  }, [id]);

  const revoke = useCallback(
    async (inviteId: string) => {
      try {
        await revokeHowlerInvite(inviteId);
        setInvites(await listHowlerInvites(id));
      } catch (e) {
        setError(
          e instanceof Error ? e.message : "Could not withdraw that link",
        );
      }
    },
    [id],
  );

  if (!project && !error) return <Skeleton />;

  const results = invites.filter((i) => i.result);
  // "Live" is a link somebody can still use: not withdrawn, and not attached
  // to an interview that has already finished.
  const usable = invites.filter(
    (i) => i.status === "unopened" || i.status === "in_progress",
  );
  const live = invites.filter((i) => !i.revoked).length;

  return (
    <div className="mx-auto max-w-6xl">
      {/* THE BAR SPANS THE WHOLE PAGE, above both columns -- it is the
          project's chrome, not the conversation's, and inside the chat column
          the tabs stopped short of the panel and read as belonging to the
          chat alone.

          App bar and tabs are ONE sticky block: two separately sticky
          elements let the bar slide under the tabs, and the tabs' bottom rule
          then cut across the title. */}
      <div
        className="sticky top-0 z-20"
        style={{ background: "var(--md-surface-container-low)" }}
      >
        <header className="flex h-16 items-center gap-2 px-6">
          <IconButton
            onClick={() => router.push("/parley/howler")}
            aria-label="All projects"
            className="shrink-0"
          >
            <IconChevron className="h-5 w-5 rotate-180" />
          </IconButton>
          <h1 className="md-title-large min-w-0 flex-1 truncate">
            {project?.title ?? "Loading"}
          </h1>
          {/* In the bar rather than over the composer: it is the action that
              settles this project, it should be reachable from any tab and any
              scroll position, and a second button beside the send key is the
              one place a stray Enter can go somewhere expensive.

              IN TERTIARY, which is the point of having a third colour role.
              Primary is spent on the drawer, the active tab and the send key,
              so a primary button here was one more purple thing in a row of
              purple things. Tertiary reads as a different KIND of action at a
              glance, which is what it is: everything else on this screen edits
              the project, and this one publishes it.

              Full size and ringed until a link exists; afterwards it drops to
              the quieter container pair, because by then it is a revision
              rather than the point. */}
          {/* ONLY WHILE THERE IS NO USABLE LINK.
              "Synthesise again" was doing nothing worth a model call. A link
              reads the project's data points when the CONVERSATION STARTS, not
              when the link was made, so one that has not been opened yet
              already gathers whatever the latest turn produced -- there is
              nothing to re-settle and no new link to hand back.

              It still matters when there is no usable link: the first one, and
              a replacement after the last was withdrawn or its interview
              finished. From then on the Links tab owns making them. */}
          {lines.length > 0 && usable.length === 0 && (
            <Button
              onClick={() => void synthesise()}
              disabled={busy !== null}
              className="shrink-0"
              style={{
                background: invites.length
                  ? "var(--md-tertiary-container)"
                  : "var(--md-tertiary)",
                color: invites.length
                  ? "var(--md-on-tertiary-container)"
                  : "var(--md-on-tertiary)",
                // A steady ring, not the tab's ping: this one sits and waits
                // rather than announcing something that just happened, and a
                // pulsing filled button is an alarm.
                ...(!invites.length && (project?.fields ?? []).length > 0
                  ? {
                      boxShadow:
                        "0 0 0 4px color-mix(in srgb, var(--md-tertiary) 24%, transparent)",
                    }
                  : {}),
              }}
            >
              {busy === "synthesise" ? <IconSpinner /> : <IconCheck />}
              {invites.length ? "New link" : "Synthesise now"}
            </Button>
          )}
        </header>

        <div className="md-tabs px-6" role="tablist">
          <TabButton active={tab === "design"} onClick={() => go("design")}>
            Design
          </TabButton>
          <TabButton
            active={tab === "links"}
            onClick={() => go("links")}
            count={live || undefined}
            alert={fresh}
          >
            Links
          </TabButton>
          <TabButton
            active={tab === "results"}
            onClick={() => go("results")}
            count={results.length || undefined}
          >
            Results
          </TabButton>
        </div>
      </div>

      {error && (
        <p
          className="md-body-medium mx-6 mt-4 rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      {tab === "design" ? (
        /* NO `items-start` HERE, and that was the bug: it shrank the aside to
           the height of its own content, so the sticky panel inside had zero
           room to travel and simply scrolled away with the conversation. A
           stretched aside is as tall as the chat beside it, which is exactly
           the distance the panel needs to stay put over. */
        <div className="lg:flex lg:gap-8">
          {/* `lg:order-2` puts the panel on the RIGHT on a wide screen and
              ABOVE the conversation on a narrow one. Below would be the
              obvious stacking and it is wrong: the composer is sticky to the
              bottom, so anything after it is the one part nobody scrolls to. */}
          <aside className="px-6 pt-5 lg:order-2 lg:w-[23rem] lg:shrink-0 lg:pl-0">
            <Panel
              project={project}
              invites={invites}
              onOpenLinks={() => go("links")}
            />
          </aside>

          {/* The column is at least a screen tall EVEN WHEN NEARLY EMPTY, and
              that is the whole reason the composer sits at the bottom of the
              window rather than floating under the last message: `flex-1` on
              the list pushes the form down, and there is nothing to push
              against unless the column has a height to fill. `7rem` is the
              sticky bar above it -- a 4rem app bar plus 3rem of tabs. */}
          <div className="flex min-h-[calc(100vh-7rem)] min-w-0 flex-1 flex-col pt-5 lg:order-1">
            <Design
              id={id}
              project={project}
              lines={lines}
              busy={busy}
              onTurn={turn}
            />
          </div>
        </div>
      ) : (
        <div className="flex min-h-[calc(100vh-7rem)] flex-col pt-6">
          {tab === "links" ? (
            <Links
              invites={invites}
              ready={(project?.fields ?? []).length > 0}
              onAdd={(label) => void addLink(label)}
              onRevoke={(inviteId) => void revoke(inviteId)}
              onSynthesise={() => void synthesise()}
            />
          ) : (
            <Results
              invites={invites}
              onOpenLinks={() => go("links")}
              onRefresh={() => void refresh()}
              refreshing={refreshing}
            />
          )}
        </div>
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  count,
  alert,
  children,
}: {
  active: boolean;
  onClick: () => void;
  count?: number;
  alert?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Ripplable
      as="button"
      type="button"
      className="md-tab"
      data-active={active}
      onClick={onClick}
      role="tab"
      aria-selected={active}
    >
      {children}
      {count !== undefined && (
        <span className="md-label-small opacity-70">{count}</span>
      )}
      {alert && (
        // A ping ring around a solid dot. The ring is `animate-ping`, which
        // scales and fades on a loop; the dot underneath stays put so the mark
        // is legible at every frame rather than blinking out of existence.
        <span className="relative ml-0.5 flex h-2 w-2" aria-label="New link">
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
            style={{ background: "var(--md-primary)" }}
          />
          <span
            className="relative inline-flex h-2 w-2 rounded-full"
            style={{ background: "var(--md-primary)" }}
          />
        </span>
      )}
    </Ripplable>
  );
}

/* --------------------------------------------------------------- the design */

function Design({
  id,
  project,
  lines,
  busy,
  onTurn,
}: {
  id: string;
  project: HowlerProject | null;
  lines: Line[];
  busy: "say" | "synthesise" | null;
  onTurn: (
    text: string,
    run: () => Promise<{
      reply: string;
      project: HowlerProject;
      invite?: HowlerInvite | null;
    }>,
    kind: "say" | "synthesise",
  ) => Promise<boolean>;
}) {
  const [message, setMessage] = useState("");
  const bottom = useRef<HTMLDivElement | null>(null);
  const composer = useRef<HTMLTextAreaElement | null>(null);
  const wasBusy = useRef(false);

  // Follow the conversation down. `block: "end"` plus the sentinel's
  // `scroll-mb-40` keeps the last line clear of the sticky composer, which
  // would otherwise cover the very thing just scrolled to.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lines.length, busy]);

  // Put the caret back in the composer once a turn finishes.
  //
  // In an EFFECT, not in the `finally` of the request. The textarea is
  // `disabled` while a turn is in flight, and focusing a disabled element does
  // nothing at all; the `finally` runs before React has re-rendered it
  // enabled, so every attempt from there was a silent no-op.
  useEffect(() => {
    if (busy) {
      wasBusy.current = true;
      return;
    }
    if (!wasBusy.current) return; // a mount, not a completion
    wasBusy.current = false;

    // Pointer devices only. On a touch device, focusing a text field throws
    // the on-screen keyboard over half the screen -- so the reply that just
    // arrived would be hidden by the act of being ready for the next message.
    if (!window.matchMedia("(pointer: fine)").matches) return;

    // preventScroll, because focus must not move the viewport: `focus()`
    // scrolls its element into view, and the composer is `sticky bottom-0`, so
    // the browser aims at where it sits in the document flow rather than at
    // the sentinel. The two scrolls race and the browser's lands last.
    composer.current?.focus({ preventScroll: true });
  }, [busy]);

  async function say(e?: React.FormEvent) {
    e?.preventDefault();
    const text = message.trim();
    if (!text || busy) return;
    setMessage("");
    const ok = await onTurn(text, () => designHowlerProject(id, text), "say");
    // Put it back in the composer on failure, so a dropped turn does not cost
    // somebody the paragraph they just typed.
    if (!ok) setMessage((m) => m || text);
  }

  const opening = lines.length === 0;

  return (
    <>
      <ol className="flex-1 space-y-6 px-6">
        {opening && (
          <li className="md-card md-card-filled px-6 py-12 text-center">
            <span
              className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
              style={{
                background: "var(--md-primary-container)",
                color: "var(--md-on-primary-container)",
              }}
            >
              <IconHowler className="h-7 w-7" />
            </span>
            <h2 className="md-title-medium">What do you want to find out?</h2>
            <p
              className="md-body-medium mx-auto mt-2 max-w-md"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              A sentence is enough to start — who you are, who you are
              interviewing, and what you are trying to learn. It drafts the data
              points as you talk and asks about the parts you have not said yet.
            </p>
          </li>
        )}

        {lines.map((line, i) =>
          line.role === "user" ? (
            <li key={`${i}-u`} className="flex justify-end">
              <p
                className="md-body-medium max-w-[80%] whitespace-pre-wrap rounded-[var(--md-shape-lg)] px-4 py-3"
                style={{
                  background: "var(--md-secondary-container)",
                  color: "var(--md-on-secondary-container)",
                  // Faded only while it is the one in flight, matching the
                  // Research Desk's pending question.
                  opacity: busy && i === lines.length - 1 ? 0.6 : 1,
                }}
              >
                {line.content}
              </p>
            </li>
          ) : (
            <li key={`${i}-a`}>
              {/* `.md-answer`, the same class the Research Desk renders an
                  answer into: it carries the prose rhythm, and a reply here
                  should not be a differently shaped thing. */}
              <div className="md-answer p-5">
                <p className="whitespace-pre-wrap">{line.content}</p>
              </div>
            </li>
          ),
        )}
      </ol>

      {/* Scroll sentinel. `scroll-mb-40` (10rem) reserves room for the sticky
          composer, which otherwise covers what we scrolled to. */}
      <div ref={bottom} className="scroll-mb-40" />

      <form
        onSubmit={say}
        className="sticky bottom-0 mt-6 px-6 pb-5 pt-3"
        style={{
          background:
            "linear-gradient(to top, var(--md-surface-container-low) 65%, transparent)",
        }}
      >
        <div className="flex items-end gap-3">
          <TextArea
            ref={composer}
            label={
              opening
                ? "Say what you want to find out"
                : "Keep going, or change something"
            }
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || e.shiftKey) return;
              // IME GUARD. While composing (Japanese, Chinese, Korean), Enter
              // CONFIRMS the candidate word -- it is not a submit. Without
              // this the message sends mid-word every time someone picks a
              // character, invisibly to anyone testing in English.
              if (e.nativeEvent.isComposing) return;
              e.preventDefault();
              void say();
            }}
            disabled={busy !== null || !project}
            // `--md-surface`, not the page's container-low: the composer
            // should read as a distinct input sitting ON the page rather than
            // a cut-out of it.
            surface="var(--md-surface)"
            className="flex-1"
            // Pill composer. `shape` moves the floating label's inset in step
            // with the radius.
            shape="var(--md-shape-xl)"
            autoComplete="off"
            autoCorrect="off"
          />
          <Fab
            type="submit"
            disabled={busy !== null || !project || !message.trim()}
            aria-label="Send"
          >
            {busy === "say" ? (
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
          // FIXED height, sized for the tallest thing that can go in it. The
          // line swaps contents on every turn; reserving the space means none
          // of them nudge the composer as they come and go.
          className="md-body-small mt-2 flex h-5 items-center gap-1.5 px-1"
          style={{
            color: busy ? "var(--md-primary)" : "var(--md-on-surface-variant)",
          }}
        >
          {busy ? (
            <>
              <IconSpinner className="h-3.5 w-3.5" />
              {busy === "synthesise" ? "Writing the data points" : "Thinking"}
            </>
          ) : (
            // Shift+Enter is invisible unless it is stated: a textarea that
            // submits on Enter looks identical to one that does not.
            <>
              <kbd className="md-kbd">Shift</kbd>
              <span aria-hidden="true"> + </span>
              <kbd className="md-kbd">Enter</kbd> for a new line
            </>
          )}
        </p>
      </form>
    </>
  );
}

/* ---------------------------------------------------------------- the panel */

function Panel({
  project,
  invites,
  onOpenLinks,
}: {
  project: HowlerProject | null;
  invites: HowlerInvite[];
  onOpenLinks: () => void;
}) {
  const fields = project?.fields ?? [];
  const vocabulary = project?.vocabulary ?? [];
  const live = invites.filter((i) => !i.revoked).length;
  /**
   * Narrow screens only. On a phone the panel sits ABOVE the conversation --
   * the only place it can go, since below a sticky composer is nowhere -- and
   * expanded it would push the chat off the first screen entirely. So: one
   * line saying what is in it, opened when wanted.
   */
  const [open, setOpen] = useState(false);

  return (
    /* STICKY, so the cards hold their place while the conversation moves.
       Sticking is only half of it: an element taller than the viewport still
       travels, because it scrolls until its own bottom edge arrives -- which
       is exactly what happened the moment the descriptions were expanded. The
       column below is therefore CAPPED as well, and scrolls internally.

       `top-[7.5rem]` clears the sticky bar above -- a 4rem app bar plus 3rem
       of tabs, plus a little air. At `top-0` the cards slid underneath it and
       the first one was decapitated by the tab rule. */
    <div className="lg:sticky lg:top-[7.5rem]">
      <button
        type="button"
        onClick={() => setOpen((w) => !w)}
        aria-expanded={open}
        className="md-card md-card-outlined md-state mb-3 flex w-full items-center gap-2 px-4 py-3 lg:hidden"
      >
        <IconChevron className="h-4 w-4 shrink-0" open={open} />
        <span className="md-label-large min-w-0 flex-1 truncate text-left">
          {fields.length
            ? `${fields.length} data point${fields.length === 1 ? "" : "s"}`
            : "No data points yet"}
          {live ? ` · ${live} link${live === 1 ? "" : "s"}` : ""}
        </span>
      </button>

      {/* CAPPED TO THE VIEWPORT, so the block cannot outgrow the screen. A
          sticky element taller than its viewport still travels -- it scrolls
          until its bottom edge arrives -- which is why expanding the
          descriptions made the whole panel start moving with the chat again.

          The cards are a flex column and only ONE of them flexes: the data
          points, which is the section that grows without bound. The other two
          are `shrink-0`, so "who you are interviewing" and the links keep
          their place on screen no matter how many fields there are. */}
      <div
        className={`space-y-3 lg:flex lg:max-h-[calc(100vh-9rem)] lg:flex-col lg:gap-3 lg:space-y-0 ${
          open ? "" : "hidden"
        }`}
      >
        <Section
          title="Data points"
          count={fields.length || "none yet"}
          // `min-h-0` is what makes the scroll work at all: a flex item's
          // default `min-height: auto` refuses to shrink below its content, so
          // without it the section grows past the cap and overflows the column
          // instead of scrolling within it.
          className="lg:flex lg:min-h-0 lg:flex-col"
          bodyClassName="md-scroll lg:min-h-0 lg:overflow-y-auto"
        >
          {(detail) =>
            fields.length === 0 ? (
              <p
                className="md-body-small"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                These appear as you talk, and change when you change your mind.
              </p>
            ) : (
              <>
                {/* THE LIST IS ALWAYS VISIBLE; collapsing hides the
                    DESCRIPTIONS, not the data points. Which fields exist is
                    the thing worth glancing at while you talk; the
                    descriptions are instruction for the interviewer, worth
                    reading once and in the way thereafter. */}
                <ul className="-mx-1">
                  {fields.map((f) => (
                    <FieldRow key={f.name} field={f} detail={detail} />
                  ))}
                </ul>
                {detail && (
                  <p
                    className="md-body-small mt-3 border-t pt-3"
                    style={{
                      color: "var(--md-on-surface-variant)",
                      borderColor: "var(--md-outline-variant)",
                    }}
                  >
                    An interview takes this list when it starts. Change them and
                    the link gathers the new ones; an interview already running
                    keeps the ones it was given.
                  </p>
                )}
              </>
            )
          }
          {/* Rendered whether open or closed -- see the render prop above. */}
        </Section>

        {project?.participant && (
          <Section
            title="Who you are interviewing"
            collapses
            className="lg:shrink-0"
            // Its own cap. A participant paragraph is usually four lines and
            // occasionally twenty, and one long one would otherwise take the
            // room the data points need.
            bodyClassName="md-scroll lg:max-h-56 lg:overflow-y-auto"
          >
            <p className="md-body-small whitespace-pre-wrap">
              {project.participant}
            </p>
            <p
              className="md-body-small mt-3 border-t pt-3"
              style={{
                color: "var(--md-on-surface-variant)",
                borderColor: "var(--md-outline-variant)",
              }}
            >
              Context for the interviewer. Never read back to them, and what
              they say always wins over it.
            </p>
          </Section>
        )}

        {vocabulary.length > 0 && (
          <Section
            title="Words it will hear"
            count={vocabulary.length}
            collapses
            className="lg:shrink-0"
            bodyClassName="md-scroll lg:max-h-40 lg:overflow-y-auto"
          >
            <div className="flex flex-wrap gap-1.5">
              {vocabulary.map((term) => (
                <span key={term} className="md-badge">
                  {term}
                </span>
              ))}
            </div>
            <p
              className="md-body-small mt-3 border-t pt-3"
              style={{
                color: "var(--md-on-surface-variant)",
                borderColor: "var(--md-outline-variant)",
              }}
            >
              {/* Said plainly because it looks like data and is not. */}
              Not things to ask about — spellings. Speech recognition mangles
              exactly the words that matter, so these are given to the
              microphone and to the interviewer. Say so in the chat if one is
              wrong or missing.
            </p>
          </Section>
        )}

        {/* A SUMMARY, not the list. The detail is a tab away, and duplicating
            it here is what made this panel taller than the screen. */}
        <section className="md-card md-card-outlined p-4 lg:shrink-0">
          <h2 className="md-title-small flex items-center gap-2">
            <IconLink className="h-4 w-4 shrink-0" />
            <span className="flex-1">Links</span>
            <span
              className="md-label-small"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {live || "none"}
            </span>
          </h2>
          <p
            className="md-body-small mt-2"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {live
              ? "Send one to whoever you are interviewing. It gathers whatever the data points say when they open it, so keep changing them if you want."
              : fields.length
                ? "Press Synthesise now and you will get a link to send."
                : "Once there are data points worth gathering, you will get a link to send."}
          </p>
          {invites.length > 0 && (
            <div className="mt-2">
              <Button variant="text" size="sm" onClick={onOpenLinks}>
                View links
              </Button>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

/**
 * One panel section: a title that collapses what is under it.
 *
 * Open by default, and the choice is REMEMBERED per section, because this is a
 * preference about somebody's screen rather than about the project. It
 * defaults open because a list nobody can see is a list nobody uses;
 * collapsing is for when a section has grown long enough to be in the way.
 *
 * `children` may be a FUNCTION of the open state rather than a node, for the
 * sections where collapsing means "show less of this" rather than "hide it" --
 * the data points being the case that wanted it.
 */
function Section({
  title,
  count,
  icon,
  collapses,
  className = "",
  bodyClassName = "",
  children,
}: {
  title: string;
  count?: number | string;
  icon?: React.ReactNode;
  /** Hide the body entirely when closed, rather than passing the state down. */
  collapses?: boolean;
  /** On the section itself -- for the one that absorbs the column's overflow. */
  className?: string;
  /** On the body -- where a max height and a scroller go. */
  bodyClassName?: string;
  children: React.ReactNode | ((open: boolean) => React.ReactNode);
}) {
  const [open, setOpen] = useState(true);
  const key = `howler.panel.${title}`;

  // Restored after mount, not in the initialiser: this is a client component
  // but Next still renders it on the server for the first HTML, where
  // `localStorage` does not exist -- reading it there is a hydration mismatch.
  useEffect(() => {
    try {
      setOpen(localStorage.getItem(key) !== "0");
    } catch {
      /* private window; the default stands */
    }
  }, [key]);

  function toggle() {
    setOpen((was) => {
      const next = !was;
      try {
        localStorage.setItem(key, next ? "1" : "0");
      } catch {
        /* nothing to do */
      }
      return next;
    });
  }

  const body =
    typeof children === "function" ? children(open) : collapses && !open ? null : children;

  return (
    <section className={`md-card md-card-outlined ${className}`}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        // `shrink-0` so the header keeps its height when the section is the
        // one absorbing a too-tall column -- otherwise flex squashes the title
        // instead of scrolling the body underneath it.
        className="md-state flex w-full shrink-0 items-center gap-2 rounded-[inherit] px-4 py-3 text-left"
      >
        <IconChevron className="h-4 w-4 shrink-0 opacity-60" open={open} />
        {icon}
        <h2 className="md-title-small min-w-0 flex-1 truncate">{title}</h2>
        {count !== undefined && (
          <span
            className="md-label-small shrink-0"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {count}
          </span>
        )}
      </button>
      {body !== null && (
        <div className={`px-4 pb-4 ${bodyClassName}`}>{body}</div>
      )}
    </section>
  );
}

function FieldRow({
  field,
  detail,
}: {
  field: BlueprintField;
  detail: boolean;
}) {
  return (
    <li className="px-1 py-1.5">
      <p className="md-body-medium flex items-baseline gap-2">
        <span className="min-w-0 flex-1 font-medium">{field.label}</span>
        <span
          className="md-label-small shrink-0"
          style={{
            color: field.required
              ? "var(--md-primary)"
              : "var(--md-on-surface-variant)",
          }}
        >
          {field.required ? "required" : "optional"}
        </span>
      </p>
      {detail && (
        <p
          className="md-body-small mt-0.5"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {field.description}
        </p>
      )}
    </li>
  );
}

/* ---------------------------------------------------------------- the links */

const STATUS: Record<HowlerInvite["status"], string> = {
  unopened: "Not opened yet",
  in_progress: "In progress",
  complete: "Complete",
  revoked: "Withdrawn",
};

function Links({
  invites,
  ready,
  onAdd,
  onRevoke,
  onSynthesise,
}: {
  invites: HowlerInvite[];
  ready: boolean;
  onAdd: (label: string) => void;
  onRevoke: (inviteId: string) => void;
  onSynthesise: () => void;
}) {
  const [label, setLabel] = useState("");
  const [adding, setAdding] = useState(false);

  if (!invites.length) {
    return (
      <Empty
        icon={<IconLink className="h-8 w-8" />}
        title="No links yet"
        body={
          ready
            ? "Synthesise settles the data points and gives you a link to send. Whoever opens it is interviewed straight away — no account, no sign-in."
            : "Once the conversation has produced some data points worth gathering, you will get a link to send to whoever you are interviewing."
        }
        action={
          ready ? (
            <Button onClick={onSynthesise}>
              <IconCheck />
              Synthesise now
            </Button>
          ) : null
        }
      />
    );
  }

  return (
    <div className="flex-1 space-y-4 px-6 pb-10">
      <p
        className="md-body-medium"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        One link per person. Each stays usable until that interview finishes, so
        a dropped call or a closed tab is nothing to worry about — reopening it
        carries on where they left off.
      </p>

      <ul className="space-y-3">
        {invites.map((invite) => (
          <LinkRow key={invite.id} invite={invite} onRevoke={onRevoke} />
        ))}
      </ul>

      {/* A SECOND link is for a SECOND person, which is why it asks for a
          name. One link per participant is what makes "who has answered" a
          question the list can answer at all. */}
      {adding ? (
        <form
          className="flex items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (!label.trim()) return;
            onAdd(label.trim());
            setLabel("");
            setAdding(false);
          }}
        >
          <TextArea
            label="Who is this one for?"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                (e.currentTarget.form as HTMLFormElement).requestSubmit();
              }
              if (e.key === "Escape") setAdding(false);
            }}
            surface="var(--md-surface)"
            className="flex-1"
            autoComplete="off"
          />
          <Button type="submit" disabled={!label.trim()}>
            Add
          </Button>
        </form>
      ) : (
        <Button variant="tonal" onClick={() => setAdding(true)}>
          <IconPlus />
          Link for someone else
        </Button>
      )}
    </div>
  );
}

function LinkRow({
  invite,
  onRevoke,
}: {
  invite: HowlerInvite;
  onRevoke: (inviteId: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  // Read at render, not at module scope: there is no `window` on the server,
  // and the origin is the one part of this link the page cannot be told.
  const url =
    typeof window === "undefined"
      ? ""
      : `${window.location.origin}/howl/${invite.token}`;

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard denied; the link is still selectable in the field */
    }
  }

  // Not usable again, for either of the two reasons a link stops working:
  // somebody withdrew it, or the interview behind it finished. `resolve`
  // refuses both, so neither has a URL worth copying and neither has anything
  // left to withdraw.
  const dead = invite.status === "revoked" || invite.status === "complete";

  return (
    <li className="md-card md-card-outlined p-4">
      <div className="flex items-baseline gap-3">
        <h3 className="md-title-small min-w-0 flex-1 truncate">
          {invite.label || "Unnamed"}
        </h3>
        <span
          className="md-label-medium shrink-0 rounded-[var(--md-shape-full)] px-2.5 py-1"
          style={{
            background:
              invite.status === "complete"
                ? "var(--md-secondary-container)"
                : "var(--md-surface-container-high)",
            color:
              invite.status === "complete"
                ? "var(--md-on-secondary-container)"
                : "var(--md-on-surface-variant)",
          }}
        >
          {STATUS[invite.status]}
        </span>
      </div>

      {!dead && (
        <div className="mt-3 flex items-center gap-2">
          <input
            readOnly
            value={url}
            onFocus={(e) => e.currentTarget.select()}
            aria-label={`Link for ${invite.label || "this participant"}`}
            className="md-body-small min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2 outline-none"
            style={{
              background: "var(--md-surface-container-high)",
              color: "var(--md-on-surface-variant)",
            }}
          />
          <Button
            variant="text"
            size="sm"
            onClick={() => void copy()}
            className="shrink-0"
          >
            {copied ? <IconCheck /> : <IconCopy />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      )}

      <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1">
        <Stat label="Opened" value={invite.opens === 0 ? "never" : `${invite.opens}×`} />
        {invite.last_opened_at && (
          <Stat
            label="Last opened"
            value={new Date(invite.last_opened_at).toLocaleString(undefined, {
              day: "numeric",
              month: "short",
              hour: "2-digit",
              minute: "2-digit",
            })}
          />
        )}
        {invite.result && (
          <Stat label="Exchanges" value={String(invite.result.turns)} />
        )}
        {invite.result && invite.result.missing.length > 0 && (
          <Stat
            label="Still missing"
            value={invite.result.missing.join(", ")}
          />
        )}
      </dl>

      {invite.result?.summary && (
        <p
          className="md-body-small mt-3 italic"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {invite.result.summary}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-1">
        {invite.result && (
          <a
            href={`/parley/howler?c=${invite.result.session_id}`}
            className="md-btn md-btn-text md-btn-sm md-state"
          >
            Open the result
          </a>
        )}
        {/* Only while there is something to withdraw. A finished interview
            has already closed its own link -- `resolve` turns it away with
            "this interview is already complete" -- so offering to withdraw it
            is a destructive-looking button that does nothing, on the one row
            where somebody might press it thinking it tidies the result away. */}
        {!dead && (
          <ConfirmButton
            label="Withdraw"
            icon={<IconTrash />}
            title="Withdraw this link?"
            // Names what is NOT lost, because that is the part people hesitate
            // over: withdrawing a link and throwing away what it gathered are
            // not the same decision.
            body="It stops working immediately. Anything already gathered through it is kept."
            confirmLabel="Withdraw"
            onConfirm={() => onRevoke(invite.id)}
          />
        )}
      </div>
    </li>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt
        className="md-label-small"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {label}
      </dt>
      <dd className="md-body-small">{value}</dd>
    </div>
  );
}

/* -------------------------------------------------------------- the results */

function Results({
  invites,
  onOpenLinks,
  onRefresh,
  refreshing,
}: {
  invites: HowlerInvite[];
  onOpenLinks: () => void;
  onRefresh: () => void;
  refreshing: boolean;
}) {
  const withResults = invites.filter((i) => i.result);

  if (!withResults.length) {
    return (
      <Empty
        icon={<IconProfile className="h-8 w-8" />}
        title="Nothing has come back yet"
        body={
          invites.length
            ? "Whatever your participants say lands here as it happens — you do not have to wait for them to finish. Check again when you expect someone to have spoken."
            : "Send someone a link and what they say will appear here, against the data points you asked for."
        }
        action={
          invites.length ? (
            <Button variant="tonal" onClick={onRefresh} disabled={refreshing}>
              {refreshing ? <IconSpinner /> : <IconRefresh />}
              Check again
            </Button>
          ) : (
            <Button onClick={onOpenLinks}>
              <IconLink />
              Go to links
            </Button>
          )
        }
      />
    );
  }

  return (
    <div className="flex-1 space-y-6 px-6 pb-10">
      {/* Results come from SOMEBODY ELSE'S browser, minutes or days later, so
          nothing on this page could know to update itself. A button beats
          reloading, which threw away whatever else was open to do the same
          thing. */}
      <div className="flex items-center justify-between gap-4">
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {withResults.length} of {invites.length} link
          {invites.length === 1 ? "" : "s"} have been opened.
        </p>
        <Button
          variant="text"
          size="sm"
          onClick={onRefresh}
          disabled={refreshing}
        >
          {refreshing ? <IconSpinner /> : <IconRefresh />}
          {refreshing ? "Checking" : "Refresh"}
        </Button>
      </div>

      {withResults.map((invite) => {
        const result = invite.result!;
        return (
          <section key={invite.id}>
            <div className="mb-2 flex items-baseline gap-3">
              <h2 className="md-title-medium min-w-0 flex-1 truncate">
                {invite.label || result.title}
              </h2>
              <span
                className="md-label-small shrink-0"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {result.turns} exchange{result.turns === 1 ? "" : "s"}
                {invite.status === "complete" ? " · finished" : ""}
              </span>
            </div>
            {/* THE SAME CARD the operator watches fill in live, given this
                conversation's OWN schema rather than the project's current
                one -- a link opened last week gathered last week's data
                points, and rendering it against today's would invent empty
                rows for questions nobody was ever asked. */}
            {/* WHERE THE EMOTION ANALYSIS LIVES, and what it says while there
                is none. Showing nothing made this look missing rather than
                pending -- and the honest answer is usually "there is no
                recording to analyse yet", which nobody could guess. */}
            {(result.voice || result.analysis) && (
              <section className="md-card md-card-outlined mb-3 p-5">
                <h3 className="md-title-small">Voice analysis</h3>
                {result.voice ? (
                  <>
                    <div className="mt-3">
                      <VoiceTimeline
                        turns={result.voice.turns}
                        baseline={result.voice.baseline}
                        moments={result.voice.moments}
                      />
                    </div>
                    {result.voice.moments.length > 0 && (
                      <>
                        <h4
                          className="md-label-medium mt-4"
                          style={{ color: "var(--md-on-surface-variant)" }}
                        >
                          Where it departed
                        </h4>
                        <ul className="mt-1 space-y-1">
                          {result.voice.moments.map((m, i) => (
                            <li key={i} className="md-body-small">
                              <span className="font-medium">Turn {m.turn + 1}</span>
                              <span style={{ color: "var(--md-on-surface-variant)" }}>
                                {" — "}
                                {m.dimension} {m.direction} than they were
                                elsewhere ({m.delta > 0 ? "+" : ""}
                                {m.delta.toFixed(2)})
                              </span>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    <p
                      className="md-body-small mt-3 border-t pt-3"
                      style={{
                        color: "var(--md-on-surface-variant)",
                        borderColor: "var(--md-outline-variant)",
                      }}
                    >
                      {result.voice.caveat}
                    </p>
                  </>
                ) : (
                  <p
                    className="md-body-small mt-2"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    {result.analysis?.status === "failed"
                      ? `Analysis failed: ${result.analysis.error || "unknown error"}`
                      : result.analysis?.status === "running"
                        ? "Analysing the audio…"
                        : result.analysis?.status === "done"
                          ? "Nothing to analyse — this interview has no audio recording."
                          : "Waiting. Interviews are queued as they finish, and analysed once recordings exist and the model is switched on."}
                  </p>
                )}
              </section>
            )}

            {result.affect && (
              <section className="md-card md-card-outlined mb-3 p-5">
                <h3 className="md-title-small">How they came across</h3>
                {result.affect.demeanour && (
                  <p className="md-body-medium mt-2">{result.affect.demeanour}</p>
                )}
                {result.affect.moments.length > 0 && (
                  <ul className="mt-3 space-y-1.5">
                    {result.affect.moments.map((moment) => (
                      <li
                        key={moment}
                        className="md-body-small flex gap-2"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        <span aria-hidden="true">·</span>
                        <span>{moment}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {/* SAID OUT LOUD, because this is the part of a profile most
                    likely to be read as fact. It is one model's impression of
                    a voice, it is not evidence, and anybody deciding about a
                    person on the strength of it should know that. */}
                <p
                  className="md-body-small mt-3 border-t pt-3"
                  style={{
                    color: "var(--md-on-surface-variant)",
                    borderColor: "var(--md-outline-variant)",
                  }}
                >
                  An impression of how the conversation sounded, not a finding
                  about the person. Tone reads differently across cultures, and
                  a bad line or a bad day sounds like a lot of things.
                </p>
              </section>
            )}

            <ProfileCard
              fields={result.fields.map((f) => ({
                name: f.name,
                required: f.required,
                kind: f.type,
              }))}
              profile={result.profile}
              missing={result.missing}
              complete={result.complete}
            />
            <ConversationPanel
              sessionId={result.session_id}
              turns={result.turns}
            />
          </section>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- the small */

function Empty({
  icon,
  title,
  body,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex-1 px-6">
      <div className="md-card md-card-filled mx-auto max-w-lg px-6 py-14 text-center">
        <span
          className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-[var(--md-shape-full)]"
          style={{
            background: "var(--md-primary-container)",
            color: "var(--md-on-primary-container)",
          }}
        >
          {icon}
        </span>
        <h2 className="md-title-large">{title}</h2>
        <p
          className="md-body-medium mx-auto mt-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {body}
        </p>
        {action && <div className="mt-6 flex justify-center">{action}</div>}
      </div>
    </div>
  );
}

function Banner({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="md-body-medium mb-4 rounded-[var(--md-shape-md)] px-4 py-3"
      style={{
        background: "var(--md-error-container)",
        color: "var(--md-on-error-container)",
      }}
    >
      {children}
    </p>
  );
}

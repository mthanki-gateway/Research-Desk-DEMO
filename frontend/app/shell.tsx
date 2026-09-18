"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Fragment, Suspense, useEffect, useState } from "react";
import { createSession } from "@/lib/api";
import { useApp } from "./providers";
import CommandPalette from "./command-palette";
import ChunkPanel from "./chunk-panel";
import { PROJECTS, SHARED_NAV, projectFor } from "./projects";
import ParleyNav from "./parleyNav";
// Rail widths live with the rail, so the margin reserved here and the rail
// itself can never disagree about how wide it is.
import { RAIL_WIDTH, RAIL_WIDTH_COLLAPSED } from "./chat/[id]/rail";
import {
  Button,
  Dialog,
  IconButton,
  LinearProgress,
  Ripplable,
  useAutoHideScroll,
} from "./md";
import {
  IconChat,
  IconClose,
  IconLab,
  IconLibrary,
  IconMenu,
  IconPlus,
  IconChevron,
  IconSearch,
  IconProfile,
  IconSignOut,
  IconSpinner,
} from "./icons";


/**
 * M3 navigation drawer.
 *
 * A light `surface` with a hairline right edge, which is what stock M3
 * specifies. It was navy for a while as brand identity; that lost to the flat
 * redesign, where a dark column was the loudest thing on screen and was loud
 * about navigation — the part of the app you look at least.
 *
 * The --md-nav-* roles are kept even though they now alias onto ordinary
 * surface tones, so the drawer can diverge again without touching every call
 * site here. Everything inside still follows M3 anatomy — 56px items,
 * pill-shaped active state, state layers and ripple.
 */
export default function Shell({ children }: { children: React.ReactNode }) {
  const {
    sessions,
    documents,
    ingesting,
    loading,
    refreshSessions,
    account,
    authReady,
    authEnabled: authOn,
    signOut,
    railActive,
    railCollapsed,
    railReady,
  } = useApp();
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [palette, setPalette] = useState(false);
  const [switcher, setSwitcher] = useState(false);
  const [creating, setCreating] = useState(false);
  // Which app owns the current route. Drives the drawer's whole contents,
  // so the nav cannot disagree with the page being shown.
  const project = projectFor(pathname);
  const isResearchDesk = project.id === "research-desk";

  // The nav href the current path belongs to: the longest one that matches.
  const activeHref = project.nav.reduce(
    (best, { href }) =>
      (pathname === href || pathname.startsWith(`${href}/`)) &&
      href.length > best.length
        ? href
        : best,
    "",
  );
  const scroller = useAutoHideScroll<HTMLElement>();

  // Routes that render WITHOUT the drawer and are never gated.
  //
  // /login and /auth/* because gating them would loop. /howl/* because it is
  // the magic link: somebody with no account, who must not be bounced to a
  // login they cannot complete, and who should see one interview rather than
  // a drawer full of apps that are not theirs.
  const isAuthRoute =
    pathname.startsWith("/login") ||
    pathname.startsWith("/auth") ||
    pathname.startsWith("/howl/");

  // Client-side gate. Middleware would avoid the brief flash, but it would
  // also need its own cookie plumbing; this is one condition and behaves
  // correctly on token expiry too, since `account` goes null.
  useEffect(() => {
    if (authOn && authReady && !account && !isAuthRoute) {
      router.replace("/login");
    }
  }, [authOn, authReady, account, isAuthRoute, router]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette((p) => !p);
      }
      if (e.key === "Escape") setPalette(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    setOpen(false);
    setSwitcher(false);
  }, [pathname]);

  async function startSession() {
    setCreating(true);
    try {
      const s = await createSession();
      // Navigate FIRST, then refresh the list.
      //
      // `await refreshSessions()` before the push meant the whole session list
      // was refetched before navigation began -- so the visible result of
      // clicking "New chat" was the sidebar gaining a row while the page
      // stayed put, and the new chat opened only after a second round trip.
      // The list is not needed to render the new chat, so it catches up in the
      // background.
      router.push(`/chat/${s.id}`);
      void refreshSessions();
    } finally {
      setCreating(false);
    }
  }

  const pending = documents.filter(
    (d) => d.status !== "ready" && d.status !== "failed",
  );

  if (isAuthRoute) return <>{children}</>;

  // Hold the frame until we know whether to gate. Rendering the app first and
  // redirecting after shows a flash of someone else's shell.
  if (authOn && !authReady) {
    return (
      <div className="grid min-h-screen place-items-center">
        <IconSpinner className="h-6 w-6 opacity-40" />
      </div>
    );
  }

  return (
    // h-screen + overflow-hidden, NOT min-h-screen: the shell is exactly one
    // viewport and never scrolls, so the scroll belongs to <main> below.
    <div className="flex h-screen overflow-hidden">
      {/* Top app bar, small — mobile only */}
      <header
        className="fixed inset-x-0 top-0 z-30 flex h-16 items-center gap-2 px-2 md:hidden"
        style={{
          background: "var(--md-nav-surface)",
          borderBottom: "1px solid var(--md-outline-variant)",
        }}
      >
        <IconButton
          onClick={() => setOpen((o) => !o)}
          aria-label="Open navigation"
          style={{ color: "var(--md-nav-on-surface)" }}
        >
          <IconMenu />
        </IconButton>
        <span
          className="md-title-large"
          style={{ color: "var(--md-nav-on-surface)" }}
        >
          {project.name}
        </span>
      </header>

      {open && (
        <div
          onClick={() => setOpen(false)}
          className="md-scrim z-30 md:hidden"
        />
      )}

      <aside
        // gap-3 rather than a margin on each child: the spacing BETWEEN the
        // groups is the thing doing the separating, so it belongs to the
        // container that owns the relationship, not to whichever child
        // happens to be above.
        className={`fixed inset-y-0 left-0 z-40 flex w-[20rem] flex-col gap-4 p-4 transition-transform md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{
          background: "var(--md-nav-surface)",
          // The drawer used to be a navy slab, so its edge was obvious. Now
          // that it is a light surface on a light page, a hairline is what
          // separates it -- same job, a fraction of the weight.
          borderRight: "1px solid var(--md-outline-variant)",
          transitionDuration: "var(--md-dur-medium)",
          transitionTimingFunction: "var(--md-ease-emphasized)",
        }}
      >
        <div className="flex items-center gap-1 px-2 pt-2">
          {/* The app's own name, and the switch to another one, in a single
              control. A separate "switch project" item lower down was the
              obvious alternative and is worse: the thing you click to change
              apps should be the thing showing which app you are in. */}
          <Ripplable
            as="button"
            type="button"
            onClick={() => setSwitcher(true)}
            // flex-1 so the control spans the drawer and the chevron sits at
            // the right edge. Sized to its content it floated mid-row with the
            // chevron hard against the text, which read as decoration rather
            // than as the affordance that opens the dialog.
            className="md-state flex min-w-0 flex-1 items-center gap-3 rounded-[var(--md-shape-md)] px-2 py-2 text-left"
            aria-haspopup="dialog"
            title="Switch app"
            // Colour set HERE rather than on the icon: icons are stroked with
            // `currentColor` and take only a className, so the parent is what
            // tints them. The text spans below set their own colours, so this
            // reaches the two icons alone.
            style={{ color: "var(--md-nav-on-surface-variant)" }}
          >
            {/* The APP'S OWN mark, not a generic grid. A grid icon says "there
                are several of these somewhere"; the app's own icon says which
                one you are in, which is the question this control answers
                every time you look at it and only occasionally by being
                clicked. */}
            <span
              className="grid h-9 w-9 shrink-0 place-items-center rounded-[var(--md-shape-md)]"
              style={{
                background: "var(--md-primary-container)",
                color: "var(--md-on-primary-container)",
              }}
            >
              <project.Icon className="h-5 w-5" />
            </span>
            <span className="min-w-0 flex-1">
              <span
                className="md-title-medium block truncate"
                style={{ color: "var(--md-nav-on-surface)" }}
              >
                {project.name}
              </span>
              <span
                className="md-body-small block"
                style={{ color: "var(--md-nav-on-surface-variant)" }}
              >
                Switch app
              </span>
            </span>
            {/* A chevron, because this OPENS something. The grid glyph that
                was here implied a grid of apps would appear in place, which is
                not what happens. */}
            <IconChevron className="h-4 w-4 shrink-0 rotate-90" />
          </Ripplable>
          <IconButton
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
            className="md:hidden"
            style={{ color: "var(--md-nav-on-surface-variant)" }}
          >
            <IconClose />
          </IconButton>
        </div>

        {/* Research Desk's action, not a universal one. A drawer that
            offers "New chat" while you are in the Model Lab is offering
            to leave the app you just opened. */}
        {isResearchDesk && (
          <Button
            onClick={() => void startSession()}
            disabled={creating}
            className="w-full"
          >
            {creating ? <IconSpinner /> : <IconPlus />}
            {creating ? "Creating" : "New chat"}
          </Button>
        )}

        {/* SCROLLS RATHER THAN PUSHING. Each collapsible list is capped on its
            own, but Parley has three of them -- Conversations, Profiles,
            Sessions -- and three capped lists still add up to more than the
            drawer is tall. The account block at the bottom was being shoved
            off the screen by the sum of them.

            `min-h-0` is what makes it work: a flex child refuses to shrink
            below its content without it, so the overflow never engages and
            the column grows instead. The spacer below sits at basis 0, so
            when there IS room it takes it and this stays its natural
            height. */}
        <nav className="md-nav-group scroll-thin min-h-0 space-y-1 overflow-y-auto pr-1">
          {project.nav.map(({ href, label, Icon }) => {
            // LONGEST MATCH WINS, not merely "starts with".
            //
            // Parley owns both /parley and /parley/interview, and a plain
            // prefix test lights BOTH rows on the interview page -- so the
            // drawer says you are in two places at once. Same rule as
            // projectFor(): the most specific route that matches is the one
            // you are in.
            const active = href === activeHref;
            return (
              /* A real <Link>, not a div with role="link" calling
                 router.push(). Two reasons, and the first is the bigger cause
                 of the perceived lag between Chat / Library / Lab:

                 1. PREFETCH. Next prefetches a <Link>'s route when it enters
                    the viewport (and on hover), so by the time you click, the
                    payload is usually already there. `router.push()` prefetches
                    nothing — every navigation started cold.
                 2. It is an anchor, so middle-click, ctrl-click, "open in new
                    tab" and screen-reader link navigation all work. A div with
                    role="link" only *claims* to be a link. */
              <Fragment key={href}>
              <Ripplable
                as={Link}
                href={href}
                prefetch
                className="md-nav-item"
                data-active={active}
              >
                {/* The icon gets its own container so it can carry the hover
                    treatment independently of the row. See .md-nav-icon. */}
                <span className="md-nav-icon">
                  <Icon className="h-6 w-6" />
                </span>
                {label}
              </Ripplable>

              {/* Parley's stored conversations hang off the nav item that
                  produces them -- Conversations under Speak, Profiles under
                  Interview. Rendered here rather than as one block after the
                  nav, because "which list belongs to which mode" is exactly
                  what the nesting is there to say.

                  Its own Suspense boundary: it reads the query string, which
                  opts the subtree into client rendering, and the drawer lives
                  OUTSIDE the page's boundary -- without one here, prerendering
                  the route fails entirely. */}
              {project.id === "parley" && (
                <Suspense fallback={null}>
                  <ParleyNav
                    pathname={pathname}
                    mode={
                      href === "/parley/interview"
                        ? "interview"
                        : href === "/parley/howler"
                          ? "howler"
                          : "speak"
                    }
                  />
                </Suspense>
              )}
              </Fragment>
            );
          })}
        </nav>

        {ingesting && (
          <div
            className="space-y-2 rounded-[var(--md-shape-md)] p-3"
            style={{ background: "var(--md-nav-surface-container)" }}
          >
            <p
              className="md-label-medium"
              style={{ color: "var(--md-nav-on-surface-variant)" }}
            >
              Indexing
            </p>
            {pending.map((d) => (
              <div key={d.id}>
                <p
                  className="md-body-small truncate"
                  style={{ color: "var(--md-nav-on-surface)" }}
                >
                  {d.filename}
                </p>
                <div className="mt-1.5">
                  <LinearProgress
                    value={d.n_chunks ? (d.n_embedded / d.n_chunks) * 100 : 4}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Conversations: their own block, and the only one that scrolls.
            The group is the flex child that takes the leftover height, with
            the scroll on the list INSIDE it -- so the label and the search row
            stay put while the titles move, and the tinted block does not
            visibly shrink as sessions are added.

            Research Desk only. The Model Lab has no sessions, and an empty
            "Sessions" panel there reads as a feature that is broken rather
            than one that does not apply. */}
        {isResearchDesk ? (
          <div className="md-nav-group flex min-h-0 flex-1 flex-col">
            <p
              className="md-label-medium mb-1 px-3 pt-1"
              style={{ color: "var(--md-nav-on-surface-variant)" }}
            >
              Sessions
            </p>
            <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
              {loading && sessions.length === 0 ? (
                <ul className="space-y-2 px-3 pt-2" aria-hidden>
                  {[80, 64, 72].map((w) => (
                    <li
                      key={w}
                      className="h-3 rounded"
                      style={{
                        width: `${w}%`,
                        // Also a leftover from the navy drawer: white-at-5% was
                        // invisible the moment the surface went light.
                        background: "var(--md-surface-container-high)",
                      }}
                    />
                  ))}
                </ul>
              ) : sessions.length === 0 ? (
                <p
                  className="md-body-small px-3"
                  style={{ color: "var(--md-nav-on-surface-variant)" }}
                >
                  No sessions yet
                </p>
              ) : (
                <ul>
                  {sessions.map((s) => {
                    const active = pathname === `/chat/${s.id}`;
                    return (
                      <li key={s.id}>
                        <Ripplable
                          as="div"
                          className="md-nav-item md-nav-item-dense"
                          data-active={active}
                          onClick={() => router.push(`/chat/${s.id}`)}
                          title={s.title}
                          role="link"
                          tabIndex={0}
                        >
                          <span className="min-w-0 flex-1 truncate">
                            {s.title}
                          </span>
                          <span className="md-label-small shrink-0 opacity-70">
                            {s.n_messages}
                          </span>
                        </Ripplable>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <Ripplable
              as="div"
              className="md-nav-item md-nav-item-dense mt-1"
              onClick={() => setPalette(true)}
              role="button"
              tabIndex={0}
            >
              <IconSearch className="h-5 w-5 shrink-0" />
              <span className="flex-1">Search sessions</span>
              {/* A token, not `rgba(255,255,255,0.10)`. That was tuned for the old
                navy drawer and became invisible the moment the drawer went
                light. */}
              <kbd
                className="md-label-small rounded px-1.5 py-0.5"
                style={{
                  background: "var(--md-surface-container-high)",
                  color: "var(--md-on-surface-variant)",
                }}
              >
                ⌘K
              </kbd>
            </Ripplable>
          </div>
        ) : (
          // Takes the leftover height so the account group stays pinned to
          // the bottom, exactly as it sits in Research Desk.
          <div className="min-h-0 flex-1" />
        )}

        {/* You: profile, then the account row. Last block in the column, which
            is where a settings-shaped destination is looked for -- and next to
            the identity it belongs to rather than beside Lab. */}
        <div className="md-nav-group space-y-1">
          {/* Shared across every app rather than owned by one, which is why it
              lives here with the account row instead of in `project.nav`. */}
          {SHARED_NAV.map(({ href, label, Icon }) => (
            <Ripplable
              key={href}
              as={Link}
              href={href}
              prefetch
              className="md-nav-item"
              data-active={pathname.startsWith(href)}
            >
              <span className="md-nav-icon">
                <Icon className="h-6 w-6" />
              </span>
              {label}
            </Ripplable>
          ))}

          {account && (
            <div
              className="flex items-center gap-3 rounded-[var(--md-shape-full)] px-3 py-2"
              style={{ background: "var(--md-nav-surface)" }}
            >
              <span
                className="md-label-large grid h-8 w-8 shrink-0 place-items-center rounded-[var(--md-shape-full)] uppercase"
                style={{
                  background: "var(--md-primary)",
                  color: "var(--md-on-primary)",
                }}
              >
                {(account.email ?? "?").charAt(0)}
              </span>
              <span
                className="md-body-small min-w-0 flex-1 truncate"
                style={{ color: "var(--md-nav-on-surface)" }}
                title={account.email ?? account.id}
              >
                {account.email ?? "Signed in"}
              </span>
              <button
                onClick={async () => {
                  await signOut();
                  router.replace("/login");
                }}
                title="Sign out"
                aria-label="Sign out"
                className="md-icon-btn md-icon-btn-sm md-state shrink-0"
                style={{ color: "var(--md-nav-on-surface-variant)" }}
              >
                <IconSignOut className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* THE scroll container for the app.
          The document no longer scrolls (the flex root above is exactly
          `h-screen overflow-hidden`), so the browser's full-height scrollbar
          is gone -- it used to run the whole right edge of the window, past
          the fixed rail, for content that only occupies the middle column.

          The right MARGIN matters as much as the scrolling. Reserving the
          rail's width here rather than as padding inside the page is what puts
          the scrollbar against the rail's edge; as padding, main still ran to
          the window edge underneath the fixed rail and its scrollbar went with
          it -- exactly the bar we are trying to move.

          Sticky descendants (the chat header and composer) now resolve against
          this element instead of the viewport, which is the same behaviour
          they had when the document scrolled. */}
      <main
        ref={scroller}
        className={`md-scroll min-w-0 flex-1 overflow-y-auto pt-16 md:ml-[20rem] md:pt-0 lg:mr-[var(--rail-pad)] ${
          railReady ? "transition-[margin]" : ""
        }`}
        style={
          {
            "--rail-pad": !railActive
              ? "0px"
              : railCollapsed
                ? RAIL_WIDTH_COLLAPSED
                : RAIL_WIDTH,
            transitionDuration: "var(--md-dur-medium)",
            transitionTimingFunction: "var(--md-ease-emphasized)",
          } as React.CSSProperties
        }
      >
        {children}
      </main>

      {palette && <CommandPalette onClose={() => setPalette(false)} />}

      <Dialog
        open={switcher}
        onClose={() => setSwitcher(false)}
        title="Switch app"
        body="These share one stack: the same login, database and API process. Only the routes differ."
        wide
        // The default wrapper is an action ROW -- right-aligned and horizontal,
        // which is right for "Cancel / Delete" and wrong for a list. Left as
        // the default, these cards were laid out as one flex item and shrank
        // to their content, which is what made this render as a narrow strip.
        contentClassName="mt-6 space-y-2"
      >
        {PROJECTS.map((p) => {
          const current = p.id === project.id;
          return (
            <Ripplable
              key={p.id}
              as={Link}
              href={p.home}
              prefetch
              className="md-card md-card-outlined md-card-interactive flex w-full items-center gap-4 p-4"
              // The current app is marked, not hidden. Removing it would make
              // the list change length depending on where you are, so the one
              // you want is never in the same place twice.
              data-active={current}
              aria-current={current ? "page" : undefined}
            >
              <span
                className="grid h-10 w-10 shrink-0 place-items-center rounded-[var(--md-shape-md)]"
                style={{
                  background: current
                    ? "var(--md-primary-container)"
                    : "var(--md-surface-container-high)",
                  color: current
                    ? "var(--md-on-primary-container)"
                    : "var(--md-on-surface-variant)",
                }}
              >
                <p.Icon className="h-5 w-5" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <span className="md-title-small">{p.name}</span>
                  {current && (
                    <span
                      className="md-label-small rounded-[var(--md-shape-full)] px-2 py-0.5"
                      style={{
                        background: "var(--md-secondary-container)",
                        color: "var(--md-on-secondary-container)",
                      }}
                    >
                      Current
                    </span>
                  )}
                </span>
                <span
                  className="md-body-small mt-0.5 block"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {p.blurb}
                </span>
              </span>
            </Ripplable>
          );
        })}
      </Dialog>

      <ChunkPanel />
    </div>
  );
}

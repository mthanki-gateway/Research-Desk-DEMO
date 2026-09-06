"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { createSession } from "@/lib/api";
import AccentPicker from "./accent-picker";
import { useApp } from "./providers";
import CommandPalette from "./command-palette";
import ChunkPanel from "./chunk-panel";
// Rail widths live with the rail, so the margin reserved here and the rail
// itself can never disagree about how wide it is.
import { RAIL_WIDTH, RAIL_WIDTH_COLLAPSED } from "./chat/[id]/rail";
import {
  Button,
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
  IconSearch,
  IconSignOut,
  IconSpinner,
} from "./icons";

const NAV = [
  { href: "/chat", label: "Chat", Icon: IconChat },
  { href: "/library", label: "Library", Icon: IconLibrary },
  { href: "/lab", label: "Lab", Icon: IconLab },
];

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
  const [creating, setCreating] = useState(false);
  const scroller = useAutoHideScroll<HTMLElement>();

  // /login and /auth/* must render without the drawer, and must never be
  // gated — gating them would loop.
  const isAuthRoute = pathname.startsWith("/login") || pathname.startsWith("/auth");

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

  useEffect(() => setOpen(false), [pathname]);

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
          Research Desk
        </span>
      </header>

      {open && (
        <div
          onClick={() => setOpen(false)}
          className="md-scrim z-30 md:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[20rem] flex-col p-3 transition-transform md:translate-x-0 ${
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
        <div className="mb-4 flex items-center justify-between px-4 pt-3">
          <Link href="/chat" className="block">
            <span
              className="md-title-large block"
              style={{ color: "var(--md-nav-on-surface)" }}
            >
              Research Desk
            </span>
            <span
              className="md-body-small block"
              style={{ color: "var(--md-nav-on-surface-variant)" }}
            >
              Document intelligence
            </span>
          </Link>
          <IconButton
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
            className="md:hidden"
            style={{ color: "var(--md-nav-on-surface-variant)" }}
          >
            <IconClose />
          </IconButton>
        </div>

        <Button
          onClick={() => void startSession()}
          disabled={creating}
          className="mb-4 w-full"
        >
          {creating ? <IconSpinner /> : <IconPlus />}
          {creating ? "Creating" : "New chat"}
        </Button>

        <nav className="mb-4 space-y-1">
          {NAV.map(({ href, label, Icon }) => {
            const active = pathname.startsWith(href);
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
              <Ripplable
                key={href}
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
            );
          })}
        </nav>

        {ingesting && (
          <div
            className="mb-4 space-y-2 rounded-[var(--md-shape-md)] p-3"
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
                    value={
                      d.n_chunks ? (d.n_embedded / d.n_chunks) * 100 : 4
                    }
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          <p
            className="md-label-medium mb-1 px-4"
            style={{ color: "var(--md-nav-on-surface-variant)" }}
          >
            Sessions
          </p>
          {loading && sessions.length === 0 ? (
            <ul className="space-y-2 px-4 pt-2" aria-hidden>
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
              className="md-body-small px-4"
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
                      <span className="min-w-0 flex-1 truncate">{s.title}</span>
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
          className="md-nav-item md-nav-item-dense mt-2"
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

        <AccentPicker />

        {account && (
          <div
            className="mt-1 flex items-center gap-3 rounded-[var(--md-shape-full)] px-3 py-2"
            style={{ background: "var(--md-nav-surface-container)" }}
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
      <ChunkPanel />
    </div>
  );
}

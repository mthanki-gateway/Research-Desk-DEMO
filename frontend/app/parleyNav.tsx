"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  type Mode,
  deleteParleyConversation,
  renameParleyConversation,
} from "@/lib/api";
import { useApp } from "./providers";
import { Ripplable } from "./md";
import { IconChevron } from "./icons";

/**
 * Parley's conversations, in the drawer beneath "Speak".
 *
 * A SEPARATE COMPONENT rather than more branching inside the shell. The shell
 * already carries one app-specific block for the Research Desk's sessions, and
 * a second one inline would make the drawer a switch statement over apps. This
 * keeps the shell's job to "render the project's nav, plus whatever that
 * project contributes".
 *
 * COLLAPSIBLE, and collapsed is not the default. A list you cannot see is a
 * list nobody uses; the toggle is for when it grows long enough to be in the
 * way. The choice is remembered per browser, because it is a preference about
 * this person's screen rather than about the data.
 */

/** What each mode calls its stored conversations, and where they live. */
const SECTION: Record<Mode, { label: string; empty: string; href: string }> = {
  speak: {
    label: "Conversations",
    empty: "Nothing spoken yet",
    href: "/parley",
  },
  interview: {
    // "Profiles", because that is what an interview PRODUCES. Calling them
    // conversations would describe the mechanism rather than the point.
    label: "Profiles",
    empty: "No interviews yet",
    href: "/parley/interview",
  },
};

export default function ParleyNav({
  pathname,
  mode,
}: {
  pathname: string;
  mode: Mode;
}) {
  const router = useRouter();
  // Which conversation the page is showing, so the row can be highlighted.
  // It lives in the URL rather than in shared state: the drawer and the page
  // are separate components, and a query parameter is the one thing they both
  // already see.
  const current = useSearchParams().get("c");
  const { parleyConversations, refreshParleyConversations } = useApp();
  const items = parleyConversations[mode];
  const section = SECTION[mode];
  const [open, setOpen] = useState(true);
  /** Which row's menu is showing. One at a time. */
  const [menu, setMenu] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const renameBox = useRef<HTMLInputElement | null>(null);

  // Restored after mount, not in the initialiser: this is a client component
  // but Next still renders it on the server for the first HTML, where
  // `localStorage` does not exist — reading it there is a hydration mismatch.
  useEffect(() => {
    try {
      setOpen(localStorage.getItem(`parley.nav.${mode}`) !== "0");
    } catch {
      /* private window; the default stands */
    }
  }, [mode]);

  function toggle() {
    setOpen((was) => {
      const next = !was;
      try {
        localStorage.setItem(`parley.nav.${mode}`, next ? "1" : "0");
      } catch {
        /* nothing to do */
      }
      return next;
    });
  }

  // Any click elsewhere closes the menu. Without this it stays open behind the
  // next thing the user does, which reads as the app having lost track.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [menu]);

  useEffect(() => {
    if (renaming) renameBox.current?.focus();
  }, [renaming]);

  async function commitRename(id: string) {
    const title = draft.trim();
    setRenaming(null);
    if (!title) return;
    try {
      await renameParleyConversation(id, title);
      await refreshParleyConversations(mode);
    } catch {
      /* the old title stands */
    }
  }

  async function remove(id: string) {
    setMenu(null);
    try {
      await deleteParleyConversation(id);
      await refreshParleyConversations(mode);
      // Leaving the page pointed at a conversation that no longer exists would
      // show its turns until the next reload.
      if (current === id) router.replace(section.href);
    } catch {
      /* it stays in the list */
    }
  }

  return (
    /* INDENTED, with a rule down the left edge.
       These conversations belong to "Speak" -- they are the thing that nav
       item produces -- and as a flat sibling group the relationship was not
       visible at all. The indent and the rule say "inside this" without
       needing a second label to explain it. */
    <div
      /* `mb-3` separates one mode's block from the next nav item. Without it
         Speak's conversations sat flush against Interview, and the indent
         alone was not enough to say where one group ended -- the last
         conversation read as though it belonged to the item below it. */
      className="ml-5 mb-3 mt-0.5 border-l pl-1"
      style={{ borderColor: "var(--md-nav-outline, rgba(0,0,0,0.10))" }}
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="md-label-medium flex w-full items-center gap-1 rounded-[var(--md-shape-sm)] px-2 py-1"
        style={{ color: "var(--md-nav-on-surface-variant)" }}
      >
        <IconChevron className="h-3.5 w-3.5" open={open} />
        <span className="flex-1 text-left">{section.label}</span>
        {items.length > 0 && (
          <span className="md-label-small opacity-70">{items.length}</span>
        )}
      </button>

      {open &&
        (items.length === 0 ? (
          <p
            className="md-body-small px-2 pb-1"
            style={{ color: "var(--md-nav-on-surface-variant)" }}
          >
            {section.empty}
          </p>
        ) : (
          <ul className="scroll-thin max-h-72 overflow-y-auto">
            {items.map((c) => {
              const active = pathname === section.href && current === c.id;
              return (
                <li key={c.id} className="relative">
                  {renaming === c.id ? (
                    <input
                      ref={renameBox}
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onBlur={() => void commitRename(c.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") void commitRename(c.id);
                        if (e.key === "Escape") setRenaming(null);
                      }}
                      className="md-body-small mx-2 my-0.5 w-[calc(100%-1rem)] rounded-[var(--md-shape-sm)] px-2 py-1.5"
                      style={{
                        background: "var(--md-surface)",
                        color: "var(--md-on-surface)",
                        border: "1px solid var(--md-outline)",
                      }}
                    />
                  ) : (
                    <Ripplable
                      as="div"
                      className="md-nav-item md-nav-item-dense"
                      data-active={active}
                      onClick={() => router.push(`${section.href}?c=${c.id}`)}
                      title={c.title}
                      role="link"
                      tabIndex={0}
                    >
                      <span className="min-w-0 flex-1 truncate">{c.title}</span>
                      <span
                        role="button"
                        tabIndex={0}
                        aria-label={`Options for ${c.title}`}
                        onClick={(e) => {
                          // Or the row navigates out from under the menu.
                          e.stopPropagation();
                          setMenu(menu === c.id ? null : c.id);
                        }}
                        className="shrink-0 px-1 opacity-60 hover:opacity-100"
                      >
                        ⋯
                      </span>
                    </Ripplable>
                  )}

                  {menu === c.id && (
                    <div
                      onClick={(e) => e.stopPropagation()}
                      role="menu"
                      className="absolute right-2 top-8 z-20 min-w-[9rem] overflow-hidden rounded-[var(--md-shape-md)] py-1"
                      style={{
                        background: "var(--md-surface-container-high)",
                        color: "var(--md-on-surface)",
                        boxShadow: "var(--md-elev-2)",
                      }}
                    >
                      <button
                        type="button"
                        role="menuitem"
                        className="md-body-small w-full px-3 py-2 text-left"
                        onClick={() => {
                          setDraft(c.title);
                          setRenaming(c.id);
                          setMenu(null);
                        }}
                      >
                        Rename
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        className="md-body-small w-full px-3 py-2 text-left"
                        style={{ color: "var(--md-error)" }}
                        onClick={() => void remove(c.id)}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        ))}
    </div>
  );
}

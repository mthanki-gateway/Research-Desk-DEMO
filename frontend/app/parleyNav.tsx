"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter, useSearchParams } from "next/navigation";
import {
  type Mode,
  deleteParleyConversation,
  renameParleyConversation,
} from "@/lib/api";
import { useApp } from "./providers";
import { Ripplable } from "./md";
import { IconChevron, IconEdit, IconMore, IconTrash } from "./icons";

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
  howler: {
    label: "Sessions",
    empty: "No sessions yet",
    href: "/parley/howler",
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
  /**
   * Which row's menu is showing, and WHERE.
   *
   * The coordinates are carried because the menu is positioned `fixed`: an
   * absolutely positioned one is clipped by the nearest scrolling ancestor,
   * and this list is exactly that -- the menu appeared cut in half, behind the
   * row beneath it.
   */
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(
    null,
  );
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

  // Any click elsewhere closes it, and so does scrolling or resizing --
  // a `fixed` menu does not follow the row it belongs to, so it would sit over
  // an unrelated one. Escape closes it because every menu should.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    window.addEventListener("click", close);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
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
          // Capped so ONE long list cannot crowd out the other two. The
          // drawer's nav scrolls as well, which handles the sum of them; this
          // is about no single app's history taking the whole column.
          <ul className="scroll-thin max-h-56 overflow-y-auto">
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
                      aria-label="Conversation name"
                      className="md-body-small mx-2 my-0.5 w-[calc(100%-1rem)] rounded-[var(--md-shape-sm)] px-2 py-1.5 outline-none"
                      style={{
                        background: "var(--md-surface-container-high)",
                        color: "var(--md-on-surface)",
                        // A focus ring rather than a border: a border changes
                        // the row's height when it appears, which makes the
                        // list jump as you start renaming.
                        boxShadow: "inset 0 0 0 2px var(--md-primary)",
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
                        aria-haspopup="menu"
                        aria-expanded={menu?.id === c.id}
                        data-open={menu?.id === c.id}
                        onClick={(e) => {
                          // Or the row navigates out from under the menu.
                          e.stopPropagation();
                          if (menu?.id === c.id) return setMenu(null);
                          // Anchored to the BUTTON's position on screen, taken
                          // at open time, because `fixed` coordinates are
                          // viewport coordinates.
                          const r = (
                            e.currentTarget as HTMLElement
                          ).getBoundingClientRect();
                          setMenu({ id: c.id, x: r.right, y: r.bottom + 4 });
                        }}
                        className="md-icon-affordance shrink-0"
                      >
                        <IconMore className="h-4 w-4" />
                      </span>
                    </Ripplable>
                  )}

                </li>
              );
            })}
          </ul>
        ))}

      {/* ONE menu for the whole list, and PORTALLED TO THE BODY.
          Rendered per-row it was clipped by the list's own overflow. Moving it
          out of the <ul> fixed that, but only until the drawer's <nav> became
          scrollable too -- `position: fixed` is still clipped by a scrolling
          ancestor when something between it and the viewport establishes a
          containing block, and the drawer does exactly that with its slide-in
          `transition-transform`.
          A portal takes it out of that subtree entirely, so no ancestor can
          clip it and no future one can either. The coordinates already assume
          the viewport, which is what they now actually get. */}
      {menu && typeof document !== "undefined" && createPortal(
        <div
          role="menu"
          className="md-menu"
          style={{ left: menu.x, top: menu.y, transform: "translateX(-100%)" }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            role="menuitem"
            className="md-body-small md-menu-item"
            onClick={() => {
              const target = parleyConversations[mode].find(
                (c) => c.id === menu.id,
              );
              setDraft(target?.title ?? "");
              setRenaming(menu.id);
              setMenu(null);
            }}
          >
            <IconEdit className="h-3.5 w-3.5" />
            Rename
          </button>
          <button
            type="button"
            role="menuitem"
            className="md-body-small md-menu-item md-menu-item-danger"
            onClick={() => void remove(menu.id)}
          >
            <IconTrash className="h-3.5 w-3.5" />
            Delete
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}

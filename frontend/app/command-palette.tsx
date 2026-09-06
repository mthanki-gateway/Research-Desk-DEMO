"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "./providers";
import { Ripplable } from "./md";
import { IconChat, IconSearch } from "./icons";

/**
 * Cmd/Ctrl+K session switcher — an M3 **full-screen search view**: a docked
 * search bar with results below, elevation 3, on a scrim.
 *
 * Type to filter, arrows to move, Enter to open.
 */
export default function CommandPalette({ onClose }: { onClose: () => void }) {
  const { sessions } = useApp();
  const router = useRouter();
  const [query, setQuery] = useState("");
  // -1 = nothing highlighted. Starting at 0 pre-selected the first row, which
  // reads as a choice already made rather than as a list to choose from.
  // Highlight appears only once the user arrows or hovers.
  const [cursor, setCursor] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => inputRef.current?.focus(), []);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? sessions.filter((s) => s.title.toLowerCase().includes(q))
      : sessions;
    return list.slice(0, 8);
  }, [sessions, query]);

  // Reset when results change, or the cursor can point past the end.
  useEffect(() => setCursor(-1), [query]);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      // Back past the top clears the highlight rather than sticking on row 0.
      setCursor((c) => Math.max(c - 1, -1));
    } else if (e.key === "Enter") {
      // Enter with nothing highlighted opens the top match, which is what a
      // search box should do — the highlight is a visual affordance, not a
      // precondition.
      const target = cursor >= 0 ? matches[cursor] : matches[0];
      if (target) open(target.id);
    }
  }

  function open(id: string) {
    router.push(`/chat/${id}`);
    onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]"
      onClick={onClose}
    >
      <div className="md-scrim" />

      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-xl overflow-hidden"
        style={{
          background: "var(--md-surface)",
          color: "var(--md-on-surface)",
          borderRadius: "var(--md-shape-xl)",
          // See chunk-panel: overlays are the one place depth survives.
          border: "1px solid var(--md-outline-variant)",
          boxShadow: "var(--md-elev-2)",
        }}
      >
        {/* Docked search bar: 56px, no outline, icon leading. */}
        <div className="flex h-14 items-center gap-3 px-4">
          <IconSearch
            className="h-5 w-5 shrink-0"
            // affordance inside the field
          />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search sessions"
            className="md-body-large w-full bg-transparent outline-none"
            style={{ color: "var(--md-on-surface)" }}
          />
          <kbd
            className="md-label-small shrink-0 rounded-[var(--md-shape-xs)] px-1.5 py-0.5"
            style={{
              background: "var(--md-surface-container-highest)",
              color: "var(--md-on-surface-variant)",
            }}
          >
            esc
          </kbd>
        </div>

        <hr className="md-divider" />

        {matches.length === 0 ? (
          <p
            className="md-body-medium px-4 py-5"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            No matching sessions.
          </p>
        ) : (
          <ul className="max-h-[50vh] overflow-y-auto py-2">
            {matches.map((s, i) => (
              <li key={s.id}>
                <Ripplable
                  as="div"
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => open(s.id)}
                  role="option"
                  aria-selected={i === cursor}
                  tabIndex={0}
                  className="flex h-14 cursor-pointer items-center gap-4 px-4"
                  style={{
                    background:
                      i === cursor
                        ? "var(--md-secondary-container)"
                        : "transparent",
                    color:
                      i === cursor
                        ? "var(--md-on-secondary-container)"
                        : "var(--md-on-surface)",
                  }}
                >
                  <IconChat className="h-5 w-5 shrink-0 opacity-70" />
                  <span className="md-body-large min-w-0 flex-1 truncate">
                    {s.title}
                  </span>
                  <span
                    className="md-label-medium shrink-0 tabular-nums"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    {s.n_messages}
                  </span>
                </Ripplable>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

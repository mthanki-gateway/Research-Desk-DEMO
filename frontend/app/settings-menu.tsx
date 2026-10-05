"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Ripplable } from "./md";

/**
 * The Settings entry at the bottom of the drawer, and its menu.
 *
 * A Material 3 menu that opens UPWARDS, because it sits at the foot of the
 * column and a downward menu would open off the screen. Each item carries a
 * colourful tile that morphs into a shape on hover -- the same 48-point
 * shapes as the icon buttons (see globals.css), so it is one motion language,
 * not a one-off.
 */

const ITEMS = [
  {
    href: "/settings#keys",
    label: "API keys",
    hint: "Add your own provider keys",
    Glyph: KeyGlyph,
  },
  {
    href: "/settings#features",
    label: "Feature access",
    hint: "What your keys unlock",
    Glyph: GridGlyph,
  },
  {
    href: "/profile",
    label: "Profile",
    hint: "Your preferences and account",
    Glyph: PersonGlyph,
  },
];

export default function SettingsMenu() {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement | null>(null);
  const pathname = usePathname();

  // Close on navigation, outside click and Escape -- a menu that has to be
  // dismissed by clicking its own button again is a trap.
  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={root} className="relative">
      {open && (
        <div
          role="menu"
          aria-label="Settings"
          className="md-settings-menu absolute bottom-full left-0 right-0 z-50 mb-2 overflow-hidden rounded-[var(--md-shape-lg)] p-1.5"
          style={{
            background: "var(--md-surface-container)",
            color: "var(--md-on-surface)",
            boxShadow: "0 4px 16px rgba(0,0,0,0.18), 0 1px 3px rgba(0,0,0,0.12)",
          }}
        >
          {ITEMS.map(({ href, label, hint, Glyph }, i) => (
            <Ripplable
              key={href}
              as={Link}
              href={href}
              role="menuitem"
              className="group flex items-center gap-3 rounded-[var(--md-shape-md)] px-2.5 py-2"
              onClick={() => setOpen(false)}
            >
              <span className={`md-morph-tile md-morph-${i % 4}`} aria-hidden>
                <Glyph />
              </span>
              <span className="min-w-0 flex-1">
                <span className="md-label-large block">{label}</span>
                <span className="md-body-small block truncate" style={{ color: "var(--md-on-surface-variant)" }}>
                  {hint}
                </span>
              </span>
            </Ripplable>
          ))}
        </div>
      )}
      <Ripplable
        className="md-nav-item w-full"
        data-active={pathname.startsWith("/settings")}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="md-nav-icon">
          <GearGlyph />
        </span>
        Settings
      </Ripplable>
    </div>
  );
}

const svg = "h-5 w-5";
function GearGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6" aria-hidden>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
    </svg>
  );
}
function KeyGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className={svg}>
      <circle cx="8" cy="15" r="4" />
      <path d="M11 12l9-9M17 6l3 3M15 8l2 2" />
    </svg>
  );
}
function GridGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className={svg}>
      <rect x="4" y="4" width="7" height="7" rx="1.5" />
      <rect x="13" y="4" width="7" height="7" rx="1.5" />
      <rect x="4" y="13" width="7" height="7" rx="1.5" />
      <path d="M16.5 13.5v6M13.5 16.5h6" />
    </svg>
  );
}
function PersonGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className={svg}>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0 1 16 0" />
    </svg>
  );
}

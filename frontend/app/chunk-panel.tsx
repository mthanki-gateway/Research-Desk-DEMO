"use client";

import { useEffect } from "react";
import { useApp } from "./providers";
import { IconButton } from "./md";
import { IconClose, IconQuote } from "./icons";

/**
 * Citation slide-over, used OUTSIDE the chat view (the Lab).
 *
 * The chat view has its own rail with a Source tab, so a modal there would
 * cover the answer you are checking the citation against — hence `railActive`.
 */
export default function ChunkPanel() {
  const { openChunk, closeChunk, railActive } = useApp();

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") closeChunk();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeChunk]);

  if (!openChunk || railActive) return null;
  const c = openChunk;

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={closeChunk}>
      <div className="md-scrim" />
      <aside
        onClick={(e) => e.stopPropagation()}
        className="relative flex w-full max-w-md flex-col"
        style={{
          background: "var(--md-surface)",
          color: "var(--md-on-surface)",
          // Overlays keep a shadow -- they genuinely float above the page, and
          // now that everything else is flat that shadow means something again.
          // Softened to elev-2 with a hairline, so it reads as a raised sheet
          // rather than a slab.
          borderLeft: "1px solid var(--md-outline-variant)",
          boxShadow: "var(--md-elev-2)",
        }}
      >
        <header className="flex items-start justify-between gap-4 p-4">
          <div className="flex min-w-0 gap-3">
            <span
              className="grid h-10 w-10 shrink-0 place-items-center rounded-[var(--md-shape-full)]"
              style={{
                background: "var(--md-primary-container)",
                color: "var(--md-on-primary-container)",
              }}
            >
              <IconQuote className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <p className="md-title-small truncate">
                {c.heading
                  ? c.heading.replace(/^#+\s*/, "")
                  : `Chunk ${c.chunk_index}`}
              </p>
              <p
                className="md-body-small mt-0.5 truncate"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {c.filename}
                {c.page !== null && ` · page ${c.page}`}
              </p>
            </div>
          </div>
          <IconButton onClick={closeChunk} aria-label="Close">
            <IconClose />
          </IconButton>
        </header>

        <div className="flex gap-2 px-4 pb-3">
          <span className="md-badge">chunk {c.chunk_index}</span>
          <span className="md-badge">{c.n_chars} chars</span>
        </div>

        <hr className="md-divider" />

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-4">
          <p className="md-body-large whitespace-pre-wrap">{c.text}</p>
        </div>

        <hr className="md-divider" />

        <footer
          className="md-body-small p-4"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Verbatim text sent to the model, including the heading prefix added
          during ingestion.
        </footer>
      </aside>
    </div>
  );
}

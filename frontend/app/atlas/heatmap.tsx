"use client";

import { useEffect, useRef, useState } from "react";
import type { AtlasPoint } from "@/lib/api";

/**
 * Every chunk against every other, as a grid.
 *
 * WHAT IT IS FOR. Three patterns are worth spotting and none of them are
 * visible anywhere else in the app:
 *
 *   bright block on the diagonal   a document whose chunks are all alike --
 *                                  possibly over-chunked
 *   bright square OFF the diagonal two documents covering the same ground, so
 *                                  retrieval spends slots on duplicates
 *   a dark row                     a chunk similar to nothing, which is
 *                                  usually a chunking failure rather than an
 *                                  unusual topic
 *
 * CANVAS, NOT DIVS. At 39 chunks a CSS grid would be 1,521 elements and fine;
 * at the 2,000-chunk cap it is four million, which no browser will lay out.
 * One canvas draws either in the same code.
 */

/**
 * Where the colour ramp starts.
 *
 * NOT ZERO. Embeddings of text from one corpus are all somewhat alike --
 * nothing here falls below about 0.3 -- so a 0..1 ramp renders the entire
 * matrix in the top third of its range and every cell looks identical. The
 * floor stretches the contrast across the part that actually varies.
 */
const FLOOR = 0.45;

export default function Heatmap({
  points,
  similarity,
  selected,
  onSelect,
}: {
  points: AtlasPoint[];
  similarity: number[][];
  selected: number | null;
  onSelect: (index: number | null) => void;
}) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const n = points.length;
  /**
   * The cell under the cursor.
   *
   * Added because the grid was unreadable without it: a wall of squares says
   * nothing about WHICH chunks any square compares, so the pattern was
   * visible and unusable. Naming both chunks and the score turns it from a
   * texture into a measurement.
   */
  const [at, setAt] = useState<{ i: number; j: number; x: number; y: number } | null>(
    null,
  );

  useEffect(() => {
    const el = canvas.current;
    if (!el || n === 0) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;

    const size = el.clientWidth;
    const dpr = Math.min(window.devicePixelRatio, 2);
    el.width = size * dpr;
    el.height = size * dpr;
    ctx.scale(dpr, dpr);
    const cell = size / n;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = similarity[i]?.[j] ?? 0;
        const t = Math.max(0, Math.min(1, (v - FLOOR) / (1 - FLOOR)));
        // Single hue, varying lightness. A rainbow ramp would imply
        // categories where there is only magnitude, and is unreadable to the
        // ~8% of men with a colour deficiency.
        const light = 97 - t * 72;
        ctx.fillStyle = `hsl(258 55% ${light}%)`;
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }

    // Document boundaries. Without them the blocks are visible but unlabelled,
    // and "which file is that bright square" is the first question anyone asks.
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.lineWidth = 1;
    for (let i = 1; i < n; i++) {
      if (points[i].filename === points[i - 1].filename) continue;
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(size, i * cell);
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, size);
      ctx.stroke();
    }

    if (selected !== null) {
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.lineWidth = 2;
      ctx.strokeRect(0, selected * cell, size, cell);
      ctx.strokeRect(selected * cell, 0, cell, size);
    }
  }, [points, similarity, selected, n]);

  const a = at ? points[at.i] : null;
  const b = at ? points[at.j] : null;
  const score = at ? similarity[at.i]?.[at.j] : null;

  const cellFrom = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const i = Math.floor((y / rect.height) * n);
    const j = Math.floor((x / rect.width) * n);
    if (i < 0 || i >= n || j < 0 || j >= n) return null;
    return { i, j, x, y };
  };

  return (
    <div className="relative">
      <canvas
        ref={canvas}
        className="aspect-square w-full cursor-crosshair rounded-[var(--md-shape-md)]"
        onMouseMove={(e) => setAt(cellFrom(e))}
        onMouseLeave={() => setAt(null)}
        onClick={(e) => {
          const cell = cellFrom(e);
          // The ROW is the chunk. Selecting on the row keeps this consistent
          // with the scatter, where a click selects one chunk rather than a
          // pair.
          onSelect(cell ? cell.i : null);
        }}
        aria-label="Chunk similarity matrix. Click a row to select that chunk."
      />

      {at && a && b && score !== null && (
        <div
          className="pointer-events-none absolute z-10 w-[17rem] rounded-[var(--md-shape-md)] p-2.5"
          style={{
            left: Math.min(at.x + 14, (canvas.current?.clientWidth ?? 0) - 280),
            top: Math.min(at.y + 14, (canvas.current?.clientHeight ?? 0) - 120),
            background: "var(--md-inverse-surface)",
            color: "var(--md-inverse-on-surface)",
          }}
        >
          <p className="md-label-large">{score.toFixed(3)}</p>
          <p className="md-body-small mt-1 break-words">
            {a.filename} · chunk {a.chunk_index}
          </p>
          <p className="md-body-small break-words" style={{ opacity: 0.75 }}>
            vs {b.filename} · chunk {b.chunk_index}
          </p>
        </div>
      )}
    </div>
  );
}

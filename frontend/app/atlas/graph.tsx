"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildGraph,
  getGraph,
  type GraphEdge,
  type GraphNode,
  type KnowledgeGraph,
} from "@/lib/api";
import { useApp } from "../providers";
import { Button } from "../md";
import { IconSpinner } from "../icons";

/**
 * Who and what the corpus mentions, and what it SAYS connects them.
 *
 * The scatter shows passages that are near each other in meaning; this shows
 * claims. Every edge is a sentence in a document, and clicking it opens that
 * passage -- an edge that cannot be checked would be the model's opinion
 * dressed as the corpus's.
 *
 * A 2D force layout on a canvas, written here rather than imported: it is a
 * hundred lines, it needs no image rebuild, and at the 250-node cap the
 * simulation is a few milliseconds a frame. It settles and then STOPS -- a
 * layout that keeps jiggling is the most distracting thing a graph can do.
 */

const TYPE_COLOURS: Record<string, string> = {
  person: "#f2a7c3",
  organization: "#8fb8ff",
  place: "#9be0b4",
  product: "#ffd27a",
  concept: "#c8b6ff",
  event: "#ffab91",
  other: "#b8bfd6",
};

type Sim = GraphNode & { x: number; y: number; vx: number; vy: number };

export default function GraphView() {
  const [data, setData] = useState<KnowledgeGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queuing, setQueuing] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const sim = useRef<Sim[]>([]);
  const view = useRef({ scale: 1, x: 0, y: 0 });
  const { showChunk } = useApp();
  // Read by the draw loop. Refs, not effect dependencies: a dependency would
  // restart the simulation on every hover and throw the settled layout away.
  const focusRef = useRef<string | null>(null);
  focusRef.current = selected ?? hovered;

  const load = useCallback(async () => {
    try {
      setData(await getGraph());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the graph");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll only while extraction is actually running, and slowly: a job is a
  // few seconds per document, and nothing here needs to be live.
  useEffect(() => {
    if (!data?.pending) return;
    const t = setTimeout(() => void load(), 4000);
    return () => clearTimeout(t);
  }, [data, load]);

  const byId = useMemo(() => {
    const m = new Map<string, number>();
    data?.nodes.forEach((n, i) => m.set(n.id, i));
    return m;
  }, [data]);

  // ---- layout ---------------------------------------------------------
  useEffect(() => {
    const el = canvas.current;
    if (!el || !data || data.nodes.length === 0) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;

    // Deterministic start on a spiral, so the same graph settles into the
    // same picture on every visit instead of a new random one.
    sim.current = data.nodes.map((n, i) => {
      const a = i * 2.39996;
      const r = 12 * Math.sqrt(i + 1);
      return { ...n, x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0 };
    });
    const nodes = sim.current;
    const links = data.edges
      .map((e) => [byId.get(e.source), byId.get(e.target)] as const)
      .filter((l): l is readonly [number, number] => l[0] !== undefined && l[1] !== undefined);

    let alpha = 1;
    let frame = 0;
    const dpr = Math.min(window.devicePixelRatio, 2);

    const stars = Array.from({ length: 160 }, (_, i) => {
      const h = Math.sin(i * 12.9898) * 43758.5453;
      const f = h - Math.floor(h);
      const g = Math.sin(i * 78.233) * 12345.678;
      return { x: f, y: g - Math.floor(g), b: 0.15 + (i % 7) / 30 };
    });

    const step = () => {
      // Repulsion between every pair: O(n^2), fine at 250 nodes.
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy || 0.01;
          const f = (900 / d2) * alpha;
          dx *= f; dy *= f;
          a.vx += dx; a.vy += dy; b.vx -= dx; b.vy -= dy;
        }
      }
      for (const [i, j] of links) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = ((d - 70) / d) * 0.04 * alpha;
        a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
      }
      for (const n of nodes) {
        // Gentle pull to the centre so disconnected islands do not drift off.
        n.vx -= n.x * 0.004 * alpha;
        n.vy -= n.y * 0.004 * alpha;
        n.x += n.vx; n.y += n.vy;
        n.vx *= 0.6; n.vy *= 0.6;
      }
      alpha *= 0.985;
    };

    const draw = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (el.width !== w * dpr) { el.width = w * dpr; el.height = h * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#0b0d16";
      ctx.fillRect(0, 0, w, h);
      for (const s of stars) {
        ctx.fillStyle = `rgba(220,228,255,${s.b})`;
        ctx.fillRect(s.x * w, s.y * h, 1, 1);
      }
      const v = view.current;
      ctx.translate(w / 2 + v.x, h / 2 + v.y);
      ctx.scale(v.scale, v.scale);

      const focus = focusRef.current;
      const near = new Set<number>();
      if (focus !== null) {
        const fi = byId.get(focus);
        if (fi !== undefined) {
          near.add(fi);
          for (const [i, j] of links) {
            if (i === fi) near.add(j);
            if (j === fi) near.add(i);
          }
        }
      }
      const dim = (i: number) => focus !== null && !near.has(i);

      ctx.lineWidth = 1 / v.scale;
      for (const [i, j] of links) {
        const a = nodes[i], b = nodes[j];
        ctx.strokeStyle = dim(i) || dim(j) ? "rgba(160,170,210,0.06)" : "rgba(160,170,210,0.35)";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      nodes.forEach((n, i) => {
        const r = 3 + Math.sqrt(n.degree) * 2;
        ctx.globalAlpha = dim(i) ? 0.2 : 1;
        ctx.fillStyle = TYPE_COLOURS[n.type] ?? TYPE_COLOURS.other;
        ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
        if (n.documents.length > 1) {
          // A ring for entities that bridge documents -- the connections
          // nothing else in the app shows.
          ctx.strokeStyle = "rgba(255,255,255,0.8)";
          ctx.lineWidth = 1.2 / v.scale;
          ctx.beginPath(); ctx.arc(n.x, n.y, r + 2.5, 0, Math.PI * 2); ctx.stroke();
        }
        // Labels only where they can be read: well-connected nodes, and
        // everything near the focus. Labelling all 250 is a wall of text.
        if (!dim(i) && (n.degree >= 3 || near.has(i))) {
          ctx.fillStyle = "rgba(230,234,250,0.9)";
          ctx.font = `${11 / v.scale}px system-ui, sans-serif`;
          ctx.fillText(n.name, n.x + r + 3, n.y + 3);
        }
        ctx.globalAlpha = 1;
      });
    };

    const loop = () => {
      if (alpha > 0.02) step();
      draw();
      frame = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(frame);
  }, [data, byId]);

  // ---- interaction ----------------------------------------------------
  const nodeAt = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    const v = view.current;
    const x = (e.clientX - rect.left - rect.width / 2 - v.x) / v.scale;
    const y = (e.clientY - rect.top - rect.height / 2 - v.y) / v.scale;
    let best: string | null = null;
    let bestD = 14 / v.scale;
    for (const n of sim.current) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < bestD) { bestD = d; best = n.id; }
    }
    return best;
  };

  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  async function backfill() {
    setQueuing(true);
    try {
      await buildGraph();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not queue extraction");
    } finally {
      setQueuing(false);
    }
  }

  if (error) {
    return <p className="md-body-medium" style={{ color: "var(--md-error)" }}>{error}</p>;
  }
  if (!data) return <div className="md-skeleton h-[26rem]" aria-hidden />;

  const missing = data.n_documents - data.n_documents_with_graph;
  const node = selected ? data.nodes[byId.get(selected) ?? -1] : undefined;
  const relations: GraphEdge[] = selected
    ? data.edges.filter((e) => e.source === selected || e.target === selected)
    : [];
  const nameOf = (id: string) => data.nodes[byId.get(id) ?? -1]?.name ?? id;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <p className="md-body-small flex-1" style={{ color: "var(--md-on-surface-variant)" }}>
          {data.nodes.length} entities, {data.edges.length} stated relations from{" "}
          {data.n_documents_with_graph} of {data.n_documents} documents.
          {data.truncated && ` Showing the ${data.nodes.length} best connected of ${data.n_entities}.`}
          {data.pending > 0 && ` Extracting ${data.pending} more…`}
        </p>
        {missing > 0 && data.pending === 0 && (
          <Button variant="tonal" onClick={() => void backfill()} disabled={queuing}>
            {queuing && <IconSpinner />}
            Build graph for {missing} document{missing === 1 ? "" : "s"}
          </Button>
        )}
      </div>

      {data.nodes.length === 0 ? (
        <div
          className="flex h-[20rem] items-center justify-center rounded-[var(--md-shape-lg)]"
          style={{ background: "#0b0d16", color: "rgba(230,234,250,0.75)" }}
        >
          <p className="md-body-medium max-w-sm text-center">
            {data.pending > 0
              ? "Reading your documents for entities and relationships…"
              : "No graph yet. New uploads are read automatically; build one for existing documents above."}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-[1fr_18rem]">
          <canvas
            ref={canvas}
            className="h-[30rem] w-full rounded-[var(--md-shape-lg)]"
            style={{ cursor: hovered ? "pointer" : "grab" }}
            aria-label="Knowledge graph. Click an entity to see its relations."
            onMouseMove={(e) => {
              const d = drag.current;
              if (d) {
                view.current.x += e.clientX - d.x;
                view.current.y += e.clientY - d.y;
                d.moved ||= Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 2;
                d.x = e.clientX; d.y = e.clientY;
                return;
              }
              setHovered(nodeAt(e));
            }}
            onMouseDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; }}
            onMouseUp={(e) => {
              const moved = drag.current?.moved;
              drag.current = null;
              if (!moved) setSelected(nodeAt(e));
            }}
            onMouseLeave={() => { drag.current = null; setHovered(null); }}
            onWheel={(e) => {
              const v = view.current;
              v.scale = Math.min(4, Math.max(0.3, v.scale * (e.deltaY < 0 ? 1.1 : 0.9)));
            }}
          />
          <aside className="space-y-3">
            {!node ? (
              <>
                <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
                  Click an entity to see what your documents say about it. A ring marks
                  an entity that appears in more than one document.
                </p>
                <ul className="md-label-small flex flex-wrap gap-2">
                  {Object.entries(TYPE_COLOURS).map(([t, c]) => (
                    <li key={t} className="flex items-center gap-1.5">
                      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />
                      {t}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                <div>
                  <p className="md-title-small">{node.name}</p>
                  <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
                    {node.type} · {node.mentions} mention{node.mentions === 1 ? "" : "s"} ·{" "}
                    {node.documents.join(", ")}
                  </p>
                </div>
                <ul className="space-y-1.5">
                  {relations.map((r, i) => (
                    <li key={i}>
                      <button
                        type="button"
                        disabled={!r.chunk_id}
                        onClick={() => r.chunk_id && void showChunk(r.chunk_id)}
                        className="md-state md-body-small w-full rounded-[var(--md-shape-sm)] px-2 py-1.5 text-left"
                        style={{ background: "var(--md-surface-container-high)" }}
                        title="Open the passage that says this"
                      >
                        <strong>{nameOf(r.source)}</strong> {r.predicate}{" "}
                        <strong>{nameOf(r.target)}</strong>
                        <span className="block opacity-70">{r.filename}</span>
                      </button>
                    </li>
                  ))}
                </ul>
                <Button variant="text" onClick={() => setSelected(null)}>
                  Clear
                </Button>
              </>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

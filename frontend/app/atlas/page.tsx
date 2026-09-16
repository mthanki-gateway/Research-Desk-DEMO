"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { getAtlas, type Atlas } from "@/lib/api";
import { useApp } from "../providers";
import { Button } from "../md";
import Scatter, { PALETTES, type PlotTheme } from "./scatter";
import Heatmap from "./heatmap";

/**
 * The corpus as geometry rather than as chat.
 *
 * Two views of the same 768-dimensional space, answering questions retrieval
 * quality depends on and nothing else in the app shows: is a document an
 * outlier, are two chunks near-duplicates, is a chunk similar to nothing at
 * all. Both are computed from vectors that already exist -- no embedding
 * calls, no model calls.
 */
export default function AtlasPage() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  // Which document the camera is framing, or null for the whole corpus.
  const [focus, setFocus] = useState<string | null>(null);
  // Lifted out of the plot so the LEGEND uses the same palette the points
  // do. Held separately from the app theme: the right ground for a point
  // cloud is not the right ground for a page of text.
  const [plotTheme, setPlotTheme] = useState<PlotTheme>("dark");
  const { showChunk } = useApp();

  const load = useCallback(async () => {
    setError(null);
    try {
      setAtlas(await getAtlas());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the atlas");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const files = useMemo(
    () => [...new Set((atlas?.points ?? []).map((p) => p.filename))].sort(),
    [atlas],
  );

  const point = selected !== null ? atlas?.points[selected] : undefined;

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-9">
        <p className="md-body-medium" style={{ color: "var(--md-error)" }}>
          {error}
        </p>
        <Button variant="text" onClick={() => void load()}>
          Retry
        </Button>
      </div>
    );
  }

  if (!atlas) {
    return (
      <div className="mx-auto max-w-4xl space-y-3 px-6 py-9" aria-hidden>
        <div className="md-skeleton h-8 w-40" />
        <div className="md-skeleton h-4 w-72" />
        <div className="md-skeleton mt-6 h-[26rem]" />
      </div>
    );
  }

  if (atlas.points.length === 0) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-9">
        <h1 className="md-headline-small">Atlas</h1>
        <p
          className="md-body-medium mt-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Nothing is indexed yet. Upload a document in the Library and its
          chunks will appear here.
        </p>
      </div>
    );
  }

  const explained = atlas.explained_variance.reduce((a, b) => a + b, 0);

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Atlas</h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {atlas.points.length} chunks from {atlas.n_documents} document
          {atlas.n_documents === 1 ? "" : "s"}, laid out by what they mean.
          Hover a point for its metadata, click to inspect it, and click a
          document below to fly to it.
        </p>
      </header>

      {/* The legend IS the navigation. A colour key that only explains is a
          wasted control when the obvious question looking at it is "show me
          that one" -- clicking flies the camera to that document and fades
          the rest. */}
      <div className="flex flex-wrap items-center gap-2">
        {files.map((f, i) => {
          const on = focus === f;
          return (
            <button
              key={f}
              type="button"
              onClick={() => setFocus(on ? null : f)}
              className="md-label-small flex items-center gap-1.5 rounded-[var(--md-shape-full)] px-2.5 py-1"
              style={{
                background: on
                  ? "var(--md-secondary-container)"
                  : "var(--md-surface-container-high)",
                color: on ? "var(--md-on-secondary-container)" : undefined,
              }}
              aria-pressed={on}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{
                  background:
                    PALETTES[plotTheme][i % PALETTES[plotTheme].length],
                }}
              />
              {f}
              <span style={{ opacity: 0.6 }}>
                {atlas.points.filter((p) => p.filename === f).length}
              </span>
            </button>
          );
        })}
        {focus && (
          <Button variant="text" onClick={() => setFocus(null)}>
            Show all
          </Button>
        )}
      </div>

      <Scatter
        points={atlas.points}
        selected={selected}
        onSelect={setSelected}
        focus={focus}
        onFocusChange={setFocus}
        theme={plotTheme}
        onThemeChange={setPlotTheme}
      />

      {/* SAID PLAINLY, because a 3D plot implies a faithful map and this one
          is not. Three linear axes cannot carry 768 dimensions; the number
          says how much of the real structure survived the projection, and
          without it the picture overclaims. */}
      <p
        className="md-body-small"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        These three axes account for{" "}
        <strong>{(explained * 100).toFixed(0)}%</strong> of the variation in
        the full 768-dimensional space (PCA). Gross structure — one document
        sitting apart from the rest — is real. Fine detail is not: chunks that
        look close here may be far apart in the space retrieval actually
        searches.
      </p>

      <section className="grid gap-6 md:grid-cols-[1fr_18rem]">
        <div>
          <h2 className="md-title-medium">Similarity</h2>
          <p
            className="md-body-small mt-0.5 mb-3"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Every chunk against every other, ordered by document. Bright blocks
            off the diagonal are duplicated coverage; a dark row is a chunk
            similar to nothing, which usually means the chunker split it badly.
          </p>
          <Heatmap
            points={atlas.points}
            similarity={atlas.similarity}
            selected={selected}
            onSelect={setSelected}
          />
        </div>

        <aside className="space-y-3">
          <h2 className="md-title-medium">Selection</h2>
          {!point ? (
            <p
              className="md-body-small"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Click a point or a heatmap row.
            </p>
          ) : (
            <div className="md-card md-card-outlined space-y-2 p-4">
              <p className="md-title-small break-words">{point.filename}</p>
              <p
                className="md-body-small"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {point.heading ?? "no heading"} · chunk {point.chunk_index} ·{" "}
                {point.n_chars} chars
              </p>
              <p className="md-body-small">{point.preview}…</p>

              {point.nearest && (
                <div
                  className="md-body-small rounded-[var(--md-shape-sm)] p-2"
                  style={{ background: "var(--md-surface-container-high)" }}
                >
                  Nearest: <strong>{point.nearest.score}</strong>
                  <br />
                  {point.nearest.filename} · chunk {point.nearest.chunk_index}
                  {/* The number that matters. Above ~0.95 the two chunks are
                      saying the same thing, and retrieval will often return
                      both -- spending two of its slots on one fact. */}
                  {point.nearest.score >= 0.95 && (
                    <span style={{ color: "var(--md-error)" }}>
                      {" "}
                      — near-duplicate
                    </span>
                  )}
                </div>
              )}

              {/* Reuses the chunk panel the Lab already uses, so "look at the
                  whole passage" is not a second thing to build. */}
              <Button variant="text" onClick={() => void showChunk(point.chunk_id)}>
                Open full chunk
              </Button>
            </div>
          )}
        </aside>
      </section>

      {atlas.truncated && (
        <p className="md-body-small" style={{ color: "var(--md-error)" }}>
          Showing the first {atlas.points.length} chunks only.
        </p>
      )}
    </div>
  );
}

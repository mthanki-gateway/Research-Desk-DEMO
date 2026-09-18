"use client";

import { useState } from "react";

/**
 * How somebody sounded, turn by turn.
 *
 * WHY A TIMELINE RATHER THAN THREE NUMBERS
 *
 * An average arousal of 0.58 says almost nothing: people differ, microphones
 * differ, rooms differ. What carries meaning is the SHAPE -- where they lifted,
 * where they went flat, which question they hurried past. Three averages throw
 * exactly that away, and it is the only part a reader could act on.
 *
 * So the turns are the x-axis and the turns are the chapters: the same
 * boundaries the profile was recorded against, which means a departure here
 * points at a specific question rather than at a timestamp nobody can place.
 *
 * INLINE SVG, NO CHART LIBRARY. Recharts and friends are hundreds of kilobytes
 * and an image rebuild for one chart of at most a few dozen points. This is
 * about eighty lines and has no opinions about theming, which matters when the
 * palette is a set of CSS variables that change with the theme.
 */

type Turn = {
  turn: number;
  arousal: number;
  dominance: number;
  valence: number;
};

type Moment = {
  turn: number;
  dimension: string;
  delta: number;
  direction: string;
};

/** The three dimensions, and the colour each keeps everywhere it appears. */
const LINES = [
  { key: "arousal", label: "Arousal", hint: "calm → animated", colour: "var(--md-primary)" },
  { key: "dominance", label: "Dominance", hint: "tentative → assertive", colour: "var(--md-tertiary)" },
  { key: "valence", label: "Valence", hint: "negative → positive", colour: "var(--md-secondary)" },
] as const;

const W = 560;
const H = 180;
const PAD = { top: 12, right: 12, bottom: 26, left: 30 };

export default function VoiceTimeline({
  turns,
  baseline,
  moments,
}: {
  turns: Turn[];
  baseline: Record<string, number>;
  moments: Moment[];
}) {
  /** Which dimensions are drawn. All three at once is a tangle on a small
   *  chart, so any of them can be muted -- and muting is remembered only for
   *  as long as the card is open, because it is a way of reading, not a
   *  setting. */
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [active, setActive] = useState<number | null>(null);

  if (turns.length === 0) return null;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  // A single turn would divide by zero and, more usefully, has no shape to
  // show -- so it is drawn as one point in the middle rather than a line.
  const step = turns.length > 1 ? plotW / (turns.length - 1) : 0;

  const x = (i: number) => PAD.left + (turns.length > 1 ? i * step : plotW / 2);
  const y = (v: number) => PAD.top + (1 - Math.max(0, Math.min(1, v))) * plotH;

  const shown = LINES.filter((l) => !hidden.has(l.key));

  return (
    <div>
      <div className="md-scroll -mx-1 overflow-x-auto px-1">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          style={{ minWidth: Math.max(320, turns.length * 44) }}
          role="img"
          aria-label="How the speaker sounded, turn by turn"
        >
          {/* Gridlines at the quarters, plus their labels. Only three, because
              the numbers are not precise enough to deserve more. */}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(v)}
                y2={y(v)}
                stroke="var(--md-outline-variant)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 6}
                y={y(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--md-on-surface-variant)"
              >
                {v.toFixed(1)}
              </text>
            </g>
          ))}

          {/* The speaker's own baseline, dashed, per visible dimension. The
              comparison that matters is against THEMSELVES -- a population
              average would be measuring the microphone as much as the person. */}
          {shown.map((line) => (
            <line
              key={`base-${line.key}`}
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(baseline[line.key] ?? 0)}
              y2={y(baseline[line.key] ?? 0)}
              stroke={line.colour}
              strokeWidth={1}
              strokeDasharray="3 4"
              opacity={0.35}
            />
          ))}

          {/* Chapter dividers: one per turn, so a departure points at a
              question rather than at a moment in a waveform. */}
          {turns.map((t, i) => (
            <line
              key={`tick-${t.turn}`}
              x1={x(i)}
              x2={x(i)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--md-outline-variant)"
              strokeWidth={active === t.turn ? 2 : 1}
              opacity={active === t.turn ? 0.9 : 0.25}
            />
          ))}

          {shown.map((line) => (
            <polyline
              key={line.key}
              fill="none"
              stroke={line.colour}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              points={turns
                .map((t, i) => `${x(i)},${y(t[line.key] as number)}`)
                .join(" ")}
            />
          ))}

          {/* A ring on every point that departed from the baseline enough to
              be called out. The list below says which and by how much; this is
              so the eye finds them without reading it. */}
          {moments.map((m, i) => {
            const index = turns.findIndex((t) => t.turn === m.turn);
            if (index < 0 || hidden.has(m.dimension)) return null;
            const line = LINES.find((l) => l.key === m.dimension);
            return (
              <circle
                key={`m-${i}`}
                cx={x(index)}
                cy={y(turns[index][m.dimension as keyof Turn] as number)}
                r={4}
                fill="var(--md-surface)"
                stroke={line?.colour ?? "var(--md-primary)"}
                strokeWidth={2}
              />
            );
          })}

          {/* Hit targets last, so they sit above everything and the whole
              column is clickable rather than a 4px dot. */}
          {turns.map((t, i) => (
            <rect
              key={`hit-${t.turn}`}
              x={x(i) - step / 2 || PAD.left}
              y={PAD.top}
              width={step || plotW}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setActive(t.turn)}
              onMouseLeave={() => setActive(null)}
            />
          ))}

          {turns.map((t, i) => (
            <text
              key={`lab-${t.turn}`}
              x={x(i)}
              y={H - 8}
              textAnchor="middle"
              fontSize={9}
              fill="var(--md-on-surface-variant)"
              opacity={turns.length > 14 && i % 2 ? 0 : 1}
            >
              {t.turn + 1}
            </text>
          ))}
        </svg>
      </div>

      {/* The legend doubles as the control. Clicking a name mutes that line,
          which is the only way three overlapping series stay readable on a
          chart this size. */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {LINES.map((line) => {
          const off = hidden.has(line.key);
          const reading = active !== null
            ? turns.find((t) => t.turn === active)?.[line.key]
            : undefined;
          return (
            <button
              key={line.key}
              type="button"
              aria-pressed={!off}
              onClick={() =>
                setHidden((prev) => {
                  const next = new Set(prev);
                  if (next.has(line.key)) next.delete(line.key);
                  else next.add(line.key);
                  return next;
                })
              }
              className="md-body-small flex items-center gap-1.5"
              style={{ opacity: off ? 0.4 : 1 }}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: line.colour }}
              />
              <span>{line.label}</span>
              <span style={{ color: "var(--md-on-surface-variant)" }}>
                {reading !== undefined
                  ? (reading as number).toFixed(2)
                  : line.hint}
              </span>
            </button>
          );
        })}
      </div>

      <p
        className="md-body-small mt-1 h-4"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {active !== null ? `Turn ${active + 1}` : "Hover a turn to read it"}
      </p>
    </div>
  );
}

"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { meetingAudioUrl } from "@/lib/api";

/**
 * One continuous player over a meeting's recorded parts.
 *
 * A meeting is stored as several short files, and the files do not carry a
 * duration the browser can use (a recorder's output reports Infinity), which is
 * what broke the stock <audio> controls: the bar sat at the end and seeking did
 * nothing. So the timeline is built from the lengths the server recorded, the
 * parts are played back to back, and a position anywhere in the meeting is
 * mapped to a part and an offset into it.
 *
 * Laid out like a music player's bar: a thin full-width seek line whose thumb
 * only appears when it is touched, then the transport row underneath.
 */

export type Part = { index: number; start: number; seconds: number };
export type PlayerHandle = { seekTo: (seconds: number, play?: boolean) => void };

const RATES = [1, 1.25, 1.5, 2, 0.75];

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function Icon({ d, size = 24 }: { d: string; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden>
      <path d={d} />
    </svg>
  );
}

const PLAY = "M8 5v14l11-7z";
const PAUSE = "M6 5h4v14H6zm8 0h4v14h-4z";
// Circular arrows with the 10 baked into the glyph, the way music apps draw it.
const BACK10 =
  "M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8zm-1.1 11h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z";
const FWD10 =
  "M18 13c0 3.31-2.69 6-6 6s-6-2.69-6-6 2.69-6 6-6v4l5-5-5-5v4c-4.42 0-8 3.58-8 8s3.58 8 8 8 8-3.58 8-8h-2zm-7.46 3h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z";
const VOL = "M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z";
const MUTE =
  "M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.8 8.8 0 0 0 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a8.99 8.99 0 0 0 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4 9.91 6.09 12 8.18V4z";

export const Player = forwardRef<PlayerHandle, { id: string; parts: Part[] }>(function Player(
  { id, parts },
  ref,
) {
  const sorted = [...parts].sort((a, b) => a.start - b.start);
  const total = sorted.length ? sorted[sorted.length - 1].start + sorted[sorted.length - 1].seconds : 0;

  const el = useRef<HTMLAudioElement>(null);
  const urls = useRef<Map<number, string>>(new Map());
  const current = useRef(0); // index into `sorted`
  const pending = useRef<{ offset: number; play: boolean } | null>(null);
  const dragging = useRef(false);

  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [rate, setRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState("");

  // A different meeting: forget everything about the last one.
  useEffect(() => {
    urls.current.forEach((u) => URL.revokeObjectURL(u));
    urls.current = new Map();
    current.current = 0;
    pending.current = null;
    setTime(0);
    setPlaying(false);
    setError("");
    const a = el.current;
    if (a) {
      a.pause();
      a.removeAttribute("src");
    }
  }, [id]);

  useEffect(
    () => () => {
      urls.current.forEach((u) => URL.revokeObjectURL(u));
    },
    [],
  );

  const urlFor = useCallback(
    async (i: number): Promise<string> => {
      const have = urls.current.get(i);
      if (have) return have;
      const u = await meetingAudioUrl(id, sorted[i].index);
      urls.current.set(i, u);
      return u;
    },
    // `sorted` is rebuilt every render from `parts`; the part list itself is
    // what changes between meetings.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [id, parts],
  );

  /** Put the global position `t` under the playhead, loading its part if needed. */
  const seekTo = useCallback(
    async (t: number, play = true) => {
      const a = el.current;
      if (!a || !sorted.length) return;
      const clamped = Math.max(0, Math.min(total, t));
      let i = 0;
      for (let k = 0; k < sorted.length; k++) if (sorted[k].start <= clamped + 0.001) i = k;
      const offset = Math.max(0, clamped - sorted[i].start);
      setTime(clamped);
      try {
        if (current.current !== i || !a.src) {
          setLoading(true);
          pending.current = { offset, play };
          current.current = i;
          a.src = await urlFor(i);
          a.playbackRate = rate;
          a.load();
        } else {
          a.currentTime = offset;
          if (play) void a.play().catch(() => {});
        }
      } catch (e) {
        setLoading(false);
        setError(e instanceof Error ? e.message : "Could not load the recording.");
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [urlFor, total, rate, parts],
  );

  useImperativeHandle(ref, () => ({ seekTo: (s, p = true) => void seekTo(s, p) }), [seekTo]);

  const toggle = useCallback(() => {
    const a = el.current;
    if (!a) return;
    if (!a.src) {
      void seekTo(time, true);
    } else if (a.paused) {
      void a.play().catch(() => {});
    } else {
      a.pause();
    }
  }, [seekTo, time]);

  // Smooth bar: the media element only reports ~4 times a second.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const a = el.current;
      if (a && !dragging.current && sorted[current.current]) {
        setTime(sorted[current.current].start + a.currentTime);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, parts]);

  const onLoaded = () => {
    const a = el.current;
    const want = pending.current;
    if (!a) return;
    pending.current = null;
    setLoading(false);
    const apply = () => {
      if (want) a.currentTime = want.offset;
      if (want?.play) void a.play().catch(() => {});
    };
    // A recorder's file reports Infinity until something forces the browser to
    // find the end. Jumping far past it does that; then the real seek is made.
    if (!Number.isFinite(a.duration)) {
      const fix = () => {
        a.removeEventListener("timeupdate", fix);
        apply();
      };
      a.addEventListener("timeupdate", fix);
      a.currentTime = 1e101;
    } else {
      apply();
    }
  };

  const onEnded = () => {
    const next = current.current + 1;
    if (next < sorted.length) {
      void seekTo(sorted[next].start, true);
    } else {
      setPlaying(false);
      setTime(total);
    }
  };

  // Fetch the next part a little before it is needed, so the join is silent.
  const onTimeUpdate = () => {
    const a = el.current;
    const part = sorted[current.current];
    if (!a || !part) return;
    if (part.seconds - a.currentTime < 15 && current.current + 1 < sorted.length) {
      void urlFor(current.current + 1).catch(() => {});
    }
  };

  useEffect(() => {
    const a = el.current;
    if (a) {
      a.volume = volume;
      a.muted = muted;
      a.playbackRate = rate;
    }
  }, [volume, muted, rate]);

  const pct = total ? Math.min(100, (time / total) * 100) : 0;

  return (
    <div
      className="rounded-[var(--md-shape-lg)] px-4 pb-2 pt-3"
      style={{ background: "var(--md-surface-container-high)" }}
    >
      <audio
        ref={el}
        preload="none"
        onLoadedMetadata={onLoaded}
        onEnded={onEnded}
        onTimeUpdate={onTimeUpdate}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onWaiting={() => setLoading(true)}
        onPlaying={() => setLoading(false)}
      />

      <input
        type="range"
        className="md-seek"
        aria-label="Seek"
        min={0}
        max={total || 1}
        step={0.1}
        value={Math.min(time, total || 1)}
        style={{ ["--p" as string]: `${pct}%` }}
        data-active={dragging.current || undefined}
        onPointerDown={() => (dragging.current = true)}
        onPointerUp={(e) => {
          dragging.current = false;
          void seekTo(Number(e.currentTarget.value), playing);
        }}
        onChange={(e) => {
          const v = Number(e.target.value);
          setTime(v);
          // Dragging only moves the thumb; the keyboard commits each step.
          if (!dragging.current) void seekTo(v, playing);
        }}
      />

      <div className="mt-1 flex items-center gap-1">
        <button
          type="button"
          aria-label="Back 10 seconds"
          onClick={() => void seekTo(time - 10, playing)}
          className="md-state grid h-10 w-10 place-items-center rounded-full"
        >
          <Icon d={BACK10} />
        </button>
        <button
          type="button"
          aria-label={playing ? "Pause" : "Play"}
          onClick={toggle}
          disabled={!sorted.length}
          className="md-state grid h-12 w-12 place-items-center rounded-full disabled:cursor-not-allowed disabled:opacity-50"
          style={{ background: "var(--md-primary)", color: "var(--md-on-primary)" }}
        >
          {loading ? (
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Icon d={playing ? PAUSE : PLAY} size={28} />
          )}
        </button>
        <button
          type="button"
          aria-label="Forward 10 seconds"
          onClick={() => void seekTo(time + 10, playing)}
          className="md-state grid h-10 w-10 place-items-center rounded-full"
        >
          <Icon d={FWD10} />
        </button>

        <span className="md-label-medium ml-2 tabular-nums" style={{ color: "var(--md-on-surface-variant)" }}>
          {clock(time)} / {clock(total)}
        </span>

        <span className="flex-1" />

        <button
          type="button"
          aria-label={`Speed ${rate}x. Change`}
          onClick={() => setRate(RATES[(RATES.indexOf(rate) + 1) % RATES.length])}
          className="md-label-medium md-state rounded-[var(--md-shape-full)] px-2.5 py-1 tabular-nums"
        >
          {rate}×
        </button>
        <button
          type="button"
          aria-label={muted ? "Unmute" : "Mute"}
          onClick={() => setMuted((m) => !m)}
          className="md-state grid h-9 w-9 place-items-center rounded-full"
        >
          <Icon d={muted || volume === 0 ? MUTE : VOL} size={22} />
        </button>
        <input
          type="range"
          className="md-seek hidden w-20 sm:block"
          aria-label="Volume"
          min={0}
          max={1}
          step={0.05}
          value={muted ? 0 : volume}
          style={{ ["--p" as string]: `${(muted ? 0 : volume) * 100}%` }}
          onChange={(e) => {
            setMuted(false);
            setVolume(Number(e.target.value));
          }}
        />
      </div>
      {error && (
        <p className="md-label-small pb-1" style={{ color: "var(--md-error)" }}>
          {error}
        </p>
      )}
    </div>
  );
});

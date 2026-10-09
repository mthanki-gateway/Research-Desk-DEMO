"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type Meeting,
  type MeetingStatus,
  analyzeMeeting,
  createMeeting,
  deleteMeeting,
  getManksStatus,
  getMeeting,
  leaveMeeting,
  listMeetings,
} from "@/lib/api";
import { Button, ConfirmButton, Switch } from "../../md";
import { IconManks, IconSpinner } from "../../icons";
import { Player, type PlayerHandle } from "./player";

/**
 * Manks: send bots to meetings, several at once, and read what they brought back.
 *
 * Each bot runs in a separate container and joins as a guest, records the
 * room's audio in short segments, and reports its progress here through the
 * status on each meeting. Afterwards the API transcribes the segments and writes
 * the notes. A BigBlueButton bot sent with "Talks" also joins with a microphone
 * and takes part, through the Duplex voice pipeline on the API.
 */

const LABEL: Record<MeetingStatus, string> = {
  queued: "Waiting for the bot",
  joining: "Opening the meeting",
  waiting: "Waiting to be let in",
  in_meeting: "In the meeting",
  transcribing: "Transcribing",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

const BUSY: MeetingStatus[] = ["queued", "joining", "waiting", "in_meeting", "transcribing"];
const LIVE: MeetingStatus[] = ["queued", "joining", "waiting", "in_meeting"];

function stamp(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/**
 * What to call a meeting when it has no title. A BigBlueButton link carries the
 * room's password and checksum in its query string, so it is never shown: only
 * the server it points at.
 */
const isBbb = (m: Pick<Meeting, "platform">) => m.platform === "bbb" || m.platform === "bbb_native";

function where(m: Pick<Meeting, "platform" | "url">): string {
  try {
    const u = new URL(m.url);
    return isBbb(m) ? `${u.host} (BigBlueButton)` : `${u.host}${u.pathname}`;
  } catch {
    return isBbb(m) ? "BigBlueButton" : m.url;
  }
}

function Banner({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
      style={{ background: "var(--md-error-container)", color: "var(--md-on-error-container)" }}
    >
      {children}
    </p>
  );
}

function Chip({ status }: { status: MeetingStatus }) {
  const good = status === "done";
  const bad = status === "failed";
  return (
    <span
      className="md-label-small flex shrink-0 items-center gap-1.5 rounded-[var(--md-shape-full)] px-2.5 py-1"
      style={{
        background: good
          ? "var(--md-primary-container)"
          : bad
            ? "var(--md-error-container)"
            : "var(--md-surface-container-high)",
        color: good
          ? "var(--md-on-primary-container)"
          : bad
            ? "var(--md-on-error-container)"
            : "var(--md-on-surface-variant)",
      }}
    >
      {BUSY.includes(status) && <IconSpinner className="h-3 w-3" />}
      {LABEL[status]}
    </span>
  );
}

/** One bot to send, as configured in the "New bots" card before it is sent. */
type Draft = {
  key: number;
  kind: "meet" | "bbb";
  // BigBlueButton only: a real browser, or the lightweight client with none.
  method: "browser" | "native";
  // BigBlueButton only: it joins with a microphone and takes part out loud.
  talk: boolean;
  link: string;
  title: string;
};

const PAGE = 15;

let nextKey = 1;
const blank = (from?: Draft): Draft => ({
  key: nextKey++,
  kind: from?.kind ?? "bbb",
  method: from?.method ?? "native",
  talk: from?.talk ?? false,
  link: "",
  title: "",
});

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly (readonly [T, string])[];
  onChange: (v: T) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="flex w-fit gap-1 rounded-[var(--md-shape-full)] p-1"
      style={{ background: "var(--md-surface-container-high)" }}
    >
      {options.map(([id, text]) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={value === id}
          onClick={() => onChange(id)}
          className="md-label-medium md-state rounded-[var(--md-shape-full)] px-3 py-1"
          style={{
            background: value === id ? "var(--md-secondary-container)" : "transparent",
            color: value === id ? "var(--md-on-secondary-container)" : "var(--md-on-surface-variant)",
          }}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function hint(d: Draft): string {
  if (d.kind === "meet") {
    return "Joins as a guest and asks to be let in, so the host has to admit it. Microphone and camera stay off.";
  }
  const how =
    d.method === "native"
      ? "Lightweight: no browser, a few MB per meeting, so many can run at once."
      : "Browser: a real Chromium, about 1 GB per meeting.";
  const voice = d.talk
    ? " It joins with a microphone and takes part out loud (the Duplex voice pipeline)."
    : " It joins listen-only.";
  return `${how}${voice} Paste the join link exactly as generated; it carries a checksum.`;
}

function DraftCard({
  d,
  index,
  removable,
  onChange,
  onRemove,
}: {
  d: Draft;
  index: number;
  removable: boolean;
  onChange: (d: Draft) => void;
  onRemove: () => void;
}) {
  const set = (patch: Partial<Draft>) => onChange({ ...d, ...patch });
  return (
    <div
      className="space-y-3 rounded-[var(--md-shape-md)] p-4"
      style={{ background: "var(--md-surface-container)" }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="md-label-large mr-1" style={{ color: "var(--md-on-surface-variant)" }}>
          Bot {index + 1}
        </span>
        <Segmented
          label="Kind of meeting"
          value={d.kind}
          options={[["bbb", "BigBlueButton"], ["meet", "Google Meet"]] as const}
          onChange={(kind) => set({ kind })}
        />
        {d.kind === "bbb" && (
          <Segmented
            label="How the bot joins"
            value={d.method}
            options={[["native", "Lightweight"], ["browser", "Browser"]] as const}
            onChange={(method) => set({ method })}
          />
        )}
        {d.kind === "bbb" && (
          <label className="md-label-large ml-1 flex items-center gap-2">
            <Switch on={d.talk} onChange={(talk) => set({ talk })} aria-label="Let the bot talk" />
            Talks
          </label>
        )}
        {removable && (
          <button
            type="button"
            onClick={onRemove}
            aria-label={`Remove bot ${index + 1}`}
            title="Remove"
            className="md-icon-btn md-icon-btn-sm md-state ml-auto"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            ✕
          </button>
        )}
      </div>
      <div className="flex flex-col gap-3 sm:flex-row">
        <input
          value={d.link}
          onChange={(e) => set({ link: e.target.value })}
          placeholder={
            d.kind === "meet"
              ? "https://meet.google.com/abc-defg-hij"
              : "https://your-server/bigbluebutton/api/join?meetingID=…&checksum=…"
          }
          aria-label={`Meeting link for bot ${index + 1}`}
          className="md-body-medium min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-3 py-2.5 outline-none"
          style={{ background: "var(--md-surface-container-highest)", color: "var(--md-on-surface)" }}
        />
        <input
          value={d.title}
          onChange={(e) => set({ title: e.target.value })}
          placeholder="Title (optional)"
          aria-label={`Title for bot ${index + 1}`}
          className="md-body-medium rounded-[var(--md-shape-sm)] px-3 py-2.5 outline-none sm:w-56"
          style={{ background: "var(--md-surface-container-highest)", color: "var(--md-on-surface)" }}
        />
      </div>
      <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
        {hint(d)}
      </p>
    </div>
  );
}

function since(iso: string | null): string {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString([], { dateStyle: "medium" });
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="md-label-small rounded-[var(--md-shape-full)] px-2 py-0.5"
      style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface-variant)" }}
    >
      {children}
    </span>
  );
}

function BotRow({
  m,
  onOpen,
  onLeave,
  onDelete,
}: {
  m: Meeting;
  onOpen: () => void;
  onLeave: () => void;
  onDelete: () => void;
}) {
  const live = LIVE.includes(m.status);
  return (
    <li>
      <div
        role="button"
        tabIndex={0}
        onClick={onOpen}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onOpen()}
        className="md-state flex flex-col gap-2 rounded-[var(--md-shape-md)] px-4 py-3 sm:flex-row sm:items-center sm:gap-4"
        style={{ background: "var(--md-surface-container)" }}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {m.status === "in_meeting" && (
              <span
                aria-hidden
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: "var(--md-primary)", boxShadow: "0 0 0 3px var(--md-primary-container)" }}
              />
            )}
            <span className="md-title-small truncate">{m.insights?.title || m.title || where(m)}</span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <Tag>{isBbb(m) ? "BigBlueButton" : "Google Meet"}</Tag>
            {isBbb(m) && <Tag>{m.platform === "bbb_native" ? "Lightweight" : "Browser"}</Tag>}
            {m.talk && <Tag>Talks</Tag>}
            <span className="md-label-small truncate" style={{ color: "var(--md-on-surface-variant)" }}>
              {m.error || m.detail || where(m)}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <span className="md-label-small w-20 text-right tabular-nums" style={{ color: "var(--md-on-surface-variant)" }}>
            {m.recorded_seconds > 0 ? stamp(m.recorded_seconds) : "—"}
            <span className="block">{since(m.created_at)}</span>
          </span>
          <Chip status={m.status} />
          {live && (
            <Button
              variant="tonal"
              size="sm"
              disabled={m.leave_requested}
              onClick={(e) => {
                e.stopPropagation();
                onLeave();
              }}
            >
              {m.status === "queued" ? "Cancel" : m.leave_requested ? "Leaving…" : "Leave"}
            </Button>
          )}
          {/* Its own click target: the row behind it opens the meeting. */}
          <span onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
            <ConfirmButton
              label="Delete"
              title="Delete this meeting?"
              body={
                live
                  ? "The bot leaves the meeting, and the recording, transcript and notes are removed. This cannot be undone."
                  : "The recording, transcript and notes are removed. This cannot be undone."
              }
              confirmLabel="Delete"
              onConfirm={onDelete}
            />
          </span>
        </div>
      </div>
    </li>
  );
}

export default function ManksSurface() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [total, setTotal] = useState(0);
  const [liveCount, setLiveCount] = useState(0);
  const [anyBusy, setAnyBusy] = useState(false);
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Meeting | null>(null);
  const [drafts, setDrafts] = useState<Draft[]>(() => [blank()]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "active" | "finished">("all");

  // Typing settles for a moment before it becomes a query.
  useEffect(() => {
    const t = window.setTimeout(() => {
      setQ(search.trim());
      setPage(0);
    }, 300);
    return () => window.clearTimeout(t);
  }, [search]);

  const refresh = useCallback(async () => {
    try {
      const res = await listMeetings({ q, show: filter, offset: page * PAGE, limit: PAGE });
      // Deleting the last row of the last page leaves an empty page behind.
      if (!res.items.length && page > 0) {
        setPage((p) => Math.max(0, p - 1));
        return;
      }
      setMeetings(res.items);
      setTotal(res.total);
      setLiveCount(res.live);
      setAnyBusy(res.busy > 0);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load meetings.");
    }
  }, [q, filter, page]);

  useEffect(() => {
    getManksStatus()
      .then((s) => setEnabled(s.enabled))
      .catch(() => setEnabled(false));
    void refresh();
  }, [refresh]);

  // Poll while anything is in motion; stop when everything has settled.
  useEffect(() => {
    if (!anyBusy) return;
    const t = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(t);
  }, [anyBusy, refresh]);

  // The open meeting is reloaded with its notes whenever its status changes.
  const current = meetings.find((m) => m.id === selected) ?? null;
  const currentStatus = current?.status;
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    getMeeting(selected)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected, currentStatus]);

  const ready = drafts.filter((d) => d.link.trim());

  async function sendAll() {
    if (!ready.length) return;
    setBusy(true);
    setError(null);
    // One at a time, so a bad link fails on its own and the rest still go.
    const failed: Draft[] = [];
    const problems: string[] = [];
    for (const d of ready) {
      try {
        await createMeeting(
          d.link.trim(),
          d.title.trim(),
          d.kind === "bbb" ? d.method : "browser",
          d.kind === "bbb" && d.talk,
        );
      } catch (e) {
        failed.push(d);
        problems.push(`${d.title || `Bot ${drafts.indexOf(d) + 1}`}: ${e instanceof Error ? e.message : "could not send"}`);
      }
    }
    // What failed stays in the form to be fixed; what went is cleared.
    setDrafts(failed.length ? failed : [blank(drafts[drafts.length - 1])]);
    if (problems.length) setError(problems.join(" · "));
    await refresh();
    setBusy(false);
  }

  async function act(fn: () => Promise<unknown>) {
    setError(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "That did not work.");
    }
  }

  const pages = Math.max(1, Math.ceil(total / PAGE));

  if (selected) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-6 py-9">
        <Button variant="text" size="sm" onClick={() => setSelected(null)}>
          ← All bots
        </Button>
        {error && <Banner>{error}</Banner>}
        {!detail ? (
          <IconSpinner className="h-5 w-5 opacity-50" />
        ) : (
          <Detail
            m={detail}
            onLeave={() => void act(() => leaveMeeting(detail.id))}
            onAnalyze={() => void act(() => analyzeMeeting(detail.id))}
            onDelete={() =>
              void act(async () => {
                await deleteMeeting(detail.id);
                setSelected(null);
              })
            }
          />
        )}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small flex items-center gap-2">
          <IconManks className="h-6 w-6" />
          Manks
        </h1>
        <p className="md-body-medium mt-1" style={{ color: "var(--md-on-surface-variant)" }}>
          Send bots into meetings, as many at once as you like. Each one listens (or takes part), and
          afterwards gives you the transcript and the notes.
        </p>
      </header>

      {enabled === false && (
        <Banner>
          The Manks bot is not set up on this deployment (no bot secret). Start the bot container and set
          MANKS_BOT_SECRET on both sides.
        </Banner>
      )}
      {error && <Banner>{error}</Banner>}

      <section className="md-card md-card-outlined space-y-3 p-5">
        <h2 className="md-title-medium">New bots</h2>
        {drafts.map((d, i) => (
          <DraftCard
            key={d.key}
            d={d}
            index={i}
            removable={drafts.length > 1}
            onChange={(next) => setDrafts(drafts.map((x) => (x.key === d.key ? next : x)))}
            onRemove={() => setDrafts(drafts.filter((x) => x.key !== d.key))}
          />
        ))}
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outlined" onClick={() => setDrafts([...drafts, blank(drafts[drafts.length - 1])])}>
            + Add another
          </Button>
          <Button onClick={() => void sendAll()} disabled={busy || !ready.length || enabled === false} className="ml-auto">
            {busy
              ? "Sending…"
              : ready.length > 1
                ? `Send ${ready.length} bots`
                : "Send the bot"}
          </Button>
        </div>
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="md-title-medium">
            All bots
            {liveCount > 0 && (
              <span className="md-label-medium ml-2" style={{ color: "var(--md-primary)" }}>
                {liveCount} live
              </span>
            )}
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search title or server"
              aria-label="Search meetings"
              className="md-body-medium w-56 rounded-[var(--md-shape-full)] px-4 py-2 outline-none"
              style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface)" }}
            />
            <Segmented
              label="Show"
              value={filter}
              options={[["all", "All"], ["active", "Active"], ["finished", "Finished"]] as const}
              onChange={(f) => {
                setFilter(f);
                setPage(0);
              }}
            />
          </div>
        </div>
        {meetings.length === 0 ? (
          <p className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
            {q || filter !== "all" ? "No meetings match." : "No bots yet. Configure one above and send it."}
          </p>
        ) : (
          <ul className="space-y-1.5">
            {meetings.map((m) => (
              <BotRow
                key={m.id}
                m={m}
                onOpen={() => setSelected(m.id)}
                onLeave={() => void act(() => leaveMeeting(m.id))}
                onDelete={() => void act(() => deleteMeeting(m.id))}
              />
            ))}
          </ul>
        )}
        {total > PAGE && (
          <div className="flex items-center justify-end gap-2">
            <span className="md-label-medium" style={{ color: "var(--md-on-surface-variant)" }}>
              {page * PAGE + 1}–{Math.min(total, (page + 1) * PAGE)} of {total}
            </span>
            <Button variant="text" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
              ← Newer
            </Button>
            <Button variant="text" size="sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>
              Older →
            </Button>
          </div>
        )}
      </section>
    </div>
  );
}

function Detail({
  m,
  onLeave,
  onAnalyze,
  onDelete,
}: {
  m: Meeting;
  onLeave: () => void;
  onAnalyze: () => void;
  onDelete: () => void;
}) {
  const live = LIVE.includes(m.status);
  const ins = m.insights;
  const player = useRef<PlayerHandle>(null);
  const lines = m.transcript?.lines ?? [];
  const sections = ins?.sections ?? [];
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set());

  // Each section runs from its start line to just before the next one's.
  const ranges = sections.map((s, i) => ({
    ...s,
    end: i + 1 < sections.length ? sections[i + 1].start_line : lines.length,
  }));
  const seek = (t: number) => player.current?.seekTo(t, true);
  const highlights = ins?.highlights?.length ? ins.highlights : (ins?.key_points ?? []);

  return (
    <>
      <div className="md-card md-card-outlined space-y-3 p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="md-title-medium truncate">{ins?.title || m.title || "Meeting"}</h2>
            {m.platform === "bbb" || m.platform === "bbb_native" ? (
              <span className="md-label-small truncate" style={{ color: "var(--md-on-surface-variant)" }}>
                {where(m)}
                {m.platform === "bbb_native" ? " · lightweight" : ""}
                {m.talk ? " · talks" : ""}
              </span>
            ) : (
              <a
                href={m.url}
                target="_blank"
                rel="noreferrer"
                className="md-label-small truncate"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                {m.url}
              </a>
            )}
          </div>
          <Chip status={m.status} />
        </div>
        {(m.detail || m.error) && (
          <p className="md-body-small" style={{ color: m.error ? "var(--md-error)" : "var(--md-on-surface-variant)" }}>
            {m.error || m.detail}
          </p>
        )}
        <p className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
          {m.recorded_seconds > 0
            ? `${stamp(m.recorded_seconds)} recorded in ${m.segments} part${m.segments === 1 ? "" : "s"}`
            : "Nothing recorded yet"}
        </p>
        <div className="flex flex-wrap gap-2">
          {live && (
            <Button variant="tonal" size="sm" onClick={onLeave} disabled={m.leave_requested}>
              {m.status === "queued" ? "Cancel" : m.leave_requested ? "Leaving…" : "Leave the meeting"}
            </Button>
          )}
          {!live && m.segments > 0 && m.status !== "transcribing" && (
            <Button variant="outlined" size="sm" onClick={onAnalyze}>
              Run the analysis again
            </Button>
          )}
          {m.status !== "joining" && m.status !== "waiting" && m.status !== "in_meeting" && (
            <ConfirmButton
              label="Delete"
              title="Delete this meeting?"
              body="The recording, transcript and notes are removed. This cannot be undone."
              confirmLabel="Delete"
              onConfirm={onDelete}
            />
          )}
        </div>
        {m.segments > 0 && <Player ref={player} id={m.id} parts={m.segment_info} />}
      </div>

      {ins && (
        <div className="md-card md-card-outlined space-y-4 p-5">
          {ins.summary && !ins.highlights?.length && <p className="md-body-medium">{ins.summary}</p>}
          <List title="Highlights" items={highlights} />
          <List title="Decisions" items={ins.decisions} />
          {ins.action_items.length > 0 && (
            <div>
              <h3 className="md-title-small mb-1.5">Action items</h3>
              <ul className="space-y-1.5">
                {ins.action_items.map((a, i) => (
                  <li key={i} className="md-body-medium flex flex-wrap gap-x-2">
                    <span>{a.task}</span>
                    {(a.owner || a.due) && (
                      <span className="md-label-medium" style={{ color: "var(--md-on-surface-variant)" }}>
                        {[a.owner, a.due].filter(Boolean).join(" · ")}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <List title="Open questions" items={ins.open_questions} />
          {ins.topics.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {ins.topics.map((t) => (
                <span
                  key={t}
                  className="md-label-small rounded-[var(--md-shape-full)] px-2.5 py-1"
                  style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface-variant)" }}
                >
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {m.transcript && (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <h3 className="md-title-small">
              {ranges.length ? "The meeting, section by section" : "Transcript"}
            </h3>
            <span className="md-label-small" style={{ color: "var(--md-on-surface-variant)" }}>
              {m.transcript.words} words
              {ranges.length > 1 && (
                <>
                  {" · "}
                  <button
                    type="button"
                    className="underline"
                    onClick={() => setCollapsed(collapsed.size ? new Set() : new Set(ranges.map((_, i) => i)))}
                  >
                    {collapsed.size ? "Expand all" : "Collapse all"}
                  </button>
                </>
              )}
            </span>
          </div>

          {ranges.length > 1 && (
            <nav aria-label="Sections" className="flex flex-wrap gap-1.5">
              {ranges.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => document.getElementById(`section-${m.id}-${i}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}
                  className="md-label-small md-state rounded-[var(--md-shape-full)] px-2.5 py-1"
                  style={{ background: "var(--md-surface-container-high)", color: "var(--md-on-surface-variant)" }}
                >
                  {s.title}
                </button>
              ))}
            </nav>
          )}

          {ranges.length === 0 ? (
            <ol className="md-card md-card-outlined space-y-2 p-5">
              {lines.map((l, i) => (
                <Line key={i} start={l.start} text={l.text} onSeek={seek} playable={m.segments > 0} />
              ))}
            </ol>
          ) : (
            ranges.map((s, i) => {
              const open = !collapsed.has(i);
              const body = lines.slice(s.start_line, s.end);
              return (
                <section
                  key={i}
                  id={`section-${m.id}-${i}`}
                  className="md-card md-card-outlined scroll-mt-4 overflow-hidden"
                >
                  <button
                    type="button"
                    aria-expanded={open}
                    onClick={() => {
                      const next = new Set(collapsed);
                      if (open) next.add(i);
                      else next.delete(i);
                      setCollapsed(next);
                    }}
                    className="md-state flex w-full items-start gap-3 px-5 py-4 text-left"
                  >
                    <span className="md-label-medium mt-0.5 shrink-0 tabular-nums" style={{ color: "var(--md-primary)" }}>
                      {stamp(lines[s.start_line]?.start ?? 0)}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="md-title-small block">{s.title}</span>
                      {s.gist && (
                        <span className="md-body-small mt-0.5 block" style={{ color: "var(--md-on-surface-variant)" }}>
                          {s.gist}
                        </span>
                      )}
                    </span>
                    <span aria-hidden className="mt-1 shrink-0" style={{ color: "var(--md-on-surface-variant)" }}>
                      {open ? "▾" : "▸"}
                    </span>
                  </button>
                  {open && (
                    <div className="space-y-3 px-5 pb-5">
                      {s.points.length > 0 && (
                        <ul
                          className="list-disc space-y-1 rounded-[var(--md-shape-md)] py-3 pl-8 pr-4"
                          style={{ background: "var(--md-surface-container)" }}
                        >
                          {s.points.map((p, k) => (
                            <li key={k} className="md-body-medium">
                              {p}
                            </li>
                          ))}
                        </ul>
                      )}
                      <ol className="space-y-1.5">
                        {body.map((l, k) => (
                          <Line key={k} start={l.start} text={l.text} onSeek={seek} playable={m.segments > 0} />
                        ))}
                      </ol>
                    </div>
                  )}
                </section>
              );
            })
          )}
        </div>
      )}
    </>
  );
}

/** One line of the transcript. The time seeks the recording to it. */
function Line({
  start,
  text,
  onSeek,
  playable,
}: {
  start: number;
  text: string;
  onSeek: (t: number) => void;
  playable: boolean;
}) {
  return (
    <li className="md-body-small flex gap-3">
      <button
        type="button"
        disabled={!playable}
        onClick={() => onSeek(start)}
        title="Play from here"
        className="md-state w-12 shrink-0 rounded-[var(--md-shape-sm)] text-left tabular-nums disabled:cursor-default"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        {stamp(start)}
      </button>
      <span>{text}</span>
    </li>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className="md-title-small mb-1.5">{title}</h3>
      <ul className="list-disc space-y-1 pl-5">
        {items.map((t, i) => (
          <li key={i} className="md-body-medium">
            {t}
          </li>
        ))}
      </ul>
    </div>
  );
}

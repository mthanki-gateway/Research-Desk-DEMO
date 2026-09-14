"use client";

import { useEffect, useState } from "react";
import type { Chunk, Document, SessionDetail } from "@/lib/api";
import {
  Checkbox,
  Chip,
  ConfirmButton,
  IconButton,
  Ripplable,
  Switch,
} from "../../md";
import {
  IconChevron,
  IconClose,
  IconDocument,
  IconQuote,
  IconTrash,
} from "../../icons";
import type { TurnSettings } from "@/lib/prefs";

// Defined in lib/prefs.ts, next to the defaults and the validation that reads
// stored values back. Re-exported here so existing importers are unaffected --
// the rail is where every consumer already looks for this type.
export type { TurnSettings };

/** Rail widths, shared so the page's padding animation matches exactly. */
export const RAIL_WIDTH = "20rem";
export const RAIL_WIDTH_COLLAPSED = "5rem";

/**
 * Collapsed rail — an M3 **navigation rail**, which is exactly the component
 * for a narrow persistent side strip.
 *
 * It keeps the state visible in miniature rather than hiding it: losing sight
 * of what is being searched was the problem the rail was built to solve, so
 * collapsing reclaims width without reclaiming the information.
 */
function CollapsedRail({
  session,
  documents,
  settings,
  visible,
  onExpand,
}: {
  session: SessionDetail;
  documents: Document[];
  settings: TurnSettings;
  visible: boolean;
  onExpand: () => void;
}) {
  const scoped = session.document_ids.length > 0;
  const docCount = scoped ? session.document_ids.length : documents.length;
  const scopeTitle = scoped
    ? session.document_ids
        .map((d) => documents.find((r) => r.id === d)?.filename ?? "unknown")
        .join(", ")
    : `All ${documents.length} document${documents.length === 1 ? "" : "s"}`;

  return (
    <aside
      className={`fixed inset-y-0 right-0 z-30 hidden flex-col items-center gap-2 py-5 ${
        visible ? "lg:flex" : ""
      }`}
      style={{
        width: RAIL_WIDTH_COLLAPSED,
        background: "var(--md-surface)",
        borderLeft: "1px solid var(--md-outline-variant)",
      }}
    >
      <IconButton onClick={onExpand} aria-label="Expand panel">
        <IconChevron className="h-5 w-5 rotate-180" />
      </IconButton>

      <RailStat
        title={`Documents searched: ${scopeTitle}`}
        onClick={onExpand}
        active={scoped}
        icon={<IconDocument className="h-5 w-5" />}
        label={String(docCount)}
      />

      <RailStat
        title={`Passages per query: ${settings.topK}`}
        onClick={onExpand}
        label={`k${settings.topK}`}
      />

      {settings.multiQuery && (
        <RailStat title="Multi-query is on" onClick={onExpand} active label="MQ" />
      )}

      {settings.clarify && (
        <RailStat
          title="Clarifying questions are on"
          onClick={onExpand}
          active
          label="ASK"
        />
      )}

      {settings.react && (
        <RailStat
          title="Research mode: searches your documents and the web"
          onClick={onExpand}
          active
          label="TOOLS"
        />
      )}

      {session.summary && (
        <RailStat
          title={`Memory: ${session.summarised_upto} messages compressed`}
          onClick={onExpand}
          icon={<IconQuote className="h-5 w-5" />}
          label={String(session.summarised_upto)}
        />
      )}
    </aside>
  );
}

/** A navigation-rail destination: icon over a label, in a 56px pill target. */
function RailStat({
  title,
  onClick,
  icon,
  label,
  active = false,
}: {
  title: string;
  onClick: () => void;
  icon?: React.ReactNode;
  label: string;
  active?: boolean;
}) {
  return (
    <Ripplable
      as="div"
      title={title}
      onClick={onClick}
      role="button"
      tabIndex={0}
      className="flex w-14 cursor-pointer flex-col items-center gap-1 rounded-[var(--md-shape-lg)] py-2.5"
      style={{
        background: active ? "var(--md-secondary-container)" : "transparent",
        color: active
          ? "var(--md-on-secondary-container)"
          : "var(--md-on-surface-variant)",
      }}
    >
      {icon}
      <span className="md-label-small tabular-nums">{label}</span>
    </Ripplable>
  );
}

export default function Rail({
  session,
  documents,
  settings,
  chunk,
  open,
  collapsed,
  animate,
  onClose,
  onCollapse,
  onToggleDoc,
  onSettings,
  onClearChunk,
  onDeleteSession,
}: {
  session: SessionDetail;
  documents: Document[];
  settings: TurnSettings;
  chunk: Chunk | null;
  open: boolean;
  collapsed: boolean;
  /** False until the persisted collapse state is known, so the rail settles
   *  into position without sliding on first paint. */
  animate: boolean;
  onClose: () => void;
  onCollapse: (v: boolean) => void;
  onToggleDoc: (id: string) => Promise<void>;
  onSettings: (s: TurnSettings) => void;
  onClearChunk: () => void;
  onDeleteSession: () => Promise<void>;
}) {
  const [tab, setTab] = useState<"controls" | "source">("controls");

  // A citation click should bring its passage forward without the user having
  // to find the tab, and should re-open the rail if it was collapsed.
  useEffect(() => {
    if (chunk) {
      setTab("source");
      onCollapse(false);
    }
    // onCollapse is stable, so this fires only on a new chunk
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunk]);

  return (
    <>
      {open && <div onClick={onClose} className="md-scrim z-30 lg:hidden" />}

      <CollapsedRail
        session={session}
        documents={documents}
        settings={settings}
        visible={collapsed}
        onExpand={() => onCollapse(false)}
      />

      {/*
        The full panel stays mounted and slides out when collapsed, so the
        transition animates rather than the element vanishing. It sits above
        the navigation rail, which is revealed underneath as it leaves.
      */}
      <aside
        className={`fixed inset-y-0 right-0 z-40 flex flex-col ${
          animate ? "transition-transform" : ""
        } ${open ? "translate-x-0" : "translate-x-full"} ${
          collapsed ? "lg:translate-x-full" : "lg:translate-x-0"
        }`}
        style={{
          width: RAIL_WIDTH,
          background: "var(--md-surface)",
          borderLeft: "1px solid var(--md-outline-variant)",
          transitionDuration: "var(--md-dur-medium)",
          transitionTimingFunction: "var(--md-ease-emphasized)",
        }}
      >
        <div className="flex h-14 items-center justify-between px-2">
          <span className="md-title-small px-2">Conversation</span>
          <span className="flex items-center">
            <IconButton
              onClick={() => onCollapse(true)}
              className="hidden lg:inline-grid"
              aria-label="Collapse panel"
            >
              <IconChevron className="h-5 w-5" />
            </IconButton>
            <IconButton
              onClick={onClose}
              className="lg:hidden"
              aria-label="Close panel"
            >
              <IconClose />
            </IconButton>
          </span>
        </div>

        {/* Real <button>s, not divs with role="tab". A focusable div shows the
            focus ring on mouse click; a button only shows it for keyboard
            users, which is the behaviour we want. */}
        <div className="md-tabs" role="tablist">
          <Ripplable
            as="button"
            type="button"
            className="md-tab"
            data-active={tab === "controls"}
            onClick={() => setTab("controls")}
            role="tab"
            aria-selected={tab === "controls"}
          >
            Controls
          </Ripplable>
          <Ripplable
            as="button"
            type="button"
            className="md-tab"
            data-active={tab === "source"}
            onClick={() => setTab("source")}
            role="tab"
            aria-selected={tab === "source"}
          >
            Source
            {chunk && (
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--md-primary)" }}
              />
            )}
          </Ripplable>
        </div>

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {tab === "controls" ? (
            <Controls
              session={session}
              documents={documents}
              settings={settings}
              onToggleDoc={onToggleDoc}
              onSettings={onSettings}
              onDeleteSession={onDeleteSession}
            />
          ) : (
            <Source chunk={chunk} onClear={onClearChunk} />
          )}
        </div>
      </aside>
    </>
  );
}

function SectionHeading({
  children,
  meta,
}: {
  children: React.ReactNode;
  meta?: React.ReactNode;
}) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-2">
      <h3 className="md-label-large" style={{ color: "var(--md-primary)" }}>
        {children}
      </h3>
      {meta && (
        <span
          className="md-label-small tabular-nums"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {meta}
        </span>
      )}
    </div>
  );
}

function Controls({
  session,
  documents,
  settings,
  onToggleDoc,
  onSettings,
  onDeleteSession,
}: {
  session: SessionDetail;
  documents: Document[];
  settings: TurnSettings;
  onToggleDoc: (id: string) => Promise<void>;
  onSettings: (s: TurnSettings) => void;
  onDeleteSession: () => Promise<void>;
}) {
  const scoped = session.document_ids.length > 0;

  return (
    <div className="space-y-7 p-5">
      <section>
        <SectionHeading
          meta={
            scoped
              ? `${session.document_ids.length} of ${documents.length}`
              : "all"
          }
        >
          Documents searched
        </SectionHeading>

        {documents.length === 0 ? (
          <p
            className="md-body-small"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            No indexed documents.
          </p>
        ) : (
          <ul className="space-y-2.5">
            {documents.map((d) => {
              const explicit = session.document_ids.includes(d.id);
              return (
                <li key={d.id}>
                  <Ripplable
                    as="div"
                    onClick={() => void onToggleDoc(d.id)}
                    role="checkbox"
                    aria-checked={explicit}
                    tabIndex={0}
                    className="flex cursor-pointer items-start gap-3 rounded-[var(--md-shape-md)] p-3"
                    style={{
                      background: explicit
                        ? "var(--md-secondary-container)"
                        : "var(--md-surface-container-high)",
                      color: explicit
                        ? "var(--md-on-secondary-container)"
                        : "var(--md-on-surface)",
                      opacity: scoped && !explicit ? 0.6 : 1,
                    }}
                    title={
                      explicit
                        ? "Remove from scope"
                        : scoped
                          ? "Add to scope"
                          : "Restrict the search to this document"
                    }
                  >
                    <span className="mt-0.5">
                      <Checkbox on={explicit} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="md-body-medium block truncate">
                        {d.filename}
                      </span>
                      <span
                        className="md-body-small block tabular-nums"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        {d.n_chunks} chunks
                      </span>
                    </span>
                  </Ripplable>
                </li>
              );
            })}
          </ul>
        )}

        <p
          className="md-body-small mt-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {scoped
            ? "Only the selected documents are searched."
            : "Nothing selected, so every document is searched."}
        </p>
      </section>

      <hr className="md-divider" />

      <section>
        <SectionHeading>Retrieval</SectionHeading>

        <div className="space-y-5">
          <div className="flex items-center justify-between gap-3">
            <span className="md-body-medium">
              Passages per query
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Lower is stricter
              </span>
            </span>
            <span className="md-segmented shrink-0">
              <Ripplable
                as="button"
                onClick={() =>
                  onSettings({
                    ...settings,
                    topK: Math.max(1, settings.topK - 1),
                  })
                }
                aria-label="Decrease"
              >
                &minus;
              </Ripplable>
              {/* A VALUE, not a selected segment.
                  It was a `<button data-active="true">` with pointer events
                  switched off, which gave it the filled "this segment is
                  chosen" treatment -- a solid block running the full 40px
                  height, so it met the pill's 1px outline from the inside and
                  read as spilling over it. Nothing here is chosen; it is the
                  number the two buttons change.

                  Being a real button also meant it was still focusable and
                  announced as one, so tabbing through the stepper stopped on
                  a control that could not be activated. */}
              <span
                className="md-segmented-value"
                aria-live="polite"
                aria-label={`${settings.topK} passages per query`}
              >
                {settings.topK}
              </span>
              <Ripplable
                as="button"
                onClick={() =>
                  onSettings({
                    ...settings,
                    topK: Math.min(20, settings.topK + 1),
                  })
                }
                aria-label="Increase"
              >
                +
              </Ripplable>
            </span>
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="md-body-medium">
              Multi-query
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Rephrase and fuse — one extra call
              </span>
            </span>
            <Switch
              on={settings.multiQuery}
              onChange={(v) => onSettings({ ...settings, multiQuery: v })}
              aria-label="Multi-query"
            />
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="md-body-medium">
              Ask if unclear
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Offer options when a question is vague
              </span>
            </span>
            <Switch
              on={settings.clarify}
              onChange={(v) => onSettings({ ...settings, clarify: v })}
              aria-label="Ask if unclear"
            />
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="md-body-medium">
              Research mode
              <span
                className="md-body-small mt-0.5 block"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                Searches your documents and the web. Off restricts
                answers to your documents only.
              </span>
            </span>
            <Switch
              on={settings.react}
              onChange={(v) => onSettings({ ...settings, react: v })}
              aria-label="Research mode"
            />
          </div>

          {/* DEVELOPMENT ONLY.
              Gated on NODE_ENV rather than hidden with CSS, so the whole block
              is dead-code-eliminated from a production bundle -- the same
              treatment the accent picker gets. A shipped product does not let
              the browser choose which model answers; this exists because the
              answer model's 5 rpm makes the agent loop impractical to iterate
              on, and Gemma's 30 rpm is the only budget that can. */}
          {process.env.NODE_ENV !== "production" && (
            <div
              className="mt-1 border-t pt-3"
              style={{ borderColor: "var(--md-outline-variant)" }}
            >
              <span className="md-body-medium">
                Model
                <span
                  className="md-body-small mt-0.5 block"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  Dev only. Gemma is far faster to iterate on (30/min vs 5/min)
                  but cannot use tools, so research mode falls back to planning.
                </span>
              </span>
              <div className="mt-2 flex flex-wrap gap-2">
                {(
                  [
                    [null, "Server default"],
                    ["gemini", "Gemini"],
                    ["gemma", "Gemma"],
                  ] as const
                ).map(([value, label]) => (
                  <Chip
                    key={label}
                    size="sm"
                    selected={settings.modelProfile === value}
                    onClick={() => onSettings({ ...settings, modelProfile: value })}
                  >
                    {label}
                  </Chip>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {session.summary && (
        <>
          <hr className="md-divider" />
          <section>
            <SectionHeading meta={`${session.summarised_upto} compressed`}>
              Memory
            </SectionHeading>
            <p
              className="md-body-small md-quote"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              {session.summary}
            </p>
            <p
              className="md-body-small mt-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Older exchanges, compressed by the model and sent with each
              question alongside the last three verbatim.
            </p>
          </section>
        </>
      )}

      <hr className="md-divider" />

      <section>
        <ConfirmButton
          label="Delete session"
          title="Delete this session?"
          body="The conversation and all its messages will be permanently removed. This cannot be undone."
          onConfirm={onDeleteSession}
          icon={<IconTrash className="h-4 w-4" />}
          className="w-full"
        />
      </section>
    </div>
  );
}

function Source({
  chunk,
  onClear,
}: {
  chunk: Chunk | null;
  onClear: () => void;
}) {
  if (!chunk) {
    return (
      <div className="p-6 text-center">
        <span
          className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
          style={{
            background: "var(--md-surface-container-highest)",
            color: "var(--md-on-surface-variant)",
          }}
        >
          <IconQuote className="h-6 w-6" />
        </span>
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Select a citation under any answer to read the passage it came from.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="md-title-small truncate">
            {chunk.heading
              ? chunk.heading.replace(/^#+\s*/, "")
              : `Chunk ${chunk.chunk_index}`}
          </p>
          <p
            className="md-body-small mt-0.5 flex items-center gap-1 truncate"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            <IconDocument className="h-3.5 w-3.5 shrink-0" />
            {chunk.filename}
            {chunk.page !== null && ` · p.${chunk.page}`}
          </p>
        </div>
        <IconButton size="sm" onClick={onClear} aria-label="Clear">
          <IconClose className="h-4 w-4" />
        </IconButton>
      </div>

      <div className="mb-3 flex gap-2">
        <span className="md-badge">chunk {chunk.chunk_index}</span>
        <span className="md-badge">{chunk.n_chars} chars</span>
      </div>

      <p className="md-body-medium md-quote">
        {chunk.text}
      </p>

      <p
        className="md-body-small mt-3"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        Verbatim text sent to the model, including the heading prefix added at
        ingestion.
      </p>
    </div>
  );
}

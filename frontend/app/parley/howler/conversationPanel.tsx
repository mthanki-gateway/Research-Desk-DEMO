"use client";

import { useEffect, useState } from "react";
import { getParleyConversation } from "@/lib/api";
import { Button } from "../../md";
import { IconChevron, IconSpinner } from "../../icons";

/**
 * The conversation, opened where it is rather than on another page.
 *
 * WHY IN PLACE
 *
 * Reading a transcript is something you do WHILE comparing results -- against
 * the profile above it, against the next participant below it. Navigating away
 * threw away that position: you came back to the top of the Results tab with
 * no memory of which of five interviews you had been reading. Expanding keeps
 * the place you are in visible the whole time, which is the point of having a
 * results tab rather than a list of links.
 *
 * FETCHED ON FIRST OPEN, then kept. Five interviews on a page would otherwise
 * be five transcript requests nobody asked for, and closing and reopening one
 * is not a reason to ask again.
 */
export default function ConversationPanel({
  sessionId,
  turns,
}: {
  sessionId: string;
  /** From the result row, so the label can be honest before anything loads. */
  turns: number;
}) {
  const [open, setOpen] = useState(false);
  const [lines, setLines] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || lines !== null || loading) return;
    setLoading(true);
    getParleyConversation(sessionId)
      .then((d) =>
        setLines(d.turns.map((t) => t.answer).filter((a): a is string => !!a)),
      )
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not load it"),
      )
      .finally(() => setLoading(false));
  }, [open, lines, loading, sessionId]);

  return (
    <div className="mt-2">
      <Button
        variant="text"
        size="sm"
        onClick={() => setOpen((w) => !w)}
        aria-expanded={open}
      >
        <IconChevron className="h-4 w-4" open={open} />
        {open ? "Hide the conversation" : "Read the conversation"}
        <span style={{ color: "var(--md-on-surface-variant)" }}>
          {turns} turn{turns === 1 ? "" : "s"}
        </span>
      </Button>

      {open && (
        <div className="mt-2">
          {loading && (
            <p
              className="md-body-small flex items-center gap-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              <IconSpinner className="h-3.5 w-3.5" />
              Loading
            </p>
          )}

          {error && (
            <p
              className="md-body-small rounded-[var(--md-shape-sm)] px-3 py-2"
              style={{
                background: "var(--md-error-container)",
                color: "var(--md-on-error-container)",
              }}
            >
              {error}
            </p>
          )}

          {/* THE INTERVIEWER'S SIDE ONLY, as everywhere else. The
              participant's transcript is a separate, lossier pass -- it
              rendered somebody saying their name was John as "madre es un" --
              and a wrong transcript beside a right answer reads as
              authoritative. What they said survives accurately in the notes
              and quotes above, recorded by the thing that heard it. */}
          {lines && (
            <ol
              className="md-scroll max-h-96 space-y-2 overflow-y-auto rounded-[var(--md-shape-md)] p-3"
              style={{ background: "var(--md-surface-container-high)" }}
            >
              {lines.map((line, i) => (
                <li key={i}>
                  <p
                    className="md-label-small"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    Interviewer
                  </p>
                  <p className="md-body-small whitespace-pre-wrap">{line}</p>
                </li>
              ))}
              {lines.length === 0 && (
                <li
                  className="md-body-small"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  Nothing was said in this conversation.
                </li>
              )}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}

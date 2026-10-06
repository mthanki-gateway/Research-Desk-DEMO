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
 * FETCHED WHEN OPEN, then refreshed while post-call transcription is running.
 * The saved turns begin with live captions, then the speech-to-text pass
 * replaces the participant's side when it completes.
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
  const [conversation, setConversation] = useState<
    Awaited<ReturnType<typeof getParleyConversation>> | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let stopped = false;
    let timer: number | undefined;

    async function refresh(showSpinner: boolean) {
      if (showSpinner) setLoading(true);
      try {
        const latest = await getParleyConversation(sessionId);
        if (stopped) return;
        setConversation(latest);
        setError(null);
        if (
          latest.transcription?.status === "queued" ||
          latest.transcription?.status === "running"
        ) {
          timer = window.setTimeout(() => void refresh(false), 2000);
        }
      } catch (e) {
        if (!stopped)
          setError(e instanceof Error ? e.message : "Could not load it");
      } finally {
        if (showSpinner && !stopped) setLoading(false);
      }
    }

    void refresh(true);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [open, sessionId]);

  const transcription = conversation?.transcription;

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

          {transcription?.status === "queued" || transcription?.status === "running" ? (
            <p
              className="md-body-small mb-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Transcribing participant turns. Live captions are shown until the
              cleaner transcript is ready.
            </p>
          ) : transcription?.status === "failed" ? (
            <p
              className="md-body-small mb-2"
              style={{ color: "var(--md-error)" }}
            >
              Transcription failed; showing the live captions.
              {transcription.error ? ` ${transcription.error}` : ""}
            </p>
          ) : transcription?.status === "done" && transcription.result.engine ? (
            <p
              className="md-body-small mb-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              Participant turns transcribed with {String(transcription.result.engine)}.
            </p>
          ) : null}

          {/* The participant's clean post-call transcript appears here beside
              the interviewer's response. While that pass is pending, it is
              refreshed from the live captions and replaced on completion. */}
          {conversation && (
            <ol
              className="md-scroll max-h-96 space-y-2 overflow-y-auto rounded-[var(--md-shape-md)] p-3"
              style={{ background: "var(--md-surface-container-high)" }}
            >
              {conversation.turns.map((turn, i) => (
                <li
                  key={i}
                  className="space-y-3 rounded-[var(--md-shape-sm)] px-3 py-2"
                  style={{ background: "var(--md-surface)" }}
                >
                  {turn.question && (
                    <div>
                      <p
                        className="md-label-small"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        Participant · turn {i + 1}
                      </p>
                      <p className="md-body-small whitespace-pre-wrap">{turn.question}</p>
                    </div>
                  )}
                  {turn.answer && (
                    <div>
                      <p
                        className="md-label-small"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        Interviewer
                      </p>
                      <p className="md-body-small whitespace-pre-wrap">{turn.answer}</p>
                    </div>
                  )}
                </li>
              ))}
              {conversation.turns.length === 0 && (
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

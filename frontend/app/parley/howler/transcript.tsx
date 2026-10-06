"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getParleyConversation,
  renameParleyConversation,
} from "@/lib/api";
import { Button, IconButton, TextArea } from "../../md";
import { IconChevron, IconEdit } from "../../icons";
import { ProfileCard } from "../surface";

/**
 * Reading what came back, with nothing that could change it.
 *
 * WHY THIS IS NOT `ParleySurface`
 *
 * Opening a finished interview used to hand you the live surface: a
 * microphone, a voice picker, a keep-the-socket-open toggle and a New
 * conversation button. Every one of those is a control for HOLDING an
 * interview, offered to somebody who came to READ one. The voice picker is the
 * clearest tell -- it configures a call that already happened.
 *
 * So this is a separate page rather than a mode of that one. The two have
 * genuinely different jobs, and the shared part -- the profile card -- is
 * imported rather than duplicated.
 *
 * It is also where a conversation gets RENAMED. A link made for a named person
 * carries that name, but an auto-made one has nothing to go on, and the
 * participant's name often lands in a note rather than a field where the
 * automatic renamer could find it. Somebody reading the result is the first
 * person who actually knows what it should be called.
 */
export default function Transcript({
  id,
  projectId,
}: {
  id: string;
  projectId: string | null;
}) {
  const router = useRouter();
  const [data, setData] = useState<
    Awaited<ReturnType<typeof getParleyConversation>> | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;

    async function refresh() {
      try {
        const latest = await getParleyConversation(id);
        if (stopped) return;
        setData(latest);
        setError(null);
        if (
          latest.transcription?.status === "queued" ||
          latest.transcription?.status === "running"
        ) {
          timer = window.setTimeout(() => void refresh(), 2000);
        }
      } catch (e) {
        if (!stopped)
          setError(e instanceof Error ? e.message : "Could not open that");
      }
    }

    void refresh();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [id]);

  async function commitRename() {
    const title = draft.trim();
    setRenaming(false);
    if (!title || !data || title === data.title) return;
    // Optimistic: the request is a PATCH that either works or leaves the old
    // title, and making somebody watch a spinner to rename a row is worse than
    // the rare case of showing a name that did not save.
    setData({ ...data, title });
    try {
      await renameParleyConversation(id, title);
    } catch {
      /* the stored title stands; a reload shows it */
    }
  }

  const back = () =>
    router.push(
      data?.project_id || projectId
        ? `/parley/howler?p=${data?.project_id ?? projectId}&tab=results`
        : "/parley/howler",
    );

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-9">
        <p
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 px-6 py-9" aria-hidden>
        <div className="md-skeleton h-8 w-52" />
        <div className="md-skeleton h-64" />
      </div>
    );
  }

  const answered = data.turns.filter((t) => t.question || t.answer);

  return (
    <div className="mx-auto max-w-3xl px-6 py-6">
      <header className="sticky top-0 z-20 mb-5 -mx-6 flex h-16 items-center gap-2 px-6"
        style={{
          background: "var(--md-surface-container-low)",
          boxShadow: "0 1px 0 0 var(--md-outline-variant)",
        }}
      >
        <IconButton onClick={back} aria-label="Back to results" className="shrink-0">
          <IconChevron className="h-5 w-5 rotate-180" />
        </IconButton>

        {renaming ? (
          <TextArea
            label="Name"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => void commitRename()}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void commitRename();
              }
              if (e.key === "Escape") setRenaming(false);
            }}
            surface="var(--md-surface)"
            className="flex-1"
            autoComplete="off"
          />
        ) : (
          <>
            <h1 className="md-title-large min-w-0 flex-1 truncate">
              {data.title}
            </h1>
            <Button
              variant="text"
              size="sm"
              onClick={() => {
                setDraft(data.title);
                setRenaming(true);
              }}
              className="shrink-0"
            >
              <IconEdit />
              Rename
            </Button>
          </>
        )}
      </header>

      <ProfileCard
        fields={data.fields.map((f) => ({
          name: f.name,
          required: f.required,
          kind: f.type,
        }))}
        profile={data.profile}
        missing={data.missing}
        complete={data.complete}
      />

      <h2 className="md-title-small mb-3 mt-8">
        Conversation
        <span
          className="md-label-small ml-2"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          {answered.length} turn{answered.length === 1 ? "" : "s"}
        </span>
      </h2>

      {data.transcription?.status === "queued" || data.transcription?.status === "running" ? (
        <p
          className="md-body-small mb-3"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Transcribing participant turns. Live captions are shown until the
          cleaner transcript is ready.
        </p>
      ) : data.transcription?.status === "failed" ? (
        <p className="md-body-small mb-3" style={{ color: "var(--md-error)" }}>
          Transcription failed; showing the live captions.
          {data.transcription.error ? ` ${data.transcription.error}` : ""}
        </p>
      ) : null}

      <ol className="space-y-3">
        {answered.map((turn, i) => (
          <li key={i} className="md-card md-card-outlined space-y-3 p-5">
            {turn.question && (
              <div>
                <p
                  className="md-label-medium"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  Participant · turn {i + 1}
                </p>
                <p className="md-body-medium whitespace-pre-wrap">{turn.question}</p>
              </div>
            )}
            {turn.answer && (
              <div>
                <p
                  className="md-label-medium"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  Interviewer
                </p>
                <p className="md-body-medium whitespace-pre-wrap">{turn.answer}</p>
              </div>
            )}
          </li>
        ))}
        {answered.length === 0 && (
          <li
            className="md-body-medium py-8 text-center"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Nothing was said in this conversation.
          </li>
        )}
      </ol>
    </div>
  );
}

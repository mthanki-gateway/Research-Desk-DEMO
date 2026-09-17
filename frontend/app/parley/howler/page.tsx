"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { type BlueprintField, createHowl, previewHowl } from "@/lib/api";
import { useApp } from "../../providers";
import { TextArea } from "../../md";
import { IconHowler, IconSpinner } from "../../icons";
import ParleySurface from "../surface";

/**
 * Howler — an interview whose data points are written by whoever runs it.
 *
 * Speak and Interview have their fields in Python. Here a brief is turned into
 * a schema, once, and frozen on the conversation. Everything downstream --
 * the tool declaration, the completeness check, the card -- reads that schema
 * instead of a constant.
 *
 * THE SETUP IS A SEPARATE STEP, not a dialog over the microphone. A brief is a
 * loose instruction and the schema is a specific list, and the moment it
 * becomes specific is exactly the moment worth reading before anyone speaks.
 * `previewHowl` exists for the same reason: adjusting a brief should not leave
 * a trail of abandoned conversations in the drawer.
 */
export default function HowlerPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-3xl space-y-6 px-6 py-9" aria-hidden>
          <div className="md-skeleton h-8 w-40" />
          <div className="md-skeleton h-[22rem]" />
        </div>
      }
    >
      <Howler />
    </Suspense>
  );
}

function Howler() {
  const params = useSearchParams();
  // A conversation in the URL means setup is done; hand over to the surface.
  if (params.get("c")) return <ParleySurface mode="howler" />;
  return <Setup />;
}

function Setup() {
  const router = useRouter();
  const { refreshParleyConversations } = useApp();
  const [brief, setBrief] = useState("");
  const [participant, setParticipant] = useState("");
  const [fields, setFields] = useState<BlueprintField[] | null>(null);
  const [busy, setBusy] = useState<"preview" | "create" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function preview() {
    setBusy("preview");
    setError(null);
    try {
      setFields(await previewHowl(brief));
    } catch (e) {
      setFields(null);
      setError(e instanceof Error ? e.message : "Could not read that brief");
    } finally {
      setBusy(null);
    }
  }

  async function begin() {
    setBusy("create");
    setError(null);
    try {
      const made = await createHowl(brief, participant);
      await refreshParleyConversations("howler");
      router.push(`/parley/howler?c=${made.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start that session");
      setBusy(null);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small flex items-center gap-2">
          <IconHowler className="h-6 w-6" />
          Howler
        </h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Say what you want to find out. It works out the data points, then
          interviews someone until it has them.
        </p>
      </header>

      {error && (
        <p
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      <section className="md-card md-card-outlined space-y-4 p-5">
        <TextArea
          label="What do you want to find out?"
          value={brief}
          onChange={(e) => setBrief(e.target.value)}
          rows={5}
          placeholder="Qualify inbound leads. Find out their budget, when they want to go live, who signs off, and what they use today. If possible, what made them reach out now."
        />
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Written as instruction, not as a list. Anything you call &ldquo;if
          possible&rdquo; becomes optional.
        </p>

        <TextArea
          label="Who is being interviewed? (optional)"
          value={participant}
          onChange={(e) => setParticipant(e.target.value)}
          rows={3}
          placeholder="Priya Raman, Head of Data at a logistics company. Filled in the pricing form last week. Technical, short on time."
        />
        <p
          className="md-body-small"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Context for the interviewer — what to skip, what to press on. It is
          never read back to them, and what they say always wins over it.
        </p>

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <button
            type="button"
            onClick={() => void preview()}
            disabled={!brief.trim() || busy !== null}
            className="md-label-large rounded-[var(--md-shape-full)] px-4 py-2 disabled:opacity-50"
            style={{
              background: "var(--md-surface-container-high)",
              color: "var(--md-on-surface)",
            }}
          >
            {busy === "preview" ? "Reading" : fields ? "Read again" : "Preview fields"}
          </button>
          <button
            type="button"
            onClick={() => void begin()}
            disabled={!brief.trim() || busy !== null}
            className="md-label-large flex items-center gap-2 rounded-[var(--md-shape-full)] px-5 py-2 disabled:opacity-50"
            style={{
              background: "var(--md-primary)",
              color: "var(--md-on-primary)",
            }}
          >
            {busy === "create" && <IconSpinner className="h-4 w-4" />}
            Start session
          </button>
        </div>
      </section>

      {fields && (
        <section className="md-card md-card-outlined p-5">
          <h2 className="md-title-medium mb-1">
            {fields.length} data point{fields.length === 1 ? "" : "s"}
          </h2>
          <p
            className="md-body-small mb-3"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {/* Said plainly, because it is the thing most likely to surprise:
                these are fixed once the session starts. A moving target cannot
                be completed, so an interview against one never ends. */}
            Fixed for the life of the session. Anything they say that these do
            not cover is still captured, as a note.
          </p>
          <ul className="space-y-3">
            {fields.map((f) => (
              <li key={f.name}>
                <p className="md-body-medium flex items-baseline gap-2">
                  <span className="font-medium">{f.label}</span>
                  <span
                    className="md-label-small"
                    style={{ color: "var(--md-on-surface-variant)" }}
                  >
                    {f.type.toLowerCase()}
                    {f.required ? " · required" : " · optional"}
                  </span>
                </p>
                <p
                  className="md-body-small"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {f.description}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import type { ClarifyDecision, InterruptEvent } from "@/lib/api";
import { Button, Ripplable } from "../../md";
import { IconSpinner } from "../../icons";

/**
 * The agent asking what the user meant.
 *
 * Shaped as **prefilled options plus a free-text box**: the model offers 2–4
 * concrete aspects drawn from the headings the documents actually have, and
 * "Something else" is there for when none of them fit. That last part is not a
 * courtesy — an option list is the model's guess at the space of answers, and
 * it will sometimes be wrong in a way the user can fix in four words.
 *
 * Nothing has been planned or retrieved at this point. The turn exists only as
 * a checkpoint keyed by `interrupt.thread_id`, which is why that id has to
 * survive the round trip.
 */
export default function Clarify({
  interrupt,
  busy,
  onDecide,
}: {
  interrupt: InterruptEvent;
  busy: boolean;
  onDecide: (decision: ClarifyDecision) => void;
}) {
  // Non-null once "Something else" is chosen. Separate from the option list
  // rather than a magic option value, so an empty box can't be submitted as if
  // it were a choice.
  const [custom, setCustom] = useState<string | null>(null);
  const box = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (custom !== null) box.current?.focus();
  }, [custom]);

  const typed = (custom ?? "").trim();

  return (
    <li
      className="rounded-[var(--md-shape-lg)] p-5"
      style={{
        background: "var(--md-surface)",
        border: "1px solid var(--md-outline-variant)",
      }}
      aria-live="polite"
    >
      <div className="flex items-start gap-2.5">
        <span
          className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full"
          style={{
            background: "var(--md-tertiary-container)",
            color: "var(--md-on-tertiary-container)",
          }}
          aria-hidden="true"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.4}
            strokeLinecap="round"
            className="h-3.5 w-3.5"
          >
            <path d="M9.5 9a2.5 2.5 0 1 1 3.6 2.2c-.7.4-1.1 1-1.1 1.8v.5M12 17h.01" />
          </svg>
        </span>
        <h2 className="md-title-medium">{interrupt.question}</h2>
      </div>

      <p
        className="md-body-small mt-1.5 pl-[2.125rem]"
        style={{ color: "var(--md-on-surface-variant)" }}
      >
        Paused before searching — pick one, or say what you're after.
      </p>

      <div className="mt-4 space-y-2">
        {interrupt.options.map((o) => (
          <Ripplable
            key={o.label}
            as="button"
            type="button"
            disabled={busy}
            onClick={() => onDecide({ action: "answer", answer: o.label })}
            className="md-state flex w-full items-start gap-3 rounded-[var(--md-shape-md)] p-3 text-left disabled:opacity-50"
            style={{ border: "1px solid var(--md-outline-variant)" }}
          >
            <span className="min-w-0">
              <span className="md-body-medium block font-medium">{o.label}</span>
              {o.description && (
                <span
                  className="md-body-small block"
                  style={{ color: "var(--md-on-surface-variant)" }}
                >
                  {o.description}
                </span>
              )}
            </span>
          </Ripplable>
        ))}

        {/* The custom option expands IN PLACE into an input, rather than
            swapping the whole card for a form. The generated options stay on
            screen while you type -- which matters, because the usual reason to
            type is that one option is nearly right and you want to say how it
            differs. Replacing them made you remember what they were. */}
        {custom === null ? (
          <Ripplable
            as="button"
            type="button"
            disabled={busy}
            onClick={() => setCustom("")}
            className="md-state flex w-full items-start gap-3 rounded-[var(--md-shape-md)] p-3 text-left disabled:opacity-50"
            style={{
              border: "1px dashed var(--md-outline)",
              color: "var(--md-on-surface-variant)",
            }}
          >
            <span className="md-body-medium block font-medium">
              Something else…
            </span>
          </Ripplable>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (typed) onDecide({ action: "answer", answer: typed });
            }}
            // Dashed, matching the button it replaced, so the row does not
            // appear to jump to a different kind of control.
            className="flex items-center gap-2 rounded-[var(--md-shape-md)] p-1.5 pl-3"
            style={{ border: "1px dashed var(--md-outline)" }}
          >
            <input
              ref={box}
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              // Escape returns to the options rather than trapping you in the
              // input -- the same key that dismisses every other transient
              // thing in the app.
              onKeyDown={(e) => {
                if (e.key === "Escape") setCustom(null);
              }}
              disabled={busy}
              maxLength={500}
              placeholder="What do you want to know?"
              aria-label="What do you want to know?"
              className="md-body-medium min-w-0 flex-1 bg-transparent py-1.5 outline-none"
              style={{ color: "var(--md-on-surface)" }}
              autoComplete="off"
            />
            <Button type="submit" size="sm" disabled={busy || !typed}>
              {busy && <IconSpinner className="h-4 w-4" />}
              Search
            </Button>
          </form>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Button
          variant="text"
          size="sm"
          disabled={busy}
          onClick={() => onDecide({ action: "skip" })}
          title="Search the question exactly as you wrote it"
        >
          Search anyway
        </Button>
        <span className="flex-1" />
        <Button
          variant="text"
          size="sm"
          disabled={busy}
          onClick={() => onDecide({ action: "cancel" })}
        >
          Cancel
        </Button>
      </div>
    </li>
  );
}

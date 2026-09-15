"use client";

import ProviderPanel, { VAD } from "./panel";

/**
 * Two ASR vendors, side by side and fully independent.
 *
 * NEITHER IS STREAMING. Both endpoints are batch: a complete clip goes in, a
 * complete transcript comes back, with no partial result to subscribe to.
 * "Live" here is a sequence of short recordings, and every rough edge -- the
 * segment minimum, the boundary artefacts, the per-segment language detection
 * -- follows from that one fact rather than from the implementation.
 *
 * THE PANELS SHARE NOTHING. Each owns its microphone, settings and transcript,
 * so both can run at once: say something once and watch Whisper and NeMo
 * transcribe the same words. A shared recorder would have forced a choice
 * before speaking, making every comparison a comparison of two different
 * utterances.
 */
export default function Transcribe() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header>
        <h1 className="md-headline-small">Transcribe</h1>
        <p
          className="md-body-medium mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Two vendors, each with its own microphone and transcript — start both
          to compare them on the same speech. Segments are at least{" "}
          {VAD.MIN_SEGMENT_MS / 1000}s, because short clips are what these
          models guess on, then cut at the next pause. Stop transcribes whatever
          is buffered.
        </p>
      </header>

      <ProviderPanel provider="groq" />
      <ProviderPanel provider="nvidia" />
    </div>
  );
}

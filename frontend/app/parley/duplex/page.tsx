import DuplexSurface from "./surface";

/**
 * Parley's fourth door: speech in, speech out, with the answer started
 * before you finish. Unlike the other three it is NOT the native-audio model;
 * it is speech-to-text, a fast language model and a streaming voice, wired
 * the way LiveKit wires them.
 */
export default function DuplexPage() {
  return <DuplexSurface />;
}

# Model Lab

Two surfaces: **Playground**, **Transcribe**.

## The idea

Call open models directly and watch what they actually do — how long they take,
how many tokens they use, and how two vendors differ on the same input. No
retrieval, no citations, no conversation. Research Desk and Parley are
applications built *on* models; this is the models themselves, with the numbers
on screen.

## Notable decisions

**Playground is deliberately not a chat.** No history, no streaming, no
retrieval. Every one of those is a thing to build and debug that teaches
nothing about the model behind the API. What it shows instead is the part worth
seeing: latency, tokens in and out, and tokens per second.

**Transcribe runs two vendors side by side, sharing nothing.** Each panel owns
its own microphone, settings and transcript, so both can run at once — say
something once and watch Whisper and NeMo transcribe the same words. A shared
recorder would have forced a choice between them and made comparison
impossible.

**Neither ASR endpoint is streaming, and the UI admits it.** Both are batch: a
complete clip in, a complete transcript back, with nothing partial to subscribe
to. "Live" transcription here is a sequence of short recordings, and every
rough edge that follows — the segment minimum, boundary artefacts, per-segment
language detection — comes from that one fact rather than from the
implementation. Saying so is more useful than hiding it behind a spinner.

**A provider that is not configured is absent, not broken.** Missing keys
remove the panel rather than leaving a control that fails when pressed.

## The tech

| | |
|---|---|
| Text | Groq (`openai/gpt-oss-120b` by default), OpenAI-compatible API |
| Speech | Groq Whisper, NVIDIA NeMo (Parakeet / Canary) |
| Capture | `MediaRecorder` per panel, independent of Parley's PCM path |

## How it works

### Playground

One `POST` per press. The response carries the completion plus timing and token
counts, which the UI renders beside the text. Model list and availability come
from `playgroundStatus`, so the picker only ever offers what the deployment can
actually call.

### Transcribe

Each panel records a clip, posts it, and appends the returned transcript. A
segment minimum exists because very short clips transcribe badly — the model
has no context to work with — and the boundary between two clips is where words
get clipped, which is visible and deliberately not smoothed over.

This is a different audio path from Parley entirely: `MediaRecorder` producing
encoded clips, rather than raw PCM frames over a socket. The two exist side by
side because they answer different questions — "how good is this ASR vendor" and
"how fast can a native-audio model reply".

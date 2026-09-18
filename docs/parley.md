# Parley

Three voice apps: **Speak**, **Interview**, **Howler**.

## The idea

Talk to a model out loud and have it talk back, with no transcript in the
middle. Speak answers questions from your documents and the web. Interview
turns a conversation into a profile of the person. Howler does the same, except
you decide by conversation what the profile should contain, and send a link to
whoever you want interviewed.

## Notable decisions

**Audio to audio, not a cascade.** The first version transcribed speech, ran
the LangGraph agent, and synthesised the answer: 12–53 seconds to first sound.
The current one streams audio into a native-audio model and streams audio
back — about 3 seconds. The audio is tokenised into the same sequence the model
generates from; nothing is transcribed on the way in.

The cascade (`POST /voice/ask`) is **deliberately kept**. It is a working
reference implementation of the other architecture, and the measured gap
between them is the most instructive thing in this repo.

**The button owns the turn.** Automatic activity detection is switched off
(`AutomaticActivityDetection(disabled=True)`); a turn opens on `activity_start`
and closes on `activity_end`, both sent when you click. This exists because
people pause constantly while speaking — to think, to find a word, to check a
figure — and every one of those was read as "they have finished", so the model
talked over the second half of the question. Tuning the silence threshold only
moves the problem: short enough to feel responsive is short enough to
interrupt, long enough never to interrupt is long enough to feel broken.

**Turn detection is an opt-in on top of that, not a replacement.** See
[below](#turn-detection-speak-only).

**One pipeline, three prompts.** The socket, audio handling, turn boundaries,
tools, persistence and resumption are shared. The only difference between modes
is the system prompt (`live_prompts.py`) and the `kind` a conversation is
stored under. Adding a mode should be an entry in `live.MODES` plus a route. If
it needs more, something that ought to be shared has been duplicated.

**Tools are shared with the typed agent.** `live.py` builds its declarations
from `agent_tools.tool_specs()` and executes them through `agent_tools.run_tool`
— the same functions Research Desk uses. A second retrieval path would diverge,
and the divergence would show up as the voice app answering differently from
the chat app about the same document.

**The input transcript is not shown.** `input_transcription` is a separate,
lossier pass than the model's own understanding — it rendered a participant
saying their name was John as "madre es un", in a turn the model answered with
"Thanks, John". A wrong transcript beside a right answer is worse than none: it
reads as authoritative, and invites doubt about the half that was correct.

**Two things fail silently, and both are pinned by tests.**

- A turn does not end without **trailing silence**. Live decides the speaker
  stopped by hearing them stop; `audio_stream_end` does not substitute. Without
  it the model accepts the audio and never replies, with no error anywhere.
- `session.receive()` **ends at a tool call**. The spoken answer arrives on the
  next generator, so treating the first end as the end of the turn yields a
  tool call and zero audio.

## The tech

| | |
|---|---|
| Model | Gemini Live (`bidiGenerateContent`), native audio in and out |
| Transport | One WebSocket: browser ⟷ our API ⟷ Gemini |
| Audio | 16 kHz PCM up, 24 kHz down — two `AudioContext`s, no resampling in the model |
| Turn detection | Silero VAD + Smart Turn v3, ONNX in a Web Worker |
| Context | 131,072 tokens; audio costs ~25 tokens/second |

## How it works

### The socket

`backend/app/api/live.py` proxies between the browser and Gemini. Tool calls
stop at our server — they never reach the browser.

```
browser  →  {"type":"greet"}    open the conversation; the model speaks first
         →  {"type":"start"}    the speaker pressed the button
         →  {"type":"end"}      pressed again; answer now
         →  {"type":"finish"}   the PARTICIPANT ended the interview
         →  binary              16 kHz PCM

server   →  binary              24 kHz PCM
         →  said / tool / turn / turn_end
         →  resume              a resumption handle was stored
         →  going_away          the server is about to drop us
         →  finished / error
```

**`turn` is assembled server-side.** The browser used to build exchanges from
streaming fragments, deciding where one ended by watching the audio queue
drain — which happens *between chunks*, so questions were split across cards
and a whole spoken sentence could land as one stray word.

### Capture

`frontend/app/parley/liveSession.ts`.

The capture graph is built **once** and reused. Every turn used to call
`getUserMedia` and build a fresh `AudioContext`; opening a device takes
100–500 ms, and the countdown starts the next turn automatically, so the
speaker was usually already talking while the microphone was still opening.
The first turn always looked fine — it is the one you start by clicking, and
*then* speak.

The context takes the **device's own rate** and we resample to 16 kHz in JS.
`new AudioContext({ sampleRate: 16000 })` followed by `createMediaStreamSource`
is a Chrome-only pattern: Firefox throws ([Bugzilla 1725336][ff1]), and it does
not implement the `sampleRate` constraint either ([1388586][ff2]). The
resampler is a box filter, not sample-dropping — decimating 48 kHz by 3 without
a low-pass folds everything above 8 kHz back into the speech band.

The microphone is released on end, on pause, and on socket close, but **not**
between turns — that is the optimisation above, and the browser's recording
indicator staying lit between two turns is correct.

[ff1]: https://bugzilla.mozilla.org/show_bug.cgi?id=1725336
[ff2]: https://bugzilla.mozilla.org/show_bug.cgi?id=1388586

### Turn detection

Off by default, and available in all three modes on the owner's surface. It
began as Speak-only, because Interview and Howler take somebody *through* a
conversation and cutting a participant off mid-answer costs more there — Speak
was simply where that risk was cheapest to carry while the thresholds were
wrong, and they were, twice.

The **guest page is still excluded**, deliberately: somebody on a magic link
has no settings to read, so enabling it for them would be the operator choosing
on their behalf. That belongs on the project, not on a toggle they never see. Two models, answering different questions:

| | Silero VAD (2.2 MB) | Smart Turn v3 (8.7 MB) |
|---|---|---|
| Asks | "is anyone speaking *right now*?" | "did that sound *finished*?" |
| Runs | every 32 ms frame | only when VAD notices a pause |
| Reads | energy/spectral | prosody — trailing intonation, final lengthening |

Silero is the trigger, Smart Turn the decision: speech → pause ≥ 600 ms →
Smart Turn on the last 8 s → P(complete) ≥ 0.7 ends the turn, with a 3 s
backstop. A four-step control (Eager → Very patient) moves the pause length and
the confidence bar **together**; a long pause with a low bar is not "in
between", it waits and then ends the turn anyway on weak evidence.

Both run in a Web Worker (`turnWorker.ts`). A decision costs ~60 ms of
spectrogram plus 150–600 ms of inference under WASM — on the main thread that
is a visible freeze of the level ring, exactly when you are watching it.

**The detector never ends a turn.** It reports; the surface acts. One place
decides, so there is one answer to "what closed this turn".

**`mel.ts` is the risky part.** Smart Turn is a Whisper Tiny encoder, so it
takes an `[1, 80, 800]` log-mel — not a waveform — and there is no
`WhisperFeatureExtractor` in a browser. A wrong mel scale, a periodic window or
the wrong padding side does not throw: it produces a plausible tensor, the model
returns confident probabilities, and detection becomes a coin flip. So the same
spec is implemented twice — TypeScript and numpy
(`backend/app/scripts/mel_reference.py`) — and compared by
`frontend/scripts/verifyMel.mjs`, currently agreeing to ~1e-8.

Two bugs it caught, and one it did not: padding must be on the **right**, and
normalisation must cover **only the real samples** with the padding left at
zero. Both were wrong at first, which made short buffers look like silence —
and silence reads as 0.99 "complete", so turns ended almost instantly. The
cross-check missed these because both implementations were mine and shared the
misreading; only the actual `zero_mean_unit_var_norm` source settled it.

### Interview and Howler

The model calls `record_profile` as it learns things, and `end_interview` when
it is done. The prompt's central instruction is **organise, do not summarise**:
every field carries `notes` and `quotes`, and anything the schema did not
anticipate goes in an `other` bucket, which is usually the largest — the fields
were chosen in advance and the person was not.

`end_interview` is separate from completeness on purpose. "Every required field
is filled" and "this conversation is over" are different facts: the interviewer
fills the last field, then asks whether there is anything to add, and *that*
answer is often the most useful thing in the profile. It is also refused if it
arrives on the same turn the closing question was asked — the instruction alone
was not enough, so the guard is structural.

**Howler's schema is generated from a brief and then frozen on the session**
(`blueprint.py`). Four things depend on that: completeness needs a fixed
denominator, the tool declaration is fixed in the setup message, a resumed
session must find the same shape, and a column that comes and goes between
renders is not a profile.

**Designing it is a conversation** (`designer.py`). It drafts data points from
the first turn and revises them as you talk; "Synthesise now" settles them and
returns a magic link.

That button disappears once a usable link exists, and the reason is worth
stating: a link reads the project's data points when the CONVERSATION STARTS,
not when the link was made. An unopened link therefore already gathers whatever
the latest turn produced, so there is nothing to re-settle and no new link to
hand back — "synthesise again" was a model call that returned the same link.
It comes back when the last link is withdrawn or its interview finishes, since
then there genuinely is a new one to make. Two scars live in its schema: `fields` is **required**,
because with only `reply` required the cheapest valid completion was a reply
and nothing else — including replies claiming to have drafted data points while
returning none. And there is **no `title` field**, because asked for one inline
the model degenerated into it (sixty words of near-synonyms one turn, a Korean
phrase thirty times the next), eating the token budget and truncating the JSON.
Naming happens in its own 40-token call instead.

**Vocabulary.** The designer also generates the terms the participant is likely
to say — React, Node.js, GCP for an engineering interview; Yggdrasil, Snorri
Sturluson for Norse mythology. These go to `AudioTranscriptionConfig.custom_vocabulary`
*and* into the interviewer's prompt, because the first steers the transcript
and the second steers the model's own understanding — and it is the model, not
the transcript, that writes the profile.

**Magic links** (`invites.py`). One link per participant, reusable until that
interview finishes. Revoked → dead; ended → dead; anything else → live. No
expiry in hours, because a dropped call, a closed tab and coming back after
lunch are exactly who needs it. The token authenticates **to one conversation**,
never as a person — it is resolved by its own function rather than through
`current_user`, so it cannot inherit permissions by being passed to the wrong
dependency. The guest page (`/howl/<token>`) renders outside the app shell.

### What is not built

- **Context-window compression** is not enabled, so a session caps at roughly
  15 minutes.
- **Resumption handles** expire 2 hours after termination, and the `resumable`
  flag does not know that.
- **Audio recordings** — see [storage.md](storage.md).
- **A dedicated emotion model.** Today "How they came across" is the
  interviewer's own impression, recorded as `demeanour` and `notable_moments`
  on `end_interview`. It is deliberately worded as observation, never
  diagnosis. The local models in this app are Silero and Smart Turn, and they
  do TURN DETECTION -- neither of them knows anything about emotion.

  A participant who ends the interview themselves now gets a closing pass:
  the live model is asked to call `end_interview` before the socket closes,
  bounded to 8 seconds. It goes to the model that HEARD the audio rather than
  to a text pass over the transcript afterwards, because that transcript is a
  separate and lossier recognition of the same sound -- asking it to describe
  how somebody sounded would be inventing from a bad reading.

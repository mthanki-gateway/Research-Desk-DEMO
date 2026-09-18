# Storage

Where original files live.

## The idea

Uploads used to be parsed in memory and thrown away. Only the extracted text
survived, under whatever chunking rules applied that day. Keeping the original
means three things become possible: re-chunking without re-uploading, opening
the actual page a citation points at, and — next — playing back the audio of an
interview beside the profile it produced.

## Notable decisions

**The provider is a URL, not a code path.** Every option worth having speaks
the S3 API, so `S3Storage` takes an endpoint and that is the whole of the
difference between them. There is deliberately no per-provider subclass — the
differences are a hostname and a quota, and a class each would be five copies
of the same six calls waiting to drift apart.

| | Free | Card? |
|---|---|---|
| **Supabase** | 1 GB, 5 GB egress | **no** |
| Filebase | 5 GB | no |
| Cloudflare R2 | 10 GB, zero egress | **yes** |
| Backblaze B2 | 10 GB | yes, for verification |
| MinIO | self-hosted | n/a |

**Supabase is the recommendation.** R2 was the original pick for one number —
zero egress, on a store whose entire job is serving files back — but enabling
it requires a credit card, which is a hard stop for a project that should be
runnable by anyone. Supabase needs no card, and this app already depends on it
for auth: no new account, no new vendor. Its endpoint is
`https://<project>.supabase.co/storage/v1/s3`.

Its 1 GB is the real constraint and is worth knowing before it bites.
Documents are small; a ten-minute interview recording is about 19 MB, so the
planned audio fills that quota in roughly fifty interviews. Postgres `bytea`
was rejected outright — it bloats every backup and burns Neon's row quota.

**Local storage is the default, and that is not a placeholder.** Object storage
should not stand between cloning this repo and seeing it work, and a backend
that only runs when a paid account is configured is a backend nobody tests.
`LocalStorage` writes to a Docker volume; `S3Storage` is the same interface over
boto3. One setting chooses, and nothing above the module knows which is in use.

**`boto3` is an optional extra.** The default path needs no cloud SDK at all.

**Store the original, never the extracted text.** Re-chunking needs the source.

**Key by UUID, never by filename** — `docs/<uuid>.pdf`. Filenames collide on
the second upload of `report.pdf`, and a filename in a path is how directory
traversal gets in. The extension is kept, because that is what makes a signed
URL open in a viewer instead of downloading as a blob.

**Storing is never fatal.** A store that is full, misconfigured or unreachable
costs the archival copy — it must not cost the indexing, because the text is
what answers questions. The column is nullable and the failure is a warning.

**Store before parsing.** Parsing is the step most likely to fail, and the
original is exactly what somebody needs in order to work out why.

**Escape is refused, not trusted.** Keys are minted from UUIDs, so a key that
escapes the root should be impossible — which is exactly the kind of assumption
worth enforcing. `LocalStorage` resolves and checks every path.

## The tech

| | |
|---|---|
| Interface | `Storage` ABC — `put`, `get`, `signed_url`, `delete` |
| Default | `LocalStorage`, a named Docker volume at `/data/files` |
| Production | `S3Storage` — boto3, S3v4 signing, any S3-compatible endpoint |
| Column | `documents.storage_key VARCHAR(512)`, nullable |

## How it works

`backend/app/services/storage.py`. Configuration:

```bash
STORAGE_BACKEND=local          # or s3
STORAGE_DIR=/data/files        # local only

# Supabase: Storage → S3 Connection in the dashboard gives all four.
S3_ENDPOINT_URL=https://<project>.supabase.co/storage/v1/s3
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET=...
S3_REGION=eu-west-2            # R2 ignores it; Supabase wants its project region
S3_PUBLIC_BASE=                # optional custom domain
```

Switching needs `pip install '.[storage]'` (or the equivalent in the image),
the variables above, and nothing else — no code change and no migration.
`STORAGE_BACKEND=r2` is still accepted, because that is what the original
design called it and a setting that silently stops working is a bad trade for
a tidier name.

**Ingest** calls `put()` before parsing and records the key.

**`GET /documents/{id}/file`** returns the original: a 307 redirect to a signed
URL when the backend can mint one, and the bytes themselves when it cannot.
An S3 backend signs; a local directory has no public host to sign for. The caller does not
need to know which — following a redirect is what a browser does anyway.
`Content-Disposition: inline`, because the point is to *open* page four, not to
download a copy of the handbook.

Ownership is checked before anything is read, and a document with no
`storage_key` returns a message saying the original was not kept — rather than
"not found", which would suggest the document itself is gone when its text is
perfectly well indexed.

---

## Planned: interview recordings

Not built. The design, so it is not re-derived later:

**Why.** A profile says what someone answered. The recording is how they said
it — and for Howler, where results are read by someone who was not present,
that is most of the value. It also makes the emotion report checkable rather
than something to take on faith.

**Where the audio already is.** Both directions pass through our socket
(`api/live.py`): the participant's 16 kHz PCM through `_uplink`, the model's
24 kHz through `_downlink`. Nothing extra needs capturing — it only needs
keeping.

**Shape.** Buffer per turn rather than per session: a turn is ~30 s ≈ 1 MB at
16 kHz mono 16-bit, where a 10-minute interview is ~19 MB. Write a WAV per turn
at `activity_end`, key it `howl/<session>/<n>.wav`, and store the key on the
`Message` row. Per-turn segmentation is what makes the player able to jump to
"the answer about budget" — a single file cannot do that without a separate
index.

**Player.** In the Results tab, beside the profile, with the segment list
driven by the same turn boundaries the profile was recorded against. Each data
point links to the turn that filled it.

**Consent is a requirement, not a setting.** Recording someone requires telling
them. The guest page must say so before the first turn, not in a tooltip.

**Video, later.** The same per-turn keys extend to video, which is what facial
expression analysis per answer would need. Worth noting that this raises the
stakes considerably on the point above.

---

## Planned: a dedicated emotion model

Today's "How they came across" is the *interviewer's* impression, recorded as
`demeanour` and `notable_moments` on `end_interview` and deliberately worded as
observation rather than diagnosis.

A dedicated model is the obvious next step, and the plumbing already exists —
`turnWorker.ts` runs ONNX in a Web Worker, with a ring buffer of recent audio.
Candidates: [`onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX`][ser]
(ready-made ONNX), or [Wav2Small][ws] at 72K parameters and ~120 KB quantised.

Two things to check before committing, both learned the hard way from Smart
Turn: **verify the real ONNX input contract** rather than trusting the model
card, and note that wav2vec2 takes a **raw waveform**, not a mel — so `mel.ts`
would not be reused, though the worker and ORT plumbing would.

And one caution worth keeping in the design: speech emotion recognition is
substantially less reliable than turn detection, and varies by culture and
accent. It belongs in a profile as a note, never as a number anyone decides on.

[ser]: https://huggingface.co/onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX
[ws]: https://arxiv.org/html/2408.13920v4

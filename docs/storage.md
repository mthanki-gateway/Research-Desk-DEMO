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

**Key by UUID, never by filename.** Filenames collide on the second upload of
`report.pdf`, and a filename in a path is how directory traversal gets in. The
extension is kept, because that is what makes a signed URL open in a viewer
instead of downloading as a blob.

**Prefix by tenant.**

```
t/<owner>/docs/<document-uuid>.pdf
t/<owner>/howl/<session-uuid>/0007.wav     (planned)
t/_local/docs/<document-uuid>.pdf          (auth off)
```

The prefix is **not** what keeps tenants apart — the API does that, checking
ownership before it ever looks at a key, and keys are never exposed to a
client. It earns its place three other ways:

- **Deleting a tenant** becomes one prefixed list instead of a full scan joined
  against the database. So does measuring what one is using, and exporting
  everything they own when they ask for it.
- **Bucket policies** can be scoped by prefix — S3 IAM and Supabase Storage
  rules both work on paths. A flat namespace cannot be divided, so every
  credential is necessarily a credential for everything.
- **Defence in depth.** Ownership is enforced in one place today; a prefix
  means a future mistake there is not automatically a cross-tenant read.

Anonymous uploads get a named segment (`_local`) rather than an empty one,
which would collapse the path and put them where a prefixed delete for any
tenant could reach them. It mirrors the `COALESCE(owner_id, '')` the SQL side
already uses for the same reason.

For recordings the owner is the **project's** owner, not the person speaking:
a guest holding a magic link has no account and owns nothing, so one delete
removes a tenant's documents and their audio together.

**Changing the scheme orphans nothing**, because `storage_key` is *stored* per
document rather than derived. Anything written under the earlier flat scheme is
still fetched from where it actually is.

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

## Interview recordings and voice analysis

Participant audio is captured by the live socket as 16 kHz mono PCM, wrapped
as one WAV per participant turn, and stored through the shared `Storage`
interface. Each user message keeps its `audio_key` in `agent_meta`; the default
local backend writes the files to the `filestore` Docker volume. Audio storage
failures are logged and do not interrupt the call.

After an interview ends, a Postgres-backed job transcribes each saved clip,
replaces the initial live caption in the message, and rebuilds the profile from
the ordered participant/interviewer turns. It uses Groq Whisper when that
owner has a Groq key, otherwise Gemini speech-to-text. This transcription is
provider-hosted; there is no local ASR model in this flow. The live Gemini
session also emits provisional input/output captions for the live screen.

The optional local emotion model is
[`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`][aud]. It scores the
stored clips on the API CPU and reports arousal, dominance and valence relative
to that participant's own median. The config defaults `EMOTION_ANALYSIS` to
`false`; the local Docker Compose stack sets it to `true` so voice analysis
runs after interviews. Set `EMOTION_ANALYSIS=false` in `.env` to disable it
locally. The development image has PyTorch and Transformers through the
`local-nlp` extra; the production image omits them. The model weights load
lazily on the first emotion job.

The local browser models Silero VAD and Smart Turn v3 answer a different
question: when a speaker has paused and whether the turn sounds finished.
They are not used to transcribe speech or detect emotion. Speech-emotion
estimates vary across speakers, accents and recording conditions; the UI
frames them as observations rather than findings about a person.

[aud]: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

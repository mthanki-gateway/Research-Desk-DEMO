# HANDOFF — read this first

Working state, decisions and open threads for **Research Desk**. Written so work
can resume cold, without the prior conversation.

`README.md` explains the *project* — architecture, measured findings, why each
choice was made. **This file explains where the work stopped and what happens
next.** Read both; they barely overlap.

Last updated: **2026-09-02**, end of the session that finished step 6 plus a
full UI pass.

---

## 1. Status

| Step | State |
|---|---|
| 1. Compose skeleton, health check, Next shell | done |
| 2. Ingestion: upload → parse → chunk → embed → Qdrant | done |
| 3. Baseline RAG (`POST /ask`) | done, kept permanently for comparison |
| 4. LangGraph agent (`POST /research`) | done, cycle verified |
| 5. Streaming | done — folded into step 6 as node-level SSE progress |
| 6. Chat sessions, checkpointer, scope, history compression | done |
| — UI pass: routes, design system, right rail, citations | done |
| — **Material 3 conversion** | done (see §12) |
| 9. **Auth — Supabase** | **DONE and live.** Signed in, corpus claimed. §9 |
| — File storage on Cloudflare R2 | after auth, specced in §8 |
| — **MCP server** | agreed, specced in §13 |
| 7. Deploy | after auth |
| 8. Agent tools (web search, charts) | not started |
| — Hybrid search (BM25 + RRF) | agreed, not started |

Everything runs locally today: `docker compose up -d`, then
http://localhost:3000.

### The user's stated order for the next session

> "We will build file storage, and auth and main things tomorrow."

They also accepted deploying **before** auth, on the condition that the public
URL is gated (see §8).

---

## 2. What this app is

A document-grounded research agent, built as a **learning vehicle** for the
current AI-app stack. The user's goal is understanding LangGraph, RAG, vector
DBs and the modern Python/Next stack — not shipping a product. That matters:

- **Explain mechanisms, don't just implement them.** They ask "why" constantly
  and the answers are the point, not an interruption.
- **Measure claims.** Nearly every decision here was settled with a probe, not
  an assertion. Several confident predictions turned out wrong (§5). Keep
  testing rather than asserting.
- **The Lab route exists** so diagnostic tooling stays visible rather than
  hidden as "internals".

### Hard constraint that shapes everything

Only free Google AI Studio access. Measured limits:

| Model | RPM | TPM | RPD |
|---|---|---|---|
| `gemma-4-26b-a4b-it` (the LLM) | 30 | **16,000** | 14,400 |
| `gemini-embedding-001` | 100 | 30,000 | **1,000** |
| Gemini 3.x Flash | 5 | 250,000 | **20/day — unusable** |

**16K tokens/minute is the binding constraint**, not context window. It drives
`top_k=5`, the history-compression design, and the agent's iteration cap.

---

## 3. File map

### Backend (`backend/app/`)

| File | Role |
|---|---|
| `main.py` | FastAPI app, lifespan wiring (tables, embeddings, vector store, checkpointer, graph) |
| `config.py` | every setting, `pydantic-settings`, one place for all knobs |
| `db/models.py` | `Document`, `Chunk`, `ChatSession`, `Message` |
| `db/session.py` | async engine, `SessionLocal`, `create_tables()` |
| `services/limiter.py` | dual token-bucket (requests **and** tokens per minute) |
| `services/embeddings.py` | `gemini` / `fastembed` providers behind one interface |
| `services/parsing.py` | pypdf + text, keeps page numbers |
| `services/chunking.py` | **section-aware splitter — highest-leverage code in the app** |
| `services/vectorstore.py` | the only file that imports `qdrant_client` |
| `services/ingest.py` | pipeline stages + `ENRICHERS` hook for future categorisation |
| `services/llm.py` | Gemma client, own limiter, `generate_json`, `extract_string_list` |
| `services/retrieval.py` | retrieve, query expansion, RRF fusion, context assembly |
| `services/rag.py` | step-3 baseline, ~90 lines, deliberately naive |
| `services/history.py` | conversation-history strategy + rolling summary |
| `agent/state.py` | `ResearchState` + reducers |
| `agent/nodes.py` | `plan`, `retrieve_node`, `draft`, `critique` |
| `agent/graph.py` | wiring, conditional edge, `run_agent`, `stream_agent` |
| `agent/checkpointer.py` | psycopg pool, `init_checkpointer`, `discard_thread` |
| `api/documents.py` | upload, list, delete, `/search`, `/stats`, `/chunks/*` |
| `api/chat.py` | `POST /ask` (baseline), `POST /research` (agent) |
| `api/sessions.py` | session CRUD, `/messages`, `/stream` (SSE) |
| `scripts/list_models.py` | which model ids the key can actually serve |
| `scripts/reset_vectors.py` | drop Qdrant collection + rows (after a dim change) |
| `scripts/prune_checkpoints.py` | clear orphaned checkpoint rows |
| `scripts/probe_resume.py` | proves accumulators leak across a reused thread |

### Frontend (`frontend/app/`)

| File | Role |
|---|---|
| `globals.css` | **design tokens + the whole component layer — start here** |
| `icons.tsx` | 14 hand-rolled inline SVGs. **No emoji anywhere, by instruction** |
| `providers.tsx` | shared sessions/documents state, citation panel, rail collapse |
| `shell.tsx` | navy sidebar, nav, ingest progress, Cmd+K trigger |
| `layout.tsx` | `<Providers><Shell>` |
| `page.tsx` | redirects `/` → `/chat` |
| `chat/page.tsx` | session list + chained empty states |
| `chat/[id]/page.tsx` | the conversation; session cache; optimistic send |
| `chat/[id]/rail.tsx` | right rail (Controls / Source tabs) + collapsed strip |
| `library/page.tsx` | upload, document cards, chunk inspector |
| `lab/page.tsx` | single-shot ask, agent/baseline toggle, raw search |
| `chunk-panel.tsx` | global citation slide-over (used outside chat only) |
| `command-palette.tsx` | Cmd/Ctrl+K session switcher |
| `lib/api.ts` | every API call + SSE frame parsing |

---

## 4. Design decisions already settled

Don't relitigate these without a new reason; each was argued through.

| Decision | Why |
|---|---|
| Qdrant, not Pinecone/pgvector | runs locally *and* has a free cloud tier; identical client code both places |
| Two databases | Qdrant answers "nearest vector"; Postgres does everything else. Not "vectors vs text" — both store text |
| Chunk UUID = Qdrant point id | one identity across stores, no translation table |
| Every LLM call uses `responseSchema` | Gemma has no thinking channel; unstructured output becomes a monologue |
| No native function-calling | Gemma ignores `functionDeclarations`. Schemas are the substitute |
| Transcript in `messages`, not checkpoints | queryable, and decouples the UI from LangGraph's internal state shape |
| One checkpoint thread per **attempt**, deleted on success | reducers merge across a reused thread; see §5 |
| `multi_query` default **off** | no measurable gain at this corpus size, costs a Gemma call per query |
| SSE, not WebSockets | one-way data; plain HTTP passes proxies and needs no library |
| Node-level progress, not token streaming | schema'd output would stream half a JSON object |
| Deploy before auth | deployment surfaces architectural problems; auth is a self-contained slice |
| Red is strictly semantic | decorative red would compete with sky and erode its warning meaning |
| Navy sidebar in both themes | gives the app an identity; sky is the only accent |

### Rejected, with reasons

- **Reranking** — Gemma could do it via schema, but 20 chunks ≈ 5K tokens against 16K TPM ≈ 3 reranks/minute. Hybrid + HyDE are better value.
- **Alembic** (until deploy) — `create_all` cannot alter existing tables, so column additions need a one-line manual `ALTER`. That is acceptable with a single database, and the only pending column addition is R2's `storage_key`. Alembic lands at deploy, when local + Neon must stay in sync and a forgotten ALTER becomes a production 500 rather than an instant local error. See §9.
- **Unifying on psycopg3** — SQLAlchemy could use it, but asyncpg is faster and it would not reduce connection count anyway.
- **A new-chat document picker dialog** — puts a decision before the task. Better: "Ask about these" from the Library, or a picker only past ~10 documents.
- **Markdown rendering in answers** — Gemma returns plain prose; a renderer is a dependency plus an XSS surface for no gain.
- **A manual dark-mode toggle** — it already follows system preference.

---

## 5. Bugs and traps already hit — do not rediscover these

### Gemma behaviour

1. **No thinking channel.** Asked for prose it wrote its entire reasoning trace — constraint checklists, three drafts, a "Self-Correction" section — then truncated. **Always pass a `responseSchema`.**
2. **Bare `responseMimeType: application/json` without a schema is unusable** — it narrates its reasoning *about* JSON instead of emitting it.
3. **Repetition loops** at `temperature=0.7` ("way's actually way's actually…") until the token cap, invalidating the JSON. Use ≤0.35 plus `topP`. There is no repetition-penalty parameter.
4. **One extra schema field destroyed a good result.** `PLAN_SCHEMA` had a `reasoning` field; Gemma produced four correct `sub_questions`, then degenerated inside `reasoning`, truncating the object — and a strict parse discarded the good plan. **Ask only for fields you use**, and `extract_string_list()` salvages arrays from truncated JSON.
5. **No function-calling.** Verified: given tool declarations it replies in prose.

### LangGraph

6. **Reducers do not reset.** `Annotated[list, append]` applies to the *input* too, so passing `[]` appends nothing. Combined with one thread per session, evidence from an earlier question leaked into later answers.
7. **A failed turn retries onto the same thread.** No messages are persisted, so the turn index is unchanged. `scripts/probe_resume.py` measured a second attempt duplicating `sub_questions` and `trace`, and polluting `tried_queries` — which makes `critique` refuse to re-run a query it thinks was tried. Hence **thread per attempt**.
8. **Checkpoints grow without bound.** ~25 turns produced 990 rows / **920KB**, roughly 4x all real app data. `discard_thread()` runs after each turn; `prune_checkpoints.py` clears orphans.
9. **The checkpointer needs psycopg3**, not asyncpg — two drivers, two pools, ~15 connections. **Neon's free tier caps connections: lower both pool sizes at deploy.**

### Frontend

10. **Tailwind v4 cannot `@apply` a custom component class** — only utilities. `.card-interactive` repeats the card properties rather than applying `.card`.
11. **Next 16 defaults to Turbopack, which ignores `WATCHPACK_POLLING`** and gets no filesystem events across a Windows bind mount — edits were silently ignored. The dev container runs `next dev --webpack`. For Turbopack speed, run the frontend natively.
12. **An unstable callback in context wiped state.** `closeChunk` was an inline arrow inside the context `useMemo`, so its identity changed whenever any value changed, re-running the chat page's mount effect and clearing the citation the instant it was set. Wrapped in `useCallback`.
13. **A `fixed` element with `translate-x-full` extends the scrollable area**, producing a horizontal scrollbar and letting the layout drift sideways. `overflow-x: hidden` on `html, body`.
14. **Full-page skeletons flash on every navigation.** Fixed with a module-level session cache, a header title taken from the already-loaded session list, and no skeleton at all in the message area.
15. **`mr-[18rem]` on a `max-w-3xl` box shrinks the column** instead of reserving space (768 → 480px). Use padding on an unconstrained wrapper.
16. **Smooth scroll on first paint reads as jank.** `behavior: "auto"` for the first scroll, `"smooth"` afterwards.

### Environment / tooling

17. **PowerShell `Get-Content | Set-Content -Encoding UTF8` double-encodes** — it turned an ellipsis into `â€¦` inside a Python file. **Use the Edit tool for file edits, never PowerShell text munging.**
18. **PowerShell 5.1 has no inline `if` expression** and no `&&`. Also `Invoke-RestMethod` mangles UTF-8 on display — an em-dash showing as `â` in the terminal does **not** mean the data is corrupt. Check with `psql` before "fixing" it.
19. **A guard blocks reading/writing any `.env*` file**, and it aborts the whole shell command when one is mentioned. Real values live in `env.example`; the user copies it to `.env` themselves.
20. **Qdrant client and server must match on major.minor.** Client 1.19 could not read storage written by server 1.12.4 and crash-looped. Both are pinned; bumping one means bumping the other and wiping the volume.

### Deploy-only — cannot be reproduced locally or in CI

21. **`output: "standalone"` breaks the Vercel build.** It was set for the Docker prod stage with the comment "harmless on Vercel". It is not. Vercel builds through its own Build Output API, and standalone mode relocates the file-trace manifests, so the deploy dies with `ENOENT: ... open '/vercel/path0/frontend/.next/next-server.js.nft.json'` — which reads like a broken `node_modules`, not a config conflict. Now gated on `process.env.VERCEL`, which Vercel sets in every build environment. **Neither `next build` locally nor CI can catch this**: both succeed *with* standalone output, because the step that fails is Vercel's own post-build trace collection. Adding `VERCEL=1` to CI would not help either — the build still passes; only a real deploy exercises it. Verified both branches: unset → `.next/standalone` present (Docker needs it), `VERCEL=1` → absent (Vercel needs it absent).
22. **`sslmode`/`channel_binding` in `DATABASE_URL` kill asyncpg.** Managed Postgres hands out libpq-style URLs; asyncpg rejects both params as unknown `connect()` kwargs, while the LangGraph checkpointer reads the *same* URL through psycopg, which understands only `sslmode`. One URL, two dialects — so it cannot be fixed by rewriting the URL. `db/session.py` translates for asyncpg only (`sslmode` → `connect_args["ssl"]`, `channel_binding` dropped) and `checkpointer.psycopg_url()` passes the libpq form through. Invisible locally: compose Postgres carries no SSL params at all.

---

## 6. Testing — what exists, honestly

| Layer | Tool | Status |
|---|---|---|
| Backend lint | `ruff` | in use, clean |
| Frontend types | `tsc --noEmit` | in use, clean |
| Browser | Playwright MCP | used constantly for verification |
| API probes | PowerShell + curl | how most claims in these docs were verified |
| Reusable probes | `scripts/probe_resume.py`, `scripts/list_models.py` | committed |
| **Unit tests** | `pytest` | **NONE WRITTEN** |
| **E2E suite** | — | none |
| **CI** | — | none |

`pytest` and `pytest-asyncio` are already in the dev group with
`asyncio_mode = "auto"` configured. Nothing has been written.

Everything so far was verified manually, which did catch real bugs (the
reducer leak, `PyJWKClientError`, the state-layer mismatch, the thin column) —
but **nothing prevents a regression**.

Highest-value tests, in order:

1. **`chunking.py`** — pure, the highest-leverage code in the app, and the most
   likely thing to be broken by a "small improvement"
2. **`reciprocal_rank_fusion`** — pure, and about to be reused by hybrid search
3. **`limiter.py`** — pure; a bug here means 429s in production
4. **ownership** — user B gets 404 on user A's session. The one test whose
   failure is a *security* bug. Needs a fake token: either a test signing key
   or monkeypatching `auth._decode`
5. **`extract_string_list`** — the truncated-JSON salvage, which exists
   because of a real observed failure

1-3 and 5 are pure functions with no I/O and are quick. Recommended before
deploy, since deploy is where changes stop being manually verifiable.

## 7. Local environment

```powershell
docker compose up -d                          # whole stack
docker compose logs -f api                    # backend logs
docker compose --profile tools up -d pgweb    # DB browser, no login needed
docker compose exec api ruff check app        # backend lint
cd frontend; npx tsc --noEmit                 # frontend typecheck
```

| Surface | URL |
|---|---|
| App | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Postgres (pgweb) | http://localhost:8081 — no credentials, `--lock-session` |

Postgres credentials (in `docker-compose.yml`): `rd` / `rd_local_dev` /
`research_desk`, host `postgres` inside the compose network.

**`.env`**: the user creates it with `Copy-Item env.example .env`. `env.example`
holds a real key and is gitignored for that reason. **The key must be rotated
before any public push.**

### Test fixtures — committed, so nothing is lost to a volume wipe

`fixtures/` holds the two documents every measurement in `README.md` refers to,
plus `fixtures/README.md` with the re-ingest commands and the expected results.

```powershell
curl.exe -s -X POST http://localhost:8000/documents -F "file=@fixtures/acme-report.md;type=text/markdown"
curl.exe -s -X POST http://localhost:8000/documents -F "file=@fixtures/antiquity.md;type=text/markdown"
```

### The canonical regression test

```
top_k=1, question: "What was operating income in 2024, and what capital
expenditure is planned for 2025?"

/ask      → gets operating income, MISSES capex   (expected failure)
/research → gets both, correct                     (expected pass)
```

If `/research` stops passing this, the agent is broken. Other useful probes are
tabulated in `fixtures/README.md`.

---

## 8. Deploy — see `DEPLOY.md`

**The full plan now lives in `DEPLOY.md`.** Provider table (Vercel / Render /
Neon / Qdrant Cloud / GitHub Actions), env vars, ordered steps, environments,
monorepo notes, and five traps in the order they bite. Do not re-derive it here.

Settled points that differ from the old plan:

- **Postgres stays on Neon** (I briefly recommended Supabase and reversed it —
  do not re-litigate). Reason: Neon's **direct endpoint is IPv4-reachable**, so
  the app skips connection pooling entirely. That *deletes* the
  prepared-statement problem rather than working around it. Supabase's direct
  connection is IPv6-only and Render's free egress is IPv4, which forces you
  onto Supavisor and straight into that problem; Supabase free also pauses a
  project after ~7 days idle, so an occasionally-demoed app needs manual
  unpausing. Supabase remains the **identity provider only**.
- **Use the Neon string WITHOUT `-pooler`.** The pooled endpoint is PgBouncer in
  transaction mode and breaks prepared statements, which this app uses through
  *both* its drivers (asyncpg via SQLAlchemy, psycopg3 via
  `AsyncPostgresSaver`). The failure is an intermittent
  `prepared statement "__asyncpg_1__" does not exist` that shows up under load,
  so it reads like a capacity problem rather than a config one. Safe to go
  direct only because the pool budget is 5 + 2.
- **Environments: local + prod only.** Staging is not justified for a
  single-developer learning app; the industry has largely replaced it with
  per-PR previews anyway. Config is all env-var driven, so adding one later is
  cheap. **Footgun:** Vercel creates previews unasked, and unscoped env vars
  make them talk to the *production* backend — scope `NEXT_PUBLIC_API_URL` to
  Production, or give previews a Neon DB branch.
- **Monorepo is fine and stays.** Both platforms take a Root Directory. But
  **never add `on.paths` filters to `ci.yml` while those checks are required** —
  GitHub reports a skipped job as pending forever, so a frontend-only PR would
  block on a `backend` check that never runs. Use `dorny/paths-filter` inside a
  single always-running job instead.

Still mandatory and unchanged: **gate the public URL before it is public.** The
Gemma quota is per *key*, not per user, so one crawler exhausts everyone's daily
budget and Google gives no per-caller breakdown to diagnose it with.

### Repo and CI — DONE (2026-09-02)

- `git init`, branch `main`. **No commit made yet** — awaiting instruction.
- `env.example` is now placeholder-only (`REPLACE_ME`) and is **committed** on
  purpose. `.gitignore` was inverted to match: `.env*` + `*.env` ignored,
  `!env.example` negated after them (negation must follow the broad pattern or
  git ignores it).
- Verified by scanning all 78 would-be-staged files for the old Gemma key, the
  Supabase project ref, and `sb_publishable_`/`sb_secret_` prefixes: clean.
- `.github/workflows/ci.yml` — ruff, `tsc --noEmit`, real `next build`, and a
  build of the prod Docker image. **Checks only, no deploy**: Vercel and Render
  build on push themselves, and two build paths that can disagree is worse than
  one.

### Two bugs CI caught immediately, both invisible in dev

1. **`next build` had never been run and did not pass.** `/login` called
   `useSearchParams()` outside a Suspense boundary — legal in `next dev`, fatal
   when prerendering. Fixed by moving the read into a null-rendering
   `<OAuthErrorReporter/>` inside `<Suspense>`. This is the whole argument for
   having CI build the production bundle.
2. **`next lint` no longer exists.** Next 16 removed it (deprecated in 15), and
   it fails *misleadingly*: it treats `lint` as a directory and dies with "no
   such directory: ./frontend/lint", which reads like a broken checkout. The
   dead script was removed from `package.json`. There is now **no frontend
   linter** — ESLint is not installed. `tsc --noEmit` is doing that work alone.

Also: Node 20 → 22 in `frontend/Dockerfile` and CI, because
`@supabase/supabase-js` warns on every build that 20 is deprecated.
`ruff format --check` is deliberately **not** in CI — the codebase is hand-wrapped
at 100 cols with comments sized to their code, and the formatter wants to rejoin
those; gating on it means one large reflow that changes no behaviour.

Backend prod image builds clean at **177 MB**, `/health` exists for Render's
health check, and `create_all` on startup means the first deploy needs no
migration step.

Older hardening list in `README.md` → "Production hardening checklist".

---

## 9. File storage — agreed design, not built

Uploads are currently parsed in memory and **thrown away**.

**Chosen: Cloudflare R2** — 10GB free and **zero egress**, S3-compatible so
`boto3` works. (Supabase Storage 1GB is the fallback. Postgres `bytea` was
rejected — bloats backups and burns Neon's quota. Render's free tier has no
persistent disk at all.)

Mirror the embeddings-provider pattern:

```python
# services/storage.py
class Storage(ABC):
    async def put(self, key: str, data: bytes, content_type: str) -> str: ...
    async def get(self, key: str) -> bytes: ...
    async def signed_url(self, key: str, ttl: int = 3600) -> str: ...

class R2Storage(Storage): ...     # boto3, S3-compatible endpoint
class LocalStorage(Storage): ...  # a Docker volume, for dev
```

Then: a `storage_key` column on `documents`, a `put()` call in
`ingest_document` before parsing, and `GET /documents/{id}/file` returning a
signed URL.

Two rules agreed:
- **Store the original, not the extracted text** — re-chunking later needs the source.
- **Key by document UUID** (`docs/{uuid}.pdf`), never by filename — avoids collisions and path traversal.

Payoff beyond archival: **re-ingest without re-upload** when chunking improves.

### The one schema change, and why Alembic is NOT needed for it

`storage_key` is the only column R2 adds, and `create_all` cannot add it to an
existing table. But that does not require a migration tool — one statement does
it:

```powershell
docker compose exec postgres psql -U rd -d research_desk -c "ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key VARCHAR(512);"
```

Safe because `create_all` runs on every boot and is a **no-op on existing
tables** (proven by dozens of restarts with data intact), and because a fresh
volume self-heals: `create_all` builds the table *with* `storage_key` since the
model declares it. The ALTER is a one-time patch for already-existing databases.

Note also that **the whole auth phase needs no migrations at all** — `owner_id`
already exists, and `api_keys` / `usage_events` are new tables, which
`create_all` handles.

**Alembic is deferred to step 7 (deploy), deliberately.** With one database and
one developer, a manual ALTER is fine and you notice a mistake instantly. With
two databases (local + Neon) the failure mode changes: forget the ALTER on Neon
and production 500s with `column "storage_key" does not exist` while local
works. That is when a migration tool stops being optional.

---

## 10. Auth — **Supabase**, DONE and live

User chose **Supabase**, with **Google OAuth + magic link**. Project being set
up by the user; needs `SUPABASE_PROJECT_URL` and the `anon` key.

### Supabase is used ONLY as an identity provider

Application data stays in our own Postgres and Qdrant. **Row Level Security is
therefore irrelevant** — RLS only protects tables inside Supabase's own
database. Authorization is enforced in FastAPI and by the Qdrant payload
filter. Worth stating because adopting Supabase is usually assumed to mean
adopting RLS; here it does not.

### JWT verification (verified 2026-09-02)

Projects created after 1 May 2025 use **asymmetric JWTs by default** (RS256,
optionally ECC/Ed25519), so the backend holds **no shared secret**. Public keys
come from the JWKS endpoint:

```
https://<project-ref>.supabase.co/auth/v1/jwks
```

Fetch once, cache in memory, read `kid` from the incoming token, verify
locally. **No network call per request**, and keys can be rotated without
redeploying. `pyjwt[crypto]`'s `PyJWKClient` handles fetch, cache and `kid`
lookup.

Claims used: `sub` (Supabase user UUID → our `owner_id`), `email`, `exp`, and
`aud` must equal `"authenticated"`.

### Plan

`backend/app/auth.py`:

```python
async def current_user(cred = Depends(bearer)) -> User: ...      # writes
async def optional_user(cred = Depends(bearer)) -> User | None: ...  # reads, during migration
```

Two dependencies deliberately: with only `current_user`, every existing row
with `owner_id = NULL` becomes invisible the moment auth lands, and the test
corpus disappears.

| Endpoint | Change |
|---|---|
| `POST /documents`, `POST /sessions` | set `owner_id = user.sub` |
| `GET /documents`, `GET /sessions` | filter by `owner_id` |
| `/sessions/{id}/*` | **404 on owner mismatch**, not 403, so ids aren't enumerable |
| `/search`, `/ask`, `/research` | pass `owner_id` into `search()` |

**No vector migration needed.** `owner_id` is already on every Qdrant payload,
indexed, and `VectorStore.search(owner_id=...)` is implemented.

Frontend: `@supabase/ssr` (not the older auth-helpers) for App Router cookie
sessions — `lib/supabase/{client,server}.ts`, `middleware.ts` for refresh and
route gating, a `/login` route, `Authorization: Bearer` on every call in
`lib/api.ts`, and an account item in the drawer.

### Two migration decisions

1. **Existing rows have `owner_id = NULL`.** Recommend claiming them for the
   first user who signs in (a one-line `UPDATE`) — simplest mental model, and
   it is the user's own data.
2. **Quota caps** need a `usage_events` table (new table, so `create_all`
   covers it). This is the real reason auth matters: the Gemma quota is per
   *key*, so without per-user caps one visitor exhausts everyone's budget.

### LIVE — current state (2026-09-02)

Auth is **on and working**. Verified end to end:

| Check | Result |
|---|---|
| `/health` | `auth_enabled: true` |
| `/documents`, `/sessions`, `/stats`, `/auth/me` unauthenticated | **401** |
| `/chat` in a browser | redirects to `/login` |
| Publishable key accepted by Supabase | yes — `sb_publishable_...` works with supabase-js 2.113 |
| Providers enabled | **google** and **email** (magic link) |
| Signed-in owner | `e9e2157c-3c6d-4aec-9e22-478b0408bbb4` |
| Claimed | 2 documents, 5 sessions, **0 unowned vectors remaining** |

Project ref is in the local `.env` (never committed) and in the Supabase
dashboard. Deliberately not written down here: this file is committed, and while
the ref is not a credential — it ships in the browser bundle — it identifies the
project to anyone reading the repo, and documentation gains nothing from it.
Keys must be rotated before any public push.

**`mailer_autoconfirm` is false**, so magic links really are emailed. Supabase's
built-in mailer is rate-limited on the free tier and often lands in spam. Do
NOT turn autoconfirm on for anything public — it would let anyone claim any
email address.

### Claiming is now switched OFF

`ALLOW_CLAIM_UNOWNED=false` (default). `POST /auth/claim` returns **403**.

It existed to adopt the pre-auth corpus, which is done. Leaving it enabled
would let any *new* account claim ownerless rows that appeared later — for
example if auth were briefly disabled during maintenance. To re-run a
migration: set the flag true, sign in, set it false again.

### No password, and no email/password sign-in

Magic link creates a Supabase user with **no password at all**, and none is
ever needed. Email/password is deliberately **not implemented** — it would add
signup, reset, strength rules and breach exposure for no benefit over magic
link plus Google. If it is ever wanted, it is `supabase.auth.signUp` /
`signInWithPassword` plus a reset route; nothing on the backend changes,
because the backend only ever sees a JWT.

### BUILT — what exists now

Project ref: see `.env`. Verified against the live JWKS endpoint:
**`/auth/v1/.well-known/jwks.json` returns 200 with one ES256 key**;
`/auth/v1/jwks` returns 401. So asymmetric, JWKS path, no shared secret.

| File | Role |
|---|---|
| `backend/app/auth.py` | `User`, JWKS verification, `current_user`, `optional_user`, `forbid_if_not_owner` |
| `backend/app/api/auth.py` | `GET /auth/me`, `POST /auth/claim` |
| `frontend/lib/supabase.ts` | browser client, `authEnabled`, `getAccessToken` |
| `frontend/app/login/page.tsx` | Google + magic link |
| `frontend/app/auth/callback/page.tsx` | session pickup, then claims pre-auth rows |

**Auth is configuration-gated.** With `SUPABASE_URL` unset, `current_user`
returns `ANONYMOUS` whose `owner_id` is `None`, and every query falls back to
unfiltered — byte-identical behaviour to before auth existed. Verified:
`/documents`, `/sessions` and `/search` all still answer unauthenticated.

Endpoints wired: documents (upload/list/get/delete), chunks (single + per
document), `/search`, `/stats`, `/ask`, `/research`, and all of `/sessions`
including `/messages` and `/stream`. `owner_id` also threads through
`ResearchState` so agent retrieval is scoped.

**To switch on:** put `SUPABASE_ANON_KEY` in `.env` (the URL is already there)
and `docker compose up -d api web`.

### Why dependencies, not middleware

FastAPI's `Depends(current_user)` replaces the Express-style
"middleware decodes token, attaches `req.user`" pattern. It is typed, appears
in `/docs`, can be swapped per route, and is **cached per request** — so
`/sessions/{id}/stream`, which needs the user for ownership, retrieval scoping
and persistence, verifies the token once.

That cache is a fresh dict created inside each request's own execution, not a
global keyed by user, so cross-user reuse is structurally impossible.
Contextvars behave the same way (per-task, copy-on-write). **Measured:** 10
concurrent requests produced 10 distinct `request_id` values.

Middleware is used for exactly one thing — the mechanical, universal,
must-run-first job of clearing the log context and assigning a request id.
Identity cannot go there: middleware runs *before* dependency resolution, so
it cannot see a decoded token.

Logging identity therefore works via a contextvar bound inside
`current_user`, merged by `structlog.contextvars.merge_contextvars`, so every
line in a request carries `owner_id` and `request_id` without threading a user
object through every signature. `X-Request-ID` is returned as a response
header so a reported problem can be grepped straight out of the logs.

### Two bugs found while building this

1. **`PyJWKClientError` is not a `jwt.InvalidTokenError`.** A token with an
   unknown or missing `kid` escaped both handlers and surfaced as a **500
   instead of a 401**. Found by probe, not by reading. Also added a distinct
   **503** for `PyJWKClientConnectionError` — a JWKS outage is our failure, and
   returning 401 would sign users out over a network blip.
2. **npm lockfile drift broke the Docker build.** `@supabase/supabase-js`
   published a new patch between `npm install` and `npm ci`, so the lock and
   `package.json` disagreed and `npm ci` refused. Re-run `npm install` on the
   host to resync before rebuilding the web image.

### MCP interaction — design now, not later

An MCP client cannot run a browser OAuth flow, so it needs a **long-lived API
key**: an `api_keys` table (hashed, `researchdesk_sk_...` prefix) accepted by
the same `current_user` dependency as an alternative to a JWT. Designing it
alongside auth avoids retrofitting the auth layer.

---

## 11. Other agreed-but-unbuilt work

**Hybrid search (BM25 + RRF)** — the highest-value retrieval improvement left.
Vector search fails on exact values: querying `167.2` beat the runner-up by
**0.007**, versus 0.102 for a conceptual query. Postgres does BM25-ish ranking
natively (`ts_rank_cd` plus a GIN index on `to_tsvector('english', text)`),
chunk text is **already stored there for this purpose**, and
`reciprocal_rank_fusion()` in `services/retrieval.py` is written and reusable.
Roughly 40 lines. Do it as its own step, with before/after numbers.

**HyDE / multi-query improvements** — multi-query is built but showed no gain at
11 chunks. Re-measure once the corpus is larger.

**Usage dashboard** — deliberately skipped; needs a `usage_events` table to be
more than a placeholder. Worth building against real traffic. The valuable
metrics: Gemma calls today / 14,400, embedding calls / 1,000, TPM headroom,
average iterations per question, and % of answers with zero citations.

**UI items raised and not done:**
- Per-answer scope provenance (which documents were searched for *that* turn)
- Scope shown on sidebar session rows
- "Ask about these" from the Library → new pre-scoped session

**Known rough edge:** retrieval is phrasing-sensitive and Gemma is not
deterministic. "Which segment was flat?" failed once (draft cited nothing, critic
accepted it) and passed on a later run. `critique` now treats zero citations as
a phrasing failure and asks for differently-worded queries — but **that fix has
not been observed firing**, only reasoned about. Cost: ~5 Gemma calls instead of
3 for genuinely-absent answers.

---

## 12. Material 3 — what was done and what remains

Converted from the custom navy/sky system to Material 3. Decision record:

**Why not MUI:** verified 2026-09-02 — MUI v9.4.0 is still **Material Design
2**. M3 is an open tracking issue (mui/material-ui#29345), M3 Expressive
another (#46148). It would have delivered the wrong visual language plus
Emotion, and made Tailwind redundant.

**Why not `@material/web`:** authentic M3 and actively published, but the
README says *"maintenance mode pending new maintainers"*, and it is Lit web
components with no React guidance.

**So:** M3 spec implemented on the existing Tailwind + component-layer
architecture.

| Piece | Where |
|---|---|
| Colour roles, generated from source `#0284c7` | `scripts/gen-theme.mjs` → `app/theme.css` (committed) |
| Type scale, shape, elevation, state layers, motion | `app/globals.css` |
| Ripple + React primitives | `app/md.tsx` |

Regenerate the palette with:

```powershell
cd frontend
node --import ./scripts/register-loader.mjs scripts/gen-theme.mjs
```

The loader hook is required: `material-color-utilities` ships extensionless
ESM imports that Node's resolver rejects, and
`--experimental-specifier-resolution=node` was removed in Node 20.

### The documented spec departure

**The navigation drawer stays navy.** Stock M3 makes it a light `surface`. It
uses its own fixed `--md-nav-*` roles, set by hand in the generator rather than
derived from the palette. Everything inside it still follows M3 anatomy: 56px
items, pill active state, state layers, ripple.

### Verified

- ripple injects a span on pointerdown, sized to reach the furthest corner (measured 298px on a 40px-tall button)
- buttons are 40px pills (`border-radius: 9999px`)
- Roboto applied, `--md-primary` = `#006398`, `--md-nav-surface` = `#071626`

### Material gotchas hit

- **Tailwind v4's PostCSS plugin cannot resolve a bare package specifier in `@import`** — `@import "@fontsource/roboto/400.css"` fails. Import fonts from `layout.tsx` (JS) so webpack resolves them.
- **Old tokens leave transparent surfaces.** After the token rename, `command-palette.tsx` and `chunk-panel.tsx` still referenced `var(--surface)` etc. Undefined variables silently produce transparent backgrounds — it looked like a hot-reload bug. Grep for old token names after any rename.

### Not yet converted to M3

- **Snackbar** — errors are still inline error-container blocks. M3 would use a Snackbar for transient failures.
- **Menus** — no dropdown menus exist yet; when they do, use M3 Menu (4px radius, elevation 2).
- **Top app bar** — the chat header approximates a small top app bar but does not implement the scroll-elevation behaviour.
- **`md-badge`** is an M3-flavoured label container, not the spec's Badge (which is a small numeric dot on an icon).

## 13. MCP server — agreed, not started

The user wants an **MCP server exposing this app**, as a learning exercise.

### Why it fits

The app's capabilities are already clean, typed service functions with no HTTP
coupling — `retrieve()`, `answer_question()`, `run_agent()`,
`list_documents()`. An MCP server is a thin second adapter over the same
services layer that `api/` already wraps. That is the payoff of the
`services/` vs `api/` split.

### Proposed shape

`backend/app/mcp/server.py`, using the official **Python MCP SDK** (`mcp`
package) over stdio for local clients such as Claude Desktop / Claude Code.

**Tools** (actions):

| Tool | Wraps |
|---|---|
| `search_documents(query, top_k, document_ids?)` | `services.retrieval.retrieve` |
| `ask_documents(question, top_k?)` | `services.rag.answer_question` |
| `research(question)` | `agent.graph.run_agent` — the full agent |
| `list_documents()` | `services.ingest.list_documents` |
| `get_chunk(chunk_id)` | direct Postgres read |

**Resources** (readable context, not actions):

| Resource URI | Content |
|---|---|
| `researchdesk://documents` | the library index |
| `researchdesk://document/{id}` | one document's chunks |
| `researchdesk://stats` | Postgres/Qdrant counts |

**Prompts** (reusable templates): a `summarise_document` and a
`compare_documents` prompt, which is the MCP feature most people skip and the
one worth learning.

### Decisions to make first

1. **stdio or HTTP/SSE transport?** stdio is simpler and works with local
   clients; HTTP/SSE is needed if the server must run alongside the deployed
   API. Start with stdio.
2. **Does the MCP server import the services directly, or call the HTTP API?**
   Direct import is cleaner and faster but ties the MCP process to the same
   container and database credentials. Calling the API keeps them independent
   and works against the deployed instance. **Recommend calling the HTTP API**,
   so the MCP server can point at either local or production.
3. **Auth interaction.** Once auth lands, an MCP server needs a token. A
   long-lived API key on the user record is the usual answer — worth designing
   at the same time as auth rather than after.

### Learning value

MCP is the current standard for exposing tools to LLM clients, and building a
server is the fastest way to understand the tools/resources/prompts
distinction. It also inverts the project: so far the app *calls* a model;
here the app *is* the tool a model calls.

## 14. Working style the user has asked for

- **Terse.** They read diffs and outputs themselves.
- **Direct recommendations** — one line on why, not a survey of options.
- **Real values in commands**, never placeholders.
- **No emoji in the UI.** Stated explicitly. Icons are inline SVG.
- **Never commit or push** unless told to in that same message.
- **Never change anything in GCP** — read-only always.
- PowerShell syntax for commands (Windows 11, PowerShell 5.1).
- They say when to move to the next step; don't run ahead.
- They ask many "how does this work" questions — the explanations are the
  point of the project, not a distraction from it.

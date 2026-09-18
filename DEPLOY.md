# Deploying Research Desk

Four providers, all on permanent free tiers, none asking for a card.

| Piece | Provider | Free tier | Why this one |
|---|---|---|---|
| Frontend | **Vercel** | Hobby, unlimited | Next.js's own host. Zero config, and `/chat/[id]` needs on-demand SSR, which static hosts can't do. |
| Backend | **Render** | Web Service, 512 MB | Runs a Dockerfile directly. `backend/Dockerfile` already has a `prod` target that honours Render's `$PORT`. |
| Postgres | **Neon** | 0.5 GB | Its **direct endpoint is IPv4-reachable**, so this app can skip connection pooling entirely — which deletes the prepared-statement trap instead of working around it (see trap 1). Also gives free DB branching, which is how preview environments get their own data. |
| Vectors | **Qdrant Cloud** | 1 GB cluster, forever | Same API as the local container, so only `QDRANT_URL` + `QDRANT_API_KEY` change. |
| CI | **GitHub Actions** | 2000 min/mo private, unlimited public | Runs checks only; see below. |

## Environments: local + prod, deliberately

The textbook answer is dev → staging → prod. The honest modern answer is that
long-lived staging has largely been displaced by **per-PR preview
environments**, and the common shape is:

| Environment | Who really has it | What justifies it |
|---|---|---|
| **Local** | everyone | `docker compose up` |
| **Preview**, per PR, ephemeral | most current teams | integration bugs with zero config drift; destroyed on merge |
| **Staging**, long-lived prod mirror | teams with a specific reason | rehearsing destructive migrations at production scale, third-party *sandbox* accounts, a QA team, compliance sign-off |
| **Prod** | everyone | |

**This project runs local + prod.** Staging earns its place when something
genuinely cannot be rehearsed in a preview; nothing here qualifies, and a second
long-lived environment means every free tier duplicated and kept in sync.

What actually makes adding environments later cheap is not the count — it is
that all configuration arrives through env vars rather than code, which is
already true. So this is a reversible decision, not a fork in the road.

### The preview-environment footgun

Vercel creates a preview deployment for **every branch and PR whether you ask
for it or not**. Env vars set without an environment scope apply to previews
too, which means a preview frontend will happily talk to the **production**
backend and write to production data.

So a second environment already exists. Either scope
`NEXT_PUBLIC_API_URL` to Production only, or give previews their own backend and
a **Neon database branch** — copy-on-write, instant, and the reason Neon's
branching is listed as a selection criterion above.

## Monorepo: yes, and it is the right shape here

Strictly this is "one repo, two apps" rather than a true monorepo — `backend/`
and `frontend/` each own their Dockerfile and `.dockerignore`, and no workspace
tooling links them. That is the easy case, and it is standard at this size.

Deployment works because both platforms take a **Root Directory** (`backend` for
Render, `frontend` for Vercel). Three things do bite:

1. **Every push rebuilds both apps.** A backend-only commit redeploys the
   frontend for nothing. Scope it with Vercel's *Ignored Build Step* and
   Render's *Build Filters*.

2. **Do not add `on.paths` filters to `ci.yml` if those checks are required.**
   GitHub reports a *skipped* job as pending, never as passed, so a
   frontend-only PR would block forever waiting on a `backend` check that will
   never run. `ci.yml` has no paths filter on purpose. If path-scoped CI is
   wanted later, use `dorny/paths-filter` **inside** a single always-running
   job, so the job always reports a conclusion.

3. **Build-root inference.** Next.js hunts upward for a lockfile and will warn
   about stray ones outside the repo. Cosmetic locally; the Root Directory
   setting makes it moot on Vercel.

## Why CI does not deploy

Vercel and Render both watch the repo and build on push by themselves. If
Actions also deployed, there would be two build paths that could disagree, and
the one that broke would be the one nobody tested. So `.github/workflows/ci.yml`
answers only *"should that deploy be allowed?"* — ruff, `tsc --noEmit`, a real
`next build`, and a build of the exact prod Docker image Render will build.

To make it a true gate rather than a notification: GitHub → Settings → Branches
→ protect `main` → require the `backend`, `frontend` and `docker` checks. Then
work on branches and merge via PR. Pushing straight to `main` runs CI *after*
the deploy has already started.

## Order of operations

1. **Rotate the Google AI Studio key** — prudent, not urgent. Verified: the
   original key never entered git history (`git grep` across `rev-list --all`
   finds nothing; `env.example` was sanitised to `REPLACE_ME` before the first
   commit). Its only exposure is the local `.env` and the chat transcript it was
   pasted into, so this is housekeeping rather than an incident. Rotate before
   the repo goes public, and thereafter keep the key only in Render's env store.
2. Create the GitHub repo **private**. Public only after step 6.
3. Neon: create a project, copy the **direct** connection string (the one
   without `-pooler`), and swap its `postgresql://` prefix for
   `postgresql+asyncpg://` — SQLAlchemy selects its driver from that prefix.
4. Qdrant Cloud: create a cluster, copy its URL + API key.
5. Render: New → Web Service → repo → Root Directory `backend`, Runtime
   Docker, Dockerfile Path `./Dockerfile`, and set **Docker target** to `prod`.
   Health check path `/health`.
6. Vercel: New Project → repo → Root Directory `frontend`. It detects Next.js.
   Scope `NEXT_PUBLIC_API_URL` to **Production** only, or previews will write to
   production data — see the preview footgun above.
7. **Add the access gate before making the repo or URL public** — see below.

Tables are created by `create_all` on startup (`app/db/session.py`), so the
first deploy needs no migration step. That stops being true the moment a model
changes — `create_all` cannot alter an existing table. Alembic is still
deferred, and R2's `documents.storage_key` column is the change that will force
the issue.

## Environment variables

**Render** (backend) — everything from `env.example` except the Supabase anon
key, which the backend never reads, plus:

```
DATABASE_URL     postgresql+asyncpg://...   # Neon DIRECT endpoint, no `-pooler`
QDRANT_URL       https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY   <from Qdrant Cloud>
GOOGLE_API_KEY   <the ROTATED key>
SUPABASE_URL     https://<ref>.supabase.co
CORS_ORIGINS     https://<your-app>.vercel.app
ALLOW_CLAIM_UNOWNED  false

STORAGE_BACKEND       s3                    # `local` LOSES FILES on Render
S3_ENDPOINT_URL       https://<ref>.supabase.co/storage/v1/s3
S3_ACCESS_KEY_ID      <Storage -> S3 -> New access key>
S3_SECRET_ACCESS_KEY  <same page, shown once>
S3_BUCKET             research-desk
S3_REGION             <shown on the S3 page>
```

`STORAGE_BACKEND=s3` is not optional on Render, and it fails in the least
useful way if forgotten. Render's free tier has **no persistent disk**, so
`local` writes originals into the container filesystem and every deploy,
restart and idle spin-down throws them away. Nothing errors: uploads succeed,
text indexes, answers cite correctly — and then `/documents/{id}/file` 404s on
files that were there an hour ago.

The S3 keys are **not** the Supabase anon or service keys. They come from
Storage → S3 → New access key, and the secret is shown once.

**Vercel** (frontend):

```
NEXT_PUBLIC_API_URL            https://<your-api>.onrender.com
NEXT_PUBLIC_SUPABASE_URL       https://<ref>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY  sb_publishable_...
API_URL_INTERNAL               https://<your-api>.onrender.com
```

`NEXT_PUBLIC_*` is inlined into the browser bundle at build time, so changing
one requires a **redeploy**, not just a restart. And nothing secret may ever
carry that prefix.

`API_URL_INTERNAL` is easy to miss and fails quietly. `lib/api.ts` keeps two
base URLs because under docker compose a server component reaches the API at
`http://api:8000` over the compose network while the browser can only reach
`localhost`. **On Vercel there is no private network to the Render service**, so
both must be the same public URL. Only `getHealth()` uses the server-side base
today, and it swallows errors into `{ error: "unreachable" }` — so forgetting
this variable shows up as a permanently red health indicator and nothing else,
with the frontend still fully working.

## Traps, in the order they will bite

### 1. Use Neon's DIRECT endpoint, not its pooled one

Neon hands out two connection strings. Take the one **without** `-pooler` in the
hostname.

The pooled endpoint is PgBouncer in transaction mode: it gives out a different
backend connection per transaction, so **prepared statements break**. This app
uses them via both of its drivers, and the failure mode is nasty — an
intermittent `prepared statement "__asyncpg_1__" does not exist` that appears
under load rather than on the first request, so it reads like a capacity problem
instead of a configuration one.

Two drivers, one database: SQLAlchemy uses **asyncpg**, and LangGraph's
`AsyncPostgresSaver` uses **psycopg3**. Both would need disabling separately
(`statement_cache_size=0` for asyncpg, `prepare_threshold=None` for psycopg), so
avoiding the pooler is one decision instead of two.

The direct endpoint is safe here only because the connection budget is tiny —
`pool_size` 5 + 2 (see "Also do these") against a cap in the hundreds. An app
with many replicas would need the pooler and would have to pay the
prepared-statement cost.

**Why not Supabase Postgres, given it is already in the stack for auth:** its
direct connection is **IPv6-only**, and Render's free egress is IPv4, so it
forces you onto Supavisor and straight back into the problem above. Its free
tier also **pauses a project after about 7 days of inactivity**, meaning a demo
you open occasionally needs manually unpausing first. Neon suspends compute too
but resumes on the next query in ~a second. Supabase stays as the *identity
provider* only.

Free-tier numbers (storage, compute hours, pause windows) drift — the
architectural facts above are stable, but re-check the quotas on the pricing
page before relying on one.

### 2. Render free spins down after 15 minutes idle

The next request pays a cold start of roughly 30–60s. The chat endpoint is SSE,
so a cold start looks exactly like a hung stream. Either warm it with a cron
ping to `/health`, or have the UI say so — do not debug a "broken stream" that
is a sleeping container.

### 3. Supabase Auth redirect allowlist

Authentication → URL Configuration: set **Site URL** to the Vercel domain and
add `https://<app>.vercel.app/auth/callback` to **Redirect URLs**. Until then
sign-in bounces back with `?error=...`, which the login page now displays.

Vercel preview deploys get a new hostname per commit, so previews will not sign
in unless the wildcard `https://<project>-*.vercel.app/**` is added too.

### 4. CORS

`CORS_ORIGINS` must be the exact Vercel origin, scheme included and no trailing
slash. A mismatch shows up as a browser CORS error while `curl` works fine.

### 5. The access gate is not optional

**The Gemma free quota is per API key, not per user.** A public URL means one
crawler or one curious visitor can exhaust the entire daily budget for
everybody, and Google gives no per-caller breakdown to diagnose it with.

Supabase Auth already gates the UI, but anyone can still sign up. Before the URL
is public, restrict signups — Supabase Authentication → Sign In / Providers
supports an allowlist by email domain, or disable public signup and invite
accounts manually. Cheapest correct option for a learning demo: leave signups
off and invite yourself.

## Also do these while deploying

Carried over from the older plan in `HANDOFF.md` §8; still all valid.

- **Lower both connection pool sizes** — SQLAlchemy 10 → 5, psycopg 5 → 2. Free
  Postgres tiers cap connections, and this app opens two independent pools
  (asyncpg for SQLAlchemy, psycopg3 for the LangGraph checkpointer) so it uses
  twice what a single-driver app would.
- **Disable `/docs` and `/redoc`** in prod, or put them behind the gate. They
  document every endpoint, including `/auth/claim`.
- **Re-ingest the corpus.** Qdrant Cloud starts empty and embedding runs at
  ~133 chunks/min, so budget real minutes for this, not seconds.
- **Keep `/health` reachable** for the keep-alive pinger from trap 2.

## Cost of going public

Making the repo public is genuinely free-er (unlimited Actions minutes vs 2000).
It is safe once the rotated key is only ever in Render's and Vercel's env var
stores — but it also publishes the git history, so anything committed once is
public forever. That is the reason step 1 is a rotation and not a deletion.

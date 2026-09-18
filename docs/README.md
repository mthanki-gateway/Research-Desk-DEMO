# Documentation

Three apps, one stack. Each page below follows the same four headings:

- **The idea** — what it is, in plain words
- **Notable decisions** — the choices that were not obvious, and why
- **The tech**
- **How it works** — the actual implementation

| | |
|---|---|
| [Research Desk](research-desk.md) | Chat, Library, Lab, Atlas — the RAG app |
| [Parley](parley.md) | Speak, Interview, Howler — the voice apps |
| [Model Lab](model-lab.md) | Playground, Transcribe — calling models directly |
| [Storage](storage.md) | Where original files live, and what is planned for audio |
| [Background jobs](jobs.md) | The queue, and why it is a Postgres table |

Other documents in the repo, which these do not replace: `README.md` (setup and
running), `DEPLOY.md` (hosting), `HANDOFF.md` (design decisions and open
threads), `CLAUDE.md` (conventions this code follows).

---

## The shared stack

Everything below is used by more than one app. Anything app-specific lives on
that app's page.

```
  Next.js (App Router, React 19, Tailwind v4, Material 3)
        │  HTTP + WebSocket
  FastAPI (async, Pydantic settings, structlog)
        │
  ├── Postgres      documents, chunks, conversations, projects, checkpoints
  ├── Qdrant        vectors
  ├── Gemini        embeddings, generation, native-audio Live
  ├── Serper        web search
  └── Storage       original uploads (local volume, or any S3-compatible store)
```

**Each app wears its own accent** — Research Desk purple, Parley brown, Model
Lab blue — so which one you are in is legible before you read a word. It is a
DEFAULT, not a lock: somebody who picks an accent explicitly gets it
everywhere, because that is a preference about their eyes rather than about
the app they happen to have open. Accents are generated from a single seed
each by `frontend/scripts/gen-theme.mjs`; the output is committed so the app
ships static CSS with no runtime colour maths.

**One process, several apps.** Research Desk, Model Lab and Parley share one
API, one database and one login. They are different doors onto the same corpus,
not separate products. Adding an app is an entry in
`frontend/app/projects.ts` plus its pages — the drawer, the app switcher and
active-item highlighting all read from that one registry.

**Ownership is explicit, never ambient.** Every query that touches user data
takes `owner_id` as an argument rather than reading it from a context variable.
Aggregates especially: a count that quietly includes another tenant's rows
still looks perfectly plausible.

**Secrets stay server-side.** The browser calls our API; our API calls the
providers. No key is ever in a bundle.

**Schema changes are additive and idempotent.** There is no Alembic. Column
additions are `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements in
`_MIGRATIONS` (`backend/app/db/session.py`), run at boot after `create_all`.
This works because every addition so far has been nullable or defaulted. A
change that renames or drops a column is the point at which a migration tool
becomes necessary, and that is a deliberate line rather than an oversight.

**Boot cannot hang silently.** `create_tables()` and `init_checkpointer()` run
behind `_boot_step()` with a timeout, and failures are recorded and reported by
`/health` rather than preventing the server from binding its port. A dependency
that is down should make the app say so, not make it look dead.

# Research Desk

A document-grounded research agent, built as a learning vehicle for the current
AI-app stack: **FastAPI + LangGraph + Qdrant + Next.js**, entirely in Docker.

> **Resuming work?** Read [HANDOFF.md](HANDOFF.md) first. It carries the current
> status, the file map, the decisions already settled, every trap already hit,
> and the agreed plans for deploy / file storage / auth. This file explains the
> project; HANDOFF explains where the work stopped.

You ask a question about your uploaded documents. Rather than one-shot RAG, a
LangGraph agent plans sub-questions, retrieves, drafts, **critiques its own
answer, and re-retrieves if the evidence is thin** — then streams a cited
answer to the UI.

## Why these choices

| Choice | Reason |
|---|---|
| **Qdrant** over Pinecone | Pinecone can't run locally, which breaks the "install nothing but Docker" goal. Qdrant runs in compose *and* has a free managed cloud tier — identical client code both places. |
| **Qdrant** over pgvector | Fewer concepts to learn, but you'd miss payload filtering, quantization, named vectors — the things a dedicated vector DB teaches. |
| **Gemma 4** as the reasoning model | On the AI Studio free tier Gemma 4 gets 30 RPM / **14.4K requests per day**; Gemini 3.x Flash gets **20 requests per day** — about three agent questions. Gemma isn't a compromise here, it's the only workable option. The trade-off is no function-calling (see below). |
| **Gemini Embedding** for vectors | Gemma has no embedding endpoint. Same key, 100 RPM / 1K req-per-day. A local `fastembed` provider sits behind the same interface for offline / post-quota work. |
| **LangGraph** over plain LangChain | The critique→re-retrieve **cycle** is the whole point. LangChain chains are DAGs; they can't loop. |
| **Next 16**, not 15 | The plan said 15, but 15.1.6 carries CVE-2025-66478 and 16.3.4 is current stable. Upgraded during step 2. |

## Verified against a real key (2026-09-01)

Probed before writing agent code, because two of these would have been painful
to discover mid-build:

| Question | Answer |
|---|---|
| Real Gemma ids | `models/gemma-4-26b-a4b-it` (MoE, 4B active, fast) and `models/gemma-4-31b-it` (dense, stronger, but 503'd under load) |
| Does Gemma do function-calling? | **No.** Given `functionDeclarations` it returns prose narrating what it would call. No `functionCall` part. |
| Then how does the agent take structured steps? | `generationConfig.responseSchema` — strict JSON — **works reliably.** This replaces tool-calling throughout. |
| Bare `responseMimeType: application/json`? | **Unusable.** Without a schema it emitted its own reasoning trace instead of JSON. Always pass a schema. |
| `systemInstruction`? | Supported. |
| Embedding dimensions | Natively **3072**; `outputDimensionality: 768` is honoured. We use 768 (4x less storage) and re-normalise client-side, since Google only L2-normalises the full-width vector. |

That third row is the notable one: on this stack the agent's control flow is
driven by JSON schemas, not tool-calls. Same LangGraph concepts either way —
nodes, edges, cycles — just a different mechanism for getting typed output.

## Free-tier limits that shape the design

Gemma 4's ceiling is **~16K tokens per minute**. One naive prompt stuffed with a
dozen chunks would spend the entire minute's budget on a single call. So:

- `RETRIEVAL_TOP_K=5` and `CHUNK_SIZE=900` keep prompts lean
- a token-bucket limiter throttles ahead of the API rather than eating 429s
- embedding calls batch (`EMBEDDING_BATCH_SIZE=64`) to stay inside 1K req/day

## Quickstart

```powershell
Copy-Item env.example .env
# paste your key from https://aistudio.google.com/apikey into GOOGLE_API_KEY
docker compose up --build
```

- Frontend — http://localhost:3000
- API docs — http://localhost:8000/docs
- Qdrant dashboard — http://localhost:6333/dashboard
- Langfuse — http://localhost:3001 (needs `--profile obs`)
- pgweb — http://localhost:8081 (needs `--profile tools`)

Or open every one that is running, in your browser, in one go:

```powershell
.\scripts\dev-open.ps1        # open what is already up
.\scripts\dev-open.ps1 -Up    # start the default stack first, then open
```

`scripts\dev-open.cmd` is the same thing, double-clickable from Explorer. It
probes each port and opens only the services that answer — Langfuse and pgweb
live behind compose profiles, so opening all five blindly would usually give
you two connection-error tabs — and it prints the `docker compose` command for
anything that is not running.

Model ids drift between releases. Confirm the ones your key actually serves and
paste real values into `.env`:

```powershell
docker compose exec api python -m app.scripts.list_models
```

## Step 2 notes — what ingestion actually does

`POST /documents` returns immediately with a document id and embeds in the
background; the UI polls and shows `embedding 142/300`. Holding the request open
would time out, since embedding runs at ~133 chunks/minute.

Pipeline: `parsing.py` (pypdf, page numbers kept for citations) →
`chunking.py` → `embeddings.py` (batched, rate-limited) → `vectorstore.py`
(Qdrant) plus a `chunks` row in Postgres. Chunk UUIDs double as Qdrant point
ids, so one identity spans both stores.

### Chunking on section boundaries — the one non-obvious decision

The first version split purely on a 900-char window. Retrieval was visibly bad:
asking *"what are the main risks?"* returned the **headcount** section, because
a 900-char window swallowed the tail of one section plus the head of the next,
and the blended embedding landed near neither topic.

Fixing it took two changes in `chunking.py`:

1. markdown headings are **hard** split boundaries — a chunk never spans two
   sections
2. each chunk is **prefixed with its own heading**, so an isolated chunk still
   states its context and `## Risks` sits next to the risk text

Measured on the sample report, top hit for "main risks":

| | top hit | score |
|---|---|---|
| before | Headcount and Operations ✗ | 0.625 |
| after | Risks ✓ | 0.710 |

Every test query returns the correct section after the change. This is the
single biggest quality lever in the whole app — worth re-reading that file
before tuning anything else.

### How retrieval actually behaves (measured, not theorised)

Gemma is **not** involved in search. `gemini-embedding-001` maps text to 768
numbers and Qdrant compares angles. No language model reads the query.

Tested against a document containing sections on Giza, mummification, Norwegian
cod and Roman aqueducts — in which the words *Egypt* and *pyramid* never appear:

| query | top hit | score | note |
|---|---|---|---|
| `Egypt` | Great Structures of Giza | 0.654 | word absent from corpus entirely |
| `pyramid` | Great Structures of Giza | 0.658 | word absent from corpus entirely |
| `How were the pyramids of Egypt built?` | Great Structures of Giza | **0.697** | longer query beats bare keyword |
| `preserving a dead body` | Mummification | 0.699 | zero shared words with the text |
| `how did they move water downhill` | Roman Aqueducts | 0.732 | zero shared words |
| `dried fish trade` | Norwegian Cod | 0.714 | |
| `staff numbers` | Headcount (other document) | 0.629 | picked the right document |

So: **concept-based, not keyword-based**, and fuller questions retrieve better
than single words.

#### The failure modes that shape the design

| query | top hit | score | 2nd | margin |
|---|---|---|---|---|
| `preserving a dead body` | Mummification ✓ | 0.699 | 0.597 | **0.102** |
| `167.2` | Headcount ✓ | 0.615 | 0.608 | **0.007** |
| `quantum computing error correction` | Risks ✗ | 0.570 | 0.557 | 0.013 |
| `What is the CEO salary?` | Financial Summary ✗ | 0.584 | 0.575 | 0.009 |

Three conclusions, each with a design consequence:

1. **Exact values barely discriminate.** `167.2` beat the runner-up by 0.007,
   versus 0.102 for a conceptual query. Embeddings encode *meaning*, and one
   number means much the same as another. → this is the case for **hybrid
   search** (BM25 over the Postgres `chunks.text` fused with vector results).
   Chunk text is stored in Postgres precisely to enable it.

2. **There is no "no results".** A query about quantum computing scored 0.570
   against a corpus with no computing content — *higher* than several correct
   matches elsewhere. Absolute scores overlap between right and wrong answers,
   so **you cannot threshold on score** to detect irrelevance. → this is the
   case for the **critique node** in step 4: something has to read the
   retrieved text and judge whether it actually supports an answer. Cosine
   distance cannot.

3. **Negation isn't a retrieval problem.** `Which segment did NOT grow?`
   correctly returned "Revenue by Segment" (0.687) — the section containing the
   answer. Whether the model then correctly identifies Hardware as the flat one
   is a *generation* problem, for step 3 onward.

## Step 3 notes — the baseline, and where it actually breaks

`POST /ask` → retrieve top_k once → one Gemma call → cited answer.
`services/rag.py` is ~90 lines and stays in the repo permanently as the thing
to compare the agent against.

### Gemma has no thinking channel — always constrain the output

Asked for prose, Gemma 4 wrote its **entire reasoning trace** into the reply:
constraint checklists, three drafts, a "Self-Correction during drafting"
section, then ran out of tokens mid-sentence. The user-visible answer was
buried inside it.

Passing a `responseSchema` (`{answer, sources_used}`) fixed it completely —
reasoning has nowhere to go except the declared string field. Same lesson as
the earlier JSON-mode finding, now confirmed twice:

> **With Gemma, structure the output or get a monologue.**

Every LLM call in this app should therefore go through `generate_json` with an
explicit schema. `generate` (raw text) exists only for step 5's streaming,
where the tradeoff has to be revisited.

### The baseline is better than expected

Predictions that turned out wrong — it handled all of these correctly:

| question | result |
|---|---|
| `What is the CEO salary?` (absent from corpus) | *"The provided documents do not contain this information."* ✅ no hallucination |
| `Which segment did not grow in 2024?` | correctly identified Hardware as flat ✅ handled negation |
| `What drove revenue growth?` | Cloud Platform, exact figures, cited [1][2] ✅ |

So plain RAG plus a firm prompt is a genuinely decent single-hop system. The
case for an agent is **not** hallucination prevention.

### Where it does break: multi-hop

One retrieval pass means one chance to gather every needed fact.

| top_k | question | answer |
|---|---|---|
| 5 | operating income **and** 2025 capex | both facts, correct ✅ |
| 1 | same question | *"...do not contain this information regarding the planned capital expenditure"* ❌ |

The capex figure **is** in the document — in a section that wasn't retrieved.
No amount of prompting fixes this, because the model never sees the text.

Why top_k=5 looks perfect here: this corpus is 6 chunks, so top-5 returns 83%
of the document. On a realistic 300-chunk corpus top-5 is 1.6% — much closer to
the top_k=1 case. **Test with top_k=1 to simulate real scale**, which is what
the UI control is for.

That is the argument for step 4: plan sub-questions, retrieve per sub-question,
then judge whether the evidence actually supports an answer.

## Step 4 notes — the LangGraph agent

`POST /research`. Same inputs as `/ask`, so the two are directly comparable on
one question; the `agent` checkbox in the UI switches between them.

```
START → plan → retrieve → draft → critique ─┬→ END
                  ▲                         │
                  └──────── retry ──────────┘
```

| file | role |
|---|---|
| `agent/state.py` | `ResearchState` + the reducers that merge node output |
| `agent/nodes.py` | the four nodes, each schema-constrained |
| `agent/graph.py` | wiring, the conditional edge, `run_agent()` |

### It beat the baseline on the target case

At `top_k=1`, *"What was operating income in 2024, and what capital expenditure
is planned for 2025?"*:

| | answer |
|---|---|
| `/ask` baseline | operating income ✓, capex *"documents do not contain this information"* ❌ |
| `/research` agent | both figures, correct, cited ✅ |

The agent split the question, retrieved once per part, and accumulated both
chunks — where the baseline's single pass could only ever see one.

### The cycle, demonstrated

A four-part question against `agent_max_subquestions=3` forces the loop, because
the planner can only cover three parts:

```
plan      → 3 sub-questions (4th part dropped by the cap)
retrieve  → 3 chunks
draft     → answers 3, reports unanswered: "Norwegian stockfish designation"
critique  → sufficient=FALSE, missing=[stockfish query]     ← loops back
retrieve  → 4th chunk
draft     → all 4 facts, cited [1,2,3,4]
critique  → sufficient=TRUE                                 ← exits
```

All four facts correct, 2 iterations, still at `top_k=1`. `GET /research`'s
`trace` field records this, and the UI renders it under "Graph path".

### Two bugs worth remembering

**1. One bad schema field destroyed a good plan.** `PLAN_SCHEMA` originally had
a `reasoning` field. Gemma emitted four correct `sub_questions`, then degenerated
inside `reasoning` (`)$\text{planned...`), truncating the JSON — and strict
`json.loads` threw away the whole object, including the good plan. Fixes:

- **removed `reasoning`** — it was only ever logged. Every extra output field is
  another chance to derail; ask only for what you use.
- `extract_string_list()` in `llm.py` salvages a complete array from truncated
  JSON, and drops degenerate strings (low unique-word ratio ⇒ repetition loop).

**2. The critic couldn't tell "absent" from "not yet retrieved".** Its prompt
said an honest *"not in the documents"* counts as sufficient — so it approved a
draft that had failed to answer three of four parts. Two fixes:

- the prompt now states explicitly that **the sources shown are only what was
  retrieved so far, not the whole library**, so "not in these sources" means
  *search again*, not *give up*
- routing trusts the drafter's own `unanswered` list over the critic's
  inference — the drafter knows exactly what it couldn't support

### Loop safety

Three independent guards, because an unbounded self-critique loop is the
easiest way to burn a daily quota:

1. `agent_max_iterations=2` — a hard cap on cycles
2. `tried_queries` accumulates; a query already run is never re-issued
3. `sufficient=False` with no *untried* queries exits anyway

Cost: ~3 Gemma calls, ~5 with one retry, against 14.4K/day.

## Step 6 notes — chat sessions

```
POST   /sessions                     create (optional document scope)
GET    /sessions                     list, newest first
GET    /sessions/{id}                session + full transcript
PATCH  /sessions/{id}                retitle, change scope
DELETE /sessions/{id}
POST   /sessions/{id}/messages       ask a turn (blocking)
POST   /sessions/{id}/stream         ask a turn, SSE node progress
```

Verified: a session and its transcript survive `docker compose restart api`,
and sessions are auto-titled from their first question.

### Two stores, on purpose

| store | holds | why |
|---|---|---|
| `chat_sessions` / `messages` | the transcript | queryable, independent of LangGraph's state shape |
| `checkpoints*` (LangGraph) | graph state per thread | resume mid-run; `interrupt()` later |

Reading history out of checkpoint blobs would couple the UI to LangGraph
internals, so the transcript is ours and `chat_context` is passed *into* the
graph as a plain string.

**Messages never store retrieved chunk text** — only the question, the answer,
and citation pointers. Storing context would re-send it on every later turn,
which is the fastest way to exhaust a 16K-tokens/minute budget.

### The bug that mattered: reducers don't reset

`ResearchState` uses append reducers (`evidence`, `sub_questions`,
`tried_queries`, `trace`). Passing `[]` as input does **not** clear them — a
reducer applies to the input too, so `[]` appends nothing. Combined with one
`thread_id` per session, that meant:

- a one-part question reported "6 sub-questions"
- worse, **evidence retrieved for an earlier question leaked into later
  answers**, inflating tokens and risking wrong context

Fix: **one thread per turn**, `f"{session_id}:{turn}"`. The conversation lives
in `messages`, not in graph state, so the checkpointer's job is durability
*within* a run — not carrying the conversation. After the fix, `sub_questions`
and `evidence` stay constant per turn while history grows (0 → 122 → 208 chars).

### Checkpoint hygiene — one thread per *attempt*, deleted on success

Two iterations of the same bug, both measured rather than reasoned about.

**Per session** (first attempt): append reducers made `evidence` and
`sub_questions` grow across turns, so an earlier question's chunks leaked into
a later answer.

**Per turn** (second attempt): still wrong. A *failed* turn persists no
messages, so a retry computes the same turn index and lands on the same thread.
Probed with `app/scripts/probe_resume.py`, invoking one thread twice:

| channel | attempt 1 | attempt 2 |
|---|---|---|
| `sub_questions` | 1 | **2** |
| `tried_queries` | 1 | **2** |
| `trace` steps | 4 | **8** |
| `evidence` | 2 | 2 (saved only by the dedupe reducer) |

Polluted `tried_queries` is the dangerous one: `critique` refuses to re-issue a
query it believes was already tried, so a legitimate retry gets blocked.

**Now:** `f"{session_id}:{turn}:{uuid4().hex[:8]}"` — every attempt starts
clean — and `discard_thread()` deletes the thread once the answer is committed
to `messages`. Verified: checkpoint count stays flat at zero across turns
instead of growing ~6 rows each.

Storage before and after, same database:

| | rows | size |
|---|---|---|
| checkpoint tables, ~25 turns, no cleanup | 990 | **920 KB** |
| all real app data (messages, docs, chunks, sessions) | 53 | 248 KB |
| checkpoint tables, with cleanup | 0 | 72 KB |

Checkpoints were **4x larger than the actual application data**. Use
`app/scripts/prune_checkpoints.py` to clear rows left by crashes or older
versions.

Deleting is safe because nothing reads a checkpoint after its turn finishes:
the transcript, citations and the full agent trace are all in `messages`.

### History strategy (layers 1-3 of 4)

`services/history.py`. Budget reasoning: Gemma is 16K tokens **per minute**, and
the agent makes 3-5 calls per question, so each call must stay near 3.5K tokens
— leaving ~1.5K for history, about 8-10 plain turns.

1. system prompt + retrieved chunks — always
2. **rolling summary** of evicted turns — incremental (summarises *previous
   summary + newly evicted turns*, never the whole history), so cost stays flat
3. **verbatim last 3 exchanges** — uncompressed; this is what pronouns and
   follow-ups resolve against
4. semantically retrieved old turns — **not implemented**, see
   `retrieve_old_turns()` for the seam

Measured: 12 messages, 4 folded into the summary, and `established_facts`
retained the figures —

```
- 2024 operating income was $112.8 million.
- 2024 gross margin was 62.1%, up from 58.7%.
```

Splitting `summary` from `established_facts` in the schema is what preserves
specifics; a prose blob degrades to "discussed revenue".

### Follow-up resolution happens in `plan`

Retrieval is stateless, so *"and how does that compare to the prior year?"* must
become a standalone query before it reaches the vector store. `plan` receives
`chat_context` and rewrites it — verified turning that question into *"What was
operating income in 2023?"* and answering correctly ($112.8m vs $74.2m).

`critique` deliberately does **not** receive history. Per-node context budgets
are where the real token savings are.

### Progress streaming, not token streaming

`POST /sessions/{id}/stream` emits SSE per completed node:

```
event: progress  {"node":"plan","detail":"planned 2 sub-questions"}
event: progress  {"node":"retrieve","detail":"retrieved 4 chunks across 2 queries"}
event: progress  {"node":"draft","detail":"drafted, cited 2 sources"}
event: progress  {"node":"critique","detail":"critique passed"}
event: done      {...full result...}
```

Token streaming is not possible here: with `responseSchema` output you would be
streaming half a JSON object. Node-level progress says *what is happening*,
which for an agent is the more useful signal. `EventSource` only does GET, so
the client uses `fetch` + `ReadableStream` and parses SSE frames by hand.

### Known rough edge

Retrieval is phrasing-sensitive and the agent is not fully deterministic. *"Which
segment was flat?"* failed once (draft cited nothing, critic accepted it) and
succeeded on a later run when `plan` happened to rephrase it. `critique` now
treats **zero citations** as evidence that the phrasing failed rather than the
information being absent, and asks for differently-worded queries.

The cost: genuinely-absent answers now retry once before giving up — ~5 Gemma
calls instead of 3. Verified bounded by `agent_max_iterations=2`: asking for a
CEO salary and board members retried once with new phrasings, found nothing, and
stopped.

## Multi-query retrieval + RRF (implemented, off by default)

`multi_query=true` on `/ask` or `/search`: Gemma rewrites the question into 3
variations, each retrieves its own top_k, and the ranked lists are fused with
Reciprocal Rank Fusion.

```
score(chunk) = Σ over lists of  1 / (60 + rank_in_that_list)
```

Rank position, not score, is deliberate: cosine scores aren't comparable across
queries (measured: an irrelevant match at 0.570 vs a correct one at 0.615), and
a BM25 score would be on a different scale again. Rank is the only shared
currency, which is why RRF needs no normalisation or tuning. It's the same
function hybrid search will reuse.

### Honest result: no measurable gain on this corpus

Rewriting works well — *"how did they keep bodies from rotting"* became
*"What methods were employed to prevent biological decomposition of human
remains?"* and two more. But A/B at tight top_k:

| query | top_k | single | multi |
|---|---|---|---|
| `what stops the water channel from eroding` | 2 | Aqueducts, Giza | identical |
| `the new office they opened abroad` | 2 | Headcount, Financial | identical |
| `Egypt` | 1 | Mummification | identical |
| `growth` | 2 | Financial, **Outlook** | Financial, **Headcount** (arguably worse) |

With 11 chunks, single-query retrieval already finds the right chunk — there is
no headroom to recover. Multi-query pays off when top_k is a small fraction of
the corpus and recall genuinely misses; at 11 chunks it mostly costs an extra
Gemma call. Hence **default off**, togglable per request, so it can be
re-measured as the corpus grows.

### Two Gemma quirks this exposed

- **Repetition loops.** At `temperature=0.7` it emitted *"way's actually way's
  actually…"* until it hit the token cap, truncating the JSON. The API offers no
  repetition penalty, so the defence is low temperature (0.35) plus `topP`.
- **Salvage truncated JSON.** `_parse_variations` recovers the complete strings
  from a cut-off array and drops degenerate ones (a low unique-word ratio marks
  a loop). Query expansion is an optimisation, so it degrades to "no variations"
  rather than failing the request.

## Frontend

Three routes, a navy sidebar, and a right rail on the chat view.

| Route | Purpose |
|---|---|
| `/chat` | session list; `/chat/[id]` is the conversation |
| `/library` | upload, document cards, expandable chunk inspector |
| `/lab` | single-shot ask, agent/baseline toggle, `top_k`, raw vector search |

`/lab` is deliberately separate: those controls are how you tell whether a bad
answer came from retrieval or generation, but they should not be the first thing
a user of the product sees.

### The design system lives in `globals.css`

Tokens plus a component layer — `.btn`, `.card`, `.input`, `.chip`, `.badge`,
`.switch`, `.checkbox`, `.stepper`, `.segmented`, `.tab`, `.skeleton`. Two
things are baked in rather than left to discipline: **`cursor: pointer` on every
interactive class**, and a **38px minimum height** (44px for primary actions).
An earlier pass used raw HTML controls and read as unfinished for exactly those
two reasons.

Palette: **navy** carries the structure (sidebar in both themes), **sky** is the
only interactive accent, **red is strictly semantic** — destructive confirms,
error banners, and the `no sources cited` badge. Red spread decoratively would
compete with sky and erode its meaning as a warning.

Icons are 14 hand-rolled inline SVGs in `icons.tsx`. **No emoji** — they render
differently per platform and read as informal.

### The right rail

Chat controls live in a rail rather than the page header, because the header
scrolled away and you could no longer tell what was in scope. It has two tabs:

- **Controls** — documents searched, passages per query, multi-query, the
  rolling summary, delete session
- **Source** — the passage behind a clicked citation

Citations open in the rail rather than a modal, since a slide-over would cover
the answer you are checking the citation *against*.

Collapsing leaves a 52px strip on the right edge showing document count, `k`
value, a multi-query marker and the memory count — reclaiming width without
reclaiming the information. The conversation's width animates to match
(measured: 288px → 106px mid-flight → 52px).

### Frontend gotchas worth knowing

- **Tailwind v4 cannot `@apply` another custom component class**, only
  utilities. `.card-interactive` repeats the card properties on purpose.
- **Next 16 defaults to Turbopack, which ignores `WATCHPACK_POLLING`** and sees
  no file events across a Windows bind mount — edits were silently dropped. The
  dev container runs `next dev --webpack`; run the frontend natively if you want
  Turbopack's speed.
- **A `fixed` element with `translate-x-full` extends the scrollable area**,
  which produced a horizontal scrollbar and let the layout drift sideways.
  `overflow-x: hidden` on `html, body`.
- **An unstable callback in context wiped state.** `closeChunk` was an inline
  arrow inside the context `useMemo`, so its identity changed on any context
  change, re-running the chat page's mount effect and clearing the citation the
  instant it was set. `useCallback` fixed it.
- **Don't put a right margin on a `max-w` box** to make room for the rail — it
  shrinks the column (768 → 480px) instead of reserving space. Padding on an
  unconstrained wrapper is correct.
- **Session details are cached** in a module-level `Map`, and the chat header
  takes its title from the already-loaded session list. Without both, switching
  chats flashed a full-page skeleton for the length of a round-trip.

## Looking at the data

| Store | Where |
|---|---|
| Qdrant | http://localhost:6333/dashboard — built in, always running |
| Postgres (GUI) | `docker compose --profile tools up -d pgweb` → http://localhost:8081 |
| Postgres (CLI) | `docker compose exec postgres psql -U rd -d research_desk` |

**pgweb needs no credentials** — it connects via `DATABASE_URL` in
`docker-compose.yml` and opens directly on the schema, with a SQL tab and
CSV/JSON export. Opt-in via the `tools` profile so a plain `up` doesn't run it.

(Adminer and pgAdmin were both tried first and dropped: each shows a login
form, and Adminer defaults its System dropdown to MySQL, so correct Postgres
credentials still fail until you notice and change it.)

For reference, the local Postgres credentials are in `docker-compose.yml`:
user `rd`, password `rd_local_dev`, db `research_desk`, host `postgres`
(from inside the compose network) or `localhost:5432` from Windows.

### Useful endpoints

```powershell
# raw retrieval, no LLM -- tells you whether retrieval or generation is at fault
curl "http://localhost:8000/search?q=what+drove+revenue+growth&limit=3"

# does Postgres agree with Qdrant? a mismatch means a failed ingest
curl http://localhost:8000/stats

# after changing EMBEDDING_DIM or the provider (collections are fixed-dimension)
docker compose exec api python -m app.scripts.reset_vectors
```

### Gotchas hit while building this

- **Qdrant client/server versions must match** on major.minor. `qdrant-client`
  1.19 could not read storage written by server 1.12.4 and the container
  crash-looped. Both are pinned now; if you bump one, bump the other and wipe
  the volume (`docker volume rm research-desk_qdrantdata`).
- **PowerShell 5.1 mangles UTF-8** in `Invoke-RestMethod` output. An em-dash
  showing as `â` in the terminal does not mean the data is corrupt — check with
  `docker compose exec postgres psql` before "fixing" anything.
- **`create_all` cannot alter existing tables.** Any model change needs
  `docker compose down -v`, or migrate to Alembic.

## Daily development

Services carry `restart: unless-stopped`, so **if Docker Desktop is set to start
on login, the whole stack comes back by itself** — no command needed. You only
run `up` after a reboot where you'd previously stopped it, or after changing
`docker-compose.yml` / a `Dockerfile` / dependencies.

```powershell
docker compose up -d          # start detached, returns your terminal
docker compose logs -f api    # follow one service (drop -f api for all)
docker compose ps             # what's running + health
```

Day-to-day you shouldn't need to restart anything. Both services hot-reload:
`app/` and `lib/` are bind-mounted, uvicorn runs `--reload`, Next has
`WATCHPACK_POLLING=true` (needed — inotify doesn't cross the Windows/WSL2 mount
boundary reliably).

When you *do* need to restart:

```powershell
docker compose restart api            # backend only, ~2s
docker compose restart web            # frontend only
docker compose up -d --build api      # after editing pyproject.toml
docker compose up -d --build web      # after editing package.json
docker compose down                   # stop all (data survives in volumes)
docker compose down -v                # stop AND wipe postgres + qdrant data
```

### Can I skip Docker and run `npm run dev`?

**Frontend: yes.** Node 20 is on your machine already.

```powershell
docker compose up -d postgres qdrant api   # infra + backend in Docker
cd frontend; npm install; npm run dev      # frontend native
```

Native `npm run dev` is meaningfully faster on Windows — file watching is
direct instead of polling through a bind mount, so HMR is near-instant. This is
the workflow to use while doing heavy UI work in step 5.

**Backend: no**, not without work — `python` isn't on your PATH, and installing
3.12 + `uv` natively defeats the "install nothing but Docker" goal. Leave it in
the container; `--reload` already makes it feel local.

`NEXT_PUBLIC_API_URL=http://localhost:8000` works identically in both modes, so
nothing needs changing when you switch.

## Security note — read before the first push

`.env` is gitignored, but **rotate the key before you push, not after**. Once a
key reaches a public repo it is compromised regardless of any later rotation;
scrapers find them within minutes.

## Build order

| # | Step | Teaches |
|---|---|---|
| 1 | ✅ Compose skeleton, health check, Next.js shell | the plumbing, container networking |
| 2 | ✅ Ingestion: upload → parse → chunk → embed → Qdrant | chunking, batching, rate limits |
| 3 | ✅ Plain RAG endpoint, no agent | the baseline to compare against |
| 4 | ✅ LangGraph rewrite: plan → retrieve → answer → critique → loop | state machines, conditional edges, cycles |
| 5 | ✅ Streaming + citations UI (folded into step 6) | SSE, `astream`, React streaming |
| 6 | ✅ Chat sessions: checkpointer, threads, per-session scope, SSE progress | durable agent state, history compression |
| 7 | **Deploy — next up** | Render + Vercel + Neon + Qdrant Cloud |
| 8 | Agent tools: web search, chart generation | tool-use loops without native function-calling |
| 9 | Auth (Firebase/Clerk) + per-user isolation | wiring `owner_id` end to end |

Steps 3 → 4 are deliberately in that order: you should *feel* what LangGraph
buys you over a plain chain instead of taking it on faith.

### What already persists, and what doesn't

Documents, chunks and vectors live in named Docker volumes and survive browser
reloads, container restarts and `docker compose down`. Only `down -v` wipes
them. What's missing is not persistence but **identity and continuity**: there
is no user, no conversation, and no saved scope, so every visit starts from a
blank query box. Steps 6 and 9 close that.

Design decisions already made to keep those steps cheap:

- `Document.owner_id` exists and is written to every vector payload, so
  per-user isolation needs no re-ingest
- `Document.meta` (JSONB) is copied to every payload, so LLM categorisation
  becomes a filter, not a migration
- `VectorStore.search()` already takes `document_ids`, `owner_id` and
  `meta_filters` — that *is* the "which documents are in scope" control a chat
  session will drive

## Deployment (all free tiers)

| Piece | Host | Catch |
|---|---|---|
| Frontend | Vercel | — |
| Backend | Render (Docker, `prod` target) | spins down after 15 min idle → ~50s cold start |
| Postgres | Neon | 0.5GB; autosuspends but wakes fast |
| Vectors | Qdrant Cloud | 1GB, no card required |
| Files | none — PDFs parsed in-memory | avoids needing S3/R2 |

Render's own free Postgres expires after 30 days; Neon's doesn't. Hence Neon.

### Production hardening checklist

Deliberately NOT done yet — the demo would be harder to learn from with all of
this in place. Tracked here so it isn't forgotten at step 7.

**Security**
- [ ] rotate `GOOGLE_API_KEY`; move it to the host's secret store, never a file
- [ ] `CORS_ORIGINS` to the exact Vercel domain — no `*`, no localhost
- [ ] rate-limit public endpoints per IP (`slowapi`); upload is expensive and
      currently unauthenticated
- [ ] `MAX_UPLOAD_BYTES` enforced at the proxy too, not just in the handler
- [ ] Qdrant Cloud API key + TLS; never expose 6333 publicly
- [ ] drop `/docs` and `/redoc` in prod (`docs_url=None`) or put them behind auth
- [ ] security headers + HTTPS redirect; `TrustedHostMiddleware`

**Correctness / reliability**
- [ ] **Alembic migrations.** `create_all` cannot alter an existing table, so
      column additions currently need a one-line manual `ALTER` (fine with one
      database — you notice a mistake instantly). This becomes mandatory with
      local **and** Neon to keep in sync: a forgotten ALTER on Neon is a
      production 500 while local works. See HANDOFF §9.
- [ ] move ingestion off `BackgroundTasks` to a real queue (Redis + arq, or
      Cloud Tasks). An API restart mid-ingest currently abandons a document in
      `embedding` forever
- [ ] reconcile job for Postgres↔Qdrant drift (`/stats` detects it; nothing
      fixes it)
- [ ] retry/backoff already exists for 429/503 — add a circuit breaker so a
      dead upstream fails fast instead of holding connections
- [ ] graceful shutdown: finish or requeue in-flight ingests
- [x] ~~prune checkpoint tables~~ — **done**, see "Checkpoint hygiene" below

**Operations**
- [ ] structured JSON logs to stdout + a log sink; add request IDs
- [ ] error tracking (Sentry free tier)
- [ ] uptime ping on `/health` to blunt Render's 15-min spin-down
- [ ] Postgres backups (Neon has PITR on paid; on free, a scheduled `pg_dump`)
- [ ] pin every image by digest, not tag

**Cost control**
- [ ] hard per-user daily caps on embedding and LLM calls — the free quota is
      shared across all users, so one enthusiastic visitor exhausts everyone's
- [ ] cache identical queries (embedding + answer) in Postgres

**Testing**
- [ ] `pytest` for chunking, RRF, and the limiter (all pure functions, easy)
- [ ] one end-to-end ingest→ask test against a throwaway DB
- [ ] CI running ruff + tsc + pytest

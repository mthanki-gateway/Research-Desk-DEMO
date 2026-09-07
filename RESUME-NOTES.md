# Research Desk — resume source material

Input for an AI agent updating a CV. Everything here is accurate and defensible
in an interview.

**Every bullet states a capability delivered, not a defect fixed.** Keep it that
way — "found and fixed a bug" reads as an admission on a CV even when the work
behind it was good. The same underlying work is framed as the system property it
produced.

**Read section 6 before writing bullets** — a few claims need context, or they
invite a follow-up that undercuts them.

---

## 1. One-line description

> **Research Desk** — a document-grounded research agent with a stateful
> multi-step reasoning loop, hybrid evaluation harness, and citation-backed
> answers, built on LangGraph, Qdrant, FastAPI and Next.js.

## 2. Short paragraph

> Designed and built an end-to-end retrieval-augmented generation (RAG) system
> that answers questions from a user's own documents with inline, verifiable
> citations. The agent runs as a **LangGraph state machine** — plan, retrieve,
> draft, self-critique, retry — with durable Postgres checkpointing that lets a
> run survive a restart and pause mid-execution to ask the user a clarifying
> question. Quality is measured by a **two-tier evaluation harness** covering
> retrieval metrics (recall@k, precision@k, MRR, MAP, NDCG) and generation
> metrics via RAGAS, judged by a separate, more capable model to avoid
> self-preference bias.

## 3. Technology keywords (for ATS)

**AI / RAG**
LangGraph · LangChain · RAG · Retrieval-Augmented Generation · Vector search ·
Qdrant · HNSW · Approximate Nearest Neighbour (ANN) · Embeddings · Semantic
search · Reciprocal Rank Fusion (RRF) · Multi-query expansion · Chunking
strategies · Prompt engineering · Structured output / JSON schema constraints ·
LLM-as-judge · RAGAS · Google Gemini API · Agentic workflows · Human-in-the-loop ·
Self-critique / reflection loops · Citation grounding · Hallucination mitigation

**Backend**
Python · FastAPI · async/await · SQLAlchemy (async) · asyncpg · psycopg3 ·
PostgreSQL · Pydantic · Server-Sent Events (SSE) · REST API design ·
Rate limiting · JWT / JWKS authentication · Multi-tenancy · pytest

**Frontend**
TypeScript · React · Next.js (App Router) · Tailwind CSS · Material Design 3 ·
Design tokens · Streaming UI · Accessibility (ARIA)

**Infrastructure / tooling**
Docker · Docker Compose · GitHub Actions · CI/CD · uv (Python packaging) ·
Ruff · structlog

---

## 4. Achievement bullets (ready to adapt)

Pick 4–6. Ordered strongest first.

### Agent architecture

- Architected a **LangGraph state machine** for multi-step research —
  plan → retrieve → draft → self-critique → conditional retry — using typed
  state, custom reducers and conditional edges; the cyclic graph expresses
  retry-with-new-evidence, which a linear chain (DAG) structurally cannot.

- Implemented a **self-critique loop** in which the model reviews its own draft
  against retrieved evidence and emits concrete follow-up search queries when
  the answer is incomplete, with four independent termination guards
  (sufficiency verdict, iteration cap, no-untried-queries, recursion limit) to
  prevent unbounded loops.

- Built **human-in-the-loop clarification** using LangGraph `interrupt()` and
  durable checkpointing: when a request is too vague to search, the agent pauses
  before any retrieval and offers concrete options derived from the user's
  actual document structure. The pause is persisted to Postgres rather than held
  in memory, so it survives process restarts and is resumed by a separate HTTP
  request.

- Added **durable execution** via LangGraph's Postgres checkpointer, allowing a
  multi-call agent run to resume from the last completed node after a crash or
  deploy instead of restarting.

### Retrieval and vector search

- Built the full RAG ingestion pipeline — parse → chunk → embed → index —
  with structure-aware chunking that preserves document hierarchy by prefixing
  heading paths onto chunk text.

- Integrated **Qdrant** for vector search with HNSW indexing, payload indexes,
  and server-side metadata filtering pushed into the index (post-filtering an
  ANN result silently under-returns).

- Implemented **multi-query expansion with Reciprocal Rank Fusion (RRF)** —
  rewriting a question into several phrasings, retrieving in parallel, and
  fusing the ranked lists by rank position rather than score, because similarity
  scores are not comparable across queries.

- Designed an **intent classifier** that distinguishes specific from broad
  requests and rewrites the query set accordingly, correcting a failure mode
  where broad instructions ("summarise this") retrieved passages matching the
  instruction rather than the content.

### Evaluation (differentiator — lead with this)

- Designed and built a **two-tier evaluation harness** against a curated golden
  dataset: Tier 1 measures retrieval (recall@k, precision@k, MRR, MAP, NDCG, hit
  rate) with no LLM calls; Tier 2 measures generation quality via **RAGAS**
  (faithfulness, answer relevancy, context precision, context recall).

- Used a **separate, more capable model as LLM-as-judge**, isolating the
  evaluator from the model under test to avoid documented self-preference bias.

- Separated the expensive generation phase from the cheap judging phase with a
  **content-addressed answer cache** keyed on every input that affects an
  answer, cutting judge-iteration time by roughly an order of magnitude;
  failures are deliberately never cached.

- Used the harness to **detect a content-integrity defect in the ingestion
  pipeline that manual testing could not surface** — chunk filtering was
  silently dropping indexed content — and drove the fix to full recall on the
  affected queries, demonstrating measurable ROI on the evaluation investment.

- Surfaced evaluation in an in-app dashboard with per-question drill-down,
  showing expected vs. retrieved passages and an automated diagnosis of whether
  a failure is a *retrieval* or a *ranking* problem (derived from the gap
  between recall@k at small and large k).

### Reliability and production concerns

- Implemented **multi-tenant data isolation** end to end — tenant scoping is a
  required parameter on every retrieval path and is pushed down into both the
  SQL and vector-store queries, so a tenant filter cannot be omitted by
  accident.

- Implemented **JWT authentication** verified against a JWKS endpoint, so the
  API holds no shared secret, with per-user data isolation enforced at the
  query layer.

- Built **dual token-bucket rate limiting** (requests/minute and tokens/minute,
  scoped per model) to keep the system within provider quotas, extending it to
  third-party libraries that call the provider directly and would otherwise
  bypass throttling and stall in exponential backoff.

- Engineered the agent to **degrade rather than fail**: every node falls back to
  simpler prior behaviour on model error, and a resilient parser salvages usable
  output from truncated or degenerate model responses rather than discarding the
  turn — turning a class of hard failure into a graceful one.

- Constrained every model call with **JSON schema / structured output**, which
  eliminated a class of failure where the model emitted reasoning traces into
  user-facing text.

### Frontend

- Built a streaming chat interface in **Next.js (App Router) and TypeScript**
  with node-level progress over **Server-Sent Events**, clickable inline
  citations that open the exact source passage, and an interactive
  human-in-the-loop prompt.

- Implemented a **Material Design 3 design system** with programmatically
  generated colour tokens — a full light/dark palette derived from a single seed
  colour, switchable at runtime with no re-render.

### Engineering practice

- 109 automated tests covering retrieval metrics, chunking, agent control flow,
  human-in-the-loop state transitions, and model-output parsing.
- CI running lint, type-checking and tests on every push; containerised
  development environment via Docker Compose.

---

## 5. Interview talking points

Design decisions worth being able to defend. Each maps to a bullet above.

**"How do you know your RAG system works?"**
Two tiers, because "the answer was bad" has two entirely different causes with
different fixes. Retrieval metrics answer *was the right passage returned*;
generation metrics answer *given that passage, was the answer faithful to it*.
Comparing recall at small vs. large k also tells you whether a failure is a
ranking problem (fix with a reranker) or a retrieval problem (fix chunking) —
the difference between a day's work and a month's.

**"Why LangGraph and not a chain?"**
A chain is a DAG and a DAG cannot loop. The core behaviour — critique the draft,
and if it's insufficient go back and retrieve differently — is a cycle. The
second reason is the checkpointer: persisting state after every node is what
makes a run resumable, and what makes a human-in-the-loop pause possible at all,
since the pause is a database row rather than a blocked process.

**"How does the human-in-the-loop pause actually work?"**
The graph decides ambiguity in one node and interrupts in a separate one,
because `interrupt()` doesn't suspend a function mid-body — on resume the node
re-executes from the top. Keeping the expensive judgement out of the
interrupting node means answering never pays for it twice. The options offered
are generated from the user's real document structure, so the agent can only
offer choices the corpus can actually answer.

**"Why a separate model as judge?"**
Models show a documented self-preference bias when grading their own output, so
the evaluator has to be a different model from the one under test. It also
suits the workload — judging sends the answer plus every retrieved passage, so
it wants a model with different throughput characteristics from the one doing
generation.

**"How do you guarantee tenant isolation?"**
Scoping is a required parameter on every retrieval path, not an optional one
with a permissive default, and it's pushed down into both the SQL and the
vector-store query rather than filtering after the fact. Post-filtering an
approximate-nearest-neighbour result also silently under-returns, so pushing the
filter into the index is a correctness requirement as well as a security one.

**"What would you do next?"**
Cross-encoder reranking — the evaluation quantifies the headroom, since the
right passage is in the candidate pool far more often than it's ranked first.
Then LLM tracing, because a turn spans several model calls across four nodes
with a retry loop, and flat logs can't show why the loop fired.

---

## 6. Scope notes — what is and isn't built

Not a list of shortcomings; a list of what the bullets may and may not claim, so
nothing above can be contradicted under questioning.

| Claim | Guidance |
|---|---|
| Specific recall/precision numbers | Real, but from a **small curated corpus**. Say "on a curated golden set" — don't quote a bare figure that invites "on what dataset?" |
| "Production" | Say **"production-shaped"** or "built to production practices". It is a solo project, not a system with real users at scale. |
| Hybrid / BM25 search | **Not built.** Evaluated and deliberately deferred when the measurements didn't support it. Strong as a judgement story ("I didn't build it because the data said the bottleneck was elsewhere"); don't list it as implemented. |
| Reranking / cross-encoder | **Not built.** The evaluation quantifies the headroom for it — the identified next step. |
| Observability / tracing | **Not built.** Structured logging exists; LLM tracing does not. Good answer to "what would you do next?" |
| Team size / scale | Solo project. Don't imply otherwise. |

**Framing that works:** present it as a **learning-driven engineering project
built to production standards** — the point isn't scale, it's that the
decisions are measured rather than guessed, and that the limitations are known
and articulated. Interviewers respond well to a candidate who says "I didn't
build hybrid search because my evaluation said the problem was elsewhere."

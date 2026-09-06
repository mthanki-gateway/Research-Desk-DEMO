# Interview cheat sheet — RAG & LangGraph

Read on the train. Everything here is either a number you measured or a
sentence you can say. Depth lives in `INTERVIEW-PREP.md`; this is recall.

**The three stories to lead with** — have these ready, they carry the interview:

1. **The chunker bug.** I concluded "dense retrieval can't match identifiers,
   we need BM25" — and I was wrong. Evidence beat intuition.
2. **The cross-tenant leak.** A security control that is a default argument is
   a security control that will be forgotten.
3. **The reducer trap.** `evidence: []` doesn't clear a reducer — it appends
   nothing. Evidence leaked between turns until thread ids became per-attempt.

---

## 1. The pipeline, in one breath

```
OFFLINE:  parse → chunk → embed → index
ONLINE:   query → rewrite → retrieve → fuse → rerank → assemble → generate → cite
```

**Diagnosis is the skill, not the list.** Bad answer → *was the right chunk in
the context?*

| recall@3 | recall@10 | Diagnosis | Fix |
|---|---|---|---|
| low | **high** | ranking — it's there, buried | **reranker** |
| low | low | retrieval — never found | chunking / hybrid |
| high | high | retrieval fine | it's generation |

> ⬅ *"Almost every 'the LLM is hallucinating' complaint is a retrieval failure.
> The model can't ground a claim in a passage it was never given."*

**Fix order:** measure → diagnose → chunking → hybrid → rerank → prompt →
guardrails. Fine-tuning last, usually never.

---

## 2. Numbers from my system

Say these; they're the difference between reading blogs and shipping.

| Metric | Value | What it means |
|---|---|---|
| **recall@1** | 0.520 | right chunk is first half the time |
| **recall@10** | 0.951 | it's almost always *in the pool* |
| ⬅ the gap | **0.43** | pure ranking loss — that's the reranker's prize |
| Corpus | 39 chunks | too small; top-10 = 26% of everything |
| Golden set | 20 questions, 16 tags | content-based labels, not chunk ids |
| Tier 2 cost | 113s/question (2 metrics), 532s (4) | judge is 15 rpm |
| Checkpoints | 920KB vs 248KB real data | LangGraph snapshots every node |
| Agent turn | 3–5 calls, ~41s, ~9,800 tokens | mostly waiting on network |

**Known weaknesses — say them before you're asked.** Corpus too small; the
`bm25` and `anaphora` golden tags now all score 1.000 so they prove nothing;
Flash Lite ignores `temperature` so Tier 2 has irreducible noise; CI isn't a
gate.

---

## 3. Chunking

- Recursive character splitting: `["\n\n", "\n", " ", ""]`, fall back only when
  still too big
- Markdown headings as **hard boundaries**; prefix the heading path onto the
  chunk text
- 900 chars / 150 overlap here
- Filter degenerate chunks — but measure the **body** length, excluding the
  heading prefix, or the filter is fooled

**★ The story.** A golden question asked about `INC-2024-1183`. Retrieval never
found it at any k. I concluded dense retrieval can't match exact identifiers and
that we needed BM25.

**Wrong.** The id lived in a heading with an empty body, `min_chunk_chars`
deleted the resulting short chunk, and the string **wasn't in the index at
all**. BM25 would have failed identically. After the fix: **recall 1.000**.

> ⬅ *"A retrieval failure is a chunking failure until proven otherwise — and
> the proof is trivial: grep the chunk table for the string."*

Also worth naming: **parent-document / small-to-big** (index small, return the
parent), **contextual retrieval** (Anthropic — LLM prepends a context line at
ingestion, −35% failures, −49% with BM25), **late chunking** (embed the whole
doc, then pool per chunk).

---

## 4. Embeddings & geometry

- **Bi-encoder** = separate encodes → two vectors → cosine. **Retrieval.**
- **Cross-encoder** = query+doc together → one score. **Reranking.** Better,
  slower, nothing to precompute.
- **Cosine** = angle, ignores length. Normalise to unit length and cosine, dot
  and L2 all rank identically.
- **MRL** (Matryoshka) = truncate the vector, then **re-normalise**.
- Model swap = **full re-index**. Always.

⚠️ **Scores are not comparable across queries.** Measured here: an irrelevant
match at **0.570**, a correct one at **0.615**. That kills naive score
thresholds — use relative signals (gap between top and mean).

---

## 5. Indexing

- **HNSW** — layered proximity graph, the industry default. `ef_search` is the
  one knob you'll actually tune.
- IVF = cluster, search nearest clusters. PQ = compress, lossy. DiskANN = SSD,
  billion-scale.
- The triangle: **recall / latency / memory** — pick two.
- ⚠️ **Filtering must be pushed into the index.** Post-filtering an ANN result
  returns fewer than `top_k`, or nothing, because traversal never visited the
  matching region. Qdrant does filtered HNSW + payload indexes.

---

## 6. Retrieval

| Strategy | Good at | Bad at |
|---|---|---|
| **Dense** | paraphrase, synonyms, intent | exact ids, rare proper nouns |
| **BM25** | identifiers, exact phrases, names | paraphrase |
| **Hybrid** | both — run in **parallel**, fuse | needs a second index |
| ColBERT | token-level MaxSim | storage |
| SPLADE | learned sparse + expansion | cost |

**BM25 = TF · IDF with k₁ saturation and b length-normalisation.** No model,
pure corpus statistics.

### RRF — say this precisely

```
score(chunk) = Σ over lists of  1 / (k + rank)        k = 60
```

**Rank, not score, because the scores are incomparable** — BM25 returns ~14.7
(unbounded), dense returns 0.61 (cosine). Rank is the only shared currency,
which is why RRF needs no normalisation and no tuning. Agreement between
independent retrievers is the signal.

`k=60` from the original TREC paper; at 60 the rank-1 to rank-2 gap is ~2%, so
one list can't dominate.

⬅ **Fusion ≠ reranking.** RRF reorders using positions that already exist and
never reads the text. A cross-encoder reads query and chunk together and can
discover that rank 8 was best. *Fusion is bookkeeping; reranking is judgement.*

⚠️ **A useless query poisons RRF** — every list gets an equal vote. "Summarize
the document" retrieves passages similar to the *word* "summarize". Hence: a
`scope` classifier, and for broad requests the original question is **dropped**
and one query per topic is written instead.

### Reranking

Retrieve 30–100 cheap → rerank to 5–10 expensive. Anthropic: contextual
embeddings + BM25 = −49% failures; **adding reranking of 150→20 took it to
−67%** — worth roughly as much as everything before it.

Models: Cohere Rerank (API), BGE-reranker (self-hosted), MiniLM (CPU-fast).
**Not an LLM** — MiniLM-L6 is 22M params.

---

## 7. Evaluation

**Tier 1 — retrieval.** Recall@k ⬅ the one that matters. Precision@k, MRR
(1/rank of first hit), MAP, NDCG (graded + rank-aware), hit rate.

⚠️ MRR and MAP must **truncate at k** or every row of your table is identical.
I shipped that bug and caught it in the UI.

**Tier 2 — generation.** RAGAS: **faithfulness** ⬅ *the* hallucination metric,
answer relevancy, context precision, context recall.

> *"Why RAGAS and not my own judge prompts? Because its definitions are the
> ones everyone else reports against. A hand-rolled faithfulness of 0.82 means
> whatever my prompt happens to do. And faithfulness isn't one prompt — it's
> decompose into atomic claims, then verify each. The decomposition is where
> the fiddly work is."*

**Judge = a different model** (Flash Lite, not Gemma): self-preference bias is
documented, and the quota shapes suit it — 250K tpm vs Gemma's 16K.

**Answer caching is what makes it usable.** Generation is expensive, judging is
cheap and on a different quota. Cache keyed on a hash of *every* input that
changes an answer (mode, top_k, multi_query, model). Failures are never cached
— that would hide a transient outage forever.

**LLM-as-judge biases:** position, verbosity, self-preference.

**Online:** 👍/👎 and logging queries + retrieved chunk ids from day one — that
log *becomes* your golden set. Cheap proxies: zero-citation rate, refusal rate,
top-score distribution.

---

## 8. LangGraph

**Pitch:** LCEL gives you a **DAG** and a DAG cannot loop. LangGraph gives you a
**state machine with typed state, cycles, and a snapshot after every node.**

```
START → plan → review_plan ─┬→ retrieve → draft → critique ─┬→ END
                 (human)    │      ▲                        │
                  cancel ───┘      └──────── retry ─────────┘
```

### Reducers — the #1 gotcha

```python
draft: str                                  # no reducer → OVERWRITE
trace: Annotated[list, append]              # accumulate
evidence: Annotated[list, merge_evidence]   # custom: dedupe by chunk_id
```

⚠️ **A reducer applies to the INPUT too.** `evidence: []` appends nothing — it
does **not** clear. Measured consequences: duplicated trace, polluted
`tried_queries` so `critique` refused to re-run queries it thought were tried,
and **evidence from an earlier question leaking into a later answer.**

Fix: **per-attempt thread ids** (`<session>:<turn>:<random>`) and keep
conversation memory in a `messages` table, not in graph state.

> ⬅ *"Graph state ≠ conversation memory. Conflating them is the most common
> LangGraph design error."*

### Conditional edges

Router must be **pure, cheap, deterministic** — no LLM calls. Put the model's
judgement in a *node* that writes a verdict; let the router read it.

My `should_continue` has three exits: model says sufficient, iteration cap
(**a cost cap, not a quality judgement** — each cycle is ~2 Gemma calls), or no
untried queries left.

### Checkpointers

One argument, and it unlocks durability, time travel, threads and interrupts —
all the same mechanism.

⚠️ Two traps I hit: **920KB of snapshots** vs 248KB of real data; and
`psycopg` defaults `min_size=4`, so `max_size=2` alone raises — and because
init was in a `try/except`, that surfaced as **the app booting fine but
silently without resume.** A failure that looked like success.

### Human-in-the-loop ⬅ built

```python
decision = interrupt({"sub_questions": planned})   # graph STOPS, state persists
# later, different request, different worker:
graph.ainvoke(Command(resume={"action": "edit", ...}), config)
```

**Why the plan and not a tool call:** this agent has no side effects to gate.
What it has is a decomposition that decides every query that follows — a bad
plan wastes 3–5 calls. That's where a human's 5 seconds buys something.

Four things to say:
- **Nothing happens before `interrupt()`** — on resume the node re-executes
  **from the top**, so any side effect above that line runs twice. The classic
  footgun.
- An edit hits the reducer trap on purpose: `pending_queries` overwrites (run
  only the human's queries), `sub_questions` appends (trace keeps the original).
- A malformed decision **degrades to approve**, never to an error.
- **Ownership checked against the session, never the thread id** — that id is
  client-supplied and proves nothing.
- The pause is **not a parked coroutine.** State is in Postgres; resume can
  arrive after a deploy.

### Error handling — degrade to the previous behaviour

| Node fails | Falls back to | Why |
|---|---|---|
| `plan` | ask the question as-is | planning is an optimisation |
| `expand_query` | original query only | "a mediocre search beats no search" |
| `critique` | accept the draft | "better a good answer with no review than a 502" |
| `document_outline` | `[]` | "a broken outline must never fail a search" |
| checkpointer | no resume | beats not starting |

⚠️ But **log it** — a `try/except` that swallows too much turns a loud failure
into a silent one. That's exactly how the `min_size` bug hid.

### Streaming

`updates` (per-node deltas) + `values` (full state) at once — saves an
`aget_state()` round trip.

⬅ **Token streaming is wrong here:** with schema-constrained output there's no
partial prose, you'd stream half a JSON object. "Retrieving 2 of 3" is a better
signal than a token crawl anyway.

### Know the names

`Send` = dynamic parallel fan-out (map-reduce). `create_react_agent` = the
current prebuilt. Supervisor = the only multi-agent topology with real traction.
`AgentExecutor` = **legacy**, and saying so signals you're current.

---

## 9. Prompting — only what matters

**Structured output is the important one.** Not a parser — a **control
surface.**

Three measured lessons:
1. A schema **suppresses reasoning leakage** — Gemma has no thinking channel,
   so asked for prose it writes its whole reasoning trace into the reply and
   truncates.
2. **Every extra field is a chance to derail.** A free-text `reasoning` field
   made it produce four correct sub-questions then degenerate inside
   `reasoning`, invalidating the whole object. **Ask only for what you use.**
3. **A constrained enum is safe where free prose wasn't** — `scope: ["specific",
   "broad"]` costs a few tokens and has no token sequence to wander into.

**Anti-hallucination, layered:** grounding instruction → forced citation →
zero-citation detection → critic pass → post-hoc validation.

⬅ **Vector search always returns something.** Quantum computing against a
financial corpus still scores 0.570. Without an abstention path the model gets
irrelevant context with no signal it's irrelevant.

Make abstention **structural**: `sources_used: []` is a typed fact;
"say you don't know" is ten unparseable phrasings.

**Temperature:** 0.0 plan/critique, 0.1 draft, 0.35 expansion (0.7 caused a
**repetition loop** that truncated JSON). ⚠️ `finishReason=RECITATION` = the
model was reciting memorised text — in RAG that's a **grounding failure**, not
a token-limit problem.

**Lost in the middle:** strongest evidence first, question last, history in the
weak middle.

---

## 10. Fast answers to likely questions

**"Why not just use a long context window?"**
Cost scales linearly, latency scales, lost-in-the-middle is real, and you lose
citations. RAG also gives access control and freshness. They compose — retrieve
into a long window.

**"Fine-tune or RAG?"**
Fine-tuning teaches **behaviour**; retrieval supplies **facts**. Using
fine-tuning to add knowledge is the most common misunderstanding in the space.
Also: fine-tuning the embedder means re-indexing everything, forever.

**"How do you know it's working?"**
Two tiers. Recall@k on a golden set for retrieval; RAGAS faithfulness for
generation. Plus cheap online proxies. And I can point at a real bug the harness
caught that I'd otherwise have "fixed" the wrong way.

**"Build or buy?"**
> *"For a typical enterprise 'search our Confluence' project I'd use Vertex AI
> Search or Bedrock Knowledge Bases — the connectors and ACL handling alone are
> months of work. I'd build when retrieval quality is the differentiator, or
> when I need to measure and tune it. On my own project I built it to be able
> to debug it — and that paid off."*

**"How would you debug this in production?"**
Not "check the logs". A turn is 3–5 calls across four nodes with a retry loop;
flat lines can't show that `critique` fired twice because retrieval missed. You
need **traces** — Langfuse via the LangGraph callback handler. Currently my
biggest gap, and I know it.

**"What's your multi-tenancy story?"**
`owner_id` pushed into the Qdrant filter with a payload index. And a real leak:
`/stream` omitted it while `/messages` passed it. ⬅ *"A security control that
is a default argument is a security control that will be forgotten."* Make the
tenant key **required**.

**"What would you do next?"**
In order, with reasons: eval-owner separation (so fixtures stop polluting the
corpus), cache question embeddings (re-runs become free), grow the corpus to
150–250 chunks (39 is too few to trust any number), golden questions that can
actually fail, then the reranker — the one **evidenced** win.

---

## 11. Traps worth naming unprompted

| Trap | The catch |
|---|---|
| Cosine scores across queries | not comparable — 0.570 wrong, 0.615 right |
| Post-filtering ANN | returns < top_k or nothing; push into the index |
| Reducer reset | `[]` appends nothing |
| MRR/MAP without k | every table row identical |
| One try/except around N metrics | one failure discards all — isolate each |
| Sampling `top_k` vs retrieval `top_k` | same name, unrelated |
| Prompt caching in RAG | chunks change every query — often nothing caches |
| RAGAS bypassing your limiter | it calls the provider directly; 2 questions took >10 min in backoff |
| `sslmode` in an asyncpg URL | libpq-only param; kills the driver |
| Deploy-only bugs | `output: "standalone"` broke Vercel — no local build catches it |

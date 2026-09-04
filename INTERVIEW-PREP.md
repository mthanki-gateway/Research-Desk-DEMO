# Interview preparation syllabus

# ⚑ SYMBOLS — read this first

Every heading carries up to three tags: **how often it ships**, **how much of it
you would actually write**, and **whether it is in this repo.**

| | Symbol | Meaning |
|---|---|---|
| **Popularity** | ★★★ | **Standard.** In most production RAG systems. Expected knowledge |
| | ★★ | **Common.** Frequently used, well supported by frameworks |
| | ★ | **Niche.** Real, but situational |
| | ○ | **Rare / research.** Papers and demos; seldom shipped |
| **You build it?** | 🔧 | **Build.** You write this code. Learn the details |
| | ⚙️ | **Configure.** A setting or one library call. Know what to set and why |
| | 📖 | **Know.** Interview/design knowledge only. Never hand-implemented |
| **In this repo?** | 🟢 | Implemented — go read the code |
| | 🔵 | Partly there, or designed but not built |
| | ⚪ | Not in the repo |
| **Inline** | ✅ ❌ ⚠️ | works / fails / caveat |
| | ⬅ | the point to notice |

**A heading tagged `○ 📖` is exam material you will never touch. A heading
tagged `★★★ 🔧` is a thing you should be able to build from memory.**

Do not memorise any formula tagged 📖 — be able to say what it *does* and what
it trades.

*(Popularity and build tags applied from Section 5 onward; retrofitted to 1–4
on a later pass.)*

---

Every topic named in the job description gets a row, whether or not this app
uses it. Work through it sequentially and tick items off.

**App notes are secondary.** Where a topic happens to be implemented in this
repo it is marked, because that is a place to go and read real code. Where it
is not, that is fine — say so in interview, or ask me to build it as a learning
exercise.

---

## Glossary

Skim this first; every term recurs throughout.

**Retrieval basics**

| Term | Meaning |
|---|---|
| **chunk** | a piece of a document, small enough to embed and cheap enough to put in a prompt |
| **embedding** | the vector produced by a model from text. "To embed" = to compute that vector, **not** to store it |
| **vector / dense vector** | a list of floats, e.g. 768 numbers, representing meaning |
| **sparse vector** | a mostly-zero vector over the vocabulary — one weight per term |
| **`top_k`** | how many results a search returns |
| **ANN** | Approximate Nearest Neighbour — fast, *approximate* vector search. The approximation is the point: exact search is O(n) |
| **payload / metadata** | fields stored beside a vector (`owner_id`, `page`), used for filtering and citation |
| **inverted index** | `term → [document ids]`. What makes keyword search fast |
| **corpus** | the whole document collection |

**Geometry**

| Term | Meaning |
|---|---|
| **cosine similarity** | the angle between two vectors, ignoring length. −1 → +1 |
| **cosine distance** | `1 − cosine similarity`. Smaller = closer |
| **dot product** | `Σ AᵢBᵢ`. Equals cosine when both vectors are unit length |
| **L2 / Euclidean** | straight-line distance |
| **L2 normalisation** | scaling a vector to length 1. Makes cosine, dot product and L2 rank identically |
| **unit vector** | a vector of length 1 |

**Models**

| Term | Meaning |
|---|---|
| **bi-encoder** | encodes query and document **separately** → two vectors → cosine. Used for **retrieval** |
| **cross-encoder** | encodes query and document **together** → one relevance score. Used for **reranking**. Much better, much slower |
| **MRL** | Matryoshka Representation Learning — training that puts most signal in the leading dimensions, so a vector can be truncated safely |
| **pooling** | collapsing per-token vectors into one text vector (CLS, mean, last-token) |
| **contrastive learning** | training on positive/negative pairs so similar texts land close together |
| **hard negative** | a training example that *looks* relevant but is not. Where the learning happens |

**Indexes**

| Term | Meaning |
|---|---|
| **flat / exact** | compare against every vector. 100% recall, O(n) |
| **HNSW** | Hierarchical Navigable Small World — layered proximity graph. **The industry default** |
| **`ef_search`** | HNSW's query-time candidate list size. The one knob you will actually tune |
| **IVF** | Inverted File — cluster vectors, search only the nearest clusters |
| **PQ** | Product Quantization — compress vectors into codebooks. Big memory savings, lossy |
| **DiskANN** | graph index designed to live on SSD, for billion-scale |

**Retrieval strategies**

| Term | Meaning |
|---|---|
| **dense retrieval** | vector similarity. "Semantic search" |
| **sparse retrieval** | keyword search — BM25, TF-IDF |
| **BM25** | the standard keyword ranking function. No model, pure corpus statistics |
| **TF / IDF** | Term Frequency (how often a term appears here) / Inverse Document Frequency (how rare it is overall) |
| **hybrid retrieval** | dense + sparse, fused |
| **RRF** | Reciprocal Rank Fusion — merge ranked lists using rank position only |
| **MMR** | Maximal Marginal Relevance — trade relevance against diversity |
| **HyDE** | Hypothetical Document Embeddings — embed a *fake answer* instead of the question |
| **ColBERT** | multi-vector late interaction — one vector per **token**, scored by MaxSim |
| **MaxSim** | for each query token, take its best match in the document, then sum |
| **SPLADE** | learned sparse retrieval — neural model outputs term weights with expansion |
| **GraphRAG** | build a knowledge graph of entities/relations, then traverse it |
| **reranking** | rescoring a shortlist with a better (usually cross-encoder) model |

**Patterns**

| Term | Meaning |
|---|---|
| **parent-document / small-to-big** | index small chunks, return their larger parent |
| **sentence-window** | index one sentence, return it plus neighbours |
| **auto-merging** | if enough sibling children match, return the parent instead |
| **contextual retrieval** | LLM prepends a document-level context line to each chunk **at ingestion** |
| **late chunking** | embed the whole document, *then* pool token vectors per chunk |
| **semantic chunking** | cut where cosine distance between consecutive sentences spikes |
| **CRAG** | Corrective RAG — grade retrieved documents *before* generating |
| **Self-RAG** | a model fine-tuned to emit reflection tokens controlling retrieval |
| **agentic RAG** | the LLM decides what to search, judges results, searches again |

**Evaluation**

| Term | Meaning |
|---|---|
| **Recall@k** | of all relevant chunks, what fraction appear in the top k. **The metric that matters most** |
| **Precision@k** | of the k returned, what fraction are relevant |
| **MRR** | Mean Reciprocal Rank — average of `1/rank` of the first relevant hit |
| **NDCG** | Normalised Discounted Cumulative Gain — rank-aware, graded relevance |
| **faithfulness** | is every claim in the answer supported by the retrieved context. **The hallucination metric** |
| **answer relevancy** | does the answer address the question asked |
| **context precision / recall** | were the retrieved chunks needed / were all needed chunks retrieved |
| **LLM-as-judge** | using an LLM to score outputs instead of human labels |
| **golden set** | a curated set of questions with known-correct answers/chunks |
| **RAGAS** | the standard Python library for RAG evaluation metrics |

**Prompting**

| Term | Meaning |
|---|---|
| **zero-shot / few-shot** | no examples / a handful of examples in the prompt |
| **CoT** | Chain-of-Thought — make the model reason step by step before answering |
| **ReAct** | Thought → Action → Observation loop for tool-using agents |
| **structured output** | forcing the model to emit JSON matching a schema |
| **grounding** | instructing the model to answer only from provided sources |
| **prompt injection** | untrusted text in the prompt hijacking the model's instructions |

Deep-dive write-ups for the harder topics are in **Part 2**, linked from the
tables.

---

# PART 1 — CHECKLIST

## 1. RAG: pipeline and architecture

### - [ ] 1.1 What RAG is, and why it beats fine-tuning for knowledge 🟢

**What it is.** Retrieval-Augmented Generation: instead of hoping the model
memorised a fact, you *retrieve* relevant text at query time and put it in the
prompt. The model's job shifts from *recall* to *reading comprehension*.

**Example.** Ask a base model "what was Acme's 2024 gross margin?" — it has
never seen your private report, so it either refuses or invents a number. With
RAG you retrieve the Financial Summary section, paste it into the prompt, and
the model reads the answer off the page and cites it.

**When to use.** Private, large, or frequently-changing knowledge. Anything
needing **citations**. Anything where being wrong is worse than being silent.

**When not to.** Teaching the model a new *skill*, *format*, or *tone* — that is
fine-tuning's job, not RAG's. Also skip it when the whole corpus fits
comfortably in the context window and cost is irrelevant.

**The interview framing:** *RAG changes what the model knows; fine-tuning
changes how the model behaves.* Facts belong in retrieval because they are
cheap to update and you can point at the source. Behaviour belongs in weights.

> 🟢 App: the whole repo.

---

### - [ ] 1.2 The full pipeline 🟢

**What it is.** Two paths that meet at the prompt:

```
INGESTION (offline, once per document)
  upload → parse → chunk → embed → index

QUERY (online, every request)
  question → [rewrite] → embed → search → [fuse] → [rerank] → assemble → generate → cite
```

**Example** — one question through this app:

| Step | What happens |
|---|---|
| rewrite | "and the prior year?" → "What was operating income in 2023?" |
| embed | question → 768-dim vector |
| search | Qdrant HNSW, filtered by `owner_id` |
| fuse | 3 query variations → RRF into one ranking |
| assemble | `[1] acme-report.md › Financial Summary\n<text>` |
| generate | Gemma reads the numbered sources, answers, cites `[1]` |

**When to use the optional steps.** `rewrite` when queries are short or
conversational; `fuse` when you run multiple queries or multiple retrievers;
`rerank` when recall is fine but precision is poor.

**When not to.** Every optional box costs latency and tokens. A single-document
FAQ needs none of them.

> 🟢 App: `services/parsing.py` → `chunking.py` → `embeddings.py` →
> `vectorstore.py` → `retrieval.py` → `agent/nodes.py`.

---

### - [ ] 1.3 Offline vs online path 🟢

**What it is.** Deciding which work happens at ingestion versus per query.
Anything you can precompute, you should.

**Example.** Chunking, embedding, contextual enrichment and metadata extraction
all happen **once** at ingestion. Query embedding, search and generation happen
**every time**. Moving work left (offline) is the single cheapest optimisation.

**When to use offline.** Deterministic, document-only work: parsing, chunking,
embedding, summarisation, entity extraction, keyword indexes.

**When not to.** Anything depending on the *query* — obviously — and anything
that changes often enough that the precomputed version would be stale.

**Trap.** Offline work is where you pay your one-off costs: this app embeds at
~133 chunks/minute against the free-tier token limit, so a large PDF takes real
minutes. Users need to see progress, not a spinner.

> 🟢 App: `services/ingest.py` runs stages (`parsing → chunking → embedding`)
> with a status column so the UI can poll progress.

---

### - [ ] 1.4 Where RAG breaks, in order of real frequency 🟢

**What it is.** A debugging order. Most people reach for prompt tweaks first,
which is the *least* likely cause.

1. **Chunking.** A boundary through the middle of a claim strands the subject
   from its number. Highest leverage, most neglected.
2. **Query formulation.** "Summarize this" has no semantic overlap with
   document content, so similarity search returns noise no matter how good the
   index is.
3. **Retrieval recall.** If the right chunk is not in top-k, no prompt saves
   the answer. *This is why Recall@k is the metric that matters most.*
4. **Generation faithfulness.** The model answers from parametric memory
   instead of the provided context.

**Example.** In this app, asking for "main risks" returned the *headcount*
section. The instinct is to blame the prompt. The cause was chunking: a 900-char
window had swallowed the tail of one section plus the head of the next, and the
blended embedding sat between both topics, near neither.

**How to tell them apart.** Have a **retrieval-only endpoint**. If the right
chunk is in the results and the answer is still wrong, it is generation. If it
is not, it is retrieval. Never debug both at once.

> 🟢 App: `GET /documents/search` exists precisely for this, and returns cosine
> scores plus which query variation found each hit.

---

### - [ ] 1.5 Naive vs Advanced vs Modular RAG ⚪

**What it is.** The taxonomy from the *Retrieval-Augmented Generation for LLMs*
survey, and a convenient way to describe maturity.

| Tier | Shape |
|---|---|
| **Naive** | chunk → embed → top-k → stuff into prompt. One pass, no rewriting, no reranking |
| **Advanced** | adds pre-retrieval (query rewriting, expansion, routing) and post-retrieval (reranking, compression, reordering) |
| **Modular** | swappable components and non-linear control flow — search modules, memory, routing, iterative and adaptive loops |

**Example.** Naive: `retriever.get_relevant_documents(q)` straight into a
prompt. Advanced: multi-query + RRF + cross-encoder rerank. Modular: an agent
that decides *whether* to retrieve, from *which* source, and whether to try
again.

**When to use which.** Start naive — it is a genuinely good baseline and you
need it to measure against. Move to advanced when you have eval numbers showing
where it fails. Go modular when different query types need genuinely different
handling.

**When not to.** Do not start modular. You will not know which module is
helping, because you never measured the baseline.

> 🟢 App: **modular** in shape (LangGraph with a critique loop), but `/ask` is
> deliberately kept as the naive baseline to compare against.

---

### - [ ] 1.6 RAG vs long-context vs fine-tuning vs tool use ⚪

**What it is.** Four ways to get knowledge into an answer.

| Approach | Cost per query | Freshness | Citations | Best for |
|---|---|---|---|---|
| **RAG** | low | instant | **yes** | large/private/changing corpora |
| **Long context** (stuff everything) | very high | instant | weak | small corpora, one-off analysis |
| **Fine-tuning** | low | stale until retrained | **no** | style, format, domain skills |
| **Tool use** | medium | live | yes | computed or real-time data |

**Example.** "What is our refund policy?" → RAG. "Summarise this one 20-page
contract I just uploaded" → long context is fine, and simpler. "Always answer in
our house style with these section headings" → fine-tune. "What is our current
AWS spend?" → tool call to an API.

**When not to use RAG.** When the answer is computed rather than written down.
No amount of retrieval finds a number that exists nowhere in the corpus — that
is a tool call.

**The "long context killed RAG" argument, and the answer:** cost scales with
every token on every request, attention degrades in the middle of long contexts
(§8.6), and you lose citations. RAG is retrieval *plus* attribution.

---

### - [ ] 1.7 Multi-tenancy 🟢

**What it is.** Guaranteeing user A can never retrieve user B's documents.
Harder in RAG than in CRUD, because the isolation must hold inside the **vector
search**, not just in your SQL.

**Example.** Store `owner_id` in the vector payload and filter on it at search
time. A missing filter does not error — it silently returns other people's
chunks, and the model happily summarises them.

**Three implementation choices:**

| Approach | Isolation | Cost |
|---|---|---|
| **Payload filter** (one collection) | logical | cheap; needs a payload index |
| **Collection per tenant** | stronger | collection overhead per tenant |
| **Cluster per tenant** | strongest | expensive; enterprise only |

**When to use which.** Payload filtering for most SaaS. Separate collections
when tenants need different embedding models or hard deletion guarantees.

**When not to.** Never rely on filtering *after* retrieval — you will return
top-k of everyone's data and then discard, which leaks via timing and breaks
recall.

> 🟢 App war story: the streaming endpoint called the agent **without
> `owner_id`**, which defaults to `None`, and `None` means *do not filter*.
> Proven: `owner_id=None` → 5 hits, `owner_id=<other user>` → 0 hits.
> **A security control that is a default argument will be forgotten.**

---

### - [ ] 1.8 Incremental indexing, updates, deletes 🔵

**What it is.** Keeping the index consistent as documents change. Harder than it
looks because one document maps to many vectors.

**Example.** A document is edited. Its 40 chunks become 37, with different
boundaries. You cannot diff chunks reliably, so the standard answer is
**delete-all-then-reinsert for that document**, keyed on `document_id`.

**Techniques.** Content hashing to skip unchanged documents; soft-delete
tombstones so search excludes rows before compaction runs; versioned
`document_id` so a failed re-ingest never half-replaces the old copy.

**When not to.** Do not attempt chunk-level diffing. Boundary shifts cascade,
and the bookkeeping costs more than re-embedding.

> 🔵 App: deleting a document removes its Postgres rows and issues a Qdrant
> `FilterSelector` delete by `document_id`. Re-ingest of an *edited* file is not
> implemented.

---

### - [ ] 1.9 Caching layers ⚪

**What it is.** Three independent caches, often confused.

| Cache | Key | Saves |
|---|---|---|
| **Embedding cache** | hash of text + model | re-embedding identical text |
| **Prompt cache** | prefix of the prompt | input tokens on repeated prefixes |
| **Answer / semantic cache** | the query (or its embedding) | the entire pipeline |

**Example.** Prompt caching is what makes **contextual retrieval** (§2.13)
affordable: the document is the large repeated prefix, so each per-chunk call
pays only for the chunk and the output.

**Semantic caching** is the interesting one: instead of exact-matching the
query string, embed it and serve a cached answer if cosine similarity to a
previous query exceeds a threshold.

**When to use.** Embedding cache always — it is free correctness. Prompt cache
whenever a long prefix repeats. Semantic cache for high-traffic, repetitive
questions.

**When not to.** Semantic caching on personalised or permission-sensitive
answers — two users can ask the same question and be entitled to different
documents. A semantic cache that ignores `owner_id` is a data leak.

## 2. Chunking

> **What is actually used in production** (answering "which are most popular?"):
>
> | Rank | Strategy | Reality |
> |---|---|---|
> | 1 | **Recursive character + overlap** | The overwhelming default. LangChain's `RecursiveCharacterTextSplitter`. Boring, dominant |
> | 2 | **Structure-aware → recursive within section** | The standard "we thought about it" combo: `MarkdownHeaderTextSplitter` → recursive. **This app** |
> | 3 | **+ heading/context prefix** | Cheap, deterministic, measurable. **This app** |
> | 4 | **Parent-document / sentence-window** | Popular in LlamaIndex codebases |
> | 5 | **+ contextual retrieval** | The current "serious production" answer since Anthropic's 2024 post |
> | 6 | **Semantic chunking** | Much discussed, **less used**. Costs an embedding pass and research finds it often fails to beat fixed-size |
> | 7 | **Propositional / late chunking** | Rare. Expensive, or needs self-hosted token-level access |
>
> **The honest interview answer to "how do you chunk?"**: *"Structure-aware
> split on the document's own boundaries, recursive character splitting within
> each section, heading prefixed into every chunk, then measure. That covers
> most of the win. Semantic chunking is the fallback when a document has no
> structure at all."*
>
> **The typical modern stack, end to end:** structure-aware split → recursive
> within section → context prefix → **hybrid index (dense + BM25)** → rerank.
> This app does 3 of those 5.

---

### - [ ] 2.1 Why chunk at all 🟢

**What it is.** Three independent forces, all pushing toward smaller units:

1. **Embedding token limits.** Most embedders cap at 512–8192 tokens and
   **silently truncate** past it — no error, just a vector for the first part.
2. **Retrieval precision.** One vector must represent one idea. A vector
   averaging ten topics is near none of them.
3. **Context cost.** You pay per token on every request, and long contexts
   suffer position effects (§8.6).

**Example.** A 50-page PDF as one chunk gives you one vector for the whole
document. Every query matches it equally badly, and you would have to send all
50 pages to answer anything.

**When not to chunk.** Short documents that already fit — a tweet, a support
ticket, a product description. Chunking a 200-word document just fragments it.

> 🟢 App: `chunk_size=900`, `chunk_overlap=150`, chosen small because Gemma's
> 16K tokens/minute makes context genuinely expensive.

---

### - [ ] 2.2 Chunk size and overlap as a trade-off 🟢

**What it is.** The central tension of RAG ingestion.

| | Small chunks (~200 chars) | Large chunks (~2000 chars) |
|---|---|---|
| Embedding precision | **high** — one idea per vector | low — meaning diluted |
| Context for the model | poor — no surroundings | **good** |
| Chunks per document | many | few |
| Tokens per answer | low | high |

**Overlap** repeats the tail of one chunk at the head of the next, so a claim
spanning a boundary survives in at least one chunk intact. Typical: 10–20% of
chunk size.

**Example.** "Operating income reached $112.8 million, compared to $74.2 million
in the prior year." Cut between those clauses and one chunk has the number
without the comparison, the other has the comparison without the subject. 150
chars of overlap keeps them together somewhere.

**When to go small.** Precise factual lookup; you have a reranker; you use
parent-document expansion to restore context.

**When to go large.** Narrative or argumentative text where meaning is spread
out; no reranker; generous token budget.

**When not to use overlap.** Highly structured records where each unit is
independent — overlap just duplicates data and inflates the index.

**The escape hatch:** you do not have to choose — **parent-document retrieval**
(§2.9) gets small-chunk precision *and* large-chunk context.

---

### - [ ] 2.3 Tokens vs characters as the unit ⚪

**What it is.** Limits are in **tokens**; most splitters count **characters**.
Roughly 1 token ≈ 4 characters of English — but that ratio collapses for code,
JSON, non-Latin scripts and long numbers.

**Example.** 900 characters of English ≈ 225 tokens. 900 characters of minified
JSON might be 400+ tokens. 900 characters of Chinese can exceed 800 tokens,
because most CJK characters are one token or more each.

**When to use character counting.** Prose in Latin scripts, where the ratio is
stable, and you want a fast splitter with no tokenizer dependency.

**When to use token counting.** Multilingual corpora, code, or when you are
close to a hard model limit. Use `tiktoken` (OpenAI) or the model's own
tokenizer via `TokenTextSplitter`.

> 🟢 App: characters, because the corpus is English prose and the 900/150
> settings were tuned empirically against measured retrieval, not against a
> token budget.

---

### - [ ] 2.4 Fixed-size chunking ⚪

**What it is.** Cut every N characters or tokens. No respect for any boundary.

**Example.** `text[0:500]`, `text[450:950]`, … A sentence, a word, even a number
can be split in half: `"$847.3 mil"` / `"lion"`.

**When to use.** Genuinely uniform machine data — log lines, sensor readings,
fixed-format records. Also as a deliberate baseline to measure against.

**When not to.** Prose. Ever. It is the strategy every other strategy exists to
improve on.

**Interview value:** worth knowing as the baseline that surprisingly often ties
with semantic chunking in published comparisons — which tells you the fancy
methods need evidence, not enthusiasm.

---

### - [ ] 2.5 Recursive character splitting 🟢

**What it is.** Try to split on the most meaningful separator that yields pieces
under the size limit. Walk down a hierarchy, and only cut mid-word as a last
resort.

**Example** — the separator list in this app:

```python
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
#              para    line  sentence  clause   word  hard-cut
```

For a 2000-char section with a 900 limit: paragraphs first. If a paragraph is
still over 900, split it into lines. Still over? Sentences. And so on. Then
greedily **merge** the fragments back up to 900 with overlap.

**When to use.** The default for prose. Cheap, deterministic, no model calls,
and it respects natural boundaries whenever it can.

**When not to.** Documents whose real structure is richer than punctuation —
use structure-aware splitting *first*, then recursive *within* each section.

> 🟢 App: `chunking._recursive_split()` and `_merge_with_overlap()`, hand-rolled
> rather than imported so the algorithm is visible and tunable. `_tail()` snaps
> the overlap forward to a word boundary so overlaps never start mid-word.

---

### - [ ] 2.6 Document-structure-aware splitting 🟢

**What it is.** Split on the boundaries the document itself declares — markdown
headings, HTML tags, PDF sections, code AST nodes. The author already told you
where the topics change; use it.

**Example, and the measured payoff in this app:**

> A 900-char window happily swallowed the tail of one section plus the head of
> the next. The blended embedding sat *between* two topics, near neither. Asking
> for "main risks" returned the **headcount** section. After making headings
> hard boundaries: the wrong section scored **0.625**, the correct one
> **0.710**.

**Also prefix the heading into the chunk**, so an isolated chunk states its own
context:

```
"## Revenue by Segment\nThe Cloud Platform segment generated $512.9 million…"
```

That is a free, deterministic approximation of contextual retrieval (§2.13).

**When to use.** Any markdown, HTML, DOCX, Confluence, or well-tagged PDF. Code
— split on function/class boundaries via AST, never mid-function.

**When not to.** Scanned PDFs and OCR output, where there is no structure to
find. Falls back to recursive or semantic.

> 🟢 App: `chunking._split_sections()` with `HEADING_RE`. Note it returns
> `[(None, text)]` for heading-less text — the branch every PDF takes.

---

### - [ ] 2.7 Semantic chunking ⚪

**What it is.** Cut where the *meaning* changes, not where a character count
runs out. Embed sentences, measure cosine distance between consecutive ones, cut
at the spikes.

**How it works, step by step:**

1. Split into sentences
2. Embed each sentence *(pass 1 — these embeddings are throwaway)*
3. Compute cosine distance between consecutive pairs
4. A **spike** means the topic changed → boundary
5. Merge the sentences between boundaries into chunks
6. **Embed the merged chunks** *(pass 2 — these are what you store)*

**Worked example** on the fixture report:

| # | Sentence | Distance to previous | |
|---|---|---|---|
| 1 | "Acme reported total revenue of $847.3 million for fiscal 2024." | — | |
| 2 | "This was an increase of 18.4% over the $715.6 million in 2023." | **0.08** | same topic |
| 3 | "Gross margin improved to 62.1% from 58.7%." | **0.15** | same topic |
| 4 | "Total headcount stood at 4,182 employees." | **0.42** | ⬅ **SPIKE → cut** |
| 5 | "R&D expense was $167.2 million, or 19.7% of revenue." | **0.12** | same topic |

Result: two chunks, split at the real topic boundary that no character count
would have found.

#### Two embedding passes — the mechanics people get wrong

**Yes, you embed twice, and the first set is discarded.**

- **Pass 1** embeds *sentences*, and exists only to **locate boundaries**.
- **Pass 2** embeds the *final merged chunks*, and those vectors are what go
  into the vector database.

Why not reuse pass 1 by averaging the sentence vectors? You *can* — some
implementations mean-pool — but averaging sentence vectors is **not** the same
as embedding the concatenated text, because in a real forward pass the tokens
attend to each other across the whole chunk. Averaging loses that. Cheaper,
slightly worse.

**Splitting is not "taken care of" for you.** `SemanticChunker` returns text
splits. Embedding and upserting them is your normal pipeline, unchanged.

#### How it meets the vector database (Qdrant and others)

**It does not.** This is the key point: *the vector database has no idea how you
chunked.* Chunking is entirely upstream. Qdrant, Pinecone, Weaviate and pgvector
all just receive `(id, vector, payload)`.

```python
# 1. chunking strategy — the ONLY thing that changes between strategies
chunks = semantic_chunk(document_text)        # or recursive, or fixed…

# 2. embed the final chunks (pass 2)
vectors = await embeddings.embed_documents([c.text for c in chunks])

# 3. upsert — identical for every chunking strategy
await client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=str(uuid4()),
            vector=vec,
            payload={
                "document_id": str(doc_id),
                "owner_id": owner_id,
                "chunk_index": i,
                "text": c.text,
                "heading": c.heading,
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ],
)
```

Swapping chunking strategy changes step 1 only. Steps 2 and 3 are untouched —
which is exactly why chunking is cheap to experiment with, and why it should be
the *first* thing you tune.

**When to use semantic chunking — concrete cases:**

| Case | Why |
|---|---|
| **Earnings call / meeting transcripts** | No headings at all. Speaker moves from revenue → guidance → a lawsuit question. Fixed-size cuts mid-topic; semantic finds the pivots |
| **Legal contracts without clause markers** | Topic shifts are real but unmarked |
| **Long-form articles, books, essays** | Narrative drift with no structural signal |
| **OCR'd scanned PDFs** | Structure was lost in scanning |

**When not to use it:**

| Case | Why |
|---|---|
| Markdown / HTML / DOCX | Structure is already there and it is **better** and free |
| Log lines, catalogs, records | Each record is already the natural unit |
| Very large corpora | One embedding call per sentence at ingestion is prohibitive |
| Code | Use AST boundaries; cosine distance between lines is meaningless |
| Short documents | Nothing to segment |

**Practical refinements.** Embed a **window** (sentence i−1 + i + i+1) rather
than single sentences — single-sentence embeddings are noisy. And pick the
**breakpoint threshold** method: percentile (cut at the 95th percentile of
distances) is the usual default because it adapts per document; standard
deviation, interquartile and gradient are the alternatives.

> ⚪🔜 **App fit — a genuinely good addition.** `_split_sections()` returns
> `[(None, text)]` when there are no markdown headings, then falls back to pure
> character splitting. That is the path **every PDF** takes: markdown gets
> structure for free, PDFs get nothing. Semantic chunking belongs exactly on
> that branch, and would be measurable against current behaviour.

---

### - [ ] 2.8 Propositional chunking ⚪

**What it is.** An LLM rewrites the text into standalone atomic facts, each
becoming a chunk. Every proposition must be self-contained — no pronouns, no
"the above".

**Example.**

Original:

> "Gross margin improved to 62.1% from 58.7%, driven primarily by a shift in
> product mix toward higher-margin subscription offerings."

Propositions:

1. "Acme Corporation's gross margin was 62.1% in fiscal 2024."
2. "Acme Corporation's gross margin was 58.7% in fiscal 2023."
3. "Acme Corporation's gross margin improvement was driven primarily by a shift
   in product mix toward higher-margin subscription offerings."

Each is independently retrievable and needs no surrounding context.

**When to use.** High-precision factual Q&A over dense reference text, where
recall of a *specific* fact matters more than narrative flow. Also good feeding
a knowledge graph.

**When not to.** Almost everywhere else. One LLM call per passage at ingestion
is expensive; nuance, hedging and qualifiers get flattened; and a
hallucinated proposition is now an indexed "fact" you will retrieve and cite
with confidence. That last risk is the real reason it stays rare.

---

### - [ ] 2.9 Parent-document retrieval (small-to-big) ⚪

**What it is.** Decouple **what you search** from **what you return**. Index
small chunks for precise vectors; return their larger parent for context.

**Example.** Query "what was the gross margin?"

Child chunk that matches (precise, high cosine):

> "Gross margin improved to 62.1% from 58.7%."

Parent returned to the model instead:

> "Acme Corporation reported total revenue of $847.3 million for fiscal year
> 2024, an increase of 18.4%… Gross margin improved to 62.1% from 58.7%,
> **driven primarily by a shift in product mix toward higher-margin subscription
> offerings.** Operating income reached $112.8 million…"

Same precise match; the answer can now explain *why*, not just recite a number.

#### How the small chunks are chosen

**They are not chosen — they are exhaustive.** This is the part that confuses
people. You do not pick "the best one or two sentences" from a paragraph. You
run a **second, finer splitting pass** over the same text, and **every** child
is indexed.

```
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=300)

for parent in parent_splitter.split(document):
    parent_id = store_parent(parent)              # docstore, not vector store
    for child in child_splitter.split(parent):    # ALL children
        index(child, metadata={"parent_id": parent_id})
```

So a 2000-char parent yields ~7 children, all embedded and all searchable. At
query time, retrieval decides which child matched; you look up its
`parent_id` and return the parent. **Selection happens at retrieval, not at
indexing.**

The vector store holds children. A separate key-value docstore holds parents.

**Variants:**

| Variant | How it expands |
|---|---|
| **Parent-document** | Child → its explicit parent (LangChain `ParentDocumentRetriever`) |
| **Sentence-window** | Children are *individual sentences*; expand ±k sentences at query time |
| **Auto-merging / hierarchical** | Multi-level tree; if enough *sibling* children match, return the parent **instead of** the children |

Auto-merging is the clever one because it adapts: scattered matches return
leaves, concentrated matches collapse into a parent, avoiding near-duplicate
context.

**When to use.** Almost always worth considering — it resolves §2.2's trade-off
rather than compromising on it. Especially good when answers need explanation,
not just a fact.

**When not to.** Tight context budgets (parents are big — cap their size or you
lose the savings small chunks bought you). Also skip it when chunks are already
independent records with no meaningful "parent".

**Trap.** **Deduplicate.** Two matching children with the same parent must not
send that parent twice.

> ⚪🔜 **App fit — sentence-window is nearly free.** `chunks` rows already carry
> `document_id` and **`chunk_index`**, so expanding a hit to its neighbours is
> one query, with no re-indexing and no migration:
>
> ```sql
> SELECT text FROM chunks
> WHERE document_id = :doc AND chunk_index BETWEEN :i - 1 AND :i + 1
> ORDER BY chunk_index;
> ```
>
> Full parent-document retrieval would want an explicit `parent_id` column —
> that is a migration.

---

### - [ ] 2.10 Sentence-window retrieval ⚪

**What it is.** The narrowest form of small-to-big: index **one sentence per
vector**, then at query time return that sentence plus a window of `k`
neighbours.

**Example.** Match on "The segment grew 27.3% year over year." Return the three
sentences before and after, which is where "Cloud Platform" is named.

**When to use.** Dense factual text where the precise sentence is findable but
meaningless alone. Very effective and simple — no parent bookkeeping, just
`chunk_index ± k`.

**When not to.** Text where sentences are long and self-contained already, or
where the window would repeatedly overlap between multiple hits (you then need
merging and dedup).

---

### - [ ] 2.11 Auto-merging / hierarchical chunking ⚪

**What it is.** Build a tree: large parents → medium → small leaves. Index the
leaves. If **enough sibling leaves under the same parent** match a query,
discard them and return the parent instead.

**Example.** A query hits 4 of the 6 leaves inside the Financial Summary
section. Rather than sending four fragments with overlapping content, return the
whole section once — fewer tokens, better coherence.

**When to use.** Broad or summarising queries over long structured documents,
where hits naturally cluster.

**When not to.** Small corpora or short documents — the tree costs more than it
saves. Also fiddly to tune: the merge threshold ("how many siblings?") is
another hyperparameter with no obvious default.

---

### - [ ] 2.12 Late chunking ⚪

**What it is.** Invert the usual order. Embed the **whole document** first so
every token has attended to the entire document, *then* apply chunk boundaries
and **mean-pool the token vectors within each boundary**.

```
NORMAL:  split → [chunk] → transformer → pool → vector    (context-blind)
LATE:    whole document → transformer → token vectors
                                          ↓ split + pool per chunk
                                        vectors            (context-aware)
```

**Example.** The chunk *"The segment grew 27.3% year over year."* embedded
independently is nearly useless — *which* segment? With late chunking, those
tokens already attended to "Cloud Platform" earlier in the document, so the
pooled vector carries the association. **The text is unchanged; the vector is
smarter.**

#### How you chunk once the document has gone through the embedder

You **do not** chunk afterwards, and you do not derive boundaries from the
embedder. Boundaries are decided **independently, by any normal method** —
fixed, recursive, structure-aware, even semantic. Late chunking only changes
*when pooling happens*:

1. **Decide boundaries** on the raw text, as character spans
   → `[(0, 900), (750, 1650), …]`
2. **Tokenize the whole document** with an offset mapping, so you know which
   character range each token covers
   → `tokenizer(text, return_offsets_mapping=True)`
3. **One forward pass** over the whole document → a token-embedding matrix
   `[n_tokens, dim]`
4. For each chunk's character span, **find its token range** via the offset
   mapping, and **mean-pool those rows**
5. Normalise → one vector per chunk → upsert as usual

```python
enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
token_vecs = model(**enc).last_hidden_state[0]          # [n_tokens, dim]
offsets = enc["offset_mapping"][0]

chunk_vectors = []
for start_char, end_char in boundaries:
    idx = [i for i, (s, e) in enumerate(offsets)
           if s >= start_char and e <= end_char]
    v = token_vecs[idx].mean(dim=0)
    chunk_vectors.append(v / v.norm())                   # L2 normalise
```

Step 2's offset mapping is the whole trick — it is what lets you map character
boundaries onto token rows.

**When to use.** Documents whose chunks depend heavily on document-wide context
(pronouns, abbreviations, entities introduced once at the top), and where you
control the embedding model.

**When not to.** With any black-box embedding API that returns one pooled vector
— it is simply impossible. Also when documents exceed the model's context
window, which needs a hierarchical variant.

> ⚪ **App fit — architecturally possible, practically blocked.**
> `EMBEDDING_PROVIDER` is an ABC with a local `fastembed` option, which is the
> right shape. But `fastembed` returns pooled vectors, not token embeddings, so
> this needs `sentence-transformers` or raw ONNX with manual pooling. The Gemini
> API path cannot do it at all.

---

### - [ ] 2.13 Contextual retrieval 🔵

**What it is.** Anthropic's technique (Sept 2024). Same problem as late chunking
— context-blind chunks — solved with **text instead of geometry**. One LLM call
per chunk, given the whole document, produces a sentence situating that chunk.
Prepend it before embedding **and** before BM25 indexing.

**Example.**

Original chunk:

> "The segment grew 27.3% year over year."

Contextualised chunk that gets indexed:

> "This chunk is from Acme Corporation's 2024 annual report, in the Revenue by
> Segment section, discussing the Cloud Platform segment which generated $512.9
> million and 60.5% of total revenue. The segment grew 27.3% year over year."

Now a query about "Cloud Platform growth" matches via embeddings **and** via
keyword search, because the words are literally present.

**Measured results (worth memorising):**

| Setup | Retrieval failure rate |
|---|---|
| Baseline (embeddings only) | 5.7% |
| + contextual embeddings | 3.7% (**−35%**) |
| + contextual BM25 | 2.9% (**−49%**) |
| + reranking (150 → 20) | 1.9% (**−67%**) |

**Cost, and what makes it viable.** One LLM call per chunk sounds ruinous. It
works because of **prompt caching**: the document is the large repeated prefix,
cached once, so each chunk call pays only for the chunk plus a short output.
Roughly $1 per million document tokens.

**Contextual retrieval vs late chunking:**

| | Contextual retrieval | Late chunking |
|---|---|---|
| Changes the **text** | **yes** | no |
| Changes the **vector** | indirectly | **directly** |
| Helps **BM25** | **yes** | no |
| Needs token-level access | no | **yes** |
| Interpretable | **yes** — you can read it | no |
| Cost | one LLM call per chunk | one long forward pass per document |

**When to use.** Whenever ingestion cost is acceptable and retrieval quality
matters — it is currently the highest-value single upgrade to a naive pipeline,
and it works with any black-box embedding API.

**When not to.** Very large or frequently-changing corpora (you re-pay per
chunk on every re-ingest), or when no LLM budget exists at ingestion time.

> 🔵 **App fit — a cheap approximation already ships.**
> `chunk_pages()` prefixes the markdown heading into every chunk, which is
> deterministic, free, needs no LLM, and **measurably worked** (0.625 → 0.710).
> Contextual retrieval is the LLM-powered generalisation: it works with no
> headings, and can pull facts from elsewhere in the document. The ingestion
> pipeline already has an `ENRICHERS` protocol hook designed for exactly this
> class of per-chunk enrichment.

---

### - [ ] 2.14 Metadata attached at chunk time 🟢

**What it is.** Whatever you store on a chunk is the *only* thing you can filter
or cite by later. Metadata is a schema decision disguised as a detail.

**Example** — the payload in this app: `document_id`, `owner_id`,
`chunk_index`, `page`, `heading`, `filename`, `text`. Each enables something
specific:

| Field | Enables |
|---|---|
| `owner_id` | multi-tenant isolation (§1.7) |
| `document_id` | per-document scoping, and delete-by-document |
| `chunk_index` | sentence-window expansion (§2.10) |
| `page` / `heading` | honest citations the user can verify |

**When to add more.** Dates for recency filtering, author, document type,
language, section level, access tags for permission filtering.

**When not to.** Anything large — payloads are held in memory alongside
vectors. Do not store the whole parent document in every child's payload.

**Trap.** You cannot filter on what you did not store, and adding a field later
means **re-ingesting everything**. Decide early and over-store cheap scalars.

> 🟢 App: metadata was deliberately over-provisioned at step 2 specifically so
> per-document filtering and citation could be added later without re-ingest.

---

### - [ ] 2.15 Tables, figures, code, headers/footers ⚪

**What it is.** The parts of a document that break naive text extraction.

**Examples of what goes wrong.**

- **Tables** flatten into unreadable token soup. Row/column association is lost,
  so "Cloud Platform | $512.9M | 27.3%" becomes three numbers next to a name
  with no relationship.
- **Repeated headers/footers** appear on every page, so every chunk gets
  polluted with "Acme Corporation — Confidential — Page 4 of 52", which
  dominates short chunks and creates false similarity between unrelated pages.
- **Code blocks** get split mid-function by prose-oriented separators.
- **Figures** are pixels; their captions are the only retrievable signal.

**Techniques.** Extract tables separately and serialise as markdown or
row-per-chunk with the header repeated; detect and strip repeating page
furniture by frequency across pages; route code to an AST splitter;
LLM/VLM-based document parsing (Unstructured, LlamaParse, Docling, Textract) for
complex layouts; multimodal embeddings or caption-then-embed for figures.

**When to invest.** Financial reports, scientific papers, invoices — documents
where the table *is* the content.

**When not to.** Plain prose corpora. The extra pipeline is not free and adds
failure modes.

> ⚪ App: `parsing.py` uses `pypdf` text extraction only. Tables flatten,
> headers/footers are not stripped. A real gap for any real PDF.

---

### - [ ] 2.16 Filtering degenerate chunks 🟢

**What it is.** Dropping chunks with too little content to answer anything. They
do active harm, not just waste.

**Example — measured in this app.** A heading with no body of its own produced a
**39-character chunk of pure title**: `"# Acme Corporation — Annual Report
2024"`. It ranked **second in every retrieval**, because a bare title embeds
close to almost any question about the document. It consumed one of five slots
and contributed nothing any answer could cite.

**Implementation detail that matters.** Measure the **body length, minus the
heading prefix**. Because every chunk carries its heading, a long title would
otherwise clear any threshold you set.

**Guard.** If filtering would empty a document, keep everything — a one-line
file is a legitimate upload, and returning zero chunks should not fail
ingestion.

**What else to filter.** Navigation boilerplate, copyright footers, "This page
intentionally left blank", pure whitespace, and near-duplicate chunks.

**When not to.** Do not set the threshold high. Genuinely short sections
("**Risks:** None identified.") are real answers.

> 🟢 App: `chunking._drop_contentless()`, `min_chunk_chars: int = 50`. Affects
> new ingests only — existing documents keep their chunks until re-ingested.

## 3. Embeddings

### - [ ] 3.1.1 What an embedding is ⚪

**What it is.** A function from text to a fixed-length list of floats. Texts
that mean similar things land in similar **directions**. "Embedding" is the act
of *computing that vector* — it has nothing to do with storing it.

**Example.** `embed("revenue grew")` → `[0.021, −0.118, 0.077, …]` (768 numbers).
`embed("sales increased")` → a vector pointing in almost the same direction.
`embed("the cat sat")` → somewhere unrelated.

**When to use.** Semantic similarity, clustering, classification, dedup,
recommendation — anything where "means the same thing" beats "contains the same
words".

**When not to.** Exact matching. An embedding of `INV-2024-0042` is not reliably
closer to that exact string than to `INV-2024-0043`. Identifiers, SKUs, error
codes and names want **keyword search** (§5.1.2).

---

### - [ ] 3.1.2 How embedding models are trained ⚪

**What it is.** **Contrastive learning.** Show the model pairs that should be
close (positives) and pairs that should be far (negatives), and train so that
cosine similarity is high for positives and low for negatives (InfoNCE loss).

**Example.** Positive pair: a question and the passage that answers it.
Negative: that same question and a random passage. **Hard negatives** — passages
that look relevant but are not — matter far more than random ones; they are what
teach the model fine distinctions.

**Why this matters practically.** The model is only good at the *kind of
similarity it was trained on*. A model trained on web Q&A pairs is good at
question→passage. It is not automatically good at code→code or
legal-clause→legal-clause. That is what domain-specific models and fine-tuning
(§3.5.6) exist for.

---

### - [ ] 3.1.3 Bi-encoder vs cross-encoder ⚪

**The single most important distinction in retrieval.** Get this right and
reranking, latency budgets, and "why not just use the good model?" all follow.

| | **Bi-encoder** | **Cross-encoder** |
|---|---|---|
| Input | query and document **separately** | query and document **together**, as one sequence |
| Output | a vector each | a single relevance **score** |
| Precompute documents? | **Yes** — embed once at ingestion | **No** — nothing to precompute |
| Cost per query | 1 embed + ANN search | **N model passes**, one per candidate |
| Accuracy | good | **much better** |
| Use for | **retrieval** over millions | **reranking** a shortlist |

**Example.** 1M chunks. A bi-encoder embeds all of them offline, then a query is
one embed plus an ANN lookup — milliseconds. A cross-encoder would need 1M
forward passes *per query*. Utterly infeasible. But over the top 100 candidates
it is 100 passes — perfectly affordable, and far more accurate because the model
can attend *across* query and document at once.

**This is why the standard pattern is retrieve-N-rerank-K** (§6.2).

**When not to use a cross-encoder.** Never for first-stage retrieval. And skip
it entirely if recall is your problem — reranking cannot surface a document that
retrieval never returned.

---

### - [ ] 3.1.4 Pooling strategies ⚪

**What it is.** A transformer produces one vector **per token**. Pooling
collapses those into one vector for the whole text.

| Strategy | How |
|---|---|
| **CLS** | Take the special `[CLS]` token's vector (BERT-style) |
| **Mean pooling** | Average all token vectors (usually attention-mask-weighted) |
| **Last token** | Take the final token (decoder-style models) |

**Example.** Mean pooling of 8 token vectors of dim 768 → one 768-dim vector.

**Why it matters here.** Mean pooling is what makes **late chunking** (§2.12)
possible: you pool over a *subset* of tokens — the ones inside a chunk boundary
— instead of over the whole input. Also: **you must use the pooling the model
was trained with.** Mean-pooling a CLS-trained model quietly degrades quality.

---

### - [ ] 3.2.1 Cosine similarity and distance 🟢

**What it is.** The angle between two vectors, ignoring their length:

```
cos_sim(A, B) = (A · B) / (|A| × |B|)

  A · B = Σ Aᵢ × Bᵢ        (dot product)
  |A|   = √(Σ Aᵢ²)          (magnitude / L2 norm)

cos_dist = 1 − cos_sim
```

Range **−1 → +1**: `+1` identical direction, `0` unrelated (90°), `−1` opposite.
Distance is 0 → 2, i.e. similarity inverted so "closer = smaller", which is what
index structures want.

**Worked example** with `A = [1, 0]`:

| B | A·B | \|B\| | cos_sim | cos_dist | Reading |
|---|---|---|---|---|---|
| `[0.9, 0.44]` | 0.9 | 1.002 | **0.898** | 0.102 | very similar |
| `[0.5, 0.87]` | 0.5 | 1.004 | **0.498** | 0.502 | loosely related |
| `[0, 1]` | 0 | 1 | **0.0** | 1.0 | unrelated |
| `[−1, 0]` | −1 | 1 | **−1.0** | 2.0 | opposite |

Real embeddings do this in 768 or 3072 dimensions. The arithmetic is identical.

**When to use.** The default for text embeddings, because magnitude usually
encodes *length*, not meaning — a long passage and a short one on the same topic
should be close, and cosine makes them so.

**When not to.** Where magnitude is meaningful — some recommender embeddings
encode popularity or confidence in the norm, and there dot product is correct.

> 🟢 App: Qdrant collection created with `Distance.COSINE`.

---

### - [ ] 3.2.2 Dot product 🟢

**What it is.** `A · B = Σ Aᵢ × Bᵢ`. Unbounded, and **sensitive to magnitude**.

**Example.** `A=[1,0]`, `B=[0.9,0.44]` → dot = 0.9. Now double B to
`[1.8, 0.88]` → dot = 1.8, but the *angle is unchanged*, so cosine is still
0.898. Dot product says "twice as similar"; cosine says "same direction".

**When to use.** When vectors are already normalised (then it equals cosine and
is **cheaper** — no division, no norms, which is why many indexes use it
internally), or when magnitude genuinely carries signal.

**When not to.** On unnormalised text embeddings, where long documents would win
purely for being long.

---

### - [ ] 3.2.3 Euclidean (L2) distance ⚪

**What it is.** Straight-line distance: `√(Σ (Aᵢ − Bᵢ)²)`. Range 0 → ∞, smaller
is closer.

**Example.** `A=[1,0]`, `B=[0,1]` → `√2 ≈ 1.414`.

**When to use.** Image embeddings and non-text vectors where absolute position
matters. Also the natural metric for many clustering algorithms (k-means).

**When not to.** Unnormalised text embeddings — same length-bias problem as dot
product.

**The key relationship:** for **unit-length** vectors,
`L2² = 2 − 2·cos_sim`. It is a monotonic transform, so **L2 and cosine produce
the identical ranking.** Choosing between them then only affects arithmetic
cost, not results.

---

### - [ ] 3.2.4 L2 normalisation 🟢

**What it is.** Scaling a vector to length 1: `A_norm = A / |A|`. Direction is
preserved, magnitude discarded.

**Example.** `A = [3, 4]`, `|A| = 5` → `A_norm = [0.6, 0.8]`, and
`√(0.6² + 0.8²) = 1`. ✓

**Why it matters — the payoff.** Once every vector is unit length:

- cosine similarity **= dot product** exactly
- Euclidean distance is a monotonic function of cosine, so **all three rank
  identically**

So normalisation lets you pick the metric on cost grounds and stop worrying.

**When not to.** When magnitude carries signal you want to keep.

> 🟢 **App — a real trap.** Gemini returns 3072-dim vectors that *are*
> L2-normalised. Requesting `outputDimensionality=768` truncates them — and
> Google normalises only the **full** vector, so the truncated one is **no
> longer unit length**. `services/embeddings.py` re-normalises client-side.
> Skip that and cosine goes subtly wrong: not broken enough to notice in a
> smoke test, just quietly worse recall forever.

---

### - [ ] 3.2.5 Scores are not comparable across queries 🟢

**What it is.** A cosine score is meaningful only *relative to other results for
the same query*. There is no absolute scale.

**Example — measured in this app.** An **irrelevant** match scored **0.570**,
while a **correct** match for a different query scored **0.615**. A global
threshold of "relevance ≥ 0.6" would have kept the wrong one and had no
consistent meaning.

**Consequences.**

- Do not set global similarity thresholds and expect them to hold.
- Do not average or compare raw scores across queries in evaluation.
- **Fuse on rank, not score** — this is the entire justification for RRF
  (§5.2.1).
- If you need calibrated confidence, use a **cross-encoder** (§3.1.3) or
  train a calibrator; bi-encoder cosine is not it.

---

### - [ ] 3.3.1 Dimensions vs storage vs latency 🟢

**What it is.** Everything scales linearly with dimension count.

**Example.** 1M chunks at float32:

| Dims | Bytes/vector | Total |
|---|---|---|
| 384 | 1,536 | ~1.5 GB |
| 768 | 3,072 | ~3.1 GB |
| 1536 | 6,144 | ~6.1 GB |
| 3072 | 12,288 | ~12.3 GB |

HNSW holds the graph **in memory**, so this is RAM, not disk. Distance
computation cost scales the same way.

**When to use high dimensions.** Large, semantically subtle corpora where you
have measured a recall gain.

**When not to.** By default. More dimensions is not automatically better — and
you should *measure* recall at 768 vs 1536 before paying 2× for memory.

> 🟢 App: 768 chosen over the native 3072, cutting Qdrant storage ~4×.

---

### - [ ] 3.3.2 Matryoshka Representation Learning (MRL) 🟢

**What it is.** A training technique that makes a vector's **leading dimensions
carry most of the signal**, so you can truncate it and keep most of the quality.
Named after nesting dolls — a 3072-dim vector contains a usable 1536, which
contains a usable 768, and so on.

**Example.** OpenAI `text-embedding-3-large` is 3072 and documented as safe to
truncate to 256 or 1024. Gemini accepts `outputDimensionality`. **Truncation is
literally `vector[:768]`** — free at query time, no re-embedding.

**When to use.** Whenever storage or latency matters more than the last few
points of recall — which is most production systems.

**When not to.** On a model **not trained with MRL.** Truncating an ordinary
embedding destroys it, because there the signal is spread across all dimensions
with no ordering. This is the trap: `vector[:768]` *runs* on any model, and only
silently ruins quality on non-MRL ones.

**Always re-normalise after truncating** — see §3.3.3.

---

### - [ ] 3.3.3 Re-normalising after truncation 🟢

**What it is.** A truncated slice of a unit vector is **not** a unit vector,
because you removed components that contributed to its length.

**Example.** Unit vector `[0.6, 0.8]` (length 1). Truncate to `[0.6]` → length
**0.6**, not 1. Re-normalise → `[1.0]`.

**Why it matters.** If you use dot product as your metric (very common, and
what many indexes do internally), unnormalised vectors give wrong distances. If
you use cosine, the *ranking* survives because cosine divides by the norm — but
any score threshold, and anything mixing normalised and unnormalised vectors,
breaks.

**When you can skip it.** Only if you are certain every consumer uses true
cosine and never dot product. Not worth the risk; it is one line.

> 🟢 App: `services/embeddings.py` re-normalises every truncated vector before
> storing.

---

### - [ ] 3.3.4 Quantization of embeddings ⚪

**What it is.** Storing each dimension in fewer bits.

| Type | Bits/dim | Memory vs float32 | Typical recall retained |
|---|---|---|---|
| float32 | 32 | 1× | 100% |
| float16 | 16 | 2× smaller | ~100% |
| **int8 / scalar** | 8 | **4× smaller** | ~98–99% |
| **binary** | 1 | **32× smaller** | ~90–95% |

**Example.** Binary quantization maps each dimension to its sign bit and
replaces cosine with **Hamming distance** — a XOR and a popcount, which is
extraordinarily fast. Usual pattern: binary search for a wide shortlist, then
**rescore** the shortlist with the full-precision vectors.

**When to use.** Above a few million vectors, where RAM is the binding cost.
int8 is nearly free quality-wise and the easy default.

**When not to.** Small collections — the complexity buys nothing. And binary
without a rescoring stage, which loses real recall.

---

### - [ ] 3.4.1 Symmetric vs asymmetric search ⚪

**What it is.** Whether the two things you compare are the *same kind* of text.

- **Symmetric**: query and target are alike — duplicate detection, "find similar
  articles", STS.
- **Asymmetric**: a **short query** against a **long passage** — which is what
  RAG is.

**Example.** Query "gross margin 2024" (4 words) against a 900-character
passage. These are structurally different objects, and models trained for
asymmetric retrieval handle the mismatch. A model trained on symmetric sentence
similarity will do measurably worse.

**When it matters.** Model selection. MTEB separates *Retrieval* (asymmetric)
from *STS* (symmetric) for exactly this reason — a model topping STS is not
necessarily good at RAG.

---

### - [ ] 3.4.2 Query/passage prefixes and task types 🔵

**What it is.** Many models require you to **label** which side you are
embedding. Get it wrong and nothing errors — quality just drops.

**Examples.**

| Model | Query side | Document side |
|---|---|---|
| E5 family | `"query: gross margin"` | `"passage: Acme reported…"` |
| BGE | an instruction prefix on queries | raw text |
| Gemini | `task_type=RETRIEVAL_QUERY` | `task_type=RETRIEVAL_DOCUMENT` |
| Nomic | `search_query:` | `search_document:` |

**When to use.** Always, if your model documents it. It is one of the cheapest
quality wins available and one of the most commonly missed.

**When not to.** Models that do not use prefixes — adding them just pollutes the
text.

**Trap.** The prefixes must be **asymmetric between ingestion and query**. Using
`passage:` on both sides is a silent regression that no test catches.

> 🔵 App: separate `embed_query()` and `embed_documents()` methods exist — the
> right seam — but worth auditing that Gemini's `task_type` is actually set
> differently on each path.

---

### - [ ] 3.4.3 Token limits and truncation ⚪

**What it is.** Every model has a max input length and most **silently
truncate** past it. No error, no warning — you get a vector for the first N
tokens and a false sense of coverage.

| Model | Max tokens |
|---|---|
| MiniLM / BGE / E5 | 512 |
| Gemini `embedding-001` | ~2,048 |
| OpenAI `text-embedding-3` | 8,191 |
| Jina v2/v3, Nomic | 8,192 |

**Example.** Feed a 4,000-token chunk to a 512-token model and ~87% of it is
invisible. Retrieval on that chunk is effectively retrieval on its first
paragraph.

**When it matters.** Always check `chunk_size` against the model limit *in
tokens*, not characters (§2.3). This is a common silent bug.

---

### - [ ] 3.4.4 Batching, rate limits, and cost 🟢

**What it is.** Embedding APIs accept arrays. One request with 100 texts is
vastly cheaper in overhead than 100 requests — but rate limits usually bind on
**tokens per minute**, not requests.

**Example — measured in this app.** `batchEmbedContents` caps at 100 items per
call (250 returns a 400). But 100 chunks × ~225 tokens ≈ **22,500 tokens in one
request**, and the limit is 30,000 TPM. So perfect batching still only allows
~1.3 batches/minute ≈ **133 chunks/minute**.

**The lesson.** A limiter that counts only *requests* will still earn you 429s.
You need a **dual token-bucket**: requests/min **and** tokens/min.

**When not to batch maximally.** When latency matters per item (interactive
single-query embedding), or when a failed batch means redoing all 100.

> 🟢 App: `services/limiter.py` implements dual buckets; `EMBEDDING_BATCH_SIZE=100`.

---

### - [ ] 3.4.5 Model swap = full re-index 🟢

**What it is.** Vectors from different models are **not comparable**, even at
the same dimension count. Different models place meaning in different
directions.

**Example.** Embed documents with `bge-small` (384) and queries with
`all-MiniLM-L6` (also 384). Both "work" — the arithmetic runs, cosine returns
numbers — and results are **garbage**. Nothing errors.

**Consequences.** Changing embedding model, or its dimension, or its pooling,
means **re-embedding the whole corpus**. Vector collections are
fixed-dimension, so it also means creating a new collection.

**Migration pattern.** Dual-write into a new collection, backfill, compare
recall on a golden set, then cut over. Never mutate in place.

> 🟢 App: `EMBEDDING_DIM` is documented as requiring a collection recreate and
> full re-ingest.

---

### - [ ] 3.4.6 Embedding drift and re-embedding ⚪

**What it is.** Two different problems often confused.

1. **Model drift** — the *provider* updates the model behind a stable name, so
   new vectors are subtly inconsistent with old ones. Mitigation: pin
   **versioned** model IDs.
2. **Data drift** — your corpus vocabulary moves (new products, new jargon) and
   the frozen model no longer represents it well. Mitigation: periodic
   re-embedding, or fine-tuning.

**When to re-embed.** Model upgrade, dimension change, chunking change, or
adding contextual enrichment (§2.13) — all of which change the text or the
mapping.

**When not to.** Merely because a newer model exists. Measure on a golden set
first; re-embedding a large corpus is expensive and sometimes a downgrade.

---

### - [ ] 3.5.1 Proprietary models ⚪

| Model | Dims | Notes |
|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 (MRL) | cheap, strong default |
| OpenAI `text-embedding-3-large` | 3072 (MRL) | best of the OpenAI line |
| Cohere `embed-v3` | 1024 | strong multilingual; has a **compression-aware** variant |
| Voyage `voyage-3` | 1024 | domain variants (code, finance, law) |
| Gemini `embedding-001` | 3072 (MRL) | `task_type` prefixes |

**When to use.** No GPU, no ops appetite, and you want a strong baseline
immediately.

**When not to.** Data residency or privacy constraints; very high volume where
per-token cost dominates; or you need token-level output (late chunking), which
APIs do not expose.

---

### - [ ] 3.5.2 Open models ⚪

| Family | Notes |
|---|---|
| **E5** (`multilingual-e5-large`) | strong, needs `query:`/`passage:` prefixes |
| **BGE** (`bge-base-en-v1.5`) | very strong for size; BAAI also ships a reranker |
| **GTE** | Alibaba; good general purpose |
| **Nomic** `nomic-embed-text-v1.5` | 8k context, MRL, open weights and data |
| **Jina** v2/v3 | 8k context, **late-chunking support** |
| **Stella / gte-Qwen** | leaderboard-toppers, larger |

**When to use.** Cost at volume, privacy, offline capability, or you need
token-level access.

**When not to.** When you have no GPU and latency matters — CPU inference on a
large model is slow. `bge-small`/`MiniLM` on ONNX is the practical CPU choice.

> 🟢 App: `fastembed` with `BAAI/bge-small-en-v1.5` is wired as an offline
> provider — no API, no quota.

---

### - [ ] 3.5.3 Multi-vector: ColBERT, ColPali ⚪

**What it is.** Instead of one vector per chunk, store **one vector per token** —
a *matrix* per document. Scoring is **MaxSim**: for each query token, take its
best match among document tokens, then sum.

```
score(q, d) = Σ over query tokens  max over doc tokens  cos(qᵢ, dⱼ)
```

**This is almost certainly what "matrix retrieval" refers to.**

**Example.** Query "Cloud Platform growth". "Cloud" matches the document token
"Cloud" almost exactly; "growth" matches "grew". Single-vector retrieval averages
everything into one point and can miss that both terms are present in the right
places. Late interaction preserves token-level evidence.

**When to use.** When quality justifies the cost, especially as a **reranker**
over a shortlist. **ColPali** is the notable variant: it embeds *page images*
directly, which is excellent for visually complex PDFs where text extraction
fails.

**When not to.** As first-stage retrieval at scale — storage is 10–100× a single
vector, and index support is limited (Qdrant, Vespa and a few others handle
multi-vectors natively).

---

### - [ ] 3.5.4 Learned sparse: SPLADE, uniCOIL ⚪

**What it is.** A neural model outputs a **sparse** vector over the vocabulary —
like BM25, but the weights are *learned*, and it performs **term expansion**: a
document about "revenue" gets non-zero weight on "sales", "income", "turnover"
even if those words never appear.

**Example.** BM25 on "income" fails to match a document saying only "revenue".
Dense retrieval matches but cannot guarantee exact terms. SPLADE gets both: the
exactness of term matching plus learned synonymy.

**When to use.** Hybrid setups where you want one model to cover keyword and
some semantic matching, and your engine supports sparse vectors (Qdrant,
OpenSearch, Vespa, Pinecone all do).

**When not to.** When BM25 already suffices — SPLADE needs a model at index
time and produces larger postings lists, so indexing is slower and heavier.

---

### - [ ] 3.5.5 Multilingual and domain-specific models ⚪

**What it is.** Models trained for cross-lingual alignment, or on a specific
domain's language.

**Example.** With `multilingual-e5`, a French query can retrieve an English
passage because both map into a shared space. With an English-only model it
cannot. Domain models (`voyage-code`, `voyage-law`, `BioBERT`-lineage) beat
general models on their domain because the *notion of similarity* differs — in
code, two functions with different names and identical structure are similar; in
prose they would not be.

**When to use.** Multilingual corpora, or a domain with heavy jargon where a
general model's similarity judgements are wrong.

**When not to.** Multilingual models generally trade a little English quality
for coverage — do not use one for an English-only corpus.

---

### - [ ] 3.5.6 Fine-tuning an embedding model ⚪

**What it is.** Continue contrastive training on **your** (query, relevant
passage) pairs so the model learns your domain's notion of similarity.

**Example.** Your users search "PTO policy" but documents say "annual leave
entitlement". A general model may rank these apart. Fine-tune on ~1,000 real
(query, correct chunk) pairs — often harvestable from click logs or generated
synthetically from your documents — and that gap closes.

**When to use.** You have (or can synthesise) labelled pairs, a stable domain,
and measured evidence that off-the-shelf retrieval is the bottleneck.

**When not to.** Before trying better chunking, hybrid search and reranking —
all cheaper and usually bigger wins. Also remember fine-tuning means
**re-embedding the entire corpus** and re-validating.

---

### - [ ] 3.5.7 MTEB, read critically ⚪

**What it is.** Massive Text Embedding Benchmark — the standard leaderboard,
spanning retrieval, STS, classification, clustering, reranking.

**How to read it properly.**

- Look at the **Retrieval** column, not the overall average, if you are doing
  RAG (§3.4.1).
- Check **dimensions and model size** — a 7B model topping the board may be
  irrelevant to your latency budget.
- Beware **benchmark contamination**: heavily-tuned models can overfit MTEB and
  underperform on your data.
- **Always validate on your own golden set.** A model 2 points lower on MTEB can
  easily be better on your corpus.

**When not to trust it.** Domain-specific corpora — MTEB is mostly general web
and academic text.

## 4. Indexing and vector databases

### - [ ] 4.1.1 Flat / brute force ⚪

**What it is.** Compare the query against **every** vector. No index structure.
Called `FLAT` in FAISS, exact search elsewhere.

**Example.** 10,000 vectors × 768 dims = 7.7M multiply-adds per query. On modern
SIMD hardware that is roughly a millisecond. Perfectly fine.

**When to use.** Below ~10k–100k vectors. It is **exact** — 100% recall by
definition — and needs no tuning, no build step, and no parameters to get
wrong. Also the correct choice as a *ground truth* when measuring an ANN index's
recall.

**When not to.** Millions of vectors, where O(n) per query becomes seconds.

**Interview point:** many teams build HNSW for 5,000 vectors. That is pure
complexity with no benefit. **Know when the index is unnecessary.**

---

### - [ ] 4.1.2 HNSW 🟢

**What it is.** Hierarchical Navigable Small World. A multi-layer proximity
graph. Top layers are sparse with long-range links; lower layers dense with
short-range links. Search enters at the top, greedily walks toward the query,
then descends — like a skip list in vector space.

**Parameters, and what they trade:**

| Param | Meaning | Effect |
|---|---|---|
| `M` | links per node | higher = better recall, more memory |
| `ef_construction` | candidate list size while **building** | higher = better graph, slower build |
| `ef_search` | candidate list size while **searching** | higher = better recall, slower query. **Tunable at query time** |

**Example.** `M=16, ef_construction=100` is a common default. If recall is poor,
raise `ef_search` first — it costs nothing to change and requires no rebuild.
`M` and `ef_construction` need a rebuild.

**When to use.** The default for most vector databases and most workloads:
excellent recall/latency, and it supports incremental inserts (unlike IVF, which
wants a training step).

**When not to.** Memory-constrained deployments — HNSW holds the graph **in
RAM** and the graph itself is a significant overhead on top of the vectors. At
very large scale, IVF+PQ or DiskANN cost far less memory.

> 🟢 App: Qdrant defaults — HNSW with cosine.

---

### - [ ] 4.1.3 IVF ⚪

**What it is.** Inverted File index. Cluster all vectors (k-means) into `nlist`
cells, each with a centroid. At query time, find the nearest `nprobe` centroids
and search **only** those cells.

**Example.** 1M vectors, `nlist=1024` → ~1,000 vectors per cell. With
`nprobe=10` you search ~10,000 vectors instead of 1M — a 100× reduction.

**The recall risk.** If the true nearest neighbour sits in a cell whose centroid
was *not* in the top `nprobe`, you never see it. Raising `nprobe` improves recall
and costs latency, linearly.

**When to use.** Large static datasets, memory-constrained, and paired with PQ
(`IVF_PQ`) for very large corpora.

**When not to.** Frequently-updated collections — IVF needs **training** on a
representative sample to pick centroids, and heavy inserts degrade the
clustering until you retrain.

---

### - [ ] 4.1.4 Product Quantization ⚪

**What it is.** Split each vector into `m` sub-vectors, run k-means on each
sub-space to build a **codebook** of 256 centroids, and store each sub-vector as
a **single byte** — the id of its nearest centroid.

**Example.** 768-dim float32 = 3,072 bytes. Split into m=96 sub-vectors of 8
dims each, one byte per sub-vector → **96 bytes**. A **32× reduction.**
Distances are then computed from precomputed lookup tables, which is also fast.

**When to use.** Above roughly 10M vectors, where memory is the dominant cost.
Almost always combined: `IVF_PQ` or `HNSW_PQ`.

**When not to.** When recall is critical and you cannot afford a rescoring pass
— PQ is **lossy**, and the reconstruction error directly costs recall. Standard
mitigation: shortlist with PQ, then **rescore** with full-precision vectors.

**Related, simpler:** *scalar* quantization (float32 → int8, ~4× smaller, almost
no loss) and *binary* quantization (1 bit/dim, 32× smaller, needs rescoring).
Scalar is the easy default; PQ is for extreme scale.

---

### - [ ] 4.1.5 DiskANN and disk-based indexes ⚪

**What it is.** Graph indexes designed so the bulk of the data lives on **SSD**,
with a compressed representation in RAM for navigation. Vamana graph plus PQ
in memory.

**Example.** A billion vectors would need terabytes of RAM under HNSW. DiskANN
serves them from NVMe with a few hundred GB of RAM, at single-digit
millisecond latency.

**When to use.** Billion-scale corpora, or when RAM cost dominates your bill.
Available in Milvus, and the basis of several managed offerings.

**When not to.** Anything that fits comfortably in memory — SSD latency is
orders of magnitude worse than RAM, and you would be paying it for nothing.

---

### - [ ] 4.1.6 The recall/latency/memory triangle ⚪

**What it is.** The single most important framing for this whole section: **ANN
search is *approximate*.** You are always choosing two of three.

```
        recall
         /  \
        /    \
   latency — memory
```

**Examples.**

| Want | Give up | How |
|---|---|---|
| High recall + low latency | memory | HNSW with large `M` |
| High recall + low memory | latency | IVF with high `nprobe`, or DiskANN |
| Low latency + low memory | recall | PQ with aggressive compression |

**How to measure it.** Build a **flat index as ground truth** on a sample, run
the same queries against your ANN index, and compute
`recall@k = |ANN results ∩ exact results| / k`. Without this you have no idea
what your index is costing you.

**Interview point:** the answer to "how do you tune your vector index?" is not a
parameter list — it is *"I measure recall against exact search and then trade
against the latency budget."*

---

### - [ ] 4.2.1 Filtered vector search 🟢

**What it is.** Combining a metadata predicate with vector similarity. Three
fundamentally different implementations, and the difference matters enormously.

| Approach | How | Problem |
|---|---|---|
| **Post-filter** | ANN search for top-k, *then* discard non-matching | Catastrophic. Ask for 10, filter to 2. Selective filters return nothing |
| **Pre-filter** | Compute the matching id set first, then search only those | Correct, but a large id set is expensive |
| **Filterable ANN** | Apply the predicate **during graph traversal** | Best of both. What Qdrant does |

**Example — why post-filtering fails.** 1M chunks across 1,000 tenants. Search
top-10 globally, then filter to tenant X: on average **0.01** of the 10 belong to
X. You return nothing, and it looks like the tenant has no data.

**The subtlety with filterable HNSW.** A restrictive filter can disconnect the
proximity graph — the greedy walk gets stuck in a region with no matching
neighbours. Qdrant handles this by estimating filter cardinality and **falling
back to a full scan** of the filtered subset when it is small enough. That
fallback is why a **payload index** (§4.2.2) matters so much: without it, Qdrant
cannot estimate cardinality and picks badly.

**When to use.** Always, for multi-tenancy — but verify your engine
**pre-filters or filters during traversal**, and never rely on post-filtering.

> 🟢 App: `services/vectorstore.py` filters on `owner_id` and `document_id`
> inside the Qdrant query.

---

### - [ ] 4.2.2 Payload indexes 🟢

**What it is.** A secondary index on a metadata field, so the engine can
evaluate a predicate quickly and, crucially, **estimate its selectivity**.

**Example.** With a payload index on `owner_id`, Qdrant knows "this filter
matches ~200 of 1M points" and chooses a filtered scan. Without it, the filter
is a linear predicate check with unknown cardinality, and the planner may take
the graph path and return poor recall — a *quality* regression, not just a slow
query.

**When to use.** Every field you filter on. It is cheap and the failure mode
without it is silent.

**When not to.** High-cardinality fields you never filter on — an index has
memory cost.

> 🟢 App: payload indexes on `owner_id` and `document_id` — the two fields the
> multi-tenant and per-document scoping paths filter by.

---

### - [ ] 4.2.3 Upserts, deletes, tombstones, compaction 🟢

**What it is.** Vector indexes are optimised for reads, so mutation is awkward.
Deletes are typically **tombstones** — the point is flagged, excluded from
results, and only physically removed during **compaction/optimisation**.

**Example.** Delete a document's 40 chunks and the index files do not shrink
immediately; memory and disk stay allocated until an optimiser pass runs.
`upsert` on an existing id is usually delete-then-insert internally, so heavy
overwriting fragments the index over time.

**Practical patterns.**

- Key points by a **deterministic id** (a UUIDv5 of `document_id:chunk_index`)
  so re-ingest overwrites cleanly rather than duplicating.
- **Delete by filter**, not by id list — `FilterSelector(document_id=X)` is one
  call instead of 40.

**When not to.** Do not treat a vector DB as a mutable primary store. Keep
Postgres (or similar) as the source of truth and the vector index as a derived,
rebuildable artifact.

> 🟢 App: deletion uses Qdrant `FilterSelector` on `document_id`; chunk text
> lives in Postgres, so the index is always rebuildable from the source.

---

### - [ ] 4.2.4 Sharding, replication, persistence ⚪

**What it is.** Standard distributed-systems concerns, applied to vectors.

- **Sharding** splits a collection across nodes. Every query must **fan out to
  all shards** and merge results, because a nearest neighbour could be anywhere
  — unlike a keyed database, you cannot route by shard key.
- **Replication** copies shards for availability and read throughput.
- **Persistence** — HNSW is a memory structure, so it is snapshotted to disk and
  reloaded on restart; a large index means slow cold starts.

**Example.** 8 shards means 8 parallel searches merged per query. Tail latency
is now the *slowest* shard, so p99 degrades as you add shards.

**When to shard.** Only when one node cannot hold the index in RAM.

**When not to.** Prematurely — it costs latency and operational complexity for
no gain below a node's capacity.

---

### - [ ] 4.2.5 Namespaces / collections for multi-tenancy 🟢

**What it is.** The alternative to payload filtering: give each tenant their own
collection (Qdrant), namespace (Pinecone), or class (Weaviate).

| | One collection + filter | Collection per tenant |
|---|---|---|
| Isolation | logical | **physical** |
| Hard delete | filter delete | drop the collection |
| Different embedding models | no | **yes** |
| Cost at 10,000 tenants | fine | **overhead per collection kills you** |
| Cross-tenant search | easy | requires fan-out |

**Example.** A B2B product with 50 enterprise customers, each wanting a
contractual guarantee of separation and their own deletion SLA →
collection-per-tenant. A consumer app with 100,000 users → one collection with
a payload filter, or you drown in collection overhead.

**Hybrid pattern:** collection per *large* tenant, one shared collection with
filtering for the long tail. Qdrant explicitly recommends single-collection +
payload filtering as the default, with **multitenancy-optimised payload
indexes**.

> 🟢 App: single collection, filtered on `owner_id`.

---

### - [ ] 4.2.6 Client/server version compatibility 🟢

**What it is.** Vector databases evolve their on-disk formats and wire
protocols. A newer client against an older server can fail in confusing ways.

**Example — measured in this app.** A Qdrant **1.19 client** against a
**1.12.4 server** could not read storage the server had written, and the
container **crash-looped**. It presented as a corrupt volume, not a version
problem.

**Practice.** Pin both, in the same commit. Keep them aligned on major.minor,
and treat an upgrade as: bump both, wipe the volume (or run the documented
migration), re-ingest.

> 🟢 App: `qdrant-client` in `pyproject.toml` and the `qdrant/qdrant` image tag
> in `docker-compose.yml` carry cross-referencing comments saying to bump them
> together.

---

### - [ ] 4.3.1 The landscape 🟢

| Engine | Model | Strengths | Weaknesses |
|---|---|---|---|
| **Qdrant** 🟢 | Rust, self-host or cloud | **filtering-first** design, payload indexes, multi-vector, sparse | smaller ecosystem than Elastic |
| **pgvector** ⚪ | Postgres extension | **one datastore**, real SQL joins, transactions | weaker at scale; HNSW build is heavy |
| **Pinecone** ⚪ | fully managed | zero ops, serverless tiers | closed, cost at scale, less control |
| **Weaviate** ⚪ | Go, self-host or cloud | hybrid search built in, GraphQL, modules | heavier to operate |
| **Milvus** ⚪ | distributed | billion-scale, many index types, DiskANN | complex architecture |
| **Chroma** ⚪ | embedded | trivial to start, great for prototyping | not for production scale |

**Special mention — pgvector.** Extremely attractive when you already run
Postgres: one backup story, one connection pool, and you can `JOIN` vectors
against relational data in a single query. Real limits: index build times, and
memory pressure on your primary database.

> **Worth noting for this app:** it already runs Postgres *and* Qdrant. pgvector
> would have collapsed that to one datastore. Qdrant was chosen for
> filtering-first multi-tenancy and to learn a purpose-built engine — but
> "why not just pgvector?" is a fair interview challenge and deserves that
> honest answer.

---

### - [ ] 4.3.2 FAISS — a library, not a database ⚪

**What it is.** Facebook AI Similarity Search. A **library** of index
implementations (Flat, IVF, HNSW, PQ, and combinations) with no server, no
persistence layer, no metadata filtering and no concurrency control.

**Example.** `faiss.IndexHNSWFlat(768, 32)` gives you an in-process index. You
must handle: saving/loading it, mapping index positions back to your document
ids, filtering, updates, and multi-process access. All of it.

**When to use.** Batch/offline similarity work, research, or embedding a small
index inside a single process. Also the reference implementation for
understanding how these indexes actually work.

**When not to.** As your production vector store, unless you enjoy rebuilding
the missing 80% of a database. Most vector DBs use FAISS-like algorithms *plus*
that 80%.

---

### - [ ] 4.3.3 OpenSearch / Elasticsearch ⚪ *(named in the JD)*

**What it is.** Mature search engines that added vector search alongside
existing **BM25**. OpenSearch is the AWS fork of Elasticsearch, available as a
managed service — **so this is the likely answer to "how would you do RAG on
AWS?"**

**Why it matters for RAG.** It gives you **hybrid search in one engine**: BM25
and kNN in a single query, fused server-side. Everywhere else you run two
systems and fuse yourself.

**Example.**

```json
{ "query": { "hybrid": { "queries": [
    { "match": { "text": "cloud platform growth" } },
    { "knn": { "vector": { "vector": [...], "k": 10 } } }
]}}}
```

**Also relevant:** OpenSearch Serverless vector collections, and it is the
default vector store behind **Bedrock Knowledge Bases** (§17.2).

**When to use.** You already run Elastic/OpenSearch; you need hybrid without
running two datastores; you are on AWS and want managed.

**When not to.** JVM memory tuning and cluster operations are a real cost, and
its vector features are newer and less specialised than a purpose-built engine's.

---

### - [ ] 4.3.4 How to choose ⚪

Ask these in order:

1. **Do you need a vector index at all?** Below ~50k vectors, brute force or
   pgvector is enough (§4.1.1).
2. **Do you already run Postgres?** → pgvector, unless you need heavy filtering
   or scale.
3. **How important is metadata filtering?** Multi-tenant → filtering-first
   engine (Qdrant, or OpenSearch).
4. **Do you need hybrid search?** → OpenSearch or Weaviate give it natively;
   Qdrant supports sparse vectors; otherwise you fuse yourself.
5. **What is your ops appetite?** None → Pinecone or a managed tier.
6. **What scale, honestly?** Most systems have fewer than 1M vectors, and
   billion-scale advice is irrelevant to them.
7. **Cost model** — per-vector, per-node, or per-query?

**The interview trap:** naming your favourite engine. The good answer explains
the *decision criteria* and then names one — because the criteria are what
transfer.

## 5. Retrieval strategies

### - [ ] 5.1.1 Dense retrieval ⚙️ 🟢

**What it is.** Embed the query, find nearest neighbours by vector distance.
"Semantic search."

**Example.** Query *"how do we handle refunds"* matches a passage saying
*"return policy: customers may send items back within 30 days"* — zero shared
keywords, correct match.

**When to use.** Paraphrase, synonymy, natural-language questions, multilingual.
The default first stage everywhere.

**When not to.** Exact identifiers (`INV-2024-0042`), rare proper nouns,
acronyms, negation ("contracts *without* an arbitration clause"), and numbers.
Dense retrieval is bad at all of these and BM25 is good at all of them.

**Implementation reality:** ⚙️ two library calls — `embed_query()` then
`client.search()`. You will not implement ANN search. What you *do* own is
`top_k`, the filter, and the embedding model choice.

> 🟢 App: `retrieval._search_one()`.

---

### - [ ] 5.1.2 Sparse retrieval: BM25 🔧 ⚪

**What it is.** Keyword search, scored statistically. **No model, no embeddings,
no training** — entirely corpus statistics. Two pieces: an **inverted index**
(`term → doc ids`) finds candidates; **BM25** ranks them.

```
score(D,Q) = Σ  IDF(t) ·        f(t,D) · (k₁ + 1)
            t∈Q            ─────────────────────────────────
                           f(t,D) + k₁ · (1 − b + b · |D|/avgdl)
```

Three ideas, each fixing a flaw in naive TF-IDF:

| Part | Does | Default |
|---|---|---|
| **IDF** | rare terms score higher | — |
| **k₁** | term-frequency **saturation** — the 10th mention adds little | 1.2–2.0 |
| **b** | **length normalisation** — long docs don't win for free | 0.75 |

**Worked example.** Corpus of 5 chunks, `avgdl = 20`, query **"Kafka
migration"**.

| Doc | Len | `kafka` | `migration` | Score |
|---|---|---|---|---|
| **D1** "…to kafka the kafka migration took six weeks" | 13 | 2 | 1 | **1.671** |
| **D5** *long rambling answer* | 52 | 2 | 1 | **1.004** |
| D3 "our migration from the monolith…" | 12 | 0 | 1 | 0.344 |
| D2 "the migration of our database…" | 13 | 0 | 1 | 0.336 |
| D4 "kubernetes … terraform …" | 10 | 0 | 0 | 0.000 |

IDF first: `kafka` in 2 of 5 docs → **0.876**. `migration` in 4 of 5 → **0.288**.
So matching `kafka` is worth **3×** matching `migration`, derived purely from the
corpus.

D1's length factor: `1.2 × (0.25 + 0.75 × 13/20) = 0.885`, so `kafka`
contributes `2 × 2.2 / 2.885 × 0.876 = 1.335` and `migration`
`2.2 / 1.885 × 0.288 = 0.336`.

**Three things the numbers demonstrate:**

1. **Rarity dominates** — `kafka` was 80% of D1's score.
2. **Length normalisation is severe** — D1 and D5 have *identical term counts*;
   D1 wins by 66% for being concise.
3. **Saturation caps repetition** — 10 mentions instead of 2 raises the score
   only 1.33×, not 5×. Keyword stuffing fails.

**Note:** BM25 runs on **analyzed** tokens — lowercased, punctuation-stripped,
**stemmed** — so `migrated`/`migration`/`migrating` all collapse. It also needs
no stopword list: `the` appears in 4 of 5 docs, so its IDF is near zero and it
is statistically ignored.

**When to use.** Always, as the other half of hybrid. Essential for
identifiers, names, jargon, error codes, exact phrases.

**When not to.** Alone, for natural-language questions. `"message queue
rollout"` scores **0.000** on every document above — no term overlap, total
failure, where dense finds D1 instantly.

**Implementation reality:** 🔧 you will genuinely wire this up.

| Option | Reality |
|---|---|
| **OpenSearch / Elasticsearch** | real BM25, the default. Hybrid in one query |
| **Qdrant sparse vectors** | store BM25/SPLADE weights beside dense |
| `rank_bm25` (Python) | in-memory, fine for small corpora |
| **Postgres FTS** (`ts_rank`) | ⚠️ **not BM25** — different function. Still good lexical search, but don't call it BM25 |
| **ParadeDB `pg_search`** | real BM25 inside Postgres |

📖 The formula itself is know-it. ⚙️ `k₁`/`b` are tunable and you will almost
never change them.

> ⚪ **App: the biggest single retrieval gap.** Chunk text already sits in
> Postgres, so `to_tsvector` + RRF is a small change — and RRF was *designed*
> for exactly this fusion.

---

### - [ ] 5.1.3 Learned sparse: SPLADE ⚙️ ⚪

**What it is.** A neural model outputs a **sparse** vector over the vocabulary —
BM25's shape, learned weights, plus **term expansion**.

**Example.** A document containing only "revenue" gets non-zero weight on
"sales", "income", "turnover". BM25 on query `"income"` misses it entirely;
SPLADE matches, while still behaving like exact term matching.

**When to use.** You want one retriever covering keyword *and* light semantics,
and your engine supports sparse vectors (Qdrant, OpenSearch, Vespa, Pinecone).

**When not to.** BM25 already suffices — SPLADE needs a model at index time,
produces larger postings lists, and slows indexing.

**Implementation reality:** ⚙️ you load a model and store its output as a
sparse vector. You do not train it. 📖 the architecture is interview knowledge.

---

### - [ ] 5.1.4 Hybrid retrieval 🔧 ⚪

**What it is.** Run dense and sparse, fuse the rankings. **The single highest-value
upgrade to a dense-only pipeline.**

**Why it works — they fail on opposite inputs:**

| Query | BM25 | Dense |
|---|---|---|
| `INV-2024-0042` | ✅ | ❌ near-identical IDs collide |
| a person's name | ✅ | ❌ names embed weakly |
| `"Kafka"` | ✅ | ⚠️ drifts to "message queue" |
| "how do we handle refunds" | ⚠️ needs literal words | ✅ matches "return policy" |
| negation, rare jargon | ✅ | ❌ |

**When to use.** Almost always in production. Anthropic's contextual-retrieval
numbers show it clearly: contextual embeddings alone cut failures 35%; **adding
contextual BM25 took it to 49%.**

**When not to.** Prototypes; or corpora that are pure natural-language prose
with no identifiers, names or codes — rare in practice.

**Implementation reality:** 🔧 real work. Two indexes to keep in sync (or one
engine that does both), plus a fusion step. The sync is the part that bites —
a document deleted from one index and not the other silently corrupts results.

---

### - [ ] 5.1.5 Multi-vector / late interaction (ColBERT) 📖 ⚪

**What it is.** One vector **per token** — a *matrix* per document. Scoring is
**MaxSim**: for each query token, take its best match among document tokens, then
sum. **This is almost certainly what "matrix retrieval" means.**

**Example.** Query "Cloud Platform growth": "Cloud" matches the token "Cloud"
almost exactly, "growth" matches "grew". Single-vector retrieval averages
everything into one point and can miss that both terms are present in the right
places.

**When to use.** As a **reranker** over a shortlist, where the quality gain
justifies the cost. **ColPali** is the notable variant — it embeds *page images*
directly, excellent for visually complex PDFs where text extraction fails.

**When not to.** First-stage retrieval at scale: storage is 10–100× a single
vector, and native index support is limited (Qdrant, Vespa).

**Implementation reality:** 📖 mostly know-it. If you do use it, it is a model
call plus an engine that supports multi-vectors — you would not implement MaxSim
yourself.

---

### - [ ] 5.1.6 Graph-based retrieval (GraphRAG) 📖 ⚪

**What it is.** Extract entities and relationships into a knowledge graph, then
traverse it. Microsoft's GraphRAG additionally clusters the graph into
communities and pre-generates summaries per community.

Two query modes, and the distinction is the interview point:

- **Local search** — start at the entities named in the question, walk their
  neighbourhood. Good for "what did X do?"
- **Global search** — map-reduce over community summaries. Good for questions
  vector search **structurally cannot** answer: *"what are the main themes across
  all 500 interviews?"* — no single chunk contains that answer.

**Example.** *"How are Alice and the payments migration connected?"* Vector
search retrieves chunks about Alice and chunks about payments, and hopes one
mentions both. A graph traverses `Alice —[led]→ payments-migration` directly.

**When to use.** Relationship questions, multi-hop over entities,
corpus-wide summarisation, and domains that are genuinely graph-shaped
(org charts, supply chains, citations, case law).

**When not to.** Most RAG. Building the graph costs an LLM pass over the entire
corpus (expensive), extraction is brittle, and it is hard to keep current as
documents change.

**Implementation reality:** 📖 know the concept and when it wins. 🔧 only if
the domain demands it — then it is a substantial project, not a feature.

---

### - [ ] 5.1.7 Agentic retrieval 🔧 🟢

**What it is.** The LLM decides *what* to search, judges what came back, and
searches again. Retrieval becomes a loop rather than a step.

**Example — measured in this app.** At `top_k=1`, a two-part question is
**unanswerable** for baseline RAG: one chunk, two facts needed. The agent
decomposes into two sub-questions, retrieves once per sub-question, accumulates
evidence, and answers correctly. **This is the cleanest demonstration that
agentic retrieval is not "RAG with extra steps".**

**When to use.** Multi-hop questions, ambiguous or under-specified queries,
anything where a single pass cannot know what it is missing.

**When not to.** Simple lookups — it costs ~3 model calls where baseline costs
1, and latency multiplies. Keep a non-agentic path for the easy case.

**Implementation reality:** 🔧 this is the code you write, and the reason
LangGraph exists. Needs hard guardrails: iteration caps, token budgets, loop
detection.

> 🟢 App: `agent/graph.py` — `plan → retrieve → draft → critique → retry`, with
> `agent_max_iterations = 2` because unbounded self-critique is the fastest way
> to burn a daily quota.

---

### - [ ] 5.1.8 Metadata / self-query retrieval 🔧 ⚪

**What it is.** An LLM converts natural language into a **structured filter**
plus a semantic query.

**Example.** *"What did the CFO say about margins in Q3 2024 interviews?"* →

```json
{ "query": "margins",
  "filter": { "speaker_role": "CFO", "quarter": "Q3", "year": 2024 } }
```

The filter narrows the search space before vector search runs, which improves
both precision and speed.

**When to use.** Rich, reliable metadata plus users who naturally express
constraints — dates, authors, categories, document types.

**When not to.** Sparse or inconsistent metadata. And be careful: a hallucinated
filter value returns **zero results**, which looks like missing data rather than
a bad filter. Always validate generated filters against the actual allowed
values.

**Implementation reality:** 🔧 a structured-output call plus filter validation.
The validation is the part people skip and then debug for a day.

---

### - [ ] 5.1.9 Recursive and iterative retrieval ⚙️ ⚪

**What it is.** Use what you retrieved to decide what to retrieve next.
*Iterative* = repeat retrieval with a refined query. *Recursive* = follow
references from one chunk to another.

**Example.** A chunk says *"as described in the Kafka migration section"*. A
recursive retriever follows that pointer and fetches it. Or: retrieve, notice
the answer mentions an entity you know nothing about, retrieve for that entity.

**When to use.** Documents with heavy internal cross-referencing — legal,
technical manuals, standards.

**When not to.** Flat corpora with no references; the extra hops cost latency
for nothing.

**Implementation reality:** ⚙️/🔧 it is the agentic loop (§5.1.7) with a
different termination condition. Same machinery.

---

### - [ ] 5.2.1 Reciprocal Rank Fusion (RRF) 🔧 🟢

**What it is.** Merge several ranked lists using **rank position only** — never
scores.

```
score(chunk) = Σ over lists  1 / (k + rank)          k = 60
```

**Worked example.** Two retrievers, `k=60`:

| Chunk | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | 5 | 1/61 + 1/65 = **0.0318** |
| B | 3 | 1 | 1/63 + 1/61 = **0.0323** ⬅ **wins** |
| C | 2 | — | 1/62 = 0.0161 |

B wins despite being 3rd in dense, because **both** retrievers liked it. That
agreement signal is the whole point.

**Why rank and not score — measured in this app:** an irrelevant match scored
**0.570** while a correct match for a different query scored **0.615**. Cosine
is not comparable across queries, and a BM25 score is on a different scale
again. **Rank is the only currency the lists share** — which is also why RRF
needs no tuning or normalisation.

`k = 60` is from the original paper; larger `k` flattens the advantage of rank 1.

**When to use.** Any time you have multiple rankings: hybrid, multi-query,
multiple retrievers. Default choice.

**When not to.** When you have genuinely calibrated, comparable scores (e.g. a
single cross-encoder) — then use the scores, since rank throws away magnitude
information.

**Implementation reality:** 🔧 about 20 lines, pure function, trivially
testable. Write it yourself.

> 🟢 App: `retrieval.reciprocal_rank_fusion()`.

---

### - [ ] 5.2.2 Weighted score fusion 📖 ⚪

**What it is.** Normalise each retriever's scores to a common range, then
combine with weights: `final = α·dense + (1−α)·sparse`.

**Example.** Min-max normalise both to [0,1], then `0.7·dense + 0.3·sparse`.
This is what OpenSearch's hybrid query does by default (normalization processor
+ arithmetic/harmonic/geometric mean).

**When to use.** You have a validation set to tune `α` on, and you want to
express a deliberate preference for one retriever.

**When not to.** Without evaluation data. Normalisation is fragile: min-max
depends on the *specific result set*, so a single outlier score reshapes
everything, and the same document scores differently depending on what else was
retrieved.

**Implementation reality:** 📖 know it exists and why RRF is usually preferred.
⚙️ if using OpenSearch, you configure it rather than write it.

---

### - [ ] 5.2.3 RAG-Fusion 🔧 🟢

**What it is.** A named pattern: **multi-query expansion + RRF**. Generate N
query variations, retrieve for each, fuse with RRF.

**Example.** *"Kafka migration risks"* becomes three queries about cutover risk,
rollback strategy, and dual-write consistency. Each retrieves its own top-k;
fusion surfaces chunks that several agreed on.

**When to use.** Short or ambiguous queries, where one phrasing under-specifies
the information need.

**When not to.** Long, already-specific queries — the variations add cost
without adding coverage. And it costs one LLM call plus N embedding calls per
question.

**Implementation reality:** 🔧 straightforward once you have RRF: one
structured-output call plus `asyncio.gather` over the variations.

> 🟢 App: `retrieval.retrieve()` with `multi_query=True`.

---

### - [ ] 5.3.1 Multi-query expansion 🔧 🟢

**What it is.** An LLM rewrites the question into several phrasings that
approach the same information need from different angles.

**Example.** *"gross margin?"* → *"What was the gross margin percentage for
fiscal 2024?"* / *"How did gross margin change year over year?"* / *"What drove
the improvement in gross margin?"*

**When to use.** Terse queries; vocabulary mismatch between users and documents.

**When not to.** When latency matters — it serialises an LLM call before
retrieval can even start.

**Implementation reality:** 🔧 a prompt plus a schema. Two measured traps:
temperature 0.7 sent Gemma into repetition loops that truncated the JSON (0.35
is stable), and the *prompt* matters more than the parameters — see the
scope-aware version below.

> 🟢 App: `retrieval.expand_query()`. Its prompt **classifies scope first**:
> `specific` → paraphrase; `broad` → do *not* paraphrase, write one query per
> **topic**. Because *"summarize the document" describes an ACTION and shares no
> meaning with the document's contents*, so searching for it returns arbitrary
> passages. For `broad`, the original query is also **dropped from the fused
> set** — RRF gives every list an equal vote, so keeping a query known to carry
> no signal actively poisons the results.

---

### - [ ] 5.3.2 HyDE 🔧 ⚪

**What it is.** Hypothetical Document Embeddings. Have the LLM **write a fake
answer**, then embed *that* and search with it. You are matching
answer-to-answer instead of question-to-answer.

**Example.** Query *"how did they mitigate cutover risk?"* → the LLM invents:
*"The team ran the old and new systems in parallel for six weeks, dual-writing
to both and reconciling nightly before cutting traffic over."* That fake
paragraph is **linguistically much closer** to the real passage than the
question was.

**When to use.** Sharp vocabulary mismatch between how users ask and how
documents are written. Strong for zero-shot domains and technical corpora.

**When not to.** When the model knows nothing about your domain — it will
hallucinate a fake answer pointing in the wrong direction, and retrieval follows
it confidently. Also adds a full generation call to the critical path.

**Implementation reality:** 🔧 trivially easy to implement (one generation, then
embed its output instead of the query), which is why it is worth trying — but
**measure it**, because it can hurt as easily as help.

---

### - [ ] 5.3.3 Query decomposition 🔧 🟢

**What it is.** Split a compound question into self-contained sub-questions and
retrieve for each.

**Example.** *"How did revenue and headcount change, and what were the main
risks?"* → three lookups. A single retrieval with `top_k=5` would likely return
five chunks about revenue and nothing about risks.

**When to use.** Multi-part questions; comparisons; anything with "and".
**Measured in this app: it is what makes `top_k=1` viable.**

**When not to.** Single-fact questions — decomposition returns the question
unchanged and you paid a model call for nothing.

**Implementation reality:** 🔧 a structured-output call, plus a **dedupe** step:
the planner emits the same lookup twice surprisingly often (especially when
resolving a follow-up, where the resolved and original phrasings collide), and
each duplicate wastes an embedding call and a retrieval slot.

> 🟢 App: `agent/nodes.py:plan`. Note `PLAN_SCHEMA` deliberately has **no
> `reasoning` field** — it once produced four correct sub-questions then
> degenerated *inside* `reasoning`, truncating the response and invalidating the
> whole object. **Ask only for what you use.**

---

### - [ ] 5.3.4 Step-back prompting 📖 ⚪

**What it is.** Ask a more *general* question first, retrieve for that, then use
the broader context to answer the specific question.

**Example.** *"Did the Q3 margin improvement come from pricing or mix?"* →
step back to *"What drove margin changes in 2024?"* → retrieve the general
discussion → then answer the specific question from it.

**When to use.** Narrow questions whose answer only makes sense inside a broader
principle — reasoning-heavy domains, physics/policy/legal style questions.

**When not to.** Factual lookups, where the general question retrieves vaguer
context than the specific one would.

**Implementation reality:** 📖 mostly a prompting idea; ⚙️ trivial to add as
one more query variation if you already have multi-query.

---

### - [ ] 5.3.5 Query rewriting and reference resolution 🔧 🟢

**What it is.** Rewrite a conversational query into a standalone one. **Retrieval
is stateless** — it has no memory of the conversation, so an unresolved pronoun
never resolves.

**Example.** *"and the prior year?"* → *"What was operating income in 2023?"*
Without this, the query embeds as a vague fragment about time and retrieves
noise.

**When to use.** Every multi-turn chat interface. Non-negotiable.

**When not to.** Single-shot search boxes with no history.

**Implementation reality:** 🔧 you write this, and the *placement* is the design
decision: it must happen **before** retrieval, and it needs the conversation
history passed in. This is also the same problem as transcript anaphora
(*"elaborate on that"*), just at the query end instead of the document end.

> 🟢 App: folded into the `plan` node's prompt — *"Each must be SELF-CONTAINED:
> no 'it', 'that', 'the prior year'. If conversation history is provided,
> resolve every reference against it."*

---

### - [ ] 5.3.6 Query routing / intent classification 🔧 🔵

**What it is.** Classify the query, then send it down a different path. Not all
questions want the same retrieval.

**Example.** Three intents needing genuinely different code:

| Intent | Correct handling |
|---|---|
| `lookup` — "what was gross margin?" | normal similarity search |
| `whole_document` — "summarize this" | **fetch all chunks in order. No similarity search at all** |
| `metadata` — "how many documents do I have?" | query Postgres, not the vector store |

**Why it matters, and this is the key insight:** *"summarize the document"* has
**no semantic overlap with document content**. Similarity search returns the
five chunks nearest a meaningless point, then confidently summarises an
arbitrary fifth of the document. **No prompt tweak fixes that — it needs a
different code path.**

**When to use.** Distinct query types with distinct correct handling.

**When not to.** Homogeneous workloads. And a misrouted query is worse than an
unrouted one, so it needs a confident classifier and a sensible default.

**Implementation reality:** 🔧 real work. The `whole_document` path needs its
own token-budget guard — a 6-chunk fixture fits in context, a 200-page PDF does
not, so it needs map-reduce summarisation or an explicit "too large" refusal.

> 🔵 App: partially done. `expand_query` classifies `specific` vs `broad` and
> changes the *queries*, but there is still no separate whole-document code
> path. Full routing is designed, not built.

---

### - [ ] 5.3.7 MMR — Maximal Marginal Relevance ⚙️ ⚪

**What it is.** Greedily select results balancing relevance against
**dissimilarity to what you have already picked**.

```
MMR = argmax [ λ · sim(q, d) − (1−λ) · max sim(d, d_selected) ]
```

`λ=1` is pure relevance; `λ=0` is pure diversity. Typical 0.5–0.7.

**Example.** Top-5 for *"revenue"* returns five near-identical chunks all
restating the same figure — you have spent five slots on one fact. MMR keeps the
best one and fills the rest with *different* aspects: segments, growth, drivers.

**When to use.** Visibly redundant results; **summarisation and broad queries,
where coverage matters more than precision**; overlapping chunks.

**When not to.** Precise factual lookup, where you *want* the several most
relevant chunks even if they agree — corroboration is useful there.

**Implementation reality:** ⚙️ built into LangChain and most vector stores as
`search_type="mmr"`. One parameter. 📖 the formula is know-it.

**Note:** this is the other candidate for what "matrix retrieval" might mean, if
the interviewer does not mean ColBERT. Worth clarifying which they intend.

---

### - [ ] 5.4.1 Self-RAG 📖 ⚪

**What it is.** A model fine-tuned to emit **reflection tokens** that control
retrieval: *do I need to retrieve?*, *is this passage relevant?*, *is my answer
supported by it?*

**Example.** For *"what is 2+2?"* it emits `[No Retrieval]` and answers
directly. For a corpus question it retrieves, then per passage emits
`[Relevant]`/`[Irrelevant]`, and for its own answer `[Fully supported]` /
`[Partially supported]` / `[No support]`.

**When to use.** When you can fine-tune, and mixed workloads where some queries
genuinely need no retrieval.

**When not to.** With an API model — it needs a specially trained model. In
practice you approximate the behaviour with prompting and a critic node.

**Implementation reality:** 📖 know the idea. 🔧 the *approximation* (a critic
that grades support) is buildable and is what most teams actually do.

---

### - [ ] 5.4.2 Corrective RAG (CRAG) 🔧 ⚪

**What it is.** **Grade the retrieved documents** before generating. If they are
irrelevant, do something else — rewrite the query, or fall back to web search.

**Example.** Retrieve for *"2025 revenue guidance"*, a grader marks all five
chunks irrelevant (the corpus only covers 2024), so instead of hallucinating
from bad context the system either says so or escalates to an external source.

**When to use.** Wherever a wrong answer is worse than "I could not find it".
The grader is cheap insurance against confident nonsense.

**When not to.** Latency-critical paths — grading adds a call before generation.

**Implementation reality:** 🔧 very practical, and the highest-value
self-correction pattern to actually build. A grader node with a boolean schema
plus a fallback branch.

> 🔵 App-adjacent: the `critique` node grades *sufficiency after* drafting, not
> relevance *before* it. Grading before generation is the CRAG variant, and
> would catch the `RECITATION` failure — where the model was asked something the
> corpus does not cover, fell back on memorised text, and got cut off by a
> Google filter.

---

### - [ ] 5.4.3 Adaptive RAG 🔧 ⚪

**What it is.** Route by **query complexity**: simple → answer directly or
single retrieval; complex → multi-step agentic loop.

**Example.** *"What is our company name?"* needs no retrieval. *"What was
margin?"* needs one. *"Compare margins across all three segments and explain the
divergence"* needs decomposition and several passes.

**When to use.** Heterogeneous traffic where always-agentic is wasteful and
never-agentic is inadequate. It is the cost-control answer.

**When not to.** Uniform workloads — the classifier is pure overhead.

**Implementation reality:** 🔧 a classifier plus a conditional edge. In
LangGraph this is a few lines; the hard part is calibrating the classifier so it
does not route hard questions down the cheap path.

---

### - [ ] 5.4.4 Critic and verifier loops 🔧 🟢

**What it is.** After drafting, a second pass asks *"is this actually answered
and supported?"* If not, retrieve more and redraft.

**Example.** The critic notices the answer covers revenue but the question also
asked about risks, so it emits a new pending query for risks and the graph loops
back to retrieval.

**When to use.** Multi-part questions; high-stakes answers; anywhere
completeness matters more than latency.

**When not to.** Unbounded. Each cycle costs ~2 model calls, and unbounded
self-critique is the single easiest way to exhaust a quota by accident.

**Implementation reality:** 🔧 this is what the conditional-edge cycle in
LangGraph is *for*. Three guardrails are mandatory: an **iteration cap**, a
check that there is actually something new to search for, and tracking of
already-tried queries so the critic cannot demand the same failed search
forever.

> 🟢 App: `graph.should_continue()` — ends on `sufficient`, on hitting
> `agent_max_iterations`, or when `pending_queries` is empty. **A cap, not a
> judgement of quality.**

## 6. Reranking

> **The one-line summary:** retrieval is optimised for *recall* at scale;
> reranking is optimised for *precision* on a shortlist. Different models,
> different stages, and **the second cannot fix a failure of the first.**

---

### - [ ] 6.1 Cross-encoder reranking ⚙️ ⚪

**What it is.** A model that takes `(query, document)` **together as one
sequence** and outputs a single relevance score. Because it attends *across*
both, it can judge relevance far better than comparing two independently-made
vectors.

```
Bi-encoder  (retrieval):  embed(query) · embed(doc)      → cosine
Cross-encoder (rerank):   model("query [SEP] doc")       → score
```

**Example.** Query *"contracts without an arbitration clause"*. A bi-encoder
embeds the query into one point; documents *containing* arbitration clauses sit
very close to it, because the words overlap heavily — **dense retrieval is
notoriously bad at negation**. A cross-encoder reads both together and can
represent "this document contains the thing you said you did *not* want".

**When to use.** After retrieval, over the top 50–100. It is the standard
second-stage everywhere quality matters.

**When not to.**

- **Never as first-stage retrieval.** 1M documents = 1M forward passes *per
  query*. Infeasible by orders of magnitude.
- **When recall is your problem.** Reranking cannot surface a document retrieval
  never returned. If the right chunk is not in your top-100, a reranker is
  useless — fix chunking or add hybrid first.

**Implementation reality:** ⚙️ one API call (Cohere) or one local model call
(BGE). You do not train or implement it. What you own is **how many candidates
you feed it** and the latency budget.

---

### - [ ] 6.2 The retrieve-N-rerank-K pattern 🔧 ⚪

**What it is.** The standard two-stage architecture: retrieve **widely and
cheaply**, then rerank **narrowly and expensively**.

```
1M chunks ──dense+BM25──> top 100 ──cross-encoder──> top 10 ──> LLM
             (cheap, high recall)     (expensive, high precision)
```

**Example with numbers.** Anthropic's contextual-retrieval results show the
final step clearly: contextual embeddings + BM25 cut retrieval failures 49%;
**adding reranking of 150 candidates down to 20 took it to 67%.** Reranking was
worth roughly as much as everything before it.

**Choosing N.** Bigger N = better recall into the reranker = better final
results, but linear cost. Typical: retrieve 50–150, rerank to 5–20. **Tune N by
measuring recall@N** — there is no point feeding 100 if recall@50 is already
0.98.

**When to use.** Any production RAG where quality matters and you have ~100ms
of latency budget.

**When not to.** Hard real-time paths, or when `top_k` is already tiny and
retrieval is already precise.

**Implementation reality:** 🔧 you wire the two stages and pick N and K. This is
the single most reliable quality upgrade after hybrid search, and it is mostly
plumbing rather than cleverness.

> ⚪ **App gap.** Retrieval returns `top_k=5` straight to the model. The natural
> change is retrieve 30 → rerank to 5, which needs no re-indexing.

---

### - [ ] 6.3 Reranker model choices ⚙️ ⚪

| Model | Type | Notes |
|---|---|---|
| **Cohere Rerank 3** | API | strongest turnkey option; multilingual; no infra |
| **BGE-reranker** (`base`/`large`/`v2-m3`) | local, open | excellent quality/size; the default self-hosted pick |
| **MiniLM cross-encoders** (`ms-marco-MiniLM-L-6-v2`) | local, tiny | very fast on CPU, good enough for many cases |
| **MonoT5 / RankT5** | local, seq2seq | strong, slower |
| **Jina Reranker** | API + open | competitive |
| **ColBERT** | multi-vector | rerank via MaxSim, cheaper than a cross-encoder |

**How to choose.** No GPU and latency matters → MiniLM on ONNX. Quality first,
no ops → Cohere. Self-hosted with a GPU → BGE-reranker-large.

**When not to bother comparing.** Below ~10k documents with good chunking,
reranking often changes little — measure before investing.

**Implementation reality:** ⚙️ swap a model name. The interesting decision is
API vs local, which is a latency/cost/privacy trade, not a quality one.

---

### - [ ] 6.4 LLM-as-reranker ⚙️ ⚪

**What it is.** Ask a general LLM to rank the candidates instead of using a
purpose-built reranker.

Three prompting shapes:

| Shape | How | Cost |
|---|---|---|
| **Pointwise** | score each doc 0–10 independently | N calls |
| **Pairwise** | "which of A or B is more relevant?" | O(N log N) comparisons |
| **Listwise** | "here are 20 docs, return them in relevance order" | **1 call** |

Listwise is what people actually use — one structured-output call returning an
ordered list of ids.

**Example.** Pass 20 numbered chunks and ask for a JSON array of the 5 most
relevant ids, in order.

**When to use.** You already pay for an LLM and do not want another model in the
stack; or you need *explainable* ranking, since you can ask for a reason.

**When not to.** Latency- or cost-sensitive paths — a cross-encoder is far
cheaper and usually better at this specific job. Also watch **position bias**:
LLMs favour items at the start and end of a list, so shuffle candidates and
consider multiple passes.

**Implementation reality:** ⚙️ a prompt plus a schema. Easy to build, easy to
get subtly wrong via position bias.

---

### - [ ] 6.5 ColBERT as a reranker ⚙️ ⚪

**What it is.** Use late-interaction MaxSim scoring (§5.1.5) over the shortlist
instead of a full cross-encoder pass.

**Why it is interesting.** Document token vectors can be **precomputed**, so
reranking is a cheap matrix operation rather than N transformer passes — much
faster than a cross-encoder, more accurate than a bi-encoder. It sits neatly
between the two.

**When to use.** You need cross-encoder-ish quality at bi-encoder-ish latency
and can afford the storage (10–100× per document).

**When not to.** Storage-constrained, or your engine has no multi-vector
support.

**Implementation reality:** ⚙️ requires an engine that stores multi-vectors
(Qdrant, Vespa) or the `RAGatouille` library. 📖 the MaxSim maths is know-it.

---

### - [ ] 6.6 Latency and cost budgeting 🔧 ⚪

**What it is.** Reranking is the stage where you *choose* how much latency to
spend, so it needs an explicit budget.

**Rough example** for a 100-candidate rerank:

| Option | Latency |
|---|---|
| MiniLM-L6, CPU, batched | ~50–150 ms |
| BGE-reranker-large, GPU | ~30–80 ms |
| Cohere Rerank API | ~100–300 ms (network dominates) |
| Listwise LLM call | ~1–3 s |

**Levers when it is too slow.** Reduce N; use a smaller reranker; batch;
truncate documents before scoring (relevance is usually decidable from the first
few hundred tokens); cache `(query, doc)` scores; or rerank only when the
first-stage scores are ambiguous.

**When not to spend it.** If your total budget is 500ms and generation takes
400ms, reranking does not fit — improve chunking and add hybrid instead, both of
which are free at query time.

**Implementation reality:** 🔧 the batching and truncation are code you write,
and they are where the real wins are.

---

### - [ ] 6.7 Rank fusion ≠ reranking 📖 🟢

**Be precise about this — it is a common interview trip-up.**

| | **Rank fusion (RRF)** | **Reranking** |
|---|---|---|
| What it combines | several rankings from the **same cheap scorer** | nothing — it **rescores** |
| Model involved | **none** | a **different, better** model |
| New information | no — only agreement between lists | **yes** — joint query/doc attention |
| Cost | negligible | a model pass per candidate |
| Fixes | which of the candidates rise | how well candidates are *judged* |

**Example.** RRF can only reorder documents that some retriever already
returned, using rank agreement as evidence. It has no capacity to notice that a
document is about the *opposite* of what you asked. A cross-encoder does,
because it reads the query and document together.

**The honest answer to "do you rerank?"** — for this app: *"I fuse multiple
rankings with RRF, which is not the same thing. A cross-encoder reranker is the
obvious next improvement and I have not added one."* Claiming RRF as reranking
is the kind of imprecision an interviewer will probe.

> 🟢 App: RRF is built (`retrieval.reciprocal_rank_fusion`); reranking is not.

## 7. RAG evaluation

> **The uncomfortable truth about production practice.** Most shipped RAG
> systems have **no formal evaluation at all**. The realistic ladder, and where
> teams actually stop:
>
> | Rung | What it is | Who does it |
> |---|---|---|
> | 0 | "Vibe check" — try some questions, eyeball answers | **almost everyone starts here, and many stay** |
> | 1 | **A written golden set of 20–50 questions**, re-run by hand before releases | ★★★ the realistic minimum |
> | 2 | Automated in CI, LLM-as-judge on faithfulness | ★★ teams that have been burned |
> | 3 | Labelled relevance judgments, Recall@k / NDCG tracked over time | ★ search-mature teams |
> | 4 | Online A/B testing with user feedback | ★ high-traffic products |
>
> **Rung 1 is the highest-value thing in this entire section.** Twenty
> questions in a file, with the answers you expect, beats every metric below if
> the alternative is nothing. The interview-winning answer is not naming
> metrics — it is *"here is the golden set, here is what regressed when I
> changed chunking, here is how I knew."*
>
> **Where interviews probe:** they will ask which metrics you tracked. Be ready
> to say what you measured, what you *didn't*, and why.

---

### - [ ] 7.1.1 Recall@k — the one that matters most ★★★ 🔧 ⚪

**What it is.** Of all the chunks that *should* have been retrieved, what
fraction appeared in the top-k.

```
Recall@k = (relevant chunks in top-k) / (all relevant chunks)
```

**Example.** A question needs 2 chunks (revenue figure + the driver). At
`top_k=5` retrieval returns 1 of them. **Recall@5 = 0.5** — and the answer will
be half right no matter how good your prompt is.

**Why it dominates.** RAG is a **two-stage pipeline, and stage two cannot repair
stage one.** A chunk that was not retrieved cannot be cited, cannot be reranked,
cannot be reasoned over. Every other quality problem is fixable downstream;
missing recall is not.

**When to use.** Whenever you change chunking, embeddings, `top_k`, or add
hybrid/reranking. It is *the* regression metric for retrieval.

**When not to.** You need labels — question → which chunk(s) answer it. That is
the cost, and why teams skip it.

**Production practice:** ★★★ the metric teams *should* track and the one they
most often approximate. The cheap version: for each golden question, note the
`chunk_id` that answers it, then assert it appears in the results. **That is a
unit test, not a research project.**

**Implementation reality:** 🔧 ~20 lines once you have a golden set. The work is
the labelling, not the maths.

> ⚪ **App:** not measured. This is the single biggest evaluation gap, and the
> reason every retrieval claim in this repo is *directional* rather than
> quantified.

---

### - [ ] 7.1.2 Precision@k ★★ 📖 ⚪

**What it is.** Of the k chunks returned, what fraction were actually relevant.

```
Precision@5 = 2 relevant / 5 returned = 0.4
```

**Example.** `top_k=5` returns 2 useful chunks and 3 unrelated ones.
Precision@5 = 0.4. The answer may still be correct — the model can ignore
noise — but you paid tokens for 3 useless chunks and increased the chance of
distraction.

**When to use.** When context cost or "lost in the middle" is your problem, not
correctness. Also the metric that improves when you add a reranker.

**When not to.** As your primary metric. **In RAG, precision matters less than
recall** — an extra irrelevant chunk is cheap; a missing relevant chunk is
fatal. Optimising precision at the cost of recall is the classic beginner
mistake.

**Production practice:** ★★ tracked alongside recall, rarely alone.

---

### - [ ] 7.1.3 MRR — Mean Reciprocal Rank ★ 📖 ⚪

**What it is.** Average of `1 / rank of the first relevant result`.

**Example.** Three queries: first relevant hit at position 1, 3, and 2.
`MRR = (1/1 + 1/3 + 1/2) / 3 = 0.61`

**When to use.** When only the **first** good result matters — "I'm feeling
lucky" search, autocomplete, a single-answer lookup.

**When not to.** RAG, mostly. You send 5 chunks to the model, so whether the
best one was rank 1 or rank 3 is largely irrelevant — it is in the context
either way.

**Production practice:** ★ more of a *search team* metric than a RAG metric. Know
the definition, expect it to come up, but do not build your regression suite on
it.

---

### - [ ] 7.1.4 NDCG@k ★ 📖 ⚪

**What it is.** Normalised Discounted Cumulative Gain. Handles **graded**
relevance (perfect / good / marginal / irrelevant) rather than binary, and
discounts by position logarithmically.

**Example.** Rating chunks 3/2/1/0, NDCG rewards you for putting the 3 above the
2, not just for retrieving both.

**When to use.** You have graded judgments — which means real annotation
investment. Standard in academic IR and in mature search organisations.

**When not to.** Almost all RAG projects. Producing graded labels is
expensive, and RAG rarely needs the resolution.

**Production practice:** ★ genuinely rare outside search teams. **Know what it
is and why you didn't use it** — "we used binary Recall@k because graded
judgments weren't worth the annotation cost" is a strong, honest answer.

---

### - [ ] 7.1.5 Hit rate and MAP ★★ 📖 ⚪

**Hit rate** — the fraction of queries where **at least one** relevant chunk was
retrieved. Blunt, binary, and genuinely useful as a headline number: *"87% of
questions retrieved something usable."*

**MAP** — Mean Average Precision. Averages precision at every relevant position,
across queries. Rank-aware, binary relevance.

**When to use.** Hit rate as a dashboard number stakeholders understand. MAP
rarely.

**Production practice:** hit rate ★★ (it's the one non-engineers grasp), MAP ★.

---

### - [ ] 7.2.1 Faithfulness / groundedness ★★★ ⚙️ ⚪

**What it is.** Is every claim in the answer actually supported by the retrieved
context? **This is the hallucination metric**, and the most important
generation-side number.

**How it is measured.** An LLM judge breaks the answer into individual claims
and checks each against the context:

```
Answer:  "Gross margin improved to 62.1%, driven by the subscription mix shift,
          and headcount grew to 4,182."
Claims:  1. margin improved to 62.1%      → supported ✅
         2. driven by subscription mix     → supported ✅
         3. headcount grew to 4,182        → NOT in context ❌
Faithfulness = 2/3 = 0.67
```

**When to use.** Always, if you measure anything on the generation side. It
needs **no ground-truth answer** — only the answer and the context — which makes
it far cheaper to adopt than correctness metrics.

**When not to.** Where the answer is legitimately expected to synthesise beyond
the sources.

**Production practice:** ★★★ the most commonly adopted generation metric,
because it's the one that maps directly onto the business risk ("does it make
things up?"). Available out of the box in RAGAS, Langfuse, LangSmith and
DeepEval.

**Implementation reality:** ⚙️ a library call, or a prompt if you want control.
Costs one judge call per answer.

> 🟢 **App has a free proxy for this:** `sources_used` tracks which `[n]` the
> model actually cited, and the UI shows a **"no sources cited"** badge. Zero
> citations is the *shape* a hallucination takes. Not a real faithfulness score,
> but honest instrumentation at zero cost — and worth saying so in interview.

---

### - [ ] 7.2.2 Answer relevancy ★★ ⚙️ ⚪

**What it is.** Does the answer actually address the question asked? Catches
evasive, partial, or off-topic answers that are nonetheless *faithful*.

**How RAGAS does it:** generate N questions *from the answer*, embed them, and
compare to the original question. High similarity = the answer was on topic.

**Example.** Q: *"What drove the margin improvement?"* A: *"Gross margin was
62.1% in 2024."* — perfectly faithful, but it did not answer *why*. Faithfulness
1.0, answer relevancy low.

**When to use.** Alongside faithfulness. The two catch different failures and
you need both: faithful-but-useless and useful-but-invented are distinct bugs.

**Production practice:** ★★ common, usually the second metric teams add.

---

### - [ ] 7.2.3 Context precision ★ ⚙️ ⚪

**What it is.** Of the retrieved chunks, how many were actually needed — and
were the useful ones ranked first?

**Example.** 5 chunks retrieved, 2 used, and they were at positions 4 and 5.
Low context precision: retrieval found the right material but ranked it badly.
**That is the signal that says "add a reranker".**

**When to use.** Diagnosing *ranking* quality specifically, and justifying
reranking work.

**Production practice:** ★ less commonly tracked than faithfulness. Useful
diagnostically rather than as a dashboard metric.

---

### - [ ] 7.2.4 Context recall ★★ ⚙️ ⚪

**What it is.** Did retrieval fetch everything the ground-truth answer needs?
Measured by decomposing the *reference answer* into claims and checking each is
attributable to the retrieved context.

**Example.** Ground truth mentions both the margin figure and its driver.
Retrieved context contains only the figure. **Context recall = 0.5** — the
generator was set up to fail.

**When to use.** When you have reference answers. **This is the LLM-judged
cousin of Recall@k**, and it needs no chunk-level labels — just a good answer —
which makes it much cheaper to produce.

**Production practice:** ★★ the pragmatic substitute for labelled Recall@k, and
the reason RAGAS is popular: you write the expected answer, not the expected
chunk ids.

---

### - [ ] 7.2.5 Answer correctness and semantic similarity ★★ ⚙️ ⚪

**Answer correctness** — does the answer match the reference answer, factually?
Usually a blend of semantic similarity and claim-level F1.

**Semantic similarity** — cosine between answer and reference embeddings. Cheap,
crude, and easily fooled: a fluent wrong answer scores well.

**When to use.** Correctness when you have a curated golden set with reference
answers — it is the closest thing to "is it right?" Similarity only as a smoke
test.

**When not to.** Similarity alone for anything that matters. It cannot tell
"revenue grew 18%" from "revenue fell 18%" reliably.

**Production practice:** correctness ★★ on golden sets; raw similarity ★ and
declining as judges get cheaper.

---

### - [ ] 7.3.1 The tooling ★★ ⚙️ ⚪

| Tool | What it is | Popularity |
|---|---|---|
| **RAGAS** | the standard RAG metrics library. Faithfulness, answer relevancy, context precision/recall | ★★★ **the default answer** |
| **Langfuse** | tracing + attach scores to traces + datasets. Eval where your observability already is | ★★ rising fast |
| **LangSmith** | LangChain's platform — tracing, datasets, evaluators, human review | ★★ standard if you're in LangChain |
| **DeepEval** | pytest-style LLM evals, CI-friendly | ★★ growing |
| **promptfoo** | config-driven prompt/RAG testing, very CI-friendly | ★★ |
| **TruLens** | feedback functions, "RAG triad" framing | ★ |
| **Phoenix (Arize)** | open-source tracing + eval | ★ |

**What RAGAS needs:** `question`, `answer`, `contexts`, and — only for context
recall and correctness — `ground_truth`. Notably **faithfulness and answer
relevancy need no labels at all**, which is why they are usually the first two
metrics anyone adopts.

**Production practice:** the common real-world stack is **Langfuse or LangSmith
for tracing + RAGAS metrics on a small golden set, run in CI.** That combination
is a completely defensible interview answer.

---

### - [ ] 7.3.2 Building a golden dataset ★★★ 🔧 ⚪

**What it is.** The actual foundation. Everything else is arithmetic on top of
it.

**What a row looks like:**

```yaml
- question: "What drove the gross margin improvement in 2024?"
  expected_answer: "A shift in product mix toward higher-margin subscriptions."
  expected_chunk_ids: ["acme-report#financial-summary"]
  tags: [single-hop, financial]
```

**Three ways to build one, in ascending realism:**

1. **Hand-write 20–50 questions** from your own documents. An afternoon.
   ★★★ and by far the highest value per hour spent.
2. **Synthetic generation** — LLM reads each chunk and writes questions it
   answers. Fast, scales, but produces *chunk-shaped* questions that are easier
   than real ones. ★★ Use it to bulk out a hand-written core.
3. **Mine production traffic** — real questions, with thumbs-up/down. The best
   source, and the reason to log queries from day one. ★★

**Coverage matters more than size.** Deliberately include: single-hop,
multi-hop, **unanswerable** (does it correctly refuse?), ambiguous, and
conversational follow-ups with pronouns.

**When not to.** Never skip it. Fifty questions is not a research programme.

**Implementation reality:** 🔧 a YAML file and a test runner. The unanswerable
cases are the ones people forget and the ones that catch hallucination.

> 🟢 **App has the seed of this:** `fixtures/` holds a committed corpus plus
> written regression questions, including the multi-hop case where baseline RAG
> fails at `top_k=1`. It is run **by hand**. Turning it into an automated golden
> set is a small job with disproportionate payoff.

---

### - [ ] 7.3.3 LLM-as-judge ★★★ ⚙️ ⚪

**What it is.** Use a strong LLM to score outputs instead of human labels. The
technique that made RAG evaluation practical at all.

**The biases you must know** — these are the interview questions:

| Bias | What happens | Mitigation |
|---|---|---|
| **Position bias** | favours whichever candidate appears first | swap order, average |
| **Verbosity bias** | prefers longer answers | constrain length; score criteria separately |
| **Self-preference** | a model prefers its own outputs | judge with a *different* model family |
| **Leniency** | reluctant to give low scores | few-shot with examples of bad answers; use a rubric |
| **Poor calibration** | 1–10 scores cluster at 7–8 | use binary or 3-point scales |

**Practical rules.** Prefer **binary or 3-point** judgments over 1–10. Give an
explicit rubric. Ask for the reason *before* the verdict. Validate the judge
against ~50 human labels before trusting it at scale.

**When not to.** When the judge is weaker than the model being judged, or for
domain judgments requiring expertise the judge lacks (clinical, legal).

**Production practice:** ★★★ effectively the default for generation-side eval.
Cost is real though — one judge call per answer per metric, so four metrics on
1,000 questions is 4,000 calls.

---

### - [ ] 7.3.4 Regression suites in CI ★★ 🔧 ⚪

**What it is.** Run the golden set on every change to chunking, prompts,
retrieval or model, and fail the build on regression.

**The problem, and the practical shape.** LLM outputs are non-deterministic, so
exact-match assertions flap. What actually works:

- **Assert on retrieval, not generation.** `expected_chunk_id in results` is
  deterministic and catches most regressions. **This is the trick.**
- **Threshold, don't exact-match:** `faithfulness >= 0.8` across the set.
- **Compare against the previous run**, not an absolute bar — "no metric dropped
  more than 5%".
- **Temperature 0** for anything scored.
- **Keep it small and fast.** 20 questions in CI, 200 nightly.

**When not to.** Do not gate a PR on a flaky LLM metric — you will teach the
team to ignore CI. Gate on the deterministic retrieval assertions; report the
generative ones.

**Production practice:** ★★ common in teams that have shipped a regression.
Rarer than it should be, and a genuine differentiator to have done.

---

### - [ ] 7.3.5 Online evaluation and feedback ★★ 🔧 ⚪

**What it is.** Measuring in production, where the real distribution lives.
Offline eval tells you about the questions *you* thought of.

**What teams actually instrument** — ordered by value per effort:

| Signal | Cost | Value |
|---|---|---|
| **Thumbs up/down** per answer | trivial | ★★★ the single best signal |
| **Log every query + retrieved chunk ids** | trivial | ★★★ becomes your golden set |
| Copy / regenerate / follow-up rate | low | ★★ implicit dissatisfaction |
| "No sources cited" rate | low | ★★ hallucination proxy |
| Latency and cost per turn | low | ★★ |
| A/B test on a retrieval change | high | ★ high-traffic products only |

**The compounding one:** logging queries and retrieved chunk ids from day one.
Six months later that log *is* your golden set, drawn from the real
distribution. Teams that skip it cannot build one retrospectively.

**Production practice:** ★★ thumbs and query logging are near-universal in
serious products. Formal A/B testing of RAG changes is ★ and mostly at scale.

---

### - [ ] 7.3.6 Cheap proxies ★★ 🔧 🟢

**What it is.** Signals that correlate with quality and cost nothing to compute
— what you use *before* you have an eval harness.

| Proxy | What it catches |
|---|---|
| **Zero-citation rate** | hallucination shape — an answer with no sources |
| **Citation coverage** | how many of the retrieved chunks were actually used |
| **Refusal rate** | too high = retrieval failing; too low = over-confidence |
| **Top score distribution** | a sudden drop in max cosine = a query the corpus can't serve |
| **Retrieval latency** | index degradation |
| **Answer length distribution** | truncation, or degeneration |

**When to use.** Immediately, on day one, before any formal eval. These are
dashboard metrics you can ship in an afternoon.

**When not to.** As a substitute for measuring recall. They are *smoke alarms*,
not thermometers — they tell you something is wrong, not how good you are.

**Production practice:** ★★ and underrated. Most teams have some of these and
call it monitoring rather than evaluation.

> 🟢 **App:** `sources_used`, the "no sources cited" badge, `iterations`,
> `sufficient`, and a per-answer trace of which query found each chunk. Genuinely
> good instrumentation — it just isn't aggregated anywhere, which is what would
> turn it into evaluation.

## 8. Improving retrieval and generation quality

- [ ] **8.1** Diagnose retrieval vs generation separately — a retrieval-only endpoint 🟢
- [ ] **8.2** Better chunking first — highest leverage 🟢
- [ ] **8.3** Add hybrid search ⚪
- [ ] **8.4** Add reranking ⚪
- [ ] **8.5** Query rewriting / expansion 🟢
- [ ] **8.6** **"Lost in the middle"** — position effects in long context 🟢
- [ ] **8.7** Context ordering and compression ⚪
- [ ] **8.8** Deduplication and diversity ⚪
- [ ] **8.9** Metadata filtering to shrink the search space 🟢
- [ ] **8.10** Fine-tuning the embedder or the generator ⚪
- [ ] **8.11** Guardrails: refusal, abstention, confidence thresholds 🟢

## 9. Prompt engineering

### 9.1 Core techniques
- [ ] **9.1.1** **Zero-shot** 🟢
- [ ] **9.1.2** **Few-shot** — example selection and ordering ⚪
- [ ] **9.1.3** **Chain-of-Thought**, and zero-shot CoT ⚪
- [ ] **9.1.4** **Self-consistency** — sample N, majority vote ⚪
- [ ] **9.1.5** **Tree of Thoughts** ⚪
- [ ] **9.1.6** **ReAct** — Thought → Action → Observation ⚪
- [ ] **9.1.7** Reflexion, ReWOO, Plan-and-Execute 🔵
- [ ] **9.1.8** **Structured output** — JSON schema, function calling, grammars 🟢
- [ ] **9.1.9** Role/system prompting, delimiters, output templates 🟢
- [ ] **9.1.10** Prompt compression ⚪

### 9.2 Anti-hallucination
- [ ] **9.2.1** Grounding instructions — "answer only from the sources" 🟢
- [ ] **9.2.2** Forced citation, and verifying citations exist 🟢
- [ ] **9.2.3** Explicit permission to abstain 🟢
- [ ] **9.2.4** Self-verification / critic passes 🟢
- [ ] **9.2.5** Constrained decoding as a hallucination control 🟢
- [ ] **9.2.6** Temperature, top_p, top_k and their effect on faithfulness 🟢

### 9.3 Safety and operations
- [ ] **9.3.1** **Prompt injection** — direct and indirect (poisoned documents) ⚪
- [ ] **9.3.2** Jailbreak resistance, system-prompt leakage ⚪
- [ ] **9.3.3** PII handling and redaction ⚪
- [ ] **9.3.4** Prompt versioning and management ⚪
- [ ] **9.3.5** Token budgeting and cost control 🟢
- [ ] **9.3.6** Model-specific failure modes 🟢

> 🟢 App: measured findings worth telling. Gemma leaks chain-of-thought into
> prose without a `responseSchema`. An extra free-text `reasoning` field made it
> degenerate *after* producing four correct sub-questions, truncating and
> invalidating the whole object — so **ask only for what you use**; a
> constrained enum is safe where free prose was not. Temperature 0.7 caused
> repetition loops; 0.35 is stable. `finishReason=RECITATION` means the model
> was reproducing memorised text — in RAG that is a **grounding failure**, not a
> token-limit problem.

## 10. LangChain

- [ ] **10.1** **LCEL / Runnables** — `|` composition, streaming, batching, async ⚪
- [ ] **10.2** **Chains** — `create_retrieval_chain`, `create_stuff_documents_chain` ⚪
- [ ] **10.3** Stuff vs map-reduce vs refine vs map-rerank ⚪
- [ ] **10.4** **Tools** — schemas, `@tool`, `bind_tools` ⚪
- [ ] **10.5** **Agents** — ReAct agent, tool-calling agent, AgentExecutor ⚪
- [ ] **10.6** **Retrievers** — `BaseRetriever`, VectorStoreRetriever ⚪
- [ ] **10.7** `EnsembleRetriever` — hybrid via RRF, out of the box ⚪
- [ ] **10.8** `ParentDocumentRetriever`, `MultiVectorRetriever` ⚪
- [ ] **10.9** `ContextualCompressionRetriever` + `LLMChainExtractor` ⚪
- [ ] **10.10** `SelfQueryRetriever`, `MultiQueryRetriever` ⚪
- [ ] **10.11** **Memory** — buffer, window, summary-buffer; `RunnableWithMessageHistory` ⚪
- [ ] **10.12** **Document processing** — loaders, transformers, text splitters ⚪
- [ ] **10.13** Output parsers, structured output, retry parsers ⚪
- [ ] **10.14** Callbacks and the tracing interface ⚪
- [ ] **10.15** Integrations: models, vector stores, tools ⚪
- [ ] **10.16** When *not* to use LangChain ⚪

> ⚪ **App reality check:** `langchain-core` and `langchain-google-genai` are
> declared in `pyproject.toml` but **never imported**. The Gemma client is
> hand-rolled `httpx` against the REST API, and chunking, retrieval and memory
> are all hand-written. Treat this section as pure study. The cheapest way to
> make it real: expose `services/retrieval.py` as a `BaseRetriever`, then add
> BM25 through `EnsembleRetriever` — which closes the hybrid gap too.

## 11. LangGraph

- [ ] **11.1** **StateGraph** — nodes, edges, typed state 🟢
- [ ] **11.2** **State and reducers** — `Annotated[list, add]`; reducers apply to the input too 🟢
- [ ] **11.3** **Conditional routing** — runtime data decides control flow 🟢
- [ ] **11.4** **Cycles** — the reason a graph beats a chain (LCEL builds DAGs, DAGs cannot loop) 🟢
- [ ] **11.5** **Checkpointers** — memory, SQLite, Postgres 🟢
- [ ] **11.6** **Durable execution** — resume after crash or restart 🟢
- [ ] **11.7** **Human-in-the-loop** — `interrupt()`, `Command(resume=...)` 🔵
- [ ] **11.8** Time travel: `get_state_history`, forking a thread ⚪
- [ ] **11.9** **Threads** and `thread_id` semantics 🟢
- [ ] **11.10** Streaming modes — `updates`, `values`, `messages`, `custom` 🟢
- [ ] **11.11** **Multi-agent** — supervisor, network, hierarchical teams ⚪
- [ ] **11.12** Subgraphs and composition ⚪
- [ ] **11.13** `Send` API and map-reduce fan-out ⚪
- [ ] **11.14** Tool nodes and `ToolNode` prebuilts ⚪
- [ ] **11.15** Error handling, retries, recursion limits 🟢
- [ ] **11.16** LangGraph Platform / Server deployment ⚪

> 🟢 App: `agent/graph.py`, `agent/nodes.py`, `agent/state.py`,
> `agent/checkpointer.py`. Two sharp lessons: **reducers cannot be reset by
> passing `[]`** (a reducer applies to the input, so it appends nothing rather
> than clearing — evidence leaked between turns until thread IDs became
> per-attempt), and **checkpoints reached 920KB against 248KB of real data**
> because every node writes a full snapshot.

## 12. Langfuse and LLM observability

- [ ] **12.1** Why LLM observability differs from ordinary APM — non-determinism, token cost, prompt as config ⚪
- [ ] **12.2** **Traces, spans, generations** — the nested model ⚪
- [ ] **12.3** Instrumenting LangChain/LangGraph via the callback handler ⚪
- [ ] **12.4** **Prompt management** — versioning, deploy without release, A/B ⚪
- [ ] **12.5** **Evaluation** — attaching scores to traces; model-based and human ⚪
- [ ] **12.6** **Datasets** — promoting production traces into a regression set ⚪
- [ ] **12.7** **Analytics** — cost, latency, token usage by model/user/prompt ⚪
- [ ] **12.8** Sessions and user-level grouping ⚪
- [ ] **12.9** Self-hosting vs cloud; PII considerations ⚪
- [ ] **12.10** Alternatives: LangSmith, Arize Phoenix, Helicone, W&B Weave, OpenTelemetry GenAI conventions ⚪

> ⚪ App: not installed. Logging is `structlog` with `contextvars` binding
> `request_id`/`owner_id` — good structured logging, **not** LLM observability.
> A turn is 3–5 model calls across four nodes with a retry loop; flat log lines
> cannot show that `critique` fired twice because retrieval missed. Cheapest
> high-value gap to close.

## 13. AWS: fundamentals

- [ ] **13.1** Regions, Availability Zones, edge locations ⚪
- [ ] **13.2** The shared responsibility model ⚪
- [ ] **13.3** Well-Architected Framework — six pillars ⚪
- [ ] **13.4** Pricing models: on-demand, reserved, savings plans, spot ⚪
- [ ] **13.5** Accounts, Organizations, control towers ⚪

## 14. AWS: compute, storage, networking

### 14.1 Compute
- [ ] **14.1.1** EC2 — instance families, AMIs, auto-scaling groups ⚪
- [ ] **14.1.2** **Lambda** — cold starts, 15-min limit, memory/CPU coupling, concurrency ⚪
- [ ] **14.1.3** Fargate ⚪
- [ ] **14.1.4** App Runner, Elastic Beanstalk, Lightsail ⚪

### 14.2 Storage
- [ ] **14.2.1** **S3** — buckets, keys, storage classes, lifecycle, versioning, presigned URLs ⚪
- [ ] **14.2.2** EBS vs EFS vs instance store ⚪
- [ ] **14.2.3** **RDS / Aurora**, Aurora Serverless v2 ⚪
- [ ] **14.2.4** DynamoDB — partition keys, GSIs, single-table design ⚪
- [ ] **14.2.5** ElastiCache (Redis) ⚪

### 14.3 Networking
- [ ] **14.3.1** **VPC** — CIDR, public vs private subnets, route tables ⚪
- [ ] **14.3.2** Internet Gateway, **NAT Gateway** (and its data-processing cost) ⚪
- [ ] **14.3.3** **Security groups (stateful) vs NACLs (stateless)** ⚪
- [ ] **14.3.4** **VPC endpoints** — reach S3 without traversing the internet ⚪
- [ ] **14.3.5** ALB vs NLB vs API Gateway ⚪
- [ ] **14.3.6** Route 53, CloudFront ⚪
- [ ] **14.3.7** VPC peering, Transit Gateway, PrivateLink ⚪

### 14.4 IAM
- [ ] **14.4.1** Users, groups, **roles**, policies ⚪
- [ ] **14.4.2** **AssumeRole** and temporary credentials ⚪
- [ ] **14.4.3** Trust policies vs permission policies ⚪
- [ ] **14.4.4** Least privilege; why instance/task roles beat long-lived keys ⚪
- [ ] **14.4.5** Resource-based policies (S3 bucket policies, KMS) ⚪
- [ ] **14.4.6** **Secrets Manager vs Parameter Store** ⚪
- [ ] **14.4.7** KMS and encryption at rest / in transit ⚪

## 15. AWS: serverless

- [ ] **15.1** Lambda execution model, handler, layers, container images ⚪
- [ ] **15.2** Triggers: API Gateway, S3, EventBridge, SQS, SNS ⚪
- [ ] **15.3** **API Gateway** — REST vs HTTP vs WebSocket APIs; authorizers; throttling ⚪
- [ ] **15.4** Step Functions — state machines vs LangGraph ⚪
- [ ] **15.5** SQS/SNS/EventBridge — queues, fan-out, DLQs ⚪
- [ ] **15.6** Cold starts, provisioned concurrency, SnapStart ⚪
- [ ] **15.7** **Why serverless is a poor fit for some AI workloads** — in-memory rate limiters and connection pools per isolated invocation; streaming/SSE duration limits ⚪
- [ ] **15.8** **RDS Proxy** for connection pooling ⚪

## 16. AWS: containers and orchestration

- [ ] **16.1** **Docker** — images, layers, multi-stage builds, caching 🟢
- [ ] **16.2** ECR — registries, image scanning ⚪
- [ ] **16.3** **ECS** — task definitions, services, Fargate vs EC2 launch type ⚪
- [ ] **16.4** **EKS** — Kubernetes control plane, node groups, IRSA ⚪
- [ ] **16.5** ECS vs EKS — when the Kubernetes tax is worth paying ⚪
- [ ] **16.6** Service discovery, load balancer integration, rolling and blue/green deploys ⚪
- [ ] **16.7** Health checks and graceful shutdown 🟢

## 17. AWS: AI/ML services

- [ ] **17.1** **Bedrock** — model access, on-demand vs provisioned throughput ⚪
- [ ] **17.2** **Bedrock Knowledge Bases** — managed chunking, embedding, retrieval ⚪
- [ ] **17.3** **Bedrock Guardrails** ⚪
- [ ] **17.4** **Bedrock Agents** — action groups, tool use ⚪
- [ ] **17.5** **SageMaker** — training, endpoints, JumpStart, Ground Truth ⚪
- [ ] **17.6** **OpenSearch Service** — kNN plugin, BM25, **hybrid search in one engine** ⚪
- [ ] **17.7** OpenSearch Serverless vector collections ⚪
- [ ] **17.8** Kendra, Textract, Comprehend, Transcribe ⚪
- [ ] **17.9** Managed RAG vs hand-built — when each wins ⚪

## 18. AWS: operations

- [ ] **18.1** **CloudWatch** — logs, metrics, alarms, Logs Insights ⚪
- [ ] **18.2** X-Ray — distributed tracing ⚪
- [ ] **18.3** CloudTrail — audit ⚪
- [ ] **18.4** **ELK / Kibana** ⚪ *(named in the JD)*
- [ ] **18.5** WAF, Shield, GuardDuty, Security Hub ⚪
- [ ] **18.6** **Cost optimisation** — right-sizing, storage classes, spot, NAT/egress traps, Cost Explorer, budgets ⚪
- [ ] **18.7** **Scalability** — horizontal vs vertical, autoscaling policies, read replicas, caching ⚪
- [ ] **18.8** **IaC** — CloudFormation, CDK, Terraform ⚪
- [ ] **18.9** CI/CD on AWS — CodePipeline, CodeBuild, CodeDeploy ⚪
- [ ] **18.10** Backup, DR, RTO/RPO, multi-AZ vs multi-region ⚪

> ⚪ App: **no AWS at all.** Stack is Render, Vercel, Neon, Qdrant Cloud,
> Supabase. Prepare by *mapping* what you built onto AWS equivalents — the
> architectural reasoning transfers even though the services do not.

## 19. Agentic AI in Python

- [ ] **19.1** What makes a system "agentic" — perceive, decide, act, observe ⚪
- [ ] **19.2** Agent loop patterns: ReAct, Plan-and-Execute, Reflexion 🔵
- [ ] **19.3** **Tool use / function calling** — schemas, validation, error recovery ⚪
- [ ] **19.4** Planning and task decomposition 🟢
- [ ] **19.5** **Memory** — short-term, long-term, episodic, semantic 🟢
- [ ] **19.6** **Multi-agent** — supervisor, hand-off, debate, role specialisation ⚪
- [ ] **19.7** **Human-in-the-loop** — approval gates, clarification, editing 🔵
- [ ] **19.8** Guardrails: iteration caps, budget caps, timeouts, loop detection 🟢
- [ ] **19.9** Determinism, reproducibility, replay 🟢
- [ ] **19.10** Frameworks: LangGraph 🟢, CrewAI ⚪, AutoGen ⚪, OpenAI Agents SDK ⚪, Pydantic AI ⚪, Smolagents ⚪
- [ ] **19.11** MCP (Model Context Protocol) — tool interoperability 🔵
- [ ] **19.12** Async Python for agents — `asyncio.gather`, streaming, backpressure 🟢
- [ ] **19.13** Rate limiting and quota management under concurrency 🟢
- [ ] **19.14** Cost accounting per run ⚪

## 20. Frontend: TypeScript, React, Next.js

- [ ] **20.1** **TypeScript** — generics, discriminated unions, narrowing, utility types 🟢
- [ ] **20.2** `unknown` vs `any`; type guards; assertion functions ⚪
- [ ] **20.3** **React** — hooks, dependency arrays, referential stability 🟢
- [ ] **20.4** `useMemo` / `useCallback` — and when they are pointless ⚪
- [ ] **20.5** Reconciliation, keys, `React.memo` ⚪
- [ ] **20.6** Suspense and error boundaries 🟢
- [ ] **20.7** React 19: Server Components, actions, `use` ⚪
- [ ] **20.8** **Next.js App Router** — server vs client components, layouts, streaming 🟢
- [ ] **20.9** Data fetching and caching semantics 🟢
- [ ] **20.10** **State management** — Context, Redux, Zustand, TanStack Query 🟢
- [ ] **20.11** **Streaming UI** — SSE, `ReadableStream`, why `EventSource` is GET-only 🟢
- [ ] **20.12** **Performance** — Core Web Vitals, bundle splitting, compositor-only animation 🟢
- [ ] **20.13** Responsive design and layout 🟢
- [ ] **20.14** **Cross-browser compatibility** ⚪
- [ ] **20.15** **Accessibility** — semantics, ARIA, keyboard, focus management, contrast 🔵
- [ ] **20.16** HTML/CSS: cascade, **cascade layers**, specificity, flex/grid 🟢

> 🟢 App: two measured traps. `useSearchParams()` outside `<Suspense>` passes
> `next dev` and **fails the production prerender**. And in Tailwind v4, an
> **unlayered CSS rule outranks everything in a layer** regardless of
> specificity — a bare `*` reset silently defeated every component's border
> colour.

## 21. Backend: Node.js and APIs

- [ ] **21.1** Node event loop, microtasks, blocking pitfalls ⚪
- [ ] **21.2** **Express / Fastify / NestJS** — routing, middleware ⚪
- [ ] **21.3** **RESTful API design** — resources, verbs, status codes, idempotency 🟢
- [ ] **21.4** **Orchestration / BFF pattern** ⚪
- [ ] **21.5** Streaming responses in Node ⚪
- [ ] **21.6** Validation: Zod, class-validator ⚪
- [ ] **21.7** Error handling and problem+json ⚪
- [ ] **21.8** OpenAPI, and generating types from a schema 🔵
- [ ] **21.9** GraphQL vs REST ⚪
- [ ] **21.10** WebSockets vs SSE vs polling 🟢
- [ ] **21.11** **Datastores** — relational, document, key-value, vector, graph, search 🟢
- [ ] **21.12** Connection pooling and exhaustion 🟢
- [ ] **21.13** Migrations 🔵

> ⚪ App: the backend is **Python/FastAPI**; Node appears only as Next.js's
> runtime. The JD wants Node.js REST APIs — the concepts transfer, the syntax
> does not. A small Node BFF in front of FastAPI would make this real.

## 22. Security

- [ ] **22.1** **Authentication** — sessions vs tokens ⚪
- [ ] **22.2** **JWT** — claims, `exp`, signature algorithms, **JWKS**, key rotation 🟢
- [ ] **22.3** Symmetric (HS256) vs asymmetric (RS256/ES256) 🟢
- [ ] **22.4** **OAuth 2.0 / OIDC** — authorization code + PKCE, flows 🟢
- [ ] **22.5** Refresh tokens, rotation, revocation 🟢
- [ ] **22.6** **Authorization** — RBAC, ABAC, row-level security, ownership checks 🟢
- [ ] **22.7** Token storage: cookies vs localStorage; `HttpOnly`, `SameSite`, CSRF 🟢
- [ ] **22.8** **OWASP Top 10** ⚪
- [ ] **22.9** XSS and why `dangerouslySetInnerHTML` on model output is dangerous 🟢
- [ ] **22.10** SQL injection and parameterised queries / ORMs 🟢
- [ ] **22.11** **CORS** — preflight, credentials, exact origins 🟢
- [ ] **22.12** Secret management and never committing secrets 🟢
- [ ] **22.13** Rate limiting and abuse prevention 🟢
- [ ] **22.14** Security headers, CSP, HTTPS/HSTS ⚪
- [ ] **22.15** Dependency and supply-chain scanning ⚪
- [ ] **22.16** **404 vs 403** — not leaking resource existence 🟢
- [ ] **22.17** **Multi-tenant isolation** — and testing it 🟢
- [ ] **22.18** LLM-specific: prompt injection, data exfiltration via tools, output handling ⚪

> 🟢 App war story: `POST /sessions/{id}/stream` called the agent **without
> `owner_id`**, which defaults to `None`, and `None` means "do not filter by
> owner" in the vector store. `/messages` passed it; `/stream` — the path the UI
> uses — did not. Proven: `owner_id=None` → 5 hits,
> `owner_id=<other user>` → 0 hits. **Lesson: a security control that is a
> default argument is a security control that will be forgotten.** Exactly what
> an ownership unit test would have caught.

## 23. Testing

- [ ] **23.1** Unit vs integration vs e2e; the test pyramid ⚪
- [ ] **23.2** **vitest** — config, mocking, coverage ⚪ *(JD mandatory)*
- [ ] **23.3** React Testing Library — queries, user-event ⚪
- [ ] **23.4** **Cypress** — e2e, selectors, network stubbing ⚪ *(JD mandatory)*
- [ ] **23.5** Playwright, and how it compares ⚪
- [ ] **23.6** **pytest** — fixtures, parametrize, `pytest-asyncio` ⚪
- [ ] **23.7** Mocking HTTP and LLM calls; recorded cassettes ⚪
- [ ] **23.8** Testing non-deterministic LLM output ⚪
- [ ] **23.9** Test data, factories, database fixtures ⚪
- [ ] **23.10** Coverage as a signal, not a target ⚪
- [ ] **23.11** Tests in CI 🔵

> ⚪ App: **zero test files.** `pytest` is a declared dev dependency with
> nothing to run; CI tolerates exit code 5 and prints a warning. The JD lists
> unit testing under **Mandatory Skills** — this is the highest-risk gap.

## 24. DevOps, tooling, process

- [ ] **24.1** **Docker** — multi-stage, layer caching, `.dockerignore`, image size 🟢
- [ ] **24.2** docker compose for local development 🟢
- [ ] **24.3** **Git** — branching, rebase vs merge, bisect, hooks 🟢
- [ ] **24.4** **GitHub/BitBucket** — PRs, reviews, branch protection, CODEOWNERS 🔵
- [ ] **24.5** **CI/CD** — CI vs Continuous Delivery vs Continuous Deployment 🟢
- [ ] **24.6** GitHub Actions — jobs, matrices, caching, concurrency 🟢
- [ ] **24.7** Reproducible builds and lockfiles 🟢
- [ ] **24.8** Monorepos — Rush, Nx, Turborepo, workspaces 🔵
- [ ] **24.9** **IaC** — Terraform/CDK ⚪
- [ ] **24.10** **Jira** and Agile: sprints, estimation, story points, ceremonies ⚪
- [ ] **24.11** Peer code review — giving and receiving ⚪
- [ ] **24.12** **Logging frameworks** — structured logging, correlation IDs, levels 🟢
- [ ] **24.13** **GitHub Copilot / MS Copilot** — effective use, review discipline ⚪
- [ ] **24.14** Feature flags, canaries, rollbacks ⚪

---

# PART 2 — HOLDING AREA

> **Being absorbed into Part 1.** §A's cosine-distance material belongs in
> §3.2.1 and moves there when Section 3 is expanded.
>
> **§B, §C and §D are now SUPERSEDED — skip them.** Their content is fully
> inline and expanded at **§2.9** (parent-document), **§2.12** (late chunking)
> and **§2.13** (contextual retrieval). They stay here only until Section 3 is
> written, then get deleted.

## A. Cosine distance (moving to §3.2.1)

### A.1 Cosine similarity, from first principles

An embedding is a list of numbers — a **direction** in high-dimensional space.
Two texts that mean similar things point in similar directions.

**Cosine similarity** measures the angle between two vectors, ignoring their
length:

```
cos_sim(A, B) = (A · B) / (|A| × |B|)

  A · B  = Σ Aᵢ × Bᵢ            (dot product)
  |A|    = √(Σ Aᵢ²)              (magnitude / L2 norm)
```

Range **−1 to +1**:

| Value | Angle | Meaning |
|---|---|---|
| `+1` | 0° | identical direction |
| `0` | 90° | unrelated |
| `−1` | 180° | opposite |

**Cosine distance = 1 − cosine similarity**, range 0 to 2. Distance is just
similarity inverted so that "closer = smaller", which is what index structures
want.

### A.2 Worked example in 2 dimensions

Take `A = [1, 0]`:

| Vector | Dot with A | \|B\| | cos_sim | cos_dist | Reading |
|---|---|---|---|---|---|
| `B = [0.9, 0.44]` | 0.9 | 1.002 | **0.898** | 0.102 | very similar |
| `C = [0.5, 0.87]` | 0.5 | 1.004 | **0.498** | 0.502 | loosely related |
| `D = [0, 1]` | 0 | 1 | **0.0** | 1.0 | unrelated |
| `E = [−1, 0]` | −1 | 1 | **−1.0** | 2.0 | opposite |

Real embeddings do this in 768 or 3072 dimensions instead of 2, but the
arithmetic is identical.

### A.3 Why cosine and not Euclidean

Because **magnitude usually encodes something you do not want**. Longer text, or
text with repeated words, tends to produce a longer vector. If you used raw
Euclidean distance, a long passage and a short passage about the same topic
could look far apart purely because of length.

Cosine throws magnitude away and keeps only direction — meaning.

**The important corollary:** if all vectors are **L2-normalised** (scaled to
length 1), then:

- `|A| = |B| = 1`, so **cosine similarity = dot product** exactly
- Euclidean distance becomes `√(2 − 2·cos_sim)` — a monotonic function of
  cosine, so **it ranks identically**

That is why normalisation matters, and why a *partially* normalised vector is a
silent bug: the geometry no longer holds.

> 🟢 **App connection.** Gemini returns 3072-dim vectors that are L2-normalised.
> Requesting `outputDimensionality=768` truncates them — and Google normalises
> only the *full* vector, so the truncated one is **no longer unit length**.
> `services/embeddings.py` re-normalises client-side. Skip that and cosine
> distance goes subtly wrong: not broken enough to notice, just quietly worse.

### A.4 Also worth knowing: scores are not comparable across queries

A cosine score is only meaningful *relative to other results for the same
query*. Measured in this app: an irrelevant match scored **0.570** while a
correct match for a different query scored **0.615**. You cannot set a global
"relevance threshold ≥ 0.6" and expect it to mean anything.

This is precisely why **RRF fuses on rank, not score**.

### A.5 Semantic chunking

**The idea:** instead of cutting at a fixed character count, cut where the
*meaning* changes.

Algorithm:

1. Split the document into sentences
2. Embed each sentence
3. Compute cosine distance between each **consecutive pair**
4. A **spike** in distance means the topic changed — cut there

Worked example on the report in `fixtures/`:

| # | Sentence | Distance to previous | |
|---|---|---|---|
| 1 | "Acme reported total revenue of $847.3 million for fiscal 2024." | — | |
| 2 | "This was an increase of 18.4% over the $715.6 million in 2023." | **0.08** | same topic |
| 3 | "Gross margin improved to 62.1% from 58.7%." | **0.15** | same topic |
| 4 | "Total headcount stood at 4,182 employees." | **0.42** | ⬅ **SPIKE → cut** |
| 5 | "R&D expense was $167.2 million, or 19.7% of revenue." | **0.12** | same topic |

Two chunks, split between 3 and 4 — at the real topic boundary, which no
character count would have found.

**Practical refinements:**

- **Buffering.** Embedding single sentences is noisy. Embed a *window*
  (sentence i−1 + i + i+1) so each point reflects local context.
- **Breakpoint threshold.** How big is a "spike"? Options: percentile (cut at
  the 95th percentile of distances), standard deviation, interquartile range, or
  gradient. Percentile is the usual default because it adapts per document.

**Costs and limits:** one embedding call per sentence at ingestion (expensive
for large corpora), it cannot exceed sentence granularity, and on documents with
no topic structure it produces arbitrary cuts.

> 🟢🔜 **App fit — genuinely useful here.** `chunking._split_sections()` returns
> `[(None, text)]` when a document has **no markdown headings**, and then falls
> back to pure recursive character splitting. That is the case for **every PDF**.
> Markdown files get structure for free; PDFs get nothing. Semantic chunking is
> exactly the right fallback for the no-heading branch — a natural place to add
> it and measure against the current behaviour.

---

## B. Parent-document retrieval (small-to-big)

### B.1 The tension it resolves

Chunk size is pulled in two directions at once:

| | Small chunks | Large chunks |
|---|---|---|
| Embedding precision | **Good** — one idea, one vector | Poor — meaning diluted across topics |
| Context for the LLM | **Poor** — a sentence with no surroundings | Good — full explanation present |

You are forced to choose... unless you **decouple what you search from what you
return**.

### B.2 The mechanism

- **Index** small chunks (a sentence or two) — precise vectors
- **Store** a pointer from each small chunk to its larger **parent** (the
  section, or a wide window)
- **At query time**: match on the small chunk, then **return the parent** to the
  LLM

The vector store holds children; the LLM sees parents.

### B.3 Worked example

Query: **"what was the gross margin?"**

**Child chunk that matches** (precise, high cosine):

> "Gross margin improved to 62.1% from 58.7%."

If you hand *only that* to the model, it can state the number but cannot say
anything about why.

**Parent returned instead** (the whole Financial Summary section):

> "Acme Corporation reported total revenue of $847.3 million for fiscal year
> 2024, an increase of 18.4% over the $715.6 million recorded in 2023. Gross
> margin improved to 62.1% from 58.7%, **driven primarily by a shift in product
> mix toward higher-margin subscription offerings.** Operating income reached
> $112.8 million…"

Same precise match, far better answer — it can now explain the driver.

### B.4 Variants

| Variant | How it expands |
|---|---|
| **Parent-document** | Child → its explicit parent document/section (LangChain `ParentDocumentRetriever`) |
| **Sentence-window** | Retrieve a sentence, expand ±k sentences around it |
| **Auto-merging / hierarchical** | Multi-level tree; if enough sibling children match, return the parent *instead of* the children |

Auto-merging is the clever one: it adapts. Scattered matches return leaves;
concentrated matches collapse into the parent, avoiding near-duplicate context.

### B.5 Costs

Storage roughly doubles (children indexed, parents stored). Parents can blow the
context window — cap parent size or you lose the token savings that made small
chunks attractive. And deduplication becomes essential: two matching children
with the same parent must not send that parent twice.

> 🟢🔜 **App fit — the metadata is already there.** Postgres `chunks` rows carry
> `document_id` and **`chunk_index`**, so expanding a hit to its neighbours is
> one cheap SQL query:
>
> ```sql
> SELECT text FROM chunks
> WHERE document_id = :doc AND chunk_index BETWEEN :i - 1 AND :i + 1
> ORDER BY chunk_index;
> ```
>
> Sentence-window retrieval is therefore nearly free to add — no re-indexing,
> no schema change. Full parent-document retrieval would want an explicit
> `parent_id`, which is a migration.

---

## C. Long-context embedders and late chunking

### C.1 Token limits are the constraint

Every embedding model has a maximum input length. Exceed it and the model
**silently truncates** — you get a vector for the first N tokens and no error.

| Model | Max tokens |
|---|---|
| BGE / E5 / MiniLM | 512 |
| OpenAI `text-embedding-3` | 8,191 |
| Gemini `embedding-001` | ~2,048 |
| **jina-embeddings-v2/v3** | **8,192** |
| **nomic-embed-text-v1.5** | **8,192** |

A "long-context embedder" is simply one whose window is large enough to hold a
whole document rather than a fragment. Achieved through positional encoding
schemes (ALiBi, RoPE scaling) that extrapolate beyond the training length.

### C.2 The problem late chunking solves

**Normal pipeline:** chunk first, then embed each chunk **independently**. Each
chunk's vector is computed with no knowledge of the rest of the document.

Consider this real chunk from the fixture:

> "The segment grew 27.3% year over year."

Embedded on its own, it is nearly useless. *Which* segment? The vector cannot
encode what the text does not say. A query for "Cloud Platform growth" may never
match it.

### C.3 The mechanism

Late chunking **inverts the order**:

1. Feed the **entire document** through the transformer
2. Every **token** embedding now carries attention over the whole document
3. **Then** apply chunk boundaries, and **mean-pool the token embeddings within
   each boundary** to get one vector per chunk

```
NORMAL:  split → [chunk] → transformer → pool → vector   (context-blind)
LATE:    whole document → transformer → token vectors
                                          ↓ split + pool per chunk
                                        vectors            (context-aware)
```

The chunk text is unchanged; the **vector** is different. Because the tokens in
"The segment grew 27.3%" attended to "Cloud Platform" earlier in the document,
the pooled vector carries that association — the chunk effectively knows which
segment it is about without a word being added.

### C.4 Requirements and limits

You need **token-level output** from the model, so it must be self-hosted or a
provider that exposes late chunking explicitly. Any API returning a single
pooled vector per input makes this impossible. The document must also fit the
context window, so very long documents need a hierarchical approach.

> ⚪ **App fit — architecturally possible, practically blocked.**
> `EMBEDDING_PROVIDER` is an ABC with a local `fastembed` option, which is the
> right shape for this. But `fastembed` returns pooled vectors, not token
> embeddings, so late chunking would need a different local runtime
> (`sentence-transformers` or raw ONNX with manual pooling). The Gemini API path
> cannot do it at all.

---

## D. Contextual retrieval

### D.1 The technique

Introduced by Anthropic (September 2024). Same problem as late chunking —
context-blind chunks — but solved with **text instead of geometry**.

For each chunk, make one LLM call with **the whole document plus that chunk**,
asking for a sentence or two situating the chunk. **Prepend that to the chunk
text** before embedding *and* before BM25 indexing.

### D.2 Worked example

**Original chunk:**

> "The segment grew 27.3% year over year."

**LLM prompt (roughly):**

> Here is the whole document: `<document>`
> Here is a chunk from it: `<chunk>`
> Give a short context that situates this chunk within the document. Answer with
> the context only.

**Contextualised chunk that gets indexed:**

> "This chunk is from Acme Corporation's 2024 annual report, in the Revenue by
> Segment section, discussing the Cloud Platform segment which generated $512.9
> million and 60.5% of total revenue. The segment grew 27.3% year over year."

Now a query about "Cloud Platform growth" matches strongly — via embeddings
*and* via keyword search, because "Cloud Platform" is literally present.

### D.3 Measured results (Anthropic's numbers)

| Setup | Retrieval failure rate |
|---|---|
| Baseline (embeddings only) | 5.7% |
| + contextual embeddings | 3.7% (**−35%**) |
| + contextual BM25 | 2.9% (**−49%**) |
| + reranking (top 150 → 20) | 1.9% (**−67%**) |

Those are among the most useful numbers to have memorised for a RAG interview.

### D.4 Cost, and what makes it viable

One LLM call **per chunk** at ingestion — naively very expensive. It became
practical because of **prompt caching**: the document is the large, repeated
part of the prompt, so it is cached once and each chunk call pays only for the
chunk plus the short output. Anthropic reported roughly $1.02 per million
document tokens.

### D.5 Contextual retrieval vs late chunking

| | Contextual retrieval | Late chunking |
|---|---|---|
| Changes the **text** | Yes — prepends real words | No |
| Changes the **vector** | Indirectly, via new text | Yes, directly |
| Helps **BM25** | **Yes** — the words are there | No |
| Cost | One LLM call per chunk | One long forward pass per document |
| Needs token-level access | No | **Yes** |
| Interpretable | Yes — you can read it | No |

They are complementary, and contextual retrieval is usually the more practical
of the two because it works with any black-box embedding API.

> 🔵 **App fit — you already ship a cheap approximation.**
> `chunking.chunk_pages()` **prefixes the markdown heading into every chunk**:
>
> ```
> "## Revenue by Segment\nThe Cloud Platform segment generated $512.9 million…"
> ```
>
> That is deterministic, free, and zero LLM calls — a poor man's contextual
> retrieval, and it **measurably worked** (correct section 0.625 → 0.710).
> Contextual retrieval is the LLM-powered generalisation: it works when there
> are no headings, and it can pull in facts from elsewhere in the document.
>
> There is also an `ENRICHERS` protocol hook already designed into the ingestion
> pipeline for exactly this class of per-chunk enrichment, so wiring it in would
> be a natural fit.

# Research Desk

Four surfaces: **Chat**, **Library**, **Lab**, **Atlas**.

## The idea

Ask questions across documents you upload and, when it helps, the web — and get
answers that cite the passages they came from, so every claim can be opened and
checked. Library is where documents go in. Lab is the developer's view of the
same machinery. Atlas is the corpus as geometry rather than as conversation.

## Notable decisions

**Citations are the product, not a garnish.** An answer with a number in it is
a claim until somebody can open the passage. Every cited chunk is clickable and
renders the source text beside the answer.

**Not every tool result is citable.** `corpus_facts` in `agent/state.py` keeps
that line: a document count or a list of filenames is a fact about the corpus,
not a retrieved passage, and letting the model cite one would produce a
footnote pointing at nothing.

**Clarify before searching, not after.** A vague question gets one clarifying
question and the graph *pauses* — a human-in-the-loop interrupt, checkpointed —
rather than searching on a guess and explaining afterwards. Nothing is searched
while it waits.

**Hybrid retrieval, with every stage optional.** Dense alone misses exact
terms; BM25 alone misses paraphrase. Each stage after the dense search degrades
to the stage before it, so the default path is exactly what it was before any
of them existed.

**Deduplication is by content, not filename.** The identity of a document is
the SHA-256 of its bytes: `report.md` and `report (1).md` are the same
document, and the same filename with an edit is not. The unique index uses
`COALESCE(owner_id, '')`, because in SQL two NULLs are distinct — without it,
anonymous uploads would never collide with each other.

**Atlas computes nothing new.** Both views are built from vectors that already
exist: no embedding calls, no model calls. It answers questions retrieval
quality depends on and nothing else in the app shows — is this document an
outlier, are these two chunks near-duplicates, is this chunk similar to nothing
at all.

## The tech

| | |
|---|---|
| Orchestration | LangGraph, with a Postgres checkpointer (psycopg3) |
| Vectors | Qdrant, 768-dim Gemini embeddings |
| Lexical | BM25 over Postgres |
| Fusion | Reciprocal Rank Fusion, then a rerank pass |
| Web | Serper |
| Parsing | PDF, DOCX, Markdown, plain text |
| Atlas | Three.js, PCA/SVD projection |

## How it works

### The graph

`backend/app/agent/graph.py`:

```
entry → route → clarify → ask_human → plan ─┐
                                   └→ react ┴→ retrieve → draft → critique → resolve
```

- **route** decides what kind of question this is. A "remember this" turn
  stores an instruction and never searches — which is why the missing-citation
  warning is suppressed for it, rather than accusing it of skipping the thing
  it was told not to do.
- **clarify / ask_human** is the interrupt. The graph stops, the browser shows
  options, and the checkpoint is resumed with the answer.
- **plan vs react** is the gather strategy: a planned multi-query sweep, or a
  ReAct loop that can call tools repeatedly (up to 6 rounds, 4 calls each).
- **critique** can send a draft back once.

Two Postgres drivers live in this app for this reason: SQLAlchemy uses asyncpg,
LangGraph's checkpointer speaks psycopg3. Same database, different pools.

A graph built **without** a checkpointer silently accepts a missing
`thread_id` — which is how an early bug passed its tests and 500'd in
production. The tests now assert on the arguments, not just the result.

### Retrieval

`backend/app/services/retrieval.py`:

```
query → [multi-query expand] → dense ─┐
                                BM25 ─┴→ floor → RRF → rerank → top_k → parents
```

Parent expansion returns the surrounding context of a matched chunk, so an
answer is not built from a sentence torn out of its paragraph.

### Ingest

`POST /documents` → hash → dedupe → **store the original** → parse → chunk →
embed → upsert. Storing before parsing is deliberate: parsing is the step most
likely to fail, and the original is what somebody needs to work out why. See
[storage.md](storage.md).

### Atlas

Two views over the same 768-dimensional space, projected with a reusable PCA
`Basis` so a query vector can be placed in the *same* projection as the corpus
rather than a fresh one:

- **Scatter** — every chunk as a point, coloured by document.
- **Query ray** — under each chat message, the line from the question to the
  passages it actually retrieved.
- **Threads** — a document's chunks in order, so drift across a long document
  is visible.
- **Territory** — a Voronoi partition of the space (`NearestFilter`, so the
  cells stay hard-edged rather than blurring into each other).

### Lab

The same endpoints as Chat, without the chat: raw `search`, `ask`, `research`
and corpus stats, plus a benchmark view. It exists so retrieval can be examined
without a conversation wrapped around it.

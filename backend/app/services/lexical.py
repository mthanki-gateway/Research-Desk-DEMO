"""BM25 lexical search, in process.

WHY SPARSE RETRIEVAL AT ALL

Dense and sparse do not fail gracefully in each other's territory -- they fail
COMPLETELY, which is why picking one is not an option:

    query                          dense              BM25
    "gemini-3.6-flash"             smoothed toward    exact, top rank
                                   "language model"
    "INC-2024-1183"                numbers carry      exact
                                   almost no signal
    "4,182"                        near-random        exact
    "hard to find" vs              strong             zero — no shared
      "couldn't locate it"                            tokens

A bi-encoder embeds meaning, so a rare identifier gets pulled toward whatever
cluster it resembles. BM25 scores exact term overlap weighted by rarity, so the
identifier -- the highest-IDF token in the query -- dominates. Each is the
other's blind spot.

WHY IN PROCESS, NOT POSTGRES FULL-TEXT

Postgres `ts_rank_cd` is not BM25: it has no document-length normalisation and
its term weighting is not IDF. Implementing the real thing over a corpus that
fits in memory is a few dozen lines and behaves the way the literature says,
which matters when the point of the exercise is to understand the algorithm.

THE SCALING CAVEAT, STATED PLAINLY

The index is rebuilt from SQL per scope and cached. At this corpus size that is
microseconds. This approach stops being reasonable somewhere in the high tens of
thousands of chunks, at which point the answer is a real inverted index
(Postgres GIN + tsvector, or Qdrant's sparse vectors) rather than a bigger
dictionary. The seam is `_build_index`.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass

import structlog
from sqlalchemy import select

from app.db.models import Chunk, Document
from app.db.session import SessionLocal

log = structlog.get_logger()

# BM25's two free parameters, at the values the original papers use.
#
# k1 controls TERM FREQUENCY SATURATION: how quickly repeating a term stops
# helping. At 1.2 the fifth occurrence adds far less than the second, which is
# what stops a chunk that says "revenue" nine times from beating one that
# actually answers the question.
#
# b controls LENGTH NORMALISATION. At 0.75 a long chunk is penalised for its
# length but not fully -- b=0 ignores length entirely (long chunks win on raw
# term counts), b=1 divides it out completely (short chunks win).
K1 = 1.2
B = 0.75

# Tokens keep INTERNAL punctuation: "4,182", "62.1%", "gemini-3.6-flash" and
# "INC-2024-1183" are single terms, and they are precisely the queries where
# BM25 beats dense retrieval. Splitting on every non-alphanumeric would shatter
# them into "4"/"182" and throw away the rarity that makes them findable.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,%/\-][a-z0-9]+)*")

# Deliberately small. A long stop list is a tuning decision that needs evidence,
# and BM25 already handles common words correctly -- a term appearing in every
# document has an IDF near zero and contributes almost nothing. These are here
# only to keep the postings lists tidy.
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were what which who with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True, slots=True)
class LexicalHit:
    chunk_id: uuid.UUID
    score: float


class BM25Index:
    """An immutable BM25 index over one scope's chunks."""

    __slots__ = ("_df", "_ids", "_tf", "_lengths", "_avgdl", "_n")

    def __init__(self, documents: list[tuple[uuid.UUID, str]]) -> None:
        self._ids: list[uuid.UUID] = []
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()

        for chunk_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._ids.append(chunk_id)
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            # Document frequency counts DOCUMENTS, not occurrences -- hence the
            # set. Counting occurrences here would make IDF meaningless for any
            # term that repeats.
            self._df.update(set(counts))

        self._n = len(self._ids)
        self._avgdl = (sum(self._lengths) / self._n) if self._n else 0.0

    def __len__(self) -> int:
        return self._n

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.

        The textbook form ln((N - df + 0.5)/(df + 0.5)) goes NEGATIVE for a term
        appearing in more than half the corpus, so on a small corpus a common
        word actively subtracts from the score and a chunk is punished for
        containing it. The `1 +` inside the log is the standard fix and is what
        Lucene uses.
        """
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, limit: int) -> list[LexicalHit]:
        if not self._n:
            return []

        terms = Counter(tokenize(query))
        if not terms:
            return []

        scored: list[LexicalHit] = []
        for i, chunk_id in enumerate(self._ids):
            tf = self._tf[i]
            length = self._lengths[i]
            total = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = 1.0 - B + B * (length / self._avgdl if self._avgdl else 1.0)
                total += self._idf(term) * (freq * (K1 + 1.0)) / (freq + K1 * norm)
            if total > 0.0:
                scored.append(LexicalHit(chunk_id=chunk_id, score=total))

        # Deterministic tie-break on the id. Without it, two chunks with equal
        # scores can swap places between runs and the evaluation numbers wobble
        # for reasons unrelated to any change.
        scored.sort(key=lambda h: (-h.score, str(h.chunk_id)))
        return scored[:limit]


# --------------------------------------------------------------------------
# scoped index cache
# --------------------------------------------------------------------------

# Keyed by SCOPE, because the index must never span tenants. Building it from a
# scoped query rather than filtering a global index afterwards means a chunk the
# caller may not see is never in the structure at all -- the isolation is
# structural instead of a step that could be forgotten.
_cache: dict[tuple, BM25Index] = {}


def _scope_key(
    owner_id: str | None, document_ids: list[uuid.UUID] | None
) -> tuple:
    return (owner_id, tuple(sorted(str(d) for d in document_ids)) if document_ids else None)


def invalidate_lexical_index() -> None:
    """Drop every cached index. Called after ingest and after deletion.

    Coarse on purpose: rebuilding is a single indexed SELECT plus a pass over
    the text, and a stale index silently omits a document the user just
    uploaded -- far worse than a rebuild nobody notices.
    """
    if _cache:
        log.info("lexical_index_invalidated", n_scopes=len(_cache))
    _cache.clear()


async def _build_index(
    owner_id: str | None, document_ids: list[uuid.UUID] | None
) -> BM25Index:
    stmt = select(Chunk.id, Chunk.heading, Chunk.text).join(
        Document, Chunk.document_id == Document.id
    )
    # Same scoping rules as the vector store, and they have to stay in step:
    # owner_id None means "no owner filter" (auth disabled), exactly as it does
    # there, while an explicit owner restricts to that tenant.
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    async with SessionLocal() as db:
        rows = (await db.execute(stmt)).all()

    # The heading is indexed with the body. It carries the section's real
    # vocabulary ("## Root Cause Analysis"), which is often the exact term a
    # lexical query uses and which the body may never repeat.
    return BM25Index([(cid, f"{heading or ''}\n{text}") for cid, heading, text in rows])


async def lexical_search(
    query: str,
    *,
    limit: int,
    owner_id: str | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[LexicalHit]:
    """BM25 over the caller's chunks. Returns [] rather than raising.

    Never load-bearing: hybrid retrieval must degrade to dense-only if the
    lexical half fails, because dense alone is the behaviour the app had before
    this existed.
    """
    key = _scope_key(owner_id, document_ids)
    index = _cache.get(key)
    if index is None:
        try:
            index = await _build_index(owner_id, document_ids)
        except Exception as exc:  # noqa: BLE001 - degrade to dense-only
            log.warning("lexical_index_failed", error=str(exc))
            return []
        _cache[key] = index
        log.info("lexical_index_built", n_chunks=len(index), scoped=bool(document_ids))

    return index.search(query, limit=limit)

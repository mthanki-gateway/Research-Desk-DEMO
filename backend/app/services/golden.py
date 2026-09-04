"""Golden evaluation set: loading, and matching retrieved chunks to labels.

This module is the bridge between `fixtures/golden.yaml` and the pure metrics
in `evaluation.py`. It exists as a separate layer for one reason: **labels
reference content, not ids.**

Chunk UUIDs are generated at ingest and change on every re-ingest, and
`chunk_index` shifts whenever the chunker changes -- which is precisely what
this suite exists to measure. So a spec matches on `filename` plus required
substrings, which survives re-chunking, re-embedding and swapping embedding
models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

def _find_fixtures() -> Path:
    """Locate fixtures/golden.yaml by walking up from this file.

    A fixed `parents[n]` cannot work, because the directory depth differs
    between the two layouts this runs in:

        host       .../DEMO/backend/app/services/golden.py   -> parents[3]
        container  /app/app/services/golden.py               -> parents[2]

    Hard-coding either one breaks the other, which is exactly what happened.
    Walking up until a `fixtures/golden.yaml` appears is layout-independent and
    also lets the file be found from a test runner in any working directory.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "fixtures" / "golden.yaml"
        if candidate.is_file():
            return candidate
    # Fall back to the container mount point so the error message names a real
    # path rather than something derived from a failed search.
    return Path("/app/fixtures/golden.yaml")


DEFAULT_GOLDEN_PATH = _find_fixtures()


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """One fact a question needs, described by content."""

    file: str
    must_contain: tuple[str, ...] = ()

    def matches(self, filename: str, text: str) -> bool:
        """Case-insensitive: labels are hand-written and shouldn't be brittle."""
        if filename.lower() != self.file.lower():
            return False
        haystack = text.lower()
        return all(needle.lower() in haystack for needle in self.must_contain)


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    question: str
    expect_chunks: tuple[ChunkSpec, ...] = ()
    expected_answer: str = ""
    expect_refusal: bool = False
    tags: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        """No expected chunks means the corpus cannot answer it."""
        return bool(self.expect_chunks)

    @property
    def total_relevant(self) -> int:
        """How many distinct facts the question needs.

        Counts SPECS, not chunks. If one spec happens to match three retrieved
        chunks, the question still needed one fact -- see `match` for why that
        distinction has to be enforced in the relevance vector.
        """
        return len(self.expect_chunks)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Two relevance vectors, because precision and recall have different
    denominators.

    `precision_relevance` marks EVERY hit that satisfies any spec. That is what
    precision, MRR and NDCG want: a second genuinely-relevant chunk is a good
    result, not a wasted slot.

    `recall_relevance` marks only the FIRST hit to satisfy each spec. Without
    that, a single spec matching three retrieved chunks would count as three
    towards recall and push it above 1.0 -- recall must be measured in facts
    found, not chunks marked.

    They are identical whenever each spec matches at most one chunk, which is
    the common case; the distinction only bites on broad questions.
    """

    precision_relevance: list[bool]
    recall_relevance: list[bool]
    # spec index -> rank (1-based) of the first hit that satisfied it. Kept so
    # the UI can show "expected at rank 12" rather than a bare pass/fail.
    satisfied_at: dict[int, int]
    total_specs: int

    @property
    def unsatisfied_specs(self) -> list[int]:
        """Spec indices no retrieved chunk satisfied — the actual failures."""
        return [i for i in range(self.total_specs) if i not in self.satisfied_at]


def match(question: GoldenQuestion, hits: list[tuple[str, str]]) -> MatchResult:
    """Score retrieved hits against a question's labels.

    `hits` is rank-ordered `(filename, text)`; index 0 is rank 1.
    """
    specs = question.expect_chunks
    precision_relevance: list[bool] = []
    recall_relevance: list[bool] = []
    satisfied_at: dict[int, int] = {}

    for rank, (filename, text) in enumerate(hits, start=1):
        matched_any = False
        first_for_spec = False
        for index, spec in enumerate(specs):
            if not spec.matches(filename, text):
                continue
            matched_any = True
            if index not in satisfied_at:
                satisfied_at[index] = rank
                first_for_spec = True
        precision_relevance.append(matched_any)
        recall_relevance.append(first_for_spec)

    return MatchResult(
        precision_relevance=precision_relevance,
        recall_relevance=recall_relevance,
        satisfied_at=satisfied_at,
        total_specs=len(specs),
    )


def _spec(raw: dict[str, Any]) -> ChunkSpec:
    contains = raw.get("must_contain") or []
    if isinstance(contains, str):  # tolerate a bare string
        contains = [contains]
    return ChunkSpec(file=raw["file"], must_contain=tuple(contains))


def load_golden_set(path: Path | None = None) -> list[GoldenQuestion]:
    """Parse fixtures/golden.yaml.

    Raises on a malformed file rather than skipping rows: a silently dropped
    question would quietly inflate every metric, which is the worst possible
    failure mode for an evaluation harness.
    """
    target = path or DEFAULT_GOLDEN_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))

    entries = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{target}: expected a `questions:` list")

    out: list[GoldenQuestion] = []
    seen: set[str] = set()
    for entry in entries:
        qid = entry["id"]
        if qid in seen:
            raise ValueError(f"{target}: duplicate question id {qid!r}")
        seen.add(qid)
        out.append(
            GoldenQuestion(
                id=qid,
                question=entry["question"],
                expect_chunks=tuple(_spec(s) for s in entry.get("expect_chunks") or []),
                expected_answer=(entry.get("expected_answer") or "").strip(),
                expect_refusal=bool(entry.get("expect_refusal", False)),
                tags=tuple(entry.get("tags") or []),
            )
        )

    log.info(
        "golden_set_loaded",
        path=str(target),
        n=len(out),
        answerable=sum(1 for q in out if q.answerable),
        unanswerable=sum(1 for q in out if not q.answerable),
    )
    return out

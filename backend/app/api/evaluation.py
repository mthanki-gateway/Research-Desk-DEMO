"""Evaluation endpoints, for the Lab UI.

Tier 1 only. A Tier 1 run is one embedding call per question and no LLM calls,
so it is fast enough to serve from a web request -- which is the whole reason
the tiers are split. Tier 2 (faithfulness, answer relevancy) runs a full agent
plus a judge per question, which is minutes of wall-clock against a 16K
tokens/minute budget, so it stays in `app.scripts.evaluate`.

Everything is scoped to the caller's `owner_id`, so the benchmark measures the
corpus that caller can actually see. That has a consequence worth knowing: a
deployment whose fixtures were never ingested will score near zero, and the
response reports `n_questions` and the corpus size so the UI can say so rather
than presenting a meaningless number.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.auth import User, current_user
from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.services.eval_runner import DEFAULT_K_VALUES, run_tier1
from app.services.golden import load_golden_set

log = structlog.get_logger()

router = APIRouter(prefix="/eval", tags=["evaluation"])

# Caps, because this endpoint spends embedding quota. A run is bounded at
# MAX_QUESTIONS embedding calls (times the expansion factor if multi-query is
# on), which at 100 requests/minute leaves plenty of headroom for normal use.
MAX_QUESTIONS = 100
MAX_TOP_K = 50


class GoldenQuestionOut(BaseModel):
    id: str
    question: str
    tags: list[str]
    answerable: bool
    n_facts: int


class GoldenSetOut(BaseModel):
    n_questions: int
    n_answerable: int
    n_unanswerable: int
    tags: list[str]
    questions: list[GoldenQuestionOut]


@router.get("/golden", response_model=GoldenSetOut)
async def get_golden_set(_: User = Depends(current_user)) -> GoldenSetOut:
    """The suite, without running it — so the UI can offer tag filters."""
    questions = load_golden_set()
    return GoldenSetOut(
        n_questions=len(questions),
        n_answerable=sum(1 for q in questions if q.answerable),
        n_unanswerable=sum(1 for q in questions if not q.answerable),
        tags=sorted({t for q in questions for t in q.tags}),
        questions=[
            GoldenQuestionOut(
                id=q.id,
                question=q.question,
                tags=list(q.tags),
                answerable=q.answerable,
                n_facts=q.total_relevant,
            )
            for q in questions
        ],
    )


class Tier1Request(BaseModel):
    # None means "max of k_values", so one retrieval serves every k.
    top_k: int | None = Field(None, ge=1, le=MAX_TOP_K)
    k_values: list[int] = Field(default_factory=lambda: list(DEFAULT_K_VALUES))
    multi_query: bool = False
    # Empty means everything.
    tags: list[str] = Field(default_factory=list)
    ids: list[str] = Field(default_factory=list)


@router.post("/tier1")
async def run_tier1_endpoint(
    req: Tier1Request, user: User = Depends(current_user)
) -> dict:
    """Run the retrieval benchmark and return the full report.

    Returns a plain dict rather than a Pydantic model on purpose: the report is
    already a dataclass tree with a `to_dict()`, and restating that shape as
    response models would mean maintaining two definitions of the same thing
    that could silently drift apart.
    """
    questions = load_golden_set()

    if req.tags:
        wanted = set(req.tags)
        questions = [q for q in questions if wanted & set(q.tags)]
    if req.ids:
        wanted_ids = set(req.ids)
        questions = [q for q in questions if q.id in wanted_ids]

    if not questions:
        raise HTTPException(status_code=400, detail="No questions matched those filters.")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{len(questions)} questions exceeds the {MAX_QUESTIONS} cap.",
        )

    k_values = tuple(sorted({k for k in req.k_values if 1 <= k <= MAX_TOP_K}))
    if not k_values:
        raise HTTPException(status_code=400, detail="No valid k values supplied.")

    report = await run_tier1(
        questions,
        top_k=req.top_k,
        k_values=k_values,
        multi_query=req.multi_query,
        owner_id=user.owner_id,
        # Recorded so the report can say it covers a subset.
        filters={
            **({"tags": req.tags} if req.tags else {}),
            **({"ids": req.ids} if req.ids else {}),
        },
    )
    log.info(
        "eval_tier1_served",
        n=len(questions),
        multi_query=req.multi_query,
        seconds=round(report.elapsed_seconds, 2),
    )
    return report.to_dict()


@router.get("/corpus")
async def corpus_summary(
    user: User = Depends(current_user),
    include_unowned: bool = Query(False),
) -> dict:
    """What the benchmark is measuring against.

    Exists because a low score has two very different causes -- bad retrieval,
    or a corpus that was never ingested -- and the UI should be able to tell
    them apart before anyone draws a conclusion.
    """
    async with SessionLocal() as db:
        doc_q = select(Document.filename, func.count(Chunk.id)).outerjoin(
            Chunk, Chunk.document_id == Document.id
        )
        if user.owner_id is not None and not include_unowned:
            doc_q = doc_q.where(Document.owner_id == user.owner_id)
        rows = (await db.execute(doc_q.group_by(Document.filename))).all()

    return {
        "documents": [{"filename": name, "chunks": n} for name, n in sorted(rows)],
        "n_documents": len(rows),
        "n_chunks": sum(n for _name, n in rows),
    }

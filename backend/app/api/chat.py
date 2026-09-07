import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.auth import User, current_user
from app.schemas.documents import SearchHitOut
from app.services.llm import LLMError
from app.services.rag import answer_question

log = structlog.get_logger()

router = APIRouter(tags=["chat"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    # Present from day one because it costs nothing here and is the hook a
    # chat session's "which documents are in scope" control will drive.
    document_ids: list[uuid.UUID] | None = None
    # None = use the MULTI_QUERY default from config. Explicit true/false lets
    # you A/B the same question in the UI.
    multi_query: bool | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchHitOut]
    # Which [n] the model actually cited. Retrieval returns top_k regardless,
    # so this is how you see how many were really used.
    sources_used: list[int]


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, user: User = Depends(current_user)) -> AskResponse:
    """Baseline RAG: one retrieval, one generation. No agent, no streaming.

    Kept after step 4 lands, as the thing to compare the agent against.
    """
    try:
        result = await answer_question(
            req.question,
            top_k=req.top_k,
            document_ids=req.document_ids,
            owner_id=user.owner_id,
            multi_query=req.multi_query,
        )
    except LLMError as exc:
        # Quota exhaustion and safety blocks both land here. A 502 is honest:
        # our request was fine, the upstream model did not answer.
        log.warning("ask_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Model error: {exc}") from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources_used=result.sources_used,
        sources=[_to_out(h) for h in result.hits],
    )


def _to_out(h) -> SearchHitOut:
    return SearchHitOut(
        chunk_id=h.chunk_id,
        document_id=h.document_id,
        filename=h.filename,
        page=h.page,
        chunk_index=h.chunk_index,
        heading=h.heading,
        text=h.text,
        score=h.score,
        meta=h.meta,
        rrf_score=h.rrf_score,
        found_by=h.found_by,
        source=h.source,
        url=h.url,
    )


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchHitOut]
    sources_used: list[int]
    # Agent-specific: how it got there.
    sub_questions: list[str]
    critique: str
    sufficient: bool
    iterations: int
    trace: list[dict]


@router.post("/research", response_model=ResearchResponse)
async def research(
    req: AskRequest, user: User = Depends(current_user)
) -> ResearchResponse:
    """The LangGraph agent: plan → retrieve → draft → critique → (loop).

    Same inputs as /ask, so the two can be compared directly on one question.
    Costs ~3 model calls, or ~5 if the critic sends it round again.
    """
    try:
        result = await run_agent(
            req.question,
            top_k=req.top_k,
            document_ids=req.document_ids,
            owner_id=user.owner_id,
            multi_query=req.multi_query,
            # Pinned so this endpoint keeps meaning what its docstring says.
            # It exists to be compared against /ask on one question, and a
            # comparison whose shape moves with a config default is not a
            # comparison. The chat UI is where ReAct is exposed.
            react=False,
        )
    except LLMError as exc:
        log.warning("research_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Model error: {exc}") from exc

    return ResearchResponse(
        question=result.question,
        answer=result.answer,
        sources=[_to_out(h) for h in result.evidence],
        sources_used=result.citations,
        sub_questions=result.sub_questions,
        critique=result.critique,
        sufficient=result.sufficient,
        iterations=result.iterations,
        trace=result.trace,
    )

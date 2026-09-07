"""The agent's state, and the reducers that merge node outputs into it.

A LangGraph node returns only the keys it changed; LangGraph merges that into
the running state. How each key merges is declared *here*, once, via
Annotated[type, reducer] -- rather than being re-implemented at every
assignment, which is what makes a hand-rolled loop over mutable dicts go bad.

Default behaviour (no reducer) is overwrite. `evidence` and `trace` accumulate.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from app.services.vectorstore import SearchHit


def merge_evidence(
    existing: list[SearchHit], incoming: list[SearchHit]
) -> list[SearchHit]:
    """Append new chunks, skipping ones already gathered.

    Plain `operator.add` would duplicate: on a retry the second retrieval
    usually re-finds some of the same chunks, and a duplicated chunk would be
    handed to the model twice -- wasting scarce prompt tokens and inflating its
    apparent importance. Dedupe by chunk_id, keeping the first occurrence so
    ordering stays best-first.
    """
    seen = {hit.chunk_id for hit in existing}
    merged = list(existing)
    for hit in incoming:
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            merged.append(hit)
    return merged


def append(existing: list, incoming: list) -> list:
    return [*existing, *incoming]


class ResearchState(TypedDict, total=False):
    # --- inputs, set once ---
    question: str
    top_k: int
    document_ids: list[str] | None
    # Set once from the authenticated caller and threaded into every retrieval,
    # so the agent can never surface another user's chunks. Carried in state
    # rather than read from a context var because nodes must stay pure
    # functions of state.
    owner_id: str | None
    multi_query: bool
    # Rolling summary + last few exchanges, pre-assembled by
    # services/history.py. Given only to `plan` and `draft`; `critique` does
    # not need it, and per-node context budgets are where the real token
    # savings are.
    chat_context: str
    # Ask the user a clarifying question when the request is too vague to
    # search on. Carried in state rather than read from settings inside the
    # node, so a single compiled graph serves both modes and the choice is per
    # REQUEST -- which is what makes it comparable in the Lab.
    clarify: bool
    # Gather evidence with the ReAct tool-calling loop instead of plan +
    # retrieve. Per REQUEST, so one compiled graph serves both strategies and
    # they stay comparable in the Lab.
    react: bool

    # --- working state ---
    # Queries the next retrieve pass should run. `plan` fills it from the
    # question; `critique` refills it with whatever was missing.
    pending_queries: list[str]
    # Everything ever asked, for the trace. Never overwritten.
    sub_questions: Annotated[list[str], append]
    # Every query already run. Guards the cycle against re-searching the same
    # thing forever -- the difference between a loop that terminates and one
    # that spends quota to learn nothing.
    tried_queries: Annotated[list[str], append]
    evidence: Annotated[list[SearchHit], merge_evidence]
    # Parts the drafter could not support. More reliable than asking the critic
    # to infer gaps, since the drafter knows what it left out.
    unanswered: list[str]

    # --- outputs ---
    draft: str
    citations: list[int]
    critique: str
    sufficient: bool
    missing: list[str]

    # --- human-in-the-loop ---
    # The question to put to the user, written by the `clarify` node:
    # {"question": str, "options": [{"label": str, "description": str}]}.
    # Written BEFORE the interrupt and read by `ask_human`, which is what keeps
    # the expensive half out of the node that replays on resume.
    pending_clarification: dict
    # What the user chose, verbatim. Absent when nothing was asked, which keeps
    # "the question was clear" distinguishable from "the user clarified it" --
    # the same reason JudgeScores fields are None-able rather than zero.
    clarification: str
    # The original wording, kept because `question` is rewritten with the
    # user's answer and the transcript should still show what they typed.
    original_question: str
    # The user declined to answer. Routes straight to END without retrieving.
    cancelled: bool

    # --- control ---
    iterations: int
    # Node-by-node record of what happened, so the UI can show the graph's
    # path. This is the payoff of an explicit state machine: the reasoning is
    # inspectable data, not buried in logs.
    trace: Annotated[list[dict], append]

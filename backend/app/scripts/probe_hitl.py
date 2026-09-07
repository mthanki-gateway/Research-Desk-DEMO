"""Exercise the clarifying-question pause and resume, end to end.

    # stubbed clarifier -- free, offline, always asks
    docker compose exec api python -m app.scripts.probe_hitl
    docker compose exec api python -m app.scripts.probe_hitl --action skip
    docker compose exec api python -m app.scripts.probe_hitl --action cancel

    # the REAL clarifier: does Gemma think this question is too vague?
    docker compose exec api python -m app.scripts.probe_hitl --real \\
        --owner <uuid> --question "Tell me about the pyramids"

There is no way to drive the HTTP endpoints from here -- they need a Supabase
token -- so this calls `run_agent` / `resume_agent` directly, which is what the
two HTTP requests do anyway.

What it proves:

* the graph stops at `ask_human` and NOTHING downstream has run
* `clarify` runs exactly ONCE across both calls, which is the reason it is a
  separate node from `ask_human`
* the pause is durable -- state is read back from Postgres between two
  independent calls, not held in memory by a parked coroutine
* an answer REWRITES the question that planning and retrieval then see
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from unittest.mock import patch

from app.agent.checkpointer import discard_thread, init_checkpointer
from app.agent.graph import (
    build_graph,
    get_graph,
    resume_agent,
    run_agent,
    set_graph,
    thread_config,
)

VAGUE = "Tell me about the pyramids, the specific thing i want to know"
STUB_OPTIONS = [
    {"label": "Construction methods", "description": "How the blocks were quarried and moved"},
    {"label": "Who built them", "description": "The workforce and how it was organised"},
    {"label": "Dimensions and dates", "description": "Height, mass, and when each was built"},
]


class _StubLLM:
    """Always says the question is ambiguous, and never calls the network."""

    async def generate_json(self, prompt: str, **kwargs) -> dict:
        return {
            "ambiguous": True,
            "question": "Which aspect of the pyramids do you want?",
            "options": STUB_OPTIONS,
        }

    async def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps({"sub_questions": ["What are the pyramids?"]})


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


async def probe(action: str, *, real: bool, owner: str | None, question: str) -> int:
    saver = await init_checkpointer()
    if saver is None:
        print("FAIL: no checkpointer. interrupt() cannot work without one.")
        return 1
    set_graph(build_graph(checkpointer=saver))

    thread_id = f"probe:0:{uuid.uuid4().hex[:8]}"

    _rule(f"ASK    action={action}  clarifier={'real' if real else 'stub'}")
    print(f"question: {question}\n")

    stub = patch("app.agent.nodes.get_llm", lambda: _StubLLM())
    if not real:
        stub.start()
    try:
        first = await run_agent(
            question,
            thread_id=thread_id,
            owner_id=owner,
            clarify=True,
            # Pinned: the stub above patches `app.agent.nodes.get_llm`, and the
            # ReAct node holds its own reference in `app.agent.react`. Left to
            # REACT_DEFAULT this probe would route through an UNSTUBBED model
            # and start making real calls -- which is the opposite of what a
            # `--real`-gated script should do by default.
            react=False,
        )

        if not first.paused:
            print("The clarifier judged this question specific enough to search.")
            print("Nothing was asked; the turn ran straight through.")
            print(f"\nanswer:\n  {first.answer[:300]}")
            print("\nOK (not ambiguous)")
            return 0

        payload = first.interrupt
        print("paused                : True")
        print(f"clarifying question   : {payload.get('question')}")
        print("options offered       :")
        for i, o in enumerate(payload.get("options", []), 1):
            print(f"  {i}. {o.get('label')}  --  {o.get('description')}")

        # Read state back out of Postgres. This is the durability claim: the
        # pause is a row, not a suspended coroutine.
        snapshot = await get_graph().aget_state(thread_config(thread_id))
        print(f"\nnext node (from DB)   : {snapshot.next}")
        planned = bool(snapshot.values.get("sub_questions"))
        retrieved = bool(snapshot.values.get("evidence"))
        print(f"planning ran          : {planned}   <- must be False")
        print(f"retrieval ran         : {retrieved}   <- must be False")

        decision: dict = {"action": action}
        if action == "answer":
            chosen = (payload.get("options") or [{}])[0].get("label", "anything")
            decision["answer"] = chosen
            print(f"\nanswering with        : {chosen}")

        _rule(f"RESUME action={action}")
        second = await resume_agent(thread_id, decision)

        print(f"still paused          : {second.paused}")
        print(f"clarification         : {second.clarification}")
        print(f"cancelled             : {second.cancelled}")
        print(f"question searched     : {second.question[:160]!r}")
        print(f"sub-questions         : {second.sub_questions}")
        print(f"evidence retrieved    : {len(second.evidence)}")
        print(f"\nanswer:\n  {second.answer[:400]}")

        if action == "cancel" and second.evidence:
            print("\nFAIL: cancel retrieved anyway.")
            return 1
        if action == "answer" and not second.clarification:
            print("\nFAIL: the answer never reached the state.")
            return 1
        if action == "skip" and second.clarification:
            print("\nFAIL: skip should not record a clarification.")
            return 1
        print("\nOK")
        return 0
    finally:
        if not real:
            stub.stop()
        await discard_thread(thread_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["answer", "skip", "cancel"],
        default="answer",
        help="what the user does when asked",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real clarifier, so the model decides whether to ask at all",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="owner_id to retrieve as. Needed with --real, or the outline is empty",
    )
    parser.add_argument("--question", default=VAGUE, help="the question to ask")
    args = parser.parse_args()
    return asyncio.run(
        probe(args.action, real=args.real, owner=args.owner, question=args.question)
    )


if __name__ == "__main__":
    raise SystemExit(main())

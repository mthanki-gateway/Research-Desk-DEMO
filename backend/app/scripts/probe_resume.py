"""Probe: what happens when the same thread_id is invoked twice?

This matters because `_prepare_turn` derives thread_id from the message count,
and a FAILED turn persists no messages -- so a user retrying gets the SAME
thread_id. If accumulator channels merge across attempts, a retry would
duplicate evidence and inflate sub_questions.

    docker compose exec api python -m app.scripts.probe_resume
"""

import asyncio

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph, initial_state, thread_config

QUESTION = "What was operating income in 2024?"
THREAD = "probe-resume-thread"


async def main() -> None:
    saver = await init_checkpointer()
    graph = build_graph(checkpointer=saver)
    config = thread_config(THREAD)

    for attempt in (1, 2):
        state = initial_state(QUESTION, top_k=2)
        final = await graph.ainvoke(state, config=config)
        print(
            f"attempt {attempt}: "
            f"evidence={len(final.get('evidence', []))} "
            f"sub_questions={len(final.get('sub_questions', []))} "
            f"tried={len(final.get('tried_queries', []))} "
            f"iterations={final.get('iterations')} "
            f"trace_steps={len(final.get('trace', []))}"
        )

    # How many checkpoints did two runs on one thread leave behind?
    count = 0
    async for _ in saver.alist(config):  # type: ignore[union-attr]
        count += 1
    print(f"checkpoints on thread: {count}")

    await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())

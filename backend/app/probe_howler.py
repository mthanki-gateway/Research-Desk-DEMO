"""A whole Howler session: a brief becomes a schema, and the schema gets filled.

The thing under test is the dynamic half. Speak and Interview have their fields
written in Python; these are generated from a sentence somebody typed, frozen
on the session, and then read back to build the tool declaration, decide
completeness and render the card.

Text turns rather than speech, because TTS quota is finite and the audio path
is proven elsewhere.

Run:  docker compose exec api python -m app.probe_howler
"""

from __future__ import annotations

import asyncio
import os

from google import genai
from google.genai import types

from app.config import get_settings
from app.services import blueprint, live, profile

BRIEF = """I want to qualify inbound leads for our B2B analytics product. Find
out their budget, when they want to go live, who signs off on purchases, and
what tooling they use today. If possible, also what made them reach out now."""

PARTICIPANT = """Priya Raman, Head of Data at a mid-sized logistics company.
Filled in our pricing form last week. Has been on a competitor's product for
two years. Technical, short on time."""

ANSWERS = [
    "Hi, yes I can hear you.",
    "We're looking at somewhere between forty and sixty thousand a year, probably.",
    "We'd want to be live by the start of Q3, ideally sooner.",
    "It'd be me and our CFO. Procurement gets involved over fifty thousand.",
    "Right now we're on Looker with a Snowflake warehouse underneath, and a lot of "
    "spreadsheets honestly.",
    "Our renewal is coming up and the pricing went up a lot. That's what prompted it.",
    "No, that's everything, thanks.",
]


async def main() -> None:
    settings = get_settings()
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"], http_options={"api_version": "v1beta"}
    )

    print("=" * 70)
    print("THE BRIEF BECOMES A SCHEMA")
    print("=" * 70)
    fields = await blueprint.from_brief(BRIEF)
    for f in fields:
        mark = "required" if f["required"] else "optional"
        print(f"  {f['name']:24} {f['type']:7} {mark}")

    chat = await live.open_conversation(None, None, "howler")
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        row = await db.get(ChatSession, chat.id)
        row.brief, row.participant, row.fields = BRIEF, PARTICIPANT, fields
        await db.commit()

    print()
    print("=" * 70)
    print("THE CONVERSATION")
    print("=" * 70)

    config = live.config("Kore", None, "howler", fields, BRIEF, PARTICIPANT)
    turns = 0

    async with client.aio.live.connect(
        model=settings.live_model.removeprefix("models/"), config=config
    ) as session:
        # The model opens, as it does behind the Start button.
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Begin. Introduce yourself in one sentence and ask "
                        "your first question."
                    )
                ],
            ),
            turn_complete=True,
        )

        for answer in [None, *ANSWERS]:
            if answer is not None:
                turns += 1
                print(f"  THEM: {answer}")
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=answer)]),
                    turn_complete=True,
                )

            said: list[str] = []
            done = False
            while not done:
                async for message in session.receive():
                    sc = message.server_content
                    if sc:
                        if sc.output_transcription and sc.output_transcription.text:
                            said.append(sc.output_transcription.text)
                        if sc.turn_complete:
                            done = True
                    if message.tool_call:
                        responses = []
                        for call in message.tool_call.function_calls:
                            response, report = await live.run_tool_call(
                                call,
                                owner_id=None,
                                top_k=5,
                                session_id=chat.id,
                                turns=turns,
                                fields=fields,
                            )
                            responses.append(response)
                            if call.name == profile.END_TOOL:
                                print(f"    [END {dict(call.args or {})}]")
                            else:
                                print(
                                    f"    [{call.name} "
                                    f"{sorted(dict(call.args or {}))}]"
                                )
                        await session.send_tool_response(function_responses=responses)
                        done = False
                if done:
                    break

            spoken = "".join(said).strip()
            if spoken:
                print(f"  HOWLER: {spoken[:170]}")
            print()

    async with SessionLocal() as db:
        final = await db.get(ChatSession, chat.id)
        stored = (final.profile if final else {}) or {}

    print("=" * 70)
    print("THE PROFILE")
    print("=" * 70)
    print(profile.render(stored, fields))


if __name__ == "__main__":
    asyncio.run(main())

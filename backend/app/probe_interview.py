"""Does the interviewer actually fill the profile, and know when it is done?

Runs a short interview as TEXT turns against the real live session, with the
real tools and the real database row. Text rather than speech because TTS quota
is finite and the audio path is proven elsewhere; what is under test is the
tool loop -- whether `record_profile` is called as things are learned, whether
the merge accumulates, and whether the model notices when nothing is missing.

Run:  docker compose exec api python -m app.probe_interview
"""

from __future__ import annotations

import asyncio
import os

from google import genai
from google.genai import types

from app.config import get_settings
from app.services import live, profile

# A participant, answering as people actually do: vaguely at first, in pieces,
# and with a correction.
ANSWERS = [
    "Hi there.",
    "I'm Sam Okafor, I'm a platform engineer at a logistics company.",
    "About five years. Sorry, six -- I forgot the contracting year.",
    "Mostly Python and Postgres. Some Kubernetes, and I've been doing a lot of Terraform lately.",
    "I'd like to get deeper into distributed systems, and I'm curious about observability.",
    "Hybrid, ideally. Two days in the office is about right.",
    # The thing that used to vanish: real information with no field to hold it.
    "One more thing -- I'm looking for something that pays well. Honestly I "
    "think I'm underpaid where I am.",
    "Nothing else really, thanks.",
]


async def main() -> None:
    settings = get_settings()
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"], http_options={"api_version": "v1beta"}
    )

    chat = await live.open_conversation(None, None, "interview")
    print(f"conversation: {chat.id}")
    print()

    # AUDIO out, as in the real app -- this exact config rejects TEXT once the
    # tools and speech settings are present. The output transcription is read
    # instead, which is what the browser shows anyway.
    config = live.config("Kore", None, "interview")

    async with client.aio.live.connect(
        model=settings.live_model.removeprefix("models/"), config=config
    ) as session:
        for answer in ANSWERS:
            print(f"  PARTICIPANT: {answer}")
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
                            )
                            responses.append(response)
                            if call.name == profile.END_TOOL:
                                print(f"    [END_INTERVIEW {dict(call.args or {})}]")
                            elif call.name == profile.TOOL_NAME:
                                print(
                                    f"    [record_profile {sorted(dict(call.args or {}))}"
                                    f" -> missing {report['missing']}]"
                                )
                            else:
                                print(f"    [{call.name}]")
                        await session.send_tool_response(function_responses=responses)
                        done = False
                if done:
                    break

            spoken = "".join(said).strip()
            if spoken:
                print(f"  INTERVIEWER: {spoken[:200]}")
            print()

    # What ended up stored.
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        final = await db.get(ChatSession, chat.id)
        stored = (final.profile if final else {}) or {}

    print("=" * 66)
    print("STORED PROFILE")
    print("=" * 66)
    for key, value in stored.items():
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else value
        print(f"  {key:18} {shown}")
    print()
    print(f"  missing  : {profile.missing(stored) or 'nothing'}")
    print(f"  complete : {profile.complete(stored)}")


if __name__ == "__main__":
    asyncio.run(main())

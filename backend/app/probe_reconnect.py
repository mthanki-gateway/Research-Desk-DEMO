"""Does a reconnect restore the conversation? Measured at the SDK seam.

A live session's history lives inside the socket. When the socket dies the
conversation dies with it -- measured: on a fresh session the model correctly
says it has no access to anything said before. Session resumption is meant to
make that invisible.

Three connections, and the third is the one that matters:

    1  state a fact, keep the handle
    2  NEW connection, resuming  -> must remember
    3  NEW connection, no handle -> must NOT remember

Without the control, a model that simply guessed "Red" would look exactly like
working resumption.

TEXT turns rather than speech, deliberately: TTS quota is finite and the audio
path is proven elsewhere. What is under test is whether CONTEXT survives a
reconnection, which is independent of how the turn arrived.

Run:  docker compose exec api python -m app.probe_reconnect
"""

from __future__ import annotations

import asyncio
import os

from google import genai
from google.genai import types

from app.config import get_settings
from app.services import live

FACT = "Remember this: my favourite pyramid is the Red Pyramid."
RECALL = "Which pyramid did I say was my favourite? Answer in one sentence."


async def one_turn(client, model, config, text: str) -> tuple[str, str | None]:
    """Open a connection, say one thing, return (transcript, resume handle)."""
    said: list[str] = []
    handle: str | None = None

    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )
        async for message in session.receive():
            sc = message.server_content
            if sc and sc.output_transcription and sc.output_transcription.text:
                said.append(sc.output_transcription.text)
            update = message.session_resumption_update
            if update and update.resumable and update.new_handle:
                # Collected as it arrives. The update can land at any point in a
                # turn, and draining for it afterwards blocks for ever when it
                # has already been and gone.
                handle = update.new_handle
            if sc and sc.turn_complete:
                break

    return "".join(said).strip(), handle


async def main() -> None:
    settings = get_settings()
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"], http_options={"api_version": "v1beta"}
    )
    model = settings.live_model.removeprefix("models/")

    print("=" * 68)
    print("1  state the fact")
    said, handle = await one_turn(client, model, live.config("Kore"), FACT)
    print("   said  :", said[:150])
    print("   handle:", f"{handle[:30]}..." if handle else "NONE ISSUED")
    if not handle:
        print()
        print("No handle issued, so there is nothing to resume.")
        return

    print()
    print("=" * 68)
    print("2  NEW connection, resuming with the handle")
    said, _ = await one_turn(client, model, live.config("Kore", handle), RECALL)
    print("   said  :", said[:200])
    print("   -> RESTORED:", "red" in said.lower())

    print()
    print("=" * 68)
    print("3  NEW connection, NO handle -- the control")
    said, _ = await one_turn(client, model, live.config("Kore"), RECALL)
    print("   said  :", said[:200])
    print("   -> remembers anyway (should be False):", "red" in said.lower())


if __name__ == "__main__":
    asyncio.run(main())

"""Audio to audio, natively. Parley's engine.

WHAT CHANGED, AND WHY IT MATTERS

Parley began as a CASCADE: speech recognised to text, text answered by the
LangGraph agent, text synthesised back to speech. Three models with text in the
middle. It worked, and it was measured at 12 to 53 seconds a turn.

This is the same app on a NATIVE AUDIO model. There is no transcript in the
middle. The audio is tokenised into the same sequence the model generates from,
and what comes back is audio tokens rather than words that later get spoken.
Measured on this deployment, same question, same corpus:

    cascade          12-53s to the first sound
    gemini-3.8-live   2.0s to the first sound

The difference is not only speed. A transcript destroys everything about HOW
something was said -- hesitation, a question's rising intonation, someone
trailing off -- and those never reach a cascade's agent at all. They reach this
one, because the waveform is the input.

WHAT WE GIVE UP

The LangGraph agent. Its planner, critic, retry loop and citation discipline
do not exist here -- the live model decides for itself when to search and what
to say. That is a real loss of rigour, and it is why the Research Desk keeps
the graph: reading an answer you can check beats hearing one you cannot.

What survives is the part that matters most: THE SAME TOOLS ON THE SAME
CORPUS. `search_documents`, `search_web`, `list_documents` and `corpus_stats`
are executed here by `app.agent.tools.run_tool` -- the identical code path the
typed agent uses, against the identical Qdrant collection. A document uploaded
in the Library is answerable out loud the moment it finishes indexing.

TWO THINGS THAT ARE NOT OBVIOUS AND COST A DAY EACH IF MISSED

1. The turn does not end without TRAILING SILENCE. Live decides the speaker
   has stopped by hearing them stop. Audio that ends on the last word gives it
   nothing to detect, and `audio_stream_end` does NOT substitute -- measured,
   the model accepted five seconds of clear speech, reported no transcript and
   never replied at all. One second of appended silence fixes it completely.

2. `session.receive()` ENDS AT A TOOL CALL. The spoken answer arrives on the
   next generator. Treating the first end as the end of the turn yields a tool
   call and zero audio, which looks exactly like a model that refused to speak.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import types

from app.agent import tools as agent_tools
from app.config import get_settings
from app.services import websearch

log = structlog.get_logger()

# What Live expects in and produces out. Neither is negotiable.
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000

# Kept, though the app no longer needs it.
#
# It was the workaround for AUTOMATIC activity detection, which decides a turn
# has ended by hearing the speaker stop -- and which this app now disables
# outright, because a person pausing mid-sentence is not a person who has
# finished. With manual activity control the turn ends when `activity_end` is
# sent, and silence means nothing at all.
#
# Left here because anything that re-enables automatic detection needs it
# again, and the failure without it is completely silent: the model accepts the
# audio, reports no transcript, and never replies.
TRAILING_SILENCE = bytes(INPUT_RATE * 2)

# Tools the live model may call.
#
# A SUBSET of the typed agent's seven, chosen for what makes sense spoken.
# `remember_preference` is omitted because a voice turn here is not part of a
# conversation with stored memory, and `read_around` because following a
# passage's neighbours is a reading behaviour -- a listener cannot hold three
# adjacent chunks in their head to compare them.
SPOKEN_TOOLS = ("search_documents", "search_web", "list_documents", "corpus_stats")

SYSTEM = """You are Parley, a research assistant that is LISTENED TO rather \
than read. Everything you say is spoken aloud and heard once.

YOUR TOOLS ARE THE POINT. You have the user's own uploaded documents and the \
public web. Before answering any factual question, search. Never answer a \
question about their material from memory -- you have not read their documents, \
you can only search them.

- search_documents: their private material. Use it first for anything about \
their reports, incidents, handbooks or transcripts.
- search_web: public knowledge, definitions, current events, anything not \
theirs. Use it ALONGSIDE the documents when a question spans both.
- list_documents: what they actually have, and what it covers. Use it for \
"what do you have", and before claiming something is not in their documents.
- corpus_stats: counts and sizes of the collection as a whole.

HOW TO SPEAK

Be brief. Aim for under eighty words. A listener cannot skim, so lead with the \
answer and stop.

Name your sources in words -- "your engineering handbook says", "according to \
the incident report". Never say a citation number; there is nothing on screen \
to match it to.

Never refer to anything visual: no "above", no "below", no "as listed", no \
"see the table".

Say numbers as they are spoken: "sixty four passages", "the eleventh of \
November".

If you searched and found nothing, say that plainly and say where you looked. \
Do not invent a plausible answer -- being wrong out loud is worse than being \
wrong in text, because there is nothing to re-read."""


def enabled() -> bool:
    return bool(get_settings().google_api_key)


def _declarations() -> list[types.FunctionDeclaration]:
    """Our tool specs, translated into the Live SDK's types.

    Built from `agent_tools.tool_specs()` rather than written out again, so a
    description improved for the typed agent improves here too. Two copies of a
    tool description drift, and a drifted description is a model that calls the
    wrong tool for reasons nobody can see.
    """
    declared = agent_tools.tool_specs()[0]["functionDeclarations"]
    out = []
    for spec in declared:
        if spec["name"] not in SPOKEN_TOOLS:
            continue
        params = spec.get("parameters") or {}
        properties = {
            name: types.Schema(
                type=(field.get("type") or "STRING").upper(),
                description=field.get("description"),
            )
            for name, field in (params.get("properties") or {}).items()
        }
        out.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=(
                    types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=params.get("required") or [],
                    )
                    if properties
                    # A tool with no arguments must declare NO schema at all.
                    # An empty OBJECT is rejected by the API.
                    else None
                ),
            )
        )
    return out


def config(voice: str, resume: str | None = None) -> types.LiveConnectConfig:
    reach = (
        ""
        if websearch.enabled()
        else (
            "\n\nWEB SEARCH IS NOT CONFIGURED on this deployment. You have only "
            "the user's documents. If a question needs public information, say "
            "that it is not in their documents AND that you cannot search the "
            "web -- never imply the information does not exist."
        )
    )
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        tools=[types.Tool(function_declarations=_declarations())],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM + reach)]),
        # Both transcripts are requested purely so the SCREEN can show what was
        # heard and said. They are a side output, not the mechanism -- the model
        # is not reading them, and nothing in the answer path depends on them.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # THE BUTTON OWNS THE TURN. Automatic detection is switched off.
        #
        # With it on, the model answers whenever it hears a pause -- and people
        # pause constantly while speaking: to think, to find a word, to check a
        # figure. Every one of those was read as "they have finished", so the
        # assistant talked over the second half of the question. Tuning the
        # silence threshold only moves the problem: short enough to feel
        # responsive is short enough to interrupt, and long enough never to
        # interrupt is long enough to feel broken.
        #
        # Disabled, the turn begins at `activity_start` and ends at
        # `activity_end`, both sent when the user clicks. Nothing about the
        # audio itself can end a turn, so a ten-second pause mid-question is
        # just part of the question.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            )
        ),
        # SESSION RESUMPTION. The conversation survives a dropped socket.
        #
        # A live session's history lives SERVER-SIDE, inside the connection --
        # we send no transcript and no prior turns. Measured: within one socket
        # the model recalls turn 1 at turn 3; on a fresh socket it correctly
        # says it has no access to anything said before.
        #
        # That matters because the socket closes on its own. Gemini drops an
        # idle one, and it announces a hard lifetime cap through `go_away`. So
        # without this, a conversation with a pause in the middle silently
        # forgets everything before the pause, and the user is given no sign
        # that it happened -- the worst kind of failure, because the assistant
        # goes on answering confidently from nothing.
        #
        # Passing a handle back restores the server-side context. An empty
        # config on a fresh session ASKS for handles without resuming anything.
        session_resumption=types.SessionResumptionConfig(handle=resume)
        if resume
        else types.SessionResumptionConfig(),
    )


def client() -> genai.Client:
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Parley.")
    return genai.Client(
        api_key=settings.google_api_key, http_options={"api_version": "v1beta"}
    )


async def run_tool_call(
    call: Any, *, owner_id: str | None, top_k: int
) -> tuple[types.FunctionResponse, dict]:
    """Execute one tool the live model asked for, on the real corpus.

    Routed through `agent_tools.run_tool`, which is the SAME function the typed
    agent uses. Not a reimplementation: a second retrieval path would diverge,
    and the divergence would show up as the voice app answering differently
    from the chat app about the same document.

    Never raises. A tool that throws would kill the session mid-sentence; the
    model can recover from "that failed" by saying so or trying another tool.
    """
    name = call.name
    args = dict(call.args or {})
    try:
        hits, observation = await agent_tools.run_tool(
            name,
            args,
            top_k=top_k,
            document_ids=None,
            owner_id=owner_id,
            session_id=None,
        )
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the call
        log.warning("live_tool_failed", tool=name, error=str(exc))
        hits, observation = [], f"The {name} tool failed: {type(exc).__name__}."

    # Sources are reported to the BROWSER separately, not squeezed into the
    # model's observation. The model should hear the content; the screen should
    # show where it came from, because a spoken citation cannot be clicked.
    sources = []
    seen: set[str] = set()
    for hit in hits:
        kind = getattr(hit, "source", "document")
        key = (hit.url or "") if kind == "web" else str(hit.document_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {"label": hit.filename, "kind": kind, "url": getattr(hit, "url", None)}
        )

    return (
        types.FunctionResponse(id=call.id, name=name, response={"result": observation}),
        {"tool": name, "args": args, "n": len(hits), "sources": sources},
    )


async def frames(audio: bytes, *, size: int = INPUT_RATE * 2 // 10) -> AsyncIterator[bytes]:
    """100ms frames, which is roughly what a microphone produces."""
    for i in range(0, len(audio), size):
        yield audio[i : i + size]
        # Yield to the loop between frames so a long buffer does not starve
        # everything else on the connection.
        await asyncio.sleep(0)

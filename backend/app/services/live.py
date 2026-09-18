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
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import types

from app.agent import tools as agent_tools
from app.config import get_settings
from app.services import live_prompts, profile, websearch

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
# Which of the agent's tools each mode may call.
#
# INTERVIEW GETS NO RETRIEVAL. It is not answering questions about the corpus,
# it is asking them about a person, and a tool that is offered will eventually
# be used -- an interviewer with document search reaches for it and starts
# explaining the user's own files back at them instead of asking anything. The
# web stays, because a passing mention of a company or a technology is worth
# being able to place.
#
# Neither mode gets `remember_preference` (a voice turn is not part of a
# conversation with stored preferences) or `read_around` (following a passage's
# neighbours is a reading behaviour; a listener cannot hold three adjacent
# chunks in their head).
SPOKEN_TOOLS: dict[str, tuple[str, ...]] = {
    "speak": ("search_documents", "search_web", "list_documents", "corpus_stats"),
    "interview": ("search_web",),
    # Same as Interview. Howler differs in WHAT it gathers, never in how.
    "howler": ("search_web",),
}

# Which prompt, and which kind of stored conversation, each mode uses.
#
# ONE PIPELINE, TWO PROMPTS. Speak and Interview share every piece of
# machinery: the socket, the audio handling, the turn boundaries, the tools,
# the persistence. What differs is what the model is told it is for -- and that
# is genuinely the only difference, which is why it is a lookup rather than a
# second implementation.
MODES: dict[str, dict[str, str]] = {
    "speak": {"system": live_prompts.SPEAK, "kind": "parley"},
    "interview": {"system": live_prompts.INTERVIEW, "kind": "interview"},
    # Howler's prompt carries {brief} and {participant} slots, filled per
    # session in `config`. The others have no slots and are used as they are.
    "howler": {"system": live_prompts.HOWLER, "kind": "howler"},
}
DEFAULT_MODE = "speak"

# Every kind a CONVERSATION can be, as opposed to a project.
#
# Listed once because three separate endpoints check it, and when Howler was
# added they were all still checking for two -- so every Howler session loaded
# as "Conversation not found" while the socket that created it worked perfectly.
CONVERSATION_KINDS = ("parley", "interview", "howler")


def mode_of(name: str) -> str:
    """The requested mode, or the default. Never raises on a bad value."""
    return name if name in MODES else DEFAULT_MODE


def kind_of(mode: str) -> str:
    return MODES[mode_of(mode)]["kind"]


def enabled() -> bool:
    return bool(get_settings().google_api_key)


def _declarations(
    mode: str = DEFAULT_MODE, fields: list[dict] | None = None
) -> list[types.FunctionDeclaration]:
    """Our tool specs, translated into the Live SDK's types.

    Built from `agent_tools.tool_specs()` rather than written out again, so a
    description improved for the typed agent improves here too. Two copies of a
    tool description drift, and a drifted description is a model that calls the
    wrong tool for reasons nobody can see.
    """
    allowed = SPOKEN_TOOLS[mode_of(mode)]
    declared = list(agent_tools.tool_specs()[0]["functionDeclarations"])
    # The interview's own tool, which belongs to no agent: it records rather
    # than retrieves, and its result is what tells the interviewer what to ask
    # next. Declared here rather than in `agent_tools` because the typed agent
    # has no use for it.
    if mode_of(mode) in ("interview", "howler"):
        # Howler passes the schema generated from its brief; Interview passes
        # nothing and gets the built-in one.
        declared.append(profile.declaration(fields))
        declared.append(profile.end_declaration())

    out = []
    for spec in declared:
        if spec["name"] not in allowed and spec["name"] not in (
            profile.TOOL_NAME,
            profile.END_TOOL,
        ):
            continue
        params = spec.get("parameters") or {}
        properties = {
            name: types.Schema(
                type=(field.get("type") or "STRING").upper(),
                description=field.get("description"),
                # ARRAY MUST DECLARE ITS ITEM TYPE. Omitting it is not a
                # tolerated default -- the API rejects the entire setup
                # message, so the session never opens and the whole mode is
                # dead rather than one tool being broken:
                #   function_declarations[1].parameters.properties[interests]
                #   .items: missing field
                items=(
                    types.Schema(
                        type=((field.get("items") or {}).get("type") or "STRING").upper()
                    )
                    if (field.get("type") or "").upper() == "ARRAY"
                    else None
                ),
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


def _system(
    mode: str, brief: str, participant: str, vocabulary: list[str] | None = None
) -> str:
    """The system prompt for this mode, with Howler's slots filled.

    `str.format` is not used: the prompts contain braces of their own in
    examples, and one stray pair turns the whole instruction into a KeyError at
    connect time -- which fails the session rather than the formatting.
    """
    text = MODES[mode_of(mode)]["system"]
    if mode_of(mode) != "howler":
        return text
    terms = ", ".join(vocabulary or [])
    return (
        text.replace("{brief}", brief.strip() or "(no brief given)")
        .replace(
            "{participant}",
            participant.strip() or "(nothing known about them yet)",
        )
        .replace("{vocabulary}", terms or "(none listed)")
    )


def config(
    voice: str,
    resume: str | None = None,
    mode: str = DEFAULT_MODE,
    fields: list[dict] | None = None,
    brief: str = "",
    participant: str = "",
    vocabulary: list[str] | None = None,
) -> types.LiveConnectConfig:
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
        tools=[types.Tool(function_declarations=_declarations(mode, fields))],
        system_instruction=types.Content(
            parts=[
                types.Part(text=_system(mode, brief, participant, vocabulary) + reach)
            ]
        ),
        # Both transcripts are requested purely so the SCREEN can show what was
        # heard and said. They are a side output, not the mechanism -- the model
        # is not reading them, and nothing in the answer path depends on them.
        input_audio_transcription=types.AudioTranscriptionConfig(
            # SPELLINGS FOR THE MICROPHONE.
            #
            # "React", "Node.js", "GCP" and "Kubernetes" are exactly the words a
            # recogniser is worst at and exactly the ones that matter in the
            # answer -- a profile that says "angular JS" or hears "jeep" for
            # "GCP" is wrong in the only part anybody reads it for.
            #
            # `custom_vocabulary` biases the ASR towards these phrases.
            # `adaptation_phrases` does the same thing and is deprecated, so it
            # is deliberately not used. The list is generated per project, from
            # ITS subject: a mythology interview gets Yggdrasil and Snorri
            # Sturluson, and feeding it a list of web frameworks would bias the
            # recogniser towards words nobody is going to say.
            #
            # The same terms also go into the system prompt, because this
            # setting steers the TRANSCRIPT and the prompt steers the model's
            # own understanding -- and it is the model, not the transcript,
            # that writes the profile.
            custom_vocabulary=list(vocabulary) if vocabulary else None,
        ),
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


async def record_profile(
    session_id: uuid.UUID,
    args: dict,
    turns: int = 0,
    fields: list[dict] | None = None,
) -> tuple[str, dict, bool]:
    """Fold what the model just learned into the stored profile.

    Returns (what to tell the model, the merged profile, whether it is
    complete). The FIRST of those is the important one: it lists the fields
    still missing, which is how the interviewer knows what to ask next without
    re-deriving it from a growing transcript.

    Read-modify-write on one row. Two calls racing would lose one update, which
    cannot happen here -- a live session has one model producing one tool call
    at a time -- and a lock across a network round trip would cost more than
    the problem it prevents.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        chat = await db.get(ChatSession, session_id)
        if chat is None:
            return "The profile could not be saved.", {}, False
        merged = profile.merge(chat.profile or {}, args, fields)
        chat.profile = merged
        await db.commit()

    await name_conversation(session_id, merged)

    done = profile.complete(merged, fields)
    if done:
        await _mark_completed(session_id, turns)
    log.info(
        "profile_recorded",
        fields=sorted(args),
        missing=profile.missing(merged, fields),
        complete=done,
    )
    return profile.render(merged, fields), merged, done


async def run_tool_call(
    call: Any,
    *,
    owner_id: str | None,
    top_k: int,
    session_id: uuid.UUID | None = None,
    turns: int = 0,
    fields: list[dict] | None = None,
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

    # The model declaring the conversation over. Nothing is retrieved and
    # nothing is asked; the browser stops reopening the microphone.
    if name == profile.END_TOOL:
        # REFUSED WHEN THE CLOSING QUESTION HAS NOT BEEN ANSWERED.
        #
        # The prompt says to ask whether there is anything to add and wait for
        # the reply. Observed doing neither: it asked "is there anything else
        # you would like to share about your career goals or background?" and
        # ended in the same breath, so the participant was shown a closed
        # session instead of a chance to answer.
        #
        # The instruction was not enough, so this is structural. `turns` counts
        # what the participant has said; an end arriving on the same turn the
        # profile completed is refused, with a tool result telling the model
        # plainly to wait. It may end on the next turn.
        if session_id is not None and not await _may_end(session_id, turns, fields):
            log.info("end_interview_too_early", turns=turns)
            return (
                types.FunctionResponse(
                    id=call.id,
                    name=name,
                    response={
                        "result": (
                            "NOT YET. You have just asked whether they want to "
                            "add anything, and they have not answered. Wait for "
                            "their reply, record it, and only then end."
                        )
                    },
                ),
                {"tool": name, "args": args, "n": 0, "sources": [], "ended": False},
            )

        summary = str(args.get("summary") or "").strip()
        moments = [
            " ".join(str(m).split())
            for m in (args.get("notable_moments") or [])
            if str(m).strip()
        ]
        affect = {
            "demeanour": " ".join(str(args.get("demeanour") or "").split()),
            "moments": moments[:8],
        }
        # STORED EVEN WITH NO SUMMARY. `ended` is set inside `store_summary`,
        # and gating the whole call on a summary meant a model that ended
        # without one left the conversation open for ever -- the link stayed
        # live and the card never said complete.
        if session_id is not None:
            await store_summary(session_id, summary, affect)
        return (
            types.FunctionResponse(
                id=call.id,
                name=name,
                response={
                    "result": (
                        "The interview is closed. Say a brief goodbye and stop "
                        "asking questions."
                    )
                },
            ),
            {"tool": name, "args": args, "n": 0, "sources": [], "ended": True,
             "summary": summary},
        )

    # Not a retrieval tool: it writes, and its result steers the next question.
    if name == profile.TOOL_NAME:
        if session_id is None:
            return (
                types.FunctionResponse(
                    id=call.id, name=name, response={"result": "No conversation to save to."}
                ),
                {"tool": name, "args": args, "n": 0, "sources": []},
            )
        observation, merged, done = await record_profile(
            session_id, args, turns, fields
        )
        return (
            types.FunctionResponse(id=call.id, name=name, response={"result": observation}),
            {
                "tool": name,
                "args": args,
                "n": 0,
                "sources": [],
                # Sent on to the browser so the profile can fill in on screen as
                # it is gathered, rather than appearing only once it is done.
                "profile": merged,
                "missing": profile.missing(merged, fields),
                "complete": done,
            },
        )

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


# ---------------------------------------------------------------------------
# Persistence
#
# Parley conversations live in the SAME tables as the Research Desk's, marked
# with `kind="parley"`. One table rather than two: a conversation is a
# conversation -- same owner scoping, same message shape, same deletion -- and
# what differs is only how the turns arrived.
#
# The transcripts are a SIDE OUTPUT here, not the mechanism. The model is not
# reading them back; they exist so the screen can show what was said and so a
# conversation can be looked at again tomorrow. The live context itself is
# restored by the resumption handle, which is a different thing entirely and is
# stored on the session.
# ---------------------------------------------------------------------------


async def open_conversation(
    owner_id: str | None, session_id: str | None, kind: str = "parley"
):
    """Find the conversation to append to, or start one."""

    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        if session_id:
            chat = await db.get(ChatSession, uuid.UUID(session_id))
            # Ownership is checked HERE rather than trusted from the client: a
            # session id in a query string is a guess anyone can make.
            if chat is not None and chat.owner_id == owner_id and chat.kind == kind:
                return chat

        chat = ChatSession(
            title=UNNAMED_INTERVIEW if kind == "interview" else "Spoken conversation",
            owner_id=owner_id,
            kind=kind,
        )
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return chat


async def save_turn(
    session_id: uuid.UUID,
    question: str,
    answer: str,
    sources: list[dict],
    tools: list[str],
) -> None:
    """Append one spoken exchange. Never raises.

    A failure to write the transcript must not end the call -- the
    conversation is happening in the socket, and losing a row is a smaller harm
    than dropping someone mid-sentence to report it.
    """

    from app.db.models import ChatSession, Message, Role
    from app.db.session import SessionLocal

    if not question and not answer:
        return

    try:
        async with SessionLocal() as db:
            db.add(
                Message(
                    session_id=session_id,
                    role=Role.user,
                    content=question or "(nothing intelligible)",
                    sources=[],
                    agent_meta={"spoken": True},
                )
            )
            db.add(
                Message(
                    session_id=session_id,
                    role=Role.assistant,
                    content=answer,
                    sources=sources,
                    agent_meta={"spoken": True, "tools": tools},
                )
            )

            # The first real question becomes the title, for SPEAK only.
            # "Spoken conversation" tells a list of conversations nothing at
            # all.
            #
            # An interview is named after the PERSON instead -- see
            # `name_conversation`. Titling it from the opening utterance gives
            # a list of rows reading "Hi there" and "Hello, can you hear me",
            # and worse, that utterance is the one the transcriber mangles most
            # because nobody has warmed up yet.
            chat = await db.get(ChatSession, session_id)
            if (
                chat is not None
                and question
                and chat.kind == "parley"
                and chat.title == "Spoken conversation"
            ):
                chat.title = question[:120]

            await db.commit()
    except Exception as exc:  # noqa: BLE001 - never interrupt a live call
        log.warning("live_save_turn_failed", error=str(exc)[:200])


# The placeholder an interview carries until it learns whose it is.
UNNAMED_INTERVIEW = "Interview"


async def name_conversation(session_id: uuid.UUID, profile_data: dict) -> None:
    """Title an interview after its participant. Never raises.

    Applied AS SOON AS the name is recorded rather than only at the end, so the
    drawer stops reading "Interview, Interview, Interview" while one is still
    running -- which is exactly when you need to tell them apart.

    Only replaces the placeholder. A title that is anything else was either set
    by a person or already carries a name, and overwriting either would be the
    app arguing with the user about what to call their own conversation.
    """
    name = profile.person_name(profile_data or {})
    if not name:
        return

    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is not None and chat.title == UNNAMED_INTERVIEW:
                chat.title = name[:120]
                await db.commit()
                log.info("interview_named", name=name[:60])
    except Exception as exc:  # noqa: BLE001 - a title is never worth a failure
        log.warning("live_name_failed", error=str(exc)[:200])


async def _may_end(
    session_id: uuid.UUID, turns: int, fields: list[dict] | None = None
) -> bool:
    """Has the participant spoken since the profile was completed?

    The turn on which the last required field lands is the same turn the model
    announces it has everything and asks whether there is anything to add. The
    answer to THAT is frequently the most useful thing in the profile, because
    it is the only part the participant chose -- so ending on that turn throws
    away the one thing they volunteered.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            completed_on = (chat.profile or {}).get("completed_on") if chat else None
        # No record of completion means there is nothing to wait for. Never
        # block an ending on missing bookkeeping.
        return completed_on is None or turns > int(completed_on)
    except Exception as exc:  # noqa: BLE001 - a read must not trap the model
        log.warning("may_end_check_failed", error=str(exc)[:200])
        return True


async def _mark_completed(session_id: uuid.UUID, turns: int) -> None:
    """Remember the turn on which the profile first became complete."""
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is None or (chat.profile or {}).get("completed_on") is not None:
                return
            merged = dict(chat.profile or {})
            merged["completed_on"] = turns
            chat.profile = merged
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("mark_completed_failed", error=str(exc)[:200])


async def adopt_project(
    session_id: uuid.UUID,
    project_id: uuid.UUID,
    brief: str,
    participant: str,
    fields: list[dict],
    invite_id: uuid.UUID | None = None,
    vocabulary: list[str] | None = None,
    title: str = "",
) -> None:
    """Stamp a guest's conversation with the project it belongs to.

    Only on a conversation that has not already been stamped: a link reused
    after a dropped call must find the SAME session with the same schema, not
    have it rewritten underneath a half-filled profile.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal
    from app.services import invites

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is None or chat.project_id is not None:
                return
            chat.project_id = project_id
            chat.brief = brief
            chat.participant = participant
            chat.fields = fields
            # Snapshotted with the schema, and for the same reason: the
            # spellings are baked into the transcription config when the socket
            # opens, so a session already running keeps the ones it started
            # with however the project changes underneath it.
            chat.vocabulary = vocabulary or []
            # NAME IT NOW, from what the link already knows.
            #
            # Howler conversations used to inherit Speak's default and sit in
            # the drawer as four identical "Spoken conversation" rows. The
            # usual renamer cannot help: it looks for a name FIELD, and a
            # generated schema often has none -- the participant's name ends up
            # in a note instead. The link's own label, or failing that the
            # project, is known before anybody speaks.
            if title and chat.title in ("Spoken conversation", UNNAMED_INTERVIEW):
                chat.title = title[:120]
            await db.commit()
        if invite_id is not None:
            await invites.attach_session(invite_id, session_id)
    except Exception as exc:  # noqa: BLE001 - never fail a call over bookkeeping
        log.warning("adopt_project_failed", error=str(exc)[:200])


async def store_summary(
    session_id: uuid.UUID, summary: str, affect: dict | None = None
) -> None:
    """Close the interview: the one-line summary, and how it sounded.

    Never raises. Both are written here because both arrive on the same tool
    call, and this is also where `ended` is set -- see the caller for why that
    must not depend on the model having bothered with a summary.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is not None:
                merged = dict(chat.profile or {})
                if summary:
                    merged["summary"] = summary
                if affect and (affect.get("demeanour") or affect.get("moments")):
                    merged["affect"] = affect
                merged["ended"] = True
                chat.profile = merged
                # Last chance to name it. If `record_profile` never carried a
                # name -- it can be learned and then corrected -- this is the
                # point at which the profile is final.
                name = profile.person_name(merged)
                if name and chat.title in DEFAULT_TITLES:
                    chat.title = name[:120]
                else:
                    await name_from_context(db, chat)
                owner = getattr(chat, "owner_id", None)
                await db.commit()
        await _queue_analysis(session_id, owner)
    except Exception as exc:  # noqa: BLE001
        log.warning("live_store_summary_failed", error=str(exc)[:200])


async def finish_interview(session_id: uuid.UUID) -> None:
    """The PARTICIPANT closed the interview. Never raises.

    Recorded separately from the model closing it, because the two mean
    opposite things about the result. The model ends when it has what it came
    for; a person ends when they have had enough -- and a half-filled profile
    that somebody walked out of is a different finding from one the interviewer
    could not get answers to. `ended_by` is what tells them apart afterwards,
    and without it both look like "complete".

    No summary is written. The summary is the model's account of the person,
    and it has not finished forming one.
    """
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is None:
                return
            merged = dict(chat.profile or {})
            # IT RUNS AFTER THE CLOSING PASS, and must not undo it.
            #
            # Pressing stop now asks the live model to call `end_interview`
            # first, so by the time this runs the profile may already carry a
            # summary, a demeanour and the notable moments -- written by the
            # only thing that heard the audio. Those are kept; `ended_by` is
            # recorded alongside them, because the participant is still who
            # ended it and a half-filled profile somebody walked out of is a
            # different finding from one the interviewer finished.
            #
            # Not set twice: an interview the model had ALREADY closed on its
            # own, before anybody pressed anything, keeps its own ending.
            if merged.get("ended") and merged.get("ended_by"):
                return
            merged["ended"] = True
            merged.setdefault("ended_by", "participant")
            chat.profile = merged
            name = profile.person_name(merged)
            if name and chat.title in DEFAULT_TITLES:
                chat.title = name[:120]
            else:
                await name_from_context(db, chat)
            owner = getattr(chat, "owner_id", None)
            await db.commit()
        await _queue_analysis(session_id, owner)
        log.info("interview_ended_by_participant", session=str(session_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("live_finish_failed", error=str(exc)[:200])


DEFAULT_TITLES = ("Spoken conversation", UNNAMED_INTERVIEW, "")


async def name_from_context(db, chat) -> None:
    """Name a conversation from what surrounds it, if it still has no name.

    THE LAST FALLBACK, after `person_name`. That one looks for a name FIELD,
    and Howler's schemas are generated -- so a participant's name usually lands
    in a note where it cannot be found, and the conversation keeps the default
    for ever.

    Everything else about it is already known: the link was made for somebody,
    or the project it belongs to has a title. Either beats a row that reads
    "Spoken conversation" next to three others saying the same.

    Runs INSIDE the caller's session and does not commit -- it is one more
    field on a write that was happening anyway.
    """
    if chat.title not in DEFAULT_TITLES:
        return

    from sqlalchemy import select

    from app.db.models import HowlerInvite, HowlerProject

    label = (
        await db.execute(
            select(HowlerInvite.label).where(HowlerInvite.session_id == chat.id)
        )
    ).scalar_one_or_none()
    if label:
        chat.title = label[:120]
        return

    if chat.project_id:
        title = (
            await db.execute(
                select(HowlerProject.title).where(HowlerProject.id == chat.project_id)
            )
        ).scalar_one_or_none()
        if title:
            chat.title = title[:120]


async def _queue_analysis(session_id: uuid.UUID, owner_id: str | None) -> None:
    """Queue the emotion pass for a finished interview. Never raises.

    ENQUEUED WHETHER OR NOT ANYTHING CAN RUN IT YET. The job is the record that
    this interview is waiting for analysis, so a deployment too small for the
    model still collects the work and anything that can run it finds it later.
    Only queueing when the feature happens to be switched on would silently
    lose every interview held before that.
    """
    from app.services import jobs
    from app.services.analysis import KIND, SUBJECT

    await jobs.enqueue(
        KIND,
        {"session_id": str(session_id)},
        subject_type=SUBJECT,
        subject_id=session_id,
        owner_id=owner_id,
    )


async def store_handle(session_id: uuid.UUID, handle: str) -> None:
    """Remember how to resume this conversation. Never raises."""
    from app.db.models import ChatSession
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            chat = await db.get(ChatSession, session_id)
            if chat is not None:
                chat.live_handle = handle
                await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("live_store_handle_failed", error=str(exc)[:200])

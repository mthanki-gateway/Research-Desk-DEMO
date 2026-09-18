"""The native-audio path, and the two things about it that fail silently.

Neither raises. Both produce a session that connects, accepts audio, and then
sits there — which in an audio-only app is indistinguishable from a crash.
"""

import asyncio
import uuid

import pytest

from app.services import live


class TestToolsAreShared:
    """The live model's tools are the TYPED AGENT'S tools, not copies.

    Two descriptions of the same tool drift, and a drifted description is a
    model that calls the wrong one for reasons nobody can see from the outside.
    Built from `agent_tools.tool_specs()` so an improvement in one place is an
    improvement in both.
    """

    def test_the_declarations_come_from_the_agent(self):
        from app.agent import tools as agent_tools

        source = {
            spec["name"]: spec["description"]
            for spec in agent_tools.tool_specs()[0]["functionDeclarations"]
        }
        for declaration in live._declarations("speak"):
            assert declaration.description == source[declaration.name]

    def test_only_the_spoken_subset_is_offered(self):
        names = {d.name for d in live._declarations("speak")}
        assert names <= set(live.SPOKEN_TOOLS["speak"])
        # The two that make no sense out loud.
        assert "remember_preference" not in names
        assert "read_around" not in names

    def test_speak_reaches_both_the_corpus_and_the_web(self):
        """A voice assistant that can only reach one of them is half the app."""
        names = {d.name for d in live._declarations("speak")}
        assert "search_documents" in names
        assert "search_web" in names

    def test_the_interview_has_no_retrieval_at_all(self):
        """It is asking about a PERSON, not answering about the corpus.

        A tool that is offered will eventually be used: an interviewer with
        document search reaches for it and starts explaining the user's own
        files back at them instead of asking anything.
        """
        names = {d.name for d in live._declarations("interview")}
        assert "search_documents" not in names
        assert "list_documents" not in names
        assert "corpus_stats" not in names

    def test_the_interview_keeps_the_web(self):
        """Worth being able to place a company or a technology they mention."""
        assert "search_web" in {d.name for d in live._declarations("interview")}

    def test_only_the_interview_can_record_a_profile(self):
        assert "record_profile" in {
            d.name for d in live._declarations("interview")
        }
        assert "record_profile" not in {d.name for d in live._declarations("speak")}

    def test_a_tool_with_no_arguments_declares_no_schema(self):
        """An empty OBJECT schema is REJECTED by the API.

        `list_documents` and `corpus_stats` take nothing. Declaring them with
        `{type: OBJECT, properties: {}}` fails the whole session at connect
        time, so the app never starts rather than the tool never working.
        """
        for declaration in live._declarations("speak"):
            if declaration.name in ("list_documents", "corpus_stats"):
                assert declaration.parameters is None, declaration.name

    def test_a_tool_with_arguments_declares_them(self):
        by_name = {d.name: d for d in live._declarations("speak")}
        search = by_name["search_documents"]
        assert search.parameters is not None
        assert "query" in (search.parameters.properties or {})
        assert "query" in (search.parameters.required or [])


class TestTheButtonOwnsTheTurn:
    """Automatic activity detection is OFF, and that is the whole design.

    With it on, the model answers whenever it hears a pause -- and people pause
    constantly while speaking: to think, to find a word, to check a figure.
    Every one of those was read as "they have finished", so the assistant talked
    over the second half of the question.

    Tuning the threshold does not solve it, it only moves it. Short enough to
    feel responsive is short enough to interrupt; long enough never to interrupt
    is long enough to feel broken. The only reliable signal for "I have finished
    speaking" is a person saying so.

    Verified end to end: a three-second pause deliberately inserted halfway
    through a question produced no reply, and the complete question -- both
    halves -- was transcribed and answered after the button.
    """

    def test_automatic_detection_is_disabled(self):
        cfg = live.config("Kore")
        assert cfg.realtime_input_config is not None
        detection = cfg.realtime_input_config.automatic_activity_detection
        assert detection is not None
        assert detection.disabled is True

    def test_no_silence_threshold_is_configured(self):
        """A threshold here would mean the decision was still being tuned.

        It is not tuned, it is removed: nothing about the audio ends a turn.
        """
        detection = live.config("Kore").realtime_input_config.automatic_activity_detection
        assert detection.silence_duration_ms is None


class TestSessionResumption:
    """The conversation has to survive a socket that dies on its own.

    A live session's history lives SERVER-SIDE, inside the connection: we send
    no transcript and no prior turns. Measured -- within one socket the model
    recalls turn 1 at turn 3; on a fresh socket it says, correctly, that it has
    no access to anything said before.

    The socket closes by itself. Gemini drops an idle one and announces a hard
    lifetime cap through `go_away`. Without resumption a conversation with a
    pause in the middle forgets everything before the pause AND SAYS NOTHING --
    the assistant simply goes on answering confidently from an empty context.

    Verified end to end with a control: state a fact, reconnect with the
    handle -> recalled; reconnect without it -> not recalled.
    """

    def test_a_fresh_session_asks_for_handles(self):
        """An empty config REQUESTS resumption without resuming anything.

        Omitting it entirely means no handle is ever issued, and the first
        disconnection is unrecoverable.
        """
        cfg = live.config("Kore")
        assert cfg.session_resumption is not None
        assert cfg.session_resumption.handle is None

    def test_a_handle_is_passed_through(self):
        cfg = live.config("Kore", "handle-abc")
        assert cfg.session_resumption.handle == "handle-abc"

    def test_an_empty_handle_is_not_treated_as_one(self):
        """The route passes `resume or None`; this pins the other half.

        A blank string sent as a handle is rejected by the API, so a client
        that omits the parameter would fail to connect at all rather than
        starting fresh.
        """
        cfg = live.config("Kore", None)
        assert cfg.session_resumption.handle is None


class TestTrailingSilence:
    """The single least obvious thing in the app.

    Live decides a turn has ended by HEARING the speaker stop. Audio that ends
    on the last word gives it nothing to detect, and `audio_stream_end` does not
    substitute.

    Measured, before this existed: five seconds of clear speech, accepted by the
    session, no transcript reported, no reply, no error — it simply sat there
    for the full ninety-second timeout. With one second of silence appended, the
    same audio is transcribed correctly and answered in two seconds.
    """

    def test_a_full_second_is_appended(self):
        assert len(live.TRAILING_SILENCE) == live.INPUT_RATE * 2

    def test_it_is_actually_silent(self):
        """Not a buffer of something. Any signal here is a sound the model hears."""
        assert set(live.TRAILING_SILENCE) == {0}

    def test_the_rates_are_the_ones_the_model_uses(self):
        """16k in and 24k out, which is not symmetric and easy to assume wrong.

        Playing 24kHz audio through a 16kHz context sounds slow and deep, and
        nothing reports an error.
        """
        assert live.INPUT_RATE == 16_000
        assert live.OUTPUT_RATE == 24_000


@pytest.mark.asyncio
class TestFraming:
    async def test_audio_is_split_into_hundred_millisecond_frames(self):
        """Roughly what a microphone produces.

        One large blob is not merely inefficient: the VAD watches audio arrive
        over time, and a whole turn delivered at once gives it no stream to
        watch. Measured — the model never replied at all.
        """
        audio = bytes(live.INPUT_RATE * 2)  # one second
        frames = [f async for f in live.frames(audio)]
        assert len(frames) == 10
        assert sum(len(f) for f in frames) == len(audio)

    async def test_a_short_buffer_still_produces_one_frame(self):
        frames = [f async for f in live.frames(b"\x00\x01" * 10)]
        assert len(frames) == 1


@pytest.mark.asyncio
class TestToolBridge:
    """Tool calls run against the REAL corpus, through the agent's own code."""

    async def test_the_agent_runner_is_what_executes(self, monkeypatch):
        seen = {}

        async def fake_run_tool(name, args, **kwargs):
            seen["name"] = name
            seen["args"] = args
            seen.update(kwargs)
            return [], "eleven documents"

        monkeypatch.setattr(live.agent_tools, "run_tool", fake_run_tool)
        response, report = await live.run_tool_call(
            _Call("list_documents", {}), owner_id="u-1", top_k=5
        )
        assert seen["name"] == "list_documents"
        # Tenant scoping must reach the retriever. A voice session that loses
        # owner_id searches every tenant's documents.
        assert seen["owner_id"] == "u-1"
        assert response.response == {"result": "eleven documents"}

    async def test_a_failing_tool_does_not_kill_the_session(self, monkeypatch):
        """A raise here ends the call mid-sentence.

        The model can recover from "that failed" by saying so or trying
        something else; it cannot recover from a closed socket.
        """

        async def boom(name, args, **kwargs):
            raise RuntimeError("qdrant is down")

        monkeypatch.setattr(live.agent_tools, "run_tool", boom)
        response, report = await live.run_tool_call(
            _Call("search_documents", {"query": "x"}), owner_id=None, top_k=5
        )
        assert "failed" in response.response["result"]
        assert report["n"] == 0

    async def test_sources_are_deduplicated_per_document(self, monkeypatch):
        """Eight chunks of one handbook is ONE source to a listener."""

        async def fake_run_tool(name, args, **kwargs):
            return [_Hit("handbook.md", "doc-1"), _Hit("handbook.md", "doc-1")], "text"

        monkeypatch.setattr(live.agent_tools, "run_tool", fake_run_tool)
        _, report = await live.run_tool_call(
            _Call("search_documents", {"query": "x"}), owner_id=None, top_k=5
        )
        assert len(report["sources"]) == 1
        assert report["n"] == 2  # the count is still the true number of hits


class TestModes:
    """Speak and Interview are one pipeline and two prompts.

    Everything else is shared -- the socket, the audio handling, the manual
    turn boundaries, the tools, the persistence, the resumption. If these two
    ever stop being the only difference, this file is where that shows up.
    """

    def test_every_mode_exists_and_differs(self):
        assert set(live.MODES) == {"speak", "interview", "howler"}
        prompts = {m: live.MODES[m]["system"] for m in live.MODES}
        assert len(set(prompts.values())) == len(prompts)

    def test_howler_inherits_the_interview_craft(self):
        """It is ASSEMBLED from the interview prompt, not written again.

        Everything that makes an interview good is identical; a second copy
        would drift the moment either was improved.
        """
        howler = live.MODES["howler"]["system"]
        for craft in (
            "ASK ONE QUESTION THAT EARNS ITS PLACE",
            "MINE THE ANSWER BEFORE YOU ASK AGAIN",
            "DO NOT LEAD",
            "QUOTE THEM",
        ):
            assert craft in " ".join(howler.split()), craft

    def test_howler_does_not_inherit_the_fixed_field_list(self):
        """Its fields come from a brief, so Interview's must not leak in.

        The split is at "HOW TO GET THERE" -- the seam between WHAT is gathered
        and HOW. One paragraph earlier and a conversation about procurement
        budgets asks how many years of professional experience they have.
        """
        howler = " ".join(live.MODES["howler"]["system"].split())
        assert "how many years of professional experience" not in howler

    def test_howler_carries_slots_for_its_brief(self):
        howler = live.MODES["howler"]["system"]
        assert "{brief}" in howler and "{participant}" in howler

    def test_the_slots_are_filled_at_connect(self):
        text = live.config(
            "Kore", None, "howler", None, "THE BRIEF", "THE PERSON"
        ).system_instruction.parts[0].text
        assert "THE BRIEF" in text and "THE PERSON" in text
        assert "{brief}" not in text

    def test_an_empty_brief_does_not_leave_a_slot_showing(self):
        """A literal "{brief}" in a system prompt is a visible bug."""
        text = live.config("Kore", None, "howler").system_instruction.parts[0].text
        assert "{brief}" not in text and "{participant}" not in text

    def test_howler_uses_the_schema_it_is_given(self):
        """Not the built-in one. The whole point of the mode."""
        fields = [
            {"name": "budget", "label": "Budget", "description": "x",
             "type": "STRING", "required": True},
        ]
        names = {
            d.name
            for d in live._declarations("howler", fields)
        }
        assert "record_profile" in names
        record = next(
            d for d in live._declarations("howler", fields) if d.name == "record_profile"
        )
        properties = set(record.parameters.properties or {})
        assert "budget" in properties
        # Interview's fields must not appear in a Howler session.
        assert "years_experience" not in properties

    def test_they_are_stored_separately(self):
        """Interviews must not appear in the Speak list, or the reverse.

        Same table, different `kind` -- one conversation table for one concept,
        with the app it belongs to as a column.
        """
        assert live.kind_of("speak") == "parley"
        assert live.kind_of("interview") == "interview"
        assert live.kind_of("howler") == "howler"

    def test_an_unknown_mode_falls_back_rather_than_failing(self):
        """A bad query parameter must not take the socket down.

        The mode arrives in a URL, so it is whatever anyone types.
        """
        assert live.mode_of("nonsense") == "speak"
        assert live.mode_of("") == "speak"

    @pytest.mark.parametrize("mode", ["speak", "interview"])
    def test_every_mode_forbids_citation_numbers(self, mode):
        """There is no screen to match "[3]" to, and it is read out as a number."""
        assert "citation number" in _flat(mode).lower()

    @pytest.mark.parametrize("mode", ["speak", "interview"])
    def test_every_mode_forbids_visual_references(self, mode):
        system = _flat(mode)
        assert "above" in system and "below" in system

    @pytest.mark.parametrize("mode", ["speak", "interview"])
    def test_every_mode_reaches_the_web(self, mode):
        """Both need to be able to look something up, for different reasons."""
        assert "search_web" in _flat(mode)


def _flat(mode: str) -> str:
    """The prompt with its hard wrapping removed.

    The prompts are wrapped prose, so a phrase worth asserting on is as likely
    as not to straddle a newline -- "doing most of
the talking". Matching the
    raw string makes the test depend on where the paragraph happened to wrap.
    """
    return " ".join(live.MODES[mode]["system"].split())


class TestInterviewPrompt:
    """The behaviours that make an interview an interview rather than a chat."""

    def test_it_asks_compound_questions_but_not_scattered_ones(self):
        """One question may cover several related things. Not unrelated ones.

        The rule used to be a flat "one question at a time", which produced an
        interrogation: twenty small questions in a row is what makes somebody
        start answering in single words. A question covering one subject from
        several sides gets a paragraph instead. What still does not work is two
        UNRELATED questions in a breath -- that reliably loses the first.
        """
        text = _flat("interview")
        assert "ASK ONE QUESTION THAT EARNS ITS PLACE" in text
        assert "UNRELATED questions in a breath" in text

    def test_it_mines_the_answer_before_asking_again(self):
        """A compound question is answered with more than it asked for.

        Taking one fact out of an answer and moving on is how an interviewer
        ends up asking about something it was just told.
        """
        assert "MINE THE ANSWER BEFORE YOU ASK AGAIN" in _flat("interview")

    def test_it_is_told_to_follow_up_on_vague_answers(self):
        """"It's going well" is a deflection, not an answer."""
        assert "FOLLOW UP ON VAGUE" in _flat("interview")

    def test_it_is_told_not_to_lead(self):
        """A leading question buys agreement, which is not information."""
        assert "DO NOT LEAD" in _flat("interview")

    def test_it_is_told_the_participant_does_the_talking(self):
        assert "most of the talking" in _flat("interview")

    def test_it_asks_who_it_is_talking_to(self):
        """A profile with no name attached is not a profile."""
        assert "their name" in _flat("interview")


class TestWebReach:
    def test_an_unconfigured_web_is_declared(self, monkeypatch):
        """Otherwise "did not search" and "cannot search" sound identical."""
        monkeypatch.setattr(live.websearch, "enabled", lambda: False)
        text = live.config("Kore").system_instruction.parts[0].text
        assert "NOT CONFIGURED" in text

    def test_a_configured_web_adds_no_caveat(self, monkeypatch):
        monkeypatch.setattr(live.websearch, "enabled", lambda: True)
        text = live.config("Kore").system_instruction.parts[0].text
        assert "NOT CONFIGURED" not in text

    @pytest.mark.parametrize("mode", ["speak", "interview"])
    def test_the_caveat_reaches_every_mode(self, mode, monkeypatch):
        monkeypatch.setattr(live.websearch, "enabled", lambda: False)
        text = live.config("Kore", None, mode).system_instruction.parts[0].text
        assert "NOT CONFIGURED" in text


class _Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.id = "call-1"


class _Hit:
    def __init__(self, filename, document_id):
        self.filename = filename
        self.document_id = document_id
        self.source = "document"
        self.url = None


class TestNaming:
    """An interview is named after its participant, not its opening line.

    Titling from the first utterance gave a drawer full of rows reading "Hi
    there" and "Hello, can you hear me" -- and one, observed in the database,
    reading "¿Qué tal? ¿Cómo estás?". That utterance is also the one the
    transcriber mangles most, because nobody has warmed up yet.

    Applied as soon as the name is recorded rather than only at the end, so the
    list stops reading "Interview, Interview, Interview" while one is still
    running -- which is precisely when you need to tell them apart.
    """

    def test_the_placeholder_is_what_marks_an_unnamed_interview(self):
        """It has to be a value nothing else produces.

        Distinguishing "not yet named" from "named by a person" is what keeps
        the app from arguing with the user about what to call their own
        conversation, and the placeholder is the only signal available.
        """
        assert live.UNNAMED_INTERVIEW == "Interview"


@pytest.mark.asyncio
class TestNameConversation:
    async def test_a_name_replaces_the_placeholder(self, monkeypatch):
        titles = _capture(monkeypatch, start="Interview")
        await live.name_conversation(_ID, {"full_name": "Mithun Tanwar"})
        assert titles[-1] == "Mithun Tanwar"

    async def test_a_manual_rename_is_never_overwritten(self, monkeypatch):
        """Only the placeholder is replaced.

        Anything else was set by a person or already carries a name, and a
        later correction to `full_name` must not undo their choice.
        """
        titles = _capture(monkeypatch, start="Renamed by hand")
        await live.name_conversation(_ID, {"full_name": "Someone Else"})
        assert titles == ["Renamed by hand"]

    async def test_a_profile_with_no_name_changes_nothing(self, monkeypatch):
        titles = _capture(monkeypatch, start="Interview")
        await live.name_conversation(_ID, {"current_role": "platform engineer"})
        assert titles == ["Interview"]

    async def test_a_blank_name_is_not_a_name(self, monkeypatch):
        titles = _capture(monkeypatch, start="Interview")
        await live.name_conversation(_ID, {"full_name": "   "})
        assert titles == ["Interview"]

    async def test_a_failure_to_rename_does_not_raise(self, monkeypatch):
        """A title is never worth interrupting a live conversation for."""

        def boom():
            raise RuntimeError("database is down")

        monkeypatch.setattr(live, "SessionLocal", boom, raising=False)
        await live.name_conversation(_ID, {"full_name": "Sam"})


_ID = uuid.UUID(int=7)


def _capture(monkeypatch, *, start: str) -> list[str]:
    """A stand-in session whose one row records every title it is given."""
    titles = [start]

    class Row:
        kind = "interview"

        @property
        def title(self) -> str:
            return titles[-1]

        @title.setter
        def title(self, value: str) -> None:
            titles.append(value)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, model, ident):
            return Row()

        async def commit(self):
            return None

    monkeypatch.setattr(
        "app.db.session.SessionLocal", lambda: Session(), raising=False
    )
    return titles


class TestParticipantEnding:
    """Ending it yourself, and why that is not the same as the model ending it.

    The model ends when it has what it came for. A person ends when they have
    had enough -- and a half-filled profile somebody walked out of is a
    different finding from one the interviewer could not get answers to.
    Without `ended_by` both read as "complete", which is the thing this records
    to prevent.
    """

    def test_it_records_who_ended_it(self, monkeypatch):
        stored: dict = {}

        class _Chat:
            profile: dict = {}
            owner_id = "owner-1"
            id = uuid.UUID(int=1)
            project_id = None
            title = live.UNNAMED_INTERVIEW
            owner_id = "owner-1"
            id = uuid.UUID(int=1)
            project_id = None

        chat = _Chat()
        _patch_session(monkeypatch, chat, stored)

        asyncio.run(live.finish_interview(uuid.uuid4()))

        assert chat.profile["ended"] is True
        assert chat.profile["ended_by"] == "participant"
        assert stored["committed"]

    def test_it_keeps_the_closing_account_the_model_just_wrote(self, monkeypatch):
        """Pressing stop asks the model to close properly FIRST.

        So by the time this runs, the profile may already carry a summary, a
        demeanour and the notable moments -- written by the only thing that
        heard the audio. Overwriting them with nothing would throw away the
        entire point of asking.
        """
        stored: dict = {}

        class _Chat:
            profile = {
                "ended": True,
                "summary": "A senior engineer.",
                "affect": {"demeanour": "Warm, unhurried.", "moments": ["x"]},
            }
            owner_id = "owner-1"
            id = uuid.UUID(int=1)
            project_id = None
            title = "Priya Raman"

        chat = _Chat()
        _patch_session(monkeypatch, chat, stored)

        asyncio.run(live.finish_interview(uuid.uuid4()))

        assert chat.profile["summary"] == "A senior engineer."
        assert chat.profile["affect"]["demeanour"] == "Warm, unhurried."
        assert chat.profile["ended_by"] == "participant"

    def test_an_ending_the_model_owned_is_not_reattributed(self, monkeypatch):
        """An interview it closed on its own keeps its own ending.

        Only reachable if somebody presses stop on an interview that had
        already finished, which the UI hides -- but reattributing a completed
        interview to the participant would misreport it as abandoned.
        """
        stored: dict = {}

        class _Chat:
            profile = {"ended": True, "ended_by": "model", "summary": "Done."}
            owner_id = "owner-1"
            id = uuid.UUID(int=1)
            project_id = None
            title = "Priya Raman"

        chat = _Chat()
        _patch_session(monkeypatch, chat, stored)

        asyncio.run(live.finish_interview(uuid.uuid4()))

        assert chat.profile["ended_by"] == "model"

    def test_a_missing_conversation_is_not_an_error(self, monkeypatch):
        """Bookkeeping must never take down a call that is already over."""
        _patch_session(monkeypatch, None, {})
        asyncio.run(live.finish_interview(uuid.uuid4()))  # does not raise


def _patch_session(monkeypatch, chat, stored):
    """Stand in for the database, so these assert on behaviour not on SQL."""

    class _Result:
        """What a lookup returns when there is nothing to find.

        `name_from_context` asks for an invite label and a project title as a
        last resort before leaving a conversation unnamed. These stubs have
        neither, so both come back empty -- which is the path worth covering
        here: the ending must still be recorded when nothing can name it.
        """

        def scalar_one_or_none(self):
            return None

    class _DB:
        async def get(self, _model, _id):
            return chat

        async def execute(self, *_args, **_kwargs):
            return _Result()

        async def commit(self):
            stored["committed"] = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr("app.db.session.SessionLocal", lambda: _DB())

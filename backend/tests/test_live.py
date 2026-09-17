"""The native-audio path, and the two things about it that fail silently.

Neither raises. Both produce a session that connects, accepts audio, and then
sits there — which in an audio-only app is indistinguishable from a crash.
"""

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

    def test_both_modes_exist_and_differ(self):
        assert set(live.MODES) == {"speak", "interview"}
        assert (
            live.MODES["speak"]["system"] != live.MODES["interview"]["system"]
        )

    def test_they_are_stored_separately(self):
        """Interviews must not appear in the Speak list, or the reverse.

        Same table, different `kind` -- one conversation table for one concept,
        with the app it belongs to as a column.
        """
        assert live.kind_of("speak") == "parley"
        assert live.kind_of("interview") == "interview"

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

    def test_one_question_at_a_time(self):
        """Two questions in one breath gets an answer to the second only."""
        assert "ONE question at a time" in _flat("interview")

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

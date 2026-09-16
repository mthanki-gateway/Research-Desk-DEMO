"""The native-audio path, and the two things about it that fail silently.

Neither raises. Both produce a session that connects, accepts audio, and then
sits there — which in an audio-only app is indistinguishable from a crash.
"""

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
        for declaration in live._declarations():
            assert declaration.description == source[declaration.name]

    def test_only_the_spoken_subset_is_offered(self):
        names = {d.name for d in live._declarations()}
        assert names <= set(live.SPOKEN_TOOLS)
        # The two that make no sense out loud.
        assert "remember_preference" not in names
        assert "read_around" not in names

    def test_retrieval_and_the_web_are_both_there(self):
        """A voice assistant that can only reach one of them is half the app."""
        names = {d.name for d in live._declarations()}
        assert "search_documents" in names
        assert "search_web" in names

    def test_a_tool_with_no_arguments_declares_no_schema(self):
        """An empty OBJECT schema is REJECTED by the API.

        `list_documents` and `corpus_stats` take nothing. Declaring them with
        `{type: OBJECT, properties: {}}` fails the whole session at connect
        time, so the app never starts rather than the tool never working.
        """
        for declaration in live._declarations():
            if declaration.name in ("list_documents", "corpus_stats"):
                assert declaration.parameters is None, declaration.name

    def test_a_tool_with_arguments_declares_them(self):
        by_name = {d.name: d for d in live._declarations()}
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


class TestSystemPrompt:
    """What the model is told about speaking, since nothing else enforces it."""

    def test_it_is_told_to_search_before_answering(self):
        assert "before answering" in live.SYSTEM.lower()

    def test_it_is_told_not_to_say_citation_numbers(self):
        """There is no screen to match "[3]" to, and it is read as a number."""
        assert "citation number" in live.SYSTEM.lower()

    def test_it_is_told_not_to_refer_to_anything_visual(self):
        assert "above" in live.SYSTEM and "below" in live.SYSTEM

    def test_an_unconfigured_web_is_declared(self, monkeypatch):
        """Otherwise "did not search" and "cannot search" sound identical."""
        monkeypatch.setattr(live.websearch, "enabled", lambda: False)
        text = live.config("Kore").system_instruction.parts[0].text
        assert "NOT CONFIGURED" in text

    def test_a_configured_web_adds_no_caveat(self, monkeypatch):
        monkeypatch.setattr(live.websearch, "enabled", lambda: True)
        text = live.config("Kore").system_instruction.parts[0].text
        assert "NOT CONFIGURED" not in text


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

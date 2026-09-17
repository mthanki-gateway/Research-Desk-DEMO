"""Parley's two silent failure modes.

Neither of these raises. Both produce a turn that completes, returns a
plausible-looking payload, and is wrong in a way nobody can see from the
screen -- which in an audio-first app means nobody can see it at all.
"""

import struct

import pytest

from app.services import voice


class TestWavHeader:
    """Raw PCM in an <audio> element is silence, with no error anywhere.

    The TTS model returns `audio/l16` -- signed 16-bit little-endian samples
    with NO container. A browser handed those bytes cannot know the sample
    rate, the channel count or the bit depth, so it plays nothing and reports
    nothing. The same class of mistake as the Riva truncation earlier in this
    project, in the opposite direction: there a container was sent where raw
    frames were wanted.
    """

    def test_the_header_is_a_playable_wav(self):
        pcm = b"\x00\x01" * 1000
        wav = voice.to_wav(pcm)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"
        # 44-byte header plus the samples, unmodified.
        assert len(wav) == 44 + len(pcm)
        assert wav[44:] == pcm

    def test_the_declared_sizes_match_the_payload(self):
        """A wrong length field truncates playback rather than failing."""
        pcm = b"\x00\x01" * 512
        wav = voice.to_wav(pcm)
        riff_size = struct.unpack("<I", wav[4:8])[0]
        data_size = struct.unpack("<I", wav[40:44])[0]
        assert riff_size == 36 + len(pcm)
        assert data_size == len(pcm)

    def test_the_rate_is_read_from_the_response_not_assumed(self):
        """A hardcoded rate does not fail -- it changes the pitch.

        If the model ever returns 16kHz and the header says 24000, the answer
        plays 1.5x fast and chipmunked. Nothing errors, nothing logs, and the
        only symptom is that the assistant sounds absurd.
        """
        assert voice._rate_from_mime("audio/l16; rate=16000") == 16_000
        assert voice._rate_from_mime("audio/l16;rate=48000; channels=1") == 48_000
        # No rate in the mime type: fall back rather than crash on a turn that
        # is otherwise fine.
        assert voice._rate_from_mime("audio/l16") == voice.TTS_SAMPLE_RATE
        assert voice._rate_from_mime("") == voice.TTS_SAMPLE_RATE

    def test_the_rate_reaches_the_header(self):
        wav = voice.to_wav(b"\x00\x01" * 10, rate=16_000)
        assert struct.unpack("<I", wav[24:28])[0] == 16_000
        # Byte rate must follow the sample rate, or players resample wrongly.
        assert struct.unpack("<I", wav[28:32])[0] == 16_000 * 2


class TestSpeakable:
    """Markup is punctuation to the eye and noise to the ear.

    A RAG answer is written to be READ. Fed to a speech model unchanged, every
    citation marker is pronounced -- "bracket three" -- and every heading hash
    and emphasis asterisk either gets read out or distorts the phrasing.
    """

    def test_citation_markers_do_not_survive(self):
        out = voice.speakable("Pool size was 64 [1]. The limit was 40 [2][5].")
        assert "[" not in out and "]" not in out
        assert "64" in out and "40" in out

    def test_multi_number_citations_go_too(self):
        assert "[" not in voice.speakable("Both agree [1, 3] and [2; 4].")

    def test_headings_lose_their_hashes(self):
        out = voice.speakable("## Findings\nThe service failed.")
        assert "#" not in out
        assert "Findings" in out

    def test_emphasis_is_unwrapped_not_deleted(self):
        """The WORD must survive. Stripping it removes the emphasised term."""
        out = voice.speakable("This was **critical** and _urgent_ and ***now***.")
        assert "critical" in out and "urgent" in out and "now" in out
        assert "*" not in out and "_" not in out

    def test_a_link_keeps_its_text_and_drops_its_url(self):
        out = voice.speakable("See [the runbook](https://example.com/rb) for more.")
        assert "the runbook" in out
        assert "example.com" not in out

    def test_bullets_lose_their_markers(self):
        out = voice.speakable("- first\n- second\n* third")
        assert "first" in out and "second" in out and "third" in out
        assert not any(line.startswith(("-", "*")) for line in out.splitlines())

    def test_a_table_is_removed_rather_than_read_aloud(self):
        """Pipes and dashes read as gibberish, and a table read linearly is
        worse than useless -- the column headers arrive once, minutes before
        the cells that need them."""
        out = voice.speakable("Totals:\n| Q1 | Q2 |\n| --- | --- |\n| 10 | 20 |\nEnds.")
        assert "|" not in out
        assert "Totals:" in out and "Ends." in out

    def test_code_is_named_rather_than_spelled_out(self):
        out = voice.speakable("Run this:\n```\nSELECT * FROM t;\n```\nThen restart.")
        assert "SELECT" not in out
        assert "code omitted" in out
        assert "Then restart." in out

    def test_inline_code_keeps_its_content(self):
        """Unlike a block, an inline span is usually a word in the sentence."""
        out = voice.speakable("Set `max_connections` to 40.")
        assert "max_connections" in out
        assert "`" not in out

    def test_blank_line_runs_collapse(self):
        """A speech model reads a run of newlines as a long dead pause."""
        assert "\n\n" not in voice.speakable("One.\n\n\n\nTwo.")

    def test_ordinary_prose_is_left_alone(self):
        """The common case must not be mangled by any of the above."""
        prose = "You have ten documents. The largest is the engineering handbook."
        assert voice.speakable(prose) == prose


class TestVoices:
    def test_the_default_is_a_real_voice(self):
        """An unknown name is rejected by the API, failing the whole turn."""
        assert voice.DEFAULT_VOICE in {v["id"] for v in voice.VOICES}

    def test_every_voice_says_how_it_sounds(self):
        """The bare names are unreadable as a menu -- nobody can pick between
        Sadaltager and Rasalgethi by reading them."""
        for v in voice.VOICES:
            assert v["character"].strip(), v["id"]


class TestNoSpeechGuard:
    """Silence must not become a question.

    Passed through, "(no speech)" is embedded, retrieved against, answered and
    spoken back: an entire turn and three model calls spent on an empty room.
    A recogniser asked to transcribe silence always returns SOMETHING, so the
    guard has to recognise the something.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "(no speech)",
            "(No speech)",
            "(no speech detected)",
            "No speech",
            "",
            "   ",
            '"(no speech)"',
        ],
    )
    def test_silence_becomes_empty(self, raw):
        assert voice.heard(raw) == ""

    @pytest.mark.parametrize(
        "raw",
        [
            "How many documents are in the collection?",
            '"What did the incident report say?"',
            "  Tell me about on-call paging.  ",
        ],
    )
    def test_real_speech_survives(self, raw):
        out = voice.heard(raw)
        assert out and not out.startswith('"') and out == out.strip()

    def test_a_question_about_silence_is_not_silence(self):
        """The stem match must not eat a genuine sentence beginning with it."""
        assert voice.heard("No speech was detected in the recording, why?")


@pytest.mark.asyncio
class TestTheTurnCanActuallyRun:
    """The graph in the RUNNING server is not the graph in a test process.

    THE BUG THIS PINS

    The route called `run_agent` with no `thread_id`, and every check here
    passed -- because a bare Python process never runs the app's lifespan, so
    it gets a graph compiled WITHOUT a checkpointer, which does not care. The
    real server installs one during startup, and a checkpointed graph refuses
    outright:

        ValueError: Checkpointer requires one or more of the following
        'configurable' keys: thread_id, checkpoint_ns, checkpoint_id

    Every spoken turn 500'd. The one configuration that mattered was the one
    not exercised, which is the whole lesson: this asserts on the ARGUMENTS the
    route passes, not on whether a particular graph tolerates them.
    """

    async def test_a_thread_id_is_passed(self, monkeypatch):
        from app.api import voice as api

        seen: dict = {}

        async def fake_run_agent(question, **kwargs):
            seen["question"] = question
            seen.update(kwargs)
            return _Result()

        async def fake_transcribe(data, **kwargs):
            return "how many documents are there"

        async def fake_speak(text, **kwargs):
            return b"RIFF" + bytes(40), 24_000

        monkeypatch.setattr(api, "run_agent", fake_run_agent)
        monkeypatch.setattr(api.voice, "transcribe", fake_transcribe)
        monkeypatch.setattr(api.voice, "speak", fake_speak)
        monkeypatch.setattr(api, "_name_hint", _no_hint)

        await api.ask(audio=_Upload(b"fake wav"), voice_name="Kore", top_k=0, user=_User())

        assert seen.get("thread_id"), "no thread_id: a checkpointed graph refuses to run"

    async def test_each_turn_gets_its_own_thread(self, monkeypatch):
        """Reused threads grow without bound and leak evidence between turns.

        The state accumulators use APPEND reducers, so a thread shared across
        questions carries the previous question's passages into the next
        answer -- which in a spoken app is invisible until the assistant cites
        something nobody asked about.
        """
        from app.api import voice as api

        threads: list[str] = []

        async def fake_run_agent(question, **kwargs):
            threads.append(kwargs.get("thread_id", ""))
            return _Result()

        async def fake_transcribe(data, **kwargs):
            return "a question"

        async def fake_speak(text, **kwargs):
            return b"RIFF" + bytes(40), 24_000

        monkeypatch.setattr(api, "run_agent", fake_run_agent)
        monkeypatch.setattr(api.voice, "transcribe", fake_transcribe)
        monkeypatch.setattr(api.voice, "speak", fake_speak)
        monkeypatch.setattr(api, "_name_hint", _no_hint)

        for _ in range(3):
            await api.ask(audio=_Upload(b"x"), voice_name="Kore", top_k=0, user=_User())

        assert len(set(threads)) == 3, threads

    async def test_clarification_is_switched_off(self, monkeypatch):
        """A human-in-the-loop pause renders as buttons, and there are none.

        A paused graph here is a turn that produces no audio at all and no way
        to continue it.
        """
        from app.api import voice as api

        seen: dict = {}

        async def fake_run_agent(question, **kwargs):
            seen.update(kwargs)
            return _Result()

        async def fake_transcribe(data, **kwargs):
            return "a question"

        async def fake_speak(text, **kwargs):
            return b"RIFF" + bytes(40), 24_000

        monkeypatch.setattr(api, "run_agent", fake_run_agent)
        monkeypatch.setattr(api.voice, "transcribe", fake_transcribe)
        monkeypatch.setattr(api.voice, "speak", fake_speak)
        monkeypatch.setattr(api, "_name_hint", _no_hint)

        await api.ask(audio=_Upload(b"x"), voice_name="Kore", top_k=0, user=_User())
        assert seen.get("clarify") is False

    async def test_silence_never_reaches_the_agent(self, monkeypatch):
        """An empty transcript must be answered with speech, not searched."""
        from app.api import voice as api

        called = False

        async def fake_run_agent(question, **kwargs):
            nonlocal called
            called = True
            return _Result()

        async def fake_transcribe(data, **kwargs):
            return ""  # the no-speech guard fired

        async def fake_speak(text, **kwargs):
            return b"RIFF" + bytes(40), 24_000

        monkeypatch.setattr(api, "run_agent", fake_run_agent)
        monkeypatch.setattr(api.voice, "transcribe", fake_transcribe)
        monkeypatch.setattr(api.voice, "speak", fake_speak)
        monkeypatch.setattr(api, "_name_hint", _no_hint)

        out = await api.ask(audio=_Upload(b"x"), voice_name="Kore", top_k=0, user=_User())
        assert not called, "silence was sent to the agent"
        assert out["heard_nothing"] is True
        # Still SPEAKS. The user is not looking at the screen -- that is the
        # premise of the app -- so a silent error banner is a dead end.
        assert out["audio"]


class _User:
    owner_id = None


class _Upload:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.content_type = "audio/wav"

    async def read(self) -> bytes:
        return self._data


class _Result:
    answer = "You have ten documents."
    evidence: list = []
    iterations = 1
    partial = False


async def _no_hint(owner_id):
    return ""


class TestRetryable:
    """Which failures are worth asking a different model about.

    Measured in one pass mid-build: 3.1-flash-tts returned 429,
    2.5-flash-preview-tts answered in 3.1 seconds, 2.5-pro-preview-tts returned
    429. Free-tier TTS quota is per model and small, and the newest is not the
    most available.

    Falling through on the WRONG failures is its own bug: a 400 means the text
    or the voice is wrong, every model will reject it identically, and trying
    three turns one clear error into three slow ones.
    """

    def test_quota_and_provider_faults_fall_through(self):
        assert voice._retryable(429)
        assert voice._retryable(500)
        assert voice._retryable(503)

    def test_a_bad_request_does_not(self):
        assert not voice._retryable(400)

    def test_an_auth_failure_does_not(self):
        """A different model does not have a different key."""
        assert not voice._retryable(401)
        assert not voice._retryable(403)


@pytest.mark.asyncio
class TestSpeakFallsThrough:
    async def test_the_second_model_answers_when_the_first_is_out_of_quota(
        self, monkeypatch
    ):
        import base64 as b64

        calls: list[str] = []
        pcm = b64.b64encode(bytes(64)).decode()

        class FakeResponse:
            def __init__(self, status, body):
                self.status_code = status
                self._body = body

            def json(self):
                return self._body

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, path, json=None):
                calls.append(path)
                if len(calls) == 1:
                    return FakeResponse(429, {"error": {"message": "quota"}})
                return FakeResponse(
                    200,
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                # Capital L and a codec field:
                                                # the real second model returns
                                                # `audio/L16;codec=pcm;rate=24000`
                                                # where the first returns
                                                # `audio/l16; rate=24000`.
                                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                                "data": pcm,
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                )

        monkeypatch.setattr(voice, "_client", lambda: FakeClient())
        wav, rate = await voice.speak("ten documents", voice="Kore")

        assert len(calls) == 2, "did not try the second model"
        assert wav[:4] == b"RIFF"
        assert rate == 24_000

    async def test_a_bad_request_stops_immediately(self, monkeypatch):
        calls: list[str] = []

        class FakeResponse:
            status_code = 400
            text = ""

            def json(self):
                return {"error": {"message": "unknown voice"}}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, path, json=None):
                calls.append(path)
                return FakeResponse()

        monkeypatch.setattr(voice, "_client", lambda: FakeClient())
        with pytest.raises(voice.VoiceError):
            await voice.speak("x", voice="Kore")
        assert len(calls) == 1, "retried a request every model will reject"

    async def test_all_exhausted_names_what_was_tried(self, monkeypatch):
        """The user is not looking at the screen. The message has to explain."""

        class FakeResponse:
            status_code = 429

            def json(self):
                return {"error": {"message": "quota"}}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, path, json=None):
                return FakeResponse()

        monkeypatch.setattr(voice, "_client", lambda: FakeClient())
        with pytest.raises(voice.VoiceError) as caught:
            await voice.speak("x", voice="Kore")
        assert "429" in str(caught.value)
        assert "on screen" in str(caught.value)

"""Earshot's two silent failure modes.

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

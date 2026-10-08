"""Duplex: the parts that decide WHICH answer gets spoken.

The models and the sockets are exercised live; what is pinned here is the logic
that sits between them, because that is where a wrong turn does damage: speaking
an answer written for words the person did not finish saying.
"""

import asyncio

import pytest

from app.services import duplex


class TestChunker:
    def test_the_first_clause_is_spoken_before_the_sentence_ends(self):
        c = duplex.Chunker()
        assert c.feed("A hash map stores keys, ") == ["A hash map stores keys,"]

    def test_a_short_opening_clause_waits_for_more(self):
        # "Well," is not worth a synthesis call.
        c = duplex.Chunker()
        assert c.feed("Well, ") == []
        assert c.feed("it depends on the data. ") == ["Well, it depends on the data."]

    def test_later_pieces_are_whole_sentences_not_clauses(self):
        c = duplex.Chunker()
        out = c.feed("It is fast and simple to use. Lookups are constant time, on average. ")
        assert out == ["It is fast and simple to use.", "Lookups are constant time, on average."]

    def test_a_decimal_is_not_a_sentence_end(self):
        c = duplex.Chunker()
        assert c.feed("Version 3.5 is out now. ") == ["Version 3.5 is out now."]

    def test_punctuation_at_the_buffer_edge_waits_for_the_next_token(self):
        # "3." might become "3.5"; it must not be cut off.
        c = duplex.Chunker()
        assert c.feed("It costs 3.") == []
        assert c.feed("5 dollars. ") == ["It costs 3.5 dollars."]

    def test_flush_returns_the_unfinished_tail(self):
        c = duplex.Chunker()
        c.feed("And that is")
        assert c.flush() == "And that is"
        assert c.flush() is None


class TestSameWords:
    def test_case_and_punctuation_do_not_matter(self):
        assert duplex.same_words("what is a hash map", "What is a hash map?")

    def test_a_truncated_question_is_not_the_same_question(self):
        # The case the whole adopt-or-discard decision exists for.
        assert not duplex.same_words("what is a hash", "what is a hash map")

    def test_empty_matches_empty(self):
        assert duplex.same_words("", "  ")


class TestChain:
    def test_parse(self):
        chain = duplex.parse_chain("groq:qwen/qwen3.8-27b, gemini:models/gemma-4-26b-a4b-it, nope:x, bad")
        assert [t.name for t in chain] == ["groq:qwen/qwen3.8-27b", "gemini:gemma-4-26b-a4b-it"]

    def test_a_cooling_model_is_skipped_until_it_cools(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        a, b = duplex.parse_chain("groq:a,groq:b")
        duplex._cool.clear()
        assert duplex.usable([a, b]) == [a, b]
        duplex.cool(a, 60)
        assert duplex.usable([a, b]) == [b]
        duplex._cool.clear()

    def test_a_provider_without_a_key_is_not_usable(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k" if provider == "gemini" else "")
        duplex._cool.clear()
        chain = duplex.parse_chain("groq:a,gemini:b")
        assert [t.name for t in duplex.usable(chain)] == ["gemini:b"]

    async def test_it_fails_over_before_the_first_token(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        duplex._cool.clear()

        async def broken(target, system, turns):
            raise duplex.LLMUnavailable("down", status=429)
            yield  # pragma: no cover

        async def working(target, system, turns):
            yield "hello "
            yield "there"

        monkeypatch.setattr(duplex, "_stream_groq", broken)
        monkeypatch.setattr(duplex, "_stream_gemini", working)
        chain = duplex.parse_chain("groq:a,gemini:b")
        seen = []
        got = [
            p async for p in duplex.stream_answer(chain, "sys", [{"role": "user", "content": "hi"}], seen.append)
        ]
        assert "".join(got) == "hello there"
        assert [t.name for t in seen] == ["gemini:b"]
        # The one that failed is now cooling, not retried on the next turn.
        assert [t.name for t in duplex.usable(chain)] == ["gemini:b"]
        duplex._cool.clear()

    async def test_it_does_not_splice_a_second_model_onto_spoken_words(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        duplex._cool.clear()

        async def dies_midway(target, system, turns):
            yield "partial "
            raise duplex.LLMUnavailable("cut", status=500)

        async def other(target, system, turns):
            yield "NEVER"

        monkeypatch.setattr(duplex, "_stream_groq", dies_midway)
        monkeypatch.setattr(duplex, "_stream_gemini", other)
        chain = duplex.parse_chain("groq:a,gemini:b")
        got: list[str] = []
        with pytest.raises(duplex.LLMUnavailable):
            async for p in duplex.stream_answer(chain, "s", [{"role": "user", "content": "x"}]):
                got.append(p)
        assert got == ["partial "]
        duplex._cool.clear()


class TestDraft:
    async def test_pieces_stream_in_order_and_end(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        duplex._cool.clear()

        async def stream(target, system, turns):
            for part in ["It is a table that maps keys to values, ", "so lookups are fast. ", "Neat."]:
                yield part

        monkeypatch.setattr(duplex, "_stream_groq", stream)
        draft = duplex.Draft(for_text="what is a hash map")
        await duplex.write_draft(draft, duplex.parse_chain("groq:m"), [])
        pieces = [p async for p in duplex.pieces_of(draft)]
        assert pieces == ["It is a table that maps keys to values,", "so lookups are fast.", "Neat."]
        assert draft.done and draft.model == "groq:m"
        duplex._cool.clear()

    async def test_a_failure_ends_the_queue_instead_of_hanging_the_speaker(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        duplex._cool.clear()

        async def broken(target, system, turns):
            raise duplex.LLMUnavailable("nope", status=500)
            yield  # pragma: no cover

        monkeypatch.setattr(duplex, "_stream_groq", broken)
        draft = duplex.Draft(for_text="x y")
        await duplex.write_draft(draft, duplex.parse_chain("groq:m"), [])
        assert [p async for p in duplex.pieces_of(draft)] == []
        assert draft.failed
        duplex._cool.clear()


class TestSession:
    """The adopt-or-discard decision, with the model and the voice faked."""

    def _session(self, monkeypatch):
        from app.api import duplex as api

        class FakeWS:
            def __init__(self):
                self.text, self.audio = [], []

            async def send_text(self, t):
                self.text.append(t)

            async def send_bytes(self, b):
                self.audio.append(b)

        s = api.Session(FakeWS(), "Kore")
        s.chain = duplex.parse_chain("groq:m")
        return s, api

    def _events(self, s):
        import json

        return [json.loads(t) for t in s.ws.text]

    async def test_a_speculative_draft_is_replaced_when_the_words_change(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        s, _ = self._session(monkeypatch)
        started = []

        async def fake_write(draft, chain, history):
            started.append(draft.for_text)
            await asyncio.sleep(10)

        monkeypatch.setattr(duplex, "write_draft", fake_write)
        assert await s.draft_for("what is a hash", final=False)
        first = s.draft
        await asyncio.sleep(0)  # let the first one actually start
        assert await s.draft_for("what is a hash map", final=False)
        await asyncio.sleep(0)
        assert s.draft is not first
        assert started == ["what is a hash", "what is a hash map"]
        assert first.task.cancelled() or first.task.cancelling() or first.task.done()
        s.discard_draft("test")

    async def test_the_same_words_do_not_cost_another_request(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        s, _ = self._session(monkeypatch)
        calls = []

        async def fake_write(draft, chain, history):
            calls.append(1)
            await asyncio.sleep(10)

        monkeypatch.setattr(duplex, "write_draft", fake_write)
        await s.draft_for("what is a hash map", final=False)
        await s.draft_for("What is a hash map?", final=False)
        await asyncio.sleep(0)
        assert len(calls) == 1
        s.discard_draft("test")

    async def test_speculation_stops_at_the_cap_but_the_final_answer_does_not(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        s, _ = self._session(monkeypatch)
        s.s.duplex_max_speculations  # the real setting
        monkeypatch.setattr(duplex, "write_draft", lambda d, c, h: asyncio.sleep(10))
        s.speculations = s.s.duplex_max_speculations
        assert not await s.draft_for("one two three", final=False)
        assert await s.draft_for("one two three", final=True)
        s.discard_draft("test")

    async def test_one_word_is_not_worth_an_answer(self, monkeypatch):
        s, _ = self._session(monkeypatch)
        assert not await s.draft_for("uh", final=False)
        assert s.draft is None

    async def test_the_answer_is_spoken_once_and_older_drafts_are_discarded(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        s, api = self._session(monkeypatch)

        async def fake_write(draft, chain, history):
            for p in (f"Answer to {draft.for_text}.",):
                draft.pieces.put_nowait(p)
            draft.model = "groq:m"
            draft.done = True
            draft.pieces.put_nowait(duplex._END)

        async def fake_speak(text, voice="Kore"):
            return b"\x00" * 44 + b"\x01\x00" * 100, 24000

        monkeypatch.setattr(duplex, "write_draft", fake_write)
        monkeypatch.setattr(api.voice, "speak", fake_speak)

        s.state = "listening"
        s.transcript = "what is a hash"
        await s.draft_for("what is a hash", final=False)
        s.transcript = "what is a hash map"
        s.transcript_at = 0.0  # long quiet: settle returns at once
        s.t_endpoint = 0.0
        s.state = "settling"
        await s._answer_inner()

        events = self._events(s)
        states = [(e.get("state"), e.get("text")) for e in events if e["type"] == "draft"]
        assert ("discarded", "what is a hash") in states
        assert [e["text"] for e in events if e["type"] == "say"] == ["Answer to what is a hash map."]
        assert any(e["type"] == "latency" for e in events)
        assert events[-1]["type"] == "turn_end"
        assert len(s.ws.audio) >= 1
        assert s.history[-2:] == [
            {"role": "user", "content": "what is a hash map"},
            {"role": "assistant", "content": "Answer to what is a hash map."},
        ]

    async def test_words_that_change_before_the_first_sound_restart_the_answer(self, monkeypatch):
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")
        s, api = self._session(monkeypatch)

        async def fake_write(draft, chain, history):
            draft.pieces.put_nowait(f"Re: {draft.for_text}.")
            draft.pieces.put_nowait(duplex._END)

        async def fake_speak(text, voice="Kore"):
            # While the voice is "synthesising", the STT finishes the sentence.
            s.transcript = "what is a hash map"
            return b"\x00" * 44 + b"\x01\x00" * 10, 24000

        monkeypatch.setattr(duplex, "write_draft", fake_write)
        monkeypatch.setattr(api.voice, "speak", fake_speak)

        s.state = "settling"
        s.transcript = "what is a hash"
        s.transcript_at = 0.0
        s.t_endpoint = 0.0
        await s._answer_inner()

        said = [e["text"] for e in self._events(s) if e["type"] == "say"]
        assert said == ["Re: what is a hash map."]  # never the truncated one

    async def test_speaking_again_while_settling_means_they_were_not_finished(self, monkeypatch):
        s, _ = self._session(monkeypatch)
        s.state = "settling"
        s.transcript = "so what I wanted to"
        s._turn = asyncio.create_task(asyncio.sleep(10))
        await s.speech_start()
        assert s.state == "listening"
        assert s.transcript == "so what I wanted to"  # not wiped
        await asyncio.sleep(0)
        assert s._turn.cancelled()

    async def test_barge_in_during_speech_stops_it(self, monkeypatch):
        s, _ = self._session(monkeypatch)
        s.state = "speaking"
        s.spoken = ["First part."]
        await s.speech_start()
        events = self._events(s)
        assert any(e["type"] == "stop" and e["reason"] == "barge-in" for e in events)
        assert s.state == "listening"
        assert s.history[-1] == {"role": "assistant", "content": "First part."}


class TestWhisperPacing:
    """Groq's free Whisper is 20 requests a minute. Spend them where they matter."""

    def _make(self, monkeypatch):
        calls: list[int] = []
        seen: list[str] = []

        async def fake_transcribe(wav, **kw):
            calls.append(len(wav))
            return {"text": f"pass {len(calls)}"}

        monkeypatch.setattr(duplex.groq_client, "transcribe", fake_transcribe)

        async def on_text(t):
            seen.append(t)

        return duplex.WhisperTranscriber(on_text), calls, seen

    async def test_silence_is_never_sent(self, monkeypatch):
        w, calls, _ = self._make(monkeypatch)
        await w.feed(bytes(64000))  # nobody is speaking
        await asyncio.sleep(0)
        assert calls == []

    async def test_passes_during_speech_are_spaced(self, monkeypatch):
        w, calls, _ = self._make(monkeypatch)
        await w.speech_start()
        await w.feed(bytes(32000))  # one second
        await asyncio.sleep(0.01)
        await w.feed(bytes(32000))  # another second, too soon after the first
        await asyncio.sleep(0.01)
        assert len(calls) == 1

    async def test_a_pause_gets_a_pass_even_inside_the_interval(self, monkeypatch):
        w, calls, seen = self._make(monkeypatch)
        await w.speech_start()
        await w.feed(bytes(32000))
        await asyncio.sleep(0.01)
        await w.feed(bytes(9600))
        await w.pass_now()
        await asyncio.sleep(0.01)
        assert len(calls) == 2 and seen[-1] == "pass 2"

    async def test_the_end_adds_a_pass_only_for_audio_not_yet_covered(self, monkeypatch):
        w, calls, _ = self._make(monkeypatch)
        await w.speech_start()
        await w.feed(bytes(32000))
        await asyncio.sleep(0.01)
        await w.pass_now()
        await w.speech_end()
        assert len(calls) == 1  # the pass already covered everything

        await w.speech_resume()
        await w.feed(bytes(32000))
        await w.speech_end()
        assert len(calls) == 2

    async def test_the_per_minute_cap_holds(self, monkeypatch):
        w, calls, _ = self._make(monkeypatch)
        monkeypatch.setattr(duplex.get_settings(), "duplex_whisper_per_minute", 2)
        await w.speech_start()
        for _ in range(5):
            await w.feed(bytes(32000))
            await w.pass_now()
            await asyncio.sleep(0.01)
        assert len(calls) == 2


class TestEarlyAnswer:
    async def test_a_pause_starts_an_answer_with_no_debounce(self, monkeypatch):
        from app.api import duplex as api
        from app.services import keys

        monkeypatch.setattr(keys, "key_for", lambda provider: "k-1234")

        class FakeWS:
            async def send_text(self, t): ...
            async def send_bytes(self, b): ...

        started = []

        async def fake_write(draft, chain, history):
            started.append(draft.for_text)
            await asyncio.sleep(10)

        monkeypatch.setattr(duplex, "write_draft", fake_write)
        s = api.Session(FakeWS(), "Kore")
        s.s.duplex_debounce_ms = 5000  # would make a normal speculation wait
        s.state = "listening"

        class T(duplex.Transcriber):
            async def pass_now(self):
                await self._publish("what is a hash map")

        s.transcriber = T(s.on_text)
        await s.pause()
        await asyncio.sleep(0.05)
        assert started == ["what is a hash map"]
        s.discard_draft("test")


class TestPhantoms:
    def test_a_stock_phrase_from_a_short_clip_is_noise(self):
        assert duplex.is_hallucination("Thank you.", 1.0)
        assert duplex.is_hallucination("I'm sorry.", 1.4)

    def test_the_same_words_in_a_real_sentence_are_kept(self):
        assert not duplex.is_hallucination("Thank you for explaining the hash map", 3.0)
        assert not duplex.is_hallucination("What is a hash map", 1.8)

    def test_a_long_clip_is_trusted_even_if_short_text(self):
        assert not duplex.is_hallucination("Thank you.", 6.0)

    def test_empty_is_noise(self):
        assert duplex.is_hallucination("  ", 1.0)

    def test_whispers_own_doubt_counts_for_short_clips(self):
        assert duplex.is_hallucination("Right then", 1.0, no_speech=0.9)
        assert not duplex.is_hallucination("Right then", 1.0, no_speech=0.1)


class TestPreRoll:
    async def test_the_audio_before_the_turn_opened_is_kept(self, monkeypatch):
        async def noop(t): ...

        w = duplex.WhisperTranscriber(noop)
        await w.feed(b"\x01" * 6400)  # not speaking yet
        await w.speech_start()
        assert bytes(w._buf) == b"\x01" * 6400

    async def test_only_the_last_second_is_kept(self, monkeypatch):
        async def noop(t): ...

        w = duplex.WhisperTranscriber(noop)
        await w.feed(bytes(100_000))
        await w.speech_start()
        assert len(w._buf) == w.PRE_ROLL_BYTES

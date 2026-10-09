"""Manks: what decides whether a link is sent a bot, and who may talk to the queue."""

import pytest
from fastapi import HTTPException

from app.services import manks


class TestParseLink:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://meet.google.com/abc-defg-hij",
            "meet.google.com/abc-defg-hij",
            "  https://meet.google.com/ABC-DEFG-HIJ?authuser=1  ",
            "https://meet.google.com/abc-defg-hij/",
        ],
    )
    def test_a_meet_code_is_canonicalised(self, raw):
        assert manks.parse_link(raw) == ("meet", "https://meet.google.com/abc-defg-hij")

    @pytest.mark.parametrize(
        "raw,fragment",
        [
            ("", "Paste"),
            ("https://zoom.us/j/12345", "Google Meet"),
            ("https://teams.microsoft.com/l/meetup-join/x", "Google Meet"),
            ("https://meet.google.com/", "meeting code"),
            ("https://meet.google.com/lookup/abc", "meeting code"),
            ("https://example.com/abc-defg-hij", "Google Meet or BigBlueButton"),
        ],
    )
    def test_anything_else_says_why(self, raw, fragment):
        with pytest.raises(manks.BadLink) as e:
            manks.parse_link(raw)
        assert fragment in str(e.value)


class TestBigBlueButton:
    JOIN = (
        "https://gatewaymeet.com/bigbluebutton/api/join?meetingID=1c65b096&password=056045e9"
        "&fullName=Tester&userdata-default_caption_locale=en-GB&checksum=cf1651ba52784fc7b911479787fd61027a46e6a2"
    )

    def test_the_join_link_is_kept_exactly_as_pasted(self):
        # Any change breaks the checksum, so nothing is normalised.
        assert manks.parse_link(self.JOIN) == ("bbb", self.JOIN)
        assert manks.parse_link("  " + self.JOIN + "  ") == ("bbb", self.JOIN)

    def test_the_client_url_it_redirects_to_is_accepted(self):
        url = "https://gatewaymeet.com/html5client/join?sessionToken=fdsahtibo56tpwrr"
        assert manks.parse_link(url) == ("bbb", url)

    @pytest.mark.parametrize(
        "raw",
        [
            "https://gatewaymeet.com/bigbluebutton/api/join?meetingID=1c65b096",  # no checksum
            "https://gatewaymeet.com/bigbluebutton/",
            "https://gatewaymeet.com/html5client/join",
        ],
    )
    def test_an_incomplete_link_says_so(self, raw):
        with pytest.raises(manks.BadLink) as e:
            manks.parse_link(raw)
        assert "incomplete" in str(e.value) or "BigBlueButton" in str(e.value)


class TestTranscript:
    def test_silence_does_not_become_a_line(self):
        assert manks._is_phantom("")
        assert manks._is_phantom("Thank you.")
        assert not manks._is_phantom("Thank you for joining, let's start with the budget")

    def test_timestamps(self):
        assert manks._stamp(5) == "00:05"
        assert manks._stamp(125) == "02:05"
        assert manks._stamp(3725) == "1:02:05"


class TestBotSecret:
    """The bot can claim work and upload audio, and it can do nothing else."""

    def _secret(self, monkeypatch, value):
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "manks_bot_secret", value)

    def test_no_secret_configured_refuses_everyone_including_an_empty_header(self, monkeypatch):
        from app.api import manks as api

        self._secret(monkeypatch, "")
        for header in (None, "", "anything"):
            with pytest.raises(HTTPException) as e:
                api._bot(header)
            assert e.value.status_code == 401

    def test_wrong_secret_is_refused(self, monkeypatch):
        from app.api import manks as api

        self._secret(monkeypatch, "right")
        with pytest.raises(HTTPException):
            api._bot("wrong")
        with pytest.raises(HTTPException):
            api._bot(None)

    def test_the_right_secret_passes(self, monkeypatch):
        from app.api import manks as api

        self._secret(monkeypatch, "right")
        assert api._bot("right") is None

    def test_the_bot_router_exposes_no_way_to_read_a_meeting(self):
        """Bot routes are claim / control / status / segment -- never a GET of notes."""
        from app.api import manks as api

        for route in api.router.routes:
            if "/bot/" in route.path:
                assert route.path.endswith(("/claim", "/control", "/status", "/segment", "/talk"))
                assert not (getattr(route, "methods", None) == {"GET"} and not route.path.endswith("/control"))

    def test_user_routes_all_require_a_signed_in_person(self):
        from app.api import manks as api

        for route in api.router.routes:
            if route.path.startswith("/manks/bot/"):
                continue
            deps = {getattr(d.call, "__name__", "") for d in route.dependant.dependencies}
            assert "current_user" in deps, route.path


class TestNotes:
    def test_a_stock_phrase_alone_in_silence_is_dropped(self):
        lines = [
            {"start": 0.0, "end": 1.0, "text": "Thank you."},
            {"start": 30.0, "end": 31.0, "text": "Thank you."},
            {"start": 60.0, "end": 64.0, "text": "Right, so the budget is approved."},
            {"start": 64.5, "end": 65.0, "text": "Thank you."},
        ]
        kept = [l["text"] for l in manks.drop_stranded_phantoms(lines)]
        # The two alone in the quiet go; the one straight after somebody spoke stays.
        assert kept == ["Right, so the budget is approved.", "Thank you."]

    def test_sections_always_partition_the_lines(self):
        raw = [
            {"title": "B", "start_line": 5, "gist": "g", "points": ["a"]},
            {"title": "A", "start_line": 2},
            {"title": "dup", "start_line": 2},
            {"title": "", "start_line": 3},
            {"title": "late", "start_line": 99},
        ]
        out = manks.clean_sections(raw, 8)
        assert [s["start_line"] for s in out] == [0, 5, 7]  # first pulled to 0, 99 clamped
        assert [s["title"] for s in out] == ["A", "B", "late"]

    def test_nothing_usable_still_yields_one_section_over_everything(self):
        assert manks.clean_sections([], 10) == [
            {"title": "The meeting", "start_line": 0, "gist": "", "points": []}
        ]
        assert manks.clean_sections([{"title": "x", "start_line": "n/a"}], 10)[0]["start_line"] == 0

    def test_lines_use_whispers_own_timings_when_there_are_some(self):
        timed = [{"start": 1.0, "end": 3.0, "text": " Hello. "}, {"start": 5.0, "end": 6.0, "text": ""}]
        out = manks.lines_from(120.0, 60.0, "Hello.", timed)
        assert out == [{"start": 121.0, "end": 123.0, "text": "Hello."}]

    def test_without_timings_sentences_are_spread_through_the_segment(self):
        out = manks.lines_from(0.0, 10.0, "One two. Three four five six.", [])
        assert [l["text"] for l in out] == ["One two.", "Three four five six."]
        assert out[0]["start"] == 0.0 and 0 < out[1]["start"] < 10


class TestTalking:
    def test_its_name_in_any_spelling_whisper_gives_it(self):
        from app.services.manks_talk import addressed

        assert addressed("Hey Manks, what's the budget?", "Manks")
        assert addressed("monks can you summarise that", "Manks")
        assert not addressed("I think the monkeys disagree", "Manks")
        assert not addressed("let's move on", "Manks")

    def test_speech_is_cut_out_of_silence(self):
        import array
        import math

        from app.services.manks_talk import FRAME, Utterances

        def tone(frames: int) -> bytes:
            a = array.array("h", (int(8000 * math.sin(i / 5)) for i in range(FRAME * frames)))
            return a.tobytes()

        quiet = bytes(FRAME * 2 * 20)
        cut = Utterances()
        assert cut.feed(quiet) == []
        assert cut.feed(tone(20)) == []  # still talking
        got = cut.feed(quiet)
        assert len(got) == 1
        assert len(got[0]) >= FRAME * 2 * 20  # the speech, plus a little run-up

    def test_a_blip_is_not_an_utterance(self):
        from app.services.manks_talk import FRAME, Utterances

        import array
        import math

        blip = array.array("h", (int(8000 * math.sin(i / 5)) for i in range(FRAME * 3))).tobytes()
        cut = Utterances()
        assert cut.feed(blip + bytes(FRAME * 2 * 30)) == []

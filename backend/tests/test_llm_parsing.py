"""Salvaging usable output from a response that degenerated or truncated.

Gemma's characteristic failure is a repetition loop: a plausible sentence, then
a fragment repeated until it hits max_output_tokens -- which also truncates the
JSON and made a strict parse throw away the good half.

The literal string in `DEGENERATED` below is from a real turn, and it reached
the user as `model returned invalid JSON`. Every test here exists because of it.
"""

import json

import pytest

from app.services.llm import (
    _unescape_newlines,
    extract_bool,
    extract_int_list,
    extract_object_list,
    extract_string,
    extract_string_list,
    is_repetitive,
    strip_degeneration,
)

# Truncated exactly as the model left it: no closing quote, no closing brace.
DEGENERATED = (
    '{"answer": "I am sorry, but the provided sources do not contain '
    "information regarding which specific pyramid you are asking about. "
    "[No source provided for this clarification/refusal/unanswered part of the "
    + "question/" * 40
)

GOOD = '{"answer": "Gross margin reached 62.1% [1].", "sources_used": [1, 3]}'


class TestStripDegeneration:
    def test_cuts_at_the_loop(self):
        text = "The answer is 62.1%. " + "question/" * 30
        out = strip_degeneration(text)
        assert out == "The answer is 62.1%."

    def test_leaves_ordinary_prose_alone(self):
        text = (
            "Gross margin improved to 62.1% in fiscal 2024, driven by a shift "
            "toward subscription revenue, which carries a higher margin than "
            "hardware."
        )
        assert strip_degeneration(text) == text

    def test_does_not_cut_short_legitimate_repeats(self):
        """Four repeats is emphasis; five is a loop. The line has to be
        somewhere, and prose rarely repeats a phrase five times running."""
        assert strip_degeneration("very very very good") == "very very very good"

    def test_returns_empty_when_the_loop_starts_immediately(self):
        """A fragment with no real answer in front of it is not worth showing."""
        assert strip_degeneration("ab" * 50) == ""

    def test_empty_input(self):
        assert strip_degeneration("") == ""

    def test_handles_a_loop_beyond_the_scan_window(self):
        """Only the tail is scanned, and a loop always runs to the end."""
        text = "x" * 3000 + " and then " + "loop/" * 40
        out = strip_degeneration(text)
        assert out.endswith("and then")


class TestExtractString:
    def test_reads_well_formed_json(self):
        assert extract_string(GOOD, "answer") == "Gross margin reached 62.1% [1]."

    def test_salvages_the_real_failure(self):
        """The bug, end to end.

        This exact input produced `model returned invalid JSON` and a 502. The
        first sentence was always usable.
        """
        out = extract_string(DEGENERATED, "answer")
        assert out.startswith("I am sorry, but the provided sources do not")
        assert "question/question" not in out

    def test_missing_key(self):
        assert extract_string(GOOD, "nope") == ""

    def test_unescapes(self):
        assert extract_string(r'{"answer": "line\nbreak \"quoted\""}', "answer") == (
            'line\nbreak "quoted"'
        )

    def test_survives_a_trailing_lone_backslash(self):
        """A truncation can cut mid-escape, which breaks a strict decode."""
        raw = '{"answer": "half an escape \\'
        assert "half an escape" in extract_string(raw, "answer")

    def test_not_json_at_all(self):
        assert extract_string("plain prose, no json here", "answer") == ""


class TestExtractIntList:
    def test_reads_well_formed_json(self):
        assert extract_int_list(GOOD, "sources_used") == [1, 3]

    def test_salvages_an_unclosed_array(self):
        assert extract_int_list('{"a":"x", "sources_used": [2, 5', "sources_used") == [
            2,
            5,
        ]

    def test_missing_key(self):
        assert extract_int_list(GOOD, "nope") == []

    def test_ignores_non_integers(self):
        assert extract_int_list('{"sources_used": [1, "two", 3]}', "sources_used") == [
            1,
            3,
        ]


class TestExtractStringList:
    """Pre-existing behaviour, pinned so the refactor above cannot break it."""

    def test_reads_well_formed_json(self):
        raw = '{"sub_questions": ["What was revenue?", "What was income?"]}'
        assert extract_string_list(raw, "sub_questions") == [
            "What was revenue?",
            "What was income?",
        ]

    def test_keeps_complete_items_from_a_truncated_array(self):
        raw = '{"sub_questions": ["complete one", "complete two", "truncated th'
        assert extract_string_list(raw, "sub_questions") == [
            "complete one",
            "complete two",
        ]


class TestIsRepetitive:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("question " * 20, True),
            ("short and varied", False),  # under the 12-word floor
            (
                "Gross margin improved to 62.1 percent in fiscal 2024 driven by "
                "a shift toward subscription revenue",
                False,
            ),
        ],
    )
    def test_detects_loops(self, text, expected):
        assert is_repetitive(text) is expected


# The literal response that disabled clarification: `ambiguous` and two
# complete options were present, the third description degenerated, and the
# document never closed.
TRUNCATED_CLARIFY = (
    '{"ambiguous":true,"question":"What specifically would you like to know?",'
    '"options":['
    '{"label":"Acme Corporation 2024 Annual Report","description":"Yearly performance."},'
    '{"label":"Financial Summary","description":"Revenue and segments."},'
    '{"label":"Payments Cutover Outage","description":"Information about the about the'
)


class TestExtractObjectList:
    def test_reads_well_formed_json(self):
        raw = '{"options":[{"label":"a"},{"label":"b"}]}'
        assert extract_object_list(raw, "options") == [{"label": "a"}, {"label": "b"}]

    def test_salvages_complete_objects_from_the_real_failure(self):
        """The bug: two good options were discarded with the truncated third."""
        out = extract_object_list(TRUNCATED_CLARIFY, "options")
        assert [o["label"] for o in out] == [
            "Acme Corporation 2024 Annual Report",
            "Financial Summary",
        ]

    def test_ignores_braces_inside_strings(self):
        """A brace in a description must not shift the nesting depth."""
        raw = '{"options":[{"label":"a","description":"uses {braces} inside"}]}'
        assert extract_object_list(raw, "options")[0]["description"] == (
            "uses {braces} inside"
        )

    def test_handles_escaped_quotes(self):
        raw = '{"options":[{"label":"say \\"hi\\""}]}'
        assert extract_object_list(raw, "options") == [{"label": 'say "hi"'}]

    def test_missing_key(self):
        assert extract_object_list('{"other":[]}', "options") == []

    def test_not_json(self):
        assert extract_object_list("prose", "options") == []

    def test_stops_at_the_closing_bracket(self):
        """Objects AFTER the array must not be swept in."""
        raw = '{"options":[{"label":"a"}],"other":{"label":"not an option"}}'
        assert extract_object_list(raw, "options") == [{"label": "a"}]


class TestExtractBool:
    def test_reads_well_formed_json(self):
        assert extract_bool('{"ambiguous":true}', "ambiguous") is True
        assert extract_bool('{"ambiguous":false}', "ambiguous") is False

    def test_salvages_from_the_real_failure(self):
        assert extract_bool(TRUNCATED_CLARIFY, "ambiguous") is True

    def test_default_when_missing(self):
        assert extract_bool('{"x":1}', "ambiguous") is False
        assert extract_bool('{"x":1}', "ambiguous", default=True) is True


class TestLongRunBeyondTheScanWindow:
    """The regression that reached the UI.

    `strip_degeneration` bounds its REGEX with a scan window, not the run. A
    loop longer than the window begins before it, and cutting at the window
    boundary left the earlier part in place -- measured 3364 chars in, 1363
    out, with ~1200 characters of "the-the the-the ..." still in the answer a
    user read. The fix extends the cut backwards while the unit keeps
    repeating.
    """

    GOOD = (
        "The provided sources do not contain information regarding an ACME "
        "PYRAMID built in Egypt. The sources mention mummification practices "
        "[1]. The sources also mention "
    )

    @pytest.mark.parametrize("repeats", [6, 400, 2000])
    def test_cuts_the_whole_run_at_any_length(self, repeats):
        text = self.GOOD + "the-the " * repeats
        assert strip_degeneration(text) == self.GOOD.rstrip()

    def test_run_far_longer_than_the_scan_window(self):
        """16K of loop -- more than 8x the window."""
        text = self.GOOD + "the-the " * 2000
        out = strip_degeneration(text)
        assert "the-the" not in out

    def test_is_repetitive_catches_what_survives(self):
        """Second line of defence: cutting is not judging."""
        assert is_repetitive("the-the " * 50) is True


class TestLiteralNewlineRecovery:
    r"""A model escaping the BACKSLASH instead of the newline.

    Asked to put line breaks in a JSON string field, the drafter emitted
    ``"\\n"`` -- which decodes to the two visible characters ``\`` and ``n``,
    so the answer reached the user reading "records [5] .\n\nRegarding the
    technical operations...".

    Telling it to "write the escape \n" is what CAUSED this: encoding is the
    serialiser's job, so the prompt now asks only for real line breaks. These
    pin the net underneath.
    """

    # Built from chr(92) rather than written as an escape, so nothing between
    # here and the assertion can quietly turn it into a real newline -- which
    # would make this test pass while testing the opposite case.
    LITERAL = chr(92) + "n"

    def test_literal_escapes_become_real_breaks(self):
        broken = f"records [5] .{self.LITERAL}{self.LITERAL}Regarding the operations"
        out = _unescape_newlines(broken)
        assert out.count(chr(10)) == 2
        assert self.LITERAL not in out

    def test_text_with_real_breaks_is_left_alone(self):
        """THE GUARD. A model that produced genuine line breaks was capable of
        it, so a literal escape still sitting in that text is deliberate
        content -- a regex, a Windows path, an explanation of escaping -- and
        rewriting it would corrupt the answer."""
        mixed = f"Real break here:{chr(10)}use {self.LITERAL} to split lines."
        assert _unescape_newlines(mixed) == mixed

    def test_text_with_neither_is_unchanged(self):
        assert _unescape_newlines("no breaks at all") == "no breaks at all"

    def test_extract_string_recovers_the_whole_field(self):
        """End to end through the parser the drafter actually uses."""
        raw = json.dumps({"answer": f"One.{self.LITERAL}{self.LITERAL}Two.", "sources_used": [1]})
        assert extract_string(raw, "answer") == f"One.{chr(10)}{chr(10)}Two."

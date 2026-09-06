"""Salvaging usable output from a response that degenerated or truncated.

Gemma's characteristic failure is a repetition loop: a plausible sentence, then
a fragment repeated until it hits max_output_tokens -- which also truncates the
JSON and made a strict parse throw away the good half.

The literal string in `DEGENERATED` below is from a real turn, and it reached
the user as `model returned invalid JSON`. Every test here exists because of it.
"""

import pytest

from app.services.llm import (
    extract_int_list,
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

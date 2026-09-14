"""Durable preferences: the router's memory, and why it is not the summary.

The interesting logic is deduplication. The router paraphrases every time it
extracts a preference, so exact matching lets one instruction accumulate as
several rows -- each taking a slot in the prompt and making the instruction look
more emphatic than it is.
"""

import pytest

from app.services import preferences as prefs_mod
from app.services.llm import LLMError
from app.services.preferences import (
    MAX_IN_PROMPT,
    _fingerprint,
    _is_near_duplicate,
    _polarity,
    already_covered,
    render_for_prompt,
)


class _Pref:
    """Just enough of the model for rendering."""

    def __init__(self, text: str):
        self.text = text


class TestNearDuplicates:
    @pytest.mark.parametrize(
        "a,b",
        [
            # MEASURED. These two arrived from two turns stating one preference,
            # and were stored as separate rows before this existed.
            (
                "Always let me know what info is from the internet and what is from the docs.",
                "Always let me know what info is from internet and what's from the docs.",
            ),
            ("Always cite page numbers.", "Always cite the page numbers."),
            ("Give me short answers", "Give me short answers."),
        ],
    )
    def test_paraphrases_are_one_preference(self, a, b):
        assert _is_near_duplicate(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            (
                "Always search both the internet and documents, and separate them.",
                "Always let me know what info is from the internet and what is from the docs.",
            ),
            ("Give me short answers.", "Keep answers brief."),
            ("Always cite page numbers.", "Never use bullet points."),
        ],
    )
    def test_different_instructions_are_kept(self, a, b):
        """Merging two genuine preferences loses one silently, which is worse
        than keeping a near-duplicate. Hence the high threshold."""
        assert not _is_near_duplicate(a, b)


class TestNegation:
    """A preference and its opposite are never the same preference.

    Bag-of-words similarity scales with length: one flipped word is a large
    difference in a short instruction and a negligible one in a long sentence.
    Without a polarity check, reversing a long instruction would overlap ~0.82
    and be silently discarded as a duplicate.
    """

    def test_opposites_are_never_merged(self):
        assert not _is_near_duplicate(
            "Never mention the incident report.",
            "Always mention the incident report.",
        )

    def test_opposites_survive_a_long_sentence(self):
        base = "cite the page number and section heading for every claim you make"
        assert not _is_near_duplicate(f"Always {base}.", f"Never {base}.")

    @pytest.mark.parametrize(
        "text", ["Never do that", "Do not do that", "Don't do that", "Avoid bullet lists"]
    )
    def test_prohibitions_are_detected(self, text):
        assert _polarity(text) is True

    def test_plain_instructions_are_not(self):
        assert _polarity("Always cite page numbers") is False


class TestFingerprint:
    def test_ignores_filler(self):
        """Two phrasings of one instruction differ almost entirely in these, so
        comparing with them in makes paraphrases look distinct."""
        assert _fingerprint("the answer is from the docs") == _fingerprint(
            "answer from docs"
        )

    def test_keeps_the_meaningful_words(self):
        assert "citations" in _fingerprint("always include citations")


class TestRenderForPrompt:
    def test_empty_renders_nothing(self):
        """Not a placeholder. A line saying there are no instructions is itself
        an instruction the model reasons about, and small models reliably
        produce a sentence apologising for having no preferences."""
        assert render_for_prompt([]) == ""

    def test_includes_each_instruction(self):
        out = render_for_prompt([_Pref("Always cite pages"), _Pref("Be brief")])
        assert "Always cite pages" in out and "Be brief" in out

    def test_says_they_override_defaults(self):
        """Without it the model treats them as context rather than orders, and
        quietly keeps its own formatting."""
        assert "override" in render_for_prompt([_Pref("Be brief")]).lower()

    def test_capped_so_evidence_is_not_crowded_out(self):
        """Preferences accumulate. Unbounded, they would eventually outweigh
        the retrieved passages the answer actually has to cite."""
        many = [_Pref(f"rule {i}") for i in range(MAX_IN_PROMPT * 3)]
        assert render_for_prompt(many).count("- rule") == MAX_IN_PROMPT


class TestAlreadyCovered:
    """The semantic dedupe that sits in front of `remember`.

    The lexical guard in this module cannot catch a paraphrase that shares no
    content words -- "what came from the internet" against "which facts are
    from the web" overlap almost nowhere -- so a model makes that judgement.
    These pin the behaviour AROUND the call, which is where the risk is: the
    call itself is exercised live, but a wrong answer on an error path would
    silently discard an instruction the user just gave.
    """

    async def test_nothing_stored_makes_no_call(self, monkeypatch):
        """Short-circuits before the LLM. With no preferences there is nothing
        to be a duplicate OF, and a call here would be one per first-ever
        preference for no possible answer but False."""
        called = False

        def _boom(*a, **kw):
            nonlocal called
            called = True
            raise AssertionError("should not reach the model")

        monkeypatch.setattr(prefs_mod, "get_llm", _boom)
        assert await already_covered("Be brief", []) is False
        assert not called

    async def test_fails_open_so_an_error_never_drops_an_instruction(
        self, monkeypatch
    ):
        """An LLM error must NOT read as "duplicate".

        Failing closed would silently discard a preference on a transient 429 --
        the user sees no confirmation and no stored row, with nothing to
        indicate anything went wrong. An extra row is the cheaper failure.
        """

        class _Failing:
            async def generate(self, *a, **kw):
                raise LLMError("429 quota exhausted")

        monkeypatch.setattr(prefs_mod, "get_llm", lambda: _Failing())
        assert await already_covered("Be brief", ["Keep it short"]) is False

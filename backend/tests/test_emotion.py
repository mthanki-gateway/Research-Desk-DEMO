"""Turning per-turn scores into something a person should read.

The model call itself is not tested here -- it needs a gigabyte of torch and a
network fetch. What IS tested is every decision made around it, because those
are the ones that make the output honest or misleading.
"""

import pytest

from app.services import emotion


def _turn(n, arousal=0.5, dominance=0.5, valence=0.5):
    return {
        "turn": n,
        "scores": {"arousal": arousal, "dominance": dominance, "valence": valence},
    }


class TestSummarise:
    def test_nothing_scored_is_not_an_error(self):
        """An interview with no recordings is a finished job, not a failed one."""
        out = emotion.summarise([])
        assert out["turns"] == [] and out["moments"] == []

    def test_the_baseline_is_this_speaker(self):
        """Their own median, never a population average.

        Comparing somebody to a corpus of strangers recorded on other
        equipment measures the microphone as much as the person.
        """
        out = emotion.summarise([_turn(0, arousal=0.2), _turn(1, arousal=0.4),
                                 _turn(2, arousal=0.9)])
        assert out["baseline"]["arousal"] == 0.4

    def test_a_flat_interview_has_no_moments(self):
        """Every turn the same is the honest answer of "nothing stood out".

        A threshold that produces a notable moment for every turn produces
        none worth reading.
        """
        out = emotion.summarise([_turn(n) for n in range(5)])
        assert out["moments"] == []

    def test_a_departure_is_reported_with_its_direction(self):
        out = emotion.summarise(
            [_turn(0), _turn(1), _turn(2, arousal=0.95), _turn(3)]
        )
        spike = [m for m in out["moments"] if m["turn"] == 2]
        assert spike and spike[0]["dimension"] == "arousal"
        assert spike[0]["direction"] == "higher"

    def test_moments_are_ordered_by_size_not_by_time(self):
        """A reader wants the two that stood out, not the list they could
        have read themselves."""
        out = emotion.summarise(
            [_turn(0), _turn(1, valence=0.95), _turn(2), _turn(3, valence=0.72)]
        )
        deltas = [abs(m["delta"]) for m in out["moments"]]
        assert deltas == sorted(deltas, reverse=True)

    def test_the_caveat_travels_with_the_result(self):
        """Stored, not only rendered.

        This output ends up in a profile that outlives the screen it was first
        shown on, and anything reading it should meet the caveat with it.
        """
        out = emotion.summarise([_turn(0), _turn(1, arousal=0.9)])
        assert "not a finding about the person" in out["caveat"]


class TestDimensions:
    def test_it_is_dimensional_rather_than_categorical(self):
        """The whole design decision, pinned.

        Categorical models are mostly trained on ACTED speech and collapse on
        natural conversation -- and a label is a verdict that gets read as
        fact about a person. Three dials are an observation.
        """
        assert emotion.DIMENSIONS == ("arousal", "dominance", "valence")

    def test_the_model_is_the_msp_podcast_one(self):
        """Fine-tuned on spontaneous speech, which an interview resembles."""
        assert "msp-dim" in emotion.MODEL_ID


class TestAvailability:
    def test_it_asks_for_the_extra_rather_than_crashing(self, monkeypatch):
        """torch is ~1GB and is not installed by default.

        The failure has to name the fix: a bare ImportError halfway through a
        background job tells whoever reads the log nothing.
        """
        monkeypatch.setattr(emotion, "_model", None)
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        with pytest.raises(emotion.EmotionUnavailable) as caught:
            emotion._load()
        assert "emotion" in str(caught.value)

"""How somebody sounded, from the audio rather than from an impression.

WHAT THIS IS

`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` takes a raw 16kHz
waveform and returns three numbers in roughly 0..1:

    arousal     how activated -- calm through to agitated
    dominance   how assertive -- tentative through to forceful
    valence     how positive -- unpleasant through to pleasant

DIMENSIONAL, NOT CATEGORICAL, and that is the important decision.

The obvious design is a label: happy, angry, sad. It is wrong here twice over.
Most categorical speech-emotion models are trained on ACTED corpora, where
somebody performs an emotion on cue; they report accuracy in the eighties and
fall apart on natural speech, because nobody in an interview is performing
anger. This model was fine-tuned on MSP-Podcast -- spontaneous speech -- which
is far closer to what an interview sounds like.

And a label is a verdict. "The model says this candidate was angry" is read as
fact by whoever is deciding about them. Three dials are an observation: how
animated, how assertive, how positive somebody sounded, which is the same
register the interviewer's own notes are held to.

CHANGE MATTERS MORE THAN LEVEL. An absolute arousal of 0.6 means very little on
its own -- people differ, microphones differ, rooms differ. Arousal rising
sharply on one question and falling on the next is a real signal, and it is
what the per-turn segmentation is for. The baseline is this speaker's own
median, never a population average.

WHY IT RUNS HERE AND NOT IN THE BROWSER

Turn detection had to be local and instant: it decides something during a live
call. This does not. It reads a recording after the fact, so it has no download
budget, no latency budget, and -- the part that matters most -- it can be RE-RUN
when a better model appears. That is only true because the audio was kept, and
it is the same argument as keeping original uploads.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any

import structlog

from app.config import get_settings

log = structlog.get_logger()

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

#: What the model returns, in the order its head emits them.
DIMENSIONS = ("arousal", "dominance", "valence")

#: How far from this speaker's own median counts as worth mentioning.
#: Deliberately blunt -- these are noisy numbers and a threshold that produces
#: a "notable moment" for every turn produces none worth reading.
NOTABLE_DELTA = 0.12


class EmotionUnavailable(RuntimeError):
    """The model could not be loaded or run."""


_model: Any = None
_processor: Any = None


def _load() -> tuple[Any, Any]:
    """The model and its feature extractor, loaded once per process.

    Imported INSIDE the function on purpose. torch and transformers are a
    gigabyte of dependencies that only this feature needs; importing them at
    module scope would add them to every request path and every test run, and
    make the API refuse to start on a box that cannot hold them.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForAudioClassification, AutoProcessor
    except ImportError as exc:
        raise EmotionUnavailable(
            "Emotion analysis needs torch and transformers. Install the "
            "`emotion` extra, or leave EMOTION_ANALYSIS off."
        ) from exc

    settings = get_settings()
    name = settings.emotion_model or MODEL_ID
    log.info("emotion_model_loading", model=name)
    _processor = AutoProcessor.from_pretrained(name)
    _model = AutoModelForAudioClassification.from_pretrained(name)
    _model.eval()
    log.info("emotion_model_ready", model=name)
    return _model, _processor


def _score_sync(samples, sample_rate: int) -> dict[str, float]:
    """One clip to three numbers. Blocking; callers use a thread."""
    import torch

    model, processor = _load()
    inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).logits.squeeze().tolist()

    values = output if isinstance(output, list) else [output]
    scored = {
        name: round(float(value), 4)
        for name, value in zip(DIMENSIONS, values, strict=False)
    }
    # The head is unbounded in principle even though it was trained to 0..1.
    # Clamping keeps a stray value from rendering as a 140% bar.
    return {k: max(0.0, min(1.0, v)) for k, v in scored.items()}


async def score_clip(samples, sample_rate: int = 16_000) -> dict[str, float]:
    """Arousal, dominance and valence for one clip.

    In a THREAD. This is CPU-bound work of a second or more, and running it on
    the event loop would stall every open WebSocket in the process -- including
    the live conversations this app exists to hold.
    """
    return await asyncio.to_thread(_score_sync, samples, sample_rate)


def summarise(turns: list[dict]) -> dict[str, Any]:
    """Per-turn scores into something worth reading.

    RELATIVE TO THIS SPEAKER, always. A median over their own turns is the
    baseline, so the output says "more animated than they were elsewhere"
    rather than "more animated than average", which would be comparing them to
    a corpus of strangers recorded on other equipment.
    """
    scored = [t for t in turns if t.get("scores")]
    if not scored:
        return {"turns": [], "baseline": {}, "moments": []}

    baseline = {
        dim: round(
            statistics.median([t["scores"].get(dim, 0.0) for t in scored]), 4
        )
        for dim in DIMENSIONS
    }

    moments = []
    for turn in scored:
        for dim in DIMENSIONS:
            delta = turn["scores"].get(dim, 0.0) - baseline[dim]
            if abs(delta) >= NOTABLE_DELTA:
                moments.append(
                    {
                        "turn": turn.get("turn"),
                        "dimension": dim,
                        "delta": round(delta, 4),
                        "direction": "higher" if delta > 0 else "lower",
                    }
                )
    # Biggest departures first: a reader wants the two that stood out, not the
    # chronological list they could have read themselves.
    moments.sort(key=lambda m: abs(m["delta"]), reverse=True)

    return {
        "model": get_settings().emotion_model or MODEL_ID,
        "baseline": baseline,
        "turns": [{"turn": t.get("turn"), **t["scores"]} for t in scored],
        "moments": moments[:6],
        # Said in the output itself, not only in the UI, because this travels:
        # anything reading the stored result should meet the caveat with it.
        "caveat": (
            "Dimensional scores from audio, relative to this speaker's own "
            "median. An impression of delivery, not a finding about the "
            "person; prosody varies by culture, accent, microphone and mood."
        ),
    }

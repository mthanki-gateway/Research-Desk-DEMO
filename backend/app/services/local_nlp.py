"""Small encoder models that run in this process, with no API behind them.

WHAT THIS IS FOR

Everything else in Model Lab calls somebody's server. These two run on the
CPU of the box serving the request, and that is the point of having them: a
~200MB encoder answers a narrow question -- which entities, which intent, how
well does each label fit -- in tens of milliseconds, for nothing, without the
text leaving the machine. Seeing that next to a 120B model over an API makes
the trade concrete.

    GLiNER     zero-shot named-entity recognition. Labels are free text given
               at request time ("invoice number", "medication"), not a fixed
               tag set baked in at training.
    GLiClass   zero-shot classification from the same family. Text plus
               candidate labels in, a score per label out.

EVERY LABEL IS SCORED ON ITS OWN, 0..1, by a sigmoid over its own logit.
Scores do not sum to anything and adding a label does not lower the others.
Both use cases read the same numbers:

    scoring   the JEV-style case: "how far does each of these statements
              hold", every label reported.
    routing   the top label is the route -- but only when it clears a cutoff
              AND beats the runner-up by a margin. Independent scores mean two
              labels can both sit at 1.0 ("charged twice, refund one" is
              billing AND cancel), and a router that silently picks one of a
              tie is wrong half the time with full confidence.

GLiClass's own single-label mode was tried and dropped: it returns only the
winner, which hides exactly the tie and the near miss a routing demo exists to
show.

THE CHECKPOINT MATTERS MORE THAN THE CODE. gliclass-small-v1.0 scored every
label at 0.99+ on every input -- "about hiring" included -- so it was
measuring nothing. base-v3.0 separates them cleanly (risk warning 1.0, about
hiring 0.0 on the same sentence).

LOADED LAZILY, IMPORTED LAZILY. Same reasoning as `emotion.py`: torch and
transformers are a gigabyte that only this feature needs, so they are imported
inside the loader and the API starts fine on an image without them.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import structlog

from app.config import get_settings

log = structlog.get_logger()


class LocalModelUnavailable(RuntimeError):
    """The extra is not installed, or the weights could not be loaded."""


_lock = threading.Lock()
_gliner: Any = None
_gliclass: Any = None


def installed() -> bool:
    try:
        import gliclass  # noqa: F401
        import gliner  # noqa: F401
    except ImportError:
        return False
    return True


def loaded() -> dict[str, bool]:
    return {"gliner": _gliner is not None, "gliclass": _gliclass is not None}


def _missing() -> LocalModelUnavailable:
    return LocalModelUnavailable(
        "Local models need the `local-nlp` extra (gliner, gliclass, torch). "
        "It is in the dev image -- rebuild with `docker compose build api`."
    )


def _load_gliner() -> Any:
    global _gliner
    # A lock, because the first two requests after a restart would otherwise
    # both download and load the weights -- twice the memory, for one model.
    with _lock:
        if _gliner is None:
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise _missing() from exc
            name = get_settings().gliner_model
            log.info("local_model_loading", model=name)
            _gliner = GLiNER.from_pretrained(name)
            log.info("local_model_ready", model=name)
    return _gliner


def _load_gliclass() -> Any:
    global _gliclass
    with _lock:
        if _gliclass is None:
            try:
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise _missing() from exc
            name = get_settings().gliclass_model
            log.info("local_model_loading", model=name)
            _gliclass = ZeroShotClassificationPipeline(
                GLiClassModel.from_pretrained(name),
                AutoTokenizer.from_pretrained(name),
                classification_type="multi-label",
                device="cpu",
            )
            log.info("local_model_ready", model=name)
    return _gliclass


def _entities_sync(text: str, labels: list[str], threshold: float) -> dict[str, Any]:
    model = _load_gliner()
    start = time.perf_counter()
    found = model.predict_entities(text, labels, threshold=threshold)
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "entities": [
            {
                "text": e["text"],
                "label": e["label"],
                "score": round(float(e["score"]), 4),
                "start": int(e["start"]),
                "end": int(e["end"]),
            }
            for e in found
        ],
        "elapsed_ms": round(elapsed, 1),
        "model": get_settings().gliner_model,
    }


def _classify_sync(text: str, labels: list[str]) -> dict[str, Any]:
    pipeline = _load_gliclass()
    start = time.perf_counter()
    # threshold=0 so EVERY label comes back. Filtering is the caller's
    # decision, and a label that scored 0.03 is information -- it is the
    # evidence that the model considered it and rejected it.
    raw = pipeline(text, labels, threshold=0.0)[0]
    elapsed = (time.perf_counter() - start) * 1000
    scores = {r["label"]: round(float(r["score"]), 4) for r in raw}
    ranked = sorted(
        ({"label": label, "score": scores.get(label, 0.0)} for label in labels),
        key=lambda r: r["score"],
        reverse=True,
    )
    return {
        "scores": ranked,
        "elapsed_ms": round(elapsed, 1),
        "model": get_settings().gliclass_model,
    }


# In a THREAD, both of them. A forward pass is tens to hundreds of
# milliseconds of CPU -- the first call far longer, while weights load -- and
# on the event loop that would stall every open Parley socket in the process.


async def entities(text: str, labels: list[str], threshold: float) -> dict[str, Any]:
    return await asyncio.to_thread(_entities_sync, text, labels, threshold)


async def classify(text: str, labels: list[str]) -> dict[str, Any]:
    return await asyncio.to_thread(_classify_sync, text, labels)

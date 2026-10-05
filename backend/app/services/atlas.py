"""The corpus as geometry: a 3D projection and a similarity matrix.

WHAT THIS IS FOR

Chat answers "what does the material say". Neither of these does that. They
answer questions about the SHAPE of the corpus, which retrieval quality
depends on and which nothing in the app currently shows:

    - is a document an outlier, sitting nowhere near anything else?
    - are two chunks near-duplicates, so retrieval wastes a slot on both?
    - is a chunk isolated -- similar to nothing, often a chunking failure?

Both views are computed from the vectors that already exist. No embedding
calls, no model calls.

WHY PCA AND NOT UMAP

UMAP and t-SNE preserve local neighbourhood structure better, and both would
mean a large dependency -- umap-learn pulls numba and llvmlite, sklearn is
~60MB -- for a corpus of a few dozen chunks. PCA is a couple of numpy calls,
deterministic run to run, and has one property the others lack: its axes are
real directions in the embedding space, so distance in the plot means
something rather than being an artefact of the optimiser's random seed.

The honest limitation is that PCA is LINEAR. It will show you gross structure
-- one document sitting apart from the rest -- and will not tease apart
clusters that are curled around each other in 768 dimensions. The explained
variance is returned so the UI can say how much of the real structure the
picture actually accounts for, rather than implying a faithful map.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from app.services.embeddings import get_embeddings
from app.services.vectorstore import get_vector_store

log = structlog.get_logger()

# Above this, neither view is useful and both get expensive: the matrix is
# O(n^2) and a scatter plot of ten thousand points is a solid block.
MAX_POINTS = 2_000

# One build at a time. The server runs a single worker, and two concurrent
# builds of a full corpus double the peak memory for no benefit -- a second
# reader is better served waiting a second than taking the process down.
_BUILD_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class Basis:
    """The projection itself, kept so other vectors can enter the same picture.

    A PCA is a mean and a set of axes. Reprojecting a query with a basis fitted
    to the query would place it in a DIFFERENT space that happens to have three
    dimensions -- the coordinates would be meaningless next to the corpus, and
    convincingly so, because the picture would still look like a picture.
    """

    mean: np.ndarray
    components: np.ndarray
    span: float


def _project(vectors: np.ndarray) -> tuple[np.ndarray, list[float], Basis]:
    """PCA to three dimensions. Returns coordinates, explained variance, basis.

    Implemented with SVD rather than an eigendecomposition of the covariance
    matrix: the covariance route squares the condition number, which on 768
    dimensions and few samples loses precision for no gain.
    """
    mean = vectors.mean(axis=0, keepdims=True)
    centred = vectors - mean
    # full_matrices=False: only the first min(n, d) components exist anyway,
    # and asking for the full 768x768 basis would allocate it.
    _, singular, components = np.linalg.svd(centred, full_matrices=False)

    k = min(3, components.shape[0])
    coords = centred @ components[:k].T

    variance = singular**2
    total = float(variance.sum()) or 1.0
    explained = [float(v / total) for v in variance[:k]]

    # Scaled to a unit-ish box so the frontend camera does not have to guess a
    # sensible zoom for an arbitrary embedding space. Shape is preserved
    # because it is ONE divisor for all axes -- scaling each axis separately
    # would stretch the projection and invent structure.
    span = float(np.abs(coords).max()) or 1.0
    coords = coords / span

    # Pad if there were fewer than 3 components (a corpus of two chunks).
    if k < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - k)))
        explained += [0.0] * (3 - k)
    return coords, explained, Basis(mean=mean, components=components[:k], span=span)


def project_into(basis: Basis, vector: np.ndarray) -> list[float]:
    """Place one more vector in an existing projection.

    Same mean, same axes, same divisor as the corpus -- so a query lands where
    it truly sits relative to the chunks, rather than at an arbitrary point
    that merely looks plausible.
    """
    coords = (vector.reshape(1, -1) - basis.mean) @ basis.components.T
    coords = (coords / basis.span).ravel()
    out = [float(v) for v in coords]
    return (out + [0.0, 0.0, 0.0])[:3]


def _similarity(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of every chunk against every other.

    The embeddings are already L2-normalised by the provider, but they are
    normalised here again anyway: relying on that is a silent dependency on a
    provider's behaviour, and getting it wrong turns cosine into a dot product
    that reports magnitude as similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def _encode_matrix(matrix: np.ndarray) -> str:
    """The matrix as base64 of row-major uint8, cosine 0..1 mapped to 0..255.

    WHY NOT A LIST OF LISTS. That is what this used to return, and it is what
    took the server down on a large corpus: 2,000 chunks is four million boxed
    Python floats (~150MB) rounded one at a time ON THE EVENT LOOP, then ~25MB
    of JSON -- and the single worker answers nothing else meanwhile. As bytes
    it is 4MB, built in one numpy call. A step of 1/255 is finer than the
    colour ramp can show; negatives clamp to 0, which the ramp's floor renders
    as "unrelated" anyway.
    """
    q = np.clip(np.rint(matrix * 255.0), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


async def _load(owner_id: str | None) -> list[tuple[list[float], dict]]:
    """Every vector, sorted by document then position.

    Sorted so the similarity matrix has its documents in contiguous blocks.
    Without this the matrix is a random permutation of itself and the block
    structure -- the thing worth seeing -- is invisible.
    """
    raw = await get_vector_store().all_vectors(owner_id, limit=MAX_POINTS)
    raw.sort(
        key=lambda pair: (
            str(pair[1].get("filename") or ""),
            int(pair[1].get("chunk_index") or 0),
        )
    )
    return raw


def _compute(
    raw: list[tuple[list[float], dict]], with_matrix: bool
) -> tuple[list[dict[str, Any]], list[float], Basis, str | None]:
    """The CPU half of a build. Synchronous, so it can run off the event loop."""
    vectors = np.asarray([v for v, _ in raw], dtype=np.float32)
    coords, explained, basis = _project(vectors)

    # Nearest OTHER chunk for every chunk, in one vectorised pass. Excluding
    # the diagonal is the whole trick: a chunk's similarity to itself is 1.0
    # and would win every time.
    matrix = _similarity(vectors)
    n = matrix.shape[0]
    nearest_idx = nearest_score = None
    if n >= 2:
        np.fill_diagonal(matrix, -1.0)
        nearest_idx = matrix.argmax(axis=1)
        nearest_score = matrix[np.arange(n), nearest_idx]
        np.fill_diagonal(matrix, 1.0)
    encoded = _encode_matrix(matrix) if with_matrix else None
    del matrix

    points = []
    for i, (_, payload) in enumerate(raw):
        text = str(payload.get("text") or "")
        nearest = None
        if nearest_idx is not None:
            other = raw[int(nearest_idx[i])][1]
            nearest = {
                "filename": str(other.get("filename") or "unknown"),
                "heading": other.get("heading"),
                "chunk_index": int(other.get("chunk_index") or 0),
                "score": round(float(nearest_score[i]), 3),
            }
        points.append(
            {
                "chunk_id": str(payload.get("chunk_id") or ""),
                "document_id": str(payload.get("document_id") or ""),
                "filename": str(payload.get("filename") or "unknown"),
                "heading": payload.get("heading"),
                "chunk_index": int(payload.get("chunk_index") or 0),
                "n_chars": int(payload.get("n_chars") or len(text)),
                # Enough to recognise the chunk on hover.
                "preview": text[:160],
                # The WHOLE passage, for the expanded reading card. Sent up
                # front because the card lives inside the fullscreen canvas,
                # where a click-time request would need its own loading state
                # over a WebGL surface. ~500 chars a chunk: 2,000 is ~1MB.
                "text": text,
                "x": float(coords[i][0]),
                "y": float(coords[i][1]),
                "z": float(coords[i][2]),
                "nearest": nearest,
            }
        )
    return points, explained, basis, encoded


async def _build(
    owner_id: str | None, with_matrix: bool
) -> tuple[dict[str, Any], Basis | None]:
    async with _BUILD_LOCK:
        raw = await _load(owner_id)
        if not raw:
            return {
                "points": [],
                "similarity": None,
                "explained_variance": [],
                "n_documents": 0,
                "truncated": False,
            }, None
        # Off the event loop: the single worker must keep answering /health
        # and every other request while a large corpus is cross-multiplied.
        points, explained, basis, encoded = await asyncio.to_thread(
            _compute, raw, with_matrix
        )
    log.info("atlas_built", n=len(points), explained=round(sum(explained), 3))
    return {
        "points": points,
        # base64 uint8, row-major, n x n -- see _encode_matrix. None on the
        # ray endpoint, whose view never draws it.
        "similarity": encoded,
        "explained_variance": explained,
        "n_documents": len({p["filename"] for p in points}),
        "truncated": len(raw) >= MAX_POINTS,
    }, basis


async def build(owner_id: str | None) -> dict[str, Any]:
    """Project the corpus and cross-multiply it. One pass over the vectors."""
    atlas, _ = await _build(owner_id, with_matrix=True)
    return atlas


async def build_ray(
    owner_id: str | None,
    question: str,
    sources: list[dict],
) -> dict[str, Any]:
    """Where one question landed, and what it pulled in.

    WHAT THIS ANSWERS THAT A SCORE LIST CANNOT

    A retrieval score of 0.68 looks identical in two completely different
    situations: the query sat inside a dense cluster and the top-k are all
    neighbours of one another, or the query sat in empty space between three
    documents and dragged one straggler out of each. The first is a good answer
    waiting to happen; the second is a bad one. The number does not distinguish
    them. The geometry does, immediately.

    The retrieved set comes from what was ACTUALLY stored on the answer -- not
    from re-running retrieval now. Re-running would quietly show a different
    picture whenever the corpus or the index had moved on, and would be most
    misleading in exactly the case someone opens this for: working out why an
    old answer was wrong.

    One embedding call, to place the query. Nothing else is recomputed.
    """
    # One fetch, one fit. This used to call build() and then fetch and refit
    # every vector a second time for the basis -- and serialise a similarity
    # matrix the ray view never draws.
    atlas, basis = await _build(owner_id, with_matrix=False)
    points = atlas["points"]
    if not points or basis is None:
        return {**atlas, "query": None, "rays": [], "question": question}

    embedded = await get_embeddings().embed_query(question)
    position = project_into(basis, np.asarray(embedded, dtype=np.float32))

    # Index by chunk id so a ray can point at a position already computed,
    # rather than re-deriving one and risking a second, disagreeing answer.
    by_chunk = {p["chunk_id"]: i for i, p in enumerate(points)}

    rays = []
    for rank, hit in enumerate(sources, start=1):
        chunk_id = str(hit.get("chunk_id") or "")
        index = by_chunk.get(chunk_id)
        if index is None:
            # A web result, or a chunk deleted since the answer was written.
            # Skipped rather than faked: there is no honest position for it.
            continue
        rays.append(
            {
                "index": index,
                "rank": rank,
                "score": float(hit.get("score") or 0.0),
                "filename": str(hit.get("filename") or "unknown"),
                "heading": hit.get("heading"),
                "chunk_index": int(hit.get("chunk_index") or 0),
            }
        )

    # Everything the query was near but did NOT retrieve. These are the recall
    # failures, and they appear in no log anywhere: a near miss and a distant
    # miss are both simply absent from the results.
    retrieved = {r["index"] for r in rays}
    distances = [
        (i, float(np.linalg.norm(np.asarray([p["x"], p["y"], p["z"]]) - position)))
        for i, p in enumerate(points)
        if i not in retrieved
    ]
    distances.sort(key=lambda pair: pair[1])
    near_misses = [{"index": i, "distance": round(d, 4)} for i, d in distances[:5]]

    log.info("atlas_ray_built", rays=len(rays), dropped=len(sources) - len(rays))
    return {
        **atlas,
        "question": question,
        "query": {"x": position[0], "y": position[1], "z": position[2]},
        "rays": rays,
        "near_misses": near_misses,
        # How many stored sources had no point to attach to, so the UI can say
        # so rather than silently drawing fewer rays than there were citations.
        "dropped": len(sources) - len(rays),
    }

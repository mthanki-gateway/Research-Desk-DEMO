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

from typing import Any

import numpy as np
import structlog

from app.services.vectorstore import get_vector_store

log = structlog.get_logger()

# Above this, neither view is useful and both get expensive: the matrix is
# O(n^2) and a scatter plot of ten thousand points is a solid block.
MAX_POINTS = 2_000


def _project(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """PCA to three dimensions. Returns coordinates and explained variance.

    Implemented with SVD rather than an eigendecomposition of the covariance
    matrix: the covariance route squares the condition number, which on 768
    dimensions and few samples loses precision for no gain.
    """
    centred = vectors - vectors.mean(axis=0, keepdims=True)
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
    return coords, explained


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


async def build(owner_id: str | None) -> dict[str, Any]:
    """Project the corpus and cross-multiply it. One pass over the vectors."""
    raw = await get_vector_store().all_vectors(owner_id, limit=MAX_POINTS)
    if not raw:
        return {
            "points": [],
            "similarity": [],
            "explained_variance": [],
            "n_documents": 0,
            "truncated": False,
        }

    # Sorted by document then position, so the similarity matrix has its
    # documents in contiguous blocks. Without this the matrix is a random
    # permutation of itself and the block structure -- the thing worth seeing
    # -- is invisible.
    raw.sort(
        key=lambda pair: (
            str(pair[1].get("filename") or ""),
            int(pair[1].get("chunk_index") or 0),
        )
    )

    vectors = np.asarray([v for v, _ in raw], dtype=np.float32)
    coords, explained = _project(vectors)
    matrix = _similarity(vectors)

    points = []
    for i, (_, payload) in enumerate(raw):
        text = str(payload.get("text") or "")
        points.append(
            {
                "chunk_id": str(payload.get("chunk_id") or ""),
                "document_id": str(payload.get("document_id") or ""),
                "filename": str(payload.get("filename") or "unknown"),
                "heading": payload.get("heading"),
                "chunk_index": int(payload.get("chunk_index") or 0),
                "n_chars": int(payload.get("n_chars") or len(text)),
                # Enough to recognise the chunk on hover. The full text is a
                # click away through the existing chunk panel, and sending all
                # of it here would make this response tens of times larger for
                # a tooltip.
                "preview": text[:160],
                "x": float(coords[i][0]),
                "y": float(coords[i][1]),
                "z": float(coords[i][2]),
                # Nearest OTHER chunk, precomputed. The UI wants it per point
                # and computing it there means shipping the whole matrix to
                # find one number.
                "nearest": _nearest(matrix, i, raw),
            }
        )

    log.info(
        "atlas_built",
        n=len(points),
        explained=round(sum(explained), 3),
    )
    return {
        "points": points,
        # Rounded to 3dp: the matrix is n^2 floats and full precision triples
        # the payload for a value rendered as a colour.
        "similarity": [[round(float(x), 3) for x in row] for row in matrix],
        "explained_variance": explained,
        "n_documents": len({p["filename"] for p in points}),
        "truncated": len(raw) >= MAX_POINTS,
    }


def _nearest(matrix: np.ndarray, i: int, raw: list) -> dict[str, Any] | None:
    """The most similar chunk to `i`, excluding itself.

    Excluding the diagonal is the whole trick: a chunk's similarity to itself
    is 1.0 and would win every time.
    """
    if matrix.shape[0] < 2:
        return None
    row = matrix[i].copy()
    row[i] = -1.0
    j = int(np.argmax(row))
    payload = raw[j][1]
    return {
        "filename": str(payload.get("filename") or "unknown"),
        "heading": payload.get("heading"),
        "chunk_index": int(payload.get("chunk_index") or 0),
        "score": round(float(row[j]), 3),
    }

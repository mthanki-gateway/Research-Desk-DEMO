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
    coords, explained, _ = _project(vectors)
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
                # Enough to recognise the chunk on hover.
                "preview": text[:160],
                # The WHOLE passage, for the expanded reading card.
                #
                # Sent up front rather than fetched on click, because the card
                # lives inside the fullscreen canvas: a request at click time
                # would need its own loading state over a WebGL surface, and
                # the saving is small. Chunks are ~500 characters, so a
                # 200-chunk corpus is about 100KB -- less than one photograph.
                "text": text,
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
    atlas = await build(owner_id)
    points = atlas["points"]
    if not points:
        return {**atlas, "query": None, "rays": [], "question": question}

    # Refit the basis on the same vectors `build` used. Recomputing rather than
    # threading it out of `build` keeps that function's return value a plain
    # serialisable dict; the SVD is the cheap half of this endpoint next to the
    # embedding round trip.
    raw = await get_vector_store().all_vectors(owner_id, limit=MAX_POINTS)
    raw.sort(
        key=lambda pair: (
            str(pair[1].get("filename") or ""),
            int(pair[1].get("chunk_index") or 0),
        )
    )
    vectors = np.asarray([v for v, _ in raw], dtype=np.float32)
    _, _, basis = _project(vectors)

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

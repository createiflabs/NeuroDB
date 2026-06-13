"""Vector similarity metrics.

Every metric is expressed as a *score* where **higher means more similar**, so
the search layer can treat all metrics uniformly (just take the largest scores).
"""

from __future__ import annotations

import numpy as np

SUPPORTED_METRICS: tuple[str, ...] = ("cosine", "dot", "euclidean")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def compute_scores(matrix: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """Return a 1-D array of similarity scores for ``query`` against ``matrix``.

    ``matrix`` has shape ``(N, D)`` and ``query`` has shape ``(D,)``. Higher
    scores are always better. For ``euclidean`` the score is the negative L2
    distance so that the ranking direction matches the other metrics.
    """

    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    matrix = matrix.astype(np.float32, copy=False)
    query = query.astype(np.float32, copy=False).reshape(-1)

    if metric == "cosine":
        normed = _normalize(matrix)
        q = query / (np.linalg.norm(query) or 1.0)
        return (normed @ q).astype(np.float32)
    if metric == "dot":
        return (matrix @ query).astype(np.float32)
    if metric == "euclidean":
        diff = matrix - query
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return (-dist).astype(np.float32)

    raise ValueError(f"Unsupported metric: {metric!r}. Choose from {SUPPORTED_METRICS}.")


def score_to_distance(score: float, metric: str) -> float:
    """Convert an internal score back into an intuitive distance value."""

    if metric == "euclidean":
        return float(-score)
    if metric == "cosine":
        return float(1.0 - score)
    # Dot product has no natural distance; expose the negated score.
    return float(-score)

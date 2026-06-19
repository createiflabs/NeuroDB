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


def compute_scores(
    matrix: np.ndarray,
    query: np.ndarray,
    metric: str,
    norms: np.ndarray | None = None,
) -> np.ndarray:
    """Return a 1-D array of similarity scores for ``query`` against ``matrix``.

    ``matrix`` has shape ``(N, D)`` and ``query`` has shape ``(D,)``. Higher
    scores are always better. For ``euclidean`` the score is the negative L2
    distance so that the ranking direction matches the other metrics.

    For ``cosine``, precomputed per-row ``norms`` (shape ``(N,)``) may be passed
    to avoid re-normalizing the whole matrix on every query — the caller is
    responsible for keeping them in sync with ``matrix``.
    """

    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    matrix = matrix.astype(np.float32, copy=False)
    query = query.astype(np.float32, copy=False).reshape(-1)

    if metric == "cosine":
        # cos(x, q) = (x · q) / (|x| |q|) — one matvec + an elementwise divide,
        # with no full-matrix normalization allocation per query.
        qnorm = float(np.linalg.norm(query)) or 1.0
        row_norms = np.linalg.norm(matrix, axis=1) if norms is None else norms
        denom = np.where(row_norms == 0.0, 1.0, row_norms) * qnorm
        return ((matrix @ query) / denom).astype(np.float32)
    if metric == "dot":
        return (matrix @ query).astype(np.float32)
    if metric == "euclidean":
        diff = matrix - query
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return (-dist).astype(np.float32)

    raise ValueError(f"Unsupported metric: {metric!r}. Choose from {SUPPORTED_METRICS}.")


def compute_scores_batch(
    matrix: np.ndarray,
    queries: np.ndarray,
    metric: str,
    norms: np.ndarray | None = None,
) -> np.ndarray:
    """Score a *batch* of queries against ``matrix`` with one shared matmul.

    ``matrix`` is ``(N, D)`` and ``queries`` is ``(B, D)``; returns ``(N, B)``
    scores (higher = better). The dominant cost — ``matrix @ queries.T`` — is
    computed once for the whole batch instead of once per query.
    """

    b = int(queries.shape[0])
    if matrix.shape[0] == 0:
        return np.empty((0, b), dtype=np.float32)

    matrix = matrix.astype(np.float32, copy=False)
    q = queries.astype(np.float32, copy=False)
    sims = matrix @ q.T  # (N, B) — the single shared matmul

    if metric == "dot":
        return sims.astype(np.float32)
    if metric == "cosine":
        row_norms = np.linalg.norm(matrix, axis=1) if norms is None else norms
        row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        qnorms = np.linalg.norm(q, axis=1)
        qnorms = np.where(qnorms == 0.0, 1.0, qnorms)
        return (sims / row_norms[:, None] / qnorms[None, :]).astype(np.float32)
    if metric == "euclidean":
        xsq = np.einsum("ij,ij->i", matrix, matrix)[:, None]
        qsq = np.einsum("ij,ij->i", q, q)[None, :]
        dist2 = np.maximum(xsq + qsq - 2.0 * sims, 0.0)
        return (-np.sqrt(dist2)).astype(np.float32)

    raise ValueError(f"Unsupported metric: {metric!r}. Choose from {SUPPORTED_METRICS}.")


def score_to_distance(score: float, metric: str) -> float:
    """Convert an internal score back into an intuitive distance value."""

    if metric == "euclidean":
        return float(-score)
    if metric == "cosine":
        return float(1.0 - score)
    # Dot product has no natural distance; expose the negated score.
    return float(-score)

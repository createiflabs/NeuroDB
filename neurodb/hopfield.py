"""Modern Hopfield network retrieval — the heart of NeuroDB.

Modern Hopfield networks (Ramsauer et al., 2020, *"Hopfield Networks is All You
Need"*) store patterns as the rows of a matrix ``X`` and retrieve with a single
attention step::

    p   = softmax(beta * X @ q)     # how strongly the query attends to each pattern
    x*  = Xᵀ @ p                    # the retrieved (reconstructed) pattern

``beta`` is the inverse temperature. With a large ``beta`` the softmax
concentrates on the single most similar stored pattern, giving exact
content-addressable recall and pattern completion; with a small ``beta`` the
result is a soft blend of patterns (a metastable state). Writing a pattern is
simply appending a row to ``X`` — there is no training.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""

    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_weights(
    X: np.ndarray, query: np.ndarray, beta: float, mask: np.ndarray | None = None
) -> np.ndarray:
    """Softmax attention of ``query`` over the stored patterns ``X``.

    For a 1-D ``query`` returns a 1-D array ``p`` of length ``N`` that sums to 1:
    the weight the query places on each stored pattern. For a 2-D ``query``
    ``(M, D)`` returns an ``(M, N)`` array, each row a distribution — the
    vectorized batch path (one matmul, no Python loop). When ``mask`` (a boolean
    array over feature dimensions) is given, similarity is measured over the
    known dimensions only — the basis for pattern completion.
    """

    if query.ndim == 2:
        if X.shape[0] == 0:
            return np.zeros((query.shape[0], 0), dtype=np.float32)
        # sims[m, n] = X[n] · query[m]; one (M,D)·(D,N) matmul over the batch.
        if mask is not None and mask.any():
            sims = query[:, mask] @ X[:, mask].T
        else:
            sims = query @ X.T
        return softmax(beta * sims.astype(np.float64), axis=1).astype(np.float32)

    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    # Measure similarity over the known dimensions whenever any are flagged
    # (a full mask reduces to the full dot product).
    if mask is not None and mask.any():
        sims = X[:, mask] @ query[mask]
    else:
        sims = X @ query
    return softmax(beta * sims.astype(np.float64)).astype(np.float32)


def retrieve(
    X: np.ndarray,
    query: np.ndarray,
    beta: float,
    mask: np.ndarray | None = None,
    steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the Modern Hopfield update and return ``(reconstruction, weights)``.

    ``reconstruction`` is the retrieved pattern (length ``d``). With a ``mask``,
    the known dimensions of ``query`` are clamped between steps and only the
    unknown dimensions are completed, so the result is the completed pattern.
    ``weights`` is the final attention distribution over stored patterns.
    """

    if query.ndim == 2:
        return _retrieve_batch(X, query, beta, mask, steps)

    q = query.astype(np.float32).copy()
    # A mask with any known dimension clamps those dims (a full mask clamps all,
    # so the result equals the input — nothing is left to complete).
    active = mask is not None and bool(mask.any())
    weights = attention_weights(X, q, beta, mask)
    recon = (X.T @ weights).astype(np.float32)
    for _ in range(max(1, steps) - 1):
        if active:
            q = recon.copy()
            q[mask] = query[mask]  # clamp the known fields, complete the rest
        else:
            q = recon
        weights = attention_weights(X, q, beta, mask)
        recon = (X.T @ weights).astype(np.float32)

    if active:
        completed = recon.copy()
        completed[mask] = query[mask]
        return completed, weights
    return recon, weights


def _retrieve_batch(
    X: np.ndarray,
    queries: np.ndarray,
    beta: float,
    mask: np.ndarray | None,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Hopfield update over a batch of queries ``(M, D)``.

    Mirrors the 1-D :func:`retrieve` exactly, but every step is a single matmul
    over the whole batch: weights ``(M, N) = softmax(β · Q·Xᵀ)`` and
    reconstruction ``(M, D) = weights·X``. Returns ``(reconstructions, weights)``.
    """

    q = queries.astype(np.float32).copy()
    known = queries  # the original known fields, for clamping under a mask
    active = mask is not None and bool(mask.any())
    weights = attention_weights(X, q, beta, mask)
    recon = (weights @ X).astype(np.float32)
    for _ in range(max(1, steps) - 1):
        if active:
            q = recon.copy()
            q[:, mask] = known[:, mask]  # clamp known fields across the batch
        else:
            q = recon
        weights = attention_weights(X, q, beta, mask)
        recon = (weights @ X).astype(np.float32)

    if active:
        completed = recon.copy()
        completed[:, mask] = known[:, mask]
        return completed, weights
    return recon, weights

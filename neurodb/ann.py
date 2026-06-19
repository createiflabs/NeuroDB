"""Optional approximate-nearest-neighbour pre-filter (Tier 1.1).

Brute-force similarity is ``O(N·d)`` per query — fine at 10k rows, untenable at
millions. This wraps `hnswlib` to pre-select the top-``M`` candidate rows so a
query runs the *exact* scoring step on a small subset instead of the full
matrix. Approximate search is **opt-in per query** (``approx=True``); exact
remains the default and the source of truth, so results are unchanged unless a
caller explicitly trades a little recall for a lot of speed.

`hnswlib` is an optional dependency (``pip install neurodb[ann]``); importing
this module never fails, but constructing an :class:`ANNIndex` without it raises
:class:`ANNUnavailable`.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency
    import hnswlib

    _HAVE_HNSWLIB = True
except ImportError:  # pragma: no cover - exercised only where hnswlib is absent
    _HAVE_HNSWLIB = False


class ANNUnavailable(RuntimeError):
    """Approximate search was requested but `hnswlib` is not installed."""


def ann_available() -> bool:
    return _HAVE_HNSWLIB


class ANNIndex:
    """A cosine HNSW index over the rows of a matrix (row index == label).

    The index is a *candidate generator* only: it returns row indices whose exact
    scores the caller then recomputes, so approximate ranking errors inside the
    index never leak into returned scores — they only affect which rows are
    considered (recall), which the ``approx_within_tolerance_of_exact`` test bounds.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        ef_construction: int = 200,
        m: int = 16,
        ef: int = 64,
    ) -> None:
        if not _HAVE_HNSWLIB:
            raise ANNUnavailable(
                "approximate search requires hnswlib (pip install 'neurodb[ann]')."
            )
        n, d = int(matrix.shape[0]), int(matrix.shape[1])
        self.size = n
        self._ef = max(ef, 1)
        self._index = hnswlib.Index(space="cosine", dim=d)
        self._index.init_index(max_elements=max(n, 1), ef_construction=ef_construction, M=m)
        if n:
            self._index.add_items(
                np.ascontiguousarray(matrix, dtype=np.float32), np.arange(n, dtype=np.int64)
            )
        self._index.set_ef(self._ef)

    def query(self, q: np.ndarray, m: int) -> np.ndarray:
        """Return up to ``m`` candidate row indices nearest ``q`` (cosine)."""

        if self.size == 0 or m <= 0:
            return np.empty((0,), dtype=np.int64)
        m = min(int(m), self.size)
        # ef must be >= the number of neighbours requested for a valid query.
        if self._ef < m:
            self._ef = m
            self._index.set_ef(self._ef)
        query = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)
        labels, _ = self._index.knn_query(query, k=m)
        return labels[0].astype(np.int64)

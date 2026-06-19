"""Tier 1.4 — ingest must be amortized O(N), not O(N^2).

The old engine rebuilt the whole matrix with ``np.vstack([self._X, block])`` on
every write, so loading a dataset one record at a time copied ~N^2/2 rows. The
fix is a geometric-growth backing buffer (amortized O(1) append). We prove the
property directly with the growth-copy counter rather than wall-clock timing,
which would be flaky in CI.
"""

from __future__ import annotations

import numpy as np

from neurodb.store import Memory


def test_bulk_append_is_linear_not_quadratic():
    d = 8
    n = 4000
    mem = Memory("bulk", d)
    for i in range(n):
        mem.write([{"vector": np.full(d, i, dtype=np.float32)}])

    assert mem.count == n
    # Geometric growth copies ~N rows total over the whole load; the old
    # vstack-per-append copied ~N^2/2. 3*N cleanly separates linear from
    # quadratic (quadratic here would be ~8,000,000).
    assert mem._grow_copies <= 3 * n, mem._grow_copies
    # Capacity stays within a small constant factor of N (geometric, not exact-fit).
    assert mem._buf.shape[0] < 2 * n + 16
    # Data integrity survives every reallocation.
    assert mem.get(mem.ids[0])["vector"][0] == 0.0
    assert mem.get(mem.ids[-1])["vector"][0] == float(n - 1)
    assert mem._X.shape == (n, d)


def test_bulk_append_single_call_is_one_block():
    """K rows in one write() call cost one allocation, no per-row copying."""

    d = 4
    mem = Memory("bulk1", d)
    before = mem._grow_copies
    ids = mem.write([{"vector": [i, i, i, i]} for i in range(1000)])
    assert mem.count == 1000
    assert len(ids) == 1000
    # Growing from empty copies zero existing rows.
    assert mem._grow_copies == before
    assert mem._X.shape == (1000, d)

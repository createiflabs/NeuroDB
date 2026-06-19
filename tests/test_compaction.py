"""Tier 3.1 — delete compacts the matrix (reclaims space, no tombstones) and
the ANN index reindexes after the mutation."""

from __future__ import annotations

import numpy as np

from neurodb.ann import ann_available
from neurodb.store import Memory


def test_compaction_reclaims_space_and_reindexes():
    rng = np.random.default_rng(0)
    d = 16
    mem = Memory("c", d)
    mem.write([{"id": str(i), "vector": rng.normal(size=d)} for i in range(1000)])

    if ann_available():
        mem.search(rng.normal(size=d), k=5, approx=True)  # build the ANN index

    removed = mem.delete([str(i) for i in range(0, 1000, 2)])  # drop the evens
    assert removed == 500
    assert mem.count == 500
    # Space is reclaimed: the backing buffer holds exactly the survivors, not
    # tombstones (capacity == count, matrix shrank).
    assert mem._buf.shape[0] == 500
    assert mem._X.shape == (500, d)
    # Survivors stay correctly indexed; deleted ids are gone.
    assert "0" not in mem._index
    assert mem.get("1")["vector"] is not None

    if ann_available():
        results = mem.search(rng.normal(size=d), k=5, approx=True)
        # The index was rebuilt to the post-delete version...
        assert mem._ann_version == mem._version
        # ...and never returns a deleted row.
        assert all(r["id"] in mem._index for r in results)

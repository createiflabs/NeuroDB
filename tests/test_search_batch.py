"""Tier 1.4 — batched search shares one matmul and matches per-query results."""

from __future__ import annotations

import numpy as np
import pytest

from neurodb.store import Memory, MemoryError


@pytest.fixture
def populated():
    mem = Memory("b", 4)
    rng = np.random.default_rng(0)
    mem.write(
        [{"vector": r, "metadata": {"i": i}} for i, r in enumerate(rng.normal(size=(50, 4)))]
    )
    return mem


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclidean"])
def test_search_batch_matches_per_query(populated, metric):
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(8, 4))
    batched = populated.search_batch(queries, k=5, metric=metric)
    assert len(batched) == 8
    for j in range(8):
        single = populated.search(queries[j], k=5, metric=metric)
        assert [r["id"] for r in batched[j]] == [r["id"] for r in single]
        for a, b in zip(batched[j], single, strict=True):
            assert a["score"] == pytest.approx(b["score"], rel=1e-5, abs=1e-5)


def test_search_batch_accepts_single_query(populated):
    one = populated.search_batch(np.zeros(4), k=3)
    assert len(one) == 1 and len(one[0]) == 3


def test_search_batch_empty_memory_returns_empty_lists():
    mem = Memory("e", 4)
    assert mem.search_batch(np.zeros((3, 4)), k=5) == [[], [], []]


def test_search_batch_rejects_bad_shape_and_nonfinite(populated):
    with pytest.raises(MemoryError):
        populated.search_batch(np.zeros((2, 3)), k=5)  # wrong dimension
    with pytest.raises(MemoryError):
        bad = np.zeros((2, 4))
        bad[0, 0] = np.inf
        populated.search_batch(bad, k=5)

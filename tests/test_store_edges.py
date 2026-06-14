"""Assorted correctness edges: empty-memory contract, defensive metadata
copies, duplicate-id counting, stable anomaly ordering, NaN rejection."""

from __future__ import annotations

import pytest

from neurodb.store import MemoryError_


def test_empty_memory_complete_returns_empty(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    out = mem.complete([1, 0])
    assert out["top"] is None
    assert out["weights"] == []
    assert out["reconstruction"] is None


def test_empty_memory_anomaly_returns_empty(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    out = mem.anomaly([1, 0])
    assert out["fields"] == []
    assert out["nearest"] is None
    assert out["score"] == 0.0


def test_empty_memory_search_returns_empty(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    assert mem.search([1, 0]) == []


def test_get_metadata_is_defensive_copy(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    mem.write([{"id": "a", "vector": [1, 0], "metadata": {"tags": [1, 2]}}])
    mem.get("a")["metadata"]["tags"].append(99)
    assert mem.get("a")["metadata"]["tags"] == [1, 2]


def test_search_metadata_is_defensive_copy(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0)
    mem.write([{"id": "a", "vector": [1, 0], "metadata": {"tags": [1, 2]}}])
    mem.search([1, 0], k=1)[0]["metadata"]["tags"].append(99)
    assert mem.search([1, 0], k=1)[0]["metadata"]["tags"] == [1, 2]


def test_write_duplicate_id_in_batch_counts_once(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    affected = mem.write([{"id": "x", "vector": [1, 0]}, {"id": "x", "vector": [0, 1]}])
    assert affected == ["x"]
    assert mem.count == 1


def test_delete_duplicate_ids_counts_once(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    mem.write([{"id": "a", "vector": [1, 0]}])
    assert mem.delete(["a", "a"]) == 1


def test_anomaly_tied_deviation_is_index_ordered(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 4, beta=50.0)
    mem.write([{"vector": [0, 0, 0, 0]}, {"vector": [0, 0, 0, 0]}])
    out = mem.anomaly([1, 0, 1, 0], top_k=4)
    idxs = [f["index"] for f in out["fields"]]
    # deviations are [1,0,1,0]; the two tied 1.0 fields come first, index-ordered
    assert idxs[:2] == [0, 2]


def test_nan_vector_rejected(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    with pytest.raises(MemoryError_):
        mem.write([{"vector": [float("nan"), 0]}])


def test_zscore_empty_memory_no_crash(store_factory):
    # std is undefined with no patterns → the stats fallback must keep the
    # empty-memory contract (None reconstruction, empty fields) intact.
    store = store_factory()
    mem = store.create_memory("m", 2, normalize="zscore")
    assert mem.complete([1, 0])["reconstruction"] is None
    assert mem.anomaly([1, 0])["fields"] == []


def test_zscore_single_pattern_no_crash(store_factory):
    # With <2 patterns the zscore stats fall back to mean=0/std=1 (identity), so
    # recall just returns the one stored row without dividing by a zero std.
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0, normalize="zscore")
    mem.write([{"vector": [3, 4]}])
    out = mem.complete([3, 4])
    assert out["reconstruction"][0] == pytest.approx(3.0, abs=1e-4)
    assert out["reconstruction"][1] == pytest.approx(4.0, abs=1e-4)

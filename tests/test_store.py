import numpy as np
import pytest

from neurodb.store import MemoryError_, NeuroStore, NotFoundError


def test_write_and_search(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=10.0)
    mem.write([{"id": "a", "vector": [1, 0, 0]}, {"id": "b", "vector": [0, 1, 0]}])
    assert mem.count == 2
    assert mem.search([1, 0, 0], k=1)[0]["id"] == "a"


def test_complete_recalls_nearest(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=30.0)
    mem.write([{"id": "a", "vector": [1, 0, 0]}, {"id": "b", "vector": [0, 1, 0]}])
    out = mem.complete([0.8, 0.2, 0.0])
    assert out["top"]["id"] == "a"
    assert out["steps"] == 1
    np.testing.assert_allclose(out["reconstruction"], [1, 0, 0], atol=1e-2)


def test_complete_with_mask_fills_unknown_fields(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=20.0, fields=["x", "y", "z"])
    mem.write([{"id": "p", "vector": [2, 4, 6]}])
    out = mem.complete([2, 0, 0], mask=[0])
    recon = out["reconstruction"]
    assert np.isclose(recon[0], 2.0, atol=1e-3)
    assert recon[1] > 1.0 and recon[2] > 1.0


def test_anomaly_flags_the_off_field(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=30.0, fields=["age", "income", "score"])
    mem.write([{"vector": [1, 1, 1]}, {"vector": [1, 1, 1]}, {"vector": [0.9, 1.1, 1.0]}])
    out = mem.anomaly([1, 5, 1])  # income wildly off
    assert out["fields"][0]["name"] == "income"
    assert out["fields"][0]["index"] == 1
    assert out["score"] > 0.0


def test_empty_complete_returns_empty(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    out = mem.complete([1, 0])
    assert out["top"] is None
    assert out["weights"] == []


def test_single_file_persistence_round_trip(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path)
    mem = store.create_memory("docs", 3, beta=12.0, fields=["a", "b", "c"])
    mem.write([{"id": "x", "vector": [1, 2, 3], "metadata": {"n": 1}}])
    store.save_all()
    assert path.exists()
    assert list(tmp_path.glob("*.npz")) == [path]  # exactly one file

    reloaded = NeuroStore(path)
    m2 = reloaded.get_memory("docs")
    assert m2.count == 1
    assert m2.beta == 12.0
    assert m2.fields == ["a", "b", "c"]
    assert m2.get("x")["metadata"] == {"n": 1}
    np.testing.assert_allclose(m2.get("x")["vector"], [1, 2, 3])


def test_dimension_mismatch_raises(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3)
    with pytest.raises(MemoryError_):
        mem.write([{"vector": [1, 2]}])


def test_fields_length_validation(store_factory):
    store = store_factory()
    with pytest.raises(MemoryError_):
        store.create_memory("m", 3, fields=["only", "two"])


def test_duplicate_memory_and_delete(store_factory):
    store = store_factory()
    store.create_memory("m", 2)
    with pytest.raises(MemoryError_):
        store.create_memory("m", 2)
    store.delete_memory("m")
    with pytest.raises(NotFoundError):
        store.get_memory("m")


def test_delete_pattern_reindexes(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0)
    mem.write([{"id": "1", "vector": [1, 0]}, {"id": "2", "vector": [0, 1]}])
    assert mem.delete(["1"]) == 1
    assert mem.count == 1
    with pytest.raises(NotFoundError):
        mem.get("1")
    assert mem.search([0, 1], k=1)[0]["id"] == "2"

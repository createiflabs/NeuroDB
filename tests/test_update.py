"""Pattern update: full-vector replace, partial metadata merge/replace, and
single-field edits — with cache/stat invalidation."""

from __future__ import annotations

import numpy as np
import pytest

from neurodb.store import MemoryError_, NotFoundError


def test_update_vector_reflected_in_recall_and_stats(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=8.0, fields=["a", "b", "c"], normalize="zscore")
    mem.write([{"id": "x", "vector": [1, 2, 3]}, {"id": "y", "vector": [4, 5, 6]}])
    mem.stats()
    mean_before = mem._mean.copy()  # noqa: SLF001

    out = mem.update("x", vector=[100, 100, 100])
    assert out["vector"] == [100.0, 100.0, 100.0]
    # zscore mean must have shifted (stats recomputed from the new matrix).
    mem.stats()
    assert not np.allclose(mean_before, mem._mean)  # noqa: SLF001
    # recall of the updated vector returns it.
    rec = mem.get("x")
    assert rec["vector"] == [100.0, 100.0, 100.0]


def test_update_metadata_merge_vs_replace(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    mem.write([{"id": "x", "vector": [1, 0], "metadata": {"a": 1, "b": 2}}])

    mem.update("x", metadata={"b": 99, "c": 3})  # merge (default)
    assert mem.get("x")["metadata"] == {"a": 1, "b": 99, "c": 3}

    mem.update("x", metadata={"only": True}, merge_metadata=False)  # replace
    assert mem.get("x")["metadata"] == {"only": True}


def test_update_missing_id_raises(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2)
    with pytest.raises(NotFoundError):
        mem.update("nope", vector=[1, 0])


def test_update_field_by_name_and_index(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, fields=["t", "h", "p"], normalize="zscore")
    mem.write([{"id": "x", "vector": [1, 2, 3]}, {"id": "y", "vector": [4, 5, 6]}])
    mem.update_field("x", "h", 50.0)
    assert mem.get("x")["vector"] == [1.0, 50.0, 3.0]
    mem.update_field("x", 2, 9.0)
    assert mem.get("x")["vector"] == [1.0, 50.0, 9.0]
    with pytest.raises(MemoryError_):
        mem.update_field("x", "nonexistent", 1.0)


def test_update_invalidates_norms_cache(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0, normalize="none")
    mem.write([{"id": "x", "vector": [1, 0]}, {"id": "y", "vector": [0, 1]}])
    # populate norm cache via a cosine search
    mem.search([1, 0], k=1)
    assert mem._norms is not None  # noqa: SLF001
    mem.update("x", vector=[3, 4])
    assert mem._norms is None  # noqa: SLF001


# -- HTTP surface ----------------------------------------------------------


def test_patch_endpoint(client):
    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    client.post(
        "/v1/memories/m/patterns",
        json={"items": [{"id": "x", "vector": [1, 0], "metadata": {"a": 1}}]},
    )

    r = client.patch("/v1/memories/m/patterns/x", json={"vector": [2, 2], "metadata": {"b": 2}})
    assert r.status_code == 200
    body = r.json()
    assert body["vector"] == [2.0, 2.0]
    assert body["metadata"] == {"a": 1, "b": 2}

    missing = client.patch("/v1/memories/m/patterns/nope", json={"vector": [1, 1]})
    assert missing.status_code == 404


def test_post_existing_id_upserts(client):
    """Documented behaviour: writing an existing id overwrites it."""

    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    client.post("/v1/memories/m/patterns", json={"items": [{"id": "x", "vector": [1, 0]}]})
    client.post("/v1/memories/m/patterns", json={"items": [{"id": "x", "vector": [0, 1]}]})
    r = client.get("/v1/memories/m/patterns/x")
    assert r.json()["vector"] == [0.0, 1.0]

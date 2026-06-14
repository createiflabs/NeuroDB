"""Batch anomaly + batch complete: vectorized over M queries in one matmul.

The correctness anchor is element-wise equivalence with calling the single-query
method M times; the throughput anchor is that a batch runs *one* attention step,
not a Python loop of M.
"""

from __future__ import annotations

import numpy as np
import pytest

import neurodb.hopfield as hopfield

SENSOR_ROWS = [
    [20.0, 50.0, 1013.0],
    [21.0, 52.0, 1012.0],
    [19.0, 48.0, 1014.0],
    [20.0, 51.0, 1013.0],
]
FIELDS = ["temperature", "humidity", "pressure"]


def _sensor_memory(store, normalize="zscore", beta=2.0):
    mem = store.create_memory("s", 3, beta=beta, fields=FIELDS, normalize=normalize)
    mem.write([{"id": str(i), "vector": r} for i, r in enumerate(SENSOR_ROWS)])
    return mem


@pytest.mark.parametrize("normalize", ["zscore", "none", "l2"])
def test_anomaly_batch_matches_single(store_factory, normalize):
    mem = _sensor_memory(store_factory(f"{normalize}.npz"), normalize=normalize)
    queries = [[20.0, 95.0, 1013.0], [21.0, 52.0, 1012.0], [50.0, 50.0, 999.0]]
    batch = mem.anomaly_batch(queries, beta=3.0)
    assert len(batch) == len(queries)
    for q, got in zip(queries, batch, strict=True):
        ref = mem.anomaly(q, beta=3.0)
        np.testing.assert_allclose(got["reconstruction"], ref["reconstruction"], atol=1e-4)
        assert got["score"] == pytest.approx(ref["score"], abs=1e-4)
        assert got["z_score"] == pytest.approx(ref["z_score"], abs=1e-4)
        # Structural fields must match exactly (ordering, ids).
        assert [f["index"] for f in got["fields"]] == [f["index"] for f in ref["fields"]]
        assert got["nearest"]["id"] == ref["nearest"]["id"]


@pytest.mark.parametrize("normalize", ["zscore", "none"])
def test_complete_batch_matches_single_masked(store_factory, normalize):
    mem = _sensor_memory(store_factory(f"{normalize}.npz"), normalize=normalize)
    queries = [[20.0, 0.0, 0.0], [19.0, 0.0, 0.0]]
    batch = mem.complete_batch(queries, mask=[0], beta=3.0, steps=2)
    for q, got in zip(queries, batch, strict=True):
        ref = mem.complete(q, mask=[0], beta=3.0, steps=2)
        np.testing.assert_allclose(got["reconstruction"], ref["reconstruction"], atol=1e-4)
        assert [w["id"] for w in got["weights"]] == [w["id"] for w in ref["weights"]]


def test_empty_batch_returns_empty(store_factory):
    mem = _sensor_memory(store_factory())
    assert mem.anomaly_batch([]) == []
    assert mem.complete_batch([]) == []


def test_batch_on_empty_memory(store_factory):
    store = store_factory()
    mem = store.create_memory("e", 3, normalize="zscore")
    out = mem.anomaly_batch([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert len(out) == 2
    assert all(r["reconstruction"] is None and r["fields"] == [] for r in out)


def test_ragged_batch_rejected(store_factory):
    from neurodb.store import MemoryError_

    mem = _sensor_memory(store_factory())
    with pytest.raises(MemoryError_):
        mem.anomaly_batch([[1.0, 2.0, 3.0], [4.0, 5.0]])  # wrong dim on row 2


def test_batch_is_one_matmul_not_a_loop(store_factory, monkeypatch):
    """A batch must run a single attention step, not M single calls."""

    store = store_factory()
    mem = store.create_memory("big", 16, beta=8.0, normalize="none")
    rng = np.random.default_rng(0)
    mem.write([{"vector": rng.normal(size=16).tolist()} for _ in range(10_000)])

    calls = {"n": 0}
    real = hopfield.attention_weights

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hopfield, "attention_weights", counting)
    out = mem.anomaly_batch(rng.normal(size=(1000, 16)).tolist())
    assert len(out) == 1000
    # One attention_weights call for the whole batch (steps=1), not 1000.
    assert calls["n"] == 1


# -- HTTP surface ----------------------------------------------------------


def _seed_http(client):
    client.post("/v1/memories", json={"name": "s", "dimension": 3, "fields": FIELDS, "beta": 2.0})
    client.post(
        "/v1/memories/s/patterns",
        json={"items": [{"id": str(i), "vector": r} for i, r in enumerate(SENSOR_ROWS)]},
    )


def test_anomaly_batch_endpoint_echoes_ids(client):
    _seed_http(client)
    items = [{"id": "q1", "vector": [20, 95, 1013]}, {"vector": [21, 52, 1012]}]
    r = client.post("/v1/memories/s/anomaly/batch", json={"items": items, "beta": 2.0})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["results"][0]["id"] == "q1"
    assert body["results"][1]["id"] is None
    assert body["results"][0]["fields"][0]["name"] == "humidity"


def test_oversize_batch_rejected(client_factory):
    client = client_factory(allow_anonymous=True, max_batch=2)
    _seed_http(client)
    # 3 items > max_batch=2 → 413.
    r = client.post(
        "/v1/memories/s/anomaly/batch",
        json={"items": [{"vector": [1, 2, 3]}] * 3, "beta": 2.0},
    )
    assert r.status_code == 413
    # 2 items == cap → fine.
    ok = client.post(
        "/v1/memories/s/anomaly/batch",
        json={"items": [{"vector": [1, 2, 3]}] * 2, "beta": 2.0},
    )
    assert ok.status_code == 200


def test_empty_batch_endpoint(client):
    _seed_http(client)
    r = client.post("/v1/memories/s/complete/batch", json={"items": []})
    assert r.status_code == 200
    assert r.json() == {"results": [], "count": 0}

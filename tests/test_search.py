"""Search performance/correctness: cached cosine norms (parity + invalidation)
and selectable distance metrics."""

from __future__ import annotations

import numpy as np

from neurodb.metrics import compute_scores


def test_cached_norms_match_uncached(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 4, beta=10.0)
    rows = [[1.0, 2.0, 0.0, 0.5], [0.0, 1.0, 1.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
    mem.write([{"id": str(i), "vector": r} for i, r in enumerate(rows)])

    q = [1.0, 1.0, 0.0, 0.0]
    # Reference: cosine via the full-normalization path (norms=None).
    ref = compute_scores(np.array(rows, dtype=np.float32), np.array(q, dtype=np.float32), "cosine")
    got = {r["id"]: r["score"] for r in mem.search(q, k=3)}
    for i in range(3):
        assert np.isclose(got[str(i)], ref[i], atol=1e-5)


def test_norms_invalidated_after_write(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0)
    mem.write([{"id": "a", "vector": [1, 0]}])
    assert mem.search([1, 0], k=5)[0]["id"] == "a"  # primes the norm cache
    mem.write([{"id": "b", "vector": [0, 1]}])  # must invalidate the cache
    assert mem.search([0, 1], k=1)[0]["id"] == "b"


def test_metric_dot_and_euclidean(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0)
    mem.write([{"id": "big", "vector": [10, 10]}, {"id": "unit", "vector": [1, 0]}])
    # dot product favors the large-magnitude vector for query [1,1]
    assert mem.search([1, 1], k=1, metric="dot")[0]["id"] == "big"
    # euclidean: nearest by L2 to [1,0] is the unit vector
    assert mem.search([1, 0], k=1, metric="euclidean")[0]["id"] == "unit"


def test_metric_exposed_via_api(client):
    client.post("/memories", json={"name": "m", "dimension": 2})
    client.post(
        "/memories/m/patterns",
        json={"items": [{"id": "big", "vector": [10, 10]}, {"id": "u", "vector": [1, 0]}]},
    )
    resp = client.post("/memories/m/search", json={"query": [1, 1], "k": 1, "metric": "dot"})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["id"] == "big"


def test_invalid_metric_rejected(client):
    client.post("/memories", json={"name": "m", "dimension": 2})
    client.post("/memories/m/patterns", json={"items": [{"vector": [1, 0]}]})
    resp = client.post("/memories/m/search", json={"query": [1, 0], "metric": "manhattan"})
    assert resp.status_code == 422

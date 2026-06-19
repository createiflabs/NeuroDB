"""Tier 1.1 — ANN candidate pre-filter: approximate search stays within
tolerance of exact while only scoring a small subset of rows."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hnswlib")  # optional ANN backend

from neurodb.store import Memory  # noqa: E402


def test_approx_within_tolerance_of_exact():
    rng = np.random.default_rng(0)
    d, n, k = 32, 5000, 10
    x = rng.normal(size=(n, d)).astype(np.float32)
    mem = Memory("a", d)
    mem.write([{"id": str(i), "vector": x[i]} for i in range(n)])

    recalls = []
    for _ in range(50):
        q = rng.normal(size=d)
        exact = [r["id"] for r in mem.search(q, k=k, approx=False)]
        approx = [r["id"] for r in mem.search(q, k=k, approx=True)]
        assert len(approx) == k  # approx still returns a full top-k
        recalls.append(len(set(exact) & set(approx)) / k)

    mean_recall = float(np.mean(recalls))
    # The approximate pre-filter must recover almost all of the exact top-k.
    assert mean_recall >= 0.9, f"mean recall@{k} = {mean_recall:.3f}"


def test_approx_equals_exact_when_candidates_cover_everything():
    # When the candidate set (>= 64) covers the whole memory, approx is exact.
    rng = np.random.default_rng(1)
    d = 8
    mem = Memory("b", d)
    mem.write([{"id": str(i), "vector": rng.normal(size=d)} for i in range(40)])
    for _ in range(20):
        q = rng.normal(size=d)
        assert [r["id"] for r in mem.search(q, k=5, approx=True)] == [
            r["id"] for r in mem.search(q, k=5, approx=False)
        ]


def test_approx_index_rebuilds_after_writes():
    rng = np.random.default_rng(2)
    d = 16
    mem = Memory("c", d)
    mem.write([{"id": str(i), "vector": rng.normal(size=d)} for i in range(200)])
    q = rng.normal(size=d)
    mem.search(q, k=5, approx=True)  # builds the index
    # Add a near-duplicate of the query; a fresh approx search must find it
    # (the index rebuilt because the version advanced).
    mem.write([{"id": "needle", "vector": q}])
    hits = [r["id"] for r in mem.search(q, k=3, approx=True)]
    assert "needle" in hits


def test_approx_non_cosine_falls_back_to_exact():
    # approx only applies to cosine; other metrics quietly use the exact scan.
    rng = np.random.default_rng(3)
    d = 8
    mem = Memory("e", d)
    mem.write([{"id": str(i), "vector": rng.normal(size=d)} for i in range(50)])
    q = rng.normal(size=d)
    assert [r["id"] for r in mem.search(q, k=5, approx=True, metric="euclidean")] == [
        r["id"] for r in mem.search(q, k=5, approx=False, metric="euclidean")
    ]


def test_approx_search_over_http(client):
    # The approx flag is reachable and correct end-to-end over the API.
    rng = np.random.default_rng(4)
    d = 16
    client.post("/memories", json={"name": "m", "dimension": d})
    items = [{"id": str(i), "vector": rng.normal(size=d).tolist()} for i in range(300)]
    client.post("/memories/m/patterns", json={"items": items})
    q = rng.normal(size=d).tolist()
    r = client.post("/memories/m/search", json={"query": q, "k": 5, "approx": True})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 5

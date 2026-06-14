"""Hopfield capacity / saturation diagnostics.

These lock in the *direction* of the signal (well-separated → healthy,
near-duplicate / over-capacity → saturated), not exact numbers.
"""

from __future__ import annotations

import numpy as np


def test_well_separated_patterns_are_healthy(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 16, beta=30.0, normalize="none")
    rng = np.random.default_rng(1)
    # Random high-dim vectors are near-orthogonal → easily self-recalled.
    mem.write([{"vector": rng.normal(size=16).tolist()} for _ in range(8)])
    rep = mem.capacity_report()
    assert rep["status"] == "healthy"
    assert rep["self_recall_fail_fraction" ] < 0.1
    assert rep["count"] == 8
    assert rep["max_pairwise_similarity"] < 0.95


def test_near_duplicate_patterns_saturate(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 8, beta=1.0, normalize="none")
    rng = np.random.default_rng(2)
    base = rng.normal(size=8)
    # 30 tiny perturbations of one vector at low beta → attractors merge.
    mem.write([{"vector": (base + 0.01 * rng.normal(size=8)).tolist()} for _ in range(30)])
    rep = mem.capacity_report()
    assert rep["status"] == "saturated"
    assert rep["self_recall_fail_fraction"] > 0.5
    assert rep["max_pairwise_similarity"] > 0.95


def test_empty_memory_capacity_is_defined(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 4, normalize="zscore")
    rep = mem.capacity_report()
    assert rep["count"] == 0
    assert rep["status"] == "healthy"
    assert rep["self_recall_fail_fraction"] == 0.0
    assert rep["mean_pairwise_similarity"] is None


def test_capacity_cached_and_invalidated_on_write(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 8, beta=20.0, normalize="none")
    rng = np.random.default_rng(3)
    mem.write([{"vector": rng.normal(size=8).tolist()} for _ in range(5)])
    first = mem.capacity_report()
    assert mem._capacity is not None  # noqa: SLF001 - cache introspection
    mem.write([{"vector": rng.normal(size=8).tolist()} for _ in range(5)])
    assert mem._capacity is None  # noqa: SLF001 - invalidated on write
    second = mem.capacity_report()
    assert second["count"] == 10 and first["count"] == 5


def test_capacity_surfaced_in_stats_and_health(client):
    client.post(
        "/v1/memories",
        json={"name": "m", "dimension": 8, "beta": 1.0, "normalize": "none"},
    )
    # near-duplicates → saturated
    base = [1.0] * 8
    client.post(
        "/v1/memories/m/patterns",
        json={"items": [{"vector": [b + 0.001 * i for b in base]} for i in range(20)]},
    )
    stats = client.get("/v1/stats").json()
    cap = stats["detail"][0]["capacity"]
    assert set(cap) == {"status", "self_recall_fail_fraction"}
    assert cap["status"] == "saturated"

    health = client.get("/health").json()
    assert health["saturated_memories"] == 1

    full = client.get("/v1/memories/m/capacity").json()
    assert full["status"] == "saturated"
    assert "suggested_beta" in full

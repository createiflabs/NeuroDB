"""Resource bounds: writes past a ceiling are rejected (before allocating) while
reads keep serving; footprint is reported in /stats and pressure flips /health."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import LimitExceededError, NeuroStore


def test_pattern_ceiling_rejects_and_preserves_data(tmp_path):
    store = NeuroStore(tmp_path / "db.npz", max_patterns_per_memory=3)
    mem = store.create_memory("m", 2)
    mem.write([{"id": "a", "vector": [1, 0]}, {"id": "b", "vector": [0, 1]}])

    # Two existing + two new would be 4 > 3 → rejected before allocating.
    with pytest.raises(LimitExceededError):
        store.check_write("m", 2, mem.dimension, mem.normalize)

    # Existing data intact and still readable.
    assert mem.count == 2
    assert mem.get("a")["vector"] == [1.0, 0.0]


def test_byte_budget_rejects_naming_memory(tmp_path):
    # Budget too small for another row in a zscore memory (D*4*2 per row).
    store = NeuroStore(tmp_path / "db.npz", max_total_bytes=64)
    mem = store.create_memory("big", 8, fields=[f"f{i}" for i in range(8)], normalize="zscore")
    with pytest.raises(LimitExceededError, match="big"):
        store.check_write("big", 10, mem.dimension, mem.normalize)


def test_unlimited_by_default(tmp_path):
    store = NeuroStore(tmp_path / "db.npz")  # both ceilings 0 = unlimited
    mem = store.create_memory("m", 2)
    store.check_write("m", 100_000, mem.dimension, mem.normalize)  # no raise


def test_footprint_reported_in_stats(tmp_path):
    settings = Settings(
        data_file=str(tmp_path / "db.npz"),
        autosave_interval=0.0,
        allow_anonymous=True,
        max_total_bytes=1_000_000,
    )
    with TestClient(create_app(settings)) as client:
        client.post("/memories", json={"name": "m", "dimension": 4})
        client.post(
            "/memories/m/patterns",
            json={"items": [{"vector": [1, 2, 3, 4]} for _ in range(10)]},
        )
        stats = client.get("/v1/stats").json()
        # 10 rows x 4 dims x 4 bytes = 160 (normalize "none" → no Z cache).
        assert stats["approx_bytes"] == pytest.approx(160, abs=64)
        assert stats["detail"][0]["approx_bytes"] >= 160
        assert stats["pct_of_budget"] is not None


def test_write_over_ceiling_returns_413_reads_still_work(tmp_path):
    settings = Settings(
        data_file=str(tmp_path / "db.npz"),
        autosave_interval=0.0,
        allow_anonymous=True,
        max_patterns_per_memory=2,
    )
    with TestClient(create_app(settings)) as client:
        client.post("/memories", json={"name": "m", "dimension": 2})
        client.post("/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        resp = client.post(
            "/memories/m/patterns",
            json={"items": [{"vector": [0, 1]}, {"vector": [1, 1]}]},
        )
        assert resp.status_code == 413, resp.text
        # Reads keep serving after the rejection.
        assert client.get("/memories/m/patterns/a").status_code == 200


def test_health_memory_pressure_flag(tmp_path):
    settings = Settings(
        data_file=str(tmp_path / "db.npz"),
        autosave_interval=0.0,
        allow_anonymous=True,
        max_total_bytes=200,
        memory_pressure_pct=50.0,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["memory_pressure"] is False
        client.post("/memories", json={"name": "m", "dimension": 4})
        # 10 x 4 x 4 = 160 bytes > 50% of 200 → pressure.
        client.post(
            "/memories/m/patterns",
            json={"items": [{"vector": [1, 2, 3, 4]} for _ in range(10)]},
        )
        assert client.get("/health").json()["memory_pressure"] is True

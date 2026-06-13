"""Lifespan: shutdown persists the store; autosave loop is cancelled cleanly."""

from __future__ import annotations

from fastapi.testclient import TestClient

from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import NeuroStore


def test_shutdown_persists_dirty_store(tmp_path):
    path = tmp_path / "db.npz"
    settings = Settings(data_file=str(path), autosave_interval=0.0, allow_anonymous=True)
    with TestClient(create_app(settings)) as client:
        client.post("/v1/memories", json={"name": "m", "dimension": 2})
        client.post("/v1/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        # No explicit flush — rely on shutdown (lifespan) to persist.

    reloaded = NeuroStore(path)
    assert reloaded.get_memory("m").count == 1


def test_autosave_loop_runs_and_persists(tmp_path):
    path = tmp_path / "db.npz"
    settings = Settings(data_file=str(path), autosave_interval=0.05, allow_anonymous=True)
    with TestClient(create_app(settings)) as client:
        client.post("/v1/memories", json={"name": "m", "dimension": 2})
        client.post("/v1/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        # The autosave loop should persist within a couple of intervals.
        import time

        time.sleep(0.2)
        on_disk = NeuroStore(path)
        assert on_disk.get_memory("m").count == 1

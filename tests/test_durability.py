"""Durability: save() fsyncs before the atomic replace, and POST /flush makes
acknowledged writes durable on demand."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

import neurodb.store as store_mod
from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import NeuroStore


def test_save_fsyncs_before_replace(store_factory, monkeypatch):
    store = store_factory()
    store.create_memory("m", 2).write([{"id": "a", "vector": [1.0, 0.0]}])

    calls: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "fsync", spy_fsync)
    monkeypatch.setattr(store_mod.os, "replace", spy_replace)

    store.save()

    assert "fsync" in calls, "save() must fsync the temp file"
    assert calls.index("fsync") < calls.index("replace"), "fsync must precede replace"


def test_flush_endpoint_makes_writes_durable(tmp_path):
    path = tmp_path / "db.npz"
    settings = Settings(data_file=str(path), autosave_interval=0.0, allow_anonymous=True)
    with TestClient(create_app(settings)) as client:
        client.post("/memories", json={"name": "m", "dimension": 2})
        client.post("/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        resp = client.post("/flush")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["durable"] is True
        assert body["persisted"] >= 1

    # A fresh process reading the same file sees the flushed write.
    reloaded = NeuroStore(path)
    assert reloaded.get_memory("m").count == 1


def test_write_without_flush_is_not_yet_persisted(tmp_path):
    # Documents the bounded loss window: autosave off + no flush => not on disk.
    path = tmp_path / "db.npz"
    settings = Settings(data_file=str(path), autosave_interval=0.0, allow_anonymous=True)
    app = create_app(settings)
    with TestClient(app) as client:
        client.post("/memories", json={"name": "m", "dimension": 2})
        client.post("/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        # Peek at the on-disk state mid-session (create_memory persisted the
        # empty memory, but the pattern write has not been flushed).
        on_disk = NeuroStore(path)
        assert on_disk.get_memory("m").count == 0

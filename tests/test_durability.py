"""Durability: save() fsyncs before the atomic replace, and POST /flush makes
acknowledged writes durable on demand."""

from __future__ import annotations

import os

import pytest
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


def test_disk_full_marks_save_failed(store_factory, monkeypatch):
    # Simulate ENOSPC during the file write: save() must surface it and flip the
    # readiness flag rather than swallowing a half-persisted state.
    store = store_factory()
    store.create_memory("m", 2).write([{"id": "a", "vector": [1.0, 0.0]}])

    def no_space(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(store_mod.np, "savez", no_space)
    with pytest.raises(OSError):
        store.save()
    assert store.last_save_ok is False


def test_failed_save_makes_ready_return_503(tmp_path):
    # The readiness contract under a *real* save failure (not just a flipped bool):
    # a failed /flush leaves the instance not-ready until a save succeeds.
    settings = Settings(
        data_file=str(tmp_path / "db.npz"), autosave_interval=0.0, allow_anonymous=True
    )
    with TestClient(create_app(settings)) as client:
        client.post("/memories", json={"name": "m", "dimension": 2})
        client.post("/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})

        store = client.app.state.store
        original = store._save_locked

        def boom():
            raise OSError("No space left on device")

        store._save_locked = boom
        try:
            # A real persist failure flips last_save_ok, which /ready surfaces.
            with pytest.raises(OSError):
                store.save()
            assert client.get("/ready").status_code == 503
        finally:
            # Restore so the lifespan shutdown save (and teardown) succeed.
            store._save_locked = original


def test_interrupt_before_rename_preserves_live_file(store_factory, monkeypatch):
    # kill -9 between the fsync'd temp write and the atomic rename: the previous
    # live file must remain intact and reload cleanly (no torn data).
    store = store_factory()
    store.create_memory("m", 2).write([{"id": "a", "vector": [1, 0]}])
    store.save_all()
    good_bytes = store.data_file.read_bytes()

    def crash_before_rename(src, dst):
        raise OSError("crash between temp write and rename")

    monkeypatch.setattr(store_mod.os, "replace", crash_before_rename)
    store.get_memory("m").write([{"id": "b", "vector": [0, 1]}])
    with pytest.raises(OSError):
        store.save()
    monkeypatch.undo()

    # Live file unchanged; a fresh process sees only the previously-saved state.
    assert store.data_file.read_bytes() == good_bytes
    reloaded = NeuroStore(store.data_file)
    assert reloaded.get_memory("m").count == 1
    assert not (store.data_file.parent / "db.npz.tmp").exists()


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

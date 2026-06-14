"""Backup / restore: consistent off-box snapshots and validated, non-destructive
restore (data-loss protection)."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodb.backup import backup_store, restore_file
from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import NeuroStore, StoreError


def _populate(store: NeuroStore) -> None:
    mem = store.create_memory("sensors", 3, fields=["t", "h", "p"], normalize="zscore")
    mem.write(
        [
            {"id": "a", "vector": [20, 50, 1013], "metadata": {"site": "A"}},
            {"id": "b", "vector": [21, 52, 1012], "metadata": {"site": "B"}},
        ]
    )
    store.save_all()


def test_backup_restore_round_trip_is_identical(store_factory, tmp_path):
    store = store_factory()
    _populate(store)

    dest = tmp_path / "backups"
    dest.mkdir()
    snapshot = backup_store(store, dest)
    assert snapshot.exists()

    # Restore into a fresh location and compare every field.
    target = tmp_path / "restored.npz"
    restore_file(snapshot, target)
    restored = NeuroStore(target)

    mem = restored.get_memory("sensors")
    assert mem.count == 2
    assert mem.ids == ["a", "b"]
    assert mem.get("a")["metadata"] == {"site": "A"}
    assert mem.normalize == "zscore"
    np.testing.assert_array_equal(
        store.get_memory("sensors")._X, mem._X
    )


def test_restore_refuses_corrupt_source_and_keeps_live(store_factory, tmp_path):
    store = store_factory()
    _populate(store)
    live = store.data_file
    live_bytes = live.read_bytes()

    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a real npz")

    with pytest.raises(StoreError):
        restore_file(bad, live)

    # Live file untouched.
    assert live.read_bytes() == live_bytes


def test_restore_preserves_previous_file(store_factory, tmp_path):
    store = store_factory()
    _populate(store)
    snapshot = backup_store(store, tmp_path / "snap.npz")

    # Mutate + save so the live file differs from the snapshot, then restore.
    store.get_memory("sensors").write([{"id": "c", "vector": [1, 2, 3]}])
    store.save_all()

    preserved = restore_file(snapshot, store.data_file)
    assert preserved is not None and preserved.exists()
    # Restored file has 2 patterns (the snapshot); preserved has 3.
    assert NeuroStore(store.data_file).get_memory("sensors").count == 2
    assert NeuroStore(preserved).get_memory("sensors").count == 3


def test_backup_under_concurrent_writes_is_consistent(store_factory, tmp_path):
    store = store_factory()
    mem = store.create_memory("m", 4)

    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            mem.write([{"vector": [i, i, i, i]}])
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(20):
            snap = backup_store(store, tmp_path / "c.npz")
            reloaded = NeuroStore(snap).get_memory("m")
            # The defining invariant: rows always match ids/metadata.
            assert reloaded._X.shape[0] == reloaded.count == len(reloaded.metadata)
    finally:
        stop.set()
        t.join()


def test_backup_endpoint(tmp_path):
    settings = Settings(
        data_file=str(tmp_path / "db.npz"),
        backup_dir=str(tmp_path / "backups"),
        autosave_interval=0.0,
        allow_anonymous=True,
    )
    with TestClient(create_app(settings)) as client:
        client.post("/memories", json={"name": "m", "dimension": 2})
        client.post("/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
        resp = client.post("/v1/backup")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["bytes"] > 0
        assert NeuroStore(body["path"]).get_memory("m").count == 1

        # Streaming download returns the bytes.
        dl = client.post("/v1/backup?download=true")
        assert dl.status_code == 200
        assert dl.content[:1] == b"P"  # .npz is a zip archive (PK\x03\x04)

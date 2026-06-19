"""Tier 2.2 — write-ahead log durability.

A snapshot-only store loses every write since the last autosave on `kill -9`.
The WAL durably records each write/delete before the call returns, so a crash
between snapshots replays the uncommitted operations on boot.
"""

from __future__ import annotations

import numpy as np

from neurodb.store import NeuroStore


def _reopen(path) -> NeuroStore:
    """Simulate a crash + restart: a fresh store over the same files, with no
    intervening clean save()."""

    return NeuroStore(path, wal=True)


def test_wal_replay_recovers_uncommitted_stores(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path, wal=True)
    store.create_memory("m", 3)  # snapshots an empty 'm'; WAL is clear
    mem = store.get_memory("m")

    # Writes that never get a clean snapshot/flush (the 5s autosave never fires).
    mem.write([{"id": "a", "vector": [1, 2, 3]}])
    mem.write([{"id": "b", "vector": [4, 5, 6], "metadata": {"k": "v"}}])
    mem.delete(["a"])
    mem.write([{"id": "c", "vector": [7, 8, 9]}])

    # Crash + restart.
    recovered = _reopen(path)
    rmem = recovered.get_memory("m")
    assert rmem.count == 2
    assert sorted(rmem.ids) == ["b", "c"]
    assert rmem.get("b")["vector"] == [4.0, 5.0, 6.0]
    assert rmem.get("b")["metadata"] == {"k": "v"}
    assert rmem.get("c")["vector"] == [7.0, 8.0, 9.0]


def test_clean_save_clears_the_wal(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path, wal=True)
    store.create_memory("m", 2)
    store.get_memory("m").write([{"id": "a", "vector": [1, 2]}])
    store.save_all()  # folds the write into the snapshot and clears the log

    wal = store._wal
    assert wal is not None
    assert not wal.path.exists() and not wal.ckpt_path.exists()

    # Recovery has nothing to replay; the snapshot alone is authoritative.
    recovered = _reopen(path)
    assert recovered.get_memory("m").get("a")["vector"] == [1.0, 2.0]


def test_wal_replay_is_idempotent_across_double_restart(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path, wal=True)
    store.create_memory("m", 4)
    store.get_memory("m").write([{"id": f"id-{i}", "vector": np.arange(4) + i} for i in range(20)])

    first = _reopen(path)
    second = _reopen(first.data_file)  # boot again over the recovered snapshot
    assert second.get_memory("m").count == 20
    assert second.get_memory("m").get("id-19")["vector"] == [19.0, 20.0, 21.0, 22.0]


def test_disabled_wal_writes_no_log(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path, wal=False)
    store.create_memory("m", 2)
    store.get_memory("m").write([{"id": "a", "vector": [1, 2]}])
    assert not (tmp_path / "db.npz.wal").exists()
    # Without a flush/save, an uncommitted write is *not* durable (documented).
    recovered = NeuroStore(path, wal=False)
    assert recovered.get_memory("m").count == 0

"""Startup recovery from corrupt / torn / mismatched data files.

``NeuroStore.load()`` runs at construction (and at import via the module-level
app). A bad file must never crash startup under the default; it is quarantined
(never deleted) and the store starts empty. ``fail_on_corrupt_load`` opts into
fail-closed instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurodb.store import NeuroStore, StoreError


def _write_npz(path, **arrays) -> None:
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def _manifest_array(manifest: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)


def test_garbage_file_starts_empty_and_quarantines(tmp_path):
    path = tmp_path / "db.npz"
    path.write_bytes(b"this is not a real npz archive")
    store = NeuroStore(path)
    assert store.list_memories() == []
    assert not path.exists()  # moved aside
    assert list(tmp_path.glob("db.npz.corrupt-*"))


def test_truncated_file_recovers(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path)
    store.create_memory("m", 3).write([{"id": "a", "vector": [1, 2, 3]}])
    store.save_all()
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # truncate mid-archive
    recovered = NeuroStore(path)
    assert recovered.list_memories() == []
    assert list(tmp_path.glob("db.npz.corrupt-*"))


def test_missing_manifest_recovers(tmp_path):
    path = tmp_path / "db.npz"
    _write_npz(path, **{"X@m": np.zeros((1, 2), dtype=np.float32)})
    store = NeuroStore(path)
    assert store.list_memories() == []


def test_ids_rows_mismatch_rejected(tmp_path):
    path = tmp_path / "db.npz"
    manifest = {
        "version": 1,
        "memories": [
            {
                "name": "m",
                "dimension": 2,
                "beta": 8.0,
                "fields": None,
                "ids": ["a"],
                "metadata": [{}],
            }
        ],
    }
    # 3 rows but only 1 id — must be rejected, not loaded inconsistently.
    _write_npz(
        path,
        __manifest__=_manifest_array(manifest),
        **{"X@m": np.zeros((3, 2), dtype=np.float32)},
    )
    store = NeuroStore(path)
    assert store.list_memories() == []


def test_future_version_rejected(tmp_path):
    path = tmp_path / "db.npz"
    _write_npz(path, __manifest__=_manifest_array({"version": 999, "memories": []}))
    store = NeuroStore(path)
    assert store.list_memories() == []


def test_corrupt_file_is_preserved_not_deleted(tmp_path):
    path = tmp_path / "db.npz"
    original = b"garbage-content-xyz"
    path.write_bytes(original)
    NeuroStore(path)
    quarantined = list(tmp_path.glob("db.npz.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == original


def test_fail_closed_raises(tmp_path):
    path = tmp_path / "db.npz"
    path.write_bytes(b"garbage")
    with pytest.raises(StoreError):
        NeuroStore(path, fail_on_corrupt_load=True)


def test_stale_tmp_file_ignored(tmp_path):
    path = tmp_path / "db.npz"
    store = NeuroStore(path)
    store.create_memory("m", 2)
    store.save_all()
    (tmp_path / "db.npz.tmp").write_bytes(b"partial interrupted save")
    reloaded = NeuroStore(path)
    assert [m["name"] for m in reloaded.list_memories()] == ["m"]
    assert not (tmp_path / "db.npz.tmp").exists()

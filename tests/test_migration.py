"""Manifest migration policy: old files migrate forward, newer files fail loudly,
current files load untouched, and a save stamps the current version.

This harness is the one that protects every future release. To cover a new
format version, add one fixture at the old version and assert it migrates.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurodb.migrations import MigrationError, migrate
from neurodb.store import _MANIFEST_VERSION, NeuroStore, StoreError, peek_manifest_version


def _manifest_array(manifest: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)


def _write_version(path, version: int) -> None:
    """Write a minimal but valid data file at ``version`` with one memory."""

    manifest = {
        "version": version,
        "memories": [
            {
                "name": "m",
                "dimension": 2,
                "beta": 8.0,
                "fields": None,
                "normalize": "none",
                "ids": ["a"],
                "metadata": [{"k": "v"}],
            }
        ],
    }
    with open(path, "wb") as handle:
        np.savez(
            handle,
            __manifest__=_manifest_array(manifest),
            **{"X@m": np.ones((1, 2), dtype=np.float32)},
        )


def test_old_version_migrates_forward(tmp_path):
    path = tmp_path / "db.npz"
    _write_version(path, _MANIFEST_VERSION - 1)

    store = NeuroStore(path, fail_on_corrupt_load=True)
    mem = store.get_memory("m")
    assert mem.count == 1
    assert mem.get("a")["metadata"] == {"k": "v"}
    np.testing.assert_array_equal(mem._X, np.ones((1, 2), dtype=np.float32))


def test_newer_version_fails_loudly(tmp_path):
    path = tmp_path / "db.npz"
    future = {"version": _MANIFEST_VERSION + 1, "memories": []}
    with open(path, "wb") as handle:
        np.savez(handle, __manifest__=_manifest_array(future))

    with pytest.raises(StoreError, match="newer NeuroDB"):
        NeuroStore(path, fail_on_corrupt_load=True)


def test_current_version_loads_without_migration(tmp_path):
    path = tmp_path / "db.npz"
    _write_version(path, _MANIFEST_VERSION)
    store = NeuroStore(path, fail_on_corrupt_load=True)
    assert store.get_memory("m").count == 1


def test_save_after_migrate_stamps_current_version(tmp_path):
    path = tmp_path / "db.npz"
    _write_version(path, _MANIFEST_VERSION - 1)
    assert peek_manifest_version(path) == _MANIFEST_VERSION - 1

    store = NeuroStore(path, fail_on_corrupt_load=True)
    store.save_all()
    assert peek_manifest_version(path) == _MANIFEST_VERSION

    # Reloads clean at the current version.
    assert NeuroStore(path, fail_on_corrupt_load=True).get_memory("m").count == 1


def test_migrate_raises_when_no_path_registered():
    # A gap in the chain must fail explicitly, never silently.
    with pytest.raises(MigrationError):
        migrate({"version": 999}, {}, target_version=1000)

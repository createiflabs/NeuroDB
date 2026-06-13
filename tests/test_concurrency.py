"""Concurrency / persistence-race regression tests.

These reproduce the save/write race: ``NeuroStore.save()`` must snapshot each
memory's matrix + ids + metadata atomically (under that memory's own lock) and
must not clear the dirty flag for a write that lands while a save is in flight.
"""

from __future__ import annotations

import threading

import numpy as np

import neurodb.store as store_mod


def test_write_during_save_stays_dirty(store_factory, monkeypatch):
    """A write that lands mid-save must remain unpersisted-but-dirty.

    Reproduces the lost-update bug: the old save() unconditionally cleared
    ``dirty`` for every memory at the end, so a write injected during
    serialization was neither in the file nor flagged for the next flush.
    """

    store = store_factory()
    mem = store.create_memory("m", 2)
    mem.write([{"id": "a", "vector": [1.0, 0.0]}])

    original_savez = np.savez

    def savez_then_inject_write(*args, **kwargs):
        # A concurrent write arrives after save() captured its snapshot.
        mem.write([{"id": "b", "vector": [0.0, 1.0]}])
        return original_savez(*args, **kwargs)

    monkeypatch.setattr(store_mod.np, "savez", savez_then_inject_write)
    store.save()

    # 'b' was not in this save, so the memory must still be dirty.
    assert mem.dirty is True


def test_save_snapshots_a_consistent_view(store_factory, monkeypatch):
    """The persisted ids and matrix must come from one atomic snapshot.

    Reproduces the torn-snapshot bug deterministically: the old save()
    serialized the manifest ids and then read ``mem._X`` separately, so a write
    landing between the two persisted a matrix whose row count disagreed with
    the ids list. The fix snapshots ids + matrix together under the memory lock.
    """

    store = store_factory()
    mem = store.create_memory("m", 2)
    mem.write([{"id": "a", "vector": [1.0, 0.0]}])

    original_dumps = store_mod.json.dumps

    def dumps_then_inject_write(obj, *args, **kwargs):
        result = original_dumps(obj, *args, **kwargs)
        # Land a write after the manifest is serialized (a no-op against a
        # consistent snapshot; a torn write against live state).
        mem.write([{"id": "b", "vector": [0.0, 1.0]}])
        return result

    monkeypatch.setattr(store_mod.json, "dumps", dumps_then_inject_write)
    store.save()

    reloaded = store_factory()
    m2 = reloaded.get_memory("m")
    assert m2._X.shape[0] == len(m2.ids) == len(m2.metadata) == m2.count


def test_concurrent_writes_and_saves_stay_consistent(store_factory):
    """After hammering writes while saves run, the reloaded store must have
    a matrix whose row count matches its id/metadata lists."""

    store = store_factory()
    mem = store.create_memory("m", 4)

    stop = threading.Event()

    def saver():
        while not stop.is_set():
            store.save()

    saver_thread = threading.Thread(target=saver)
    saver_thread.start()
    try:
        for i in range(400):
            mem.write([{"id": f"id-{i}", "vector": [float(i), 0.0, 0.0, 1.0]}])
    finally:
        stop.set()
        saver_thread.join()

    store.save()
    reloaded = store_factory()
    m2 = reloaded.get_memory("m")
    n_rows = m2._X.shape[0]
    assert n_rows == len(m2.ids) == len(m2.metadata) == m2.count
    # Every id resolves to a finite, correct-dimension row.
    for _id in m2.ids:
        vec = np.asarray(m2.get(_id)["vector"], dtype=np.float64)
        assert vec.shape == (4,)
        assert np.all(np.isfinite(vec))

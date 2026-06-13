"""Mask-based pattern completion edge cases.

A mask lists the *known* dimensions. A full mask must clamp every known dim, so
the output equals the input (nothing to complete). Empty / duplicate / out-of-
range masks are rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodb.store import MemoryError_


def test_full_mask_returns_query(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=20.0)
    mem.write([{"id": "p", "vector": [2, 4, 6]}])
    out = mem.complete([1, 2, 3], mask=[0, 1, 2])
    np.testing.assert_allclose(out["reconstruction"], [1, 2, 3], atol=1e-4)


def test_empty_mask_rejected(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3)
    mem.write([{"vector": [1, 2, 3]}])
    with pytest.raises(MemoryError_):
        mem.complete([1, 0, 0], mask=[])


def test_duplicate_mask_index_rejected(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3)
    mem.write([{"vector": [1, 2, 3]}])
    with pytest.raises(MemoryError_):
        mem.complete([1, 0, 0], mask=[0, 0])


def test_out_of_range_mask_index_rejected(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3)
    mem.write([{"vector": [1, 2, 3]}])
    with pytest.raises(MemoryError_):
        mem.complete([1, 0, 0], mask=[5])


@pytest.mark.parametrize("steps", [1, 4])
def test_partial_mask_completes_unknown(store_factory, steps):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=20.0)
    mem.write([{"vector": [2, 4, 6]}])
    out = mem.complete([2, 0, 0], mask=[0], steps=steps)
    recon = out["reconstruction"]
    assert np.isclose(recon[0], 2.0, atol=1e-3)  # known dim preserved
    assert recon[1] > 1.0 and recon[2] > 1.0  # unknown dims completed

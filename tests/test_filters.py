"""Metadata filter operator semantics + robustness.

Malformed filters must yield a 400-class StoreError (not a 500/TypeError), and
``$in``/``$nin`` must mean set membership — not Python substring containment.
"""

from __future__ import annotations

import pytest

from neurodb.store import MemoryError_


def _populated(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 2, beta=10.0)
    mem.write(
        [
            {"id": "a", "vector": [1, 0], "metadata": {"tag": "cat", "price": 10, "qty": 3}},
            {"id": "b", "vector": [0, 1], "metadata": {"tag": "category", "price": 20, "qty": 5}},
            {"id": "c", "vector": [1, 1], "metadata": {"tag": "dog", "price": "cheap"}},
        ]
    )
    return mem


@pytest.mark.parametrize(
    "flt,expected",
    [
        ({"tag": "cat"}, {"a"}),
        ({"tag": {"$eq": "cat"}}, {"a"}),
        ({"tag": {"$ne": "cat"}}, {"b", "c"}),
        ({"tag": {"$in": ["cat", "dog"]}}, {"a", "c"}),
        ({"tag": {"$nin": ["cat", "dog"]}}, {"b"}),
        ({"price": {"$gt": 10}}, {"b"}),  # c (price "cheap") excluded, not a crash
        ({"price": {"$gte": 10}}, {"a", "b"}),
        ({"price": {"$lt": 20}}, {"a"}),
        ({"price": {"$lte": 20}}, {"a", "b"}),
        ({"tag": ["cat", "dog"]}, {"a", "c"}),  # bare list == membership
    ],
)
def test_filter_operators(store_factory, flt, expected):
    mem = _populated(store_factory)
    got = {r["id"] for r in mem.search([1, 1], k=10, flt=flt)}
    assert got == expected


def test_unknown_operator_rejected(store_factory):
    mem = _populated(store_factory)
    with pytest.raises(MemoryError_):
        mem.search([1, 1], k=10, flt={"price": {"$bogus": 1}})


def test_in_with_non_list_operand_rejected(store_factory):
    # The old substring bug: {"$in": "category"} matched "cat". Now rejected.
    mem = _populated(store_factory)
    with pytest.raises(MemoryError_):
        mem.search([1, 1], k=10, flt={"tag": {"$in": "category"}})


def test_gt_against_incomparable_excludes_row(store_factory):
    # price "cheap" vs 5 must not raise; the row is simply excluded.
    mem = _populated(store_factory)
    got = {r["id"] for r in mem.search([1, 1], k=10, flt={"price": {"$gt": 5}})}
    assert got == {"a", "b"}


def test_missing_key_excluded_by_comparison(store_factory):
    mem = _populated(store_factory)  # c has no "qty"
    got = {r["id"] for r in mem.search([1, 1], k=10, flt={"qty": {"$gte": 1}})}
    assert got == {"a", "b"}

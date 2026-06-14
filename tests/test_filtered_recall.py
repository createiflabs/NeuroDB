"""Filtered complete / anomaly: score a record only against patterns matching a
metadata filter (e.g. same sensor_type)."""

from __future__ import annotations


def _two_type_memory(store):
    mem = store.create_memory("m", 2, beta=6.0, fields=["x", "y"], normalize="zscore")
    items = []
    for i in range(4):
        items.append({"id": f"a{i}", "vector": [10.0 + i * 0.1, 10.0], "metadata": {"type": "A"}})
    for i in range(4):
        items.append({"id": f"b{i}", "vector": [90.0 + i * 0.1, 90.0], "metadata": {"type": "B"}})
    mem.write(items)
    return mem


def test_anomaly_filtered_to_type_recalls_only_that_type(store_factory):
    mem = _two_type_memory(store_factory())
    query = [90.0, 90.0]  # a normal type-B record

    against_b = mem.anomaly(query, flt={"type": "B"})
    against_a = mem.anomaly(query, flt={"type": "A"})

    # Recall only used the filtered patterns.
    assert against_b["nearest"]["id"].startswith("b")
    assert against_a["nearest"]["id"].startswith("a")
    # Normal-for-B → small residual; scored against A → large residual.
    assert against_b["score"] < against_a["score"]


def test_complete_filtered(store_factory):
    mem = _two_type_memory(store_factory())
    out = mem.complete([90.0, 0.0], mask=[0], flt={"type": "B"})
    # completion uses only B patterns → second field ~90, not ~10.
    assert out["reconstruction"][1] > 50
    assert out["top"]["id"].startswith("b")


def test_filter_excluding_all_returns_empty_shape(store_factory):
    mem = _two_type_memory(store_factory())
    a = mem.anomaly([1.0, 1.0], flt={"type": "ZZZ"})
    assert a["reconstruction"] is None and a["fields"] == []
    c = mem.complete([1.0, 1.0], flt={"type": "ZZZ"})
    assert c["reconstruction"] is None and c["top"] is None


def test_batch_anomaly_with_filter(store_factory):
    mem = _two_type_memory(store_factory())
    out = mem.anomaly_batch([[90.0, 90.0], [10.0, 10.0]], flt={"type": "B"})
    assert out[0]["nearest"]["id"].startswith("b")
    assert out[1]["nearest"]["id"].startswith("b")  # only B candidates exist


def test_filter_via_http(client):
    client.post(
        "/v1/memories",
        json={"name": "m", "dimension": 2, "fields": ["x", "y"], "beta": 6.0},
    )
    items = [{"id": f"a{i}", "vector": [10.0, 10.0], "metadata": {"type": "A"}} for i in range(3)]
    items += [{"id": f"b{i}", "vector": [90.0, 90.0], "metadata": {"type": "B"}} for i in range(3)]
    client.post("/v1/memories/m/patterns", json={"items": items})
    r = client.post("/v1/memories/m/anomaly", json={"query": [90, 90], "filter": {"type": "B"}})
    assert r.status_code == 200
    assert r.json()["nearest"]["id"].startswith("b")

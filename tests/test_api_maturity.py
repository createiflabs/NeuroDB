"""API maturity: /v1 versioning + legacy alias, pagination, error envelope,
and request-id propagation."""

from __future__ import annotations


def test_v1_routes_work(client):
    assert client.post("/v1/memories", json={"name": "m", "dimension": 2}).status_code == 201
    assert client.get("/v1/memories/m").status_code == 200


def test_legacy_routes_work_with_deprecation_header(client):
    resp = client.get("/v1/memories")  # prime nothing; just hit legacy below
    resp = client.post("/memories", json={"name": "leg", "dimension": 2})
    assert resp.status_code == 201
    assert resp.headers.get("Deprecation") == "true"
    assert "successor-version" in resp.headers.get("Link", "")


def test_v1_has_no_deprecation_header(client):
    resp = client.post("/v1/memories", json={"name": "m2", "dimension": 2})
    assert "Deprecation" not in resp.headers


def test_list_memories_pagination(client):
    for i in range(5):
        client.post("/v1/memories", json={"name": f"m{i}", "dimension": 2})
    body = client.get("/v1/memories", params={"limit": 2, "offset": 0}).json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert len(body["memories"]) == 2
    page2 = client.get("/v1/memories", params={"limit": 2, "offset": 4}).json()
    assert len(page2["memories"]) == 1


def test_error_envelope_and_request_id(client):
    resp = client.get("/v1/memories/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["request_id"]
    assert body["detail"]  # backward-compatible mirror
    assert resp.headers["X-Request-ID"] == body["error"]["request_id"]


def test_inbound_request_id_is_echoed(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_validation_error_envelope(client):
    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    resp = client.post("/v1/memories/m/patterns", json={"items": []})  # below min_length
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"

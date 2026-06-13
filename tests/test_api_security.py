"""Security hardening: fail-closed auth, CORS lockdown, security headers,
body-size cap, rate limiting, and input bounds."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurodb.server import create_app


def test_fail_closed_without_key(settings_factory):
    # No api_key and no allow_anonymous => refuse to start (lifespan raises).
    app = create_app(settings_factory())
    with pytest.raises(RuntimeError):  # noqa: PT012
        with TestClient(app):
            pass


def test_anonymous_allowed_when_opted_in(client):
    # `client` fixture sets allow_anonymous=True.
    assert client.get("/memories").status_code == 200


def test_auth_required_when_key_set(auth_client):
    assert auth_client.get("/memories").status_code == 401
    assert auth_client.get("/memories", headers={"X-API-Key": "secret"}).status_code == 200


def test_health_and_version_public(auth_client):
    assert auth_client.get("/health").status_code == 200
    assert auth_client.get("/version").status_code == 200


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in resp.headers


def test_cors_disabled_by_default(client):
    # No Access-Control-Allow-Origin echoed when cors_origins is empty.
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_cors_enabled_for_configured_origin(client_factory):
    client = client_factory(allow_anonymous=True, cors_origins=["https://ok.example"])
    resp = client.get("/health", headers={"Origin": "https://ok.example"})
    assert resp.headers.get("access-control-allow-origin") == "https://ok.example"


def test_body_size_limit_returns_413(client_factory):
    client = client_factory(allow_anonymous=True, max_request_bytes=2000)
    client.post("/memories", json={"name": "m", "dimension": 4})
    big = [{"id": str(i), "vector": [0, 0, 0, 0], "metadata": {"x": "y" * 200}} for i in range(50)]
    resp = client.post("/memories/m/patterns", json={"items": big})
    assert resp.status_code == 413


def test_rate_limit_returns_429(client_factory):
    client = client_factory(allow_anonymous=True, rate_limit_per_minute=3)
    client.post("/memories", json={"name": "m", "dimension": 2})
    codes = [
        client.post("/memories/m/search", json={"query": [1, 0], "k": 1}).status_code
        for _ in range(6)
    ]
    assert 429 in codes


def test_oversized_batch_rejected(client):
    client.post("/memories", json={"name": "m", "dimension": 2})
    items = [{"vector": [1, 0]} for _ in range(1001)]  # over BULK_MAX
    resp = client.post("/memories/m/patterns", json={"items": items})
    assert resp.status_code == 422

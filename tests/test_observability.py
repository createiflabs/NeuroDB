"""Observability: /health (liveness), /ready (readiness), /metrics, JSON logs."""

from __future__ import annotations

import json
import logging

from neurodb.observability import JsonFormatter, request_id_var


def test_health_is_public_and_live(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_returns_200_when_healthy(client):
    assert client.get("/ready").status_code == 200


def test_ready_503_after_save_failure(client_factory):
    client = client_factory(allow_anonymous=True)
    client.app.state.store.last_save_ok = False
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_metrics_endpoint_exposes_counters(client):
    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    client.post("/v1/memories/m/patterns", json={"items": [{"vector": [1, 0]}]})
    body = client.get("/metrics").text
    assert "neurodb_http_requests_total" in body
    assert "neurodb_patterns_total" in body


def test_metrics_and_ready_are_public(auth_client):
    assert auth_client.get("/ready").status_code == 200
    assert auth_client.get("/metrics").status_code == 200


def test_hot_path_latency_and_op_metrics(client):
    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    client.post("/v1/memories/m/patterns", json={"items": [{"id": "a", "vector": [1, 0]}]})
    client.post("/v1/memories/m/anomaly/batch", json={"items": [{"vector": [1, 0]}]})
    body = client.get("/metrics").text
    # Per-route latency histogram and the new batch-size / op counters populate.
    assert "neurodb_http_request_duration_seconds" in body
    assert "neurodb_batch_size" in body
    assert 'neurodb_ops_total{op="write"}' in body


def test_slowlog_captures_and_omits_contents(client_factory):
    # threshold 0 → every data op is recorded.
    client = client_factory(allow_anonymous=True, slowlog_ms=0.0)
    client.post("/v1/memories", json={"name": "m", "dimension": 2})
    client.post(
        "/v1/memories/m/anomaly/batch",
        json={"items": [{"vector": [1, 0]}, {"vector": [0, 1]}]},
    )

    body = client.get("/v1/slowlog").json()
    assert body["threshold_ms"] == 0.0
    assert body["entries"], "expected slow entries"

    entry = body["entries"][-1]
    # Shape metadata only — never record contents.
    allowed = {
        "ts", "duration_ms", "request_id", "method",
        "route", "status", "memory", "batch_size", "pattern_count",
    }
    assert set(entry) <= allowed
    assert entry["memory"] == "m"
    assert entry["batch_size"] == 2


def test_slowlog_is_auth_guarded(auth_client):
    # No API key header → 401 (slowlog is on the auth-protected data router).
    assert auth_client.get("/v1/slowlog").status_code == 401


def test_json_formatter_includes_request_id():
    formatter = JsonFormatter()
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", None, None)
    token = request_id_var.set("rid-xyz")
    try:
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    assert payload["msg"] == "hello"
    assert payload["request_id"] == "rid-xyz"

"""Observability: structured logging with request ids, and Prometheus metrics.

The metrics layer is optional — if ``prometheus_client`` is not installed the
``/metrics`` endpoint reports that gracefully and instrumentation is a no-op.
"""

from __future__ import annotations

import collections
import contextvars
import json
import logging
import threading
import time
from typing import Any

# Request id for the in-flight request, surfaced in every log line.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

try:  # pragma: no cover - import shim
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"


class JsonFormatter(logging.Formatter):
    """Minimal structured-log formatter that injects the request id."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "info", fmt: str = "json") -> None:
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


class Metrics:
    """A small Prometheus metrics facade with its own registry.

    All methods are safe no-ops when prometheus_client is unavailable.
    """

    def __init__(self) -> None:
        self.enabled = PROMETHEUS_AVAILABLE
        if not self.enabled:
            return
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "neurodb_http_requests_total",
            "HTTP requests by method, route and status.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "neurodb_http_request_duration_seconds",
            "HTTP request latency by method and route.",
            ["method", "path"],
            registry=self.registry,
        )
        self.memories = Gauge(
            "neurodb_memories", "Number of memories.", registry=self.registry
        )
        self.patterns = Gauge(
            "neurodb_patterns_total", "Number of stored patterns.", registry=self.registry
        )
        self.saves = Counter(
            "neurodb_save_total",
            "Store save attempts by result.",
            ["result"],
            registry=self.registry,
        )
        # Batch-size distribution per batch route (spot misuse / tune limits).
        self.batch_size = Histogram(
            "neurodb_batch_size",
            "Items per batch request, by route.",
            ["path"],
            buckets=(1, 8, 32, 128, 512, 1024, 4096),
            registry=self.registry,
        )
        # Data-mutation counters (low cardinality: by op, not by memory name).
        self.ops = Counter(
            "neurodb_ops_total",
            "Data operations by kind (write/delete/update).",
            ["op"],
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, duration: float) -> None:
        if not self.enabled:
            return
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(duration)

    def observe_batch(self, path: str, size: int) -> None:
        if not self.enabled:
            return
        self.batch_size.labels(path).observe(size)

    def record_op(self, op: str) -> None:
        if not self.enabled:
            return
        self.ops.labels(op).inc()

    def record_save(self, ok: bool) -> None:
        if not self.enabled:
            return
        self.saves.labels("ok" if ok else "error").inc()

    def render(self, memories: int, patterns: int) -> bytes:
        if not self.enabled:
            return b"# prometheus_client not installed\n"
        self.memories.set(memories)
        self.patterns.set(patterns)
        return generate_latest(self.registry)


class SlowLog:
    """A bounded, in-memory ring buffer of slow data operations (Redis SLOWLOG
    equivalent). Entries carry timing + shape metadata only — never record
    contents — and the newest evicts the oldest past ``maxlen``."""

    def __init__(self, threshold_ms: float, maxlen: int) -> None:
        self.threshold_ms = threshold_ms
        self._buf: collections.deque[dict[str, Any]] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("neurodb.slowlog")

    def record(self, duration_ms: float, **fields: Any) -> None:
        """Record an op if it crossed the threshold. Returns silently otherwise."""

        if duration_ms < self.threshold_ms:
            return
        entry = {"ts": time.time(), "duration_ms": round(duration_ms, 2), **fields}
        with self._lock:
            self._buf.append(entry)
        self._logger.warning("slow operation", extra={"slowlog": entry})

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        return items[-limit:] if limit else items

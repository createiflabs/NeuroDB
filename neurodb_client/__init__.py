"""NeuroDB Python client — a thin, dependency-free wrapper over the HTTP API.

    import neurodb_client as neurodb

    db = neurodb.connect("http://localhost:8000", api_key="...")
    mem = db.create("sensors", dimension=3, beta=12,
                    fields=["temperature", "humidity", "pressure"], normalize="zscore")
    mem.write([{"vector": [20, 50, 1013], "metadata": {"site": "A"}}])
    mem.anomaly([20, 95, 1013])           # single → dict
    mem.anomaly_batch([[20, 95, 1013], [21, 52, 1012]])
    mem.complete([20, 0, 0], mask=[0])
    mem.capacity()                         # Hopfield saturation diagnostic

Uses only the standard library (``urllib``), matching NeuroDB's lean ethos.
"""

from __future__ import annotations

from .client import (
    BadRequest,
    Client,
    Memory,
    NeuroDBError,
    NotFound,
    Unauthorized,
    connect,
)

__all__ = [
    "connect",
    "Client",
    "Memory",
    "NeuroDBError",
    "BadRequest",
    "NotFound",
    "Unauthorized",
]

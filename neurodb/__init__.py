"""NeuroDB — a lightweight, container-native vector database.

NeuroDB stores high-dimensional vectors (embeddings) alongside arbitrary JSON
metadata and serves fast nearest-neighbour similarity search over a clean REST
API. It is designed to be the long-term memory layer for AI applications:
semantic search, retrieval-augmented generation (RAG), recommendations and
deduplication.
"""

__version__ = "0.1.0"

# Re-export the Python client so the installed package users `pip install` —
# `neurodb` — exposes the high-level API directly:
#
#     from neurodb import connect, ValidationReport
#
# `neurodb_client` remains the dependency-free implementation home.
from neurodb_client import (  # noqa: E402
    BadRequest,
    Client,
    FieldResult,
    Memory,
    NeuroDBError,
    NotFound,
    RecordResult,
    Unauthorized,
    ValidationReport,
    connect,
    run_validation,
    telemetry,
)

__all__ = [
    "__version__",
    "connect",
    "Client",
    "Memory",
    "NeuroDBError",
    "BadRequest",
    "NotFound",
    "Unauthorized",
    "run_validation",
    "ValidationReport",
    "RecordResult",
    "FieldResult",
    "telemetry",
]

"""Runtime configuration for NeuroDB, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """All knobs are configured via ``NEURODB_*`` environment variables."""

    data_file: str = field(
        default_factory=lambda: _env_str("NEURODB_DATA_FILE", "./data/neurodb.npz")
    )
    host: str = field(default_factory=lambda: _env_str("NEURODB_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("NEURODB_PORT", 8000))
    autosave_interval: float = field(
        default_factory=lambda: _env_float("NEURODB_AUTOSAVE_INTERVAL", 5.0)
    )
    api_key: str | None = field(default_factory=lambda: os.environ.get("NEURODB_API_KEY") or None)
    # Secure by default: refuse to start without a key unless this is set.
    allow_anonymous: bool = field(
        default_factory=lambda: _env_bool("NEURODB_ALLOW_ANONYMOUS", False)
    )
    # Locked-down by default (no cross-origin); set explicit origins to open up.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("NEURODB_CORS_ORIGINS", [])
    )
    # Requests larger than this (bytes) are rejected with 413 before reading.
    max_request_bytes: int = field(
        default_factory=lambda: _env_int("NEURODB_MAX_REQUEST_BYTES", 8 * 1024 * 1024)
    )
    # Per-client (API key, else IP) request budget per minute; 0 disables.
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("NEURODB_RATE_LIMIT_PER_MINUTE", 600)
    )
    # Fail-closed on an unreadable data file instead of quarantining + empty start.
    fail_on_corrupt_load: bool = field(
        default_factory=lambda: _env_bool("NEURODB_FAIL_ON_CORRUPT_LOAD", False)
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("NEURODB_EMBEDDING_DIM", 256))
    log_level: str = field(default_factory=lambda: _env_str("NEURODB_LOG_LEVEL", "info"))
    log_format: str = field(default_factory=lambda: _env_str("NEURODB_LOG_FORMAT", "json"))


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` instance from the current environment."""

    return Settings()

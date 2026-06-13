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
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("NEURODB_CORS_ORIGINS", ["*"])
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("NEURODB_EMBEDDING_DIM", 256))
    log_level: str = field(default_factory=lambda: _env_str("NEURODB_LOG_LEVEL", "info"))


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` instance from the current environment."""

    return Settings()

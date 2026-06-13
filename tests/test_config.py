"""Environment-variable parsing in neurodb.config."""

from __future__ import annotations

import pytest

from neurodb.config import Settings, _env_bool, _env_float, _env_int, _env_list


@pytest.mark.parametrize(
    "raw,default,expected",
    [("42", 0, 42), ("", 7, 7), ("notanumber", 7, 7), (None, 5, 5)],
)
def test_env_int(monkeypatch, raw, default, expected):
    if raw is None:
        monkeypatch.delenv("X_TEST", raising=False)
    else:
        monkeypatch.setenv("X_TEST", raw)
    assert _env_int("X_TEST", default) == expected


@pytest.mark.parametrize(
    "raw,default,expected",
    [("1.5", 0.0, 1.5), ("", 2.0, 2.0), ("bad", 2.0, 2.0)],
)
def test_env_float(monkeypatch, raw, default, expected):
    monkeypatch.setenv("X_TEST", raw)
    assert _env_float("X_TEST", default) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("no", False)],
)
def test_env_bool(monkeypatch, raw, expected):
    monkeypatch.setenv("X_TEST", raw)
    assert _env_bool("X_TEST", False) is expected


def test_env_list_splits_and_trims(monkeypatch):
    monkeypatch.setenv("X_TEST", " a , b ,, c ")
    assert _env_list("X_TEST", []) == ["a", "b", "c"]


def test_secure_defaults(monkeypatch):
    for var in ("NEURODB_API_KEY", "NEURODB_ALLOW_ANONYMOUS", "NEURODB_CORS_ORIGINS"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    assert settings.api_key is None
    assert settings.allow_anonymous is False
    assert settings.cors_origins == []  # locked down
    assert settings.max_request_bytes > 0

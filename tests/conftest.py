"""Shared pytest fixtures and configuration for the NeuroDB test suite.

Every fixture isolates state under pytest's ``tmp_path`` and defaults the
autosave interval to ``0.0`` so the background autosave loop stays quiet unless
a test explicitly opts in. New test modules should build on these rather than
re-creating ``Settings``/``TestClient`` by hand.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import NeuroStore

# -- slow / load tests ----------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (load/perf smoke tests)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: load/perf smoke tests; only run with --run-slow"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# -- store / settings / client factories ----------------------------------


@pytest.fixture()
def data_file(tmp_path) -> str:
    """Absolute path to an (initially absent) data file under tmp_path."""

    return str(tmp_path / "db.npz")


@pytest.fixture()
def store_factory(tmp_path) -> Callable[..., NeuroStore]:
    """Factory for a :class:`NeuroStore` backed by a tmp_path data file.

    Call ``make()`` twice with the same ``filename`` to exercise a
    save → reload round-trip against the same path.
    """

    def make(filename: str = "db.npz", **kwargs) -> NeuroStore:
        return NeuroStore(tmp_path / filename, **kwargs)

    return make


@pytest.fixture()
def settings_factory(tmp_path) -> Callable[..., Settings]:
    """Factory for :class:`Settings` with safe test defaults + overrides."""

    def make(**overrides) -> Settings:
        params: dict = {
            "data_file": str(tmp_path / "db.npz"),
            "autosave_interval": 0.0,
        }
        params.update(overrides)
        return Settings(**params)

    return make


@pytest.fixture()
def client_factory(settings_factory) -> Callable[..., TestClient]:
    """Factory yielding context-managed TestClients (lifespan exercised).

    Created clients are closed at fixture teardown, so tests never need to
    manage the ``with`` block themselves.
    """

    created: list[TestClient] = []

    def make(**setting_overrides) -> TestClient:
        settings = settings_factory(**setting_overrides)
        client = TestClient(create_app(settings))
        client.__enter__()
        created.append(client)
        return client

    yield make

    for client in created:
        client.__exit__(None, None, None)


@pytest.fixture()
def client(client_factory) -> TestClient:
    """Default anonymous client (auth explicitly disabled for convenience)."""

    return client_factory(allow_anonymous=True)


@pytest.fixture()
def db(client):
    """A NeuroDB Python client whose transport drives the in-process TestClient."""

    from neurodb_client import connect

    def transport(method, url, headers, data, timeout):
        resp = client.request(method, url, content=data, headers=headers)
        return resp.status_code, resp.content

    return connect("", transport=transport)


@pytest.fixture()
def api_headers() -> dict[str, str]:
    """Headers carrying the key used by :func:`auth_client`."""

    return {"X-API-Key": "secret"}


@pytest.fixture()
def auth_client(client_factory) -> TestClient:
    """Client whose app requires the API key ``"secret"``."""

    return client_factory(api_key="secret")

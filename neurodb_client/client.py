"""Stdlib HTTP client for the NeuroDB API.

The transport is pluggable (``transport=`` on :class:`Client`): the default
drives ``urllib`` against a real server, and the test suite injects one backed by
FastAPI's in-process ``TestClient``. Everything else is plain dicts in/out.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from typing import Any

Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


# -- typed errors mapping the server's error envelope ----------------------
class NeuroDBError(Exception):
    """Base error for any non-2xx response."""


class BadRequest(NeuroDBError):
    """400/413/422 — malformed request, oversize batch, validation failure."""


class NotFound(NeuroDBError):
    """404 — memory or pattern does not exist."""


class Unauthorized(NeuroDBError):
    """401 — missing or invalid API key."""


def _urllib_transport(
    method: str, url: str, headers: dict[str, str], data: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - caller's URL
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx carry a JSON body we want
        return exc.code, exc.read()


def connect(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    transport: Transport | None = None,
) -> Client:
    """Create a :class:`Client` bound to ``base_url``."""

    return Client(base_url, api_key=api_key, timeout=timeout, transport=transport)


class Client:
    """A connection to a NeuroDB server."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport or _urllib_transport

    # -- request plumbing ------------------------------------------------
    def _request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        status, raw = self._transport(method, self.base_url + path, headers, data, self.timeout)
        if status >= 400:
            raise self._error(status, raw)
        return json.loads(raw) if raw else None

    @staticmethod
    def _error(status: int, raw: bytes) -> NeuroDBError:
        message: str = f"HTTP {status}"
        try:
            payload = json.loads(raw)
            message = (
                payload.get("detail")
                or payload.get("error", {}).get("message")
                or message
            )
        except (ValueError, AttributeError):
            if raw:
                message = raw.decode("utf-8", "replace")
        if status == 404:
            return NotFound(message)
        if status == 401:
            return Unauthorized(message)
        if status in (400, 413, 422):
            return BadRequest(message)
        return NeuroDBError(message)

    # -- memory lifecycle ------------------------------------------------
    def create(
        self,
        name: str,
        dimension: int,
        beta: float = 8.0,
        fields: Sequence[str] | None = None,
        normalize: str | None = None,
    ) -> Memory:
        body: dict[str, Any] = {"name": name, "dimension": dimension, "beta": beta}
        if fields is not None:
            body["fields"] = list(fields)
        if normalize is not None:
            body["normalize"] = normalize
        info = self._request("POST", "/v1/memories", body)
        return Memory(self, name, info)

    def memory(self, name: str) -> Memory:
        """Bind to an existing memory (fetches its info)."""

        info = self._request("GET", f"/v1/memories/{name}")
        return Memory(self, name, info)

    def memories(self) -> Any:
        return self._request("GET", "/v1/memories")

    def delete(self, name: str) -> Any:
        return self._request("DELETE", f"/v1/memories/{name}")

    def stats(self) -> Any:
        return self._request("GET", "/v1/stats")

    def health(self) -> Any:
        return self._request("GET", "/health")


def _as_items(queries: Iterable[Any]) -> list[dict[str, Any]]:
    """Accept a list of plain vectors or ``{vector, id?}`` dicts."""

    items: list[dict[str, Any]] = []
    for q in queries:
        items.append(q if isinstance(q, dict) else {"vector": list(q)})
    return items


class Memory:
    """A handle to one memory; methods map 1:1 to the HTTP endpoints."""

    def __init__(self, client: Client, name: str, info: dict[str, Any] | None = None) -> None:
        self._client = client
        self.name = name
        self.info = info or {}

    def _path(self, suffix: str) -> str:
        return f"/v1/memories/{self.name}{suffix}"

    # -- patterns --------------------------------------------------------
    def write(self, items: Sequence[dict[str, Any]]) -> Any:
        return self._client._request("POST", self._path("/patterns"), {"items": list(items)})

    def get(self, pattern_id: str) -> Any:
        return self._client._request("GET", self._path(f"/patterns/{pattern_id}"))

    def update(
        self,
        pattern_id: str,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> Any:
        body: dict[str, Any] = {"merge_metadata": merge_metadata}
        if vector is not None:
            body["vector"] = list(vector)
        if metadata is not None:
            body["metadata"] = metadata
        return self._client._request("PATCH", self._path(f"/patterns/{pattern_id}"), body)

    def delete_pattern(self, pattern_id: str) -> Any:
        return self._client._request("DELETE", self._path(f"/patterns/{pattern_id}"))

    # -- recall ----------------------------------------------------------
    def complete(
        self,
        query: Sequence[float],
        beta: float | None = None,
        mask: Sequence[int] | None = None,
        steps: int = 1,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {"query": list(query), "steps": steps, "top_k": top_k}
        if beta is not None:
            body["beta"] = beta
        if mask is not None:
            body["mask"] = list(mask)
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", self._path("/complete"), body)

    def complete_batch(
        self,
        queries: Iterable[Any],
        beta: float | None = None,
        mask: Sequence[int] | None = None,
        steps: int = 1,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {"items": _as_items(queries), "steps": steps, "top_k": top_k}
        if beta is not None:
            body["beta"] = beta
        if mask is not None:
            body["mask"] = list(mask)
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", self._path("/complete/batch"), body)

    def anomaly(
        self,
        query: Sequence[float],
        beta: float | None = None,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {"query": list(query), "top_k": top_k}
        if beta is not None:
            body["beta"] = beta
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", self._path("/anomaly"), body)

    def anomaly_batch(
        self,
        queries: Iterable[Any],
        beta: float | None = None,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {"items": _as_items(queries), "top_k": top_k}
        if beta is not None:
            body["beta"] = beta
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", self._path("/anomaly/batch"), body)

    def search(
        self,
        query: Sequence[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_vectors: bool = False,
        metric: str = "cosine",
    ) -> Any:
        body: dict[str, Any] = {
            "query": list(query),
            "k": k,
            "include_vectors": include_vectors,
            "metric": metric,
        }
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", self._path("/search"), body)

    def capacity(self) -> Any:
        return self._client._request("GET", self._path("/capacity"))

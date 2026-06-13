"""The NeuroDB storage engine: collections of vectors with metadata.

The engine keeps each collection's vectors in a contiguous ``float32`` matrix so
that similarity search is a single vectorised numpy operation. Collections are
persisted to disk as a ``vectors.npy`` matrix plus a ``meta.json`` sidecar and
are reloaded on startup. All public mutating operations are guarded by a lock so
the engine is safe to use from a threaded ASGI server.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import SUPPORTED_METRICS, compute_scores

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StoreError(Exception):
    """Base class for storage-layer errors (maps to HTTP 400)."""


class CollectionError(StoreError):
    """Invalid collection request (bad name, dimension mismatch, conflict)."""


class NotFoundError(StoreError):
    """A requested collection or vector does not exist (maps to HTTP 404)."""


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise CollectionError(
            "Invalid name. Use 1-128 chars of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    return name


def _match_filter(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Mongo-ish metadata filtering.

    Plain values mean equality; a list means "value in list"; a dict may use the
    operators ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$gt``, ``$gte``, ``$lt``,
    ``$lte``.
    """

    for key, want in flt.items():
        have = metadata.get(key)
        if isinstance(want, dict):
            for op, val in want.items():
                if op == "$eq" and have != val:
                    return False
                if op == "$ne" and have == val:
                    return False
                if op == "$in" and have not in val:
                    return False
                if op == "$nin" and have in val:
                    return False
                if op == "$gt" and not (have is not None and have > val):
                    return False
                if op == "$gte" and not (have is not None and have >= val):
                    return False
                if op == "$lt" and not (have is not None and have < val):
                    return False
                if op == "$lte" and not (have is not None and have <= val):
                    return False
        elif isinstance(want, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class Collection:
    """A named set of equal-length vectors with attached metadata."""

    def __init__(self, name: str, dimension: int, metric: str = "cosine") -> None:
        validate_name(name)
        if metric not in SUPPORTED_METRICS:
            raise CollectionError(
                f"Unsupported metric {metric!r}. Choose from {SUPPORTED_METRICS}."
            )
        if int(dimension) <= 0:
            raise CollectionError("dimension must be a positive integer")
        self.name = name
        self.dimension = int(dimension)
        self.metric = metric
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)
        self._lock = threading.RLock()
        self.dirty = False

    # -- introspection ----------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.ids)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric,
            "count": self.count,
        }

    # -- validation helpers ----------------------------------------------
    def _coerce_vector(self, vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise CollectionError(
                f"Vector has dimension {arr.shape[0]} but collection "
                f"{self.name!r} expects {self.dimension}."
            )
        if not np.all(np.isfinite(arr)):
            raise CollectionError("Vector contains NaN or infinite values.")
        return arr

    # -- mutations --------------------------------------------------------
    def upsert(self, items: Iterable[dict[str, Any]]) -> list[str]:
        """Insert or update a batch of items.

        Each item is a dict with ``vector`` (required), optional ``id`` (a random
        hex id is generated when absent) and optional ``metadata``. Returns the
        list of affected ids.
        """

        with self._lock:
            affected: list[str] = []
            pending_pos: dict[str, int] = {}
            new_rows: list[np.ndarray] = []
            new_ids: list[str] = []
            new_meta: list[dict[str, Any]] = []

            for item in items:
                if "vector" not in item:
                    raise CollectionError("Each item must include a 'vector'.")
                vec = self._coerce_vector(item["vector"])
                _id = str(item.get("id") or uuid.uuid4().hex)
                meta = dict(item.get("metadata") or {})
                affected.append(_id)

                if _id in self._index:
                    idx = self._index[_id]
                    self._matrix[idx] = vec
                    self.metadata[idx] = meta
                elif _id in pending_pos:
                    pos = pending_pos[_id]
                    new_rows[pos] = vec
                    new_meta[pos] = meta
                else:
                    pending_pos[_id] = len(new_rows)
                    new_rows.append(vec)
                    new_ids.append(_id)
                    new_meta.append(meta)

            if new_rows:
                block = np.vstack(new_rows).astype(np.float32)
                if self._matrix.shape[0] == 0:
                    self._matrix = block
                else:
                    self._matrix = np.vstack([self._matrix, block])
                start = len(self.ids)
                for offset, _id in enumerate(new_ids):
                    self._index[_id] = start + offset
                self.ids.extend(new_ids)
                self.metadata.extend(new_meta)

            self.dirty = True
            return affected

    def get(self, _id: str) -> dict[str, Any]:
        with self._lock:
            if _id not in self._index:
                raise NotFoundError(f"Vector {_id!r} not found in collection {self.name!r}.")
            idx = self._index[_id]
            return {
                "id": _id,
                "vector": self._matrix[idx].tolist(),
                "metadata": self.metadata[idx],
            }

    def delete(self, ids: Iterable[str]) -> int:
        with self._lock:
            targets = [i for i in ids if i in self._index]
            if not targets:
                return 0
            drop = np.array(sorted(self._index[i] for i in targets), dtype=np.int64)
            mask = np.ones(self._matrix.shape[0], dtype=bool)
            mask[drop] = False
            self._matrix = self._matrix[mask]
            self.ids = [i for j, i in enumerate(self.ids) if mask[j]]
            self.metadata = [m for j, m in enumerate(self.metadata) if mask[j]]
            self._index = {i: j for j, i in enumerate(self.ids)}
            self.dirty = True
            return len(targets)

    # -- search -----------------------------------------------------------
    def search(
        self,
        query: Iterable[float],
        k: int = 10,
        flt: dict[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            n = self._matrix.shape[0]
            if n == 0 or k <= 0:
                return []
            q = self._coerce_vector(query)
            scores = compute_scores(self._matrix, q, self.metric)

            if flt:
                keep = np.fromiter(
                    (_match_filter(self.metadata[i], flt) for i in range(n)),
                    dtype=bool,
                    count=n,
                )
                if not keep.any():
                    return []
                scores = np.where(keep, scores, -np.inf)

            limit = min(k, n)
            # Partial sort for the top-`limit`, then order that slice exactly.
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top], kind="stable")]

            results: list[dict[str, Any]] = []
            for i in top:
                i = int(i)
                score = float(scores[i])
                if not np.isfinite(score):
                    continue
                row: dict[str, Any] = {
                    "id": self.ids[i],
                    "score": score,
                    "metadata": self.metadata[i],
                }
                if include_vectors:
                    row["vector"] = self._matrix[i].tolist()
                results.append(row)
            return results

    # -- persistence ------------------------------------------------------
    def save(self, directory: Path) -> None:
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)

            vec_tmp = directory / "vectors.npy.tmp"
            with open(vec_tmp, "wb") as handle:
                np.save(handle, self._matrix)
            vec_tmp.replace(directory / "vectors.npy")

            meta = {
                "name": self.name,
                "dimension": self.dimension,
                "metric": self.metric,
                "ids": self.ids,
                "metadata": self.metadata,
            }
            meta_tmp = directory / "meta.json.tmp"
            meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
            meta_tmp.replace(directory / "meta.json")

            self.dirty = False

    @classmethod
    def load(cls, directory: Path) -> Collection:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        col = cls(meta["name"], meta["dimension"], meta["metric"])
        col.ids = list(meta["ids"])
        col.metadata = list(meta["metadata"])
        col._index = {i: j for j, i in enumerate(col.ids)}
        matrix_path = directory / "vectors.npy"
        if matrix_path.exists():
            col._matrix = np.load(matrix_path).astype(np.float32, copy=False)
        else:
            col._matrix = np.zeros((0, col.dimension), dtype=np.float32)
        col.dirty = False
        return col


class VectorStore:
    """Top-level registry that owns every :class:`Collection` and the data dir."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._collections: dict[str, Collection] = {}
        self._lock = threading.RLock()
        self.load_all()

    # -- collection lifecycle --------------------------------------------
    def _dir(self, name: str) -> Path:
        return self.data_dir / name

    def load_all(self) -> None:
        if not self.data_dir.exists():
            return
        for child in sorted(self.data_dir.iterdir()):
            if child.is_dir() and (child / "meta.json").exists():
                col = Collection.load(child)
                self._collections[col.name] = col

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> Collection:
        with self._lock:
            validate_name(name)
            if name in self._collections:
                raise CollectionError(f"Collection {name!r} already exists.")
            col = Collection(name, dimension, metric)
            self._collections[name] = col
            col.save(self._dir(name))
            return col

    def get_collection(self, name: str) -> Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise NotFoundError(f"Collection {name!r} not found.") from None

    def list_collections(self) -> list[dict[str, Any]]:
        with self._lock:
            return [col.info() for col in self._collections.values()]

    def delete_collection(self, name: str) -> None:
        with self._lock:
            if name not in self._collections:
                raise NotFoundError(f"Collection {name!r} not found.")
            del self._collections[name]
            directory = self._dir(name)
            if directory.exists():
                for child in directory.iterdir():
                    child.unlink()
                directory.rmdir()

    # -- stats & persistence ---------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            collections = [col.info() for col in self._collections.values()]
            return {
                "collections": len(collections),
                "vectors": sum(c["count"] for c in collections),
                "detail": collections,
            }

    def flush(self) -> int:
        """Persist any collections with unsaved changes. Returns how many."""

        with self._lock:
            flushed = 0
            for name, col in self._collections.items():
                if col.dirty:
                    col.save(self._dir(name))
                    flushed += 1
            return flushed

    def save_all(self) -> None:
        with self._lock:
            for name, col in self._collections.items():
                col.save(self._dir(name))

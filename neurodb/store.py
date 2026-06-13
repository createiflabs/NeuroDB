"""The NeuroDB storage engine: Modern Hopfield associative memories.

A :class:`Memory` stores patterns (vectors) as the rows of a matrix and offers
three content-addressable operations, all built on a single Hopfield attention
step:

* **complete** — recall / pattern completion from a full or partial query,
* **search** — nearest stored patterns by cosine similarity,
* **anomaly** — per-field deviation of an input from its recalled pattern.

:class:`NeuroStore` owns every memory and persists the whole store to a single
``.npz`` file (single-file persistence) that is reloaded on startup.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .hopfield import retrieve
from .metrics import compute_scores

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANIFEST_VERSION = 1


class StoreError(Exception):
    """Base class for storage-layer errors (maps to HTTP 400)."""


class MemoryError_(StoreError):
    """Invalid memory request (bad name, dimension mismatch, conflict)."""


class NotFoundError(StoreError):
    """A requested memory or pattern does not exist (maps to HTTP 404)."""


# Public, clearly-named alias (``MemoryError`` is a Python builtin).
MemoryError = MemoryError_


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise MemoryError_(
            "Invalid name. Use 1-128 chars of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    return name


def _match_filter(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Mongo-ish metadata filtering used by :meth:`Memory.search`."""

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


class Memory:
    """A Modern Hopfield associative memory: patterns + content-addressable recall."""

    def __init__(
        self,
        name: str,
        dimension: int,
        beta: float = 8.0,
        fields: list[str] | None = None,
    ) -> None:
        validate_name(name)
        if int(dimension) <= 0:
            raise MemoryError_("dimension must be a positive integer")
        if beta <= 0:
            raise MemoryError_("beta (inverse temperature) must be positive")
        if fields is not None and len(fields) != int(dimension):
            raise MemoryError_(
                f"fields has {len(fields)} names but dimension is {dimension}."
            )
        self.name = name
        self.dimension = int(dimension)
        self.beta = float(beta)
        self.fields: list[str] | None = list(fields) if fields else None
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._X = np.zeros((0, self.dimension), dtype=np.float32)
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
            "beta": self.beta,
            "fields": self.fields,
            "count": self.count,
        }

    # -- validation helpers ----------------------------------------------
    def _coerce_vector(self, vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise MemoryError_(
                f"Vector has dimension {arr.shape[0]} but memory "
                f"{self.name!r} expects {self.dimension}."
            )
        if not np.all(np.isfinite(arr)):
            raise MemoryError_("Vector contains NaN or infinite values.")
        return arr

    def _mask_from_indices(self, indices: Iterable[int] | None) -> np.ndarray | None:
        if indices is None:
            return None
        mask = np.zeros(self.dimension, dtype=bool)
        for i in indices:
            if not (0 <= int(i) < self.dimension):
                raise MemoryError_(f"mask index {i} out of range [0, {self.dimension}).")
            mask[int(i)] = True
        return mask

    # -- writing (append a pattern) --------------------------------------
    def write(self, items: Iterable[dict[str, Any]]) -> list[str]:
        """Append patterns. Each item: ``{vector, id?, metadata?}``. Existing ids
        are overwritten. Returns the affected ids."""

        with self._lock:
            affected: list[str] = []
            pending_pos: dict[str, int] = {}
            new_rows: list[np.ndarray] = []
            new_ids: list[str] = []
            new_meta: list[dict[str, Any]] = []

            for item in items:
                if "vector" not in item:
                    raise MemoryError_("Each item must include a 'vector'.")
                vec = self._coerce_vector(item["vector"])
                _id = str(item.get("id") or uuid.uuid4().hex)
                meta = dict(item.get("metadata") or {})
                affected.append(_id)

                if _id in self._index:
                    idx = self._index[_id]
                    self._X[idx] = vec
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
                self._X = block if self._X.shape[0] == 0 else np.vstack([self._X, block])
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
                raise NotFoundError(f"Pattern {_id!r} not found in memory {self.name!r}.")
            idx = self._index[_id]
            return {"id": _id, "vector": self._X[idx].tolist(), "metadata": self.metadata[idx]}

    def delete(self, ids: Iterable[str]) -> int:
        with self._lock:
            targets = [i for i in ids if i in self._index]
            if not targets:
                return 0
            drop = np.array(sorted(self._index[i] for i in targets), dtype=np.int64)
            mask = np.ones(self._X.shape[0], dtype=bool)
            mask[drop] = False
            self._X = self._X[mask]
            self.ids = [i for j, i in enumerate(self.ids) if mask[j]]
            self.metadata = [m for j, m in enumerate(self.metadata) if mask[j]]
            self._index = {i: j for j, i in enumerate(self.ids)}
            self.dirty = True
            return len(targets)

    # -- content-addressable operations ----------------------------------
    def _contributors(self, weights: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        n = weights.shape[0]
        if n == 0:
            return []
        k = min(max(top_k, 0), n)
        if k == 0:
            return []
        top = np.argpartition(-weights, k - 1)[:k]
        top = top[np.argsort(-weights[top], kind="stable")]
        return [
            {
                "id": self.ids[int(i)],
                "weight": float(weights[int(i)]),
                "metadata": self.metadata[int(i)],
            }
            for i in top
        ]

    def complete(
        self,
        query: Iterable[float],
        beta: float | None = None,
        mask: Iterable[int] | None = None,
        steps: int = 1,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Content-addressable recall / pattern completion via one (or more)
        Hopfield attention steps."""

        with self._lock:
            if self._X.shape[0] == 0:
                raise MemoryError_(f"Memory {self.name!r} is empty; write patterns first.")
            q = self._coerce_vector(query)
            b = float(beta) if beta is not None else self.beta
            mask_arr = self._mask_from_indices(mask)
            recon, weights = retrieve(self._X, q, b, mask_arr, steps)
            contributors = self._contributors(weights, top_k)
            return {
                "reconstruction": recon.tolist(),
                "weights": contributors,
                "top": contributors[0] if contributors else None,
                "beta": b,
                "steps": max(1, steps),
            }

    def search(
        self,
        query: Iterable[float],
        k: int = 10,
        flt: dict[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Nearest stored patterns by cosine similarity."""

        with self._lock:
            n = self._X.shape[0]
            if n == 0 or k <= 0:
                return []
            q = self._coerce_vector(query)
            scores = compute_scores(self._X, q, "cosine")
            if flt:
                keep = np.fromiter(
                    (_match_filter(self.metadata[i], flt) for i in range(n)), dtype=bool, count=n
                )
                if not keep.any():
                    return []
                scores = np.where(keep, scores, -np.inf)
            limit = min(k, n)
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
                    row["vector"] = self._X[i].tolist()
                results.append(row)
            return results

    def anomaly(
        self,
        query: Iterable[float],
        beta: float | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Per-field anomaly detection.

        Recall the pattern the input most resembles, then report where the input
        deviates from that reconstruction field-by-field. Fields with the largest
        absolute deviation are the most anomalous.
        """

        with self._lock:
            if self._X.shape[0] == 0:
                raise MemoryError_(f"Memory {self.name!r} is empty; write patterns first.")
            q = self._coerce_vector(query)
            b = float(beta) if beta is not None else self.beta
            recon, weights = retrieve(self._X, q, b, None, 1)
            residual = np.abs(q - recon)
            score = float(np.linalg.norm(q - recon))
            order = np.argsort(-residual)
            limit = min(max(top_k, 0), self.dimension)
            fields = []
            for idx in order[:limit]:
                idx = int(idx)
                fields.append(
                    {
                        "index": idx,
                        "name": self.fields[idx] if self.fields else None,
                        "value": float(q[idx]),
                        "expected": float(recon[idx]),
                        "deviation": float(residual[idx]),
                    }
                )
            nearest = self._contributors(weights, 1)
            return {
                "score": score,
                "reconstruction": recon.tolist(),
                "residual": residual.tolist(),
                "fields": fields,
                "nearest": nearest[0] if nearest else None,
                "beta": b,
            }

    # -- (de)serialisation for single-file persistence -------------------
    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "beta": self.beta,
            "fields": self.fields,
            "ids": self.ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], matrix: np.ndarray) -> Memory:
        mem = cls(
            manifest["name"],
            manifest["dimension"],
            manifest.get("beta", 8.0),
            manifest.get("fields"),
        )
        mem.ids = list(manifest["ids"])
        mem.metadata = list(manifest["metadata"])
        mem._index = {i: j for j, i in enumerate(mem.ids)}
        mem._X = matrix.astype(np.float32, copy=False)
        mem.dirty = False
        return mem


class NeuroStore:
    """Owns every :class:`Memory` and persists them all to a single ``.npz`` file."""

    def __init__(self, data_file: str | Path) -> None:
        self.data_file = Path(data_file)
        self._memories: dict[str, Memory] = {}
        self._lock = threading.RLock()
        self.load()

    # -- memory lifecycle -------------------------------------------------
    def create_memory(
        self, name: str, dimension: int, beta: float = 8.0, fields: list[str] | None = None
    ) -> Memory:
        with self._lock:
            validate_name(name)
            if name in self._memories:
                raise MemoryError_(f"Memory {name!r} already exists.")
            mem = Memory(name, dimension, beta, fields)
            self._memories[name] = mem
            self.save()
            return mem

    def get_memory(self, name: str) -> Memory:
        try:
            return self._memories[name]
        except KeyError:
            raise NotFoundError(f"Memory {name!r} not found.") from None

    def list_memories(self) -> list[dict[str, Any]]:
        with self._lock:
            return [mem.info() for mem in self._memories.values()]

    def delete_memory(self, name: str) -> None:
        with self._lock:
            if name not in self._memories:
                raise NotFoundError(f"Memory {name!r} not found.")
            del self._memories[name]
            self.save()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            memories = [mem.info() for mem in self._memories.values()]
            return {
                "memories": len(memories),
                "patterns": sum(m["count"] for m in memories),
                "data_file": str(self.data_file),
                "detail": memories,
            }

    # -- single-file persistence -----------------------------------------
    def load(self) -> None:
        if not self.data_file.exists():
            return
        with np.load(self.data_file, allow_pickle=False) as data:
            manifest = json.loads(bytes(data["__manifest__"]).decode("utf-8"))
            for entry in manifest.get("memories", []):
                matrix = data[f"X@{entry['name']}"]
                mem = Memory.from_manifest(entry, matrix)
                self._memories[mem.name] = mem

    def save(self) -> None:
        with self._lock:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "version": _MANIFEST_VERSION,
                "memories": [mem.manifest() for mem in self._memories.values()],
            }
            arrays: dict[str, np.ndarray] = {
                "__manifest__": np.frombuffer(
                    json.dumps(manifest).encode("utf-8"), dtype=np.uint8
                )
            }
            for mem in self._memories.values():
                arrays[f"X@{mem.name}"] = mem._X
            tmp = self.data_file.with_suffix(self.data_file.suffix + ".tmp")
            with open(tmp, "wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp, self.data_file)
            for mem in self._memories.values():
                mem.dirty = False

    def flush(self) -> int:
        with self._lock:
            dirty = sum(1 for mem in self._memories.values() if mem.dirty)
            if dirty:
                self.save()
            return dirty

    def save_all(self) -> None:
        self.save()

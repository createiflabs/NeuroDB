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

import copy
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hopfield import retrieve
from .metrics import compute_scores

logger = logging.getLogger("neurodb.store")

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


_COMPARISON_OPS = ("$gt", "$gte", "$lt", "$lte")
_FILTER_OPS = ("$eq", "$ne", "$in", "$nin", *_COMPARISON_OPS)


def validate_filter(flt: dict[str, Any]) -> None:
    """Reject malformed filters up front (→ 400) instead of crashing mid-scan.

    ``$in``/``$nin`` require a list/tuple/set operand; all operator keys must be
    recognised.
    """

    for want in flt.values():
        if not isinstance(want, dict):
            continue
        for op, val in want.items():
            if op not in _FILTER_OPS:
                raise MemoryError_(f"Unknown filter operator {op!r}.")
            if op in ("$in", "$nin") and not isinstance(val, (list, tuple, set)):
                raise MemoryError_(
                    f"Operator {op} expects a list, got {type(val).__name__}."
                )


def _compare(have: Any, val: Any, op: str) -> bool:
    """Total ordering comparison: incomparable types are simply non-matching."""

    if have is None:
        return False
    try:
        if op == "$gt":
            return have > val
        if op == "$gte":
            return have >= val
        if op == "$lt":
            return have < val
        return have <= val  # "$lte"
    except TypeError:
        return False


def _match_filter(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Mongo-ish metadata filtering used by :meth:`Memory.search`.

    Assumes the filter has already passed :func:`validate_filter`.
    """

    for key, want in flt.items():
        have = metadata.get(key)
        if isinstance(want, dict):
            for op, val in want.items():
                if op == "$eq" and have != val:
                    return False
                if op == "$ne" and have == val:
                    return False
                if op == "$in" and have not in val:  # membership (val is a list)
                    return False
                if op == "$nin" and have in val:
                    return False
                if op in _COMPARISON_OPS and not _compare(have, val, op):
                    return False
        elif isinstance(want, list):
            if have not in want:  # bare list == membership
                return False
        elif have != want:
            return False
    return True


@dataclass(frozen=True)
class MemorySnapshot:
    """An immutable, point-in-time copy of a :class:`Memory` for persistence.

    Produced under the memory's own lock so the matrix and the id/metadata
    lists are always mutually consistent, even if the memory is mutated
    concurrently while the snapshot is being serialized.
    """

    name: str
    dimension: int
    beta: float
    fields: list[str] | None
    ids: list[str]
    metadata: list[dict[str, Any]]
    matrix: np.ndarray
    version: int

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "beta": self.beta,
            "fields": self.fields,
            "ids": self.ids,
            "metadata": self.metadata,
        }


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
        # Cached per-row L2 norms for cosine search; invalidated on mutation.
        self._norms: np.ndarray | None = None
        self._lock = threading.RLock()
        # Monotonic write version vs. the version last persisted. A memory is
        # dirty when they differ. This (rather than a bare bool) means a write
        # landing during a save is never silently marked clean — see save().
        self._version = 0
        self._saved_version = 0

    # -- introspection ----------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def dirty(self) -> bool:
        return self._version != self._saved_version

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
        indices = list(indices)
        if not indices:
            raise MemoryError_("mask must name at least one known dimension.")
        mask = np.zeros(self.dimension, dtype=bool)
        for raw in indices:
            i = int(raw)
            if not (0 <= i < self.dimension):
                raise MemoryError_(f"mask index {raw} out of range [0, {self.dimension}).")
            if mask[i]:
                raise MemoryError_(f"duplicate mask index {raw}.")
            mask[i] = True
        return mask

    # -- writing (append a pattern) --------------------------------------
    def write(self, items: Iterable[dict[str, Any]]) -> list[str]:
        """Append patterns. Each item: ``{vector, id?, metadata?}``. Existing ids
        are overwritten. Returns the distinct affected ids (in first-seen order)."""

        with self._lock:
            affected: dict[str, None] = {}  # ordered set of distinct ids
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
                affected[_id] = None

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

            self._norms = None  # invalidate cached norms
            self._version += 1
            return list(affected)

    def get(self, _id: str) -> dict[str, Any]:
        with self._lock:
            if _id not in self._index:
                raise NotFoundError(f"Pattern {_id!r} not found in memory {self.name!r}.")
            idx = self._index[_id]
            return {
                "id": _id,
                "vector": self._X[idx].tolist(),
                "metadata": copy.deepcopy(self.metadata[idx]),
            }

    def delete(self, ids: Iterable[str]) -> int:
        with self._lock:
            # De-duplicate input ids so the returned count reflects distinct
            # patterns actually removed.
            targets = [i for i in dict.fromkeys(ids) if i in self._index]
            if not targets:
                return 0
            drop = np.array(sorted(self._index[i] for i in targets), dtype=np.int64)
            mask = np.ones(self._X.shape[0], dtype=bool)
            mask[drop] = False
            self._X = self._X[mask]
            self.ids = [i for j, i in enumerate(self.ids) if mask[j]]
            self.metadata = [m for j, m in enumerate(self.metadata) if mask[j]]
            self._index = {i: j for j, i in enumerate(self.ids)}
            self._norms = None  # invalidate cached norms
            self._version += 1
            return len(targets)

    # -- content-addressable operations ----------------------------------
    @staticmethod
    def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the ``k`` highest scores, descending, ties broken stably."""

        n = scores.shape[0]
        k = min(max(k, 0), n)
        if k == 0:
            return np.empty((0,), dtype=np.int64)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind="stable")]

    def _contributors(self, weights: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if weights.shape[0] == 0:
            return []
        top = self._top_k_indices(weights, top_k)
        return [
            {
                "id": self.ids[int(i)],
                "weight": float(weights[int(i)]),
                "metadata": copy.deepcopy(self.metadata[int(i)]),
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
            b = float(beta) if beta is not None else self.beta
            if self._X.shape[0] == 0:
                # Empty memory has nothing to recall — mirror search()'s empty
                # result rather than erroring (consistent 200 contract).
                return {
                    "reconstruction": None,
                    "weights": [],
                    "top": None,
                    "beta": b,
                    "steps": max(1, steps),
                }
            q = self._coerce_vector(query)
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
        metric: str = "cosine",
    ) -> list[dict[str, Any]]:
        """Nearest stored patterns by similarity (cosine by default)."""

        with self._lock:
            n = self._X.shape[0]
            if n == 0 or k <= 0:
                return []
            q = self._coerce_vector(query)
            if metric == "cosine":
                if self._norms is None:
                    self._norms = np.linalg.norm(self._X, axis=1).astype(np.float32)
                scores = compute_scores(self._X, q, metric, norms=self._norms)
            else:
                scores = compute_scores(self._X, q, metric)
            if flt:
                validate_filter(flt)
                keep = np.fromiter(
                    (_match_filter(self.metadata[i], flt) for i in range(n)), dtype=bool, count=n
                )
                if not keep.any():
                    return []
                scores = np.where(keep, scores, -np.inf)
            top = self._top_k_indices(scores, k)
            results: list[dict[str, Any]] = []
            for i in top:
                i = int(i)
                score = float(scores[i])
                if not np.isfinite(score):
                    continue
                row: dict[str, Any] = {
                    "id": self.ids[i],
                    "score": score,
                    "metadata": copy.deepcopy(self.metadata[i]),
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
            b = float(beta) if beta is not None else self.beta
            if self._X.shape[0] == 0:
                return {
                    "score": 0.0,
                    "reconstruction": None,
                    "residual": [],
                    "fields": [],
                    "nearest": None,
                    "beta": b,
                }
            q = self._coerce_vector(query)
            recon, weights = retrieve(self._X, q, b, None, 1)
            residual = np.abs(q - recon)
            score = float(np.linalg.norm(q - recon))
            order = np.argsort(-residual, kind="stable")
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
    def snapshot(self) -> MemorySnapshot:
        """An immutable, consistent copy for persistence (taken under the lock).

        The matrix and id/metadata lists are copied together while holding the
        memory lock, so they can never disagree even under a concurrent write.
        """

        with self._lock:
            return MemorySnapshot(
                name=self.name,
                dimension=self.dimension,
                beta=self.beta,
                fields=list(self.fields) if self.fields else None,
                ids=list(self.ids),
                metadata=[dict(m) for m in self.metadata],
                matrix=self._X.copy(),
                version=self._version,
            )

    def mark_saved(self, version: int) -> None:
        """Record that state up to ``version`` is durably persisted.

        Uses ``max`` so a write that bumped the version *after* the snapshot was
        taken leaves the memory dirty for the next flush — no lost update.
        """

        with self._lock:
            self._saved_version = max(self._saved_version, version)

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], matrix: np.ndarray) -> Memory:
        mem = cls(
            manifest["name"],
            manifest["dimension"],
            manifest.get("beta", 8.0),
            manifest.get("fields"),
        )
        ids = list(manifest["ids"])
        metadata = list(manifest["metadata"])
        matrix = np.asarray(matrix)
        # Reject a torn/mismatched file rather than silently misaligning rows.
        if matrix.ndim != 2 or matrix.shape[1] != mem.dimension:
            raise StoreError(
                f"memory {mem.name!r}: matrix shape {matrix.shape} is not "
                f"(*, {mem.dimension})."
            )
        if matrix.shape[0] != len(ids) or len(ids) != len(metadata):
            raise StoreError(
                f"memory {mem.name!r}: row/id/metadata length mismatch "
                f"(rows={matrix.shape[0]}, ids={len(ids)}, metadata={len(metadata)})."
            )
        mem.ids = ids
        mem.metadata = metadata
        mem._index = {i: j for j, i in enumerate(mem.ids)}
        mem._X = matrix.astype(np.float32, copy=False)
        # Loaded state mirrors disk, so it starts clean (version 0 == saved 0).
        return mem


class NeuroStore:
    """Owns every :class:`Memory` and persists them all to a single ``.npz`` file."""

    def __init__(self, data_file: str | Path, fail_on_corrupt_load: bool = False) -> None:
        self.data_file = Path(data_file)
        self.fail_on_corrupt_load = fail_on_corrupt_load
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
    def _tmp_path(self) -> Path:
        return self.data_file.with_suffix(self.data_file.suffix + ".tmp")

    def load(self) -> None:
        # Drop any temp file left behind by an interrupted save.
        tmp = self._tmp_path()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                logger.warning("could not remove stale temp file %s", tmp)
        if not self.data_file.exists():
            return
        try:
            self._load_from_file()
        except Exception as exc:
            if self.fail_on_corrupt_load:
                raise StoreError(
                    f"Failed to load data file {self.data_file}: {exc}"
                ) from exc
            self._quarantine_corrupt(exc)

    def _load_from_file(self) -> None:
        loaded: dict[str, Memory] = {}
        # Own the file handle so it is always closed before any quarantine
        # rename — otherwise np.load leaves it open on Windows when it raises.
        with open(self.data_file, "rb") as handle:
            with np.load(handle, allow_pickle=False) as data:
                if "__manifest__" not in data:
                    raise StoreError("data file is missing its __manifest__ entry.")
                manifest = json.loads(bytes(data["__manifest__"]).decode("utf-8"))
                version = manifest.get("version")
                if version is None or version > _MANIFEST_VERSION:
                    raise StoreError(f"unsupported manifest version {version!r}.")
                for entry in manifest.get("memories", []):
                    key = f"X@{entry['name']}"
                    if key not in data:
                        raise StoreError(f"missing matrix for memory {entry['name']!r}.")
                    mem = Memory.from_manifest(entry, data[key])
                    loaded[mem.name] = mem
        # Commit only after the whole file parses + validates (all-or-nothing).
        self._memories.update(loaded)

    def _quarantine_corrupt(self, exc: Exception) -> None:
        self._memories.clear()
        quarantine = self.data_file.with_name(
            f"{self.data_file.name}.corrupt-{uuid.uuid4().hex[:8]}"
        )
        try:
            os.replace(self.data_file, quarantine)
            logger.warning(
                "data file %s is corrupt (%s); quarantined to %s; starting empty.",
                self.data_file,
                exc,
                quarantine,
            )
        except OSError:
            logger.error(
                "data file %s is corrupt (%s) and could not be quarantined.",
                self.data_file,
                exc,
            )

    def save(self) -> None:
        with self._lock:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            mems = list(self._memories.values())
            # Snapshot every memory first (each under its own lock) so the
            # serialized matrix and ids/metadata are mutually consistent even
            # if a write lands while we are writing the file.
            snapshots = [mem.snapshot() for mem in mems]
            manifest = {
                "version": _MANIFEST_VERSION,
                "memories": [snap.manifest() for snap in snapshots],
            }
            arrays: dict[str, np.ndarray] = {
                "__manifest__": np.frombuffer(
                    json.dumps(manifest).encode("utf-8"), dtype=np.uint8
                )
            }
            for snap in snapshots:
                arrays[f"X@{snap.name}"] = snap.matrix
            tmp = self._tmp_path()
            with open(tmp, "wb") as handle:
                np.savez(handle, **arrays)
                # Force the bytes to disk before the atomic rename so a crash
                # can never leave a half-written file as the live data file.
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.data_file)
            self._fsync_dir()
            # Only clear dirtiness up to each snapshot's version; a newer write
            # keeps the memory dirty for the next flush.
            for mem, snap in zip(mems, snapshots, strict=True):
                mem.mark_saved(snap.version)

    def _fsync_dir(self) -> None:
        """fsync the parent directory so the rename itself is durable (POSIX).

        Windows has no directory-fsync; there the replace of an fsync'd temp
        file is the available atomicity guarantee.
        """

        if os.name == "nt":
            return
        try:
            dir_fd = os.open(self.data_file.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            logger.warning("could not fsync directory %s", self.data_file.parent)

    def flush(self) -> int:
        with self._lock:
            dirty = sum(1 for mem in self._memories.values() if mem.dirty)
            if dirty:
                self.save()
            return dirty

    def save_all(self) -> None:
        self.save()

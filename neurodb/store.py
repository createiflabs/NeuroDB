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

from .hopfield import retrieve, softmax
from .metrics import compute_scores

logger = logging.getLogger("neurodb.store")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANIFEST_VERSION = 1

# Per-memory normalization modes applied before the Hopfield attention step.
NORMALIZE_MODES = ("none", "zscore", "l2")
# Floor for per-dimension std / row norm so a degenerate (constant / zero)
# dimension or pattern never produces a division by zero (NaN/inf).
_NORM_EPS = 1e-6

# Capacity diagnostics: cap the pairwise/self-recall cost on large memories by
# sampling, and the self-recall margin / status thresholds.
_CAPACITY_SAMPLE = 256
_SELF_RECALL_THRESHOLD = 0.9  # top weight must land on self and exceed this
_CAPACITY_BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)  # coarse suggested-β search
# self-recall failure fraction → status
_CROWDED_FRACTION = 0.1
_SATURATED_FRACTION = 0.5


def resolve_normalize(normalize: str | None, fields: list[str] | None) -> str:
    """Resolve the effective normalization mode.

    ``None`` selects the structured-record-friendly default: ``"zscore"`` when
    per-dimension ``fields`` are given (a strong signal the rows are real-world
    records on disparate scales), otherwise ``"none"`` (raw dot product, the
    pre-patch behaviour — e.g. for already unit-norm embeddings).
    """

    if normalize is None:
        return "zscore" if fields else "none"
    if normalize not in NORMALIZE_MODES:
        raise MemoryError_(
            f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}."
        )
    return normalize


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
    normalize: str
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
            # Additive, optional key: legacy files omit it and load as "none".
            # mean/std/Z are NOT persisted — they are recomputed from the matrix
            # on load, so the file can never carry stale statistics.
            "normalize": self.normalize,
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
        normalize: str | None = None,
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
        # How patterns are scaled before the Hopfield step (see module-level
        # NORMALIZE_MODES). Defaults to "zscore" for structured records (fields
        # given), "none" otherwise; legacy loads resolve to "none".
        self.normalize = resolve_normalize(normalize, self.fields)
        self.ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._X = np.zeros((0, self.dimension), dtype=np.float32)
        # Cached per-row L2 norms for cosine search; invalidated on mutation.
        self._norms: np.ndarray | None = None
        # Normalization caches, recomputed from self._X on demand and
        # invalidated alongside self._norms on every mutation. _Z is the
        # normalized matrix the Hopfield step runs on; _mean/_std are the
        # per-dimension zscore statistics used to (de)normalize queries.
        self._Z: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        # Cached capacity/saturation diagnostic (lazy; invalidated on mutation).
        self._capacity: dict[str, Any] | None = None
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
            "normalize": self.normalize,
        }

    def stats(self) -> dict[str, Any]:
        """:meth:`info` plus, for ``zscore`` memories, the per-dimension
        ``mean``/``std`` used to normalize — useful for debugging why a given
        field was (or wasn't) flagged as anomalous."""

        with self._lock:
            detail = self.info()
            if self.normalize == "zscore":
                self._ensure_stats()
                detail["mean"] = self._mean.tolist() if self._mean is not None else None
                detail["std"] = self._std.tolist() if self._std is not None else None
            detail["capacity"] = self.capacity_compact()
            return detail

    # -- normalization ----------------------------------------------------
    def _ensure_stats(self) -> None:
        """Compute and cache the normalized matrix ``_Z`` (and, for zscore, the
        ``_mean``/``_std`` statistics) from the current matrix. Must be called
        under ``self._lock``. A no-op for ``"none"`` mode (the Hopfield step runs
        on the raw matrix, exactly as before this patch)."""

        if self.normalize == "none" or self._Z is not None:
            return
        X = self._X
        if self.normalize == "zscore":
            if X.shape[0] < 2:
                # std is undefined with <2 patterns → identity transform, so
                # the memory behaves like "none" until there is data to estimate
                # the statistics from.
                self._mean = np.zeros(self.dimension, dtype=np.float32)
                self._std = np.ones(self.dimension, dtype=np.float32)
            else:
                self._mean = X.mean(axis=0).astype(np.float32)
                # Floor the std so a constant dimension (zero variance) maps to 0
                # in normalized space rather than producing NaN/inf: it then
                # contributes nothing to similarity, which is correct.
                self._std = np.maximum(X.std(axis=0), _NORM_EPS).astype(np.float32)
            self._Z = ((X - self._mean) / self._std).astype(np.float32)
        else:  # "l2": project every row onto the unit sphere
            norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _NORM_EPS)
            self._Z = (X / norms).astype(np.float32)

    def _matrix_Z(self) -> np.ndarray:
        """The (cached) matrix the Hopfield step runs on for the active mode.
        Must be called under ``self._lock``. For ``"none"`` this is ``self._X``
        unchanged, so that path is bit-for-bit identical."""

        if self.normalize == "none":
            return self._X
        self._ensure_stats()
        return self._Z

    def _normalize_query(self, q: np.ndarray) -> np.ndarray:
        """Normalize a 1-D query ``(D,)`` or a 2-D batch ``(M, D)`` into the
        active space. Must be called under ``self._lock``."""

        if self.normalize == "none":
            return q
        self._ensure_stats()
        if self.normalize == "zscore":
            # Broadcasts over both the 1-D and the (M, D) batch case.
            return ((q - self._mean) / self._std).astype(np.float32)
        # "l2": project each row (or the single vector) onto the unit sphere;
        # a zero vector stays zero.
        if q.ndim == 1:
            qn = float(np.linalg.norm(q))
            return (q / qn).astype(np.float32) if qn > _NORM_EPS else q.astype(np.float32)
        norms = np.maximum(np.linalg.norm(q, axis=1, keepdims=True), _NORM_EPS)
        return (q / norms).astype(np.float32)

    def _normalized(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: ``(matrix_Z, normalized_query)`` for the active mode."""

        return self._matrix_Z(), self._normalize_query(q)

    def _denormalize(self, recon_z: np.ndarray) -> np.ndarray:
        """Map a reconstruction from normalized space back to raw units. For
        ``zscore`` this inverts the standardization; for ``none`` and ``l2`` the
        reconstruction is already returned as-is (an l2 recon is directional and
        lives on the unit sphere, *not* in the original magnitude units)."""

        if self.normalize == "zscore":
            return (recon_z * self._std + self._mean).astype(np.float32)
        return recon_z

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

    def _coerce_batch(self, queries: Iterable[Iterable[float]]) -> np.ndarray:
        """Validate a batch of query vectors into an ``(M, D)`` float32 array."""

        try:
            arr = np.asarray(queries, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise MemoryError_(f"Invalid batch (ragged or non-numeric): {exc}") from exc
        if arr.size == 0:  # empty batch (e.g. [] or shape (0, D))
            return np.zeros((0, self.dimension), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise MemoryError_(
                f"Batch must be a 2-D array of {self.dimension}-d vectors; "
                f"got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise MemoryError_("Batch contains NaN or infinite values.")
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

    def _invalidate_caches(self) -> None:
        """Drop every matrix-derived cache after a mutation. Must be called
        under ``self._lock``; the same invalidation point for all of them keeps
        the norms and the normalization statistics consistent with ``self._X``."""

        self._norms = None
        self._Z = None
        self._mean = None
        self._std = None
        self._capacity = None

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

            self._invalidate_caches()
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
            self._invalidate_caches()
            self._version += 1
            return len(targets)

    # -- updating an existing pattern ------------------------------------
    def _resolve_field(self, field: int | str) -> int:
        """Map a field index or (when ``fields`` are named) a field name to an
        in-range dimension index."""

        if isinstance(field, str):
            if not self.fields or field not in self.fields:
                raise MemoryError_(f"unknown field name {field!r}.")
            i = self.fields.index(field)
        else:
            i = int(field)
        if not (0 <= i < self.dimension):
            raise MemoryError_(f"field index {i} out of range [0, {self.dimension}).")
        return i

    def update(
        self,
        _id: str,
        vector: Iterable[float] | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> dict[str, Any]:
        """Update a stored pattern in place. ``vector`` replaces the row;
        ``metadata`` shallow-merges (or replaces when ``merge_metadata`` is
        False). Returns the updated pattern (same shape as :meth:`get`)."""

        with self._lock:
            if _id not in self._index:
                raise NotFoundError(f"Pattern {_id!r} not found in memory {self.name!r}.")
            idx = self._index[_id]
            changed = False
            if vector is not None:
                self._X[idx] = self._coerce_vector(vector)
                self._invalidate_caches()  # norms/stats/Z depend on the matrix
                changed = True
            if metadata is not None:
                if merge_metadata:
                    merged = dict(self.metadata[idx])
                    merged.update(metadata)
                    self.metadata[idx] = merged
                else:
                    self.metadata[idx] = dict(metadata)
                changed = True
            if changed:
                self._version += 1
            return {
                "id": _id,
                "vector": self._X[idx].tolist(),
                "metadata": copy.deepcopy(self.metadata[idx]),
            }

    def update_field(self, _id: str, field: int | str, value: float) -> dict[str, Any]:
        """Edit a single field of a structured record (by index or field name)."""

        with self._lock:
            if _id not in self._index:
                raise NotFoundError(f"Pattern {_id!r} not found in memory {self.name!r}.")
            fi = self._resolve_field(field)
            v = float(value)
            if not np.isfinite(v):
                raise MemoryError_("field value must be finite.")
            idx = self._index[_id]
            self._X[idx, fi] = v
            self._invalidate_caches()
            self._version += 1
            return {
                "id": _id,
                "vector": self._X[idx].tolist(),
                "metadata": copy.deepcopy(self.metadata[idx]),
            }

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

    def _contributors(
        self,
        weights: np.ndarray,
        top_k: int,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        # ``ids``/``metadata`` default to the whole memory, but a filtered call
        # passes the candidate-aligned slices so weights map to the right rows.
        if weights.shape[0] == 0:
            return []
        ids = self.ids if ids is None else ids
        metadata = self.metadata if metadata is None else metadata
        top = self._top_k_indices(weights, top_k)
        return [
            {
                "id": ids[int(i)],
                "weight": float(weights[int(i)]),
                "metadata": copy.deepcopy(metadata[int(i)]),
            }
            for i in top
        ]

    # -- shared result builders / candidate selection --------------------
    def _empty_complete(self, b: float, steps: int) -> dict[str, Any]:
        # Empty memory (or a filter matching nothing) has nothing to recall —
        # mirror search()'s empty result rather than erroring (200 contract).
        return {
            "reconstruction": None,
            "weights": [],
            "top": None,
            "beta": b,
            "steps": max(1, steps),
        }

    def _empty_anomaly(self, b: float) -> dict[str, Any]:
        return {
            "score": 0.0,
            "z_score": 0.0,
            "reconstruction": None,
            "residual": [],
            "fields": [],
            "nearest": None,
            "beta": b,
        }

    def _candidates(
        self, flt: dict[str, Any] | None
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]] | None:
        """Normalized candidate matrix + aligned ids/metadata for the metadata
        filter (same semantics as :meth:`search`). Returns ``None`` when the
        filter excludes every pattern. Must be called under ``self._lock``.

        zscore/l2 statistics are always computed over the **full** memory (stable
        and already cached); only the candidate *rows* are filtered out — so a
        record is scored against same-type patterns but normalized by the global
        distribution.
        """

        Z = self._matrix_Z()
        if not flt:
            return Z, self.ids, self.metadata
        validate_filter(flt)
        keep = [i for i in range(self._X.shape[0]) if _match_filter(self.metadata[i], flt)]
        if not keep:
            return None
        idx = np.asarray(keep, dtype=np.int64)
        return Z[idx], [self.ids[i] for i in keep], [self.metadata[i] for i in keep]

    def complete(
        self,
        query: Iterable[float],
        beta: float | None = None,
        mask: Iterable[int] | None = None,
        steps: int = 1,
        top_k: int = 5,
        flt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Content-addressable recall / pattern completion via one (or more)
        Hopfield attention steps. ``flt`` restricts recall to patterns whose
        metadata matches (same filter syntax as :meth:`search`)."""

        with self._lock:
            b = float(beta) if beta is not None else self.beta
            if self._X.shape[0] == 0:
                return self._empty_complete(b, steps)
            q = self._coerce_vector(query)
            mask_arr = self._mask_from_indices(mask)
            cand = self._candidates(flt)
            if cand is None:
                return self._empty_complete(b, steps)
            Z, ids, meta = cand
            # Run the Hopfield step in normalized space, then map the result
            # back to raw units. The mask clamp inside retrieve() therefore
            # clamps in normalized space (qz[mask]); de-normalizing afterwards
            # restores the known fields to their exact raw values.
            qz = self._normalize_query(q)
            recon_z, weights = retrieve(Z, qz, b, mask_arr, steps)
            recon = self._denormalize(recon_z)
            contributors = self._contributors(weights, top_k, ids, meta)
            return {
                "reconstruction": recon.tolist(),
                "weights": contributors,
                "top": contributors[0] if contributors else None,
                "beta": b,
                "steps": max(1, steps),
            }

    def complete_batch(
        self,
        queries: Iterable[Iterable[float]],
        beta: float | None = None,
        mask: Iterable[int] | None = None,
        steps: int = 1,
        top_k: int = 5,
        flt: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vectorized :meth:`complete` over a batch of queries. The whole batch
        is one matmul under a single lock — no per-query Python loop over the
        attention step. Returns one result per input row, in order."""

        with self._lock:
            b = float(beta) if beta is not None else self.beta
            Q = self._coerce_batch(queries)
            m = Q.shape[0]
            if m == 0:
                return []
            mask_arr = self._mask_from_indices(mask)
            cand = self._candidates(flt) if self._X.shape[0] else None
            if cand is None:
                return [self._empty_complete(b, steps) for _ in range(m)]
            Z, ids, meta = cand
            Qz = self._normalize_query(Q)
            recon_z, weights = retrieve(Z, Qz, b, mask_arr, steps)
            recon = self._denormalize(recon_z)
            results: list[dict[str, Any]] = []
            for i in range(m):
                contributors = self._contributors(weights[i], top_k, ids, meta)
                results.append(
                    {
                        "reconstruction": recon[i].tolist(),
                        "weights": contributors,
                        "top": contributors[0] if contributors else None,
                        "beta": b,
                        "steps": max(1, steps),
                    }
                )
            return results

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
            # search deliberately ranks on the RAW vectors (cosine already
            # normalizes magnitude away), independent of self.normalize. This
            # keeps existing search results stable and avoids double-normalizing
            # a zscore/l2 memory — the normalize mode only affects the Hopfield
            # complete/anomaly path, not similarity search.
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

    def _anomaly_result(
        self,
        q: np.ndarray,
        recon: np.ndarray,
        qz: np.ndarray,
        recon_z: np.ndarray,
        weights: np.ndarray,
        ids: list[str],
        meta: list[dict[str, Any]],
        b: float,
        top_k: int,
    ) -> dict[str, Any]:
        """Assemble one anomaly report from a (single or per-row) recall result.

        Shared by :meth:`anomaly` and :meth:`anomaly_batch` so a batch row is
        element-for-element identical to the single-query call.
        """

        # Raw residual (original contract) and its normalized counterpart.
        # z_residual is "how many std-devs off" per field, comparable across
        # fields of different scale — the meaningful anomaly signal under
        # zscore. For "none"/"l2" qz==q and recon_z==recon (l2 on the unit
        # sphere), so the z-* figures coincide with the raw ones.
        residual = np.abs(q - recon)
        z_residual = np.abs(qz - recon_z)
        score = float(np.linalg.norm(q - recon))
        z_score = float(np.linalg.norm(qz - recon_z))
        # Rank by the standardized deviation when zscore (cross-field
        # comparable), otherwise by the raw deviation as before.
        ranking = z_residual if self.normalize == "zscore" else residual
        order = np.argsort(-ranking, kind="stable")
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
                    "z_deviation": float(z_residual[idx]),
                }
            )
        nearest = self._contributors(weights, 1, ids, meta)
        return {
            "score": score,
            "z_score": z_score,
            "reconstruction": recon.tolist(),
            "residual": residual.tolist(),
            "fields": fields,
            "nearest": nearest[0] if nearest else None,
            "beta": b,
        }

    def anomaly(
        self,
        query: Iterable[float],
        beta: float | None = None,
        top_k: int = 5,
        flt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Per-field anomaly detection.

        Recall the pattern the input most resembles, then report where the input
        deviates from that reconstruction field-by-field. Fields with the largest
        absolute deviation are the most anomalous. ``flt`` restricts recall to
        patterns whose metadata matches (same syntax as :meth:`search`) — e.g.
        score a record only against others of the same type.
        """

        with self._lock:
            b = float(beta) if beta is not None else self.beta
            if self._X.shape[0] == 0:
                return self._empty_anomaly(b)
            q = self._coerce_vector(query)
            cand = self._candidates(flt)
            if cand is None:
                return self._empty_anomaly(b)
            Z, ids, meta = cand
            qz = self._normalize_query(q)
            recon_z, weights = retrieve(Z, qz, b, None, 1)
            recon = self._denormalize(recon_z)
            return self._anomaly_result(q, recon, qz, recon_z, weights, ids, meta, b, top_k)

    def anomaly_batch(
        self,
        queries: Iterable[Iterable[float]],
        beta: float | None = None,
        top_k: int = 5,
        flt: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vectorized :meth:`anomaly` over a batch of queries — one matmul under
        a single lock. Returns one report per input row, in order."""

        with self._lock:
            b = float(beta) if beta is not None else self.beta
            Q = self._coerce_batch(queries)
            m = Q.shape[0]
            if m == 0:
                return []
            cand = self._candidates(flt) if self._X.shape[0] else None
            if cand is None:
                return [self._empty_anomaly(b) for _ in range(m)]
            Z, ids, meta = cand
            Qz = self._normalize_query(Q)
            recon_z, weights = retrieve(Z, Qz, b, None, 1)
            recon = self._denormalize(recon_z)
            return [
                self._anomaly_result(
                    Q[i], recon[i], Qz[i], recon_z[i], weights[i], ids, meta, b, top_k
                )
                for i in range(m)
            ]

    # -- capacity / saturation diagnostics -------------------------------
    def capacity_report(self) -> dict[str, Any]:
        """Hopfield storage-capacity / saturation diagnostic.

        Modern Hopfield memories have finite capacity: past it, distinct
        patterns' attractors merge and recall *silently* returns blends. This
        reports honest signals of that:

        * ``mean/max_pairwise_similarity`` — cosine similarity between (normalized)
          stored patterns; a high max means those two will be confused at any β.
        * ``self_recall_fail_fraction`` — fraction of sampled patterns that, when
          queried with their *own* stored vector at the memory's β, do **not**
          retrieve themselves with top weight ≥ 0.9. If a pattern can't recall
          itself, the memory is over capacity for its β.
        * ``suggested_beta`` — smallest β (coarse, heuristic search) at which the
          sample self-recalls cleanly, or ``None`` if none of the tried β reach it.
        * ``status`` — ``healthy`` (< 10% fail) / ``crowded`` (< 50%) / ``saturated``.

        On large memories the pairwise and self-recall costs are bounded by
        sampling ``_CAPACITY_SAMPLE`` patterns (deterministic seed). The result
        is cached until the next mutation.
        """

        with self._lock:
            if self._capacity is not None:
                return copy.deepcopy(self._capacity)
            n = self._X.shape[0]
            report: dict[str, Any] = {
                "count": n,
                "dimension": self.dimension,
                "beta": self.beta,
                "normalize": self.normalize,
                "sampled": min(n, _CAPACITY_SAMPLE),
            }
            if n == 0:
                report.update(
                    {
                        "mean_pairwise_similarity": None,
                        "max_pairwise_similarity": None,
                        "self_recall_fail_fraction": 0.0,
                        "suggested_beta": None,
                        "status": "healthy",
                    }
                )
                self._capacity = report
                return copy.deepcopy(report)

            Z = self._matrix_Z()
            # Deterministic sample so the diagnostic is reproducible.
            if n > _CAPACITY_SAMPLE:
                rng = np.random.default_rng(0)
                sample = np.sort(rng.choice(n, size=_CAPACITY_SAMPLE, replace=False))
            else:
                sample = np.arange(n)

            # Pairwise cosine similarity among sampled (normalized) patterns.
            rows = Z[sample].astype(np.float64)
            unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), _NORM_EPS)
            sim = unit @ unit.T
            k = sim.shape[0]
            if k > 1:
                off = sim[np.triu_indices(k, k=1)]
                report["mean_pairwise_similarity"] = float(off.mean())
                report["max_pairwise_similarity"] = float(off.max())
            else:
                report["mean_pairwise_similarity"] = 0.0
                report["max_pairwise_similarity"] = 0.0

            # Self-recall: query the FULL memory with each sampled pattern's own
            # normalized vector; pre-softmax sims are reused across betas.
            Qz = self._normalize_query(self._X[sample])
            sims = (Qz @ Z.T).astype(np.float64)  # (sample, N)
            rowidx = np.arange(sample.shape[0])

            def fail_fraction(beta: float) -> float:
                w = softmax(beta * sims, axis=1)
                top_idx = np.argmax(w, axis=1)
                top_w = w[rowidx, top_idx]
                hits = (top_idx == sample) & (top_w >= _SELF_RECALL_THRESHOLD)
                return float(1.0 - hits.mean())

            fail = fail_fraction(self.beta)
            report["self_recall_fail_fraction"] = fail
            suggested = next(
                (b for b in _CAPACITY_BETAS if fail_fraction(b) <= _CROWDED_FRACTION), None
            )
            report["suggested_beta"] = suggested
            if fail >= _SATURATED_FRACTION:
                report["status"] = "saturated"
            elif fail >= _CROWDED_FRACTION:
                report["status"] = "crowded"
            else:
                report["status"] = "healthy"

            self._capacity = report
            return copy.deepcopy(report)

    def capacity_compact(self) -> dict[str, Any]:
        """The headline capacity fields for /stats and /health (cached)."""

        full = self.capacity_report()
        return {
            "status": full["status"],
            "self_recall_fail_fraction": full["self_recall_fail_fraction"],
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
                normalize=self.normalize,
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
            # Legacy files predate this key → "none", preserving exact prior
            # behaviour. Pass it explicitly so the fields-based default-selection
            # rule never silently upgrades a loaded raw-vector memory to zscore.
            manifest.get("normalize", "none"),
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
        # Readiness signal: did the most recent persist succeed?
        self.last_save_ok = True
        self.load()

    # -- memory lifecycle -------------------------------------------------
    def create_memory(
        self,
        name: str,
        dimension: int,
        beta: float = 8.0,
        fields: list[str] | None = None,
        normalize: str | None = None,
    ) -> Memory:
        with self._lock:
            validate_name(name)
            if name in self._memories:
                raise MemoryError_(f"Memory {name!r} already exists.")
            mem = Memory(name, dimension, beta, fields, normalize)
            self._memories[name] = mem
            self.save()
            return mem

    def get_memory(self, name: str) -> Memory:
        try:
            return self._memories[name]
        except KeyError:
            raise NotFoundError(f"Memory {name!r} not found.") from None

    def list_memories(
        self, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self._lock:
            infos = [mem.info() for mem in self._memories.values()]
            if limit is None:
                return infos
            return infos[offset : offset + limit]

    def count_memories(self) -> int:
        with self._lock:
            return len(self._memories)

    def counts(self) -> tuple[int, int]:
        """Cheap ``(memories, patterns)`` totals without the capacity diagnostic
        (used by hot endpoints like /metrics)."""

        with self._lock:
            return len(self._memories), sum(m.count for m in self._memories.values())

    def delete_memory(self, name: str) -> None:
        with self._lock:
            if name not in self._memories:
                raise NotFoundError(f"Memory {name!r} not found.")
            del self._memories[name]
            self.save()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            # Per-memory detail carries the normalize mode and, for zscore, the
            # per-dimension mean/std (handy for debugging an anomaly flag).
            memories = [mem.stats() for mem in self._memories.values()]
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
        try:
            self._save_locked()
        except Exception:
            self.last_save_ok = False
            raise
        self.last_save_ok = True

    def _save_locked(self) -> None:
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
                np.savez(handle, **arrays)  # type: ignore[arg-type]
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

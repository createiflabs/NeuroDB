"""Correlated synthetic population generation + realism diagnostics.

A :class:`CriteriaSpec` describes the fields (valid ranges/categories + target
marginals), an optional cross-field correlation matrix, and hard cross-field
constraints. :func:`generate` draws a *correlated* population (not independent
draws — the classic "too clean" failure), enforces the hard constraints by
rejection, and returns the engine's float32 matrix. :func:`diagnose` /
:func:`is_too_clean` measure whether the result is realistically spread, and
:func:`calibrate` fits marginals/correlations from a sample **without retaining
the sample's records**.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FieldSpec:
    """One field's valid domain and target marginal.

    Numeric: ``low``/``high`` bounds and target ``mean``/``std``. Categorical:
    ``categories`` (encoded as their index) with optional ``probs``.
    """

    name: str
    kind: str = "numeric"  # "numeric" | "categorical"
    low: float = 0.0
    high: float = 1.0
    mean: float = 0.0
    std: float = 1.0
    categories: list[Any] = field(default_factory=list)
    probs: list[float] | None = None


@dataclass
class CriteriaSpec:
    """A criteria specification: fields + correlations + hard constraints."""

    fields: list[FieldSpec]
    # Correlation matrix over the numeric fields, in their order of appearance.
    correlation: np.ndarray | None = None
    # Cross-field hard constraints: predicate(record_dict) -> bool (True = valid).
    constraints: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)

    @property
    def numeric_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.kind == "numeric"]


def generate(spec: CriteriaSpec, n: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Generate ``n`` records satisfying ``spec``. Returns ``(matrix, field_names)``.

    Numeric fields are drawn jointly from a multivariate normal with the target
    means/stds and (optional) correlation, then clipped to bounds; categorical
    fields are drawn from their category distribution. Rows failing any hard
    constraint are rejected and resampled.
    """

    rng = np.random.default_rng(seed)
    numeric = spec.numeric_fields
    means = np.array([f.mean for f in numeric], dtype=float)
    stds = np.array([f.std for f in numeric], dtype=float)
    corr = (
        np.asarray(spec.correlation, dtype=float)
        if spec.correlation is not None
        else np.eye(len(numeric))
    )
    cov = np.outer(stds, stds) * corr if len(numeric) else np.empty((0, 0))

    name_to_idx = {f.name: i for i, f in enumerate(spec.fields)}
    numeric_cols = [name_to_idx[f.name] for f in numeric]

    collected = np.empty((0, len(spec.fields)), dtype=np.float32)
    for _ in range(200):  # rejection-sampling rounds
        if collected.shape[0] >= n:
            break
        draw = max((n - collected.shape[0]) * 2, 16)
        block = np.zeros((draw, len(spec.fields)), dtype=np.float64)
        if numeric:
            num = rng.multivariate_normal(means, cov, size=draw)
            for j, f in enumerate(numeric):
                num[:, j] = np.clip(num[:, j], f.low, f.high)
            block[:, numeric_cols] = num
        for f in spec.fields:
            if f.kind == "categorical" and f.categories:
                k = len(f.categories)
                idx = rng.choice(k, size=draw, p=f.probs)
                block[:, name_to_idx[f.name]] = idx
        block = _filter_constraints(block, spec)
        collected = np.vstack([collected, block.astype(np.float32)])

    if collected.shape[0] < n:
        raise ValueError(
            f"could not generate {n} records satisfying the constraints "
            f"(got {collected.shape[0]}); loosen constraints or marginals."
        )
    return collected[:n], [f.name for f in spec.fields]


def _filter_constraints(block: np.ndarray, spec: CriteriaSpec) -> np.ndarray:
    if not spec.constraints:
        return block
    names = [f.name for f in spec.fields]
    keep = []
    for row in block:
        record = dict(zip(names, row.tolist(), strict=True))
        if all(predicate(record) for predicate in spec.constraints):
            keep.append(row)
    return np.array(keep, dtype=block.dtype) if keep else np.empty((0, block.shape[1]))


def diagnose(matrix: np.ndarray, target_correlation: np.ndarray | None = None) -> dict[str, Any]:
    """Realism diagnostics: per-field variance, realized correlations, a capacity
    self-recall check, and a ``degenerate`` verdict (see :func:`is_too_clean`)."""

    matrix = np.asarray(matrix, dtype=np.float64)
    variances = matrix.var(axis=0)
    realized_corr = np.corrcoef(matrix, rowvar=False) if matrix.shape[1] > 1 else np.array([[1.0]])

    # Capacity / self-recall on a zscore memory built from the population.
    from ..store import Memory

    mem = Memory("diag", matrix.shape[1], 8.0, None, "zscore")
    mem.write([{"vector": row.tolist()} for row in matrix.astype(np.float32)])
    capacity = mem.capacity_compact()

    return {
        "n": int(matrix.shape[0]),
        "variances": variances.tolist(),
        "min_variance": float(variances.min()),
        "realized_correlation": np.nan_to_num(realized_corr).tolist(),
        "capacity": capacity,
        "degenerate": is_too_clean(matrix, target_correlation),
    }


def is_too_clean(
    matrix: np.ndarray,
    target_correlation: np.ndarray | None = None,
    *,
    var_eps: float = 1e-6,
    corr_tol: float = 0.1,
    target_min: float = 0.3,
) -> bool:
    """True if the population is unrealistically clean: a near-constant field, or
    target correlations that the population failed to reproduce (the independent-
    draw failure that makes valid real records look anomalous)."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] > 1:
        if matrix.var(axis=0).min() < var_eps:
            return True
        if target_correlation is not None:
            realized = np.nan_to_num(np.corrcoef(matrix, rowvar=False))
            target = np.asarray(target_correlation, dtype=float)
            off = ~np.eye(target.shape[0], dtype=bool)
            # A target correlation present (|t|>=target_min) but realized near 0.
            structured = off & (np.abs(target) >= target_min)
            if structured.any() and np.abs(realized[structured]).max() < corr_tol:
                return True
    elif matrix.var(axis=0).min() < var_eps:
        return True
    return False


def calibrate(sample: np.ndarray) -> dict[str, Any]:
    """Fit target marginals + correlation from a (possibly anonymized) sample and
    return **parameters only** — the source records are never retained or shipped.
    This is how v1 stays synthetic-first yet realistic: derive the parameters,
    generate the records."""

    sample = np.asarray(sample, dtype=np.float64)
    corr = np.corrcoef(sample, rowvar=False) if sample.shape[1] > 1 else np.array([[1.0]])
    return {
        "means": sample.mean(axis=0).tolist(),
        "stds": sample.std(axis=0).tolist(),
        "correlation": np.nan_to_num(corr).tolist(),
        "n_samples": int(sample.shape[0]),
    }

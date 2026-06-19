"""Benchmark datasets for the Tier 0 completion gate.

Two datasets, both standardized per feature so per-field RMSE is comparable
across fields and in interpretable (standard-deviation) units:

* :func:`make_multimodal` — a deterministic Gaussian mixture where a partial
  record's correct completion is *one specific cluster*, never the global mean.
  This isolates the snap-to-attractor effect NeuroDB claims and that averaging
  imputers (IterativeImputer especially) miss. Deterministic → drives CI.
* :func:`load_wine` — UCI Wine (3 cultivars, 13 numeric features) via sklearn:
  a real, all-numeric, genuinely multimodal dataset (no categorical-encoding
  confound). Ships offline with scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """A standardized benchmark dataset plus its ground-truth modality labels."""

    name: str
    X: np.ndarray  # (n, d) float64, z-scored per feature
    labels: np.ndarray  # (n,) int — ground-truth cluster/class (the "modality")
    feature_names: list[str]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.size else 0


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column; zero-variance columns are left centered (scale 1)."""

    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def make_multimodal(
    n_clusters: int = 6,
    dim: int = 12,
    per_cluster: int = 200,
    separation: float = 6.0,
    within_std: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """A seeded Gaussian mixture of ``n_clusters`` well-separated blobs.

    Cluster centres are drawn ``separation`` apart (in raw units) and each blob
    has ``within_std`` spread, so after standardization the variance is
    dominated by *between*-cluster structure: the global mean is a poor
    completion for any partial query, while the correct answer is the centre of
    whichever cluster the observed fields identify. This is the regime the
    differentiation thesis claims NeuroDB wins in — built fairly, not rigged.
    """

    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(n_clusters, dim))
    blocks = []
    labels = []
    for c in range(n_clusters):
        pts = centers[c] + rng.normal(scale=within_std, size=(per_cluster, dim))
        blocks.append(pts)
        labels.append(np.full(per_cluster, c, dtype=np.int64))
    X = _standardize(np.vstack(blocks))
    names = [f"f{i}" for i in range(dim)]
    return Dataset("synthetic-multimodal", X, np.concatenate(labels), names)


def load_wine() -> Dataset:
    """UCI Wine: 178 rows, 13 numeric features, 3 cultivars (z-scored)."""

    from sklearn.datasets import load_wine as _load_wine

    bunch = _load_wine()
    X = _standardize(np.asarray(bunch.data, dtype=np.float64))
    y = np.asarray(bunch.target, dtype=np.int64)
    return Dataset("wine", X, y, list(bunch.feature_names))

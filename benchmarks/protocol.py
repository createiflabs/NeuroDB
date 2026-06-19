"""The completion benchmark protocol: a *fair* head-to-head.

For each method we measure per-field imputation error on held-out records whose
fields have been partially masked (MCAR). The comparison is made fair by:

* **Equal tuning budget** — every method picks its hyperparameters on a
  validation split (carved from train), never on the test set. NeuroDB tunes
  ``beta``/``steps``; KNNImputer tunes ``n_neighbors``; IterativeImputer tunes
  ``max_iter``. Grids are recorded so the budget is auditable.
* **Same data** — baselines ``.fit`` on the complete train matrix; NeuroDB
  stores the same train rows as patterns. Final evaluation refits/restores on
  the full train split and scores the test split.
* **Masked-only scoring** — error is computed only on the hidden cells (what
  each method actually had to predict), in standardized (per-feature std) units,
  accumulated in float64 so BLAS thread count cannot flip the result.

``run_benchmark`` returns a :class:`BenchmarkReport`; it does not assert
anything — the gate test and the report writer interpret the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurodb.store import Memory

from .datasets import Dataset

# Per-method hyperparameter grids (the "equal tuning budget"). Each method tunes
# its own natural knobs over a comparable range; the best validation config is
# used for the final test evaluation.
GRIDS: dict[str, list[dict]] = {
    "neurodb": [
        {"beta": b, "steps": s}
        for b in (4.0, 8.0, 16.0, 32.0)
        for s in (1, 2, 3)
    ],
    "knn": [{"n_neighbors": k} for k in (1, 3, 5, 7, 15, 31)],
    "iterative": [{"max_iter": m} for m in (10, 25, 50)],
}


@dataclass(frozen=True)
class MethodReport:
    method: str
    config: dict
    overall_rmse: float
    per_field_rmse: list[float]


@dataclass(frozen=True)
class BenchmarkReport:
    dataset: str
    n_train: int
    n_test: int
    d: int
    n_modes: int
    mask_fraction: float
    feature_names: list[str]
    methods: tuple[MethodReport, ...]

    def by_name(self, name: str) -> MethodReport:
        for m in self.methods:
            if m.method == name:
                return m
        raise KeyError(name)


# --- masking / scoring -----------------------------------------------------
def _stratified_split(
    labels: np.ndarray, frac_b: int | float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into (A, B) with ~``frac_b`` of each class in B."""

    idx_a: list[int] = []
    idx_b: list[int] = []
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        rng.shuffle(ids)
        cut = int(round(len(ids) * float(frac_b)))
        idx_b.extend(ids[:cut].tolist())
        idx_a.extend(ids[cut:].tolist())
    return np.array(sorted(idx_a), dtype=np.int64), np.array(sorted(idx_b), dtype=np.int64)


def _random_known_mask(
    n: int, d: int, mask_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean (n, d): True = observed. Always leaves ≥1 known and ≥1 hidden."""

    n_hidden = max(1, min(d - 1, int(round(mask_fraction * d))))
    known = np.ones((n, d), dtype=bool)
    for i in range(n):
        hidden = rng.choice(d, size=n_hidden, replace=False)
        known[i, hidden] = False
    return known


def _apply_mask(X: np.ndarray, known: np.ndarray) -> np.ndarray:
    masked = np.asarray(X, dtype=np.float64).copy()
    masked[~known] = np.nan
    return masked


def _score(
    pred: np.ndarray, truth: np.ndarray, known: np.ndarray
) -> tuple[float, list[float]]:
    """Per-field and overall RMSE over the hidden cells only (float64)."""

    unknown = ~known
    diff = pred.astype(np.float64) - truth.astype(np.float64)
    sq = np.where(unknown, diff * diff, 0.0)
    counts = unknown.sum(axis=0)
    per_field: list[float] = []
    for j in range(truth.shape[1]):
        c = int(counts[j])
        per_field.append(float(np.sqrt(sq[:, j].sum() / c)) if c else float("nan"))
    total = int(unknown.sum())
    overall = float(np.sqrt(sq.sum() / total)) if total else float("nan")
    return overall, per_field


# --- the three methods (uniform signature: fit on train, impute masked) -----
def _impute_knn(X_train: np.ndarray, X_masked: np.ndarray, n_neighbors: int) -> np.ndarray:
    from sklearn.impute import KNNImputer

    imp = KNNImputer(n_neighbors=n_neighbors)
    imp.fit(X_train)
    return np.asarray(imp.transform(X_masked), dtype=np.float64)


def _impute_iterative(X_train: np.ndarray, X_masked: np.ndarray, max_iter: int) -> np.ndarray:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imp = IterativeImputer(max_iter=max_iter, random_state=0)
    imp.fit(X_train)
    return np.asarray(imp.transform(X_masked), dtype=np.float64)


def _complete_neurodb(
    X_train: np.ndarray, X_masked: np.ndarray, beta: float, steps: int
) -> np.ndarray:
    """Store train rows as Hopfield patterns; complete each masked test row.

    Known fields are clamped (the mask), unknown fields are reconstructed. The
    NaN placeholders for unknown fields are replaced with 0 before the call —
    they are excluded from the similarity by the mask and overwritten by the
    reconstruction, so the placeholder value is irrelevant.
    """

    d = int(X_train.shape[1])
    mem = Memory("bench", d, beta=float(beta))
    mem.write([{"vector": row} for row in np.asarray(X_train, dtype=np.float32)])
    out = np.asarray(X_masked, dtype=np.float64).copy()
    for i in range(X_masked.shape[0]):
        row = X_masked[i]
        known_idx = np.where(~np.isnan(row))[0]
        query = np.where(np.isnan(row), 0.0, row).astype(np.float32)
        res = mem.complete(query, beta=float(beta), mask=known_idx.tolist(), steps=int(steps))
        recon = res["reconstruction"]
        if recon is not None:
            out[i] = np.asarray(recon, dtype=np.float64)
    return out


_METHODS = {
    "neurodb": _complete_neurodb,
    "knn": _impute_knn,
    "iterative": _impute_iterative,
}


def _best_config(
    method: str,
    X_fit: np.ndarray,
    X_val_masked: np.ndarray,
    truth_val: np.ndarray,
    known_val: np.ndarray,
) -> dict:
    """Pick the grid config with the lowest validation RMSE (ties → first)."""

    fn = _METHODS[method]
    best_cfg: dict | None = None
    best_score = float("inf")
    for cfg in GRIDS[method]:
        pred = fn(X_fit, X_val_masked, **cfg)
        score, _ = _score(pred, truth_val, known_val)
        if score < best_score:
            best_score, best_cfg = score, cfg
    assert best_cfg is not None
    return best_cfg


def run_benchmark(
    dataset: Dataset,
    mask_fraction: float = 0.4,
    test_frac: float = 0.3,
    val_frac: float = 0.25,
    seed: int = 0,
) -> BenchmarkReport:
    """Tune each method on a validation split, then score it on held-out test."""

    rng = np.random.default_rng(seed)
    X = dataset.X.astype(np.float64)
    y = dataset.labels

    train_idx, test_idx = _stratified_split(y, test_frac, rng)
    core_rel, val_rel = _stratified_split(y[train_idx], val_frac, rng)
    core_idx, val_idx = train_idx[core_rel], train_idx[val_rel]

    X_core, X_val = X[core_idx], X[val_idx]
    X_trainfull, X_test = X[train_idx], X[test_idx]

    mask_rng = np.random.default_rng(seed + 1)
    known_val = _random_known_mask(len(val_idx), dataset.d, mask_fraction, mask_rng)
    known_test = _random_known_mask(len(test_idx), dataset.d, mask_fraction, mask_rng)
    X_val_masked = _apply_mask(X_val, known_val)
    X_test_masked = _apply_mask(X_test, known_test)

    reports: list[MethodReport] = []
    for name in ("neurodb", "knn", "iterative"):
        cfg = _best_config(name, X_core, X_val_masked, X_val, known_val)
        pred = _METHODS[name](X_trainfull, X_test_masked, **cfg)
        overall, per_field = _score(pred, X_test, known_test)
        reports.append(MethodReport(name, cfg, overall, per_field))

    return BenchmarkReport(
        dataset=dataset.name,
        n_train=len(train_idx),
        n_test=len(test_idx),
        d=dataset.d,
        n_modes=dataset.n_modes,
        mask_fraction=mask_fraction,
        feature_names=list(dataset.feature_names),
        methods=tuple(reports),
    )

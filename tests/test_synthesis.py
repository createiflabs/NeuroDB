"""Synthetic generation: hard constraints satisfied, requested correlations
reproduced, the too-clean trap detected, and calibration without retaining data."""

from __future__ import annotations

import numpy as np

from neurodb.synthesis import (
    CriteriaSpec,
    FieldSpec,
    calibrate,
    diagnose,
    generate,
    is_too_clean,
)


def _spec(correlation=None, constraints=None):
    return CriteriaSpec(
        fields=[
            FieldSpec("a", "numeric", low=0, high=100, mean=50, std=10),
            FieldSpec("b", "numeric", low=0, high=100, mean=40, std=8),
        ],
        correlation=correlation,
        constraints=constraints or [],
    )


def test_hard_constraints_satisfied():
    # a must always exceed b.
    spec = _spec(constraints=[lambda r: r["a"] > r["b"]])
    matrix, names = generate(spec, 500, seed=1)
    assert names == ["a", "b"]
    assert matrix.shape == (500, 2)
    assert np.all(matrix[:, 0] > matrix[:, 1])


def test_requested_correlation_reproduced():
    target = np.array([[1.0, 0.7], [0.7, 1.0]])
    matrix, _ = generate(_spec(correlation=target), 8000, seed=2)
    realized = np.corrcoef(matrix, rowvar=False)[0, 1]
    assert abs(realized - 0.7) < 0.1


def test_bounds_respected():
    matrix, _ = generate(_spec(), 1000, seed=3)
    assert matrix.min() >= 0.0 and matrix.max() <= 100.0


def test_diagnostics_flag_too_clean_population():
    # Target has real cross-field structure...
    target = np.array([[1.0, 0.8], [0.8, 1.0]])
    # ...but the population was drawn independently (the classic failure).
    rng = np.random.default_rng(0)
    independent = np.column_stack(
        [rng.normal(50, 10, 4000), rng.normal(40, 8, 4000)]
    ).astype(np.float32)
    assert is_too_clean(independent, target) is True
    assert diagnose(independent, target)["degenerate"] is True


def test_diagnostics_pass_realistic_population():
    target = np.array([[1.0, 0.7], [0.7, 1.0]])
    matrix, _ = generate(_spec(correlation=target), 4000, seed=4)
    assert is_too_clean(matrix, target) is False
    d = diagnose(matrix, target)
    assert d["degenerate"] is False
    assert d["min_variance"] > 1.0


def test_constant_field_is_degenerate():
    constant = np.column_stack(
        [np.full(100, 5.0), np.linspace(0, 10, 100)]
    ).astype(np.float32)
    assert is_too_clean(constant) is True


def test_calibration_matches_marginals_without_retaining_records():
    rng = np.random.default_rng(5)
    sample = np.column_stack(
        [rng.normal(50, 10, 2000), rng.normal(40, 8, 2000)]
    )
    params = calibrate(sample)
    # Parameters only — no record array is returned.
    assert set(params) == {"means", "stds", "correlation", "n_samples"}
    assert abs(params["means"][0] - 50) < 1.0
    assert abs(params["stds"][1] - 8) < 1.0
    assert params["n_samples"] == 2000

    # Those parameters reproduce the marginals in a fresh synthetic population.
    m, s = params["means"], params["stds"]
    spec = CriteriaSpec(
        fields=[
            FieldSpec("a", "numeric", low=0, high=200, mean=m[0], std=s[0]),
            FieldSpec("b", "numeric", low=0, high=200, mean=m[1], std=s[1]),
        ],
        correlation=np.array(params["correlation"]),
    )
    matrix, _ = generate(spec, 4000, seed=6)
    assert abs(matrix[:, 0].mean() - 50) < 1.5

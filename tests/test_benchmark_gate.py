"""Tier 0 — the existential gate, in executable form.

The differentiation thesis (goal doc Tier 0.1): on multimodal data NeuroDB's
``complete`` should beat trivial sklearn imputers because it "snaps to the
correct attractor" while k-NN/iterative imputers "average between clusters".

The benchmark **disproves** this. A distance-based ``KNNImputer`` beats NeuroDB's
dot-product–softmax completion in every regime tested, and the gap *widens* as
clusters separate. So the headline gate test is recorded as a strict ``xfail``:
it documents the unmet gate, runs in CI, and will turn into a loud failure
(XPASS) the day someone actually makes NeuroDB win — at which point it should
become a hard assertion. The honest sub-findings (NeuroDB beats IterativeImputer
but loses to KNN) are locked in as ordinary passing tests so a regression in
*either* direction is caught.

See ``docs/ACCURACY.md`` for the full characterization.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")  # benchmark-only dep; core suite stays light.

from benchmarks.datasets import make_multimodal  # noqa: E402
from benchmarks.protocol import (  # noqa: E402
    _apply_mask,
    _random_known_mask,
    _score,
    run_benchmark,
)

GATE_RATIO = 0.75  # NeuroDB must reach <= 0.75 x best baseline to pass the gate.


@pytest.fixture(scope="module")
def synth_report():
    """One fair benchmark on a small, clearly-multimodal synthetic mixture."""

    ds = make_multimodal(n_clusters=5, dim=10, per_cluster=80, separation=6.0, seed=0)
    return run_benchmark(ds, mask_fraction=0.4, seed=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Tier 0 GATE UNMET: NeuroDB completion loses to KNNImputer on multimodal "
        "data (~1.3-2.6x worse). See docs/ACCURACY.md. If this XPASSes, NeuroDB's "
        "completion was improved past the gate — convert this to a hard assertion."
    ),
)
def test_completion_beats_knn_imputer_on_multimodal(synth_report):
    nb = synth_report.by_name("neurodb").overall_rmse
    knn = synth_report.by_name("knn").overall_rmse
    iterative = synth_report.by_name("iterative").overall_rmse
    best = min(knn, iterative)
    assert nb <= GATE_RATIO * best, (
        f"NeuroDB completion RMSE {nb:.4f} must be <= {GATE_RATIO} x best baseline "
        f"{best:.4f} (knn={knn:.4f}, iterative={iterative:.4f}); ratio={nb / best:.3f}."
    )


def test_neurodb_beats_iterative_imputer_on_multimodal(synth_report):
    """The part of the thesis that holds: attractor recall beats pure regression
    imputation, which regresses toward a global conditional mean on multimodal
    data."""

    nb = synth_report.by_name("neurodb").overall_rmse
    iterative = synth_report.by_name("iterative").overall_rmse
    assert nb < iterative


def test_knn_is_the_baseline_to_beat(synth_report):
    """The honest finding: a distance-based KNNImputer, not NeuroDB, is best at
    completion here. Locking this in means a future improvement to NeuroDB that
    flips it will (correctly) break this test and the strict-xfail gate together."""

    nb = synth_report.by_name("neurodb").overall_rmse
    knn = synth_report.by_name("knn").overall_rmse
    assert knn < nb


def test_benchmark_is_deterministic():
    """Same seed -> identical numbers (float64 scoring; BLAS-thread independent)."""

    ds = make_multimodal(n_clusters=4, dim=8, per_cluster=50, separation=6.0, seed=1)
    a = run_benchmark(ds, mask_fraction=0.4, seed=3)
    b = run_benchmark(ds, mask_fraction=0.4, seed=3)
    for name in ("neurodb", "knn", "iterative"):
        assert a.by_name(name).overall_rmse == pytest.approx(
            b.by_name(name).overall_rmse, abs=1e-9
        )


def test_scoring_counts_masked_cells_only():
    truth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred = np.array([[1.0, 2.0, 99.0], [4.0, 88.0, 6.0]])  # errors only on hidden
    known = np.array([[True, True, False], [True, False, True]])
    overall, per_field = _score(pred, truth, known)
    expected = np.sqrt((96.0**2 + 83.0**2) / 2)
    assert overall == pytest.approx(expected)
    assert np.isnan(per_field[0])  # field 0 was never hidden → undefined
    assert per_field[1] == pytest.approx(83.0)
    assert per_field[2] == pytest.approx(96.0)


def test_masks_leave_at_least_one_known_and_one_hidden():
    rng = np.random.default_rng(0)
    known = _random_known_mask(50, 6, 0.4, rng)
    assert known.all(axis=1).sum() == 0  # no fully-known rows
    assert (~known).all(axis=1).sum() == 0  # no fully-hidden rows
    masked = _apply_mask(np.zeros((50, 6)), known)
    assert np.isnan(masked).any(axis=1).all()  # every row has a hidden cell

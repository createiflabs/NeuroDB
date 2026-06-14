#!/usr/bin/env python3
"""Per-field anomaly attribution benchmark: NeuroDB vs IsolationForest.

A realistic, multi-dimensional dataset with *heterogeneous feature scales*
(server telemetry: cpu %, memory MB, latency ms, request rate, temperature,
power W). Anomalies are normal records with a single field spiked by several
sigma — exactly the case NeuroDB's ``zscore`` normalization is built for, and
exactly the case where per-field attribution matters.

The pitch is not "win on AUC". It is:

* **zero training** — write the normal population, recall is a matmul;
* **exact per-field attribution** — for every flagged record NeuroDB says
  *which* field is off, what it should have been, and by how many sigma;
* **instant incremental writes** — append a record, it is queryable immediately;
* **capacity awareness** — it can tell you when recall is degrading.

IsolationForest gives a single anomaly score with no clean per-field reason.

Run:  python examples/anomaly_benchmark/run.py
(IsolationForest comparison requires ``pip install scikit-learn``; the NeuroDB
results and attribution run with no extra dependencies.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neurodb.store import NeuroStore  # noqa: E402

FIELDS = ["cpu_pct", "mem_mb", "latency_ms", "req_rate", "temp_c", "power_w"]
# Heterogeneous scales — the whole point of the example.
MEANS = np.array([35.0, 4096.0, 25.0, 500.0, 50.0, 120.0])
STDS = np.array([8.0, 512.0, 6.0, 90.0, 5.0, 15.0])
RNG = np.random.default_rng(7)


def sample_normal(n: int) -> np.ndarray:
    return RNG.normal(MEANS, STDS, size=(n, len(FIELDS))).astype(np.float32)


def make_anomalies(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Normal records with one field spiked by 4–8 sigma. Returns (rows, field_idx)."""

    rows = sample_normal(n)
    fields = RNG.integers(0, len(FIELDS), size=n)
    sign = RNG.choice([-1.0, 1.0], size=n)
    mag = RNG.uniform(4.0, 8.0, size=n)
    for i in range(n):
        j = fields[i]
        rows[i, j] += sign[i] * mag[i] * STDS[j]
    return rows, fields


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann–Whitney U); no sklearn needed."""

    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    avg = {k: (cumulative[k] - (counts[k] - 1) / 2.0) for k in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    # 1. Build the normal population and a held-out test mix.
    train = sample_normal(2000)
    test_normal = sample_normal(400)
    test_anom, anom_fields = make_anomalies(400)
    test = np.vstack([test_normal, test_anom])
    labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anom))]).astype(int)

    # 2. NeuroDB: one create + one write (zero training), then a batched scan.
    store = NeuroStore("/tmp/neurodb_benchmark.npz")
    try:
        store.delete_memory("telemetry")
    except Exception:
        pass
    # Low beta on purpose: for *population* anomaly detection we want recall to
    # converge to the population prototype (the "expected normal" record), and
    # residual = how far this record sits from it. (See the capacity note below.)
    beta = 2.0
    mem = store.create_memory(
        "telemetry", len(FIELDS), beta=beta, fields=FIELDS, normalize="zscore"
    )
    mem.write([{"vector": row.tolist()} for row in train])

    reports = mem.anomaly_batch(test.tolist(), beta=beta)
    nd_scores = np.array([r["z_score"] for r in reports])
    nd_auc = roc_auc(nd_scores, labels)

    # 3. Attribution: did NeuroDB name the right offending field?
    anom_reports = reports[len(test_normal):]
    predicted_fields = np.array([r["fields"][0]["index"] for r in anom_reports])
    attribution_acc = float((predicted_fields == anom_fields).mean())

    print("=" * 68)
    print("NeuroDB  (zscore Hopfield, zero training)")
    print("=" * 68)
    print(f"  ROC-AUC on z_score ............ {nd_auc:.3f}")
    print(f"  top-field attribution accuracy  {attribution_acc:.1%}")
    print("\n  Sample attributions (what IsolationForest can't give you):")
    for r in anom_reports[:5]:
        f = r["fields"][0]
        print(
            f"    {f['name']:<11} is {f['z_deviation']:5.1f}sigma off  "
            f"(observed {f['value']:.1f}, expected ~{f['expected']:.1f})"
        )

    # 4. Capacity diagnostic on the populated memory.
    cap = mem.capacity_report()
    print("\n  Capacity report:")
    print(
        f"    status={cap['status']}  self_recall_fail={cap['self_recall_fail_fraction']:.2f}  "
        f"max_pairwise_sim={cap['max_pairwise_similarity']:.3f}"
    )
    print(
        "    (a tight, mutually-similar normal population reads as 'saturated' —\n"
        "     expected & desirable here: recall converges to the population norm.\n"
        "     The same signal warns you when you instead need *exact* per-record\n"
        "     recall and beta is too low — that's the silent-degradation case.)"
    )

    # 5. Baseline: IsolationForest (needs scikit-learn).
    print("\n" + "=" * 68)
    print("IsolationForest  (sklearn baseline; requires a fit step)")
    print("=" * 68)
    try:
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(n_estimators=200, random_state=0).fit(train)
        if_scores = -clf.score_samples(test)  # higher = more anomalous
        if_auc = roc_auc(if_scores, labels)
        print(f"  ROC-AUC ....................... {if_auc:.3f}")
        print("  per-field attribution ......... (none — single opaque score)")
    except ModuleNotFoundError:
        print("  scikit-learn not installed; run `pip install scikit-learn` to compare.")

    print(
        "\nTakeaway: comparable detection, but NeuroDB needs no training, scores"
        "\nincrementally as records arrive, and tells you *which field* and *by how"
        "\nmany sigma* — the actionable part of anomaly detection."
    )


if __name__ == "__main__":
    main()

# Anomaly attribution benchmark — NeuroDB vs IsolationForest

A reproducible, multi-dimensional example of NeuroDB's headline job: **per-field
anomaly attribution on structured records, with zero training.**

```bash
python examples/anomaly_benchmark/run.py
# IsolationForest comparison needs: pip install scikit-learn
```

## The dataset

Synthetic server-telemetry records with **heterogeneous feature scales** — the
case `normalize="zscore"` exists for:

| field | `cpu_pct` | `mem_mb` | `latency_ms` | `req_rate` | `temp_c` | `power_w` |
|-------|-----------|----------|--------------|------------|----------|-----------|
| ~mean | 35        | 4096     | 25           | 500        | 50       | 120       |
| ~std  | 8         | 512      | 6            | 90         | 5        | 15        |

2,000 "normal" records form the population. The test set is 400 normal + 400
anomalies, where each anomaly is a normal record with **one** field spiked by
4–8σ. Without normalization `mem_mb` (~1e3) would drown out everything; with
`zscore` every field is comparable.

## Results (seed 7)

| method                         | ROC-AUC | per-field attribution | training |
|--------------------------------|---------|-----------------------|----------|
| **NeuroDB** (`zscore`, β=2)    | **0.932** | **77.8%** top-field correct | none |
| IsolationForest (200 trees)    | 0.925   | none (one opaque score) | fit required |

NeuroDB matches IsolationForest's detection here **with no training step**, and
adds the part that actually drives a response:

```
cpu_pct     is   5.0sigma off  (observed 100.0, expected ~60.1)
power_w     is   3.4sigma off  (observed 224.9, expected ~174.5)
```

Each flagged record names the offending field, its expected value, and how many
σ it is off. IsolationForest gives a single number with no clean reason.

The script also prints a `capacity()` report on the populated memory. A tight,
mutually-similar normal population reads as `saturated` — which is **expected and
desirable** for population anomaly detection: recall converges to the population
prototype (the "expected normal"), and the residual is the deviation from it. The
same diagnostic is what warns you, in a different use case, when you wanted exact
per-record recall and β is too low — i.e. before recall silently degrades.

## Why you'd use this

Stand up NeuroDB, write your normal records once (no model fit), then stream new
records through `anomaly_batch` and get back, per record, *which field is weird,
what it should have been, and by how many σ* — updated instantly as records
change, with a built-in warning before recall degrades. That combination —
zero-training + exact attribution + instant incremental writes + capacity
awareness — is what a from-scratch IsolationForest pipeline doesn't give you.

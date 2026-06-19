# NeuroDB accuracy: where it works, where it doesn't

This is the honest accuracy characterization (goal Tier 0.2). It is deliberately
unflattering where the numbers are unflattering — overclaiming kills database
projects. All numbers are reproducible with `python -m benchmarks.run_benchmark`
and the probes described inline; raw benchmark output lives in
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## TL;DR

- **Exact content-addressable recall works well.** Given a full (or nearly full)
  query, NeuroDB returns the stored pattern with ~100% accuracy as long as the
  dimension is adequate for the number of stored patterns. This is the real,
  defensible capability.
- **Partial-record *completion* does not beat a trivial baseline.** On the very
  multimodal data where the design thesis predicted a win, NeuroDB's
  `complete` **loses to scikit-learn's `KNNImputer`** in every regime tested —
  by 1.3x to 2.6x — and the gap *widens* as the data becomes more cleanly
  clustered. The Tier 0 differentiation gate is **not met**. See
  [the repositioning note](#what-this-means-repositioning) below.
- **`anomaly` and `search` are honest but undifferentiated.** They do what they
  say; they do not beat purpose-built tools (`LocalOutlierFactor`, any ANN
  index). `search` is a brute-force O(N·d) scan today.

## The differentiation thesis, tested and disproven

The thesis (goal Tier 0.1): on multimodal data, k-NN / iterative imputers
"average between clusters" while NeuroDB "snaps to the correct attractor",
yielding lower per-field imputation error. We tested it fairly: every method
tunes its hyperparameters on a validation split (equal budget), trains/stores on
the same data, and is scored on the masked cells of held-out records in
standardized (per-feature std) units.

Headline (`benchmarks/RESULTS.md`, 40% of fields masked):

| dataset | NeuroDB | KNNImputer | IterativeImputer | NeuroDB / best |
|---|---|---|---|---|
| synthetic multimodal (6 clusters, d=12) | 0.393 | **0.232** | 0.557 | **1.69x worse** |
| UCI Wine (3 cultivars, d=13) | 1.003 | 0.744 | **0.739** | **1.36x worse** |

The gate target was NeuroDB <= 0.75 x best baseline. It is not met on any
dataset.

### It gets worse as the data gets *more* multimodal

Sweeping cluster separation on the synthetic set (40% / 25% masked):

| separation | NeuroDB | KNNImputer | NeuroDB / KNN |
|---|---|---|---|
| 2 (overlapping) | 0.80–0.84 | 0.58–0.61 | ~1.4x worse |
| 4 | 0.52 | 0.34 | ~1.5x worse |
| 6 | 0.40–0.43 | 0.23–0.24 | ~1.7x worse |
| 10 (well separated) | 0.32–0.37 | 0.14–0.15 | **~2.2–2.6x worse** |

The better-defined the clusters, the *more* KNN beats NeuroDB. This is the exact
inverse of the thesis: KNN snaps within-cluster (its k neighbors are all in the
right cluster) and averages out the within-cluster noise, while NeuroDB's
softmax leaks weight across clusters unless `beta` is large — and large `beta`
makes it collapse onto a single noisy neighbor.

### Why: `beta` sensitivity points the wrong way

Completion error vs `beta` on the synthetic set (lower is better):

| beta | 1 | 2 | 4 | 8 (default) | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| NeuroDB RMSE | 0.393 | 0.407 | 0.437 | 0.465 | 0.483 | 0.493 | 0.497 | 0.500 |

(KNN, k=5, on the same split: **0.242**.)

Error *increases monotonically* with `beta`. "Snapping" (high `beta`) is strictly
worse than "blending" (low `beta`) for completion here, because the best estimate
of a masked field is the cluster-conditional mean, which low `beta` approximates
by averaging many in-cluster patterns — exactly what KNN does, but KNN's
distance-based, uniform k-average is a cleaner estimator than a
dot-product-weighted softmax. NeuroDB's own default (`beta=8`) is a poor setting
for completion.

### The classic associative-memory case, too

With *distinct* (non-clustered) random patterns and a 40%-masked cue — the
textbook content-addressable-memory scenario — a trivial 1-nearest-neighbor
recovers the pattern **exactly (0.0000 RMSE)** at every `beta`, while NeuroDB
blends to 0.13–0.18. `KNNImputer(n_neighbors=1)` *is* that 1-NN.

## Where NeuroDB is genuinely good: exact recall and capacity

Given a full query (content addressing / dedup / "have I seen this vector?"),
recall is excellent and the modern-Hopfield capacity is large. Fraction of
stored patterns recovered within 1e-2 L2 from their exact query (`beta=64`):

| dimension | N=10 | N=100 | N=1000 | N=5000 |
|---|---|---|---|---|
| d=32 | 1.00 | 1.00 | 1.00 | 0.99 |
| d=128 | 1.00 | 1.00 | 1.00 | 1.00 |

Recall degrades only when the dimension is small relative to N (e.g. d=16 with
hundreds of patterns starts to show ~0.3 L2 error on exact queries). Practical
rule of thumb: keep `d` comfortably large relative to the number of *mutually
similar* patterns; well-separated patterns pack far more densely than the
worst case.

### Capacity / memory footprint

The store is a single in-memory `float32` matrix: **bytes ≈ 4 · d · N** for the
vectors, plus per-pattern id + JSON metadata overhead. Examples:

| dimension | bytes/pattern (vector only) | 1M patterns |
|---|---|---|
| 64 | 256 B | ~256 MB |
| 256 | 1 KB | ~1 GB |
| 1024 | 4 KB | ~4 GB |

The entire store currently must fit in RAM and is reloaded whole on boot (see the
beyond-RAM and memory-map gaps in the roadmap).

## Known failure mode: a single anomalous field hidden by good fields

`anomaly` recalls the nearest stored pattern and reports per-field deviation.
A record that is normal on most fields but anomalous on one is recalled to a
close neighbour, so the *aggregate* residual norm stays small and the anomaly
can be masked. Mitigation, already implemented: `anomaly` returns **per-field**
residuals (`fields[].deviation`), so the offending field surfaces with a large
single-field deviation even when the overall `score` is modest. Consumers should
threshold on per-field deviation, not only on the aggregate `score`.

## When NOT to use NeuroDB

- **You need best-in-class imputation / missing-value completion.** Use
  `sklearn`'s `KNNImputer` or `IterativeImputer` (or a gradient-boosted imputer).
  They are simpler and measurably better on the data we tried.
- **You need approximate nearest-neighbour search at scale.** Use a real vector
  database / ANN index (FAISS, hnswlib, Qdrant, pgvector). NeuroDB `search` is a
  brute-force O(N·d) scan.
- **You need best-in-class outlier detection.** Use `LocalOutlierFactor`,
  IsolationForest, etc.

## When NeuroDB is a reasonable fit

- In-memory, **exact** content-addressable recall / dedup / "nearest stored
  record" with a tiny dependency footprint and no training step.
- Small-to-medium datasets that fit in RAM where operational simplicity (one
  process, one file, one HTTP API) matters more than raw retrieval quality.
- A soft-recall `complete` that returns a plausible attractor blend — useful as a
  prior or a sketch, **not** as a precision imputer.

## What this means: repositioning

The only claimed differentiator (multimodal completion) does not survive a fair
benchmark against a five-line scikit-learn baseline. Per the goal document, this
is the gate, and it says: reposition before further hardening. The honest
positioning is **"a simple, container-native, in-memory content-addressable
vector store with strong exact recall"** — not "a completion engine that beats
imputers." Hardening work (the O(N^2) ingest bug, durability/WAL, concurrency,
ANN search, resource limits, security) remains valuable under that honest
positioning, because it makes the *store* production-grade regardless of the
completion claim.

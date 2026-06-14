# Performance & memory characterization

Measures the latency and footprint of the hot paths across a grid of
`(N patterns, D dims, M batch)` and prints a table. This is **honesty, not
optimization** — it documents where single-node NeuroDB is and isn't appropriate.

```bash
python examples/benchmark_perf/run.py
```

NeuroDB recall is a full scan: every query is an `O(N x D)` matmul. That is exact
and trivial to operate, but **latency grows linearly with N**. The table shows
the constant and the slope on your hardware; `approx_mem` is the in-process
footprint (a `zscore`/`l2` memory keeps a cached normalized matrix, ~2x raw).

Representative run (commodity laptop CPU, numpy BLAS; absolute numbers vary):

```
       N     D     M |  write/1k   anomaly   a_batch  validate | approx_mem
    1000    32   100 |     9.50m     0.11m     4.90m     5.73m |     250KiB
   10000    64   100 |    11.19m     0.26m    27.97m    27.70m |       5MiB
   50000   128   256 |    14.89m     2.91m   537.68m   555.78m |      49MiB
  100000   256   256 |    31.40m     8.32m  1016.25m  1010.40m |     195MiB
```

(`m` = milliseconds; `write/1k` is per 1,000 rows.) Single-record `anomaly` stays
sub-10ms into the 100k range, but batch/validate latency and memory scale with N:
past a few hundred thousand patterns per memory on one node, prefer sharding the
population across memories, or a purpose-built ANN index if you need approximate
billion-scale search. NeuroDB favours **exact** recall and per-field attribution
at single-node scale.

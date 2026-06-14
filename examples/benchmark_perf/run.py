#!/usr/bin/env python3
"""Performance & memory characterization for single-node NeuroDB.

This is **honesty, not optimization**: it measures the latency and footprint of
the hot paths (`write`, `anomaly`, `anomaly_batch`, `validate`) across a grid of
(N patterns, D dims, M batch) and prints a table. Use it to set expectations and
to see where the single-node, full-scan design stops being appropriate.

NeuroDB recall is a full scan: every query is an O(N x D) matmul. That is exact
and trivial to operate, but latency grows linearly with N. This script shows you
the constant and the slope for your hardware.

Run:  python examples/benchmark_perf/run.py
(No extra dependencies beyond NeuroDB itself.)
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neurodb.store import NeuroStore  # noqa: E402
from neurodb_client.validate import run_validation  # noqa: E402

# (N patterns, D dims, M batch) grid — small, representative points.
GRID = [
    (1_000, 32, 100),
    (10_000, 64, 100),
    (50_000, 128, 256),
    (100_000, 256, 256),
]
REPEATS = 5


class _LocalMemory:
    """Adapt a store Memory to the client's ``anomaly_batch`` shape so the real
    client-side ``run_validation`` can be benchmarked without a live server."""

    def __init__(self, mem):
        self._mem = mem
        self.name = mem.name

    def anomaly_batch(self, items, *, beta=None, top_k=5, filter=None):
        vectors = [it["vector"] for it in items]
        results = self._mem.anomaly_batch(vectors, beta, top_k, filter)
        return {"results": results, "count": len(results)}


def _median_ms(fn, repeats=REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TiB"


def bench(tmp_dir: Path) -> None:
    rng = np.random.default_rng(0)
    header = (
        f"{'N':>8} {'D':>5} {'M':>5} | {'write/1k':>9} {'anomaly':>9} "
        f"{'a_batch':>9} {'validate':>9} | {'approx_mem':>10}"
    )
    print(header)
    print("-" * len(header))

    for n, d, m in GRID:
        store = NeuroStore(tmp_dir / f"bench_{n}_{d}.npz")
        mem = store.create_memory(
            "b", d, fields=[f"f{i}" for i in range(d)], normalize="zscore"
        )
        data = rng.standard_normal((n, d)).astype(np.float32)

        # Bulk-write all N rows once; report write cost normalized per 1k rows.
        rows = data.tolist()
        t0 = time.perf_counter()
        mem.write([{"vector": r} for r in rows])
        write_ms = (time.perf_counter() - t0) * 1000.0 / (n / 1000.0)

        # Warm the normalization cache so recall timings are steady-state, then
        # measure recall against the full N-pattern population. Default-arg binding
        # keeps each lambda tied to this iteration's objects.
        mem.anomaly(data[0].tolist())
        queries = data[:m].tolist()
        one = data[0].tolist()
        anomaly_ms = _median_ms(lambda mem=mem, one=one: mem.anomaly(one))
        batch_ms = _median_ms(
            lambda mem=mem, q=queries: mem.anomaly_batch(q, None, 5, None)
        )
        client = _LocalMemory(mem)
        validate_ms = _median_ms(
            lambda c=client, q=queries: run_validation(c, q, threshold=3.0)
        )

        print(
            f"{n:>8} {d:>5} {m:>5} | {write_ms:>8.2f}m {anomaly_ms:>8.2f}m "
            f"{batch_ms:>8.2f}m {validate_ms:>8.2f}m | "
            f"{_human_bytes(mem.approx_bytes):>10}"
        )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bench(Path(d))

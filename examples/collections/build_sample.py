#!/usr/bin/env python3
"""Build the toy sample collection shipped in this repo.

This is an **obviously-toy, non-commercial sample** so users can exercise the
loader (`neurodb collection info/load`) end to end. It is NOT a real
domain-validated collection — it targets a fictional criteria set and carries no
attestation. Real collections (e.g. DRV-Prüfkatalog) are built privately with
this same open tooling and distributed only to customers.

Run:  python examples/collections/build_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neurodb.collections import CollectionSchema, build_bundle, synthetic_provenance  # noqa: E402
from neurodb.synthesis import CriteriaSpec, FieldSpec, diagnose, generate  # noqa: E402

OUT = Path(__file__).resolve().parent / "sample_service_health.ndcoll"

# A fictional 16-feature "service health" record. Higher dimension keeps the
# Hopfield self-recall comfortable (capacity stays healthy) while still carrying
# realistic cross-field correlation.
FIELDS = [f"feat_{i:02d}" for i in range(16)]
MEANS = [35, 4096, 25, 500, 50, 120, 60, 300, 12, 0.4, 200, 75, 18, 90, 6, 1.2]
STDS = [8, 512, 6, 90, 5, 15, 12, 60, 3, 0.1, 40, 10, 4, 8, 1.5, 0.3]


def main() -> None:
    fields = [
        FieldSpec(name, "numeric", low=0, high=mu + 8 * sd, mean=mu, std=sd)
        for name, mu, sd in zip(FIELDS, MEANS, STDS, strict=True)
    ]
    # A little correlation structure so the population isn't independent draws.
    corr = np.eye(16)
    for i, j, r in [(0, 5, 0.5), (2, 3, 0.4), (0, 2, 0.3), (10, 11, 0.45)]:
        corr[i, j] = corr[j, i] = r
    spec = CriteriaSpec(fields=fields, correlation=corr)
    patterns, _ = generate(spec, 150, seed=42)

    report = diagnose(patterns, corr)
    print(f"realism: degenerate={report['degenerate']} "
          f"min_variance={report['min_variance']:.2f} capacity={report['capacity']['status']}")

    schema = CollectionSchema(
        fields=[{"name": n, "type": "numeric", "unit": "u"} for n in FIELDS],
        dimension=16,
    )
    provenance = synthetic_provenance(
        "neurodb sample builder",
        generator="neurodb.synthesis",
        generation_params={"n": 150, "seed": 42, "dimension": 16, "correlated": True},
        calibration_source="hand-specified illustrative targets (no real data)",
        summary="Toy, non-commercial sample collection for trying the loader.",
    )
    build_bundle(
        OUT,
        name="sample_service_health",
        patterns=patterns,
        schema=schema,
        criteria={"set": "SAMPLE-Katalog (fictional)", "version": "0.1", "coverage": {}},
        provenance=provenance,
        beta=30.0,
        normalize="zscore",
    )  # unsigned: a community sample, not a signed commercial artifact
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

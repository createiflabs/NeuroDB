"""Tier 4.1 — logical JSONL export/import keeps data out of a NumPy-version-
specific binary format and round-trips losslessly."""

from __future__ import annotations

import json

import numpy as np

from neurodb.portability import export_memory_jsonl, import_memory_jsonl


def test_export_import_roundtrip_across_numpy_versions(tmp_path, store_factory):
    rng = np.random.default_rng(0)
    src = store_factory("a.npz")
    mem = src.create_memory("m", 5, beta=12.0, fields=list("abcde"), normalize="zscore")
    x = rng.normal(size=(50, 5)).astype(np.float32)
    mem.write(
        [
            {"id": f"r{i}", "vector": x[i], "metadata": {"i": i, "tag": f"t{i % 3}"}}
            for i in range(50)
        ]
    )

    path = tmp_path / "export.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        written = export_memory_jsonl(mem, fh)
    assert written == 50

    # The on-disk format is plain JSON — no NumPy binary, so it is portable
    # across NumPy versions and tooling (every line parses as JSON).
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 51  # header + 50 patterns
    assert "__neurodb_memory__" in records[0]

    # Import into a *fresh* store and verify full fidelity.
    dst = store_factory("b.npz")
    with open(path, encoding="utf-8") as fh:
        m2 = import_memory_jsonl(dst, fh)
    assert m2.count == 50
    assert m2.dimension == 5
    assert m2.beta == 12.0
    assert m2.fields == list("abcde")
    assert m2.normalize == "zscore"
    for i in range(50):
        got = m2.get(f"r{i}")
        assert np.allclose(got["vector"], x[i], atol=1e-6)
        assert got["metadata"] == {"i": i, "tag": f"t{i % 3}"}


def test_import_rejects_export_without_header(tmp_path, store_factory):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "vector": [1, 2]}\n', encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        try:
            import_memory_jsonl(store_factory(), fh)
        except ValueError as exc:
            assert "header" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing header")

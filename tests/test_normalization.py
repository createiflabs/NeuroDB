"""Per-memory normalization (zscore / l2) for the Hopfield complete/anomaly path.

These lock in *why* normalization matters: without it the largest-magnitude
field dominates the softmax and anomalies on smaller-scale fields are invisible
(the README sensor example). With ``zscore`` every field contributes comparably
and the anomaly residual becomes a "standard deviations from expected" signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurodb.hopfield import retrieve
from neurodb.store import Memory, resolve_normalize

# The README's sensor dataset: temperature ~20, humidity ~50, pressure ~1013.
SENSOR_ROWS = [
    [20.0, 50.0, 1013.0],
    [21.0, 52.0, 1012.0],
    [19.0, 48.0, 1014.0],
    [20.0, 51.0, 1013.0],
]
SENSOR_FIELDS = ["temperature", "humidity", "pressure"]


def _sensor_memory(store, normalize="zscore", beta=2.0):
    mem = store.create_memory(
        "sensors", 3, beta=beta, fields=SENSOR_FIELDS, normalize=normalize
    )
    mem.write([{"vector": row} for row in SENSOR_ROWS])
    return mem


# -- default selection -----------------------------------------------------


def test_default_is_zscore_when_fields_given():
    assert resolve_normalize(None, ["a", "b"]) == "zscore"


def test_default_is_none_without_fields():
    assert resolve_normalize(None, None) == "none"


def test_explicit_mode_overrides_default():
    assert resolve_normalize("none", ["a", "b"]) == "none"
    assert resolve_normalize("l2", None) == "l2"


def test_invalid_mode_rejected():
    from neurodb.store import MemoryError_

    with pytest.raises(MemoryError_):
        resolve_normalize("bogus", None)


# -- 1. README sensor example regression (the headline) --------------------


def test_sensor_completion_returns_means_not_a_saturated_row(store_factory):
    mem = _sensor_memory(store_factory(), normalize="zscore", beta=12.0)
    # Know only the temperature (the dataset mean) → the unknown fields should
    # complete to the dataset means, NOT a single saturated row.
    out = mem.complete([20.0, 0.0, 0.0], mask=[0])
    recon = out["reconstruction"]
    assert recon[0] == pytest.approx(20.0, abs=1e-3)  # known field preserved
    assert recon[1] == pytest.approx(np.mean([r[1] for r in SENSOR_ROWS]), abs=0.5)
    assert recon[2] == pytest.approx(np.mean([r[2] for r in SENSOR_ROWS]), abs=0.5)
    # Not collapsed onto one stored row: attention is spread, not one-hot.
    weights = [c["weight"] for c in out["weights"]]
    assert max(weights) < 0.9


def test_sensor_anomaly_flags_humidity_by_z_deviation(store_factory):
    mem = _sensor_memory(store_factory(), normalize="zscore", beta=2.0)
    out = mem.anomaly([20.0, 95.0, 1013.0])
    top = out["fields"][0]
    assert top["name"] == "humidity"
    # Reported by both raw and normalized deviation.
    assert "z_deviation" in top
    assert "z_score" in out
    assert top["z_deviation"] == max(f["z_deviation"] for f in out["fields"])


def test_sensor_anomaly_under_none_misses_small_scale_field(store_factory):
    # Documents the bug the patch fixes: with "none" the pressure-scale dot
    # product swamps everything, so changing the temperature far from normal
    # leaves the reconstruction essentially unchanged.
    mem = _sensor_memory(store_factory(), normalize="none", beta=12.0)
    base = mem.complete([20.0, 50.0, 1013.0])["reconstruction"]
    bumped = mem.complete([99.0, 50.0, 1013.0])["reconstruction"]
    np.testing.assert_allclose(base, bumped, atol=1e-3)


# -- 2. scale-invariance ---------------------------------------------------


def _weights_by_id(mem, query):
    out = mem.complete(query, top_k=mem.count)
    return {c["id"]: c["weight"] for c in out["weights"]}


def test_zscore_is_scale_invariant_but_none_is_not(store_factory):
    rows = [[1.0, 2.0], [3.0, 1.0], [2.0, 3.0], [0.5, 2.5]]
    query = [2.0, 2.0]
    scale = 1000.0

    def build(store, normalize, beta):
        plain = store.create_memory("plain", 2, beta=beta, normalize=normalize)
        plain.write([{"id": str(i), "vector": r} for i, r in enumerate(rows)])
        scaled = store.create_memory("scaled", 2, beta=beta, normalize=normalize)
        scaled.write(
            [{"id": str(i), "vector": [r[0], r[1] * scale]} for i, r in enumerate(rows)]
        )
        return plain, scaled

    # zscore standardizes both stores to the same Z, so the attention weights
    # are identical regardless of beta (true scale-invariance).
    zp, zs = build(store_factory(), "zscore", beta=4.0)
    w_plain = _weights_by_id(zp, query)
    w_scaled = _weights_by_id(zs, [query[0], query[1] * scale])
    for _id in w_plain:
        assert w_plain[_id] == pytest.approx(w_scaled[_id], abs=1e-4)

    # none: raw dot products. A low beta keeps the plain store a soft blend
    # while the ×1000 store's huge sims saturate → the weights diverge, which is
    # exactly the magnitude-dominance the patch fixes.
    np_, ns = build(store_factory("none.npz"), "none", beta=0.1)
    n_plain = _weights_by_id(np_, query)
    n_scaled = _weights_by_id(ns, [query[0], query[1] * scale])
    # With raw dot product the scaled dimension dominates → weights differ.
    assert any(
        abs(n_plain[_id] - n_scaled[_id]) > 1e-3 for _id in n_plain
    )


# -- 3. anomalous-field visibility -----------------------------------------


def test_zscore_z_deviation_tracks_std_devs(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=30.0, normalize="zscore")
    # Patterns that differ sharply per dimension, so a query matching one of
    # them in two dimensions recalls it cleanly at high beta.
    rows = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    mem.write([{"vector": r} for r in rows])
    mem.stats()  # populate the mean/std caches for introspection
    std = mem._std.copy()  # noqa: SLF001 - deliberate test introspection
    # Query == pattern [10,0,0] except field 2, shifted by N std-devs.
    n = 2.0
    q = [10.0, 0.0, n * float(std[2])]
    out = mem.anomaly(q, top_k=3)
    top = out["fields"][0]
    assert top["index"] == 2
    assert top["z_deviation"] == pytest.approx(n, abs=0.3)
    # Recall still locks onto [10,0,0]: the reconstruction reports the EXPECTED
    # value for the shifted field, not the anomalous query value — so the
    # anomaly is *visible* (the bug "none" mode exhibits on small-scale fields).
    assert abs(top["expected"]) < abs(q[2]) / 2
    assert not np.allclose(out["reconstruction"], q, atol=1e-2)


# -- 4. backward-compat load -----------------------------------------------


def test_manifest_without_normalize_loads_as_none(store_factory):
    matrix = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
    manifest = {
        "name": "legacy",
        "dimension": 3,
        "beta": 8.0,
        "fields": None,
        # NOTE: deliberately no "normalize" key (pre-patch file).
        "ids": ["a", "b"],
        "metadata": [{}, {}],
    }
    mem = Memory.from_manifest(manifest, matrix)
    assert mem.normalize == "none"

    # Byte-identical to the pure pre-patch engine on a fixed input.
    q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    expected, _ = retrieve(matrix, q, 8.0, None, 1)
    got = mem.complete(q.tolist())["reconstruction"]
    np.testing.assert_array_equal(np.asarray(got, dtype=np.float32), expected)


def test_normalize_round_trips_through_save_load(store_factory):
    store = store_factory()
    _sensor_memory(store, normalize="zscore")
    before = store.get_memory("sensors").anomaly([20.0, 95.0, 1013.0])
    store.save_all()  # writes only persist on save/flush

    reloaded = store_factory()  # same tmp_path data file → reload from disk
    mem = reloaded.get_memory("sensors")
    assert mem.normalize == "zscore"
    after = mem.anomaly([20.0, 95.0, 1013.0])
    np.testing.assert_allclose(after["reconstruction"], before["reconstruction"])
    assert after["fields"][0]["name"] == "humidity"


# -- 5. degenerate variance ------------------------------------------------


def test_constant_dimension_is_finite_and_zero_deviation(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=8.0, normalize="zscore")
    # Dimension 2 is constant across every row (zero variance).
    mem.write(
        [
            {"vector": [1.0, 5.0, 7.0]},
            {"vector": [2.0, 6.0, 7.0]},
            {"vector": [3.0, 4.0, 7.0]},
        ]
    )
    out = mem.anomaly([2.0, 5.0, 7.0], top_k=3)
    assert np.all(np.isfinite(out["reconstruction"]))
    assert np.all(np.isfinite([f["z_deviation"] for f in out["fields"]]))
    const = next(f for f in out["fields"] if f["index"] == 2)
    assert const["z_deviation"] == pytest.approx(0.0, abs=1e-3)


# -- 6. l2 mode ------------------------------------------------------------


def test_l2_recall_is_unit_norm_and_ranks_nearest_first(store_factory):
    store = store_factory()
    mem = store.create_memory("emb", 3, beta=30.0, normalize="l2")
    vecs = {
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }
    mem.write([{"id": k, "vector": v} for k, v in vecs.items()])
    # A query pointing mostly along +x, but with non-unit magnitude.
    out = mem.complete([5.0, 0.5, 0.0])
    assert out["top"]["id"] == "x"
    assert float(np.linalg.norm(out["reconstruction"])) == pytest.approx(1.0, abs=0.05)


# -- 7. single-pattern / empty memory --------------------------------------


def test_zscore_empty_memory_does_not_crash(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, normalize="zscore")
    assert mem.complete([1.0, 2.0, 3.0])["reconstruction"] is None
    assert mem.anomaly([1.0, 2.0, 3.0])["fields"] == []


def test_zscore_single_pattern_behaves_like_identity(store_factory):
    store = store_factory()
    mem = store.create_memory("m", 3, beta=8.0, normalize="zscore")
    mem.write([{"vector": [4.0, 5.0, 6.0]}])
    out = mem.complete([4.0, 5.0, 6.0])
    # std fallback (mean=0, std=1) → recon equals the single stored row.
    np.testing.assert_allclose(out["reconstruction"], [4.0, 5.0, 6.0], atol=1e-4)
    an = mem.anomaly([4.0, 5.0, 6.0])
    assert np.all(np.isfinite(an["reconstruction"]))


# -- API surface -----------------------------------------------------------


def test_create_memory_api_threads_normalize(client):
    r = client.post(
        "/v1/memories",
        json={"name": "s", "dimension": 3, "fields": SENSOR_FIELDS},
    )
    assert r.status_code == 201
    assert r.json()["normalize"] == "zscore"  # default from fields

    r2 = client.post(
        "/v1/memories",
        json={"name": "raw", "dimension": 3, "normalize": "l2"},
    )
    assert r2.json()["normalize"] == "l2"

    stats = client.get("/v1/stats").json()
    by_name = {m["name"]: m for m in stats["detail"]}
    assert by_name["s"]["normalize"] == "zscore"
    assert by_name["raw"]["normalize"] == "l2"

"""Collection bundle format: build → load round-trip, info without patterns,
required provenance, signing/verification, attestation, license seam, migration."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodb.collections import bundle as bundle_mod
from neurodb.collections import signing
from neurodb.collections.bundle import (
    BundleError,
    CollectionSchema,
    build_bundle,
    info,
    read_manifest,
    sign_attestation,
    synthetic_provenance,
    verify_attestation,
    verify_bundle,
)
from neurodb.collections.license import SignatureOnlyLicense, set_license
from neurodb.config import Settings
from neurodb.server import create_app
from neurodb.store import NeuroStore

PATTERNS = np.array([[20.0, 50.0, 1013.0], [21.0, 52.0, 1012.0], [19.0, 48.0, 1014.0],
                     [20.0, 51.0, 1013.0], [21.0, 49.0, 1012.0]], dtype=np.float32)
SCHEMA = CollectionSchema(
    fields=[
        {"name": "temperature", "type": "numeric", "unit": "C"},
        {"name": "humidity", "type": "numeric", "unit": "%"},
        {"name": "pressure", "type": "numeric", "unit": "hPa"},
    ],
    dimension=3,
)
CRITERIA = {"set": "TEST-Katalog", "version": "1.0", "coverage": {"humidity": ["H1"]}}


def _provenance():
    return synthetic_provenance("test-suite", generator="neurodb-synth 0", summary="toy")


def _build(path, *, sign_key=None, attestation=None):
    return build_bundle(
        path, name="sensors", patterns=PATTERNS, schema=SCHEMA, criteria=CRITERIA,
        provenance=_provenance(), beta=12.0, normalize="zscore",
        sign_key=sign_key, attestation=attestation,
    )


def test_build_load_round_trip(tmp_path):
    bundle = _build(tmp_path / "c.ndcoll")
    store = NeuroStore(tmp_path / "db.npz")
    mem = store.load_collection(bundle)

    assert mem.name == "sensors"
    assert mem.count == len(PATTERNS)
    assert mem.fields == ["temperature", "humidity", "pressure"]
    assert mem.normalize == "zscore"
    np.testing.assert_allclose(mem._X, PATTERNS, rtol=1e-5)
    assert mem.collection["criteria"]["set"] == "TEST-Katalog"


def test_info_reads_metadata_without_patterns(tmp_path):
    bundle = _build(tmp_path / "c.ndcoll")
    summary = info(bundle)
    assert summary["name"] == "sensors"
    assert summary["schema_fields"] == ["temperature", "humidity", "pressure"]
    assert summary["criteria"]["version"] == "1.0"
    assert summary["provenance"]["source"] == "synthetic"
    assert summary["capacity"]["status"] in {"healthy", "crowded", "saturated"}
    assert summary["attested"] is False


def test_baseline_present(tmp_path):
    manifest = read_manifest(_build(tmp_path / "c.ndcoll"))
    base = manifest["baseline"]
    assert len(base["mean"]) == 3 and len(base["std"]) == 3
    assert base["recommended_beta"] == 12.0
    assert "humidity" in base["thresholds"]


def test_missing_provenance_rejected(tmp_path):
    # Hand-craft a bundle whose manifest omits provenance → loader rejects it.
    path = tmp_path / "bad.ndcoll"
    manifest = {"name": "x", "dimension": 3, "schema": SCHEMA.to_dict(), "criteria": CRITERIA,
                "format_version": 1}
    buf = io.BytesIO()
    np.save(buf, PATTERNS)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("collection.json", json.dumps(manifest))
        zf.writestr("patterns.npy", buf.getvalue())
    store = NeuroStore(tmp_path / "db.npz")
    with pytest.raises(BundleError, match="provenance"):
        store.load_collection(path)


def test_malformed_bundle_rejected(tmp_path):
    path = tmp_path / "empty.ndcoll"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("nonsense.txt", "hi")
    with pytest.raises(BundleError):
        read_manifest(path)


def test_synthetic_provenance_asserts_no_personal_data(tmp_path):
    manifest = read_manifest(_build(tmp_path / "c.ndcoll"))
    prov = manifest["provenance"]
    assert prov["contains_personal_data"] is False
    assert "GDPR" in prov["dsgvo_basis"] or "personal data" in prov["dsgvo_basis"]


# -- signing / verification (§5) ----------------------------------------------
def test_signed_bundle_verifies_and_loads(tmp_path):
    sk, pk = signing.generate_keypair()
    bundle = _build(tmp_path / "c.ndcoll", sign_key=sk)
    result = verify_bundle(bundle, trusted_keys={pk.hex()})
    assert result.signed and result.valid and result.trusted

    store = NeuroStore(tmp_path / "db.npz")
    assert store.load_collection(bundle).count == len(PATTERNS)


def test_tampered_signed_bundle_fails_verification_and_load(tmp_path):
    sk, _ = signing.generate_keypair()
    bundle = _build(tmp_path / "c.ndcoll", sign_key=sk)

    # Rewrite the manifest while keeping the old signature → digest mismatch.
    with zipfile.ZipFile(bundle) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members["collection.json"])
    manifest["name"] = "tampered"
    tampered = tmp_path / "tampered.ndcoll"
    with zipfile.ZipFile(tampered, "w") as zf:
        zf.writestr("collection.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        zf.writestr("patterns.npy", members["patterns.npy"])
        zf.writestr("signature.json", members["signature.json"])

    assert verify_bundle(tampered).valid is False
    store = NeuroStore(tmp_path / "db.npz")
    with pytest.raises(BundleError, match="signature"):
        store.load_collection(tampered)


def test_license_seam_default_is_signature_only(tmp_path):
    # Unsigned community bundle loads (no entitlement gate); the default policy
    # only blocks an *invalid* signature.
    assert isinstance(bundle_mod, type(bundle_mod))  # module import sanity
    set_license(SignatureOnlyLicense())
    bundle = _build(tmp_path / "c.ndcoll")  # unsigned
    store = NeuroStore(tmp_path / "db.npz")
    assert store.load_collection(bundle).count == len(PATTERNS)


# -- attestation (§4) ---------------------------------------------------------
def test_attestation_signed_surfaced_and_verifiable(tmp_path):
    partner_sk, _ = signing.generate_keypair()
    attestation = sign_attestation(
        {
            "party": "DRV Domain Authority",
            "role": "auditor",
            "credential": "cert-123",
            "criteria_version": "TEST-Katalog 1.0",
            "scope": "population distribution is realistic for a valid population",
            "date": "2026-06-14",
        },
        partner_sk,
    )
    assert verify_attestation(attestation)

    bundle = _build(tmp_path / "c.ndcoll", attestation=attestation)
    summary = info(bundle)
    assert summary["attested"] is True
    assert summary["attestation"]["verified"] is True
    assert "realistic" in summary["attestation"]["scope"]


def test_tampered_attestation_rejected(tmp_path):
    partner_sk, _ = signing.generate_keypair()
    attestation = sign_attestation(
        {"party": "P", "criteria_version": "1.0", "scope": "s", "date": "2026-06-14"},
        partner_sk,
    )
    attestation["scope"] = "forged scope"  # invalidates the partner signature
    assert verify_attestation(attestation) is False
    with pytest.raises(BundleError, match="attestation"):
        _build(tmp_path / "c.ndcoll", attestation=attestation)


# -- format migration (§1/§7) -------------------------------------------------
def test_manifest_migration_identity_and_gap(tmp_path):
    manifest = read_manifest(_build(tmp_path / "c.ndcoll"))
    # Current-version manifest is returned unchanged.
    assert bundle_mod.migrate_manifest(manifest, target=bundle_mod._BUNDLE_FORMAT_VERSION)
    # A gap with no registered migration fails loudly.
    with pytest.raises(BundleError):
        bundle_mod.migrate_manifest({"format_version": 999}, target=1000)


# -- HTTP surface -------------------------------------------------------------
def test_collection_http_load_and_info(tmp_path):
    bundle = _build(tmp_path / "c.ndcoll")
    settings = Settings(
        data_file=str(tmp_path / "db.npz"), autosave_interval=0.0, allow_anonymous=True
    )
    with TestClient(create_app(settings)) as client:
        resp = client.post("/v1/collections/load", json={"path": str(bundle)})
        assert resp.status_code == 201, resp.text
        assert resp.json()["collection"]["criteria"]["set"] == "TEST-Katalog"

        got = client.get("/v1/collections/sensors")
        assert got.status_code == 200
        assert got.json()["provenance"]["source"] == "synthetic"

        # A loaded collection scores like any memory.
        a = client.post("/v1/memories/sensors/anomaly", json={"query": [20, 95, 1013]})
        assert a.status_code == 200
        assert a.json()["fields"][0]["name"] == "humidity"

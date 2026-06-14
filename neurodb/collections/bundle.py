"""The open, versioned, integrity-checked **collection bundle** format.

A *collection* is a portable reference population — the "normal" records every
incoming record is scored against — packaged so it loads into the engine as a
ready-to-score memory (patterns + baseline + schema + metadata). The format is
open and collection-agnostic: anyone can build one. *Specific* collections (e.g.
a DRV-Prüfkatalog collection) are licensed content, signed by the publisher and
distributed only to customers — the format never hard-codes any domain.

Layout — a ``.ndcoll`` zip with three members:

* ``collection.json`` — the manifest: format_version, name, dimension, beta,
  normalize, schema, baseline, criteria, provenance, and optional attestation.
* ``patterns.npy``     — the reference matrix in the engine's float32 layout.
* ``signature.json``   — optional Ed25519 signature over the other two members.

The manifest is migratable across ``format_version`` (mirroring the store's
manifest migrations) so a bundle built today still loads after the format evolves.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import signing

# Bundle format version — independent of the store's _MANIFEST_VERSION. Bump on
# any bundle-format change and register a migration in _BUNDLE_MIGRATIONS.
_BUNDLE_FORMAT_VERSION = 1

_COLLECTION_ENTRY = "collection.json"
_PATTERNS_ENTRY = "patterns.npy"
_SIGNATURE_ENTRY = "signature.json"

_VALID_SOURCES = ("synthetic", "anonymized", "mixed")

# Blocks a well-formed manifest must carry (so info()/load fail with a clear
# BundleError rather than a KeyError on a structurally-incomplete bundle).
_REQUIRED_MANIFEST_KEYS = (
    "format_version", "name", "dimension", "schema", "baseline", "criteria", "provenance",
)


class BundleError(Exception):
    """A bundle is malformed, unverifiable, or fails a required-block check."""


# -- manifest migration (mirrors store.migrations; collection-agnostic) -------
_BUNDLE_MIGRATIONS: dict[int, Any] = {}


def migrate_manifest(manifest: dict[str, Any], *, target: int) -> dict[str, Any]:
    """Bring a bundle manifest forward to ``target`` format version."""

    version = int(manifest.get("format_version", 0))
    while version < target:
        fn = _BUNDLE_MIGRATIONS.get(version)
        if fn is None:
            raise BundleError(
                f"no bundle migration registered for format v{version} -> v{version + 1}"
            )
        manifest = fn(manifest)
        version = int(manifest["format_version"])
    return manifest


# -- provenance / attestation validation (§3, §4) -----------------------------
def validate_provenance(prov: dict[str, Any]) -> None:
    """Reject a missing or malformed provenance block (no unprovenanced data)."""

    if not isinstance(prov, dict):
        raise BundleError("provenance block is missing or not an object")
    source = prov.get("source")
    if source not in _VALID_SOURCES:
        raise BundleError(f"provenance.source must be one of {_VALID_SOURCES}, got {source!r}")
    for required in ("build_date", "builder", "dsgvo_basis"):
        if not prov.get(required):
            raise BundleError(f"provenance.{required} is required")
    if source == "synthetic" and prov.get("contains_personal_data") is not False:
        raise BundleError(
            "synthetic provenance must assert contains_personal_data=false "
            "(synthetic data is generated, not collected)"
        )


def validate_attestation(att: dict[str, Any]) -> None:
    """Validate an attestation block's shape and its independent signature."""

    if not isinstance(att, dict):
        raise BundleError("attestation must be an object")
    for required in ("party", "criteria_version", "scope", "public_key", "signature"):
        if not att.get(required):
            raise BundleError(f"attestation.{required} is required")
    if not verify_attestation(att):
        raise BundleError("attestation signature does not verify")


def _attestation_signing_bytes(att: dict[str, Any]) -> bytes:
    """Canonical bytes the attester signs — the block minus the signature."""

    payload = {k: v for k, v in att.items() if k != "signature"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_attestation(att: dict[str, Any], seed: bytes) -> dict[str, Any]:
    """Return ``att`` with ``public_key``/``signature`` filled by the partner key."""

    pub = signing.public_key(seed)
    att = dict(att)
    att["public_key"] = pub.hex()
    att["signature"] = signing.sign(_attestation_signing_bytes(att), seed, pub).hex()
    return att


def verify_attestation(att: dict[str, Any]) -> bool:
    """Independently verify the attester's signature over the attestation block."""

    try:
        pub = bytes.fromhex(att["public_key"])
        sig = bytes.fromhex(att["signature"])
    except (KeyError, ValueError):
        return False
    return signing.verify(_attestation_signing_bytes(att), sig, pub)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Enforce required blocks. A collection without provenance is rejected."""

    for required in ("name", "dimension", "schema", "criteria", "provenance"):
        if required not in manifest:
            raise BundleError(f"manifest is missing required block: {required!r}")
    validate_provenance(manifest["provenance"])
    if manifest.get("attestation"):
        validate_attestation(manifest["attestation"])


# -- baseline (§1: precomputed zscore stats + recommended beta + capacity) -----
def compute_baseline(
    patterns: np.ndarray,
    fields: list[str] | None,
    beta: float,
    normalize: str,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Derive the baseline a buyer needs to score on day one: per-field mean/std,
    a recommended beta, per-field thresholds, and a build-time capacity report so
    they can see the collection is healthy (not saturated)."""

    # Local import avoids a module import cycle (store imports collections indirectly).
    from ..store import _NORM_EPS, Memory

    mem = Memory("baseline", patterns.shape[1], beta, fields, normalize)
    mem.write([{"vector": row.tolist()} for row in patterns])
    names = fields or [f"field_{i}" for i in range(patterns.shape[1])]
    if thresholds is None:
        thresholds = {name: 3.0 for name in names}
    # Floor std with the engine's _NORM_EPS so the published baseline matches how
    # the loaded memory actually normalizes (a constant column → eps, not 0.0).
    std = np.maximum(np.std(patterns, axis=0), _NORM_EPS)
    return {
        "mean": np.mean(patterns, axis=0).astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "recommended_beta": float(beta),
        "thresholds": thresholds,
        "capacity": mem.capacity_compact(),
    }


# -- build / read --------------------------------------------------------------
@dataclass
class CollectionSchema:
    """Self-describing field layout so a user maps their table correctly."""

    fields: list[dict[str, Any]]  # [{"name","type","unit"}, ...]
    dimension: int
    column_map: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "dimension": self.dimension,
            "column_map": self.column_map or {f["name"]: i for i, f in enumerate(self.fields)},
        }


def _digest(collection_bytes: bytes, patterns_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(collection_bytes)
    h.update(patterns_bytes)
    return h.hexdigest()


def _patterns_bytes(patterns: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, patterns.astype(np.float32, copy=False))
    return buf.getvalue()


def build_bundle(
    path: str | Path,
    *,
    name: str,
    patterns: np.ndarray,
    schema: CollectionSchema,
    criteria: dict[str, Any],
    provenance: dict[str, Any],
    beta: float = 8.0,
    normalize: str = "zscore",
    thresholds: dict[str, float] | None = None,
    attestation: dict[str, Any] | None = None,
    sign_key: bytes | None = None,
) -> Path:
    """Build, validate, and (optionally) sign a collection bundle at ``path``."""

    patterns = np.asarray(patterns, dtype=np.float32)
    if patterns.ndim != 2:
        raise BundleError("patterns must be a 2-D (N, D) matrix")
    field_names = [f["name"] for f in schema.fields]
    provenance = {**provenance, "format_version": _BUNDLE_FORMAT_VERSION}
    manifest: dict[str, Any] = {
        "format_version": _BUNDLE_FORMAT_VERSION,
        "name": name,
        "dimension": int(patterns.shape[1]),
        "beta": float(beta),
        "normalize": normalize,
        "schema": schema.to_dict(),
        "baseline": compute_baseline(patterns, field_names, beta, normalize, thresholds),
        "criteria": criteria,
        "provenance": provenance,
        "attestation": attestation,
    }
    validate_manifest(manifest)

    collection_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    patterns_bytes = _patterns_bytes(patterns)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_COLLECTION_ENTRY, collection_bytes)
        zf.writestr(_PATTERNS_ENTRY, patterns_bytes)
        if sign_key is not None:
            pub = signing.public_key(sign_key)
            digest = _digest(collection_bytes, patterns_bytes)
            sig = signing.sign(bytes.fromhex(digest), sign_key, pub)
            signature = {
                "algorithm": "ed25519",
                "digest": "sha256",
                "public_key": pub.hex(),
                "signature": sig.hex(),
            }
            zf.writestr(_SIGNATURE_ENTRY, json.dumps(signature).encode("utf-8"))
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read, version-check, migrate, and structurally validate the manifest
    **without** materializing the patterns."""

    with zipfile.ZipFile(path) as zf:
        if _COLLECTION_ENTRY not in zf.namelist():
            raise BundleError(f"bundle is missing {_COLLECTION_ENTRY}")
        manifest = json.loads(zf.read(_COLLECTION_ENTRY))
    version = int(manifest.get("format_version", 0))
    if version > _BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"bundle was written by a newer NeuroDB (format v{version}); this "
            f"build supports up to v{_BUNDLE_FORMAT_VERSION}. Upgrade NeuroDB."
        )
    manifest = migrate_manifest(manifest, target=_BUNDLE_FORMAT_VERSION)
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise BundleError(f"bundle manifest is missing required blocks: {missing}")
    return manifest


def read_patterns(path: str | Path) -> np.ndarray:
    with zipfile.ZipFile(path) as zf:
        if _PATTERNS_ENTRY not in zf.namelist():
            raise BundleError(f"bundle is missing {_PATTERNS_ENTRY}")
        return np.load(io.BytesIO(zf.read(_PATTERNS_ENTRY)), allow_pickle=False)


@dataclass
class VerificationResult:
    signed: bool
    valid: bool
    public_key: str | None
    trusted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "signed": self.signed,
            "valid": self.valid,
            "public_key": self.public_key,
            "trusted": self.trusted,
        }


def verify_bundle(
    path: str | Path, trusted_keys: set[str] | None = None
) -> VerificationResult:
    """Verify a bundle's signature (if present) over its exact stored bytes.

    ``valid`` is True for an unsigned bundle (nothing to forge) and for a signed
    bundle whose signature checks out. ``trusted`` is True only when the signer's
    public key is in ``trusted_keys`` (used to assert "signed by the publisher").
    """

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if _SIGNATURE_ENTRY not in names:
            return VerificationResult(signed=False, valid=True, public_key=None, trusted=False)
        # A signed bundle missing a signed member can't verify — report invalid
        # rather than crashing with a KeyError on the read below.
        if _COLLECTION_ENTRY not in names or _PATTERNS_ENTRY not in names:
            return VerificationResult(signed=True, valid=False, public_key=None, trusted=False)
        collection_bytes = zf.read(_COLLECTION_ENTRY)
        patterns_bytes = zf.read(_PATTERNS_ENTRY)
        sig_info = json.loads(zf.read(_SIGNATURE_ENTRY))

    try:
        pub = bytes.fromhex(sig_info["public_key"])
        sig = bytes.fromhex(sig_info["signature"])
    except (KeyError, ValueError):
        return VerificationResult(signed=True, valid=False, public_key=None, trusted=False)

    digest = _digest(collection_bytes, patterns_bytes)
    valid = signing.verify(bytes.fromhex(digest), sig, pub)
    trusted = valid and trusted_keys is not None and sig_info["public_key"] in trusted_keys
    return VerificationResult(
        signed=True, valid=valid, public_key=sig_info["public_key"], trusted=trusted
    )


def info(path: str | Path, trusted_keys: set[str] | None = None) -> dict[str, Any]:
    """A buyer-facing summary: schema, criteria, provenance, attestation, capacity
    and signature status — read **without** materializing the patterns."""

    manifest = read_manifest(path)
    verification = verify_bundle(path, trusted_keys)
    att = manifest.get("attestation")
    return {
        "name": manifest["name"],
        "format_version": manifest["format_version"],
        "dimension": manifest["dimension"],
        "normalize": manifest.get("normalize"),
        "criteria": manifest.get("criteria"),
        "schema_fields": [f["name"] for f in manifest["schema"]["fields"]],
        "capacity": manifest["baseline"].get("capacity"),
        "provenance": {
            "source": manifest["provenance"].get("source"),
            "dsgvo_basis": manifest["provenance"].get("dsgvo_basis"),
            "build_date": manifest["provenance"].get("build_date"),
            "builder": manifest["provenance"].get("builder"),
            "summary": manifest["provenance"].get("summary"),
        },
        "attested": bool(att),
        "attestation": (
            {
                "party": att.get("party"),
                "criteria_version": att.get("criteria_version"),
                "scope": att.get("scope"),
                "verified": verify_attestation(att),
            }
            if att
            else None
        ),
        "signature": verification.to_dict(),
    }


def synthetic_provenance(
    builder: str,
    *,
    generator: str,
    generation_params: dict[str, Any] | None = None,
    calibration_source: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Build a valid synthetic provenance block (the clean v1 legal position)."""

    return {
        "source": "synthetic",
        "contains_personal_data": False,
        "dsgvo_basis": (
            "Collection is fully synthetic (generated, not collected) and contains "
            "no personal data within the meaning of GDPR Art. 4(1)."
        ),
        "generator": generator,
        "generation_params": generation_params or {},
        "calibration_source": calibration_source,
        "build_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": builder,
        "summary": summary,
    }

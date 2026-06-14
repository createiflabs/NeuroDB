"""Collection bundles: the open format and tooling for portable, signed,
domain-validated reference populations.

The format, loader, signing/verification, provenance/attestation mechanisms and
the synthesis framework are all open (MIT). *Specific* collections (e.g. a
DRV-Prüfkatalog collection) are licensed content built privately with this
tooling and never committed here.
"""

from __future__ import annotations

from .bundle import (
    BundleError,
    CollectionSchema,
    VerificationResult,
    build_bundle,
    info,
    read_manifest,
    read_patterns,
    sign_attestation,
    synthetic_provenance,
    verify_attestation,
    verify_bundle,
)
from .license import CollectionLicense, SignatureOnlyLicense, get_license, set_license

__all__ = [
    "BundleError",
    "CollectionSchema",
    "VerificationResult",
    "build_bundle",
    "info",
    "read_manifest",
    "read_patterns",
    "verify_bundle",
    "verify_attestation",
    "sign_attestation",
    "synthetic_provenance",
    "CollectionLicense",
    "SignatureOnlyLicense",
    "get_license",
    "set_license",
]

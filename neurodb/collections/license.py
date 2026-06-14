"""The licensing seam — open mechanism, commercial scarcity.

The deliberate design choice (see the spec): the open core loads any *validly
signed* (or unsigned community) collection. Licensing is enforced at
**distribution** — you only hand the signed bundle to paying customers — not by
crippling the loader. This keeps the open core uncrippled while the scarcity is
the artifact itself.

``CollectionLicense`` is the seam a future managed/commercial edition can plug
into to add load-time entitlement checks without forking. The open default,
:class:`SignatureOnlyLicense`, verifies the signature and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .bundle import BundleError

if TYPE_CHECKING:
    from .bundle import VerificationResult


@runtime_checkable
class CollectionLicense(Protocol):
    """Decides whether a verified bundle is allowed to load."""

    def check(self, manifest: dict, verification: VerificationResult) -> None:
        """Raise :class:`BundleError` to refuse the load; return to allow."""
        ...


class SignatureOnlyLicense:
    """Open default: a signed bundle must verify; entitlement is never gated."""

    def check(self, manifest: dict, verification: VerificationResult) -> None:
        if verification.signed and not verification.valid:
            raise BundleError("bundle signature is invalid; refusing to load")


_license: CollectionLicense = SignatureOnlyLicense()


def set_license(license_obj: CollectionLicense) -> None:
    """Register the active license policy (e.g. a commercial entitlement check)."""

    global _license
    _license = license_obj


def get_license() -> CollectionLicense:
    return _license

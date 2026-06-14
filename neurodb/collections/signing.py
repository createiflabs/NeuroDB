"""Ed25519 signing for collection bundles — dependency-free (RFC 8032).

Asymmetric signatures give bundles two properties the format needs: *integrity*
(tamper-evidence) and *authenticity* (who built it). A community user self-signs
their own collections; licensed collections are signed by the publisher's key,
which any open-core install can verify.

This is a faithful transcription of the RFC 8032 reference Ed25519, using
Python's fast built-in modular ``pow``. It is intentionally pure-stdlib so the
verification path ships in the open core with no native dependency. Keys are
32-byte seeds (private) and 32-byte points (public); signatures are 64 bytes.
"""

from __future__ import annotations

import hashlib
import secrets

_b = 256
_q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _inv(x: int) -> int:
    return pow(x, _q - 2, _q)


_d = -121665 * _inv(121666) % _q
_I = pow(2, (_q - 1) // 4, _q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = pow(xx, (_q + 3) // 8, _q)
    if (x * x - xx) % _q != 0:
        x = (x * _I) % _q
    if x % 2 != 0:
        x = _q - x
    return x


_By = 4 * _inv(5) % _q
_Bx = _xrecover(_By)
_B = (_Bx % _q, _By % _q)


def _edwards_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _d * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _d * x1 * x2 * y1 * y2)
    return (x3 % _q, y3 % _q)


def _scalarmult(p: tuple[int, int], e: int) -> tuple[int, int]:
    # Iterative double-and-add (avoids deep recursion for 256-bit scalars).
    result = (0, 1)
    addend = p
    while e > 0:
        if e & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        e >>= 1
    return result


def _bit(h: bytes, i: int) -> int:
    return (h[i // 8] >> (i % 8)) & 1


def _encodeint(y: int) -> bytes:
    return y.to_bytes(_b // 8, "little")


def _encodepoint(p: tuple[int, int]) -> bytes:
    x, y = p
    bits = [(y >> i) & 1 for i in range(_b - 1)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(_b // 8))


def _secret_scalar(h: bytes) -> int:
    return 2 ** (_b - 2) + sum(2**i * _bit(h, i) for i in range(3, _b - 2))


def _hint(m: bytes) -> int:
    h = _sha512(m)
    return sum(2**i * _bit(h, i) for i in range(2 * _b))


def public_key(seed: bytes) -> bytes:
    """Derive the 32-byte public key from a 32-byte private seed."""

    if len(seed) != 32:
        raise ValueError("seed must be 32 bytes")
    h = _sha512(seed)
    return _encodepoint(_scalarmult(_B, _secret_scalar(h)))


def generate_keypair() -> tuple[bytes, bytes]:
    """Return ``(private_seed, public_key)`` — 32 bytes each."""

    seed = secrets.token_bytes(32)
    return seed, public_key(seed)


def sign(message: bytes, seed: bytes, pub: bytes | None = None) -> bytes:
    """Sign ``message`` with the 32-byte private ``seed``. Returns 64 bytes."""

    if pub is None:
        pub = public_key(seed)
    h = _sha512(seed)
    a = _secret_scalar(h)
    r = _hint(h[_b // 8 : _b // 4] + message)
    R = _scalarmult(_B, r)
    S = (r + _hint(_encodepoint(R) + pub + message) * a) % _L
    return _encodepoint(R) + _encodeint(S)


def _isoncurve(p: tuple[int, int]) -> bool:
    x, y = p
    return (-x * x + y * y - 1 - _d * x * x * y * y) % _q == 0


def _decodepoint(s: bytes) -> tuple[int, int]:
    y = int.from_bytes(s, "little") & ((1 << (_b - 1)) - 1)
    x = _xrecover(y)
    if x & 1 != _bit(s, _b - 1):
        x = _q - x
    p = (x, y)
    if not _isoncurve(p):
        raise ValueError("point is not on the curve")
    return p


def verify(message: bytes, signature: bytes, pub: bytes) -> bool:
    """Return True iff ``signature`` is a valid Ed25519 signature of ``message``
    under public key ``pub``. Never raises on a bad signature — returns False."""

    try:
        if len(signature) != 64 or len(pub) != 32:
            return False
        R = _decodepoint(signature[: _b // 8])
        A = _decodepoint(pub)
        S = int.from_bytes(signature[_b // 8 : _b // 4], "little")
        h = _hint(_encodepoint(R) + pub + message)
        return _scalarmult(_B, S) == _edwards_add(R, _scalarmult(A, h))
    except (ValueError, IndexError):
        return False

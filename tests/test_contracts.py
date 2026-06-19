"""Production-readiness contract tests (several mandated by name in the goal doc).

These pin behaviour that is part of NeuroDB's correctness/safety contract:

* input validation rejects (never coerces) bad vectors — Tier 5.2,
* memories are isolated from one another — Tier 4.4,
* the text encoder is deterministic across process restarts and independent of
  ``PYTHONHASHSEED`` — Tier 3.4 (the old PoC bug was an encoder that consumed
  shared RNG state; this proves the current SHA-style hashing has no such hazard).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neurodb.store import Memory, NotFoundError, StoreError

_REPO_ROOT = Path(__file__).resolve().parents[1]


# -- Tier 5.2: reject, don't coerce ---------------------------------------
def test_reject_dimension_mismatch_and_nonfinite_vectors():
    mem = Memory("m", 3)

    # Wrong dimension is rejected on write (not silently padded/truncated).
    with pytest.raises(StoreError):
        mem.write([{"vector": [1.0, 2.0]}])
    with pytest.raises(StoreError):
        mem.write([{"vector": [1.0, 2.0, 3.0, 4.0]}])
    # NaN / Inf are rejected on write.
    with pytest.raises(StoreError):
        mem.write([{"vector": [1.0, 2.0, float("nan")]}])
    with pytest.raises(StoreError):
        mem.write([{"vector": [1.0, float("inf"), 3.0]}])
    # A rejected write leaves the memory empty (nothing partially applied).
    assert mem.count == 0

    # The read/query paths reject the same way.
    mem.write([{"vector": [1.0, 2.0, 3.0]}])
    with pytest.raises(StoreError):
        mem.search([1.0, 2.0])
    with pytest.raises(StoreError):
        mem.complete([1.0, 2.0, float("inf")])
    with pytest.raises(StoreError):
        mem.anomaly([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(StoreError):
        mem.search_batch([[1.0, 2.0, float("nan")]])


# -- Tier 4.4: namespace isolation ----------------------------------------
def test_namespace_isolation(store_factory):
    store = store_factory()
    a = store.create_memory("a", 3)
    b = store.create_memory("b", 5)

    # Independent locks: a long write-hold on one must not block the other.
    assert a._lock is not b._lock

    a.write([{"vector": [1.0, 2.0, 3.0]} for _ in range(10)])
    # b is untouched by a's writes (data, dimension, matrix all independent).
    assert b.count == 0
    assert b.dimension == 5
    assert b._X.shape == (0, 5)

    # Persisting and reloading keeps the two namespaces independent.
    store.save()
    reloaded = store_factory()  # same tmp_path data file
    assert reloaded.get_memory("a").count == 10
    assert reloaded.get_memory("b").count == 0

    # Deleting one leaves the other intact.
    store.delete_memory("a")
    b.write([{"vector": [1.0, 2.0, 3.0, 4.0, 5.0]}])
    assert store.get_memory("b").count == 1
    with pytest.raises(NotFoundError):
        store.get_memory("a")


# -- Tier 3.4: deterministic encoding across restarts ---------------------
def _embed_digest(text: str, dim: int, hashseed: str) -> str:
    """Embed ``text`` in a *fresh* interpreter under a given PYTHONHASHSEED.

    The subprocess gets a tightly-controlled environment (only what the OS needs
    to launch Python and import an installed package, plus the hash seed), so no
    app-level env from a concurrently-running test can influence the result.
    """

    code = (
        "import hashlib;"
        "from neurodb.embedding import embed_text;"
        f"v = embed_text({text!r}, {dim});"
        "print(hashlib.sha256(v.tobytes()).hexdigest())"
    )
    keep = (
        "PATH", "PATHEXT", "SYSTEMROOT", "SystemRoot", "WINDIR",
        "TEMP", "TMP", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
    )
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONHASHSEED"] = hashseed
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"subprocess failed (seed={hashseed}): {proc.stderr}"
    return proc.stdout.strip()


def test_unseen_category_deterministic_across_restart():
    text = "a never-before-seen token xyzzy q7 plover"
    # Different process invocations AND different hash seeds must agree exactly:
    # no reliance on Python's salted hash() or any per-process RNG state.
    d0 = _embed_digest(text, 256, "0")
    d1 = _embed_digest(text, 256, "12345")
    d2 = _embed_digest(text, 256, "random")
    assert d0 == d1 == d2

    # And it matches the in-process embedding (no hidden state, order-independent).
    from neurodb.embedding import embed_text

    here = hashlib.sha256(embed_text(text, 256).tobytes()).hexdigest()
    assert here == d0

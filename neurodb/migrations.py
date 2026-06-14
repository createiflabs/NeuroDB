"""Explicit, ordered manifest migrations for the single-file store.

The store stamps every save with a manifest ``version``. On load, a file older
than the current version is brought forward by applying each registered migration
in sequence **before** any :class:`Memory` is constructed. The policy is "every
version transition has an explicit, tested function" — not "we assume new keys
are additive and hope old files still parse."

Migrations are **read-time and non-destructive**: they transform the in-memory
manifest/arrays only. The upgraded data is written back on the next normal save
(which stamps the current version); the source file on disk is never mutated
during load.

To add a format change: bump ``_MANIFEST_VERSION`` in ``store.py`` and register a
``@migration(from_version=N)`` that returns the manifest/arrays at version N+1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger("neurodb.migrations")

Manifest = dict[str, Any]
Arrays = dict[str, np.ndarray]
Migration = Callable[[Manifest, Arrays], tuple[Manifest, Arrays]]

# from_version -> function producing the manifest/arrays at from_version + 1.
_MIGRATIONS: dict[int, Migration] = {}


class MigrationError(Exception):
    """No migration is registered for a needed version transition."""


def migration(from_version: int) -> Callable[[Migration], Migration]:
    def decorator(fn: Migration) -> Migration:
        if from_version in _MIGRATIONS:  # pragma: no cover - programming error
            raise MigrationError(f"duplicate migration registered for v{from_version}")
        _MIGRATIONS[from_version] = fn
        return fn

    return decorator


@migration(from_version=1)
def _v1_to_v2(manifest: Manifest, arrays: Arrays) -> tuple[Manifest, Arrays]:
    """v1 → v2: identity.

    v1 manifests were forward-compatible by additive keys (e.g. ``normalize``
    resolved via ``.get`` defaults). No structural change is needed; this
    migration exists so the chain is complete and the upgrade path is exercised
    rather than assumed.
    """

    manifest = dict(manifest)
    manifest["version"] = 2
    return manifest, arrays


def migrate(
    manifest: Manifest, arrays: Arrays, *, target_version: int
) -> tuple[Manifest, Arrays]:
    """Apply migrations until ``manifest`` reaches ``target_version``.

    Assumes the caller has already verified ``version <= target_version`` (a
    newer file is rejected loudly by the loader, not migrated). Raises
    :class:`MigrationError` if a needed transition has no registered function.
    """

    version = int(manifest["version"])
    while version < target_version:
        fn = _MIGRATIONS.get(version)
        if fn is None:
            raise MigrationError(
                f"no migration registered for manifest v{version} -> v{version + 1}"
            )
        logger.info("migrating manifest v%d -> v%d", version, version + 1)
        manifest, arrays = fn(manifest, arrays)
        new_version = int(manifest["version"])
        if new_version != version + 1:  # pragma: no cover - programming error
            raise MigrationError(
                f"migration from v{version} stamped v{new_version}, expected v{version + 1}"
            )
        version = new_version
    return manifest, arrays

"""Consistent off-box backup and validated restore for the single-file store.

Backup reuses the store's existing atomic-save discipline: it forces a save under
the store lock, then copies the resulting ``.npz`` — so the snapshot is
point-in-time consistent (matrix rows always agree with ids/metadata) even if a
write lands concurrently. Restore validates the source *parses* before it ever
touches the live file, and preserves the current file as ``*.pre-restore`` so a
bad restore can never destroy data.

Cloud/remote targets are deliberately out of scope: ``backup`` writes a local
file; ship it elsewhere with your own tool (``aws s3 cp``, ``rclone``, a sidecar)
so no cloud SDK leaks into core.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from .store import NeuroStore

logger = logging.getLogger("neurodb.backup")


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _resolve_dest(dest: Path) -> Path:
    """An existing directory → a timestamped file inside it; otherwise ``dest``
    is treated as the target file path."""

    if dest.is_dir():
        return dest / f"neurodb-backup-{_timestamp()}.npz"
    return dest


def backup_store(store: NeuroStore, dest: str | Path) -> Path:
    """Write a consistent snapshot of ``store`` to ``dest`` and return its path.

    Taken under the store lock after a forced save, so the copied file is exactly
    the all-or-nothing image ``_save_locked`` produces — never a torn read.
    """

    target = _resolve_dest(Path(dest))
    with store._lock:
        # Force a current, fsync-durable on-disk image under the lock, then copy
        # it before releasing the lock so no write can interleave.
        store.save_all()
        src = store.data_file
        if not src.exists():  # pragma: no cover - save_all always writes a file
            raise FileNotFoundError(f"no data file to back up at {src}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    logger.info("backup written to %s", target)
    return target


def restore_file(src: str | Path, data_file: str | Path) -> Path | None:
    """Replace ``data_file`` with ``src`` after validating ``src`` loads.

    Returns the path of the preserved previous live file (``*.pre-restore``), or
    ``None`` if there was no live file. Raises before swapping if ``src`` is
    missing, corrupt, or written by an incompatible (newer) NeuroDB — the live
    file is never touched on failure.
    """

    src = Path(src)
    data_file = Path(data_file)
    if not src.exists():
        raise FileNotFoundError(f"backup source {src} does not exist")

    # Validate by loading: raises StoreError on corrupt / incompatible files.
    # Read-only — load never mutates the source, and migration is in-memory only.
    NeuroStore(src, fail_on_corrupt_load=True)

    data_file.parent.mkdir(parents=True, exist_ok=True)
    preserved: Path | None = None
    if data_file.exists():
        preserved = data_file.with_name(data_file.name + ".pre-restore")
        shutil.copy2(data_file, preserved)
        logger.info("preserved current data file to %s", preserved)

    # Stage into the destination dir then atomically replace, so an interrupted
    # restore cannot leave a half-written live file.
    tmp = data_file.with_suffix(data_file.suffix + ".restore-tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, data_file)
    logger.info("restored %s from %s", data_file, src)
    return preserved

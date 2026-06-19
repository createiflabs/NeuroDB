"""Logical export / import to a stable, NumPy-version-independent format (JSONL).

The ``.npz`` snapshot is compact but tied to NumPy's on-disk array format, which
is a forward-compatibility risk across NumPy versions. A *logical* JSONL export
keeps data portable: one header line with the memory's config, then one plain
JSON object per pattern (``id``, ``vector`` as a list, ``metadata``). It
round-trips losslessly at float32 precision and depends on nothing but JSON, so
data is never trapped in a version-specific binary format
(``export_import_roundtrip_across_numpy_versions``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .store import Memory, NeuroStore

_HEADER_KEY = "__neurodb_memory__"


def export_memory_jsonl(mem: Memory, fh: TextIO) -> int:
    """Write ``mem`` to ``fh`` as JSONL (header + one row per pattern).

    Uses a consistent snapshot, so the export is coherent even if the memory is
    written concurrently. Returns the number of pattern rows written.
    """

    snap = mem.snapshot()
    header = {
        _HEADER_KEY: {
            "name": snap.name,
            "dimension": snap.dimension,
            "beta": snap.beta,
            "fields": snap.fields,
            "normalize": snap.normalize,
        }
    }
    fh.write(json.dumps(header) + "\n")
    written = 0
    for _id, meta, row in zip(snap.ids, snap.metadata, snap.matrix, strict=True):
        fh.write(
            json.dumps({"id": _id, "vector": [float(x) for x in row], "metadata": meta})
            + "\n"
        )
        written += 1
    return written


def import_memory_jsonl(store: NeuroStore, fh: TextIO, name: str | None = None) -> Memory:
    """Read a JSONL export from ``fh`` and create a memory in ``store``.

    ``name`` overrides the memory name from the header (e.g. to avoid a clash).
    Patterns are written in batches so a large export does not buffer in memory.
    """

    lines = (line for line in (raw.strip() for raw in fh) if line)
    try:
        header = json.loads(next(lines))
    except StopIteration:
        raise ValueError("empty export: no header line.") from None
    cfg = header.get(_HEADER_KEY)
    if not isinstance(cfg, dict):
        raise ValueError(f"export is missing its {_HEADER_KEY!r} header line.")

    mem = store.create_memory(
        name or cfg["name"],
        int(cfg["dimension"]),
        float(cfg.get("beta", 8.0)),
        cfg.get("fields"),
        cfg.get("normalize"),
    )

    batch: list[dict] = []
    for line in lines:
        rec = json.loads(line)
        batch.append(
            {"id": rec.get("id"), "vector": rec["vector"], "metadata": rec.get("metadata") or {}}
        )
        if len(batch) >= 1000:
            mem.write(batch)
            batch = []
    if batch:
        mem.write(batch)
    return mem

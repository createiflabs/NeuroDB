"""Dataset validation built on NeuroDB's per-field anomaly detection.

NeuroDB already answers "how anomalous is *this* row?" via the ``/anomaly``
endpoints. This module composes that primitive into "is *this dataset* clean?":
it streams records through :meth:`Memory.anomaly_batch`, flags any field whose
standardized deviation exceeds a threshold, and aggregates the outcome into a
:class:`ValidationReport`.

It is pure client-side composition — no new server endpoint, stdlib only — so it
honours NeuroDB's lean ethos. The report is plain data (`to_dict`) so it drops
cleanly into the Great Expectations / Airflow / Dagster adapters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from . import telemetry

if TYPE_CHECKING:  # avoid a runtime import cycle with client.py
    from .client import Memory

# Per-write/anomaly batch cap enforced by the server (neurodb/models.py:BULK_MAX).
DEFAULT_BATCH_SIZE = 1000
# A z-deviation of 3 standard deviations is the conventional outlier line.
DEFAULT_THRESHOLD = 3.0


@dataclass
class FieldResult:
    """One field's deviation for a single record."""

    name: str
    deviation: float
    passed: bool


@dataclass
class RecordResult:
    """The validation outcome for a single record."""

    id: str | None
    passed: bool
    max_deviation: float
    fields: list[FieldResult] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated result of validating a dataset against a memory."""

    memory: str
    threshold: float
    total: int
    passed: int
    failed: int
    pass_rate: float
    records: list[RecordResult] = field(default_factory=list)
    field_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no record failed validation."""

        return self.failed == 0

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"NeuroDB validation [{self.memory}]: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.1%}), {self.failed} failed, threshold={self.threshold}"
        )

    def __str__(self) -> str:
        return self.summary()


def _metric(field_obj: dict[str, Any]) -> float:
    """Deviation used for thresholding: prefer the standardized ``z_deviation``
    (cross-field comparable), fall back to the raw ``deviation``."""

    value = field_obj.get("z_deviation")
    if value is None:
        value = field_obj.get("deviation", 0.0)
    return abs(float(value))


def run_validation(
    memory: Memory,
    records: Iterable[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    fields: Sequence[str] | None = None,
    beta: float | None = None,
    filter: dict[str, Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ValidationReport:
    """Validate ``records`` against ``memory`` and return a :class:`ValidationReport`.

    Each record is scored with :meth:`Memory.anomaly_batch`; a field fails when
    its (standardized) deviation exceeds ``threshold``. ``records`` may be plain
    vectors or ``{"vector": [...], "id": ...}`` dicts. ``fields`` restricts which
    field names are considered (default: all). Emits one telemetry event.
    """

    # Reuse the client's item-coercion so vectors/dicts behave like elsewhere.
    from .client import _as_items

    items = _as_items(records)
    only = set(fields) if fields is not None else None

    # Score *every* field, not just the anomaly endpoint's default top-5 by
    # deviation: otherwise field_stats undercounts and a `fields=` filter can
    # silently miss a breach that ranks below five larger (in-tolerance)
    # deviations. (The server caps top_k at 1000; collections with more fields
    # than that are not the structured-record target of validate.)
    dim = (getattr(memory, "info", None) or {}).get("dimension")
    top_k = min(dim or 1000, 1000)

    record_results: list[RecordResult] = []
    stats: dict[str, dict[str, float]] = {}

    for start in range(0, len(items), max(1, batch_size)):
        chunk = items[start : start + max(1, batch_size)]
        if not chunk:
            continue
        resp = memory.anomaly_batch(chunk, beta=beta, top_k=top_k, filter=filter)
        for result in resp.get("results", []):
            record_results.append(
                _score_record(result, threshold, only, stats)
            )

    total = len(record_results)
    passed = sum(1 for r in record_results if r.passed)
    failed = total - passed
    pass_rate = (passed / total) if total else 1.0

    # Finalize per-field aggregates (mean deviation across all records seen).
    for agg in stats.values():
        seen = agg.pop("_seen", 0.0)
        agg["mean_deviation"] = (agg["mean_deviation"] / seen) if seen else 0.0

    report = ValidationReport(
        memory=memory.name,
        threshold=threshold,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        records=record_results,
        field_stats=stats,
    )

    telemetry.record(
        telemetry.TelemetryEvent(
            event="validate",
            memory=memory.name,
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            threshold=threshold,
        )
    )
    return report


def _score_record(
    result: dict[str, Any],
    threshold: float,
    only: set[str] | None,
    stats: dict[str, dict[str, float]],
) -> RecordResult:
    """Turn one anomaly result into a :class:`RecordResult` and fold its fields
    into the running ``stats`` aggregates."""

    field_results: list[FieldResult] = []
    record_passed = True
    max_dev = 0.0

    for idx, field_obj in enumerate(result.get("fields", [])):
        name = field_obj.get("name") or f"field_{field_obj.get('index', idx)}"
        if only is not None and name not in only:
            continue
        dev = _metric(field_obj)
        field_passed = dev <= threshold
        record_passed = record_passed and field_passed
        max_dev = max(max_dev, dev)
        field_results.append(FieldResult(name=name, deviation=dev, passed=field_passed))

        agg = stats.setdefault(
            name, {"mean_deviation": 0.0, "max_deviation": 0.0, "failures": 0.0, "_seen": 0.0}
        )
        agg["mean_deviation"] += dev
        agg["max_deviation"] = max(agg["max_deviation"], dev)
        agg["_seen"] += 1.0
        if not field_passed:
            agg["failures"] += 1.0

    return RecordResult(
        id=result.get("id"),
        passed=record_passed,
        max_deviation=max_dev,
        fields=field_results,
    )

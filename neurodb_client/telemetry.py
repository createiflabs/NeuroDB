"""Local-first telemetry for NeuroDB validation runs.

This module records *what happened* during dataset validation (how many records
were checked, how many failed) without ever touching the network. By default the
active sink is :class:`NullSink`, so nothing is collected.

The remote story is a **seam, not an implementation**: callers who want to ship
events somewhere implement the tiny :class:`TelemetrySink` protocol and register
it with :func:`set_sink`. NeuroDB itself ships no remote endpoint, no URL and no
background uploader — keeping the library dependency-free and private by default.

    from neurodb_client import telemetry

    agg = telemetry.LocalAggregator()
    telemetry.set_sink(agg)
    ...                                  # run validations
    print(agg.snapshot())                # {"runs": 3, "records": 900, ...}

To forward events to your own collector, implement ``emit`` and register it::

    class MySink:
        def emit(self, event): requests.post(MY_URL, json=event.to_dict())

    telemetry.set_sink(MySink())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TelemetryEvent:
    """A single validation outcome, ready to be aggregated or forwarded."""

    event: str
    memory: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    threshold: float
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "memory": self.memory,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "threshold": self.threshold,
            "ts": self.ts,
        }


@runtime_checkable
class TelemetrySink(Protocol):
    """Anything with ``emit(event)`` can receive telemetry."""

    def emit(self, event: TelemetryEvent) -> None: ...


class NullSink:
    """The default sink: records nothing. Telemetry is opt-in."""

    def emit(self, event: TelemetryEvent) -> None:  # noqa: D401 - trivial
        return None


class LocalAggregator:
    """In-process counters over validation runs. Never leaves the machine."""

    def __init__(self) -> None:
        self.runs = 0
        self.records = 0
        self.failures = 0
        self.last: TelemetryEvent | None = None

    def emit(self, event: TelemetryEvent) -> None:
        self.runs += 1
        self.records += event.total
        self.failures += event.failed
        self.last = event

    def snapshot(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "records": self.records,
            "failures": self.failures,
            "last": self.last.to_dict() if self.last else None,
        }


# Module-level sink seam. Default is the no-op sink (local-only, privacy-first).
_sink: TelemetrySink = NullSink()


def set_sink(sink: TelemetrySink) -> None:
    """Register the active telemetry sink (e.g. a :class:`LocalAggregator` or a
    custom remote forwarder). Pass :class:`NullSink` to disable again."""

    global _sink
    _sink = sink


def get_sink() -> TelemetrySink:
    """Return the currently active sink."""

    return _sink


def record(event: TelemetryEvent) -> None:
    """Send ``event`` to the active sink. Swallows sink errors so telemetry can
    never break a validation run."""

    try:
        _sink.emit(event)
    except Exception:  # noqa: BLE001 - telemetry must never raise into callers
        pass

"""Local-only telemetry seam."""

from __future__ import annotations

import pytest

from neurodb_client import telemetry


@pytest.fixture(autouse=True)
def _restore_sink():
    """Keep the global sink isolated per test."""

    original = telemetry.get_sink()
    yield
    telemetry.set_sink(original)


def _event(failed: int = 0, total: int = 10) -> telemetry.TelemetryEvent:
    return telemetry.TelemetryEvent(
        event="validate",
        memory="m",
        total=total,
        passed=total - failed,
        failed=failed,
        pass_rate=(total - failed) / total,
        threshold=3.0,
    )


def test_default_sink_is_null():
    assert isinstance(telemetry.get_sink(), telemetry.NullSink)
    telemetry.record(_event())  # no-op, must not raise


def test_local_aggregator_accumulates():
    agg = telemetry.LocalAggregator()
    telemetry.set_sink(agg)

    telemetry.record(_event(failed=1, total=10))
    telemetry.record(_event(failed=2, total=20))

    snap = agg.snapshot()
    assert snap["runs"] == 2
    assert snap["records"] == 30
    assert snap["failures"] == 3
    assert snap["last"]["total"] == 20


def test_custom_sink_receives_events():
    received: list[telemetry.TelemetryEvent] = []

    class Sink:
        def emit(self, event):
            received.append(event)

    telemetry.set_sink(Sink())
    telemetry.record(_event(total=5))
    assert received and received[0].total == 5


def test_record_swallows_sink_errors():
    class Boom:
        def emit(self, event):
            raise RuntimeError("downstream down")

    telemetry.set_sink(Boom())
    telemetry.record(_event())  # must not propagate


def test_validation_emits_telemetry(db):
    agg = telemetry.LocalAggregator()
    telemetry.set_sink(agg)

    mem = db.create("t", dimension=2, beta=8.0, fields=["a", "b"], normalize="zscore")
    mem.write([{"vector": [1, 2]}, {"vector": [1.1, 2.1]}, {"vector": [0.9, 1.9]}])
    mem.validate([[1, 2], [1.1, 2.0]], threshold=3.0)

    assert agg.runs == 1
    assert agg.records == 2

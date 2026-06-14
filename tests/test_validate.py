"""Dataset validation via Memory.validate / run_validation.

Drives the validation API over the in-process transport (see the ``db`` fixture
in conftest), confirming pass/fail aggregation, per-field flagging, the ``fields``
filter, batching and the report's serialization helpers.
"""

from __future__ import annotations

import pytest

from neurodb import ValidationReport, connect  # re-export smoke
from neurodb_client.validate import run_validation

# A tight, high-beta sensor cluster so recall snaps and inliers reconstruct cleanly.
NORMAL = [
    [20, 50, 1013],
    [21, 52, 1012],
    [19, 48, 1014],
    [20, 51, 1013],
    [21, 49, 1012],
]


@pytest.fixture()
def sensors(db):
    mem = db.create(
        "sensors", dimension=3, beta=16.0,
        fields=["temperature", "humidity", "pressure"], normalize="zscore",
    )
    mem.write([{"vector": v} for v in NORMAL])
    return mem


def test_reexport_from_neurodb():
    # `from neurodb import connect, ValidationReport` resolves to the client.
    assert connect is not None
    assert ValidationReport.__name__ == "ValidationReport"


def test_clean_dataset_passes(sensors):
    report = sensors.validate(NORMAL, threshold=3.0)
    assert isinstance(report, ValidationReport)
    assert report.ok and bool(report) is True
    assert report.total == len(NORMAL)
    assert report.passed == len(NORMAL)
    assert report.failed == 0
    assert report.pass_rate == pytest.approx(1.0)


def test_outlier_record_fails_with_field_flag(sensors):
    report = sensors.validate(
        [{"id": "bad", "vector": [20, 95, 1013]}, {"id": "ok", "vector": [20, 51, 1013]}],
        threshold=3.0,
    )
    assert not report.ok
    assert report.failed == 1

    bad = next(r for r in report.records if r.id == "bad")
    assert not bad.passed
    assert any(f.name == "humidity" and not f.passed for f in bad.fields)
    assert bad.max_deviation > 3.0

    ok = next(r for r in report.records if r.id == "ok")
    assert ok.passed

    # per-field aggregates recorded humidity's failure
    assert report.field_stats["humidity"]["failures"] >= 1
    assert "mean_deviation" in report.field_stats["humidity"]


def test_fields_filter_scopes_check(sensors):
    # The outlier is in humidity; restricting to temperature should pass it.
    report = sensors.validate([[20, 95, 1013]], threshold=3.0, fields=["temperature"])
    assert report.ok
    assert all(f.name == "temperature" for r in report.records for f in r.fields)


def test_batching_covers_all_records(sensors):
    # batch_size below the record count exercises the chunking loop.
    report = run_validation(sensors, NORMAL, threshold=3.0, batch_size=2)
    assert report.total == len(NORMAL)


def test_report_serialization(sensors):
    report = sensors.validate(NORMAL, threshold=3.0)
    d = report.to_dict()
    assert d["memory"] == "sensors"
    assert d["total"] == len(NORMAL)
    assert isinstance(d["records"], list)
    assert "passed" in str(report)

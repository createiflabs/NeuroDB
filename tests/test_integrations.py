"""Adapter contracts for Great Expectations / Airflow / Dagster.

Each adapter is import-guarded: with its library installed the adapter wraps
``Memory.validate``; without it, the adapter raises a clear ``ImportError`` naming
the extra to install. These tests assert whichever branch applies in the current
environment, so they pass with or without the optional extras.
"""

from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture()
def sensors(db):
    mem = db.create(
        "s", dimension=3, beta=16.0,
        fields=["temperature", "humidity", "pressure"], normalize="zscore",
    )
    mem.write([{"vector": v} for v in ([20, 50, 1013], [21, 52, 1012], [19, 48, 1014])])
    return mem


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_great_expectations_adapter(sensors):
    from neurodb_client.integrations import great_expectations as ge

    if not _installed("great_expectations"):
        with pytest.raises(ImportError):
            ge.neurodb_expectation(sensors)
        return

    expectation = ge.neurodb_expectation(sensors, threshold=3.0)
    result = expectation([[20, 50, 1013]])
    assert result["success"] is True
    assert result["result"]["memory"] == "s"


def test_airflow_operator(sensors):
    from neurodb_client.integrations.airflow import NeuroDBValidateOperator

    if not _installed("airflow"):
        with pytest.raises(ImportError):
            NeuroDBValidateOperator(memory=sensors, records=[[20, 50, 1013]], task_id="t")
        return

    op = NeuroDBValidateOperator(
        task_id="validate", memory=sensors, records=[[20, 50, 1013]], threshold=3.0
    )
    result = op.execute({})
    assert result["total"] == 1
    assert result["failed"] == 0


def test_dagster_asset_check(sensors):
    from neurodb_client.integrations.dagster import neurodb_asset_check

    if not _installed("dagster"):
        with pytest.raises(ImportError):
            neurodb_asset_check(asset="data", memory=sensors, records=[[20, 50, 1013]])
        return

    import dagster as dg

    check = neurodb_asset_check(
        asset=dg.AssetKey("data"), memory=sensors, records=[[20, 50, 1013]], threshold=3.0
    )
    assert check is not None

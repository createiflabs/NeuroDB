"""Optional adapters that plug NeuroDB validation into data-stack tools.

Each submodule import-guards its third-party dependency and raises a clear
:class:`ImportError` (naming the extra to install) when it is missing, so simply
importing this subpackage never drags in Great Expectations, Airflow or Dagster.

    from neurodb_client.integrations.great_expectations import neurodb_expectation
    from neurodb_client.integrations.airflow import NeuroDBValidateOperator
    from neurodb_client.integrations.dagster import neurodb_asset_check
"""

from __future__ import annotations

__all__ = [
    "great_expectations",
    "airflow",
    "dagster",
]

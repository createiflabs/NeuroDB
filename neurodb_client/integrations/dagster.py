"""Dagster asset-check adapter for NeuroDB validation.

``neurodb_asset_check`` builds an ``@asset_check`` that validates a dataset
against a NeuroDB memory and reports pass/fail (with summary metadata) on the
target asset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import Memory

_EXTRA = "pip install 'neurodb[dagster]'"


def _require_dagster() -> Any:
    try:
        import dagster as dg
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise ImportError(f"dagster is required for this adapter ({_EXTRA})") from exc
    return dg


def neurodb_asset_check(
    *,
    asset: Any,
    memory: Memory,
    records: Iterable[Any],
    name: str = "neurodb_validation",
    threshold: float = 3.0,
    fields: Sequence[str] | None = None,
    beta: float | None = None,
    filter: dict[str, Any] | None = None,
) -> Any:
    """Return a Dagster ``@asset_check`` that validates ``records`` against ``memory``."""

    dg = _require_dagster()

    @dg.asset_check(asset=asset, name=name)
    def _check() -> Any:
        report = memory.validate(
            records, threshold=threshold, fields=fields, beta=beta, filter=filter
        )
        return dg.AssetCheckResult(
            passed=report.ok,
            metadata={
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "pass_rate": report.pass_rate,
                "threshold": report.threshold,
            },
        )

    return _check

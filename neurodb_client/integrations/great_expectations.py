"""Great Expectations adapter for NeuroDB validation.

Wraps :meth:`Memory.validate` into a callable that returns a Great-Expectations
style result (``success`` flag + structured ``result``), so a NeuroDB anomaly
check can sit inside a GE suite alongside ordinary expectations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import Memory

_EXTRA = "pip install 'neurodb[great-expectations]'"


def _require_ge() -> Any:
    try:
        import great_expectations as gx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise ImportError(
            f"great_expectations is required for this adapter ({_EXTRA})"
        ) from exc
    return gx


def neurodb_expectation(
    memory: Memory,
    *,
    threshold: float = 3.0,
    fields: Sequence[str] | None = None,
    beta: float | None = None,
    filter: dict[str, Any] | None = None,
) -> Callable[[Iterable[Any]], dict[str, Any]]:
    """Return a callable ``expectation(records) -> result`` for a GE suite.

    The returned ``result`` mirrors GE's ``ExpectationValidationResult`` shape:
    ``{"success": bool, "result": {...report...}}``.
    """

    _require_ge()

    def _expectation(records: Iterable[Any]) -> dict[str, Any]:
        report = memory.validate(
            records, threshold=threshold, fields=fields, beta=beta, filter=filter
        )
        return {"success": report.ok, "result": report.to_dict()}

    return _expectation

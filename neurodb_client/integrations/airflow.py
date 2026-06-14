"""Apache Airflow operator for NeuroDB validation.

``NeuroDBValidateOperator`` runs :meth:`Memory.validate` inside a task, pushes the
report summary to XCom, and (by default) fails the task when any record fails.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import Memory

_EXTRA = "pip install 'neurodb[airflow]'"

try:  # pragma: no cover - import shim
    from airflow.models import BaseOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    BaseOperator = object  # type: ignore[assignment,misc]
    _AIRFLOW_AVAILABLE = False


class NeuroDBValidateOperator(BaseOperator):
    """Validate a dataset against a NeuroDB memory as an Airflow task."""

    def __init__(
        self,
        *,
        memory: Memory,
        records: Iterable[Any],
        threshold: float = 3.0,
        fields: Sequence[str] | None = None,
        beta: float | None = None,
        filter: dict[str, Any] | None = None,
        fail_on_error: bool = True,
        **kwargs: Any,
    ) -> None:
        if not _AIRFLOW_AVAILABLE:  # pragma: no cover - exercised when extra absent
            raise ImportError(f"apache-airflow is required for this operator ({_EXTRA})")
        super().__init__(**kwargs)
        self.memory = memory
        self.records = records
        self.threshold = threshold
        self.fields = fields
        self.beta = beta
        self.filter = filter
        self.fail_on_error = fail_on_error

    def execute(self, context: Any) -> dict[str, Any]:
        report = self.memory.validate(
            self.records,
            threshold=self.threshold,
            fields=self.fields,
            beta=self.beta,
            filter=self.filter,
        )
        if self.fail_on_error and not report.ok:
            raise ValueError(report.summary())
        return report.to_dict()

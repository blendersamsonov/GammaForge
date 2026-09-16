"""Run history data structures for the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

from ..io.calculation import CalculationRequest
from ..io.results import Results

if TYPE_CHECKING:
    from ..engines.base import Engine


@dataclass
class Run:
    """A single calculation run in the history."""

    id: int
    name: str
    engine_name: str
    request: CalculationRequest
    results: Results | None = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    error: str = ""
    timestamp: float = field(default_factory=time)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed" and self.results is not None

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def display_name(self) -> str:
        return self.name

    def with_results(self, results: Results, status: str = "completed", error: str = "") -> "Run":
        """Return a new Run with results attached (immutable update pattern)."""
        return Run(
            id=self.id,
            name=self.name,
            engine_name=self.engine_name,
            request=self.request,
            results=results,
            status=status,
            error=error,
            timestamp=self.timestamp,
        )

    def with_status(self, status: str, error: str = "") -> "Run":
        """Return a new Run with updated status."""
        return Run(
            id=self.id,
            name=self.name,
            engine_name=self.engine_name,
            request=self.request,
            results=self.results,
            status=status,
            error=error,
            timestamp=self.timestamp,
        )

    def with_name(self, name: str) -> "Run":
        """Return a new Run with updated name."""
        return Run(
            id=self.id,
            name=name,
            engine_name=self.engine_name,
            request=self.request,
            results=self.results,
            status=self.status,
            error=self.error,
            timestamp=self.timestamp,
        )
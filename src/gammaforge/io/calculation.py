"""Immutable input snapshots for local calculation execution.

This is deliberately a small shared boundary: it carries physical inputs and validated
engine parameters, never browser state or execution machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .bunch import GaussianElectronBeam
from .interaction import SamplingSpec
from .laser import LaserField
from .schema import Parameters
from .target import Target

__all__ = ["CalculationRequest"]


@dataclass(frozen=True)
class CalculationRequest:
    """One self-contained calculation snapshot for a local or future remote runner."""

    beam: GaussianElectronBeam
    laser: LaserField
    target: Target
    sampling: SamplingSpec
    engine_params: Mapping[str, Parameters]

    def __post_init__(self) -> None:
        """Freeze the nested parameter mapping at the execution boundary.

        ``frozen=True`` protects the attribute, not a caller-owned dict passed into it.
        ``Parameters`` is already immutable, so a shallow mapping snapshot is the exact
        ownership boundary needed here.
        """
        if any(not isinstance(value, Parameters) for value in self.engine_params.values()):
            raise TypeError("CalculationRequest.engine_params values must be Parameters")
        object.__setattr__(self, "engine_params", MappingProxyType(dict(self.engine_params)))

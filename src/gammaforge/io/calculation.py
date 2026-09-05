"""Immutable input snapshots for local calculation execution.

This is deliberately a small shared boundary: it carries physical inputs and validated
engine parameters, never browser state or execution machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    engine_params: dict[str, Parameters]

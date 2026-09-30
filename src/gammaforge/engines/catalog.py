"""The one public enumeration of the engines GammaForge ships.

`LocalRunner`, calculation serialization, the GUI and ordinary scripts all resolve engines
from here instead of importing an implementation module, so adding an in-repository engine
is one entry rather than an edit to every frontend (RES095).

Two declarations carry that: an `EngineRole` saying what an entry is *for*, and a factory
so a caller gets a fresh engine rather than a shared mutable one. Role is why the estimate
overlay is never offered as a calculation choice and why the validation reference is not
an engine tab — declaratively, with no frontend branching on a name.

This is deliberately not a plugin manager. There is no mutable global registry, no
capability negotiation and no dynamic loading of separately installed packages: the
capability contract remains each engine's own `schema`, `supported_outputs` and
`recompute_costs` (RES018).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..io.schema import Parameters
from .analytical.engine import AnalyticalEngine
from .base import Engine
from .delta.engine import DeltaEngine
from .xigma.engine import XigmaEngine

__all__ = [
    "ENGINES",
    "EngineRole",
    "EngineSpec",
    "engine_names",
    "engine_schemas",
    "estimate_engine",
    "get_engine",
    "selectable_engines",
]


class EngineRole(Enum):
    """What a catalog entry is for, so no frontend has to know an engine by name."""

    #: An ordinary user-selectable calculation engine.
    CALCULATION = "calculation"
    #: The always-visible analytical preview and result overlay. Never a selectable engine.
    ESTIMATE = "estimate"
    #: An internal or validation-only reference: resolvable by name, not offered as a choice.
    REFERENCE = "reference"


@dataclass(frozen=True)
class EngineSpec:
    """One catalog entry: a stable name, the role it plays, and how to build it."""

    name: str
    role: EngineRole
    factory: Callable[[], Engine]


#: Insertion order is catalog order; the first entry is the GUI's initial engine choice.
ENGINES: Mapping[str, EngineSpec] = MappingProxyType({
    "xigma": EngineSpec("xigma", EngineRole.CALCULATION, XigmaEngine),
    "analytical": EngineSpec("analytical", EngineRole.ESTIMATE, AnalyticalEngine),
    "delta": EngineSpec("delta", EngineRole.REFERENCE, DeltaEngine),
})


def engine_names(roles: tuple[EngineRole, ...] | None = None) -> tuple[str, ...]:
    """Return catalog names in catalog order, optionally filtered to ``roles``."""
    return tuple(
        name for name, spec in ENGINES.items() if roles is None or spec.role in roles
    )


def get_engine(name: str) -> Engine:
    """Return a new engine instance for the stable catalog name ``name``."""
    try:
        spec = ENGINES[name]
    except KeyError:
        raise ValueError(
            f"unknown calculation engine: {name} (available: {', '.join(ENGINES)})"
        ) from None
    return spec.factory()


def selectable_engines() -> dict[str, Engine]:
    """Return fresh instances of the user-selectable calculation engines, keyed by name.

    This is what a frontend offers as an engine choice; estimate and reference roles are
    excluded, so no caller has to know which engine a name refers to in order to skip it.
    """
    return {
        name: spec.factory()
        for name, spec in ENGINES.items()
        if spec.role is EngineRole.CALCULATION
    }


def engine_schemas() -> dict[str, Parameters]:
    """Return every catalogued engine's typed schema, for restoring a recorded request.

    Reference engines are included so a request recorded against one still round-trips; a
    name the catalog never had is what makes a restore fail.
    """
    return {name: spec.factory().schema for name, spec in ENGINES.items()}


def estimate_engine() -> Engine:
    """Return a new instance of the single estimate engine used for previews and overlays."""
    for spec in ENGINES.values():
        if spec.role is EngineRole.ESTIMATE:
            return spec.factory()
    raise LookupError("no estimate engine is registered in the catalog")

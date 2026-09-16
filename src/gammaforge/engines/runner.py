"""Synchronous local execution behind the GUI's calculation boundary."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Callable

import numpy as np

from ..io.bunch import Bunch
from ..io.calculation import CalculationRequest
from ..io.interaction import InteractionParameters, build_interaction
from ..io.results import Results
from ..io.target import OutputKind, OutputRequest
from .analytical.engine import AnalyticalEngine
from .base import Engine
from .delta.engine import DeltaEngine
from .kascade.engine import KascadeEngine
from .xigma.engine import XigmaEngine

__all__ = ["LocalRunner"]

StatusCallback = Callable[[str, str], None]


class LocalRunner:
    """Run the concrete local engines sequentially and retain one sampled interaction.

    The cache deliberately stops at the sampled bunch.  It avoids resampling for target,
    output, engine-numeric, and charge-only edits without claiming any xigma stage reuse.
    """

    def __init__(self, engines: dict[str, Engine] | None = None) -> None:
        self.engines: dict[str, Engine] = (
            {"xigma": XigmaEngine(), "kascade": KascadeEngine(), "delta": DeltaEngine()}
            if engines is None
            else engines
        )
        self.errors: dict[str, str] = {}
        self._cached_sampling_key: tuple[object, ...] | None = None
        self._cached_interaction: InteractionParameters | None = None

    def estimate(self, request: CalculationRequest) -> Results:
        """Return the analytical total-yield preview without sampling macroparticles."""
        target = replace(request.target, outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
        return self._analytical(request, target)

    def _analytical(self, request: CalculationRequest, target) -> Results:
        """Evaluate the bunch-independent analytical engine for ``target``."""
        interaction = InteractionParameters(
            beam=request.beam,
            laser=request.laser,
            bunch=_empty_bunch(),
            target=target,
            N_e=request.beam.n_electrons(),
            sampling=request.sampling,
        )
        return AnalyticalEngine().run(interaction, AnalyticalEngine.schema)

    def calculate(
        self, request: CalculationRequest, on_status: StatusCallback | None = None
    ) -> dict[str, Results]:
        """Run selected engines in request order, preserving results from successful ones."""
        unknown = set(request.engine_params) - set(self.engines)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown calculation engine(s): {names}")

        self.errors.clear()
        results: dict[str, Results] = {}

        # Analytical is always an overlay, not a selectable calculation engine.  Calculate
        # retains every analytical output from this snapshot; preview uses `estimate()`.
        self._run_one("analytical", lambda: self._analytical(request, request.target), results, on_status)

        if not request.engine_params:
            return results

        try:
            interaction = self._interaction(request)
        except Exception as error:
            # Sampling is common to all selected engines; report its failure against each
            # selected engine so the caller can display meaningful per-engine state.
            for name in request.engine_params:
                self.errors[name] = str(error)
                _status(on_status, name, "failed")
            return results

        for name, params in request.engine_params.items():
            engine = self.engines[name]
            self._run_one(name, lambda engine=engine, params=params: engine.run(interaction, params), results, on_status)
        return results

    def calculate_one(
        self, engine_name: str, request: CalculationRequest, on_status: StatusCallback | None = None
    ) -> Results | None:
        """Run a single engine (plus analytical overlay) and return its results.

        Returns the engine's Results on success, None on failure.
        """
        if engine_name not in self.engines:
            raise ValueError(f"unknown calculation engine: {engine_name}")

        self.errors.clear()
        results: dict[str, Results] = {}

        # Always run analytical as overlay
        self._run_one("analytical", lambda: self._analytical(request, request.target), results, on_status)

        try:
            interaction = self._interaction(request)
        except Exception as error:
            self.errors[engine_name] = str(error)
            _status(on_status, engine_name, "failed")
            return None

        engine = self.engines[engine_name]
        self._run_one(engine_name, lambda engine=engine, params=request.engine_params[engine_name]: engine.run(interaction, params), results, on_status)

        if engine_name in self.errors:
            return None
        return results.get(engine_name)

    def _interaction(self, request: CalculationRequest) -> InteractionParameters:
        key = _sampling_key(request)
        if self._cached_sampling_key == key and self._cached_interaction is not None:
            cached = self._cached_interaction
            # Charge is the one beam field absent from the key: preserve its bunch while
            # refreshing the physical scalar electron count.
            return replace(
                cached,
                beam=request.beam,
                laser=request.laser,
                target=request.target,
                N_e=request.beam.n_electrons(),
                sampling=request.sampling,
            )
        interaction = build_interaction(request.beam, request.laser, request.target, request.sampling)
        self._cached_sampling_key = key
        self._cached_interaction = interaction
        return interaction

    def _run_one(
        self, name: str, operation: Callable[[], Results], results: dict[str, Results], on_status: StatusCallback | None
    ) -> None:
        _status(on_status, name, "running")
        try:
            result = operation()
            if not isinstance(result, Results):
                raise TypeError(f"{name} did not return Results")
            results[name] = result
        except Exception as error:
            self.errors[name] = str(error)
            _status(on_status, name, "failed")
        else:
            self.errors.pop(name, None)
            _status(on_status, name, "completed")


def _sampling_key(request: CalculationRequest) -> tuple[object, ...]:
    """The exact inputs that determine the sampled and prefiltered bunch."""
    beam_without_charge = tuple(
        getattr(request.beam, field.name) for field in fields(request.beam) if field.name != "bunch_charge"
    )
    return (beam_without_charge, request.laser, request.sampling)


def _empty_bunch() -> Bunch:
    """The analytical engine never consumes particles; avoid allocating a sampled bunch."""
    empty = np.empty(0)
    return Bunch(x=empty, y=empty, z=empty, thx=empty, thy=empty, gamma=empty, weight=empty)


def _status(callback: StatusCallback | None, name: str, status: str) -> None:
    if callback is not None:
        callback(name, status)
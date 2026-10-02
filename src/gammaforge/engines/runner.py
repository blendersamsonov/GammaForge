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
from .base import Engine
from .catalog import estimate_engine, selectable_engines

__all__ = ["LocalRunner"]

StatusCallback = Callable[[str, str], None]


class LocalRunner:
    """Run the concrete local engines sequentially and retain one sampled interaction.

    Default engines come from the public catalog rather than an import list here, so this
    module never names an engine implementation (RES095). `engines` stays injectable for
    tests and specialized callers.

    The cache deliberately stops at the sampled bunch.  It avoids resampling for target,
    output, engine-numeric, and charge-only edits without claiming any xigma stage reuse.
    """

    def __init__(self, engines: dict[str, Engine] | None = None) -> None:
        self.engines: dict[str, Engine] = (
            selectable_engines() if engines is None else engines
        )
        self.errors: dict[str, str] = {}
        self._cached_sampling_key: tuple[object, ...] | None = None
        self._cached_interaction: InteractionParameters | None = None

    @property
    def estimate_name(self) -> str:
        """The catalog name of the estimate engine, without hardcoding it at a call site.

        A frontend labels its estimate run from this rather than writing `"analytical"`,
        which is the same reason RES095 moved engine enumeration into the catalog.
        """
        return estimate_engine().name

    def estimate(self, request: CalculationRequest) -> Results:
        """Return the analytical total-yield preview without sampling macroparticles."""
        target = replace(request.target, outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
        return self._run_estimate(request, target)

    def estimate_requested(self, request: CalculationRequest) -> Results:
        """Return the analytical engine's full answer to the *requested* outputs.

        The counterpart to :meth:`estimate`, which deliberately narrows to `TOTAL_YIELD`
        because it runs on every keystroke. This one keeps whatever the caller asked for, so
        the same engine can also produce a spectra or angular-distribution estimate on
        demand rather than only a scalar.

        Which outputs actually come back is the analytical engine's decision, not this
        method's: its model planner refuses a geometry it has no accepted tier for, so
        asking for something it cannot model raises instead of quietly returning less.
        """
        return self._run_estimate(request, request.target)

    def _estimate_interaction(self, request: CalculationRequest, target) -> InteractionParameters:
        """Build the bunch-independent interaction the estimate engine evaluates."""
        return InteractionParameters(
            beam=request.beam,
            laser=request.laser,
            bunch=_empty_bunch(),
            target=target,
            N_e=request.beam.n_electrons(),
            sampling=request.sampling,
        )

    def _run_estimate(self, request: CalculationRequest, target) -> Results:
        """Evaluate the catalog's estimate engine for ``target``."""
        engine = estimate_engine()
        return engine.run(self._estimate_interaction(request, target), engine.schema)

    def _overlay(self, request: CalculationRequest, results: dict[str, Results],
                 on_status: StatusCallback | None) -> None:
        """Run the estimate engine as an always-present overlay, per RES058.

        The estimate role is why this is not an ordinary selectable engine, so the name
        comes from the catalog rather than being written here.
        """
        engine = estimate_engine()
        self._run_one(
            engine.name,
            lambda: engine.run(self._estimate_interaction(request, request.target), engine.schema),
            results,
            on_status,
        )

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

        # Calculate retains every analytical output from this snapshot; preview uses
        # `estimate()`, which is why the overlay runs before the engine loop below.
        self._overlay(request, results, on_status)

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
        return self.calculate_one_all(engine_name, request, on_status)[0]

    def calculate_one_all(
        self, engine_name: str, request: CalculationRequest, on_status: StatusCallback | None = None
    ) -> tuple[Results | None, Results | None]:
        """Run one engine and return ``(engine_results, estimate_results)``.

        The estimate engine's result used to be computed by `calculate_one` and then thrown
        away — it was a display overlay, and the caller received only the selected engine's
        `Results`. Now that the analytical engine produces full output kinds of its own
        (DER019 `ANGULAR_DISTRIBUTION` and `COLLIMATED_SPECTRUM`), that result is worth
        keeping: a caller can present it as a run in its own right, badge it as an estimate,
        and export it alongside the calculated one.

        Returned as a pair rather than folded into a single `Results` so the estimate stays
        distinguishable — it is a different engine with a different role (RES095), and
        merging them would erase exactly the provenance the GUI needs to label it.

        The estimate is still produced by the always-present overlay (RES058); this changes
        only whether the caller may keep it.
        """
        if engine_name not in self.engines:
            raise ValueError(f"unknown calculation engine: {engine_name}")

        self.errors.clear()
        results: dict[str, Results] = {}

        # Always run the estimate engine as an overlay (RES058).
        self._overlay(request, results, on_status)
        estimate_results = results.get(estimate_engine().name)

        try:
            interaction = self._interaction(request)
        except Exception as error:
            self.errors[engine_name] = str(error)
            _status(on_status, engine_name, "failed")
            return None, estimate_results

        engine = self.engines[engine_name]
        self._run_one(engine_name, lambda engine=engine, params=request.engine_params[engine_name]: engine.run(interaction, params), results, on_status)

        if engine_name in self.errors:
            return None, estimate_results
        return results.get(engine_name), estimate_results

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
"""Page-owned calculation state, independent of NiceGUI widgets."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from ..engines.runner import LocalRunner
from ..io.calculation import CalculationRequest
from ..io.results import Results
from .state import InputState


class Workspace:
    """Draft inputs and completed snapshot; never attach configuration to Results."""

    def __init__(self, runner: LocalRunner | None = None) -> None:
        self.runner = runner or LocalRunner()
        self.inputs = InputState(self.runner.engines)
        self.results: dict[str, Results] = {}
        self.completed_request: CalculationRequest | None = None
        self.revision = 0
        self.busy = False
        self.stale = False
        self.locked = False
        self.statuses: dict[str, str] = {}
        self.error = ""

    def changed(self, group: str, key: str) -> None:
        self.revision += 1
        self.error = ""
        if not self.results:
            return
        charge_only = (group, key) == ("beam", "bunch_charge")
        if charge_only and not self.stale and not self.busy and not self.inputs.errors:
            try:
                request = self.inputs.request()
                previous = self.completed_request
                if previous is not None:
                    old_charge = previous.beam.n_electrons()
                    if old_charge > 0:
                        factor = request.beam.n_electrons() / old_charge
                        self.results = {name: result.scaled(factor)
                                        for name, result in self.results.items()}
                        self.completed_request = replace(previous, beam=request.beam)
                        return
            except ValueError:
                pass
        self.stale = True

    async def calculate(self) -> None:
        if self.busy:
            return
        try:
            request = self.inputs.request()
            if not request.engine_params:
                raise ValueError("Select at least one engine for calculation.")
        except ValueError as exc:
            self.error = str(exc)
            return
        self.busy = True
        self.error = ""
        self.statuses = {name: "queued" for name in request.engine_params}
        revision = self.revision
        loop = asyncio.get_running_loop()

        def status(name: str, value: str) -> None:
            loop.call_soon_threadsafe(self.statuses.__setitem__, name, value)

        try:
            results = await asyncio.to_thread(self.runner.calculate, request, status)
            failures = self.runner.errors
            successful_engines = set(results) & set(request.engine_params)
            if failures:
                # `LocalRunner` includes analytical in every response, even when every
                # selected calculation engine failed.  That is useful as an overlay, but
                # not a completed calculation: preserve a prior useful result in the
                # all-failed case and leave retry inputs unlocked.  A mixed response is
                # honest partial output for this snapshot, likewise never lockable.
                if successful_engines:
                    self.results = results
                    self.completed_request = request
                self.stale = True
                self.locked = False
            elif results:
                self.results = results
                self.completed_request = request
                self.stale = self.revision != revision
                self.locked = self.revision == revision
            else:
                self.stale = bool(self.results)
            self.error = "\n".join(f"{name}: {error}" for name, error in failures.items())
        except Exception as exc:
            self.error = str(exc)
            self.stale = bool(self.results)
            self.locked = False
        finally:
            self.busy = False

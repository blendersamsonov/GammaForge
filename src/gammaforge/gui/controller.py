"""Page-owned calculation state with run history, independent of NiceGUI widgets."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import time

from ..engines.runner import LocalRunner
from ..io.calculation import CalculationRequest
from ..io.results import Results
from .run import Run
from .state import InputState


class Workspace:
    """Draft inputs and run history; never attach configuration to Results."""

    def __init__(self, runner: LocalRunner | None = None) -> None:
        self.runner = runner or LocalRunner()
        self.inputs = InputState(self.runner.engines)
        self.runs: list[Run] = []
        self.next_run_id = 1
        self.current_run_id: int | None = None  # Selected run in history panel
        self.revision = 0
        self.busy = False
        self.stale = False
        self.locked = False
        self.statuses: dict[str, str] = {}
        self.error = ""

    @property
    def current_run(self) -> Run | None:
        if self.current_run_id is None:
            return None
        for run in self.runs:
            if run.id == self.current_run_id:
                return run
        return None

    @property
    def completed_runs(self) -> list[Run]:
        return [run for run in self.runs if run.is_completed]

    @property
    def latest_completed_run(self) -> Run | None:
        completed = self.completed_runs
        return completed[-1] if completed else None

    def changed(self, group: str, key: str) -> None:
        self.revision += 1
        self.error = ""
        # Charge-only scaling now applies to the current run's results if viewing a run
        if not self.runs:
            return
        charge_only = (group, key) == ("beam", "bunch_charge")
        if charge_only and not self.stale and not self.busy and not self.inputs.errors:
            try:
                request = self.inputs.request()
                # Scale the current run's results if it's a completed run
                current = self.current_run
                if current and current.is_completed and current.results:
                    old_charge = current.request.beam.n_electrons()
                    if old_charge > 0:
                        factor = request.beam.n_electrons() / old_charge
                        scaled_results = current.results.scaled(factor)
                        # Update the run in the list
                        self.runs = [
                            run.with_results(scaled_results) if run.id == current.id else run
                            for run in self.runs
                        ]
                        return
            except ValueError:
                pass
        self.stale = True

    def select_run(self, run_id: int | None) -> None:
        """Select a run in the history panel."""
        self.current_run_id = run_id

    def fork_run(self, run_id: int) -> bool:
        """Load a historical run's inputs into the draft for editing."""
        for run in self.runs:
            if run.id == run_id:
                self._load_request_into_inputs(run.request)
                self.current_run_id = None  # Clear selection when forking
                return True
        return False

    def delete_run(self, run_id: int) -> bool:
        """Delete a run from history."""
        for i, run in enumerate(self.runs):
            if run.id == run_id:
                self.runs.pop(i)
                if self.current_run_id == run_id:
                    self.current_run_id = None
                return True
        return False

    def rename_run(self, run_id: int, new_name: str) -> bool:
        """Rename a run."""
        for i, run in enumerate(self.runs):
            if run.id == run_id:
                self.runs[i] = run.with_name(new_name)
                return True
        return False

    def _load_request_into_inputs(self, request: CalculationRequest) -> None:
        """Load a CalculationRequest into the InputState draft."""
        from ..io.fields import (
            beam_from_parameters,
            laser_from_parameters,
            laser_to_parameters,
            sampling_from_parameters,
            to_parameters,
        )
        from ..io.schema import Parameters

        # Load beam
        beam_params = to_parameters(request.beam, self.inputs.groups["beam"].specs)
        self.inputs.groups["beam"] = Parameters.from_specs(self.inputs.groups["beam"].specs)
        for spec in self.inputs.groups["beam"].specs:
            if spec.key in beam_params.values:
                self.inputs.groups["beam"] = self.inputs.groups["beam"].with_values(**{spec.key: beam_params.values[spec.key]})

        # Load laser (use laser_to_parameters to handle laser_type correctly)
        laser_params = laser_to_parameters(request.laser)
        self.inputs.groups["laser"] = Parameters.from_specs(self.inputs.groups["laser"].specs)
        for spec in self.inputs.groups["laser"].specs:
            if spec.key in laser_params.values:
                self.inputs.groups["laser"] = self.inputs.groups["laser"].with_values(**{spec.key: laser_params.values[spec.key]})

        # Load sampling
        sampling_params = to_parameters(request.sampling, self.inputs.groups["sampling"].specs)
        self.inputs.groups["sampling"] = Parameters.from_specs(self.inputs.groups["sampling"].specs)
        for spec in self.inputs.groups["sampling"].specs:
            if spec.key in sampling_params.values:
                self.inputs.groups["sampling"] = self.inputs.groups["sampling"].with_values(**{spec.key: sampling_params.values[spec.key]})

        # Load target
        target_params = to_parameters(request.target, self.inputs.groups["target"].specs)
        self.inputs.groups["target"] = Parameters.from_specs(self.inputs.groups["target"].specs)
        for spec in self.inputs.groups["target"].specs:
            if spec.key in target_params.values:
                self.inputs.groups["target"] = self.inputs.groups["target"].with_values(**{spec.key: target_params.values[spec.key]})

        # Load engine params
        for engine_name, params in request.engine_params.items():
            if f"engine:{engine_name}" in self.inputs.groups:
                self.inputs.groups[f"engine:{engine_name}"] = params

        # Set selected engine
        if request.engine_params:
            engine_name = next(iter(request.engine_params))
            self.inputs.selected_engine = engine_name

        # Load requested outputs
        self.inputs.requested = {req.kind: req.resolution for req in request.target.outputs}
        self.inputs.manual_ranges = {
            req.kind: dict(req.manual_ranges)
            for req in request.target.outputs
            if req.manual_ranges
        }
        self.inputs.manual_axes = {
            (kind, axis)
            for kind, ranges in self.inputs.manual_ranges.items()
            for axis in ranges
        }

        self.revision += 1
        self.stale = True

    async def calculate(self) -> None:
        if self.busy:
            return
        try:
            request = self.inputs.request()
            if not request.engine_params:
                raise ValueError("Select an engine for calculation.")
        except ValueError as exc:
            self.error = str(exc)
            return

        # Create new run
        engine_name = next(iter(request.engine_params))
        run = Run(
            id=self.next_run_id,
            name=f"Run {self.next_run_id}",
            engine_name=engine_name,
            request=request,
            status="running",
        )
        self.next_run_id += 1
        self.runs.append(run)
        self.current_run_id = run.id

        self.busy = True
        self.error = ""
        self.statuses = {engine_name: "queued"}
        revision = self.revision
        loop = asyncio.get_running_loop()

        def status(name: str, value: str) -> None:
            loop.call_soon_threadsafe(self.statuses.__setitem__, name, value)

        try:
            # Update run status to running
            self.runs = [
                r.with_status("running") if r.id == run.id else r
                for r in self.runs
            ]

            # Run single engine (analytical is always run as overlay)
            results, estimate_results = await asyncio.to_thread(
                self.runner.calculate_one_all, engine_name, request, status
            )
            failures = self.runner.errors

            if failures and engine_name in failures:
                self.runs = [
                    r.with_status("failed", failures[engine_name]) if r.id == run.id else r
                    for r in self.runs
                ]
                self.stale = True
                self.locked = False
            elif results:
                self.runs = [
                    r.with_results(results, "completed") if r.id == run.id else r
                    for r in self.runs
                ]
                self.stale = self.revision != revision
                self.locked = self.revision == revision
            else:
                self.runs = [
                    r.with_status("failed", "No results returned") if r.id == run.id else r
                    for r in self.runs
                ]
                self.stale = bool(self.runs)

            # The analytical estimate becomes a run of its own (badged as an estimate), so it
            # can be compared against the calculated engine in the same plot, renamed,
            # forked, exported to HDF5, and kept across a later re-run. Previously it was
            # computed as an overlay and then discarded, which meant its spectra were
            # computed every time and never shown.
            # Skip when the estimate was itself the selected engine: it already produced its
            # own run above, and a second identical one would double-count it in the history
            # and show two entries that look like different results.
            if estimate_results is not None and engine_name != self.runner.estimate_name:
                self.runs.append(
                    Run(
                        id=self.next_run_id,
                        name=f"Run {self.next_run_id}",
                        engine_name=self.runner.estimate_name,
                        request=request,
                        results=estimate_results,
                        status="completed",
                    )
                )
                self.next_run_id += 1
                self.locked = self.revision == revision

            self.error = "\n".join(f"{name}: {error}" for name, error in failures.items())
        except Exception as exc:
            self.error = str(exc)
            self.runs = [
                r.with_status("failed", str(exc)) if r.id == run.id else r
                for r in self.runs
            ]
            self.stale = bool(self.runs)
            self.locked = False
        finally:
            self.busy = False

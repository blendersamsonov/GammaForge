"""Calculation lifecycle assertions without a running browser or NiceGUI."""

import asyncio
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.engines.base import RecomputeCost
from gammaforge.gui.controller import Workspace
from gammaforge.io.results import PhasespaceSlice, Results
from gammaforge.io.schema import Parameters
from gammaforge.io.target import OutputKind


class Engine:
    name = "test"
    schema = Parameters.from_specs(())
    supported_outputs = (OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM)
    recompute_costs = {"n_e": RecomputeCost.QUERY_ONLY}


class Runner:
    def __init__(self):
        self.engines = {"test": Engine()}
        self.errors = {}
        self.calls = 0
        self.request = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def calculate(self, request, status):
        self.calls += 1
        self.request = request
        status("test", "running")
        self.started.set()
        assert self.release.wait(5)
        status("test", "completed")
        return {"test": Results({OutputKind.TOTAL_YIELD: PhasespaceSlice({}, np.array(100.0))})}


def test_charge_only_rescale_and_combined_edits_stale():
    runner = Runner()
    workspace = Workspace(runner)
    asyncio.run(workspace.calculate())
    old = workspace.inputs.groups["beam"].get_float("bunch_charge")
    workspace.inputs.groups["beam"] = workspace.inputs.groups["beam"].with_values(bunch_charge=2 * old)
    workspace.changed("beam", "bunch_charge")
    assert not workspace.stale
    assert float(workspace.results["test"].photon_slices[OutputKind.TOTAL_YIELD].distr) == 200
    workspace.changed("laser", "pulse_energy")
    workspace.inputs.groups["beam"] = workspace.inputs.groups["beam"].with_values(bunch_charge=3 * old)
    workspace.changed("beam", "bunch_charge")
    assert workspace.stale
    assert float(workspace.results["test"].photon_slices[OutputKind.TOTAL_YIELD].distr) == 200
    assert runner.calls == 1


def test_edit_during_run_and_duplicate_click():
    async def exercise():
        runner = Runner()
        runner.release.clear()
        workspace = Workspace(runner)
        task = asyncio.create_task(workspace.calculate())
        for _ in range(300):
            if runner.started.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("runner did not start")
        assert workspace.busy
        workspace.inputs.groups["laser"] = workspace.inputs.groups["laser"].with_values(pulse_energy=40.0)
        workspace.changed("laser", "pulse_energy")
        await workspace.calculate()
        runner.release.set()
        await task
        assert runner.calls == 1
        assert workspace.stale
        assert not workspace.busy
        assert not workspace.locked
        assert runner.request.laser.m("pulse_energy") != workspace.inputs.request().laser.m("pulse_energy")
    asyncio.run(exercise())


def test_invalid_draft_never_runs_old_valid_values():
    runner = Runner()
    workspace = Workspace(runner)
    workspace.inputs.set_value("beam", "bunch_charge", "not a number")
    asyncio.run(workspace.calculate())
    assert runner.calls == 0
    assert workspace.error


def test_failed_run_keeps_last_result_outdated():
    class Fails(Runner):
        def calculate(self, request, status):
            raise RuntimeError("test failure")
    workspace = Workspace(Runner())
    asyncio.run(workspace.calculate())
    previous = workspace.results
    workspace.runner = Fails()
    asyncio.run(workspace.calculate())
    assert workspace.results is previous
    assert workspace.stale
    assert "test failure" in workspace.error
    assert not workspace.busy


def test_all_selected_engine_failures_keep_previous_results_and_unlock():
    class Fails(Runner):
        def calculate(self, request, status):
            self.calls += 1
            self.errors = {"test": "engine failure"}
            return {"analytical": Results({})}

    workspace = Workspace(Runner())
    asyncio.run(workspace.calculate())
    previous = workspace.results
    workspace.locked = True
    workspace.runner = Fails()

    asyncio.run(workspace.calculate())

    assert workspace.results is previous
    assert workspace.stale
    assert not workspace.locked
    assert "test: engine failure" in workspace.error


def test_partial_engine_failure_keeps_only_current_successful_results_unlocked():
    class Partial(Runner):
        def __init__(self):
            super().__init__()
            self.engines["other"] = Engine()

        def calculate(self, request, status):
            self.errors = {"other": "engine failure"}
            return {
                "analytical": Results({}),
                "test": Results({OutputKind.TOTAL_YIELD: PhasespaceSlice({}, np.array(50.0))}),
            }

    workspace = Workspace(Partial())

    asyncio.run(workspace.calculate())

    assert set(workspace.results) == {"analytical", "test"}
    assert workspace.stale
    assert not workspace.locked
    assert "other: engine failure" in workspace.error


def test_exception_after_a_locked_result_unlocks_for_retry():
    class Fails(Runner):
        def calculate(self, request, status):
            raise RuntimeError("test failure")

    workspace = Workspace(Runner())
    asyncio.run(workspace.calculate())
    assert workspace.locked
    workspace.runner = Fails()

    asyncio.run(workspace.calculate())

    assert not workspace.locked

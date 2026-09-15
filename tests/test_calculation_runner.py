"""The calculation runner's execution and sampled-bunch reuse contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.engines.runner import LocalRunner
from gammaforge.engines.kascade.engine import KascadeEngine
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.results import Results
from gammaforge.io.schema import Parameters
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


class _FakeEngine:
    schema = Parameters.from_specs(())
    supported_outputs = ()
    recompute_costs = {}

    def __init__(self, name: str, calls: list[tuple[str, object]], fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def run(self, interaction, params):
        self.calls.append((self.name, interaction.bunch))
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        return Results(photon_slices={}, model_specific={"engine": self.name})


def test_default_runner_exposes_xigma_then_opt_in_kascade():
    runner = LocalRunner()

    assert list(runner.engines) == ["xigma", "kascade", "delta"]
    assert isinstance(runner.engines["kascade"], KascadeEngine)


def _request(**changes) -> CalculationRequest:
    scenario = scenarios.BASELINE
    base = CalculationRequest(
        beam=scenario.beam,
        laser=scenario.laser,
        target=replace(scenario.target, outputs=(OutputRequest(OutputKind.TOTAL_YIELD),)),
        sampling=replace(scenario.sampling, n_particles=32, prefilter=0.0),
        engine_params={"first": Parameters.from_specs(()), "second": Parameters.from_specs(())},
    )
    return replace(base, **changes)


def test_calculate_is_ordered_continues_after_failure_and_shares_one_bunch():
    calls: list[tuple[str, object]] = []
    runner = LocalRunner({"first": _FakeEngine("first", calls, fail=True), "second": _FakeEngine("second", calls)})
    statuses: list[tuple[str, str]] = []

    results = runner.calculate(_request(), lambda *event: statuses.append(event))

    assert list(results) == ["analytical", "second"]
    assert [name for name, _ in calls] == ["first", "second"]
    assert calls[0][1] is calls[1][1]
    assert runner.errors == {"first": "first broke"}
    assert statuses == [
        ("analytical", "running"), ("analytical", "completed"),
        ("first", "running"), ("first", "failed"),
        ("second", "running"), ("second", "completed"),
    ]


def test_non_results_engine_return_is_failed_and_later_engines_continue():
    class _NoneEngine(_FakeEngine):
        def run(self, interaction, params):
            self.calls.append((self.name, interaction.bunch))
            return None

    calls: list[tuple[str, object]] = []
    runner = LocalRunner({"first": _NoneEngine("first", calls), "second": _FakeEngine("second", calls)})
    statuses: list[tuple[str, str]] = []

    results = runner.calculate(_request(), lambda *event: statuses.append(event))

    assert list(results) == ["analytical", "second"]
    assert "first" not in results
    assert runner.errors == {"first": "first did not return Results"}
    assert statuses[-4:] == [
        ("first", "running"), ("first", "failed"),
        ("second", "running"), ("second", "completed"),
    ]


def test_sampling_cache_ignores_target_numerics_and_charge_but_not_beam_laser_or_sampling():
    calls: list[tuple[str, object]] = []
    runner = LocalRunner({"first": _FakeEngine("first", calls)})
    request = _request(engine_params={"first": Parameters.from_specs(())})
    runner.calculate(request)
    first_bunch = calls[-1][1]

    changed_target = replace(request, target=replace(request.target, outputs=()))
    changed_charge = replace(request, beam=replace(request.beam, bunch_charge=Quantity(11.0, "nC")))
    runner.calculate(changed_target)
    runner.calculate(changed_charge)
    assert calls[-1][1] is first_bunch
    assert calls[-2][1] is first_bunch

    runner.calculate(replace(request, sampling=replace(request.sampling, seed=1)))
    assert calls[-1][1] is not first_bunch
    second_bunch = calls[-1][1]

    runner.calculate(replace(request, beam=replace(request.beam, sigma_x=Quantity(11.0, "um"))))
    assert calls[-1][1] is not second_bunch
    third_bunch = calls[-1][1]

    runner.calculate(replace(request, laser=replace(request.laser, pulse_energy=Quantity(21.0, "J"))))
    assert calls[-1][1] is not third_bunch


def test_estimate_never_samples_and_only_requests_total_yield(monkeypatch):
    runner = LocalRunner()
    request = _request()

    monkeypatch.setattr("gammaforge.engines.runner.build_interaction", lambda *args: pytest.fail("sampled"))
    result = runner.estimate(request)

    assert set(result.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_calculate_preserves_requested_analytical_spectrum_without_sampling(monkeypatch):
    runner = LocalRunner(engines={})
    request = _request(
        target=replace(
            scenarios.BASELINE.target,
            outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(8,))),
        ),
        engine_params={},
    )
    monkeypatch.setattr("gammaforge.engines.runner.build_interaction", lambda *args: pytest.fail("sampled"))

    results = runner.calculate(request)

    assert set(results["analytical"].photon_slices) == {OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM}


def test_unknown_engine_names_are_rejected_before_sampling():
    runner = LocalRunner()
    with pytest.raises(ValueError, match="unknown calculation engine"):
        runner.calculate(_request(engine_params={"missing": Parameters.from_specs(())}))


def test_calculation_request_snapshots_engine_parameter_mapping():
    params = Parameters.from_specs(())
    supplied = {"first": params}
    request = _request(engine_params=supplied)
    supplied["second"] = params
    assert dict(request.engine_params) == {"first": params}
    with pytest.raises(TypeError):
        request.engine_params["third"] = params  # type: ignore[index]


def test_small_real_xigma_execution():
    scenario = scenarios.BASELINE
    params = XigmaEngine.schema.with_values(
        n_bins_gamma=8,
        n_bins_theta_x=8,
        n_bins_theta_y=8,
        n_bins_a0_shape=8,
        n_bins_ahat=4,
    )
    request = CalculationRequest(
        beam=scenario.beam,
        laser=scenario.laser,
        target=replace(scenario.target, outputs=(OutputRequest(OutputKind.TOTAL_YIELD),)),
        sampling=SamplingSpec(n_particles=64, seed=0, prefilter=0.0),
        engine_params={"xigma": params},
    )

    results = LocalRunner().calculate(request)

    assert OutputKind.TOTAL_YIELD in results["xigma"].photon_slices

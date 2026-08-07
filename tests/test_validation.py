"""The validation harness itself (GRAND_PLAN.md §7, Phase 2 exit).

Two different things are being tested here and it is worth keeping them apart.

*Real physics checks*, which need nothing beyond what exists: the committed goldens agree
with this repo's own closed forms, and the shared core's invariance properties hold.

*Harness mechanics*, which need an engine — and there is none until Phase 3a. Those are
exercised against deliberately trivial stub engines defined below. A stub proves the
harness reports what actually happened: one that is invariant passes, one that is not
fails. Testing that a checker catches a violation matters more than testing that it
passes, since a checker that always passes looks identical to a correct one from the
outside.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.base import Engine, RecomputeCost
from gammaforge.io.results import Axis, PhasespaceSlice, Results
from gammaforge.io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters
from gammaforge.io.target import OutputKind, compton_edge_energy
from gammaforge.io.units import Quantity
from gammaforge.validation import golden as golden_module
from gammaforge.validation import invariance, metrics, scenarios
from gammaforge.validation.golden import (
    Provenance,
    available_goldens,
    compare_to_golden,
    load_golden,
    save_golden,
)
from gammaforge.validation.make_references import (
    _AXIS_TRANSLATION,
    _KIND_BY_AXES,
    _SCALAR_TRANSLATION,
    _scenario_payload,
)
from gammaforge.validation.run import run_suite
from gammaforge.validation.runners import derived_scalars, run_bank, run_engine

# ---------------------------------------------------------------------------
# Stub engines
# ---------------------------------------------------------------------------
CHUNK = FieldSpec(key="chunk", label="Chunk size", kind=FieldKind.SCALAR, unit=DIMENSIONLESS,
                  default=4096, value_range=(1, 1e9), integer=True)
DEVICE = FieldSpec(key="device", label="Backend", kind=FieldKind.CHOICE, unit=DIMENSIONLESS,
                   default="numpy", choices=("numpy", "cupy", "numba"))
STUB_SCHEMA = Parameters.from_specs((CHUNK, DEVICE))


class ConstantEngine:
    """Depends on the beam and laser, never on how the work was arranged.

    Physically meaningless by construction — a Gaussian of the right width in the right
    place, scaled by ``N_e``. That is exactly enough for the harness to have something
    whose invariances are known to hold by inspection.
    """

    name = "stub"
    schema = STUB_SCHEMA
    supported_outputs = (OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM)
    recompute_costs = {"chunk": RecomputeCost.FULL_RERUN, "device": RecomputeCost.FULL_RERUN}

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def _spectrum(self, interaction, offset: float = 0.0):
        from gammaforge.io.laser import fit_gaussian_paraxial

        edge = compton_edge_energy(interaction.beam, fit_gaussian_paraxial(interaction.laser).photon_energy())
        energy = np.linspace(0.0, 1.2 * edge, 128)
        density = np.exp(-0.5 * ((energy - 0.6 * edge * (1.0 + offset)) / (0.2 * edge)) ** 2)
        density *= self.scale * interaction.N_e
        return PhasespaceSlice(axes={Axis.ENERGY: energy}, distr=density)

    def run(self, interaction, params: Parameters) -> Results:
        spectrum = self._spectrum(interaction)
        return Results(photon_slices={
            OutputKind.SPECTRUM: spectrum,
            OutputKind.TOTAL_YIELD: PhasespaceSlice(axes={}, distr=np.asarray(spectrum.integrate())),
        })


class ParameterLeakEngine(ConstantEngine):
    """Its answer depends on a numerics knob — the bug chunk/backend checks exist to find."""

    name = "leaky"

    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def run(self, interaction, params: Parameters) -> Results:
        value = params[self.field]
        offset = 1e-3 * (hash(str(value)) % 7)
        spectrum = self._spectrum(interaction, offset)
        return Results(photon_slices={
            OutputKind.SPECTRUM: spectrum,
            OutputKind.TOTAL_YIELD: PhasespaceSlice(axes={}, distr=np.asarray(spectrum.integrate())),
        })


class ParticleCountEngine(ConstantEngine):
    """Its answer scales with how many macroparticles it was handed.

    That makes it sensitive to the prefilter, which is precisely the property
    `check_prefilter_invariance` must reject — the prefilter drops particles that
    contribute nothing, and an engine whose answer moves when they go is wrong.
    """

    name = "counter"

    def run(self, interaction, params: Parameters) -> Results:
        spectrum = self._spectrum(interaction)
        scaled = spectrum.scaled(float(interaction.bunch.n_particles))
        return Results(photon_slices={
            OutputKind.SPECTRUM: scaled,
            OutputKind.TOTAL_YIELD: PhasespaceSlice(axes={}, distr=np.asarray(scaled.integrate())),
        })


@pytest.fixture(scope="module")
def small_scenario():
    """The baseline at a size a test can afford, physics unchanged."""
    return replace(scenarios.BASELINE, name="baseline",
                   sampling=replace(scenarios.BASELINE.sampling, n_particles=2000))


@pytest.fixture(scope="module")
def prefilter_scenario():
    """A bunch far wider than the pulse it meets, so the prefilter actually discards.

    The bank's own scenarios overlap completely — correct physics, and no exercise at all
    for the filter. A short pulse against a wide beam is the configuration where it bites,
    and the transverse extent is what does it: head-on, the pulse sweeps through the whole
    bunch length whatever its duration, so a *long* bunch is never enough on its own.
    """
    beam = replace(scenarios.BASELINE.beam, sigma_x=Quantity(1.0, "mm"), sigma_y=Quantity(1.0, "mm"))
    laser = replace(scenarios.BASELINE.laser, duration=Quantity(30.0, "fs"))
    return replace(scenarios.BASELINE, name="prefilter_probe", beam=beam, laser=laser,
                   sampling=replace(scenarios.BASELINE.sampling, n_particles=2000))


# ---------------------------------------------------------------------------
# The scenario bank
# ---------------------------------------------------------------------------
def test_scenario_names_are_unique_and_looked_up_by_name():
    names = [scenario.name for scenario in scenarios.SCENARIOS]
    assert len(set(names)) == len(names)
    assert scenarios.by_name("baseline") is scenarios.BASELINE
    with pytest.raises(KeyError):
        scenarios.by_name("no_such_scenario")


def test_pulse_energy_is_the_only_thing_the_scan_varies():
    for scenario in (scenarios.LOW_A0, scenarios.NEAR_A0_MAX):
        assert scenario.beam == scenarios.BASELINE.beam
        assert scenario.target == scenarios.BASELINE.target
        assert scenario.laser == replace(scenarios.BASELINE.laser,
                                         pulse_energy=scenario.laser.pulse_energy)


def test_a0_scales_as_the_square_root_of_pulse_energy():
    ratio = scenarios.NEAR_A0_MAX.laser.m("pulse_energy") / scenarios.BASELINE.laser.m("pulse_energy")
    assert scenarios.NEAR_A0_MAX.laser.a0_peak() == pytest.approx(
        scenarios.BASELINE.laser.a0_peak() * np.sqrt(ratio)
    )


def test_scenarios_build_into_a_usable_interaction(small_scenario):
    interaction = scenarios.build(small_scenario)
    assert interaction.bunch.n_particles > 0
    assert interaction.N_e == pytest.approx(small_scenario.beam.n_electrons())
    assert interaction.target is small_scenario.target


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spectrum(values, low=0.0, high=1e-5):
    energy = np.linspace(low, high, len(values))
    return PhasespaceSlice(axes={Axis.ENERGY: energy}, distr=np.asarray(values, dtype=float))


def test_identical_slices_have_zero_deviation():
    spectrum = _spectrum(np.exp(-np.linspace(-2, 2, 64) ** 2))
    deviation = metrics.compare_slices(spectrum, spectrum)
    assert deviation.worst() == pytest.approx(0.0, abs=1e-15)


def test_a_uniform_rescale_shows_up_as_exactly_that():
    reference = _spectrum(np.exp(-np.linspace(-2, 2, 64) ** 2))
    scaled = reference.scaled(1.05)
    deviation = metrics.compare_slices(scaled, reference)
    assert deviation.weighted_l1 == pytest.approx(0.05, rel=1e-6)
    assert deviation.yield_error == pytest.approx(0.05, rel=1e-6)


def test_a_localized_defect_is_caught_by_the_max_window_not_the_weighted_one():
    # The whole point of reporting two numbers: a single bad window near a feature is what
    # a weighted average is designed to hide.
    values = np.exp(-np.linspace(-2, 2, 256) ** 2)
    broken = values.copy()
    broken[8:12] *= 4.0
    deviation = metrics.compare_slices(_spectrum(broken), _spectrum(values))
    assert deviation.max_window > 20 * deviation.weighted_l1


def test_comparison_resamples_onto_the_reference_grid():
    fine = np.linspace(0.0, 1e-5, 257)
    coarse = np.linspace(0.0, 1e-5, 65)
    shape = lambda x: np.exp(-((x - 5e-6) / 2e-6) ** 2)
    deviation = metrics.compare_slices(
        PhasespaceSlice(axes={Axis.ENERGY: fine}, distr=shape(fine)),
        PhasespaceSlice(axes={Axis.ENERGY: coarse}, distr=shape(coarse)),
    )
    assert deviation.weighted_l1 < 1e-3


def test_resampling_invents_no_flux_outside_the_source_range():
    values = metrics.resample_to([0.0, 5.0, 10.0], [4.0, 6.0], [1.0, 1.0])
    assert values[0] == 0.0 and values[-1] == 0.0 and values[1] == pytest.approx(1.0)


def test_comparing_different_observables_is_an_error():
    with pytest.raises(ValueError, match="different observables"):
        metrics.compare_slices(
            _spectrum(np.ones(8)),
            PhasespaceSlice(axes={Axis.TIME: np.linspace(0, 1, 8)}, distr=np.ones(8)),
        )


def test_zero_reference_falls_back_to_the_absolute_difference():
    assert metrics.relative_error(3.0, 0.0) == 3.0


# ---------------------------------------------------------------------------
# Golden snapshots
# ---------------------------------------------------------------------------
def test_golden_round_trips_through_disk(tmp_path):
    results = Results(photon_slices={OutputKind.SPECTRUM: _spectrum(np.arange(1.0, 9.0))})
    provenance = Provenance(model="fake", scenario="probe", source_repo="/nowhere",
                            source_commit="0" * 40, source_dirty=False,
                            generated_utc="2026-08-07T00:00:00+00:00", note="test")
    save_golden(results, provenance, {"gamma0": 2000.0}, directory=tmp_path)

    loaded = golden_module.load_golden("probe", "fake", directory=tmp_path)
    assert loaded.provenance == provenance
    assert loaded.scalars == {"gamma0": 2000.0}
    assert loaded.results.photon_slices[OutputKind.SPECTRUM].integrate() == pytest.approx(
        results.photon_slices[OutputKind.SPECTRUM].integrate()
    )
    assert golden_module.available_goldens(tmp_path) == [("probe", "fake")]


def test_a_missing_golden_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_references"):
        golden_module.load_golden("probe", "absent", directory=tmp_path)


def test_every_scenario_has_a_golden_from_every_predecessor_model():
    from gammaforge.validation.make_references import MODELS

    expected = {(scenario.name, model) for scenario in scenarios.SCENARIOS for model in MODELS}
    assert set(available_goldens()) >= expected


def test_committed_goldens_were_generated_from_a_clean_tree():
    for scenario_name, model in available_goldens():
        provenance = load_golden(scenario_name, model).provenance
        assert not provenance.source_dirty, f"{scenario_name}/{model} came from a dirty tree"
        assert len(provenance.source_commit) == 40


def test_goldens_agree_with_this_repos_closed_forms():
    """The real cross-implementation check available today, and the sharpest one.

    Two independently written codebases, the same physical input, the same analytic
    quantity. Nothing statistical is involved, so a disagreement here means a constant,
    a unit or a convention differs — hence the tight bound.
    """
    compared = 0
    for scenario_name, model in available_goldens():
        golden = load_golden(scenario_name, model)
        fresh = derived_scalars(scenarios.by_name(scenario_name))
        for name, reference in golden.scalars.items():
            assert fresh[name] == pytest.approx(reference, rel=1e-9), f"{scenario_name}/{model} {name}"
            compared += 1
    assert compared > 0, "no golden carries any scalar to compare"


def test_the_predecessors_two_tabulated_methods_agree_with_each_other():
    """A property of the reference data itself, asserted so a bad regeneration is loud.

    xigma and delta share Stage 0 and differ in the Stage-2 kernel (§4.5), so agreeing to
    ~1e-7 is expected — and a regeneration that broke one of them would not be.
    """
    for scenario in scenarios.SCENARIOS:
        xigma = load_golden(scenario.name, "xigma").results.photon_slices[OutputKind.SPECTRUM]
        delta = load_golden(scenario.name, "delta").results.photon_slices[OutputKind.SPECTRUM]
        assert metrics.compare_slices(xigma, delta).weighted_l1 < 1e-6


def test_golden_spectra_carry_energies_in_erg_around_the_compton_edge():
    """Guards the unit translation in `make_references`, which is easy to get backwards."""
    from gammaforge.io.laser import fit_gaussian_paraxial

    scenario = scenarios.BASELINE
    edge = compton_edge_energy(scenario.beam, fit_gaussian_paraxial(scenario.laser).photon_energy())
    spectrum = load_golden("baseline", "xigma").results.photon_slices[OutputKind.SPECTRUM]
    top = spectrum.axes[Axis.ENERGY].max()
    assert 0.5 * edge < top < 3.0 * edge


def test_golden_yields_are_linear_in_pulse_energy_at_low_a0():
    """Physics the snapshots must already satisfy: at a0 << 1 the yield is linear in N_l."""
    baseline = load_golden("baseline", "xigma").results.photon_slices[OutputKind.TOTAL_YIELD]
    low = load_golden("low_a0", "xigma").results.photon_slices[OutputKind.TOTAL_YIELD]
    ratio = scenarios.BASELINE.laser.m("pulse_energy") / scenarios.LOW_A0.laser.m("pulse_energy")
    assert float(baseline.distr) == pytest.approx(float(low.distr) * ratio, rel=1e-3)


# ---------------------------------------------------------------------------
# make_references' boundary translation
# ---------------------------------------------------------------------------
def test_the_payload_describes_the_scenario_in_the_old_repos_units():
    payload = _scenario_payload(scenarios.BASELINE, n_energy_bins=64)
    assert payload["beam"]["bunch_charge_C"] == pytest.approx(10e-9)
    assert payload["beam"]["sigma_x_m"] == pytest.approx(10e-6)
    assert payload["laser"]["wavelength_m"] == pytest.approx(1030e-9)
    assert payload["laser"]["pulse_energy_J"] == pytest.approx(20.0)
    assert payload["beam"]["kinetic_energy_eV"] == pytest.approx(
        (2000.0 - 1.0) * 510998.9, rel=1e-5
    )


def test_every_axis_and_scalar_the_old_repo_reports_has_a_translation():
    assert set(_AXIS_TRANSLATION) == {"E_eV", "t_seconds", "x", "y", "theta_x", "theta_y"}
    assert set(_SCALAR_TRANSLATION.values()) <= set(derived_scalars(scenarios.BASELINE))
    # Every grouping an old slice can have resolves to exactly one OutputKind.
    assert frozenset({Axis.ENERGY}) in _KIND_BY_AXES
    assert _KIND_BY_AXES[frozenset()] is OutputKind.TOTAL_YIELD
    assert OutputKind.COLLIMATED_SPECTRUM not in _KIND_BY_AXES.values()


# ---------------------------------------------------------------------------
# Invariance properties of the core
# ---------------------------------------------------------------------------
def test_bunch_seed_determinism_holds(small_scenario):
    check = invariance.check_bunch_seed_determinism(small_scenario)
    assert check.passed, str(check)


def test_the_prefilter_discards_only_particles_the_pulse_never_reaches(prefilter_scenario):
    check = invariance.check_prefilter_discards_only_dark_particles(prefilter_scenario, n_times=256)
    assert check.passed, str(check)
    # If nothing were discarded the check would pass vacuously, which would make this test
    # a decoration rather than a test.
    assert "discarded" in check.detail and not check.detail.startswith("the pulse reaches all")


def test_the_bank_itself_reports_full_overlap():
    for check in invariance.core_checks(scenarios.SCENARIOS[:1]):
        assert check.passed, str(check)


# ---------------------------------------------------------------------------
# The engine-facing harness, against stubs
# ---------------------------------------------------------------------------
def test_the_stub_satisfies_the_engine_protocol():
    assert isinstance(ConstantEngine(), Engine)
    assert not isinstance(object(), Engine)


def test_running_something_that_is_not_an_engine_says_so(small_scenario):
    with pytest.raises(TypeError, match="Engine protocol"):
        run_engine(object(), small_scenario)


def test_a_run_reports_what_it_ran(small_scenario):
    run = run_engine(ConstantEngine(), small_scenario)
    assert run.engine == "stub" and run.scenario == small_scenario.name
    assert run.interaction.bunch.n_particles == small_scenario.sampling.n_particles
    assert set(run.results.photon_slices) == {OutputKind.SPECTRUM, OutputKind.TOTAL_YIELD}


def test_run_bank_covers_every_scenario(small_scenario):
    runs = run_bank(ConstantEngine(), [small_scenario, replace(small_scenario, name="again")])
    assert [run.scenario for run in runs] == [small_scenario.name, "again"]


def test_an_invariant_engine_passes_every_invariance_leg(small_scenario):
    engine = ConstantEngine()
    for check in invariance.engine_checks(engine, [small_scenario]):
        assert check.passed, str(check)
    for field, values in [("chunk", [1024, 4096]), ("device", ["numpy", "numba"])]:
        checker = (invariance.check_chunk_invariance if field == "chunk"
                   else invariance.check_backend_agreement)
        check = checker(engine, small_scenario, field, values)
        assert check.passed, str(check)


def test_an_engine_whose_answer_depends_on_a_numerics_knob_is_caught(small_scenario):
    check = invariance.check_chunk_invariance(
        ParameterLeakEngine("chunk"), small_scenario, "chunk", [1024, 4096, 8192]
    )
    assert not check.passed
    assert "chunk=" in check.detail


def test_a_backend_disagreement_larger_than_tolerance_is_caught(small_scenario):
    check = invariance.check_backend_agreement(
        ParameterLeakEngine("device"), small_scenario, "device", ["numpy", "cupy", "numba"]
    )
    assert not check.passed


def test_prefilter_sensitivity_is_caught(prefilter_scenario):
    check = invariance.check_prefilter_invariance(ParticleCountEngine(), prefilter_scenario)
    assert not check.passed, "an engine whose answer scales with particle count must fail"


def test_a_check_over_an_unknown_or_single_valued_parameter_fails_loudly(small_scenario):
    engine = ConstantEngine()
    assert not invariance.check_chunk_invariance(engine, small_scenario, "nope", [1, 2]).passed
    assert not invariance.check_chunk_invariance(engine, small_scenario, "chunk", [1]).passed


def test_results_with_different_outputs_never_compare_as_equal():
    full = Results(photon_slices={OutputKind.SPECTRUM: _spectrum(np.ones(4))})
    empty = Results(photon_slices={})
    passed, detail = invariance.compare_results(full, empty, invariance.EXACT_TOLERANCE)
    assert not passed and "different outputs" in detail


# ---------------------------------------------------------------------------
# Golden comparison, end to end
# ---------------------------------------------------------------------------
def test_an_engine_reproducing_a_golden_passes_and_a_shifted_one_fails(tmp_path, small_scenario):
    engine = ConstantEngine()
    results = run_engine(engine, small_scenario).results
    provenance = Provenance(model="stub", scenario=small_scenario.name, source_repo="test",
                            source_commit="0" * 40, source_dirty=False,
                            generated_utc="2026-08-07T00:00:00+00:00")
    save_golden(results, provenance, derived_scalars(small_scenario), directory=tmp_path)
    golden = golden_module.load_golden(small_scenario.name, "stub", directory=tmp_path)

    same = compare_to_golden(results, golden, scalars=derived_scalars(small_scenario))
    assert same.passed, same.summary()

    shifted = run_engine(ConstantEngine(scale=1.5), small_scenario).results
    assert not compare_to_golden(shifted, golden).passed


def test_an_output_the_engine_does_not_produce_is_reported_not_failed(tmp_path, small_scenario):
    results = run_engine(ConstantEngine(), small_scenario).results
    provenance = Provenance(model="stub", scenario=small_scenario.name, source_repo="test",
                            source_commit="0" * 40, source_dirty=False,
                            generated_utc="2026-08-07T00:00:00+00:00")
    save_golden(results, provenance, directory=tmp_path)
    golden = golden_module.load_golden(small_scenario.name, "stub", directory=tmp_path)

    partial = Results(photon_slices={OutputKind.SPECTRUM: results.photon_slices[OutputKind.SPECTRUM]})
    comparison = compare_to_golden(partial, golden)
    assert comparison.passed
    assert comparison.missing == (OutputKind.TOTAL_YIELD,)
    assert "not produced by this engine" in comparison.summary()


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
def test_the_suite_runs_green_with_no_engines(small_scenario):
    report = run_suite(scenarios=[small_scenario])
    text = str(report)
    assert report.failures == 0, text
    assert "no engines registered yet" in text
    assert "ALL CHECKS PASS" in text


def test_the_suite_reports_engine_failures(prefilter_scenario):
    report = run_suite(engines=[ParticleCountEngine()], scenarios=[prefilter_scenario])
    assert report.failures > 0
    assert "CHECK(S) FAILED" in str(report)

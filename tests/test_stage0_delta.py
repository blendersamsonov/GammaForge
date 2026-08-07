"""Stage 0, the shared chunking utility, and delta (GRAND_PLAN.md §4.2/§4.5, Phase 2.5 exit).

The physics assertions here are the ones §7 asks for: closed-form identities where the
contract guarantees them, invariance where a knob must not matter, and convergence where
a discretization must vanish. Where two methods genuinely disagree — the ``2 pi`` of §9.1
— the test pins the *derived* value rather than the one that would be convenient, so the
disagreement stays visible instead of being absorbed into a tolerance.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import chunking
from gammaforge.engines.xigma.stages import (
    RELATIVE_VELOCITY,
    TrajectorySamples,
    integrate_trajectories,
    photon_density_scale,
)
from gammaforge.io.interaction import PREFILTER_OFF
from gammaforge.io.target import OutputKind
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.golden import load_golden
from gammaforge.validation.references import delta

N_STEPS = 64


def _samples(scenario, n_particles=2000, **kwargs):
    small = replace(scenario, sampling=replace(scenario.sampling, n_particles=n_particles))
    interaction = scenarios.build(small)
    return integrate_trajectories(
        interaction.bunch, interaction.laser, interaction.N_e, n_steps=N_STEPS, **kwargs
    )


@pytest.fixture(scope="module")
def baseline():
    return _samples(scenarios.BASELINE)


# ---------------------------------------------------------------------------
# The shared chunking utility (§4.2)
# ---------------------------------------------------------------------------
def test_chunks_partition_exactly_once():
    seen = []
    chunking.run_in_chunks(10, lambda start, stop: seen.append((start, stop)), chunk=3)
    assert seen == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert chunking.run_in_chunks(0, lambda start, stop: 1, chunk=3) == []


def test_results_come_back_in_order():
    values = chunking.run_in_chunks(7, lambda start, stop: (start, stop), chunk=2)
    assert values == [(0, 2), (2, 4), (4, 6), (6, 7)]


def test_an_out_of_memory_failure_halves_the_chunk_and_retries_the_same_slice():
    """The retried slice must be the failed one — no item skipped, none done twice."""
    attempted, completed = [], []

    def work(start, stop):
        attempted.append((start, stop))
        if stop - start > 2:
            raise MemoryError("too big")
        completed.append((start, stop))
        return stop - start

    sizes = chunking.run_in_chunks(6, work, chunk=8)
    assert sum(sizes) == 6, "every item processed exactly once"
    assert [start for start, _ in completed] == sorted(start for start, _ in completed)
    # 8 is clamped to the 6 available, then halved until the work stops refusing it.
    assert attempted[0] == (0, 6) and attempted[1] == (0, 3) and attempted[2] == (0, 1)
    assert all(stop - start <= 2 for start, stop in completed)


def test_a_persistent_out_of_memory_failure_is_re_raised():
    def always_fails(start, stop):
        raise MemoryError("never fits")

    with pytest.raises(MemoryError):
        chunking.run_in_chunks(1024, always_fails, chunk=1024)


def test_estimate_chunk_respects_the_ceiling_and_never_returns_zero():
    assert chunking.estimate_chunk(10_000, 1, "numpy", ceiling=32) <= 32
    assert chunking.estimate_chunk(10_000, 10**18, "numpy") >= 1
    assert chunking.estimate_chunk(0, 1, "numpy") == 1
    with pytest.raises(ValueError, match="backend must be"):
        chunking.estimate_chunk(10, 1, "opencl")


def test_an_unmeasurable_machine_is_not_chunked(monkeypatch):
    # Guessing small on a machine that declines to be measured is a large silent slowdown.
    monkeypatch.setattr(chunking, "available_ram_bytes", lambda: None)
    assert chunking.estimate_chunk(5000, 10**9, "numpy") == 5000
    assert chunking.estimate_chunk(5000, 10**9, "numpy", ceiling=64) == 64


def test_memory_queries_answer_or_say_they_cannot():
    for value in (chunking.available_ram_bytes(), chunking.available_vram_bytes()):
        assert value is None or value > 0


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------
def test_stage_0_produces_one_sample_per_macroparticle(baseline):
    assert baseline.n_particles == 2000
    for values in (baseline.gamma, baseline.theta_x, baseline.theta_y,
                   baseline.a0_shape, baseline.luminosity):
        assert values.shape == (2000,)
    assert np.all(baseline.luminosity > 0.0)
    assert np.all(np.isfinite(baseline.a0_shape))
    assert baseline.total_yield() > 0.0


def test_the_chunk_size_cannot_change_the_answer():
    """§7's chunk invariance, now against a real stage rather than a stub.

    Exactly equal, not merely close: chunking partitions particles, whose trajectories are
    independent, and each particle's own reduction runs identically whichever slice it
    lands in. A tolerance here would hide a real repartitioning bug.
    """
    reference = _samples(scenarios.BASELINE, n_particles=1500)
    for chunk in (1, 7, 499, 1500, 100_000):
        other = _samples(scenarios.BASELINE, n_particles=1500, chunk=chunk)
        assert np.array_equal(other.luminosity, reference.luminosity), chunk
        assert np.array_equal(other.a0_shape, reference.a0_shape), chunk


def test_the_prefilter_cannot_change_the_answer():
    """§3.2's central claim, testable for the first time now that a stage exists.

    The prefilter discards particles that contribute nothing, so the total yield must not
    move — and it must not move *because the weights are never renormalized*, which is the
    part that would be easy to get wrong.
    """
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=2000))
    with_filter = _samples(scenario)
    without = _samples(replace(scenario,
                               sampling=replace(scenario.sampling, prefilter=PREFILTER_OFF)))
    assert with_filter.total_yield() == pytest.approx(without.total_yield(), rel=1e-12)


def test_a_wide_bunch_still_gives_the_same_yield_with_and_without_the_prefilter():
    """The same property where the prefilter actually discards most of the bunch."""
    scenario = replace(
        scenarios.BASELINE,
        name="prefilter_probe",
        beam=replace(scenarios.BASELINE.beam, sigma_x=Quantity(1.0, "mm"), sigma_y=Quantity(1.0, "mm")),
        laser=replace(scenarios.BASELINE.laser, duration=Quantity(30.0, "fs")),
        sampling=replace(scenarios.BASELINE.sampling, n_particles=2000),
    )
    filtered = _samples(scenario)
    unfiltered = _samples(replace(scenario,
                                  sampling=replace(scenario.sampling, prefilter=PREFILTER_OFF)))
    assert filtered.n_particles < unfiltered.n_particles
    assert filtered.total_yield() == pytest.approx(unfiltered.total_yield(), rel=1e-9)


def test_the_yield_converges_in_the_number_of_steps():
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=1000))
    interaction = scenarios.build(scenario)
    yields = [
        integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e,
                               n_steps=n).total_yield()
        for n in (16, 64, 256)
    ]
    # The statement worth making is that the sequence converges, not that any one step is
    # small: the midpoint rule's error falls fast enough that 64 and 256 agree to parts in
    # a million while 16 is still visibly coarse.
    coarse_step = abs(yields[1] / yields[0] - 1.0)
    fine_step = abs(yields[2] / yields[1] - 1.0)
    assert coarse_step < 1e-2
    assert fine_step < coarse_step / 100.0
    assert yields[2] == pytest.approx(yields[1], rel=1e-6)


def test_the_yield_is_exactly_linear_in_charge(baseline):
    """§3.5: no space charge, so every output scales with N_e — the QUERY_ONLY rescale."""
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=1000))
    interaction = scenarios.build(scenario)
    single = integrate_trajectories(interaction.bunch, interaction.laser,
                                    interaction.N_e, n_steps=N_STEPS)
    doubled = integrate_trajectories(interaction.bunch, interaction.laser,
                                     2.0 * interaction.N_e, n_steps=N_STEPS)
    assert doubled.total_yield() == pytest.approx(2.0 * single.total_yield(), rel=1e-14)
    assert np.array_equal(doubled.a0_shape, single.a0_shape)  # a shape, not a count


def test_the_yield_is_linear_in_pulse_energy_and_a0_shape_is_not():
    """The retarget claim (§5): ``a0_shape`` carries no a0, so it survives an energy change.

    Yield goes as the photon density and therefore as the pulse energy; ``a0_shape`` is a
    ratio of envelope moments, in which the pulse energy cancels, so it must come out
    unchanged to round-off. That is what allows a pulse-energy edit to reuse Stage 0
    instead of rerunning it.
    """
    baseline_samples = _samples(scenarios.BASELINE, n_particles=1000)
    brighter = _samples(scenarios.NEAR_A0_MAX, n_particles=1000)
    ratio = (scenarios.NEAR_A0_MAX.laser.m("pulse_energy")
             / scenarios.BASELINE.laser.m("pulse_energy"))
    assert brighter.total_yield() == pytest.approx(ratio * baseline_samples.total_yield(), rel=1e-12)
    assert brighter.a0_shape == pytest.approx(baseline_samples.a0_shape, rel=1e-12)
    assert brighter.ahat() == pytest.approx(ratio * baseline_samples.ahat(), rel=1e-12)


def test_retargeting_ahat_matches_running_the_other_pulse(baseline):
    other = _samples(scenarios.NEAR_A0_MAX)
    assert baseline.retargeted_ahat(other.a0_peak) == pytest.approx(other.ahat(), rel=1e-12)


def test_the_photon_density_scale_inverts_the_lasers_own_a0_chain():
    """Stage 0 reads the whole laser through ``a0_profile``; this is why that is enough."""
    laser = scenarios.BASELINE.laser
    scale = photon_density_scale(laser)
    for point in [(0.0, 0.0, 0.0, 0.0), (5e-4, 3e-4, 0.02, 1e-12), (2e-3, 0.0, -0.5, -2e-11)]:
        direct = laser.n_photons() * laser.photon_density(*point)
        assert scale * laser.a0_profile(*point) ** 2 == pytest.approx(direct, rel=1e-14)


def test_stage_0_agrees_with_the_predecessors_total_yield():
    """New-vs-golden, on a real computation — what Phase 2's machinery was built for.

    A fraction of a percent, and the residual is *systematic*: the two repos draw
    different bunches from the same nominal seed and bound the interaction window
    differently, so this is agreement to the accuracy the comparison can support, not a
    coincidence to tighten later.
    """
    for name in ("baseline", "low_a0", "near_a0_max"):
        scenario = scenarios.by_name(name)
        interaction = scenarios.build(scenario)
        computed = integrate_trajectories(interaction.bunch, interaction.laser,
                                          interaction.N_e, n_steps=N_STEPS).total_yield()
        golden = float(load_golden(name, "xigma").results
                       .photon_slices[OutputKind.TOTAL_YIELD].distr)
        assert computed == pytest.approx(golden, rel=5e-3), name


def test_an_unbuilt_backend_says_so_rather_than_running_on_the_host():
    interaction = scenarios.build(
        replace(scenarios.BASELINE,
                sampling=replace(scenarios.BASELINE.sampling, n_particles=10))
    )
    with pytest.raises(NotImplementedError, match="numpy-only until"):
        integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e,
                               backend="cupy")
    with pytest.raises(ValueError, match="n_steps"):
        integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e,
                               n_steps=0)


def test_the_relative_velocity_factor_is_the_head_on_one():
    assert RELATIVE_VELOCITY == 2.0


# ---------------------------------------------------------------------------
# delta (§4.5)
# ---------------------------------------------------------------------------
def test_the_closed_form_spectrum_integrates_to_the_total_yield_exactly(baseline):
    """An identity, not a tolerance (§7): the single-electron shape integrates to 1.

    This is what makes the closed form the anchor — it reproduces Stage 0's own count with
    no free constant, so any method that does not agree with it disagrees with a photon
    count, not with a convention.
    """
    edge = float(np.max(baseline.gamma) ** 2)
    s = np.linspace(0.0, edge, 20001)
    assert np.trapezoid(delta.single_electron_spectrum(baseline, s), s) == pytest.approx(
        baseline.total_yield(), rel=1e-4
    )


def test_delta_produces_a_spectrum_peaked_below_the_compton_edge(baseline):
    edge = float(np.mean(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 129)
    on_axis = delta.resonance_spectrum(baseline, s_edges, 0.0, 0.0)
    assert on_axis.shape == (128,)
    assert np.all(np.isfinite(on_axis))
    # Viewed head-on, every particle resonates within its own redshift of the edge.
    assert on_axis[:100].sum() < on_axis[100:].sum()


def test_delta_off_axis_is_redshifted_relative_to_on_axis(baseline):
    edge = float(np.mean(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 129)
    centres = 0.5 * (s_edges[:-1] + s_edges[1:])

    def mean_energy(theta):
        spectrum = delta.resonance_spectrum(baseline, s_edges, theta, 0.0)
        return float(np.sum(spectrum * centres) / np.sum(spectrum))

    assert mean_energy(2.0 / float(np.mean(baseline.gamma))) < mean_energy(0.0)


def test_delta_overcounts_stage_0_by_exactly_two_pi(baseline):
    """§9.1, reduced to a derived number.

    ``int dOmega`` of delta's prefactor is ``2 pi`` analytically (see the module
    docstring), so this pins the *derivation*, not an observation. The predecessor
    recorded the same ratio as "~6.3x ... not yet explained"; reproducing it from an
    independent CGS implementation rules out its coordinate normalization as the cause.

    The tolerance covers the angular grid's truncation, which `expected_ratio` already
    corrects for approximately — a square grid reaches past the disc the correction
    assumes, and the beam's own divergence broadens the distribution slightly.
    """
    check = delta.check_normalization(baseline, n_angles=65, cone_factor=8.0)
    # The anchor is exactly 1 in the continuum; the ~0.6% here is the trapezoid error of
    # the shared 128-bin grid, which both sides of the ratio are computed on.
    assert check.anchor_ratio == pytest.approx(1.0, rel=1e-2)
    assert check.ratio == pytest.approx(2.0 * math.pi * check.captured_fraction, rel=2e-2)
    assert abs(check.deviation) < 2e-2


def test_the_two_pi_is_stable_as_the_angular_grid_is_refined(baseline):
    """A constant that survives refinement is a constant, not a discretization artefact."""
    coarse = delta.check_normalization(baseline, n_angles=17, cone_factor=4.0)
    fine = delta.check_normalization(baseline, n_angles=65, cone_factor=4.0)
    assert fine.ratio == pytest.approx(coarse.ratio, rel=2e-3)


def test_widening_the_cone_captures_more_and_moves_towards_two_pi(baseline):
    ratios = [delta.check_normalization(baseline, n_angles=n, cone_factor=c).ratio
              for c, n in ((4.0, 33), (8.0, 65))]
    assert ratios[0] < ratios[1] < 2.0 * math.pi


def test_delta_is_linear_in_charge(baseline):
    doubled = TrajectorySamples(
        gamma=baseline.gamma, theta_x=baseline.theta_x, theta_y=baseline.theta_y,
        a0_shape=baseline.a0_shape, luminosity=2.0 * baseline.luminosity,
        a0_peak=baseline.a0_peak, n_steps=baseline.n_steps,
    )
    edge = float(np.max(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 65)
    assert delta.resonance_spectrum(doubled, s_edges, 0.0, 0.0) == pytest.approx(
        2.0 * delta.resonance_spectrum(baseline, s_edges, 0.0, 0.0), rel=1e-14
    )

"""Stage 0, the shared chunking utility, and delta (GRAND_PLAN.md §4.2/§4.5, Phase 2.5 exit).

The physics assertions here are the ones §7 asks for: closed-form identities where the
contract guarantees them, invariance where a knob must not matter, and convergence where
a discretization must vanish. The ``2 pi`` of §9.1 used to be the exception — two methods
that genuinely disagreed, with the test pinning the *derived* value rather than the
convenient one so the disagreement stayed visible. Phase 3b closed it (RES033), so those tests now pin one; what they still do is fail loudly if the factor comes
back, which is the same job under a different expected number.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier2]

from gammaforge.engines.xigma import chunking
from gammaforge.engines.xigma.stages import (
    relative_velocity,
    doppler_factor_per_particle,
    TrajectorySamples,
    ahat_from_shape,
    integrate_trajectories,
    photon_density_scale,
    polarization_factor,
    polarization_factor_vectorized,
)
from gammaforge.io.interaction import PREFILTER_OFF
from gammaforge.io.target import OutputKind
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.golden import load_golden
from gammaforge.validation.references import delta

N_STEPS = 64


def _polarization_factor_from_udef(
    gamma, theta_x, theta_y, theta_x_obs, theta_y_obs,
    ellipticity, psi_pol, theta_xz=0.0, theta_yz=0.0,
):
    """Direct lab-frame evaluation of manuscript Eq. (udef), independent of the kernel."""
    cos_xz, cos_yz = math.cos(theta_xz), math.cos(theta_yz)
    sin_xz, sin_yz = math.sin(theta_xz), math.sin(theta_yz)
    rotation = np.array([
        [cos_xz, sin_xz * sin_yz, sin_xz * cos_yz],
        [0.0, cos_yz, -sin_yz],
        [-sin_xz, cos_xz * sin_yz, cos_xz * cos_yz],
    ])
    e0_raw = rotation @ np.array([math.cos(psi_pol), math.sin(psi_pol), 0.0])
    e1_raw = rotation @ np.array([-math.sin(psi_pol), math.cos(psi_pol), 0.0])
    u_e = np.array([theta_x, theta_y, 1.0])
    u_e /= math.sqrt(1.0 + theta_x**2 + theta_y**2)
    u_dot_e0 = np.dot(u_e, e0_raw)
    p0 = e0_raw - u_dot_e0 * u_e
    e0 = p0 / np.linalg.norm(p0)
    e1 = np.cross(u_e, e0)
    if np.dot(e1, e1_raw) < 0.0:
        e1 = -e1

    n = np.array([theta_x_obs, theta_y_obs, 1.0])
    n /= np.linalg.norm(n)
    beta = math.sqrt(1.0 - gamma**-2)
    v = beta * u_e
    one_minus_vn = 1.0 - np.dot(v, n)
    u0 = (n - v) * np.dot(n, e0) / one_minus_vn - e0
    u1 = (n - v) * np.dot(n, e1) / one_minus_vn - e1
    eps2 = ellipticity**2
    return (np.dot(u0, u0) + eps2 * np.dot(u1, u1)) / (1.0 + eps2)


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


@pytest.mark.parametrize("ellipticity", [0.0, 0.3, 1.0])
def test_stage_0_is_bit_identical_under_any_polarization(ellipticity):
    """RES054's central claim, asserted as an **invariance** rather than as
    the value of a constant.

    At fixed pulse energy the cycle-averaged intensity ``<a^2>`` does not depend on the
    polarization state: the elliptical ``a0`` is smaller by ``sqrt(2C)`` while its cycle
    average is larger by ``C``, and the two offset exactly. Since every quantity Stage 0
    produces — ``luminosity``, ``a0_shape``, and therefore ``ahat`` — is a functional of
    ``<a^2>`` alone, all of them must come out **bit-identical** from linear to circular.

    This is a stronger and more useful statement than pinning ``C = 1/2`` was: it fails if
    anyone reintroduces a polarization factor anywhere on the yield or red-shift path, in
    either direction, without needing to know where they put it. `test_laser.py` covers the
    other side — that ``a0_peak`` itself genuinely *does* move with ``ellipticity``, so
    this invariance is a real cancellation and not both sides being constant.
    """
    linear = _samples(scenarios.BASELINE, n_particles=1500)
    scenario = replace(
        scenarios.BASELINE, laser=replace(scenarios.BASELINE.laser, ellipticity=ellipticity)
    )
    polarized = _samples(scenario, n_particles=1500)

    assert polarized.intensity_peak == pytest.approx(linear.intensity_peak, rel=1e-14)
    assert np.array_equal(polarized.luminosity, linear.luminosity)
    assert np.array_equal(polarized.a0_shape, linear.a0_shape)
    assert np.array_equal(polarized.ahat(), linear.ahat())


def test_ahat_is_a_plain_product_of_shape_and_peak_intensity(baseline):
    """No cycle-average factor survives in `ahat_from_shape` (RES054).

    The paper's ``ahat = (a0^2 Tr Xi / 2) int|E|^4 / int|E|^2`` becomes
    ``<a^2>_peak * int|E|^4 / int|E|^2`` once ``<a^2> = C a0^2`` is substituted, so with
    Stage 0 already carrying ``<a^2>`` there is nothing left to apply.

    The second half is the reason the count is not double-corrected:
    `photon_density_scale` inverts the same energy→intensity chain, so ``luminosity`` is
    untouched by anything that moves ``ahat``.
    """
    assert baseline.ahat() == pytest.approx(baseline.intensity_peak * baseline.a0_shape, rel=1e-14)
    assert ahat_from_shape(baseline.a0_shape, baseline.intensity_peak) == pytest.approx(
        baseline.ahat(), rel=1e-14
    )

    shifted = replace(baseline, a0_shape=2.0 * baseline.a0_shape)
    assert shifted.total_yield() == pytest.approx(baseline.total_yield(), rel=1e-14)
    assert shifted.ahat() == pytest.approx(2.0 * baseline.ahat(), rel=1e-14)


def test_retargeting_ahat_matches_running_the_other_pulse(baseline):
    other = _samples(scenarios.NEAR_A0_MAX)
    assert baseline.retargeted_ahat(other.intensity_peak) == pytest.approx(other.ahat(), rel=1e-12)


@pytest.mark.parametrize("ellipticity", [0.0, 0.5, 1.0])
def test_the_photon_density_scale_inverts_the_lasers_own_intensity_chain(ellipticity):
    """Stage 0 reads the whole laser through ``intensity_profile``; this is why that is
    enough — and why the conversion needs no polarization input (RES054).

    Parametrized over ``ellipticity`` deliberately: the identity is exact for every
    polarization state because both sides are built from the same photon density, which is
    what makes `photon_density_scale` a pure ``4 pi`` conversion with no ``C`` in it.
    """
    laser = replace(scenarios.BASELINE.laser, ellipticity=ellipticity)
    scale = photon_density_scale(laser)
    for point in [(0.0, 0.0, 0.0, 0.0), (5e-4, 3e-4, 0.02, 1e-12), (2e-3, 0.0, -0.5, -2e-11)]:
        direct = laser.n_photons() * laser.photon_density(*point)
        assert scale * laser.intensity_profile(*point) == pytest.approx(direct, rel=1e-14)


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


def test_integrate_trajectories_validates_n_steps():
    interaction = scenarios.build(
        replace(scenarios.BASELINE,
                sampling=replace(scenarios.BASELINE.sampling, n_particles=10))
    )
    with pytest.raises(ValueError, match="n_steps"):
        integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e,
                               n_steps=0)


def test_the_relative_velocity_factor_is_the_head_on_one():
    assert relative_velocity(beta=1.0) == 2.0


def test_doppler_factor_per_particle_head_on():
    """Per-particle Doppler factor for head-on collision with zero divergence."""
    from gammaforge.engines.xigma.stages import doppler_factor_per_particle

    gamma = np.array([2.0, 2000.0, 10000.0])
    theta_x = np.zeros_like(gamma)
    theta_y = np.zeros_like(gamma)

    factors = doppler_factor_per_particle(gamma, theta_x, theta_y, theta_xz=0.0, theta_yz=0.0)

    # For head-on, zero divergence: factor = (1 + beta) / 2
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
    expected = (1.0 + beta) / 2.0

    np.testing.assert_allclose(factors, expected, rtol=1e-12, atol=1e-15)


def test_doppler_factor_per_particle_with_divergence():
    """Per-particle Doppler factor with non-zero electron divergence."""
    from gammaforge.engines.xigma.stages import doppler_factor_per_particle

    gamma = np.array([2000.0, 2000.0, 2000.0])
    theta_x = np.array([0.0, 0.001, -0.0005])
    theta_y = np.array([0.0, -0.0006, 0.0003])

    factors = doppler_factor_per_particle(gamma, theta_x, theta_y, theta_xz=0.0, theta_yz=0.0)

    # For head-on: n0 = [0, 0, -1], so 1 - v.n0 = 1 + vz
    # vz = beta / sqrt(1 + theta_x^2 + theta_y^2)
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
    norm = np.sqrt(1.0 + theta_x**2 + theta_y**2)
    vz = beta / norm
    one_minus_v_dot_n0 = 1.0 + vz
    nominal = 2.0  # 1 + cos(0)*cos(0) = 2
    expected = one_minus_v_dot_n0 / nominal

    np.testing.assert_allclose(factors, expected, rtol=1e-12, atol=1e-15)


def test_doppler_factor_per_particle_crossing_angle():
    """Per-particle Doppler factor with crossing angles."""
    from gammaforge.engines.xigma.stages import doppler_factor_per_particle

    gamma = np.array([2000.0, 2000.0])
    theta_x = np.array([0.0, 0.001])
    theta_y = np.array([0.0, -0.0006])
    theta_xz = 0.3
    theta_yz = 0.2

    factors = doppler_factor_per_particle(gamma, theta_x, theta_y, theta_xz=theta_xz, theta_yz=theta_yz)

    # Verify against manual calculation
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
    norm = np.sqrt(1.0 + theta_x**2 + theta_y**2)
    vx = beta * theta_x / norm
    vy = beta * theta_y / norm
    vz = beta / norm

    cos_xz = math.cos(theta_xz)
    cos_yz = math.cos(theta_yz)
    sin_xz = math.sin(theta_xz)
    sin_yz = math.sin(theta_yz)
    n0 = np.array([-sin_xz * cos_yz, sin_yz, -cos_xz * cos_yz])

    v_dot_n0 = vx * n0[0] + vy * n0[1] + vz * n0[2]
    one_minus_v_dot_n0 = 1.0 - v_dot_n0
    nominal = 1.0 + cos_xz * cos_yz
    expected = one_minus_v_dot_n0 / nominal

    np.testing.assert_allclose(factors, expected, rtol=1e-12, atol=1e-15)


def test_doppler_factor_per_particle_k_hat():
    """Per-particle Doppler factor with explicit k_hat."""
    from gammaforge.engines.xigma.stages import doppler_factor_per_particle

    gamma = np.array([2000.0, 2000.0])
    theta_x = np.array([0.0, 0.001])
    theta_y = np.array([0.0, -0.0006])
    k_hat = np.array([-0.1, 0.05, -0.99])

    factors = doppler_factor_per_particle(gamma, theta_x, theta_y, k_hat=k_hat)

    # Verify against manual calculation
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
    norm = np.sqrt(1.0 + theta_x**2 + theta_y**2)
    vx = beta * theta_x / norm
    vy = beta * theta_y / norm
    vz = beta / norm

    v_dot_n0 = vx * k_hat[0] + vy * k_hat[1] + vz * k_hat[2]
    one_minus_v_dot_n0 = 1.0 - v_dot_n0
    nominal = 1.0 - k_hat[2]  # beam axis (0,0,1) dot k_hat
    expected = one_minus_v_dot_n0 / nominal

    np.testing.assert_allclose(factors, expected, rtol=1e-12, atol=1e-15)


def test_doppler_factor_per_particle_input_validation():
    """Test input validation for per-particle Doppler factor."""
    from gammaforge.engines.xigma.stages import doppler_factor_per_particle

    gamma = np.array([2000.0, 2000.0])
    theta_x = np.array([0.0, 0.001])
    theta_y = np.array([0.0, -0.0006])

    # Mismatched shapes
    try:
        doppler_factor_per_particle(gamma[:1], theta_x, theta_y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same shape" in str(e)

    # Non-1D arrays
    try:
        doppler_factor_per_particle(gamma.reshape(2, 1), theta_x, theta_y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "1D arrays" in str(e)

    # Invalid k_hat shape
    try:
        doppler_factor_per_particle(gamma, theta_x, theta_y, k_hat=np.array([1.0, 2.0]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "shape (3,)" in str(e)


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


def test_delta_counts_the_same_photons_as_stage_0(baseline):
    """§9.1, closed (RES033) — and the tripwire that keeps it closed.

    ``int dOmega`` of the paper's bare prefactor is ``2 pi`` analytically (delta's module
    docstring), which is exactly the factor `delta.DIFFERENTIAL_PREFACTOR` now removes, so
    two paths that count the same photons report the same number — this assertion is what
    would surface the factor coming back.

    The tolerance covers the angular grid's truncation, which `expected_ratio` already
    corrects for approximately — a square grid reaches past the disc the correction
    assumes, and the beam's own divergence broadens the distribution slightly.
    """
    check = delta.check_normalization(baseline, n_angles=65, cone_factor=8.0)
    assert check.anchor_ratio == pytest.approx(1.0, rel=1e-4)
    assert check.ratio == pytest.approx(check.captured_fraction, rel=2e-2)
    assert abs(check.deviation) < 2e-2
    # Explicitly not 2*pi off: the thing this test exists to notice.
    assert delta.DIFFERENTIAL_PREFACTOR == pytest.approx(3.0 / (2.0 * math.pi), rel=1e-14)


def test_an_empty_bunch_yields_zero_rather_than_raising():
    """Reachable, not hypothetical — and if it raises, the prefilter is not neutral.

    A mistimed pulse or a bunch far wider than the spot leaves the prefilter with nothing.
    With the filter off the same configuration returns 0.0, so an exception here would mean
    the filter turns a zero into a crash: the opposite of the pure optimization §3.2 claims.
    """
    interaction = scenarios.build(
        replace(scenarios.BASELINE,
                sampling=replace(scenarios.BASELINE.sampling, n_particles=64))
    )
    empty = interaction.bunch.select(np.zeros(interaction.bunch.n_particles, dtype=bool))
    samples = integrate_trajectories(empty, interaction.laser, interaction.N_e, n_steps=8)
    assert samples.n_particles == 0
    assert samples.total_yield() == 0.0
    assert samples.a0_shape.shape == (0,)


def test_the_capture_correction_accounts_for_the_polarization_factor():
    """``X/(1+X)`` is the Lorentz factor alone and overstates what the cone holds.

    It is the correction the normalization arbitration divides by, so an error here shows
    up as a cone-dependent drift in a number that is supposed to be a constant.
    """
    for cone in (2.0, 4.0, 8.0, 20.0):
        x = cone**2
        assert delta.captured_fraction(cone) < x / (1.0 + x)
    assert delta.captured_fraction(4.0) == pytest.approx(0.9168, abs=1e-4)
    assert delta.captured_fraction(1e4) == pytest.approx(1.0, abs=1e-6)


def test_polarization_uses_the_particle_lab_velocity_from_manuscript_udef():
    """A tilted electron changes the projection while observer and basis stay lab-frame."""
    args = dict(
        gamma=2000.0, theta_x=0.0011, theta_y=-0.0007,
        theta_x_obs=0.0003, theta_y_obs=-0.0002,
        ellipticity=0.35, psi_pol=0.6, theta_xz=0.04, theta_yz=-0.03,
    )
    expected = _polarization_factor_from_udef(**args)
    actual = polarization_factor(**args)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert actual != pytest.approx(polarization_factor(
        args["gamma"], 0.0, 0.0, args["theta_x_obs"], args["theta_y_obs"],
        args["ellipticity"], args["psi_pol"], args["theta_xz"], args["theta_yz"],
    ), rel=1e-3)


def test_vectorized_polarization_accepts_each_particle_direction():
    gamma = np.array([1800.0, 2200.0])
    theta_x = np.array([-0.0008, 0.0011])
    theta_y = np.array([0.0005, -0.0007])
    kwargs = dict(theta_x_obs=0.0003, theta_y_obs=-0.0002,
                  ellipticity=0.35, psi_pol=0.6, theta_xz=0.04, theta_yz=-0.03)
    expected = np.array([
        _polarization_factor_from_udef(g, tx, ty, **kwargs)
        for g, tx, ty in zip(gamma, theta_x, theta_y)
    ])
    assert np.allclose(
        polarization_factor_vectorized(gamma, theta_x, theta_y, **kwargs), expected, rtol=1e-9
    )


@pytest.mark.parametrize("gamma", [2000.0, 10000.0])
def test_polarization_factor_collinear_crossing_limit_is_unity(gamma):
    """Under physical transverse dipole projection (DER012), on-axis collinear limit is unity."""
    crossing = 0.3
    actual_scalar = polarization_factor(
        gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, theta_xz=crossing, theta_yz=0.0,
    )
    assert actual_scalar == pytest.approx(1.0, rel=1e-12, abs=1e-12)

    actual_vectorized = polarization_factor_vectorized(
        np.array([gamma]), np.array([0.0]), np.array([0.0]),
        0.0, 0.0, 0.0, 0.0, theta_xz=crossing, theta_yz=0.0,
    )
    np.testing.assert_allclose(actual_vectorized, [1.0], rtol=1e-12, atol=1e-12)


def test_delta_passes_each_particle_direction_to_the_polarization_projection():
    samples = TrajectorySamples(
        gamma=np.array([2000.0]), theta_x=np.array([0.0011]), theta_y=np.array([-0.0007]),
        a0_shape=np.array([1.0]), luminosity=np.array([7.0]), intensity_peak=0.02, n_steps=1,
    )
    theta_x_obs, theta_y_obs = 0.0003, -0.0002
    kwargs = dict(ellipticity=0.35, psi_pol=0.6, theta_xz=0.04, theta_yz=-0.03)
    r_squared = (samples.theta_x[0] - theta_x_obs)**2 + (samples.theta_y[0] - theta_y_obs)**2
    s_res = samples.gamma[0] ** 2 / (1.0 + samples.ahat()[0] + samples.gamma[0] ** 2 * r_squared)
    s_edges = np.array([s_res * 0.999, s_res * 1.001])
    expected_weight = (
        delta.DIFFERENTIAL_PREFACTOR * samples.luminosity[0]
        * _polarization_factor_from_udef(samples.gamma[0], samples.theta_x[0], samples.theta_y[0],
                                          theta_x_obs, theta_y_obs, **kwargs)
        * samples.gamma[0] ** 2 / (1.0 + r_squared * samples.gamma[0] ** 2) ** 2
    )
    density = delta.resonance_spectrum(samples, s_edges, theta_x_obs, theta_y_obs, **kwargs)
    assert density[0] * np.diff(s_edges)[0] == pytest.approx(expected_weight, rel=1e-9)


def test_the_closed_form_anchor_really_is_one(baseline):
    """It is an identity, so it should read as one — a trapezoid over bin centres did not.

    Integrating a bin-centre density with the trapezoid rule drops half of the first and
    last bin, which put a -0.6% bias on a quantity documented as exact and leaked it into
    the section-9.1 headline number, where nothing corrected for it.
    """
    check = delta.check_normalization(baseline, n_angles=17, cone_factor=4.0)
    assert check.anchor_ratio == pytest.approx(1.0, rel=1e-4)


def test_the_grid_residue_shrinks_as_the_cone_widens(baseline):
    """The leftover deviation is the square grid's corners, and it must behave like it."""
    narrow = delta.check_normalization(baseline, n_angles=33, cone_factor=4.0).deviation
    wide = delta.check_normalization(baseline, n_angles=65, cone_factor=8.0).deviation
    assert 0.0 < wide < narrow


def test_the_normalization_ratio_is_stable_as_the_angular_grid_is_refined(baseline):
    """A constant that survives refinement is a constant, not a discretization artefact."""
    coarse = delta.check_normalization(baseline, n_angles=17, cone_factor=4.0)
    fine = delta.check_normalization(baseline, n_angles=65, cone_factor=4.0)
    assert fine.ratio == pytest.approx(coarse.ratio, rel=2e-3)


def test_widening_the_cone_captures_more_and_moves_towards_one(baseline):
    """A wider cone can only add photons, and cannot reach more than the cone holds.

    The ceiling is `captured_fraction`, not 1: at eight cone widths the ratio is 0.981
    against a capture of 0.977, and the ~0.4% it sits *above* the disc correction is the
    square grid's corners reaching past it. Asserting against a bare 1.0 would leave a 2%
    margin that anyone retuning the cone or angle count would trip over for no reason.
    """
    # ``captured_fraction`` is the closed-form zero-divergence result.  The production
    # reference now correctly uses each particle direction, so retain the analytical
    # comparison on the configuration to which that closed form applies.
    collinear = replace(
        baseline, theta_x=np.zeros_like(baseline.theta_x), theta_y=np.zeros_like(baseline.theta_y)
    )
    settings = ((4.0, 33), (8.0, 65))
    checks = [delta.check_normalization(collinear, n_angles=n, cone_factor=c) for c, n in settings]
    assert checks[0].ratio < checks[1].ratio
    for check, (cone, _) in zip(checks, settings):
        assert check.ratio == pytest.approx(delta.captured_fraction(cone), rel=2e-2)


def test_delta_is_linear_in_charge(baseline):
    doubled = TrajectorySamples(
        gamma=baseline.gamma, theta_x=baseline.theta_x, theta_y=baseline.theta_y,
        a0_shape=baseline.a0_shape, luminosity=2.0 * baseline.luminosity,
        intensity_peak=baseline.intensity_peak, n_steps=baseline.n_steps,
    )
    edge = float(np.max(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 65)
    assert delta.resonance_spectrum(doubled, s_edges, 0.0, 0.0) == pytest.approx(
        2.0 * delta.resonance_spectrum(baseline, s_edges, 0.0, 0.0), rel=1e-14
    )

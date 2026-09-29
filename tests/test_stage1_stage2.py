"""Stage 1 (a0_shape deposition), the ahat retarget, and Stage 2 (spectrum queries),
GRAND_PLAN.md §4.2, Phase 3a (RES032).

Stage 1's job is conservation: every particle's weight lands somewhere in the shape
table, and the table's own total agrees with Stage 0's regardless of resolution or scheme.
The retarget's job is a second, independent conservation (total weight is preserved exactly
through the regrid) plus placing mass where the chosen peak a0 actually puts it — denser
near `ahat_max`, folded into the floor below `ahat_min`. Stage 2's job is the resonance
condition: the spectrum it reports must depend on ahat the way the physics does (the
nonlinear redshift), and it must agree with `delta` — an independent, table-free
implementation of the same differential form — to within grid/interpolation error.

Most of those are ratios between two paths carrying the same normalization, which is what
makes them robust and also what makes them blind: `test_the_table_kernel_angle_integrates_to_stage_0_total`
is the one absolute check, and the one that pins §9.1's constant (RES033).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier2]

from gammaforge.engines.xigma import stages
from gammaforge.engines.xigma.stages import (
    SpectralMoments,
    Table,
    TrajectorySamples,
    _ahat_target_edges,
    _inverse_resonance_gamma_sq,
    _inverse_resonance_jacobian,
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    nonuniform_derivative,
    query_spectral_moments,
    reconstruct_second_order,
    retarget_ahat,
    spectrum_from_table,
    spectrum_in_angular_range,
)
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta import (
    captured_fraction,
    resonance_spectral_moments,
    resonance_spectrum,
)


def _samples(scenario, n_particles=20_000, **kwargs):
    small = replace(scenario, sampling=replace(scenario.sampling, n_particles=n_particles))
    interaction = scenarios.build(small)
    return integrate_trajectories(
        interaction.bunch, interaction.laser, interaction.N_e, n_steps=64, **kwargs
    )


@pytest.fixture(scope="module")
def baseline():
    return _samples(scenarios.BASELINE)


def _synthetic_samples(n=20_000, gamma0=2000.0, seed=0, a0_shape=1.0, intensity_peak=0.045):
    """``intensity_peak`` is the peak cycle-averaged ``<a^2>`` (RES054), so ``ahat`` is just
    ``intensity_peak * a0_shape``. The default reproduces the ``a0_peak=0.3`` linear pulse
    these fixtures used before the rename: ``0.5 * 0.3**2 = 0.045``."""
    rng = np.random.default_rng(seed)
    gamma = gamma0 + rng.normal(0.0, gamma0 * 5e-3, n)
    theta_x = rng.normal(0.0, 3.0 / gamma0, n)
    theta_y = rng.normal(0.0, 3.0 / gamma0, n)
    return TrajectorySamples(
        gamma=gamma,
        theta_x=theta_x,
        theta_y=theta_y,
        a0_shape=np.full(n, a0_shape),
        luminosity=np.full(n, 1e4),
        intensity_peak=intensity_peak,
        n_steps=10,
        chirp_mean=np.ones(n),
        var_a_shape=np.zeros(n),
        var_chirp=np.zeros(n),
        cov_a_chirp_shape=np.zeros(n),
    )


def _table(samples, *, shape_bins=(16, 16, 16, 16, 8), scheme="nearest", **retarget_kwargs):
    """Shorthand: the two-call chain most tests need, at a modest, fast scale."""
    shape_table = deposit_shape_table(samples, n_bins=shape_bins, scheme=scheme)
    return retarget_ahat(shape_table, samples.intensity_peak, **retarget_kwargs)


# ---------------------------------------------------------------------------
# Stage 1: deposit_shape_table / ShapeTable
# ---------------------------------------------------------------------------
def test_deposit_conserves_total_weight_nearest():
    samples = _synthetic_samples()
    table = deposit_shape_table(samples, n_bins=(24, 24, 24, 8, 8), scheme="nearest")
    assert table.total_weight == pytest.approx(samples.total_yield(), rel=1e-12)
    assert table.H.sum() * table.bin_volume == pytest.approx(samples.total_yield(), rel=1e-9)


def test_deposit_conserves_total_weight_cic():
    samples = _synthetic_samples()
    table = deposit_shape_table(samples, n_bins=(24, 24, 24, 8, 8), scheme="cic")
    assert table.total_weight == pytest.approx(samples.total_yield(), rel=1e-9)


@pytest.mark.parametrize("scheme", ["nearest", "cic"])
def test_5d_deposition_and_retarget_conserve_all_moment_channels(scheme):
    samples = _synthetic_samples(n=200)
    luminosity = np.linspace(1.0, 2.0, samples.n_particles)
    chirp = np.linspace(0.8, 1.2, samples.n_particles)
    var_a = np.linspace(0.01, 0.03, samples.n_particles)
    var_chirp = np.linspace(0.02, 0.05, samples.n_particles)
    covariance = np.linspace(-0.01, 0.015, samples.n_particles)
    samples = replace(
        samples,
        luminosity=luminosity,
        chirp_mean=chirp,
        var_a_shape=var_a,
        var_chirp=var_chirp,
        cov_a_chirp_shape=covariance,
    )
    shape = deposit_shape_table(samples, n_bins=(4, 4, 4, 5, 6), scheme=scheme)

    assert shape.H.shape == (4, 4, 4, 5, 6)
    assert all(channel.shape == shape.H.shape for channel in (
        shape.H_var_a_shape, shape.H_var_chirp, shape.H_cov_a_chirp_shape
    ))
    expected_shape_masses = (
        luminosity.sum(),
        np.sum(luminosity * var_a),
        np.sum(luminosity * var_chirp),
        np.sum(luminosity * covariance),
    )
    for channel, expected in zip(
        (shape.H, shape.H_var_a_shape, shape.H_var_chirp, shape.H_cov_a_chirp_shape),
        expected_shape_masses,
    ):
        assert channel.sum() * shape.bin_volume == pytest.approx(expected, rel=2e-13, abs=1e-13)

    peak = 1.7 * samples.intensity_peak
    table = retarget_ahat(shape, peak, ahat_max=0.2, n_bins=7)
    dg = np.diff(table.gamma_edges)[:, None, None, None, None]
    dx = np.diff(table.theta_x_edges)[None, :, None, None, None]
    dy = np.diff(table.theta_y_edges)[None, None, :, None, None]
    da = np.diff(table.ahat_edges)[None, None, None, :, None]
    dc = np.diff(table.chirp_edges)[None, None, None, None, :]
    volume = dg * dx * dy * da * dc
    scale = peak / samples.intensity_peak
    expected_table_masses = (
        scale * expected_shape_masses[0],
        scale * peak**2 * expected_shape_masses[1],
        scale * expected_shape_masses[2],
        scale * peak * expected_shape_masses[3],
    )
    for channel, expected in zip(
        (table.H, table.H_var_a, table.H_var_chirp, table.H_cov_a_chirp),
        expected_table_masses,
    ):
        assert np.sum(channel * volume) == pytest.approx(expected, rel=3e-13, abs=1e-13)

    shape_chirp_mass = shape.H.sum(axis=(0, 1, 2, 3)) * shape.bin_volume
    table_chirp_mass = np.sum(table.H * dg * dx * dy * da, axis=(0, 1, 2, 3)) * dc.ravel()
    np.testing.assert_allclose(table_chirp_mass, scale * shape_chirp_mass, rtol=3e-13, atol=1e-13)


@pytest.mark.parametrize("scheme", ["nearest", "cic"])
def test_constant_chirp_collapses_to_one_exact_evaluation_bin(scheme):
    samples = _synthetic_samples(n=100)
    shape = deposit_shape_table(samples, n_bins=(4, 4, 4, 5, 11), scheme=scheme)
    assert shape.H.shape == (4, 4, 4, 5, 1)
    np.testing.assert_array_equal(shape.chirp_eval_points, np.array([1.0]))
    assert all(not np.any(channel) for channel in (
        shape.H_var_a_shape, shape.H_var_chirp, shape.H_cov_a_chirp_shape
    ))

    table = retarget_ahat(shape, samples.intensity_peak, ahat_max=0.2, n_bins=7)
    assert table.H.shape[4] == 1
    np.testing.assert_array_equal(table.chirp_eval_points, np.array([1.0]))


def test_single_nonlinear_and_chirp_bins_produce_an_effective_3d_table():
    samples = _synthetic_samples(n=100)
    shape = deposit_shape_table(samples, n_bins=(4, 4, 4, 1, 1))
    assert shape.H.shape == (4, 4, 4, 1, 1)

    table = retarget_ahat(shape, samples.intensity_peak, n_bins=1)
    assert table.H.shape == (4, 4, 4, 1, 1)
    np.testing.assert_array_equal(table.ahat_eval_points, np.array([0.0]))
    np.testing.assert_array_equal(table.chirp_eval_points, np.array([1.0]))


def test_single_chirp_bin_is_invariant_to_particle_order():
    samples = _synthetic_samples(n=400)
    samples = replace(samples, theta_x=samples.theta_x / 15,
                      theta_y=samples.theta_y / 15)
    chirp = np.linspace(0.8, 1.2, samples.n_particles)
    luminosity = np.linspace(1.0, 2.0, samples.n_particles)
    samples = replace(samples, chirp_mean=chirp, luminosity=luminosity)
    reversed_samples = replace(
        samples,
        **{name: getattr(samples, name)[::-1].copy() for name in (
            "gamma", "theta_x", "theta_y", "a0_shape", "luminosity",
            "chirp_mean", "var_a_shape", "var_chirp", "cov_a_chirp_shape",
        )},
    )
    first, second = (
        _table(item, shape_bins=(8, 8, 8, 2, 1), n_bins=1)
        for item in (samples, reversed_samples)
    )
    expected = np.average(chirp, weights=luminosity)
    assert first.chirp_eval_points[0] == pytest.approx(expected)
    assert second.chirp_eval_points[0] == pytest.approx(expected)
    np.testing.assert_allclose(first.H, second.H, rtol=1e-14, atol=1e-14)
    energy = np.linspace(1e6, 6e6, 32)
    assert np.any(spectrum_from_table(first, 0.0, 0.0, energy) > 0)
    np.testing.assert_allclose(
        spectrum_from_table(first, 0.0, 0.0, energy),
        spectrum_from_table(second, 0.0, 0.0, energy),
        rtol=1e-13,
    )


def test_zero_weight_particles_do_not_change_table_or_spectrum():
    samples = _synthetic_samples(n=100)
    dark = replace(samples, **{
        name: np.append(getattr(samples, name), value)
        for name, value in (
            ("gamma", 10_000.0), ("theta_x", 0.1), ("theta_y", 0.1),
            ("a0_shape", 0.0), ("luminosity", 0.0), ("chirp_mean", 0.0),
            ("var_a_shape", 0.0), ("var_chirp", 0.0),
            ("cov_a_chirp_shape", 0.0),
        )
    })
    first = _table(samples, shape_bins=(8, 8, 8, 2, 8), n_bins=1)
    second = _table(dark, shape_bins=(8, 8, 8, 2, 8), n_bins=1)
    assert first.H.shape == second.H.shape
    np.testing.assert_array_equal(second.chirp_eval_points, [1.0])
    np.testing.assert_allclose(first.H, second.H, rtol=1e-14)
    energy = np.linspace(1e6, 6e6, 32)
    np.testing.assert_allclose(
        spectrum_from_table(first, 0.0, 0.0, energy),
        spectrum_from_table(second, 0.0, 0.0, energy),
        rtol=1e-14,
    )


def test_zero_emission_bunch_has_zero_spectral_moments():
    samples = _synthetic_samples(n=100)
    dark = replace(
        samples,
        luminosity=np.zeros(samples.n_particles),
        chirp_mean=np.zeros(samples.n_particles),
    )
    table = _table(dark, shape_bins=(8, 8, 8, 2, 8), n_bins=1)
    moments = query_spectral_moments(table, 0.0, 0.0, [1e6, 2e6])
    assert table.total_weight == 0.0
    np.testing.assert_array_equal(table.chirp_eval_points, [1.0])
    np.testing.assert_array_equal(moments.rho0, [0.0, 0.0])
    np.testing.assert_array_equal(moments.rho1, [0.0, 0.0])
    np.testing.assert_array_equal(moments.rho2, [0.0, 0.0])


def test_single_ahat_bin_omits_nonlinear_moments_but_keeps_carrier_variance():
    samples = _synthetic_samples(n=400)
    samples = replace(
        samples,
        theta_x=samples.theta_x / 15,
        theta_y=samples.theta_y / 15,
        chirp_mean=np.full(samples.n_particles, 1.1),
        var_a_shape=np.full(samples.n_particles, 0.02),
        var_chirp=np.full(samples.n_particles, 0.004),
        cov_a_chirp_shape=np.full(samples.n_particles, 0.003),
    )
    table = _table(samples, shape_bins=(8, 8, 8, 2, 8), n_bins=1)
    assert np.any(table.H > 0)
    assert np.any(table.H_var_chirp > 0)
    assert not np.any(table.H_var_a)
    assert not np.any(table.H_cov_a_chirp)
    moments = query_spectral_moments(table, 0.0, 0.0, np.linspace(1e6, 4e6, 32))
    assert np.any(moments.rho0 > 0)
    np.testing.assert_array_equal(moments.rho1, np.zeros_like(moments.rho1))
    assert np.any(moments.rho2 > 0)


def test_deposit_uses_raw_shape_independent_of_observation_geometry():
    samples = TrajectorySamples(
        gamma=np.full(3, 2000.0),
        theta_x=np.array([-.4, 0.0, .5]),
        theta_y=np.array([.1, -.2, .3]),
        a0_shape=np.array([.6, 1.0, 1.4]),
        luminosity=np.ones(3),
        intensity_peak=.2,
        n_steps=8,
        chirp_mean=np.ones(3),
        var_a_shape=np.zeros(3),
        var_chirp=np.zeros(3),
        cov_a_chirp_shape=np.zeros(3),
    )
    table = deposit_shape_table(samples, n_bins=(2, 3, 3, 6, 4), margin=.02)

    np.testing.assert_allclose(
        table.a0_shape_edges,
        stages._uniform_edges(samples.a0_shape, 6, .02, floor_zero=True),
    )
    n0 = np.array([-.5, .25, -np.sqrt(.6875)])
    q_left = stages.observer_ponderomotive_factor(
        samples.theta_x, samples.theta_y, -.2, .1, k_hat=n0
    )
    q_right = stages.observer_ponderomotive_factor(
        samples.theta_x, samples.theta_y, .3, -.15, k_hat=n0
    )
    assert not np.allclose(q_left, q_right)
    np.testing.assert_array_equal(samples.a0_shape, np.array([.6, 1.0, 1.4]))


def test_deposit_handles_a_monoenergetic_zero_divergence_beam():
    n = 5000
    samples = TrajectorySamples(
        gamma=np.full(n, 2000.0),
        theta_x=np.zeros(n),
        theta_y=np.zeros(n),
        a0_shape=np.full(n, 1.0),
        luminosity=np.full(n, 1.0),
        intensity_peak=0.045,
        n_steps=10,
        chirp_mean=np.ones(n),
        var_a_shape=np.zeros(n),
        var_chirp=np.zeros(n),
        cov_a_chirp_shape=np.zeros(n),
    )
    table = deposit_shape_table(samples, n_bins=(8, 8, 8, 4, 4))
    assert np.all(np.isfinite(table.H))
    assert np.all(np.diff(table.gamma_edges) > 0.0)
    assert table.total_weight == pytest.approx(n, rel=1e-12)


def test_retarget_ahat_conserves_total_weight_exactly():
    """The identity `retarget_ahat`'s cheapness rests on: because the target edges are
    extended to +-inf for overlap purposes, every source bin's mass lands somewhere in the
    target, regardless of how much folds into the floor/ceiling bins.
    """
    samples = _synthetic_samples(a0_shape=1.0)
    shape_table = deposit_shape_table(samples, n_bins=(16, 16, 16, 16, 8))
    for intensity_peak in (0.01, 0.045, 0.18, 0.4):
        table = retarget_ahat(shape_table, intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)
        # Linear, not quadratic: the retarget parameter is now an intensity, which is the
        # already-squared quantity (RES054).
        expected = shape_table.total_weight * (intensity_peak / samples.intensity_peak)
        assert table.total_weight == pytest.approx(expected, rel=1e-9)


def test_retarget_ahat_redistributes_mass_toward_higher_ahat_as_the_pulse_strengthens():
    samples = _synthetic_samples(a0_shape=1.0)
    shape_table = deposit_shape_table(samples, n_bins=(16, 16, 16, 16, 8))
    table_own = retarget_ahat(shape_table, samples.intensity_peak)
    table_other = retarget_ahat(shape_table, 2.0 * samples.intensity_peak)
    # Both total_weight (exact identity above) and where that mass sits move: twice the
    # peak intensity is twice the yield *and* twice the ahat, so the populated range's own
    # top edge (after truncation, §Truncation) reaches a higher ahat (RES054).
    assert table_other.total_weight == pytest.approx(2.0 * table_own.total_weight, rel=1e-9)
    assert table_other.ahat_edges[-1] > table_own.ahat_edges[-1]


def test_retarget_ahat_folds_mass_below_ahat_min_into_the_floor_bin():
    """The floor-fold path — now exercises the explicit zeroth bin at ahat=0.
    
    With the new design, when ahat_min > 0, an explicit zeroth bin [0, ahat_min]
    is prepended to catch sub-floor contributions, evaluated at ahat=0.
    """
    samples = _synthetic_samples(a0_shape=1.0)  # ahat = 0.045 (RES053: C = 1/2)
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16, 8))
    # ahat_min well above the population's actual ahat (0.045): everything must fold into
    # the explicit zeroth bin [0, ahat_min], and total weight must still be exactly conserved.
    table = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.2, ahat_max=0.5, n_bins=16, decades=1.0)
    # First edge is now 0 (explicit zeroth bin for sub-floor contributions)
    assert table.ahat_edges[0] == pytest.approx(0.0)
    # The mass should be in the first bin (index 0), which spans [0, 0.2]
    assert table.H[:, :, :, 0, :].sum() * np.diff(table.ahat_edges)[0] * np.diff(table.chirp_edges)[0] * table.gamma_theta_cell_area * (
        table.gamma_edges[-1] - table.gamma_edges[0]
    ) / table.H.shape[0] == pytest.approx(table.total_weight, rel=1e-6)
    assert table.total_weight == pytest.approx(shape_table.total_weight, rel=1e-9)
    # The evaluation point for the first bin should be 0 (linear limit)
    assert table.ahat_eval_points[0] == pytest.approx(0.0)


def test_retarget_ahat_truncates_unpopulated_bins():
    """Trailing target bins the rescaled source never reaches carry exactly zero mass
    (RES032) — `retarget_ahat` drops them rather than returning a table
    padded with zeros out to `ahat_max`.
    """
    samples = _synthetic_samples(a0_shape=1.0, intensity_peak=0.00125)  # ahat = 0.00125, tiny
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16, 8))
    table = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)
    assert table.ahat_edges[-1] < 0.5
    assert table.H.shape[3] < 32


def test_retarget_ahat_truncation_does_not_change_the_kernel_output():
    """Truncation is a pure performance optimization: comparing a truncated table's
    `spectrum_from_table` output against a table padded back out with explicit zero bins
    must agree exactly.
    """
    samples = _synthetic_samples(n=5_000, a0_shape=1.0, intensity_peak=0.00125)
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16, 8))
    truncated = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)

    from gammaforge.engines.xigma.stages import Table

    pad = 5
    padded_edges = np.concatenate([truncated.ahat_edges, truncated.ahat_edges[-1] + np.arange(1, pad + 1) * 1e-3])
    padded_H = np.concatenate(
        [truncated.H, np.zeros((*truncated.H.shape[:3], pad, truncated.H.shape[4]))], axis=3
    )
    padded_channels = [
        np.concatenate([channel, np.zeros((*channel.shape[:3], pad, channel.shape[4]))], axis=3)
        for channel in (truncated.H_var_a, truncated.H_var_chirp, truncated.H_cov_a_chirp)
    ]
    # Pass the same evaluation points so the heuristic doesn't misinterpret the padded edges
    padded = Table(
        gamma_edges=truncated.gamma_edges,
        theta_x_edges=truncated.theta_x_edges,
        theta_y_edges=truncated.theta_y_edges,
        ahat_edges=padded_edges,
        chirp_edges=truncated.chirp_edges,
        H=padded_H,
        H_var_a=padded_channels[0],
        H_var_chirp=padded_channels[1],
        H_cov_a_chirp=padded_channels[2],
        total_weight=truncated.total_weight,
        scheme=truncated.scheme,
        _ahat_eval_points=truncated._ahat_eval_points,
        _chirp_eval_points=truncated._chirp_eval_points,
    )

    s = np.linspace(1.0, samples.gamma.max() ** 2, 20)
    assert np.allclose(
        spectrum_from_table(truncated, 0.0, 0.0, s), spectrum_from_table(padded, 0.0, 0.0, s)
    )


# ---------------------------------------------------------------------------
# Stage 2: spectrum_from_table / angular_spectrum_from_table / spectrum_in_angular_range
# ---------------------------------------------------------------------------
def test_spectrum_from_table_is_nonnegative_and_zero_past_the_compton_edge():
    samples = _synthetic_samples(gamma0=1000.0)
    table = _table(samples, shape_bins=(20, 20, 20, 16, 8))
    edge = float(np.max(samples.gamma) ** 2)
    s = np.linspace(0.0, 1.5 * edge, 60)
    spec = spectrum_from_table(table, 0.0, 0.0, s)
    assert np.all(spec >= 0.0)
    assert np.all(spec[s > 1.1 * edge] == 0.0)


def test_chirped_resonance_inverse_and_jacobian_match_forward_and_finite_difference():
    rng = np.random.default_rng(20260927)
    gamma = rng.uniform(50.0, 5000.0, 200)
    A = rng.uniform(1.0, 1.8, 200)
    K = rng.uniform(0.7, 1.3, 200)
    r_sq = rng.uniform(0.0, 0.4 / gamma**2)
    s = K * gamma**2 / (A + gamma**2 * r_sq)

    inverted, valid = _inverse_resonance_gamma_sq(A, K, r_sq, s)
    assert np.all(valid)
    np.testing.assert_allclose(inverted, gamma**2, rtol=3e-15)

    step = 1e-6 * s
    plus = np.sqrt(_inverse_resonance_gamma_sq(A, K, r_sq, s + step)[0])
    minus = np.sqrt(_inverse_resonance_gamma_sq(A, K, r_sq, s - step)[0])
    numerical = (plus - minus) / (2.0 * step)
    expected = _inverse_resonance_jacobian(K, gamma, A, s)
    np.testing.assert_allclose(numerical, expected, rtol=2e-9)


def _single_cell_moment_table(var_a=0.0, var_chirp=0.0, covariance=0.0):
    gamma_edges = np.linspace(80.0, 120.0, 81)
    H = np.ones((80, 1, 1, 1, 1))
    return Table(
        gamma_edges=gamma_edges,
        theta_x_edges=np.array([-1e-6, 1e-6]),
        theta_y_edges=np.array([-1e-6, 1e-6]),
        ahat_edges=np.array([0.1, 0.2]),
        chirp_edges=np.array([0.9, 1.1]),
        H=H,
        H_var_a=var_a * H,
        H_var_chirp=var_chirp * H,
        H_cov_a_chirp=covariance * H,
        total_weight=1.0,
        scheme="synthetic-moments",
        _ahat_eval_points=np.array([0.15]),
        _chirp_eval_points=np.array([1.0]),
    )


def test_raw_spectral_moment_channels_match_the_closed_form_weights():
    s = np.array([7000.0])
    var_a, var_chirp, covariance = 0.02, 0.03, 0.01
    moments = query_spectral_moments(
        _single_cell_moment_table(var_a, var_chirp, covariance), 0.0, 0.0, s
    )
    B = 1.15
    expected_1 = s * (var_a / B**2 - covariance / B)
    expected_2 = s**2 * (var_chirp + var_a / B**2 - 2.0 * covariance / B)
    np.testing.assert_allclose(moments.rho1 / moments.rho0, expected_1, rtol=2e-12)
    np.testing.assert_allclose(moments.rho2 / moments.rho0, expected_2, rtol=2e-12)

    zero = query_spectral_moments(_single_cell_moment_table(), 0.0, 0.0, s)
    np.testing.assert_array_equal(zero.rho1, np.zeros(1))
    np.testing.assert_array_equal(zero.rho2, np.zeros(1))

    pure_chirp = query_spectral_moments(
        _single_cell_moment_table(var_chirp=var_chirp), 0.0, 0.0, s
    )
    np.testing.assert_array_equal(pure_chirp.rho1, np.zeros(1))
    np.testing.assert_allclose(pure_chirp.rho2 / pure_chirp.rho0, s**2 * var_chirp)

    uncorrelated = query_spectral_moments(
        _single_cell_moment_table(var_a, var_chirp, 0.0), 0.0, 0.0, s
    )
    assert moments.rho2[0] < uncorrelated.rho2[0]


def test_nonuniform_derivative_is_exact_for_its_local_polynomial_order():
    x = np.array([0.2, 0.31, 0.57, 0.9, 1.4, 2.1, 3.0, 4.4])
    values = 2.0 - 3.0 * x + 0.7 * x**2 + 0.2 * x**3 - 0.04 * x**4
    expected_first = -3.0 + 1.4 * x + 0.6 * x**2 - 0.16 * x**3
    expected_second = 1.4 + 1.2 * x - 0.48 * x**2
    np.testing.assert_allclose(
        nonuniform_derivative(x, values, 1), expected_first, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        nonuniform_derivative(x, values, 2), expected_second, rtol=2e-10, atol=2e-10
    )


def test_second_order_reconstruction_matches_the_settled_formula_without_clipping():
    s = np.array([0.5, 0.73, 1.1, 1.8, 2.7, 4.0])
    p1 = 0.2 + 0.3 * s - 0.1 * s**2 + 0.02 * s**3
    p2 = 0.4 - 0.2 * s + 0.05 * s**2 + 0.01 * s**4
    rho0 = 1.5 + 0.1 * s
    moments = SpectralMoments(s=s, rho0=rho0, rho1=p1 / s, rho2=p2 / s)
    expected = rho0 - (0.3 - 0.2 * s + 0.06 * s**2) / s
    expected += 0.5 * (0.1 + 0.12 * s**2) / s
    np.testing.assert_allclose(reconstruct_second_order(moments), expected, rtol=2e-11)

    negative = SpectralMoments(s=s, rho0=np.ones_like(s), rho1=s, rho2=np.zeros_like(s))
    np.testing.assert_allclose(reconstruct_second_order(negative), -np.ones_like(s), atol=2e-13)


def test_nonuniform_second_order_reconstruction_converges():
    def evaluate(n_points):
        unit = np.linspace(0.0, 1.0, n_points)
        s = 0.5 + 1.5 * unit**1.4
        moments = SpectralMoments(
            s=s,
            rho0=np.exp(-s),
            rho1=np.sin(s) / s,
            rho2=np.exp(s) / s,
        )
        expected = np.exp(-s) - np.cos(s) / s + 0.5 * np.exp(s) / s
        return np.max(np.abs(reconstruct_second_order(moments) - expected))

    assert evaluate(33) < evaluate(17) / 5.0


def test_line_model_switch_is_explicit_and_delta_compatible():
    table = _single_cell_moment_table()
    s = np.array([6500.0, 6800.0, 7100.0, 7400.0, 7700.0])
    delta = spectrum_from_table(table, 0.0, 0.0, s, line_model="delta")
    moment2 = spectrum_from_table(table, 0.0, 0.0, s, line_model="moment2")
    np.testing.assert_allclose(moment2, delta)
    with pytest.raises(ValueError, match="at least three"):
        spectrum_from_table(table, 0.0, 0.0, 7000.0, line_model="moment2")
    with pytest.raises(ValueError, match="line_model"):
        spectrum_from_table(table, 0.0, 0.0, s, line_model="unknown")


def test_the_kernel_and_delta_agree_on_where_the_redshift_puts_the_photons():
    """The one check that watches ``ahat`` itself, rather than integrating it away.

    Every other kernel-vs-`delta` comparison here and in `validation.run` compares *photon
    counts* — a sum over ``s``. The nonlinear redshift moves photons along ``s`` and
    conserves that sum exactly, so those checks are structurally blind to ``ahat``: the
    fourth identity leg reads 0.9996 whether ``ahat`` is right, doubled, or halved
    (measured, RES053). The **centroid** in ``s`` is the quantity ``ahat``
    controls, so that is what this compares.

    It cannot catch a wrong ``ahat`` *convention* — the two paths share
    `TrajectorySamples`, so a common-mode factor cancels here as it does everywhere else
    (RES053). What it does catch is the kernel losing the redshift to grid coarseness: the
    ``ahat`` axis
    must actually resolve the population for the table to reproduce `delta`'s centroid, and
    at the production defaults ``NEAR_A0_MAX`` lands in two bins.

    Teeth, measured by biasing the table's ``ahat`` axis alone (a kernel-side error `delta`
    does not share): a 1.2x bias moves the centroid 0.53%, a 2x bias 2.6% — the ``rel=2e-3``
    below catches all of them with margin, against a 0.009% residual when nothing is wrong.
    """
    samples = _samples(scenarios.NEAR_A0_MAX)
    shape_table = deposit_shape_table(samples, n_bins=(32, 48, 48, 64, 8), scheme="cic")
    edge = float(np.max(samples.gamma)) ** 2
    s_edges = np.linspace(0.0, 1.05 * edge, 150)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    def centroid(spectrum):
        return float(np.sum(s_centers * spectrum) / np.sum(spectrum))

    # `ahat_max` narrowed to the bank's own scale: the production 0.5 is headroom for
    # pulses far brighter than any scenario here, and spends 30 of its 32 bins above them.
    fine = retarget_ahat(shape_table, samples.intensity_peak, ahat_max=0.1, n_bins=32)
    assert fine.H.shape[3] >= 8, "the point of this test is a resolved ahat axis"
    kernel = angular_spectrum_from_table(fine, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)
    assert centroid(kernel) == pytest.approx(centroid(reference), rel=2e-3)

    # And the redshift is present, not merely consistent: four times the ahat pulls the
    # centroid down by several percent, on both paths independently. Four times the
    # *intensity* now, where this used to double an amplitude for the same effect (RES054).
    bright = retarget_ahat(shape_table, 4.0 * samples.intensity_peak, ahat_max=0.4, n_bins=32)
    kernel_bright = angular_spectrum_from_table(bright, [0.0], [0.0], s_centers)[0, 0, :]
    reference_bright = resonance_spectrum(
        replace(samples, intensity_peak=4.0 * samples.intensity_peak), s_edges, 0.0, 0.0
    )
    assert centroid(kernel_bright) == pytest.approx(centroid(reference_bright), rel=2e-3)
    assert centroid(kernel_bright) < 0.97 * centroid(kernel)


@pytest.mark.parametrize(
    "scenario, expected_bias, expected_bins",
    [(scenarios.BASELINE, -1.131, 1), (scenarios.LOW_A0, -1.627, 1), (scenarios.NEAR_A0_MAX, -0.036, 2)],
)
def test_the_production_ahat_grid_under_resolves_the_bank_by_a_known_amount(
    scenario, expected_bias, expected_bins
):
    """What the *shipping* configuration actually does, pinned rather than described.

    The test above resolves the ``ahat`` axis deliberately (``ahat_max=0.1``) — but
    `Collision._table` builds the grid from the schema defaults, and nothing else here
    exercises those on the quantity ``ahat`` controls. At RES032's defaults
    (``ahat_max=0.5``, ``n_bins=32``, ``decades=1.0``) the first non-floor edge sits at
    0.035, so the whole scenario bank lands at or near the floor bin and the kernel uses
    that bin's own centre — 0.0174 — in place of population means of 0.0057, 0.00057 and
    0.028. The centroid it reports is biased low by the amounts below.

    This is **not a regression from RES053** and not something to fix by widening a tolerance:
    the bias is dominated by floor-bin coarseness and predates the cycle-average correction
    (``low_a0`` moved -1.59% -> -1.65% across it). It is pinned so that a future change to
    `_ahat_target_edges` or to the defaults has to move these numbers deliberately.
    RES053's last section records the measured alternative (``decades=0.3`` at the same
    ``n_bins``/``ahat_max``), which is the author's call rather than this test's.
    """
    samples = _samples(scenario)
    shape_table = deposit_shape_table(samples, n_bins=(32, 48, 48, 64, 8), scheme="cic")
    table = retarget_ahat(shape_table, samples.intensity_peak)  # production defaults, as Collision does
    assert table.H.shape[3] == expected_bins

    # The floor bin's centre is a fixed property of the grid, the same for every scenario,
    # and it is what the kernel uses for the whole population below 0.035.
    assert table.ahat_centers[0] == pytest.approx(0.5 * _ahat_target_edges(0.0, 0.5, 32, 1.0)[1])

    edge = float(np.max(samples.gamma)) ** 2
    s_edges = np.linspace(0.0, 1.05 * edge, 150)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    kernel = angular_spectrum_from_table(table, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)

    def centroid(spectrum):
        return float(np.sum(s_centers * spectrum) / np.sum(spectrum))

    bias = 100.0 * (centroid(kernel) / centroid(reference) - 1.0)
    assert bias == pytest.approx(expected_bias, abs=0.15)


def test_spectrum_shifts_with_ahat_not_merely_rescales():
    """Regression guard: g/prefac must be recomputed inside the ahat loop. An
    ahat-independent shortcut would rescale the spectrum's amplitude but
    never move where its edge falls; the nonlinear redshift must move the edge.

    a0_shape=0.05 and 4.0 at intensity_peak=0.045 give ahat = 0.00225 and 0.18 — bins 0 and 6 of
    the production target grid (RES053), still well apart.
    """
    low = _synthetic_samples(a0_shape=0.05)
    high = _synthetic_samples(a0_shape=4.0, seed=0)
    table_low = _table(low, shape_bins=(24, 24, 24, 8, 8))
    table_high = _table(high, shape_bins=(24, 24, 24, 8, 8))

    edge = float(np.max(low.gamma) ** 2)
    s = np.linspace(0.0, 1.2 * edge, 200)
    spec_low = spectrum_from_table(table_low, 0.0, 0.0, s)
    spec_high = spectrum_from_table(table_high, 0.0, 0.0, s)

    def edge_index(spec):
        nonzero = np.nonzero(spec > 1e-6 * spec.max())[0]
        return nonzero[-1]

    # Higher ahat redshifts the resonance (s_res = gamma^2 / (1 + ahat + ...)), so the
    # high-ahat table's populated s range ends at a strictly lower edge than the low-ahat
    # one's — not just a smaller peak at the same edge.
    assert s[edge_index(spec_high)] < s[edge_index(spec_low)]


def test_angular_spectrum_matches_a_manual_grid_of_spectrum_from_table():
    samples = _synthetic_samples(n=5_000)
    table = _table(samples, shape_bins=(16, 16, 16, 16, 8))
    s = np.linspace(1.0, samples.gamma.max() ** 2, 5)
    tx, ty = [-1e-3, 0.0], [0.0, 1e-3]
    cube = angular_spectrum_from_table(table, tx, ty, s)
    for i, x in enumerate(tx):
        for j, y in enumerate(ty):
            assert np.allclose(cube[i, j, :], spectrum_from_table(table, x, y, s))


def test_spectrum_in_angular_range_photon_count_matches_the_cube_integral():
    samples = _synthetic_samples(n=5_000)
    table = _table(samples, shape_bins=(16, 16, 16, 16, 8))
    s_edges = np.linspace(0.0, 1.05 * samples.gamma.max() ** 2, 20)
    cube, dN_ds, n_photons = spectrum_in_angular_range(
        table, (-2e-3, 2e-3), (-2e-3, 2e-3), s_edges, resolution=(9, 9)
    )
    assert cube.shape == (9, 9, 19)
    assert dN_ds.shape == (19,)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    assert n_photons == pytest.approx(float(np.trapezoid(dN_ds, s_centers)), rel=1e-12)
    assert n_photons >= 0.0


# ---------------------------------------------------------------------------
# Stage 2 vs delta: an identity gate insensitive to §9.1 (both carry the same factor, so
# their ratio was ~1 before RES033 set it and is ~1 after). The absolute normalization the
# ratio cannot see gets its own test at the bottom of this section.
# ---------------------------------------------------------------------------
def test_stage2_kernel_agrees_with_delta_at_a_point():
    """CIC, not nearest: evaluating exactly at the beam's own angular centre — the natural
    point to check — lands exactly on a cell boundary of a *nearest*-deposited table, and
    a narrow beam spans few enough theta cells that this aliases into ratios anywhere from
    0.5 to 1.7 depending on resolution alone (measured while writing this test, not a
    hypothetical). CIC deposition removes the aliasing (measured stable to +-2% from 40
    to 250 theta bins) because it
    never lets a single cell speak for the beam centre alone.
    """
    samples = _synthetic_samples(n=200_000, seed=2)  # a0_shape=1.0, intensity_peak=0.045 -> ahat=0.045
    table = _table(samples, shape_bins=(48, 64, 64, 32, 8), scheme="cic")

    edge = float(np.max(samples.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 200)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    kernel = angular_spectrum_from_table(table, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)

    ratio = float(np.sum(kernel)) / float(np.sum(reference))
    assert ratio == pytest.approx(1.0, abs=0.1)


def test_chirped_stage2_kernel_tracks_particle_oracle_for_head_on_and_crossed_geometry():
    samples = _synthetic_samples(n=80_000, gamma0=1200.0, seed=7, a0_shape=0.7)
    chirp = np.random.default_rng(18).uniform(0.94, 1.06, len(samples.gamma))
    samples = replace(samples, chirp_mean=chirp)
    shape_table = deposit_shape_table(samples, n_bins=(40, 64, 64, 1, 16), scheme="cic")
    table = retarget_ahat(
        shape_table, samples.intensity_peak, ahat_max=0.1, n_bins=1
    )

    edge = 1.2 * float(np.max(samples.gamma) ** 2)
    s_edges = np.linspace(0.0, edge, 220)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    def centroid(spectrum):
        return float(np.sum(s_centers * spectrum) / np.sum(spectrum))

    geometries = [
        dict(theta_x=0.0, theta_y=0.0, theta_xz=0.0, theta_yz=0.0),
        dict(theta_x=3e-4, theta_y=-2e-4, theta_xz=0.05, theta_yz=-0.03),
    ]
    for geometry in geometries:
        kernel = spectrum_from_table(table, s=s_centers, **geometry)
        reference = resonance_spectrum(samples, s_edges=s_edges, **geometry)
        assert centroid(kernel) == pytest.approx(centroid(reference), rel=0.03)
        assert np.sum(kernel) == pytest.approx(np.sum(reference), rel=0.05)


def test_table_moment_channels_converge_to_direct_particle_oracle():
    samples = _synthetic_samples(n=80_000, gamma0=1200.0, seed=9, a0_shape=0.7)
    n_particles = len(samples.gamma)
    samples = replace(
        samples,
        chirp_mean=np.full(n_particles, 1.03),
        var_a_shape=np.full(n_particles, 0.02),
        var_chirp=np.full(n_particles, 0.003),
        cov_a_chirp_shape=np.full(n_particles, 0.002),
    )
    s_edges = np.linspace(0.0, 1.2 * float(np.max(samples.gamma) ** 2), 240)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    widths = np.diff(s_edges)
    geometry = dict(theta_x=3e-4, theta_y=-2e-4, theta_xz=0.05, theta_yz=-0.03)
    direct = resonance_spectral_moments(samples, s_edges, **geometry)
    np.testing.assert_array_equal(
        direct.rho0, resonance_spectrum(samples, s_edges, **geometry)
    )

    def integrated_channels(shape_bins):
        shape_table = deposit_shape_table(samples, n_bins=shape_bins, scheme="cic")
        table = retarget_ahat(
            shape_table, samples.intensity_peak, ahat_max=0.1, n_bins=16
        )
        moments = query_spectral_moments(table, s=s_centers, **geometry)
        return np.array([
            np.sum(moments.rho0 * widths),
            np.sum(moments.rho1 * widths),
            np.sum(moments.rho2 * widths),
        ])

    expected = np.array([
        np.sum(direct.rho0 * widths),
        np.sum(direct.rho1 * widths),
        np.sum(direct.rho2 * widths),
    ])
    coarse = integrated_channels((24, 32, 32, 1, 1))
    fine = integrated_channels((40, 64, 64, 1, 1))
    assert np.all(np.abs(fine / expected - 1.0) < np.abs(coarse / expected - 1.0))
    np.testing.assert_allclose(fine, expected, rtol=0.06)


@pytest.mark.tier3
@pytest.mark.heavy
def test_the_table_kernel_angle_integrates_to_stage_0_total(baseline):
    """§9.1's closure, on the one quantity that can actually see it (RES033).

    Every other Stage-2 check in this file is a *ratio* between two paths that carry the
    same normalization constant, so all of them stayed green through a factor of ``2 pi``
    and would stay green through any other. This one is absolute: integrate the table
    kernel over solid angle and over ``s``, and compare with Stage 0's elementary
    ``flux x cross-section x time`` photon count. It is the same arbitration
    `delta.check_normalization` performs for delta, applied to the kernel that actually
    ships. RES033 has the full derivation-then-measurement sequence that justified the fix.

    **The tolerance is resolution, not doubt.** Both integrals are midpoint sums over grids
    sized for a ten-second test: the angular one samples a ``1/gamma``-wide cone, and the
    ``s`` one a spectrum narrower still. Refining either walks the ratio straight toward 1
    (RES033 has the convergence sequence) — ±15% covers the grid this test can afford, and is
    nowhere near wide enough to blur the only distinction it exists to make, which is
    between 1 and 6.28.
    """
    n_angles, cone = 17, 4.0
    table = _table(baseline, shape_bins=(32, 24, 24, 64, 8), scheme="cic")

    edge = float(np.max(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 241)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    widths = np.diff(s_edges)

    half = cone / float(np.mean(baseline.gamma))
    step = 2.0 * half / n_angles
    offsets = -half + step * (np.arange(n_angles) + 0.5)
    grid_x = float(np.mean(baseline.theta_x)) + offsets
    grid_y = float(np.mean(baseline.theta_y)) + offsets

    cube = angular_spectrum_from_table(table, grid_x, grid_y, s_centers)
    photons = float(np.sum(cube * widths[None, None, :])) * step * step
    # What the finite cone could not see — the same closed-form correction delta's own
    # arbitration divides by, and for the same reason.
    ratio = photons / baseline.total_yield() / captured_fraction(cone)

    assert ratio == pytest.approx(1.0, rel=0.15)
    assert ratio < 2.0  # i.e. nowhere near the 2*pi this used to be

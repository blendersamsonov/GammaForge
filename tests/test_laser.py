"""Laser geometry, sampling contract and descriptive fit."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.io.laser import (
    ELLIPTICITY_IS_NOOP,
    EMISSION_IS_HEAD_ON,
    ActiveRegion,
    GaussianParaxialLaser,
    LaserField,
    lab_frame_axes,
    rotation_matrix,
    validate,
)
from gammaforge.io.units import C_CGS, Quantity as Q


def make_laser(**overrides) -> GaussianParaxialLaser:
    defaults = dict(pulse_energy=Q(1, "J"), wavelength=Q(800, "nm"),
                    sigma_x=Q(10, "um"), sigma_y=Q(10, "um"), duration=Q(30, "fs"))
    return GaussianParaxialLaser(**{**defaults, **overrides})


# -- geometry (§2.2) ---------------------------------------------------------
def test_head_on_is_minus_z():
    k_hat, e1, e2 = lab_frame_axes(0.0, 0.0, 0.0)
    assert np.allclose(k_hat, [0.0, 0.0, -1.0])
    assert np.allclose(e1, [1.0, 0.0, 0.0])
    assert np.allclose(e2, [0.0, 1.0, 0.0])


def test_rotation_is_orthonormal_and_right_handed():
    rot = rotation_matrix(0.3, -0.2)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_geometry_round_trip_recovers_head_on():
    # The §2.2 exit criterion: applying R^-1 recovers the head-on configuration.
    for theta_xz, theta_yz in [(0.3, 0.2), (-0.5, 0.1), (1.2, -0.9)]:
        rot = rotation_matrix(theta_xz, theta_yz)
        k_hat, e1, e2 = lab_frame_axes(theta_xz, theta_yz, 0.0)
        inverse = np.linalg.inv(rot)
        assert np.allclose(inverse @ k_hat, [0.0, 0.0, -1.0], atol=1e-13)
        assert np.allclose(inverse @ e1, [1.0, 0.0, 0.0], atol=1e-13)
        assert np.allclose(inverse @ e2, [0.0, 1.0, 0.0], atol=1e-13)


def test_transverse_axes_stay_orthonormal_and_perpendicular_to_k():
    for psi in [0.0, 0.4, -1.1, math.pi / 2]:
        k_hat, e1, e2 = lab_frame_axes(0.35, -0.25, psi)
        for vector in (k_hat, e1, e2):
            assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert abs(np.dot(e1, k_hat)) < 1e-14
        assert abs(np.dot(e2, k_hat)) < 1e-14
        assert abs(np.dot(e1, e2)) < 1e-14


def test_psi_rotates_within_the_transverse_plane():
    _, e1_zero, _ = lab_frame_axes(0.3, 0.2, 0.0)
    _, e1_psi, _ = lab_frame_axes(0.3, 0.2, 0.7)
    assert np.dot(e1_zero, e1_psi) == pytest.approx(math.cos(0.7))
    # psi is measured from the transported x-axis, so a full turn is the identity.
    _, e1_full, _ = lab_frame_axes(0.3, 0.2, 2.0 * math.pi)
    assert np.allclose(e1_full, e1_zero)


# -- the LaserField contract -------------------------------------------------
def test_photon_density_integrates_to_one_over_space():
    # The normalization the whole energy->a0 chain rests on (§3.3).
    laser = make_laser()
    grid = np.linspace(-6e-3, 6e-3, 161)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    step = grid[1] - grid[0]
    total = laser.photon_density(x, y, z, 0.0).sum() * step**3
    assert total == pytest.approx(1.0, rel=1e-4)


def test_unchirped_laser_returns_zero_additional_phase_gradient_with_broadcasting():
    laser = make_laser()
    gradient = laser.carrier_phase_four_gradient(
        np.zeros((2, 1)), np.zeros((1, 3)), 0.0, np.arange(3)[None, :]
    )
    assert len(gradient) == 4
    for component in gradient:
        np.testing.assert_array_equal(component, np.zeros((2, 3)))


def test_a0_matches_the_textbook_intensity_relation():
    # a0 = 8.55e-10 * lambda[um] * sqrt(I[W/cm^2]) for linear polarization. Derived here
    # from CGS-Gaussian first principles, so agreement checks the whole chain.
    laser = make_laser()
    density = laser.photon_density(0.0, 0.0, 0.0, 0.0)
    intensity_w_per_cm2 = C_CGS * laser.m("pulse_energy") * density / 1e7
    expected = 8.55e-10 * (laser.m("wavelength") * 1e4) * math.sqrt(intensity_w_per_cm2)
    assert laser.a0_peak() == pytest.approx(expected, rel=1e-3)


def test_a0_scales_as_the_square_root_of_pulse_energy():
    assert make_laser(pulse_energy=Q(4, "J")).a0_peak() == pytest.approx(2.0 * make_laser().a0_peak())


def _peak_offset_along_axis(laser: GaussianParaxialLaser, t: float, expected: float) -> float:
    """Where along k_hat the envelope peaks at time ``t``, found by scanning around ``expected``.

    The peak's *position* is what "the pulse travels at c" means. Its *amplitude* is not
    conserved and must not be asserted to be: a Gaussian beam diffracts, so the envelope
    at the moving peak falls off as the pulse leaves its focus.
    """
    k_hat, _, _ = laser.focusing_axes()
    offsets = expected + np.linspace(-2.0, 2.0, 4001) * laser.sigma_ct()
    points = offsets[:, None] * k_hat[None, :]
    envelope = laser.a0_profile(points[:, 0], points[:, 1], points[:, 2], t)
    return float(offsets[int(np.argmax(envelope))])


def test_pulse_travels_along_k_hat_at_the_speed_of_light():
    laser = make_laser()
    dt = 1e-13
    assert _peak_offset_along_axis(laser, 0.0, 0.0) == pytest.approx(0.0, abs=1e-5)
    assert _peak_offset_along_axis(laser, dt, C_CGS * dt) == pytest.approx(C_CGS * dt, rel=1e-3)


def test_crossing_angle_moves_the_pulse_along_the_rotated_axis():
    laser = make_laser(theta_xz=Q(0.4, "rad"), theta_yz=Q(-0.2, "rad"))
    dt = 1e-13
    # Head-on and tilted pulses travel the same distance in the same time; only the
    # direction differs, and it is entirely carried by k_hat.
    assert _peak_offset_along_axis(laser, dt, C_CGS * dt) == pytest.approx(C_CGS * dt, rel=1e-3)


def test_spot_expands_by_sqrt_two_at_one_rayleigh_range():
    laser = make_laser()
    s1, _ = laser.spot_sizes(np.array([laser.rayleigh_x()]))
    assert float(s1[0]) == pytest.approx(laser.m("sigma_x") * math.sqrt(2.0))


@pytest.mark.parametrize("ellipticity", [0.0, 0.3, 0.5, 1.0])
def test_the_cycle_averaged_intensity_is_polarization_agnostic(ellipticity):
    """RES054's premise, and the other half of
    `test_stage0_delta.py::test_stage_0_is_bit_identical_under_any_polarization`.

    At fixed pulse energy, ``<a^2>`` is the same for every polarization state: an
    elliptical pulse's ``a0`` is smaller by ``sqrt(2C)`` and its cycle average larger by
    ``C``, exactly offsetting. That is why nothing on the yield/red-shift path needs to
    know the polarization.

    ``C = (1 + eps^2) / 2`` follows from the paper's own ``sum_i |eps_i|^2 = 1``, and is
    checked here against a *period-resolved* ellipse rather than restated: the mean of
    ``|a|^2`` over one cycle divided by its peak.
    """
    laser = make_laser(ellipticity=ellipticity)
    linear = make_laser()

    # C from the paper's normalization, against a directly sampled ellipse.
    phase = np.linspace(0.0, 2.0 * math.pi, 200_001)
    norm = 1.0 / math.sqrt(1.0 + ellipticity**2)
    a_sq = (np.cos(phase) * norm) ** 2 + (ellipticity * norm * np.sin(phase)) ** 2
    assert laser.cycle_average_factor() == pytest.approx(float(a_sq.mean() / a_sq.max()), rel=1e-5)

    # The invariance itself, at several points in the pulse.
    for point in [(0.0, 0.0, 0.0, 0.0), (4e-4, 2e-4, 0.01, 5e-13)]:
        assert laser.intensity_profile(*point) == pytest.approx(
            linear.intensity_profile(*point), rel=1e-14
        )
    assert laser.intensity_peak() == pytest.approx(linear.intensity_peak(), rel=1e-14)

    # And it is a genuine cancellation, not both sides being constant: the *reported*
    # a0_peak is the linear-equivalent amplitude, so `C * a0_peak**2` is NOT the invariant
    # — computing it that way is the trap `intensity_peak()` exists to avoid.
    if ellipticity > 0.0:
        assert laser.cycle_average_factor() > linear.cycle_average_factor()
        assert laser.cycle_average_factor() * laser.a0_peak() ** 2 != pytest.approx(
            laser.intensity_peak(), rel=1e-6
        )


def test_the_rayleigh_range_converts_rms_to_the_1_over_e_squared_convention():
    """The conversion `rayleigh_x` exists to perform, pinned against the profile itself
    rather than against its own formula (author-confirmed 2026-08-10, RES040).

    ``z_R = pi w0^2 / lambda`` is stated in the **1/e² convention**, but this class stores
    intensity **RMS** widths — at ``r = sigma`` the density is down by ``e^-1/2``, not
    ``e^-2``. Skipping the conversion is a factor of 4 in ``z_R`` and in the far-field
    angle — the convention error guarded by RES040. So this measures ``w0`` off `photon_density`
    directly, and only then checks the textbook formula against it — a test written as
    ``4 pi sigma^2 / lambda`` would restate
    the implementation and pass however wrong the convention was.
    """
    laser = make_laser()
    sigma, lam = laser.m("sigma_x"), laser.m("wavelength")

    on_axis = float(laser.photon_density(0.0, 0.0, 0.0, 0.0))
    # The stored width is the density RMS: one sigma out, the density is down by e^-1/2.
    assert float(laser.photon_density(sigma, 0.0, 0.0, 0.0)) / on_axis == pytest.approx(
        math.exp(-0.5), rel=1e-9
    )
    # The 1/e^2 radius is therefore at 2 sigma, not at sigma.
    w0 = 2.0 * sigma
    assert float(laser.photon_density(w0, 0.0, 0.0, 0.0)) / on_axis == pytest.approx(
        math.exp(-2.0), rel=1e-9
    )
    # Textbook formula, in the convention it is actually stated in.
    assert laser.rayleigh_x() == pytest.approx(math.pi * w0**2 / lam, rel=1e-12)
    # And the far-field divergence that follows from it — the quantity RES040 is about.
    assert sigma / laser.rayleigh_x() == pytest.approx(lam / (4.0 * math.pi * sigma), rel=1e-12)


def test_astigmatism_puts_the_two_waists_at_different_places():
    laser = make_laser(z_fx=Q(-0.05, "cm"), z_fy=Q(0.05, "cm"))
    s1, s2 = laser.spot_sizes(np.array([-0.05]))
    assert float(s1[0]) == pytest.approx(laser.m("sigma_x"))  # axis 1 at its own waist
    assert float(s2[0]) > laser.m("sigma_y")  # axis 2 defocused there
    # The joint peak sits between the two waists, below the stigmatic value.
    assert laser.a0_peak() < make_laser().a0_peak()


def test_elliptical_spot_is_independent_of_astigmatism():
    laser = make_laser(sigma_x=Q(5, "um"), sigma_y=Q(20, "um"))
    s1, s2 = laser.spot_sizes(np.array([0.0]))
    assert float(s1[0]) == pytest.approx(5e-4)
    assert float(s2[0]) == pytest.approx(20e-4)


def test_field_envelope_matches_a0_profile():
    laser = make_laser()
    points = np.linspace(-1e-3, 1e-3, 11)
    zeros = np.zeros_like(points)
    carrier = laser.field(points, zeros, zeros, 0.0)
    magnitude = np.linalg.norm(carrier, axis=0)
    envelope = laser.a0_profile(points, zeros, zeros, 0.0)
    assert np.all(magnitude <= envelope * (1.0 + 1e-12))


def test_field_is_polarized_along_p1():
    laser = make_laser(psi_pol=Q(0.6, "rad"), theta_xz=Q(0.2, "rad"))
    _, p1, _ = laser.polarization_axes()
    carrier = laser.field(1e-4, 1e-4, 0.0, 0.0)
    direction = carrier / np.linalg.norm(carrier)
    assert abs(abs(float(np.dot(direction, p1))) - 1.0) < 1e-12


def test_field_oscillates_at_the_carrier_wavelength():
    laser = make_laser()
    # Sampling on-axis in z at fixed t, the carrier period along the propagation
    # direction is the wavelength.
    z = np.linspace(-2e-4, 2e-4, 4001)
    carrier = laser.field(np.zeros_like(z), np.zeros_like(z), z, 0.0)
    signal = carrier[0]  # head-on with psi_pol = 0, the polarization is along x
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    expected = 2.0 * (z[-1] - z[0]) / laser.m("wavelength")
    assert zero_crossings == pytest.approx(expected, rel=0.02)


# -- active region and the prefilter contract --------------------------------
def test_active_region_contains_the_peak_and_excludes_the_far_field():
    laser = make_laser()
    region = laser.active_region(1e-3)
    assert isinstance(region, ActiveRegion)
    assert region.contains(0.0, 0.0, 0.0, 0.0)
    assert not region.contains(10.0 * region.radius, 0.0, 0.0, 0.0)


def test_active_region_tracks_timing_offset():
    """ActiveRegion.contains evaluates to True at the pulse center for non-zero t_off."""
    t_off_s = 50e-15
    laser = make_laser(t_off=Q(t_off_s, "s"))
    region = laser.active_region(1e-3)
    assert region.contains(0.0, 0.0, 0.0, t_off_s)


def test_active_region_is_conservative_at_its_own_threshold():
    # Everything at or above the threshold must be inside — that is what makes the
    # prefilter a pure optimization (§3.2).
    laser = make_laser()
    threshold = 1e-3
    region = laser.active_region(threshold)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-4e-3, 4e-3, 4000), rng.uniform(-4e-3, 4e-3, 4000)
    z = rng.uniform(-8e-3, 8e-3, 4000)
    above = laser.a0_profile(x, y, z, 0.0) >= threshold * region.a0_peak
    assert np.all(region.contains(x[above], y[above], z[above], 0.0))


def test_active_region_is_still_conservative_far_from_focus():
    """The pulse diverges, and the region has to diverge with it.

    Regression guard for a real defect: the region was a cylinder whose radius came from
    the spot *near focus*, so beyond a few Rayleigh ranges the expanded pulse reached
    particles the prefilter had already discarded. The Phase-2 harness found it by
    sampling ``a0`` along discarded trajectories; this is the same statement at the
    laser level, where the fix lives.
    """
    for beta_ff in (0.0, 0.5, 2.0, -0.5):
        laser = make_laser(beta_ff=beta_ff)
        threshold = 1e-3
        region = laser.active_region(threshold)
        rng = np.random.default_rng(4)
        for distance in (0.0, 5.0, 25.0):
            u = distance * laser.rayleigh_x()
            # The pulse centre is at this u at t = u / c, so the longitudinal factor is 1;
            # the spot is set by the flying-focus coordinate, not by u.
            spot = max(laser.spot_sizes(u * (1.0 + beta_ff)))
            x, y = (rng.uniform(-4 * spot, 4 * spot, 3000) for _ in range(2))
            z = np.full_like(x, -u)  # head-on: the axis is -z, so u = -z
            t = np.full_like(x, u / C_CGS)
            above = laser.a0_profile(x, y, z, t) >= threshold * region.a0_peak
            assert np.any(above), f"nothing above threshold at {distance} Rayleigh ranges"
            assert np.all(region.contains(x[above], y[above], z[above], t[above])), (
                f"beta_ff={beta_ff}, {distance} Rayleigh ranges from focus"
            )


def test_the_flying_focus_cancels_out_of_the_cone_slope_but_not_its_intercept():
    """Two ``beta_ff`` effects meet in the cone, and only one of them cancels.

    The spot is evaluated at ``u + beta_ff*ct``, which steepens the cone by
    ``|1 + beta_ff|`` — exactly undoing the ``(1 + beta_ff)`` stretch already in the
    Rayleigh range, so the *slope* is beta_ff-independent. The intercept is not: within
    the pulse length ``ct`` drifts from ``u`` by up to ``half_length``, and the region has
    to allow for the spot that drift reaches. Dropping the slide leaves the slope short by
    that same factor — the region narrower than the pulse, in the one direction a
    conservative bound may not err.
    """
    still, flying = make_laser(), make_laser(beta_ff=1.0)
    assert flying.active_region(1e-3).radius_slope == pytest.approx(
        still.active_region(1e-3).radius_slope
    )
    assert flying.active_region(1e-3).radius > still.active_region(1e-3).radius


def test_psi_focus_decides_which_lab_direction_carries_which_waist():
    """``psi_focus`` was serialized and round-tripped, but nothing checked it does anything.

    It is a **no-op for a round, stigmatic beam**, which is exactly why an untested one
    looks healthy: every existing laser test used ``sigma_x == sigma_y`` and both waists at
    the origin, where rotating the focusing axes cannot change a single number. For the
    elliptical, astigmatic beam §3.3 supports it decides which lab direction is measured
    against ``sigma_x``/``z_fx`` and which against ``sigma_y``/``z_fy``.

    This is the same shape of gap as the flying focus: a first-class schema parameter that
    reaches the serializer but never meets the physics.
    """
    elliptical = dict(sigma_x=Q(10, "um"), sigma_y=Q(40, "um"),
                      z_fx=Q(0, "um"), z_fy=Q(400, "um"))
    upright = make_laser(**elliptical)
    turned = make_laser(psi_focus=Q(math.pi / 2, "rad"), **elliptical)

    # A quarter turn takes focusing axis 1 from the lab x-axis to the lab y-axis.
    _, f1_upright, _ = upright.focusing_axes()
    _, f1_turned, _ = turned.focusing_axes()
    assert np.allclose(f1_upright, [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(f1_turned, [0.0, 1.0, 0.0], atol=1e-12)

    # Sample off-axis and away from both waists, so the elliptical *and* astigmatic parts
    # of the profile are both in play; t places the pulse centre at this u (head-on, u = -z).
    offset, z = 25e-4, 0.03
    t = -z / C_CGS
    assert turned.a0_profile(offset, 0.0, z, t) == pytest.approx(
        upright.a0_profile(0.0, offset, z, t), rel=1e-12
    )
    # ...and the rotation genuinely changed the profile, so the equality above is not the
    # trivial one a round beam would also satisfy.
    assert turned.a0_profile(offset, 0.0, z, t) != pytest.approx(
        upright.a0_profile(offset, 0.0, z, t), rel=1e-3
    )


def test_psi_focus_is_inert_for_a_round_stigmatic_beam():
    """The degenerate case, asserted rather than assumed — it is why the gap was invisible.

    This one **cannot** catch a dropped ``psi_focus``: an implementation that ignored the
    parameter entirely would pass it. That is the point. It records that the previous
    silence was the beam being round, not the parameter being verified, so the guard above
    is not mistaken for redundant.
    """
    round_beam = make_laser()
    turned = make_laser(psi_focus=Q(0.7, "rad"))
    for point in [(0.0, 0.0, 0.0, 0.0), (12e-4, -8e-4, 0.01, -0.01 / C_CGS)]:
        assert turned.a0_profile(*point) == pytest.approx(round_beam.a0_profile(*point), rel=1e-12)


# -- descriptive fit (§3.3) --------------------------------------------------
def test_quasi_monochromatic_conforming_laser_runs_without_gaussian_fitter():
    """A conforming LaserField with carrier/polarization invariants executes across engines without fit_gaussian_paraxial (RES067)."""
    from gammaforge.engines.delta.engine import DeltaEngine
    from gammaforge.engines.xigma.engine import XigmaEngine
    from gammaforge.io.bunch import GaussianElectronBeam
    from gammaforge.io.interaction import build_interaction, SamplingSpec
    from gammaforge.io.target import auto_ranges, OutputKind, OutputRequest, Target

    inner = make_laser()

    class QuasiMonochromaticLaser:
        def __init__(self, inner):
            self._inner = inner
            self.ellipticity = inner.ellipticity

        def intensity_profile(self, x, y, z, t):
            return self._inner.intensity_profile(x, y, z, t)

        def carrier_phase_four_gradient(self, x, y, z, t):
            return self._inner.carrier_phase_four_gradient(x, y, z, t)

        def a0_profile(self, x, y, z, t):
            return self._inner.a0_profile(x, y, z, t)

        def field(self, x, y, z, t):
            return self._inner.field(x, y, z, t)

        def active_region(self, threshold: float):
            return self._inner.active_region(threshold)

        def omega0(self):
            return self._inner.omega0()

        def photon_energy(self):
            return self._inner.photon_energy()

        def intensity_peak(self):
            return self._inner.intensity_peak()

        def focusing_axes(self):
            return self._inner.focusing_axes()

        def polarization_axes(self):
            return self._inner.polarization_axes()

    laser = QuasiMonochromaticLaser(inner)
    assert isinstance(laser, LaserField)

    beam = GaussianElectronBeam(
        bunch_charge=Q(100, "pC"),
        kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.001,
        sigma_x=Q(10, "um"),
        sigma_y=Q(10, "um"),
        emit_x=Q(0.05, "um") * Q(1.0, "rad"),
        emit_y=Q(0.05, "um") * Q(1.0, "rad"),
        sigma_z=Q(1.0, "ps"),
    )
    target = Target(
        theta_x_col=Q(1.0, "mrad"),
        theta_y_col=Q(1.0, "mrad"),
        outputs=(OutputRequest(OutputKind.SPECTRUM, resolution=(16,)),),
    )

    # auto_ranges succeeds without fit_gaussian_paraxial
    ranges = auto_ranges(target, beam, laser)
    assert OutputKind.SPECTRUM in ranges

    # xigma and the validation-only delta reference both execute without
    # fit_gaussian_paraxial; analytical is the documented exception (RES067).
    interaction = build_interaction(beam, laser, target, SamplingSpec(n_particles=16, seed=1))
    xigma_res = XigmaEngine().run(
        interaction, XigmaEngine.schema.with_values(backend="numpy")
    )
    assert OutputKind.SPECTRUM in xigma_res.photon_slices

    delta_res = DeltaEngine().run(
        interaction, DeltaEngine.schema.with_values(backend="numpy")
    )
    assert OutputKind.SPECTRUM in delta_res.photon_slices


# -- validation --------------------------------------------------------------
def test_ellipticity_is_applied_to_angle_resolved_kernel():
    # §9.2/DER004: ellipticity is now applied to the angle-resolved kernel.
    # The total yield and mean red-shift (ahat) are polarization-agnostic by invariance,
    # but the angle-resolved spectrum shape depends on ellipticity.
    assert not ELLIPTICITY_IS_NOOP
    # a0_peak is still linear-equivalent by convention (RES054) — no change there
    assert make_laser(ellipticity=0.5).a0_peak() == pytest.approx(make_laser().a0_peak())
    # validate no longer warns about ellipticity being unapplied
    warnings = validate(make_laser(ellipticity=0.5))
    assert not any("ellipticity" in warning for warning in warnings)
    # Linear polarization should also not warn
    warnings = validate(make_laser())
    assert not any("ellipticity" in warning for warning in warnings)

def test_crossing_angle_is_applied_to_emission_physics():
    """§9.3/DER005/DER006: crossing angle now enters emission physics in three places:
    (1) relative-velocity factor, (2) resonance/energy conversion cos²(α/2),
    (3) polarization structure v·e_i terms.
    """
    assert not EMISSION_IS_HEAD_ON
    tilted = make_laser(theta_xz=Q(0.2, "rad"))
    # validate no longer warns about crossing angle physics being unapplied
    warnings = validate(tilted)
    assert not any("crossing angle" in warning for warning in warnings)
    # Geometry is still applied (rotation matrix)
    assert not np.allclose(lab_frame_axes(0.2, 0.0, 0.0)[0], lab_frame_axes(0.0, 0.0, 0.0)[0])
    # Head-on should also not warn
    warnings = validate(make_laser())
    assert not any("crossing angle" in warning for warning in warnings)

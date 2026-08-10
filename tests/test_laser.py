"""Laser geometry, sampling contract and descriptive fit (GRAND_PLAN.md §2.2/§3.3, Phase 1 exit)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gammaforge.io.laser import (
    ELLIPTICITY_IS_NOOP,
    EMISSION_IS_HEAD_ON,
    ActiveRegion,
    GaussianParaxialLaser,
    LaserField,
    fit_gaussian_paraxial,
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
    _, e1_psi, e2_psi = lab_frame_axes(0.3, 0.2, 0.7)
    assert np.dot(e1_zero, e1_psi) == pytest.approx(math.cos(0.7))
    # psi is measured from the transported x-axis, so a full turn is the identity.
    _, e1_full, _ = lab_frame_axes(0.3, 0.2, 2.0 * math.pi)
    assert np.allclose(e1_full, e1_zero)


# -- the LaserField contract -------------------------------------------------
def test_gaussian_paraxial_laser_satisfies_the_protocol():
    assert isinstance(make_laser(), LaserField)


def test_a0_profile_and_field_are_array_callable_and_broadcast():
    laser = make_laser()
    x = np.linspace(-2e-3, 2e-3, 7)
    y = np.zeros_like(x)
    envelope = laser.a0_profile(x, y, y, 0.0)
    assert envelope.shape == x.shape
    carrier = laser.field(x, y, y, 0.0)
    assert carrier.shape == (3, *x.shape)
    # Scalars work too.
    assert np.isscalar(float(laser.a0_profile(0.0, 0.0, 0.0, 0.0)))


def test_photon_density_integrates_to_one_over_space():
    # The normalization the whole energy->a0 chain rests on (§3.3).
    laser = make_laser()
    grid = np.linspace(-6e-3, 6e-3, 161)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    step = grid[1] - grid[0]
    total = laser.photon_density(x, y, z, 0.0).sum() * step**3
    assert total == pytest.approx(1.0, rel=1e-4)


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


def test_the_rayleigh_range_converts_rms_to_the_1_over_e_squared_convention():
    """The conversion `rayleigh_x` exists to perform, pinned against the profile itself
    rather than against its own formula (author-confirmed 2026-08-10, `DECISIONS.md` D040).

    ``z_R = pi w0^2 / lambda`` is stated in the **1/e² convention**, but this class stores
    intensity **RMS** widths — at ``r = sigma`` the density is down by ``e^-1/2``, not
    ``e^-2``. Skipping the conversion is a factor of 4 in ``z_R`` and in the far-field
    angle, which is exactly the predecessor error D040 pins at 3.285x in the baseline
    yield. So this measures ``w0`` off `photon_density` directly, and only then checks the
    textbook formula against it — a test written as ``4 pi sigma^2 / lambda`` would restate
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
    # And the far-field divergence that follows from it — the quantity D040 is about.
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


def test_a_looser_threshold_gives_a_larger_region():
    laser = make_laser()
    assert laser.active_region(1e-2).radius < laser.active_region(1e-6).radius
    assert laser.active_region(1e-2).radius_slope < laser.active_region(1e-6).radius_slope
    assert laser.active_region(1e-2).half_length < laser.active_region(1e-6).half_length


def test_active_region_rejects_a_threshold_outside_the_unit_interval():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="must be in"):
            make_laser().active_region(bad)


# -- descriptive fit (§3.3) --------------------------------------------------
def test_fit_gaussian_paraxial_is_the_identity_on_a_gaussian_laser():
    laser = make_laser(sigma_x=Q(5, "um"), sigma_y=Q(15, "um"), z_fx=Q(0.01, "cm"), theta_xz=Q(0.3, "rad"), beta_ff=0.2)
    assert fit_gaussian_paraxial(laser) is laser


def test_fit_gaussian_paraxial_refuses_an_unknown_field_rather_than_guessing():
    class Elsewhere:
        def a0_profile(self, x, y, z, t): ...
        def field(self, x, y, z, t): ...
        def active_region(self, threshold): ...

    with pytest.raises(NotImplementedError, match="no numerical path yet"):
        fit_gaussian_paraxial(Elsewhere())


# -- validation --------------------------------------------------------------
def test_validate_rejects_impossible_values():
    for bad in [dict(pulse_energy=Q(0.0, "J")), dict(wavelength=Q(-1.0, "nm")), dict(sigma_x=Q(0.0, "um")),
                dict(duration=Q(0.0, "fs")), dict(beta_ff=-1.0), dict(ellipticity=1.5)]:
        with pytest.raises(ValueError):
            validate(make_laser(**bad))


def test_ellipticity_is_a_documented_no_op_not_a_silent_one():
    # §9.2/P14c: the derivation does not exist, so the parameter is carried, ignored, and
    # *said* to be ignored — never silently approximated.
    assert ELLIPTICITY_IS_NOOP
    assert make_laser(ellipticity=0.5).a0_peak() == pytest.approx(make_laser().a0_peak())
    warnings = validate(make_laser(ellipticity=0.5))
    assert any("ellipticity" in warning for warning in warnings)
    assert not any("ellipticity" in warning for warning in validate(make_laser()))


def test_a_crossing_angle_warns_that_only_the_geometry_is_applied():
    """§9.3/P14c, the sibling of the ellipticity no-op — and the harder one to notice.

    A crossing angle is not ignored: `rotation_matrix` carries it into every sampling
    position, so overlap and timing genuinely change. What does not change is the
    *emission* — xigma holds `RELATIVE_VELOCITY` at 2 and measures kernel angles from the
    collinear axis. A caller who sees the geometry respond has every reason to assume the
    physics did too, which is exactly why this one has to be said rather than inferred.
    """
    assert EMISSION_IS_HEAD_ON
    tilted = make_laser(theta_xz=Q(0.2, "rad"))
    warnings = validate(tilted)
    assert any("crossing angle" in warning for warning in warnings)
    assert any("crossing angle" in warning for warning in validate(make_laser(theta_yz=Q(0.2, "rad"))))
    assert not any("crossing angle" in warning for warning in validate(make_laser()))
    # The geometry really is applied — this is a warning about the physics, not about a
    # parameter that does nothing.
    assert not np.allclose(lab_frame_axes(0.2, 0.0, 0.0)[0], lab_frame_axes(0.0, 0.0, 0.0)[0])

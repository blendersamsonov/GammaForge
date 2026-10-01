"""The energy-integrated angular distribution (DER019 §24.2).

An `OutputKind.ANGULAR_DISTRIBUTION` for the analytical engine: `dN/dOmega` in the bunch
frame, normalized to the analytical total yield, computed by a fixed deterministic
quadrature over gamma with **no macroparticles**.

**Why this is a clean validation target.** Integrating over photon energy removes the
resonance delta function, so the angular probability of a single scattering event does not
depend on the nonlinear line shift or on the laser envelope at all (DER019 §24). Those set
photon *energy*, not the normalized Thomson angular probability. So this quantity can be
checked against Xigma without trusting any of the nonlinear machinery built in
`nonlinear_spectrum` or `fixed_width` — which is exactly why the handoff calls it a
particularly good semi-analytical validation target.

**The normalization identity, and how it is established.** With `u = gamma^2 theta^2` and
the azimuthally averaged linear-polarization kernel, the verified single-electron `u`-density
(see `diagnostics.circular_capture_fraction` and the note there on DER019's `Pbar`) is

    dP/du = 3 (u^2 + 1) / (2 (1 + u)^4),

whose CDF is DER019's `F_cap`. The solid-angle Jacobian is `dOmega = 2 pi theta dtheta` with
`theta dtheta = du/(2 gamma^2)`, so

    dN/dOmega / Y = gamma^2 / pi * dP/du

for a monoenergetic beam, and an average over the energy spread for a general one. Two
consequences are asserted in `tests/test_analytical.py`: a circular aperture of half-angle
`theta_c` captures exactly `F_cap(gamma^2 theta_c^2)` of the yield (agreement to ~1e-13),
and the integral over a sufficiently wide solid angle reproduces the total yield.

**Scope.** Head-on, zero-emittance, unchirped, first-harmonic — the tier DER019 §24.2
specifies. Finite emittance adds a deterministic electron-angle quadrature and crossing
angle needs the DER012 locally transverse polarization basis; both are deferred to the
collimated tier rather than approximated here. The model descriptor says so, so the planner
records the restriction instead of the engine quietly returning a head-on answer for a
crossed collision.
"""

from __future__ import annotations

import math

import numpy as np

from ...io.bunch import GaussianElectronBeam

__all__ = ["angular_density_per_solid_angle", "thomson_u_density"]


def thomson_u_density(u):
    """``dP/du = 3 (u^2 + 1) / (2 (1 + u)^4)`` — the azimuthally averaged `u`-density.

    This is the density whose CDF is `diagnostics.circular_capture_fraction`'s `F_cap`, and
    whose `z`-form is the verified `G(z; 0)`. Exposed separately because it is the one piece
    of the angular model that is pure physics with no geometry in it, and therefore the piece
    worth checking on its own.
    """
    u = np.asarray(u, dtype=float)
    return 3.0 * (u**2 + 1.0) / (2.0 * (1.0 + u) ** 4)


def angular_density_per_solid_angle(
    beam: GaussianElectronBeam,
    theta_x,
    theta_y,
    total_yield: float,
    n_quad: int = 401,
):
    """``dN/dOmega`` on a ``(theta_x, theta_y)`` grid, normalized to ``total_yield``.

    Evaluated per requested angular point as a one-dimensional deterministic gamma average
    (DER019 §24.2), so cost is ``O(len(theta) * n_quad)`` and independent of any particle
    count — the property that makes this engine usable for broad optimization scans.

    ``theta_x``/``theta_y`` broadcast against each other, so pass a meshgrid for an image.
    The azimuthal average is taken about the electron direction, which for a zero-emittance
    head-on bunch is the bunch axis; the result is a function of ``r = hypot(theta_x, theta_y)``.
    """
    theta_x = np.asarray(theta_x, dtype=float)
    theta_y = np.asarray(theta_y, dtype=float)
    radius = np.hypot(theta_x, theta_y)

    gamma0, sigma_gamma = beam.gamma0(), beam.sigma_gamma()
    if sigma_gamma <= 0.0:
        # `io.bunch.validate` permits exactly zero spread; that is the closed case, not a
        # degenerate grid, so take it rather than dividing by a zero-width Gaussian.
        gammas = np.array([gamma0])
        quadrature_weights = np.array([1.0])
    else:
        span = 6.0 * sigma_gamma
        gammas = np.linspace(max(gamma0 - span, 1.0 + 1e-9), gamma0 + span, n_quad)
        pdf = np.exp(-0.5 * ((gammas - gamma0) / sigma_gamma) ** 2) / (
            sigma_gamma * math.sqrt(2.0 * math.pi)
        )
        quadrature_weights = pdf * np.gradient(gammas)

    gamma_sq = gammas**2
    # (n_gamma, n_theta): u = gamma^2 r^2 broadcast over the requested angles.
    u = gamma_sq[:, None] * radius.ravel()[None, :] ** 2
    per_point = (gamma_sq[:, None] / np.pi * thomson_u_density(u) * quadrature_weights[:, None]).sum(axis=0)
    return total_yield * per_point.reshape(radius.shape)

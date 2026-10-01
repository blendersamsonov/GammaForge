"""Fixed-width deterministic Gaussian source moments (DER019 §3, §4, §22).

This module adds the *distributional* side of the analytical engine: where DER001/DER002
answer "what is the total overlap", and `formulas.overlap_mean_a0_sq` answers "what single
mean intensity does the bunch sample", these functions answer "how is that intensity
distributed across the bunch, without drawing macroparticles" (DER019 §4).

The theorem being implemented (DER019 §3): with the laser's transverse widths frozen and its
envelope Gaussian, completing the square in time along any ballistic electron trajectory
leaves

    A_L(xi) = A_max exp[-1/2 (xi - xi_0)^T K (xi - xi_0)],   rank(K) <= 2

for Gaussian electron labels ``xi ~ N(mu, Sigma)``. The rank-two fact is the whole reason
this stays analytic — it is what makes the nonlinear-amplitude distribution a generalized
chi-square rather than an arbitrary distribution.

From that single closed form come the luminosity-weighted moments, because the luminosity
weight is proportional to ``A_L`` and therefore contributes exactly one extra factor:

    E_L[A_L^m] = J_(m+1) / J_1,     J_n = A_max^n det(I + n Sigma K)^{-1/2} exp(...)

`J_n` needs no special function: it is matrix algebra, so this stays numpy-only. The rank-two
*density* (DER019 §4.1) would want a Bessel ``I_0`` and the noncentral form wants a weighted
chi-square — both are deferred to the deterministic quadrature the handoff permits, because
adding scipy for them would be a dependency this project does not otherwise carry.

**The approximation is the frozen width, and nothing else.** Head-on/elliptical/offset/
crossing-angle geometry is handled exactly (DER019 §3.1: the trajectory constants are
unchanged by crossing angle because the completed-square temporal profile is still Gaussian).
What is *not* handled is the evolution of the spot along the collision, which belongs to the
focused and flying-focus tiers (DER019 §11, §12) and is exactly what `epsilon_L` in
:func:`fixed_width_diagnostics` reports.

Deliberately **not** here: anything that would duplicate `formulas.py`'s overlap model.
The reduction below is built from the same laser/beam inputs as `formulas.overlap_yield`,
and `tests/test_analytical.py` pins the two to agree in their common limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser

__all__ = [
    "FixedWidthReduction",
    "NonlinearMoments",
    "fixed_width_diagnostics",
    "fixed_width_reduction",
    "nonlinear_moments",
    "round_nonlinear_moments",
]

#: `kappa_G = 2/sqrt(3) - 1` (DER019 §3.1): the within-trajectory variance `V_a,shape` as a
#: multiple of `a_shape**2`, for an unchirped Gaussian envelope. It is a property of the
#: temporal profile alone — independent of geometry — so it is a constant here and stays a
#: constant across every tier that keeps the Gaussian envelope.
KAPPA_G = 2.0 / math.sqrt(3.0) - 1.0

#: `c_a` for an unchirped Gaussian envelope: `a_shape = A_L / sqrt(2)` (DER019 §3.1).
#:
#: **What `A_L` is, precisely.** It is the *normalized intensity* along the trajectory, i.e.
#: the laser's `intensity_profile` divided by its peak — which for `GaussianParaxialLaser` is
#: `intensity_peak() = a0_peak()**2 / 2`, already the cycle average of `cos^2` for linear
#: polarization (verified: `a0_peak()**2 == 2 * intensity_peak()`). It is **not** the
#: normalized photon *density*, and `a0**2` is not proportional to that density's square.
#:
#: This distinction is easy to get wrong and silently wrong by a factor of two, because
#: `a_shape` here is dimensionless and normalized to 1 at the trajectory peak, while
#: `formulas.overlap_mean_a0_sq` returns a dimensional `<a0^2>` in the laser's own units. To
#: compare the two:

#:     <a_shape>_L  =  (cycle_average_factor / intensity_peak()) * overlap_mean_a0_sq
#:
#: in the fixed-width limit. `tests/test_analytical.py` asserts exactly that relation rather
#: than comparing the bare numbers, which would only pass for one pulse energy.
C_A_UNCHIRPED = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class FixedWidthReduction:
    """The fixed-width Gaussian quadratic form and everything needed to weight it.

    Holds the *unweighted* Gaussian label moments, so that `J_n` is a pure function of this
    object and the luminosity weighting stays visible at the call site as ``J_(m+1)/J_1``
    rather than hiding inside a helper (DER019 §5).
    """

    #: ``K = A^T W_perp A``: positive semidefinite with rank <= 2 (DER019 §3).
    K: np.ndarray
    #: Electron label covariance, ``Sigma``.
    sigma: np.ndarray
    #: ``mu - xi_0``: the displacement that makes the form noncentral.
    delta: np.ndarray
    #: ``A_max``, the peak normalized intensity reachable along the best trajectory.
    a_max: float
    #: ``h = a^T W a``: the temporal completion coefficient. Crossing angle changes it, but
    #: not the normalized trajectory constants (DER019 §3.1).
    h: float

    def j(self, n: int) -> float:
        """``J_n = E[A_L^n]`` under the *unweighted* Gaussian label distribution.

        The standard Gaussian quadratic-form integral (DER019 §22):

            J_n = A_max^n det(I + n Sigma K)^{-1/2}
                  exp{-n/2 delta^T K (I + n Sigma K)^{-1} delta}

        Exact for every geometry the fixed-width model covers, including offsets and a
        crossing angle, and correct in the rank-one limit because ``det`` then carries the
        single surviving direction.
        """
        if n < 1:
            raise ValueError(f"J_n is defined for n >= 1, got n={n}")
        scaled = np.eye(self.K.shape[0]) + n * self.sigma @ self.K
        sign, logdet = np.linalg.slogdet(scaled)
        if sign <= 0:
            raise ValueError(
                f"I + n Sigma K is not positive definite for n={n} (det={sign}); "
                "K must be positive semidefinite and Sigma positive definite"
            )
        quadratic = float(self.delta @ self.K @ np.linalg.solve(scaled, self.delta))
        return float(self.a_max**n * math.exp(-0.5 * logdet - 0.5 * n * quadratic))


@dataclass(frozen=True)
class NonlinearMoments:
    """Luminosity-weighted nonlinear moments and their two-variance decomposition.

    The two variances are physically different and DER019 §22.2 insists on keeping them
    apart, because conflating them is what forced the empirical
    `formulas.NONLINEAR_BROADENING_RANGE` bracket in the first place:

    * ``var_between`` — spread of *trajectory means* across electrons. Broadens the
      distribution of nominal resonances.
    * ``var_finite_line`` — the DER016 variance *inside* one trajectory, carried
      unresolved by DER016/DER017.

    They add: the law of total variance gives the complete luminosity-weighted instantaneous
    variance, which is what a bandwidth diagnostic actually wants.
    """

    mean_a: float
    mean_a_sq: float
    var_a: float
    var_between: float
    var_finite_line: float
    a_max: float

    @property
    def var_total(self) -> float:
        """``<V_a,shape>_L + Var_L(a_shape)`` — the full instantaneous variance."""
        return self.var_between + self.var_finite_line

    @property
    def relative_spread(self) -> float:
        """``sigma_a / <a>``, the quantity the empirical bracket used to stand in for."""
        if self.mean_a <= 0.0:
            return 0.0
        return math.sqrt(max(self.var_between, 0.0)) / self.mean_a


def fixed_width_reduction(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser
) -> FixedWidthReduction:
    """Build the DER019 §3 quadratic form for a frozen-width Gaussian collision.

    Completed in the lab frame rather than the laser's, because the reduction has to hold
    for **arbitrary** crossing angle. In lab coordinates the normalized intensity along a
    ballistic trajectory with ``w = c t`` is

        E(w, xi) = 1/2 (u + w a)^T W (u + w a),   u = xi - d,  a = beta_0 zhat

    with ``W`` the laser's 3x3 lab-frame precision. Minimizing over ``w`` removes exactly one
    direction,

        K = W - W a a^T W / h,   h = a^T W a,

    so ``K`` is 3x3 of rank 2 for *any* geometry — which is the rank-two fact DER019 §3
    rests on. Building it in the laser's own frame and projecting onto ``(f1, f2)`` would be
    simpler and would be wrong off head-on: there the removed direction is not ``k_hat``, so
    ``(f1, f2)`` is not the range of ``W_perp``.

    The one approximation is the **frozen width**: the laser's transverse precision uses its
    waist sizes rather than its sizes at the collision point. Head-on, elliptical, offset and
    crossing-angle geometry are all handled exactly (DER019 §3.1 — the normalized trajectory
    constants do not change with crossing angle, because the completed-square temporal profile
    is still Gaussian). Evolving the spot along the collision is the focused/flying-focus
    tiers (DER019 §11, §12) and is what :func:`fixed_width_diagnostics` reports as
    ``epsilon_L``; it is not a user-facing knob because DER019 §17.8 leaves the tier's
    automatic acceptance to evidence rather than to a schema field.
    """
    if laser.beta_ff != 0.0:
        raise ValueError(
            "fixed_width_reduction: a flying focus (beta_ff="
            f"{laser.beta_ff!r}) makes the spot size time-dependent, and the frozen-width "
            "reduction freezes it. DER019 §16 is explicit that the 1D frozen-width path is "
            "unsafe here; use the deterministic source map instead."
        )

    k_hat, f1, f2 = laser.focusing_axes()
    # Frozen transverse widths: the waist, i.e. the `u = 0` spot sizes. This is the only line
    # where the focused tiers would differ, which keeps the difference auditable.
    s1, s2 = (float(v) for v in laser.spot_sizes(0.0))

    # Laser precision in lab coordinates: the local (transverse, longitudinal) precision
    # rotated by the laser's own axes, so a crossing angle and psi_focus need no branch.
    w_local = np.diag([1.0 / s1**2, 1.0 / s2**2, 1.0 / laser.sigma_ct() ** 2])
    frame = np.stack([f1, f2, k_hat])  # rows: local axes expressed in lab coordinates
    w_lab = frame.T @ w_local @ frame

    # Relative worldline direction in lab coordinates. Only `a a^T` enters K, so the sign
    # (and whether the bunch runs with or against the pulse) cancels.
    a_dir = np.array([0.0, 0.0, beam.beta0()])
    h = float(a_dir @ w_lab @ a_dir)
    if h <= 0.0:
        raise ValueError(f"fixed_width_reduction: degenerate temporal coefficient h={h!r}")
    w_a = w_lab @ a_dir
    K = w_lab - np.outer(w_a, w_a) / h

    # Electron labels: a Gaussian bunch. The longitudinal entry is kept even though K is
    # blind to it, so the same 3x3 `J_n` algebra covers the zero-emittance case and a
    # conditioned one without a second code path.
    sigma = np.diag([beam.m("sigma_x") ** 2, beam.m("sigma_y") ** 2, beam.m("sigma_z") ** 2])

    # Transverse/timing misalignment enters only as the displacement from the laser's own
    # centre; a centered collision has delta = 0 and the form is central. The lab z origin is
    # where the pulse envelope peaks, so a transverse offset is the whole of the shift here.
    delta = np.array([-laser.m("x_off"), -laser.m("y_off"), 0.0])

    return FixedWidthReduction(K=K, sigma=sigma, delta=delta, a_max=1.0, h=h)


def nonlinear_moments(
    reduction: FixedWidthReduction, *, cycle_average: float = C_A_UNCHIRPED
) -> NonlinearMoments:
    """Exact luminosity-weighted nonlinear moments from a fixed-width reduction.

    ``cycle_average`` is ``c_a`` from DER019 §3.1 — the factor by which the trajectory power
    integral turns normalized intensity into ``a_shape``. It defaults to the unchirped
    Gaussian value and is taken as an argument (rather than hardcoded) so a caller with a
    different carrier convention can pin it, matching how `formulas` reads
    `laser.cycle_average_factor()` rather than assuming ``0.5`` (RES053/RES054).
    """
    j1 = reduction.j(1)
    if j1 <= 0.0:
        raise ValueError(f"J_1 must be positive, got {j1!r}")
    mean_a = cycle_average * reduction.j(2) / j1
    mean_a_sq = cycle_average**2 * reduction.j(3) / j1
    # A negative variance would mean the model is inconsistent, not that the physics allows
    # it; failing loudly beats returning a sqrt-domain NaN later.
    var_between = cycle_average**2 * (reduction.j(3) / j1 - (reduction.j(2) / j1) ** 2)
    if var_between < -1e-12 * max(mean_a_sq, 1.0):
        raise ValueError(
            f"negative luminosity-weighted variance ({var_between!r}); J_n are inconsistent "
            "with a positive-semidefinite K and positive-definite Sigma"
        )
    var_between = max(var_between, 0.0)
    # <V_a,shape>_L = v_a J_3/J_1 with v_a = kappa_G a_shape^2's coefficient (DER019 §22.2).
    var_finite_line = KAPPA_G * mean_a_sq
    return NonlinearMoments(
        mean_a=mean_a,
        mean_a_sq=mean_a_sq,
        var_a=var_between + var_finite_line,
        var_between=var_between,
        var_finite_line=var_finite_line,
        a_max=cycle_average * reduction.a_max,
    )


def round_nonlinear_moments(sigma_e: float, sigma_l: float) -> NonlinearMoments:
    """The closed round head-on form (DER019 §4.2, §22.1), with no matrices at all.

        f_a(a) = (nu / a_max) (a / a_max)^(nu-1),   nu = 1 + sigma_L^2 / sigma_e^2
        <a^m> = [nu / (nu + m)] a_max^m
        sigma_a / <a> = 1 / sqrt[nu (nu + 2)]

    This is the analytic limit :func:`nonlinear_moments` must reproduce, and the place the
    eigenvalue degeneracy is visible by hand: a round beam gives two equal nonzero
    eigenvalues of ``Sigma_L^{1/2} K Sigma_L^{1/2}``, so the rank-two Bessel density of
    DER019 §4.1 collapses to this power law. Kept as an independent implementation on
    purpose — a reduction identity checked against a second expression of the same physics
    is the strongest form this check can take (DER019 §18.4).
    """
    if sigma_e <= 0.0 or sigma_l <= 0.0:
        raise ValueError(f"round_nonlinear_moments needs positive widths, got {sigma_e!r}, {sigma_l!r}")
    nu = 1.0 + sigma_l**2 / sigma_e**2
    a_max = C_A_UNCHIRPED  # unchirped Gaussian, DER019 §4.2
    mean_a = nu / (nu + 1.0) * a_max
    mean_a_sq = nu / (nu + 2.0) * a_max**2
    var_between = mean_a_sq - mean_a**2
    var_finite_line = KAPPA_G * mean_a_sq
    return NonlinearMoments(
        mean_a=mean_a,
        mean_a_sq=mean_a_sq,
        var_a=var_between + var_finite_line,
        var_between=var_between,
        var_finite_line=var_finite_line,
        a_max=a_max,
    )


def fixed_width_diagnostics(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser
) -> dict[str, float]:
    """The dimensionless validity measures DER019 §16 asks the engine to report.

        epsilon_L      = L_int / z_R          spot evolution across the interaction
        epsilon_beta   = L_int / beta_e*      bunch evolution across the interaction
        epsilon_drift  = sigma_theta L_int / sigma_L

    Reported, not thresholded. DER019 §17.8 requires automatic acceptance thresholds to come
    from convergence evidence rather than guessed constants, so these are measurements a user
    (and a later sweep) can reason about; nothing in this module reads them as a decision.
    """
    # The interaction length along the pulse: the longitudinal scales that must be resolved.
    l_int = math.hypot(beam.m("sigma_z"), laser.sigma_ct())
    z_r = min(laser.rayleigh_x(), laser.rayleigh_y())
    beta_e = min(beam.beta_star_x(), beam.beta_star_y())
    sigma_theta = math.sqrt(beam.divergence_x() * beam.divergence_y())
    sigma_l = math.sqrt(laser.m("sigma_x") * laser.m("sigma_y"))
    return {
        "epsilon_L": l_int / z_r if z_r > 0 else math.inf,
        "epsilon_beta": l_int / beta_e if beta_e > 0 else math.inf,
        "epsilon_drift": sigma_theta * l_int / sigma_l if sigma_l > 0 else math.inf,
        "L_int": l_int,
    }

"""Closed-form Compton-source physics: total yield, spectrum-width breakdown, and an
angle-integrated spectrum (GRAND_PLAN.md §4.3). No per-particle Monte Carlo — every
function here costs `O(1)` or `O(n_quad)`, never `O(n_particles)` (the predecessor's
``angle_integrated_spectrum`` used to sum over real macroparticle ``gamma`` samples and
caused a 76.3 GiB allocation at 5,000,000 particles x 2048 energy bins; this module has no
macroparticle argument anywhere, by construction).

Ported (algorithm and constants, not code) from the predecessor's
``ComptonSuite/src/gammaforge/models/analytical.py`` — SI/pint ``CollisionParams``
throughout there, CGS-Gaussian ``GaussianElectronBeam``/``GaussianParaxialLaser`` here
(P1).

**Two yield functions, and which to use.** :func:`overlap_yield` evaluates the general
Gaussian luminosity overlap integral: non-round beams, per-axis focusing, displaced and
astigmatic foci, and a rotated laser ellipse, all exactly. It is what `AnalyticalEngine`
calls, and the derivation behind it is written out in `docs/DERIVATIONS.md` §A.
:func:`estimate_yield` is the predecessor's round-beam closed form, kept because the
general integral reduces to it analytically — which makes it a real regression anchor —
and because it pins port fidelity. It carries an approximation *and* a laser-divergence
convention error; its own docstring says so. Prefer :func:`overlap_yield`.

This closes two of the three growth items `DECISIONS.md` D035 left open (non-round yield,
foci displacement). The third, constructing the collimated spectrum, is still open, as is
the crossing angle (`GRAND_PLAN.md` §9.3) — :func:`overlap_yield` refuses one rather than
returning a number its derivation does not cover, which is what P14c actually asks for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser
from ...io.units import C_CGS, SIGMA_T_CGS

__all__ = [
    "estimate_yield",
    "overlap_det",
    "overlap_yield",
    "SpectrumWidthBreakdown",
    "estimate_spectrum_width",
    "angle_integrated_spectrum",
]

#: Above this, `math.erfc(nu) * math.exp(nu * nu)` is at real risk of overflowing before
#: `erfc` underflows to zero — the exact regime `erfcx` exists to protect against. Never
#: reached by a physically sane scenario (the baseline scenario has ``nu ~ 0.06``, the
#: predecessor's own worked example ``nu ~ 0.48``), but a GUI quick-estimate panel can see
#: extreme user-entered values, so the asymptotic branch below keeps this finite rather
#: than raising `OverflowError`.
_ERFCX_ASYMPTOTIC_THRESHOLD = 25.0


def _erfcx(nu: float) -> float:
    """``exp(nu**2) * erfc(nu)``, the scaled complementary error function.

    Hand-rolled from `math.erfc` rather than `scipy.special.erfcx` — `pyproject.toml`
    declares only `numpy` today, and `gammaforge.io.bunch._chi2_6_cdf` already sets the
    precedent of writing out a closed-form special function rather than adding `scipy` to
    keep the dependency surface where `pyproject.toml` already draws it (`DECISIONS.md`
    D037). Direct evaluation is exact (to `math.erfc`'s own precision) below the overflow
    threshold; above it, the standard asymptotic expansion
    ``erfcx(x) ~ (1/(x*sqrt(pi))) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6))`` takes over.

    ``estimate_yield`` only ever evaluates this at ``nu >= 0`` (built from sums of squares
    under square roots), so a negative branch is not needed.
    """
    if nu < _ERFCX_ASYMPTOTIC_THRESHOLD:
        return math.exp(nu * nu) * math.erfc(nu)
    inv2 = 1.0 / (nu * nu)
    series = 1.0 - 0.5 * inv2 + 0.75 * inv2**2 - 1.875 * inv2**3
    return series / (nu * math.sqrt(math.pi))


def estimate_yield(beam: GaussianElectronBeam, laser: GaussianParaxialLaser, N_e: float) -> float:
    """Cheap analytic total-photon-yield estimate from a Gaussian-bunch overlap integral.

    Not a replacement for a per-particle computation — a sanity-check anchor (§4.3/§7).
    ``laser`` must be a `GaussianParaxialLaser` — the *fitted* descriptive metrics
    (`gammaforge.io.laser.fit_gaussian_paraxial`), never a raw `LaserField`, mirroring how
    `xigma.collision.Collision.run`/`gammaforge.io.target.auto_ranges` already read laser
    scalars (P15). ``N_e`` is explicit rather than derived from ``beam`` so this honors the
    same cheap charge-only recompute path `XigmaEngine` declares (`RecomputeCost.QUERY_ONLY`
    on ``"n_e"``, handled entirely at the `io` level) — defaulting to
    ``beam.n_electrons()`` would desync from a charge-rescaled `InteractionParameters`.

    The laser's transverse profile is treated as round (geometric-mean effective size
    ``sqrt(sigma_x * sigma_y)``) since the underlying formula assumes a round beam — an
    elliptical laser is only approximated, not modeled exactly (see the module docstring).

    .. warning::

       **This function's laser hourglass term disagrees with this repository's own
       Rayleigh-range convention by a factor of 4 in the angle**, and it is kept only as
       a port-fidelity anchor. :func:`overlap_yield` is the one to use.

       The ``lambda^2 / (pi^2 sigma_lr0^2)`` term below is a laser divergence of
       ``lambda / (pi sigma)``. The generalized derivation (`docs/DERIVATIONS.md` §A)
       shows the coefficient is exactly ``sigma_l / z_R``, and both this repository's
       `GaussianParaxialLaser.rayleigh_x` *and the predecessor's own pulse class* define
       ``z_R = 4 pi sigma^2 / lambda`` (``w0 = 2 sigma``), giving ``lambda / (4 pi
       sigma)``. The predecessor's ``analytical.py`` is therefore inconsistent with the
       predecessor's *own* laser model; the port carried that faithfully rather than
       introducing it. On the baseline scenario — where the hourglass is almost entirely
       laser-driven — the discrepancy is a factor of 3.3 in the yield
       (`DECISIONS.md` D040).
    """
    sigma_ex = beam.m("sigma_x")
    sigma_ey = beam.m("sigma_y")
    beta_x = beam.beta_star_x()
    beta_y = beam.beta_star_y()
    sigma_ez = beam.m("sigma_z")
    sigma_lr0 = math.sqrt(laser.m("sigma_x") * laser.m("sigma_y"))
    sigma_lz = laser.m("duration") * C_CGS
    lambda_l = laser.m("wavelength")

    sb_av = math.sqrt(sigma_ex * sigma_ey / beta_x / beta_y)
    sigma0 = math.sqrt(sigma_ex**2 + sigma_lr0**2)
    nu = (
        math.sqrt(2.0)
        * sigma0
        / math.sqrt(sigma_ez**2 + sigma_lz**2)
        / math.sqrt(sb_av**2 + lambda_l**2 / math.pi**2 / sigma_lr0**2)
    )
    return N_e * laser.n_photons() * SIGMA_T_CGS / 2.0 / math.sqrt(math.pi) / sigma0**2 * nu * _erfcx(nu)


def _electron_sigma2(beam: GaussianElectronBeam, z):
    """``(sigma_ex^2(z), sigma_ey^2(z))``: the bunch's transverse variances at lab
    position ``z``, the reference point being ``z = 0``.

    The standard Twiss drift, identical to the one `gammaforge.io.bunch._drift_plane`
    applies (``beta -> beta - 2 alpha L + gamma_twiss L^2``) — the electron-side
    "foci displacement" needs **no new schema field**, because `GaussianElectronBeam`
    already carries it as ``alpha_x``/``alpha_y``: the waist sits at
    ``z_w = alpha beta_0 / (1 + alpha^2)``, so ``alpha > 0`` means a still-converging
    bunch whose waist is downstream. A beam given at its waist (``alpha == 0``) reduces
    to the symmetric ``sigma^2 (1 + z^2 / beta_0^2)`` hourglass.
    """
    out = []
    for sigma, emit, alpha in (
        (beam.m("sigma_x"), beam.m("emit_x"), beam.alpha_x),
        (beam.m("sigma_y"), beam.m("emit_y"), beam.alpha_y),
    ):
        beta_0 = sigma**2 / emit
        gamma_twiss = (1.0 + alpha**2) / beta_0
        out.append(emit * (beta_0 - 2.0 * alpha * z + gamma_twiss * z**2))
    return out[0], out[1]


def _laser_covariance(laser: GaussianParaxialLaser, z):
    """The pulse's transverse covariance ``(c_xx, c_xy, c_yy)`` in **lab** x/y at lab ``z``.

    Two coordinate facts, both read off `GaussianParaxialLaser` rather than assumed:
    the pulse propagates along ``k_hat``, which is ``-z`` head-on, so its own longitudinal
    coordinate is ``u = -z`` and its per-axis waists ``z_fx``/``z_fy`` (offsets *along*
    ``k_hat``) sit at lab ``z = -z_fx``/``-z_fy``; and ``psi_focus`` rotates the focusing
    axes within the transverse plane, so the pulse's variance ellipse is generally **not**
    diagonal in the bunch's own x/y. Carrying the full 2x2 covariance rather than a pair
    of widths is what lets `overlap_yield` handle a rotated elliptical spot exactly
    instead of approximating it (`DECISIONS.md` D039).
    """
    s1, s2 = laser.spot_sizes(-np.asarray(z, dtype=float))
    psi = laser.m("psi_focus")
    c, s = math.cos(psi), math.sin(psi)
    s1sq, s2sq = s1**2, s2**2
    return (
        c * c * s1sq + s * s * s2sq,
        c * s * (s1sq - s2sq),
        s * s * s1sq + c * c * s2sq,
    )


def overlap_det(beam: GaussianElectronBeam, laser: GaussianParaxialLaser, z):
    """``det(C_e(z) + C_l(z))``, the determinant of the summed transverse covariances.

    The transverse part of the luminosity overlap integral is
    ``1 / (2 pi sqrt(det(C_e + C_l)))`` — a single scalar that already accounts for
    unequal x/y sizes, unequal x/y focusing, astigmatic laser waists and a ``psi_focus``
    rotation between the two ellipses. Reducing it to the round-beam ``1 / (2 pi
    sigma_0^2)`` requires *both* ellipses to be circular; nothing here assumes that.
    """
    ex2, ey2 = _electron_sigma2(beam, z)
    c_xx, c_xy, c_yy = _laser_covariance(laser, z)
    return (ex2 + c_xx) * (ey2 + c_yy) - c_xy**2


def _overlap_grid(beam, laser, sigma_z_eff: float, n_quad: int) -> np.ndarray:
    """Quadrature nodes for :func:`overlap_yield`, resolving every longitudinal scale.

    The integrand is a Gaussian of width ``sigma_z_eff`` times ``1/sqrt(det)``, and those
    two carry independent scales: the four hourglass/Rayleigh lengths can each be orders
    of magnitude *shorter* than the Gaussian (the baseline scenario's ``z_R`` is 1.2 mm
    against a 9.5 mm longitudinal overlap), and displaced foci push their structure away
    from ``z = 0``. A single uniform grid over the Gaussian support would silently
    under-resolve exactly the peak that dominates the answer, so the nodes are the union
    of a grid over the Gaussian support and one locally refined window per waist — which
    keeps convergence independent of how the foci are placed, not merely true at the
    aligned default.
    """
    span = 8.0 * sigma_z_eff
    grids = [np.linspace(-span, span, n_quad)]

    beta_0x = beam.m("sigma_x") ** 2 / beam.m("emit_x")
    beta_0y = beam.m("sigma_y") ** 2 / beam.m("emit_y")
    waists = (
        (beam.alpha_x * beta_0x / (1.0 + beam.alpha_x**2), beta_0x / (1.0 + beam.alpha_x**2)),
        (beam.alpha_y * beta_0y / (1.0 + beam.alpha_y**2), beta_0y / (1.0 + beam.alpha_y**2)),
        (-laser.m("z_fx"), laser.rayleigh_x()),
        (-laser.m("z_fy"), laser.rayleigh_y()),
    )
    for center, scale in waists:
        lo = max(center - 8.0 * scale, -span)
        hi = min(center + 8.0 * scale, span)
        if hi > lo:
            grids.append(np.linspace(lo, hi, n_quad))
    return np.unique(np.concatenate(grids))


def overlap_yield(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, N_e: float, n_quad: int = 2001
) -> float:
    """Total photon yield from the **general** Gaussian luminosity overlap integral.

    Supersedes :func:`estimate_yield`'s round-beam closed form: it drops the round-beam
    and aligned-foci approximations and keeps the collision geometry exactly as
    `gammaforge.io` already describes it — per-axis bunch sizes and emittances, per-axis
    Twiss ``alpha`` (electron waist displacement), per-axis laser waists and Rayleigh
    ranges, astigmatic ``z_fx``/``z_fy`` focal offsets, and the ``psi_focus`` rotation
    between the two transverse ellipses. Head-on only (see the guards below).

    Derivation in `docs/DERIVATIONS.md` §A. In outline: the yield is
    ``sigma_T (1 + beta_0) c`` times the space-time overlap of the two densities; the two
    transverse integrals are Gaussian and collapse to ``1 / (2 pi sqrt(det(C_e + C_l)))``
    (:func:`overlap_det`), and the time integral is Gaussian and collapses to a
    longitudinal weight of width ``sigma_z_eff = D / (1 + beta_0)`` with
    ``D = sqrt(sigma_ez^2 + beta_0^2 sigma_lz^2)``. What is left is the single
    longitudinal quadrature this function evaluates::

        N = sigma_T (1 + beta_0) N_e N_L / (2 pi sqrt(2 pi) D)
            * Int dz exp(-z^2 / (2 sigma_z_eff^2)) / sqrt(det(C_e(z) + C_l(z)))

    The integrand is strictly positive and smooth, so a fixed grid converges fast and
    monotonically — there is no cancellation to lose precision to.

    In the round, aligned, ``alpha = 0`` limit this reduces **analytically** to
    :func:`estimate_yield`'s closed form with ``nu = L (1 + beta_0) / (sqrt(2) D)``, and
    `tests/test_analytical.py` pins that reduction numerically to ~1e-12 rather than
    asserting it in a comment.

    Raises ``ValueError`` for a crossing angle (``theta_xz``/``theta_yz`` nonzero) or a
    flying focus (``beta_ff`` nonzero). Neither is an oversight: a crossing angle is
    `GRAND_PLAN.md` §9.3's open derivation, and a flying focus makes the spot-size
    evaluation point ``u_spot = u + beta_ff * ct`` time-dependent, which is precisely what
    breaks the analytic time integration this whole result rests on. Refusing beats
    returning a number whose derivation does not apply (P14c).
    """
    if laser.m("theta_xz") != 0.0 or laser.m("theta_yz") != 0.0:
        raise ValueError(
            "overlap_yield: the overlap integral is derived head-on; a crossing angle "
            f"(theta_xz={laser.m('theta_xz')!r}, theta_yz={laser.m('theta_yz')!r}) is "
            "GRAND_PLAN.md §9.3's open derivation, not something this closed form covers"
        )
    if laser.beta_ff != 0.0:
        raise ValueError(
            f"overlap_yield: a flying focus (beta_ff={laser.beta_ff!r}) makes the spot size "
            "depend on time as well as position, which breaks the analytic time integration "
            "this result is derived from"
        )
    if n_quad < 11:
        raise ValueError(f"overlap_yield: n_quad must be >= 11 (got {n_quad!r})")

    beta_0 = beam.beta0()
    sigma_lz = laser.m("duration") * C_CGS
    D = math.sqrt(beam.m("sigma_z") ** 2 + (beta_0 * sigma_lz) ** 2)
    sigma_z_eff = D / (1.0 + beta_0)

    z = _overlap_grid(beam, laser, sigma_z_eff, n_quad)
    integrand = np.exp(-(z**2) / (2.0 * sigma_z_eff**2)) / np.sqrt(overlap_det(beam, laser, z))
    integral = float(np.trapezoid(integrand, z))

    return SIGMA_T_CGS * (1.0 + beta_0) * N_e * laser.n_photons() / (2.0 * math.pi * math.sqrt(2.0 * math.pi) * D) * integral


@dataclass(frozen=True)
class SpectrumWidthBreakdown:
    """The collimated-spectrum FWHM estimate (units of the Compton edge, dimensionless),
    as four independently-reported components (§4.3: "GUI shows a component table; total
    in quadrature") rather than the predecessor's single summed float.

    Each field already carries the predecessor's ``0.5 * 2.355`` FWHM-from-sigma prefactor
    applied to its own term, so :attr:`total` — `math.hypot` of the four fields — squares
    and re-sums them, exactly reproducing the predecessor's single
    ``0.5 * 2.355 * sqrt(term1 + term2 + term3 + term4)`` formula. The leading ``0.5`` is
    carried unexplained, as it was in the ported source — not retrofitted with a
    justification the original never had.
    """

    collimation: float  #: from angular collimation, ``(gamma * theta_col)^2``
    emittance: float  #: from angular divergence, ``(gamma * sqrt(div_x * div_y))^2``
    energy_spread: float  #: from beam energy spread, ``sigma_gamma / gamma``
    nonlinearity: float  #: from ponderomotive broadening, ``0.5 * a0_peak^2``

    @property
    def total(self) -> float:
        return math.hypot(self.collimation, self.emittance, self.energy_spread, self.nonlinearity)


def estimate_spectrum_width(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, theta_col: float
) -> SpectrumWidthBreakdown:
    """Collimated-spectrum FWHM estimate, broken into its four components (§4.3).

    ``theta_col``: collimation half-angle (rad) — a single scalar; a caller combining
    `gammaforge.io.target.Target`'s separate ``theta_x_col``/``theta_y_col`` should use
    their geometric mean (`DECISIONS.md` D038), the same x/y-combining convention this
    module already uses for the laser waist (``sigma_lr0``) and the emittance term below.

    ``laser`` is the fitted `GaussianParaxialLaser` (see :func:`estimate_yield`);
    ``laser.a0_peak()`` stands in for the predecessor's ``pulse.a0_interaction`` — the
    pulse's own maximum a0, not the a0 at the electron bunch's actual position, which is
    the "foci displacement" growth item §4.3 lists as open (`DECISIONS.md` D035).
    """
    gamma0 = beam.gamma0()
    sigma_gamma = beam.sigma_gamma()
    emit_width = math.sqrt(beam.divergence_x() * beam.divergence_y())
    a0 = laser.a0_peak()
    prefactor = 0.5 * 2.355
    return SpectrumWidthBreakdown(
        collimation=prefactor * (gamma0 * theta_col) ** 2,
        emittance=prefactor * (gamma0 * emit_width) ** 2,
        energy_spread=prefactor * (sigma_gamma / gamma0),
        nonlinearity=prefactor * (0.5 * a0**2),
    )


def angle_integrated_spectrum(gamma0: float, sigma_gamma: float, N_e: float, s, n_quad: int = 401):
    """``dN/ds``, integrated over all emission solid angle and over the beam's own
    (assumed Gaussian) energy distribution — a fixed-size quadrature over ``gamma``, no
    macroparticles (§4.3): cost is ``O(n_quad * len(s))``, independent of ``n_particles``
    by construction.

    Implements its **own** copy of the standard linear-Compton kinematic shape
    (``1.5 * (1 - 2y(1-y))`` for ``y = s / gamma**2``) — the same shape
    `gammaforge.engines.xigma.stages.angle_integrated_spectrum` and
    `gammaforge.validation.references.delta.single_electron_spectrum` also implement,
    **deliberately not imported from either**. Analytical is the third independent leg of
    §7's four-method cross-validation (xigma vs delta vs analytical vs kascade); importing
    the shape from either of the other two would make that specific comparison circular,
    the same reasoning `xigma/stages.py`'s own copy already documents.

    ``s``: scalar or array of normalized photon energies (``s = E / (4 * photon_energy)``,
    matching the convention `xigma.collision.Collision` uses for the same axis). ``N_e``
    scales the output to an absolute photon count: the quadrature weights integrate to 1
    (to quadrature precision) over the sampled +-6 ``sigma_gamma`` window, so multiplying
    by ``N_e`` reproduces "one scattering attempt per electron," not yet a photon count —
    see `gammaforge.engines.analytical.engine` for how the caller turns this into a real
    spectrum by construction against `estimate_yield`.

    ``n_quad``: quadrature points spanning +-6 ``sigma_gamma`` around ``gamma0`` —
    independent of ``n_particles``, so a generous default costs nothing.

    Raises ``ValueError`` for ``sigma_gamma <= 0``: `gammaforge.io.bunch.validate` permits
    a beam with exactly zero energy spread (only rejects negative), but the quadrature
    grid this function builds is degenerate there (a zero-width Gaussian divided by its
    own zero width) and would otherwise return `nan` silently rather than raising —
    exactly the silent-fallback failure mode this repo's conventions reject in favor of
    an explicit error (see `engines.xigma.stages._check_backend`'s docstring for the same
    argument in a different context).
    """
    if sigma_gamma <= 0.0:
        raise ValueError(
            f"angle_integrated_spectrum: sigma_gamma must be > 0 (got {sigma_gamma!r}) — "
            "a zero-width energy spread makes the quadrature grid degenerate"
        )
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

    span = 6.0 * sigma_gamma
    gamma_grid = np.linspace(max(gamma0 - span, 1.0 + 1e-9), gamma0 + span, n_quad)
    pdf = np.exp(-0.5 * ((gamma_grid - gamma0) / sigma_gamma) ** 2) / (sigma_gamma * math.sqrt(2.0 * math.pi))
    quad_weight = pdf * np.gradient(gamma_grid) * N_e

    gamma2 = (gamma_grid**2)[:, None]
    y = s_arr[None, :] / gamma2
    shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    out = np.sum(quad_weight[:, None] * shape / gamma2, axis=0)

    return out if np.ndim(s) else out[0]

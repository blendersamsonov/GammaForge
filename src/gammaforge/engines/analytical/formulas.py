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

**What this closes, precisely.** Of `DECISIONS.md` D035's three growth items, non-round
beams and foci displacement are closed **for the total yield** — that is
:func:`overlap_yield`'s whole point. They are *not* closed for
:func:`estimate_spectrum_width`, whose nonlinearity term still uses the pulse's own peak
a0 rather than the a0 the bunch actually samples; that needs an overlap-weighted
``<a0^2>``, a further derivation nobody has done here. The third item, constructing the
collimated spectrum, is untouched. So is the crossing angle (`GRAND_PLAN.md` §9.3) —
:func:`overlap_yield` refuses one rather than returning a number its derivation does not
cover, which is what P14c actually asks for.
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


def _overlap_quadratic_form(beam: GaussianElectronBeam, laser: GaussianParaxialLaser, z, u_shift: float = 0.0):
    """The reduced integrand of the overlap integral at lab positions ``z``.

    Returns ``(S, det_A, sigma_ex, sigma_ey, s1, s2, h)`` where the longitudinal weight is
    ``exp(-S z^2 / 2)``, the transverse integrals have contributed ``1 / sqrt(det_A)``, and
    ``h`` is the (z-independent) time-integration coefficient the caller needs for the
    overall prefactor. See `docs/DERIVATIONS.md` §A.6; in outline, at fixed ``t`` the
    combined exponent is a quadratic form ``r^T M r / 2`` in lab coordinates with

        M_e = xx^T/sigma_ex^2(z) + yy^T/sigma_ey^2(z) + zz^T/sigma_ez^2
        M_l = f1 f1^T/s1^2(u) + f2 f2^T/s2^2(u) + k k^T/s_ct^2

    Doing the (Gaussian) time integral first replaces ``M`` by ``M' = M - g g^T / h`` with
    ``g = beta_0 zhat/sigma_ez^2 + khat/s_ct^2`` and ``h = beta_0^2/sigma_ez^2 + 1/s_ct^2``;
    integrating the two transverse directions out of ``M'`` then leaves ``det_A`` (its
    upper-left 2x2 block) and the Schur complement ``S = M'_zz - b^T A^-1 b``.

    Head-on this collapses to the §A.4 result exactly — ``S = (1 + beta_0)^2 / D^2`` and
    ``sigma_ex sigma_ey s1 s2 sqrt(det_A) = sqrt(det(C_e + C_l))`` — which is why the
    crossing-angle generalization did not need a second code path.

    ``u_shift`` displaces the point at which the *slowly varying* spot sizes are sampled;
    it exists so `tests/test_analytical.py` can measure the one approximation this
    function makes (see :func:`overlap_yield`), not for production use.
    """
    z = np.asarray(z, dtype=float)
    k_hat, f1, f2 = laser.focusing_axes()
    beta_0 = beam.beta0()
    inv_sez2 = 1.0 / beam.m("sigma_z") ** 2
    s_ct = laser.sigma_ct()
    inv_sct2 = 1.0 / s_ct**2

    ex2, ey2 = _electron_sigma2(beam, z)
    s1, s2 = laser.spot_sizes(k_hat[2] * z + u_shift)

    g = beta_0 * np.array([0.0, 0.0, 1.0]) * inv_sez2 + k_hat * inv_sct2
    h = beta_0**2 * inv_sez2 + inv_sct2

    def m_prime(i: int, j: int):
        diag_e = (1.0 / ex2 if i == 0 else 1.0 / ey2 if i == 1 else inv_sez2) if i == j else 0.0
        return (
            diag_e
            + f1[i] * f1[j] / s1**2
            + f2[i] * f2[j] / s2**2
            + k_hat[i] * k_hat[j] * inv_sct2
            - g[i] * g[j] / h
        )

    a00, a01, a11 = m_prime(0, 0), m_prime(0, 1), m_prime(1, 1)
    b0, b1 = m_prime(0, 2), m_prime(1, 2)
    det_a = a00 * a11 - a01**2
    schur = m_prime(2, 2) - (a11 * b0**2 - 2.0 * a01 * b0 * b1 + a00 * b1**2) / det_a
    return schur, det_a, np.sqrt(ex2), np.sqrt(ey2), s1, s2, h


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

    With a crossing angle two further things change and both are handled here rather than
    assumed away. The longitudinal weight is no longer ``exp(-z^2/2 sigma_z_eff^2)`` but
    ``exp(-S(z) z^2/2)``, and `S` is not monotonic once foci are displaced — so the extent
    comes from a coarse pre-scan (``1/sqrt(min S)``) and the refined core from
    ``1/sqrt(max S)``, neither of them from an assumed inequality. And the crossing angle
    introduces a **new** longitudinal scale, ``s_i / sin(theta)``: the bunch's longitudinal
    extent maps into the pulse's *transverse* coordinate through ``xi_1 ~ x cos - z sin``,
    which is the whole Piwinski suppression, and it can be far shorter than every other
    scale here. A grid windowed only on the waists would silently under-resolve exactly
    the effect the crossing angle is about.
    """
    scan = np.linspace(-12.0 * sigma_z_eff, 12.0 * sigma_z_eff, 401)
    schur = _overlap_quadratic_form(beam, laser, scan)[0]
    schur = schur[np.isfinite(schur) & (schur > 0.0)]
    span = 8.0 / math.sqrt(float(np.min(schur))) if schur.size else 8.0 * sigma_z_eff
    grids = [np.linspace(-span, span, n_quad)]

    beta_0x = beam.m("sigma_x") ** 2 / beam.m("emit_x")
    beta_0y = beam.m("sigma_y") ** 2 / beam.m("emit_y")
    windows = [
        (beam.alpha_x * beta_0x / (1.0 + beam.alpha_x**2), beta_0x / (1.0 + beam.alpha_x**2)),
        (beam.alpha_y * beta_0y / (1.0 + beam.alpha_y**2), beta_0y / (1.0 + beam.alpha_y**2)),
        (-laser.m("z_fx"), laser.rayleigh_x()),
        (-laser.m("z_fy"), laser.rayleigh_y()),
    ]
    if schur.size:
        windows.append((0.0, 1.0 / math.sqrt(float(np.max(schur)))))
    k_hat, _, _ = laser.focusing_axes()
    sin_cross = math.hypot(k_hat[0], k_hat[1])
    if sin_cross > 0.0:
        windows.append((0.0, laser.m("sigma_x") / sin_cross))
        windows.append((0.0, laser.m("sigma_y") / sin_cross))

    for center, scale in windows:
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
    ranges, astigmatic ``z_fx``/``z_fy`` focal offsets, the ``psi_focus`` rotation between
    the two transverse ellipses, **and a crossing angle** (``theta_xz``/``theta_yz``).

    Derivation in `docs/DERIVATIONS.md` §A. In outline: the yield is
    ``sigma_T (1 + beta_0) c`` times the space-time overlap of the two densities, whose
    combined exponent is a quadratic form in ``(x, y, z, t)``. The time integral and the
    two transverse integrals are Gaussian and close analytically
    (:func:`_overlap_quadratic_form`), leaving one longitudinal quadrature::

        N = sigma_T (1 + beta_0) N_e N_L sqrt(2 pi / h) / (4 pi^2 sigma_ez sigma_lz)
            * Int dz exp(-S(z) z^2 / 2) / (sigma_ex sigma_ey s1 s2 sqrt(det A(z)))

    The integrand is strictly positive and smooth, so a fixed grid converges fast and
    monotonically — there is no cancellation to lose precision to.

    Head-on, ``S = (1 + beta_0)^2 / D^2`` and the widths regroup into
    ``sqrt(det(C_e + C_l))`` (:func:`overlap_det`), recovering the simpler §A.4 form; in
    the round, aligned, ``alpha = 0`` limit that reduces **analytically** to
    :func:`estimate_yield`'s closed form. `tests/test_analytical.py` pins both reductions
    numerically rather than asserting them in a comment.

    **The one approximation.** With a crossing angle the bunch's hourglass varies along
    ``z`` while the pulse's varies along ``u = k_hat . r``, which are different directions;
    an exact reduction would leave a 2D quadrature. The spot sizes are therefore sampled at
    ``u = (k_hat . zhat) z``, dropping the transverse contribution
    ``delta = k_x x + k_y y``. Everything in the *exponent* stays exact, including the
    ``xi_1 ~ x cos - z sin`` term that produces the whole crossing-angle (Piwinski)
    suppression — only the argument of the slowly varying widths is approximated, and the
    error is second order in ``delta / z_R``. That ratio, not ``delta / sigma_z``, is what
    must be small: `test_crossing_angle_width_sampling_approximation_is_negligible`
    measures the resulting shift directly, at deliberately adversarial parameters. The
    regime to watch is a large crossing angle with a tight focus and a wide bunch.

    Raises ``ValueError`` for a flying focus (``beta_ff`` nonzero), which makes the
    spot-size evaluation point ``u_spot = u + beta_ff * ct`` time-dependent and so breaks
    the time integration this result rests on. Refusing beats returning a number whose
    derivation does not apply (P14c).

    .. note::

       A crossing angle is covered **for the total yield**. It is not covered for the
       emitted *spectrum*: `GRAND_PLAN.md` §9.3's open item is the polarization structure
       of the emission kernel, a different question from this overlap geometry.
       `AnalyticalEngine` says so on the `Results` when both are in play.
    """
    if laser.beta_ff != 0.0:
        raise ValueError(
            f"overlap_yield: a flying focus (beta_ff={laser.beta_ff!r}) makes the spot size "
            "depend on time as well as position, which breaks the analytic time integration "
            "this result is derived from"
        )
    if n_quad < 11:
        raise ValueError(f"overlap_yield: n_quad must be >= 11 (got {n_quad!r})")

    beta_0 = beam.beta0()
    sigma_lz = laser.sigma_ct()
    D = math.sqrt(beam.m("sigma_z") ** 2 + (beta_0 * sigma_lz) ** 2)

    z = _overlap_grid(beam, laser, D / (1.0 + beta_0), n_quad)
    schur, det_a, sigma_ex, sigma_ey, s1, s2, h = _overlap_quadratic_form(beam, laser, z)
    integrand = np.exp(-0.5 * schur * z**2) / (sigma_ex * sigma_ey * s1 * s2 * np.sqrt(det_a))
    integral = float(np.trapezoid(integrand, z))

    return (
        SIGMA_T_CGS
        * (1.0 + beta_0)
        * N_e
        * laser.n_photons()
        * math.sqrt(2.0 * math.pi / h)
        / (4.0 * math.pi**2 * beam.m("sigma_z") * sigma_lz)
        * integral
    )


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
    pulse's own maximum a0, not the a0 at the electron bunch's actual position.

    **This is the one place foci displacement is still approximated.**
    :func:`overlap_yield` closed that growth item for the *total yield*
    (`DECISIONS.md` D039), but not here: the nonlinearity term needs the a0 the bunch
    actually samples, i.e. an ``<a0^2>`` weighted by the same overlap integral, which is a
    further derivation that has not been done. Displacing the foci therefore changes the
    yield correctly while leaving this width component unmoved — a real remaining
    limitation, not an oversight, and the reason `GRAND_PLAN.md` §11's Phase 4 row still
    carries an exception.
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

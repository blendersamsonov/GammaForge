"""Closed-form Compton-source physics: total yield, spectrum-width breakdown, and an
angle-integrated spectrum. No per-particle Monte Carlo — every
function here costs `O(1)` or `O(n_quad)`, never `O(n_particles)`. This module has no
macroparticle argument anywhere, by construction, so its memory use cannot scale as a
particle-by-energy broadcast.

**Yield evaluation.** :func:`overlap_yield` evaluates the general
Gaussian luminosity overlap integral: non-round beams, per-axis focusing, displaced and
astigmatic foci, and a rotated laser ellipse, all exactly. It is what `AnalyticalEngine`
calls, and the derivation behind it is written out in DER001.

**Quadrature cost tiers** (RES043), because §4.3 calls analytical the only real-time engine:

======================================  ==========  ===================================
tier                                    cost        what it assumes
======================================  ==========  ===================================
:func:`overlap_yield` (1D, default)     ~1-2 ms     spot sizes sampled along ``z`` only
:func:`overlap_yield` (``n_quad_u>1``)  ~40-800 ms  nothing — exact
======================================  ==========  ===================================

The default 1D quadrature is real-time at any interaction rate; the second is an opt-in exact check,
not something a caller reaches for by default (RES043).

**What this closes, precisely.** All three of RES035's growth items for the
**total yield**: non-round beams, foci displacement, and (with :func:`overlap_mean_a0_sq`
feeding :func:`estimate_spectrum_width`) the a0 the bunch actually samples. A crossing
angle is covered too (RES041), for the yield. Still open: constructing the collimated
spectrum, and the emitted *spectrum's shape* under a crossing angle, which is
an emission-kernel question rather than an overlap-geometry one.

:func:`overlap_time_profile` and :func:`overlap_transverse_profile` resolve the same
integral in time and across the transverse plane, for cheap preview plots before an
expensive run is launched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser
from ...io.units import C_CGS, SIGMA_T_CGS

__all__ = [
    "overlap_det",
    "overlap_yield",
    "overlap_mean_a0_sq",
    "overlap_time_profile",
    "overlap_transverse_profile",
    "SpectrumWidthBreakdown",
    "NONLINEAR_BROADENING_RANGE",
    "estimate_spectrum_width",
    "angle_integrated_spectrum",
]

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
    instead of approximating it (RES039).
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


def _overlap_quadratic_form(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, z, u_shift=0.0, laser_power: float = 1.0
):
    """The reduced integrand of the overlap integral at lab positions ``z``.

    Returns ``(S, det_A, sigma_ex, sigma_ey, s1, s2, h)`` where the longitudinal weight is
    ``exp(-S z^2 / 2)``, the transverse integrals have contributed ``1 / sqrt(det_A)``, and
    ``h`` is the (z-independent) time-integration coefficient the caller needs for the
    overall prefactor. See DER001 §A.6; in outline, at fixed ``t`` the
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
    pieces = _form_pieces(beam, laser, z, u_shift, laser_power)
    m = pieces.m_prime()
    det_a = m[0, 0] * m[1, 1] - m[0, 1] ** 2
    schur = m[2, 2] - (m[1, 1] * m[0, 2] ** 2 - 2.0 * m[0, 1] * m[0, 2] * m[1, 2] + m[0, 0] * m[1, 2] ** 2) / det_a
    return schur, det_a, pieces.sigma_ex, pieces.sigma_ey, pieces.s1, pieces.s2, pieces.h


@dataclass(frozen=True)
class _FormPieces:
    """The overlap integrand's quadratic form at a set of longitudinal positions.

    ``m`` is the combined exponent matrix *before* the time integration, ``g``/``h`` are
    its time couplings, and :meth:`m_prime` applies the elimination. Entries are numpy
    arrays broadcast over whatever shape ``z`` had, so one call serves a 1D grid, a 2D
    ``(z, q1)`` mesh, or a profile mesh.
    """

    m: np.ndarray  # (3, 3, ...) symmetric
    g: np.ndarray  # (3,)
    h: float
    #: Linear term and constant from a misaligned pulse. Writing the laser's contribution
    #: as a quadratic form in ``(v - D)`` with ``D = (x_off, y_off, 0, c t_off)`` gives
    #: ``E = v^T M v / 2 - v^T L + D^T M_l D / 2``, so an offset enters as exactly one
    #: linear term — no new structure, but it has to be carried through *both*
    #: completions of the square below, and a dropped piece shifts the answer rather than
    #: blowing it up.
    lin: np.ndarray  # (4, ...) conjugate to (x, y, z, ct)
    const: np.ndarray
    sigma_ex: np.ndarray
    sigma_ey: np.ndarray
    s1: np.ndarray
    s2: np.ndarray

    def m_prime(self) -> np.ndarray:
        """``M' = M - g g^T / h``: the exponent left after integrating over time."""
        outer = self.g[:, None] * self.g[None, :] / self.h
        return self.m - outer.reshape(outer.shape + (1,) * (self.m.ndim - 2))

    def after_time_integration(self):
        """``(lin', const')`` once ``ct`` is integrated out.

        Completing the square in ``w`` turns ``-w[(g.r) + L_w]`` into a shift of both the
        quadratic and the linear part: ``M -> M'`` (:meth:`m_prime`) and
        ``L_r -> L_r + g L_w / h``, with the constant picking up ``-L_w^2 / (2h)``.
        """
        return self.lin[:3] + self.g[:, None] * (self.lin[3] / self.h), self.const - self.lin[3] ** 2 / (2.0 * self.h)

    def transverse_norm(self, laser_power: float) -> np.ndarray:
        """The width normalization ``sigma_ex sigma_ey (s1 s2)^n`` that sits under the
        integrand — the laser factor enters to the same power as its density."""
        return self.sigma_ex * self.sigma_ey * (self.s1 * self.s2) ** laser_power


def _form_pieces(beam, laser, z, u_shift=0.0, laser_power: float = 1.0) -> _FormPieces:
    """Assemble the quadratic form at lab positions ``z`` (see DER001 §A.6).

    ``laser_power`` is the power the *normalized* laser density enters at: 1 for the
    luminosity itself, 2 for an ``a0^2``-weighted average (`overlap_mean_a0_sq`), since
    ``a0^2`` is exactly proportional to that density. Raising the density to a power scales
    every laser term in the exponent — including its longitudinal one, hence ``g`` and
    ``h`` — which is why one parameter covers it and no second derivation is needed.
    """
    z = np.asarray(z, dtype=float)
    k_hat, f1, f2 = laser.focusing_axes()
    beta_0 = beam.beta0()
    inv_sez2 = 1.0 / beam.m("sigma_z") ** 2
    inv_sct2 = laser_power / laser.sigma_ct() ** 2

    ex2, ey2 = _electron_sigma2(beam, z)
    s1, s2 = laser.spot_sizes(k_hat[2] * z + u_shift)
    inv_s1, inv_s2 = laser_power / s1**2, laser_power / s2**2

    shape = np.broadcast(z, ex2, s1).shape
    m = np.zeros((3, 3) + shape)
    diag_e = (1.0 / ex2, 1.0 / ey2, np.broadcast_to(inv_sez2, shape))
    for i in range(3):
        for j in range(3):
            m[i, j] = (
                f1[i] * f1[j] * inv_s1
                + f2[i] * f2[j] * inv_s2
                + k_hat[i] * k_hat[j] * inv_sct2
                + (diag_e[i] if i == j else 0.0)
            )
    g = beta_0 * np.array([0.0, 0.0, 1.0]) * inv_sez2 + k_hat * inv_sct2
    h = beta_0**2 * inv_sez2 + inv_sct2

    # Laser-only 4x4 block acting on (x, y, z, ct), needed for the offset's linear term.
    d = np.array([laser.m("x_off"), laser.m("y_off"), 0.0, C_CGS * laser.m("t_off")])
    lin = np.zeros((4,) + shape)
    const = np.zeros(shape)
    if np.any(d != 0.0):
        gl = k_hat * inv_sct2  # laser's own (r, ct) coupling
        for i in range(3):
            for j in range(3):
                ml_ij = f1[i] * f1[j] * inv_s1 + f2[i] * f2[j] * inv_s2 + k_hat[i] * k_hat[j] * inv_sct2
                lin[i] += ml_ij * d[j]
            lin[i] += -gl[i] * d[3]
        for j in range(3):
            lin[3] += -gl[j] * d[j]
        lin[3] += inv_sct2 * d[3]
        const = 0.5 * (sum(lin[i] * d[i] for i in range(4)))
    return _FormPieces(
        m=m, g=g, h=h, lin=lin, const=const,
        sigma_ex=np.sqrt(ex2), sigma_ey=np.sqrt(ey2), s1=s1, s2=s2,
    )


def _overlap_grid(beam, laser, sigma_z_eff: float, n_quad: int, laser_power: float = 1.0) -> np.ndarray:
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
    schur = _overlap_quadratic_form(beam, laser, scan, laser_power=laser_power)[0]
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


#: Longitudinal nodes evaluated per block. `_form_pieces` materializes nine arrays the
#: shape of its grid, so an unchunked ``(n_x, n_y, n_quad)`` profile mesh would allocate
#: gigabytes for an image a GUI draws in a moment. Everything that meshes ``z`` against
#: something else accumulates in blocks of this size instead — the same reason §4.2 has a
#: shared chunking utility for the per-particle kernels, applied to a much smaller problem.
_Z_BLOCK = 256


def _trapezoid_weights(z: np.ndarray) -> np.ndarray:
    """Trapezoid weights for a (possibly non-uniform) grid, so an integral can be
    accumulated blockwise as ``sum(f * w)`` instead of needing the whole array at once."""
    w = np.empty_like(z)
    w[1:-1] = 0.5 * (z[2:] - z[:-2])
    w[0] = 0.5 * (z[1] - z[0])
    w[-1] = 0.5 * (z[-1] - z[-2])
    return w


def _z_blocks(z: np.ndarray):
    """Yield ``(z_block, weight_block)`` pairs covering the grid."""
    w = _trapezoid_weights(z)
    for start in range(0, z.size, _Z_BLOCK):
        stop = start + _Z_BLOCK
        yield z[start:stop], w[start:stop]


#: Fraction of a Rayleigh range a flying-focus quadrature node may advance the spot-size
#: coordinate. The `(z, w)` Gaussian is nearly degenerate — the collision lives on a thin
#: diagonal ridge — so a grid sized from the marginals under-resolves it badly (measured:
#: 3.8% low on a short bunch). Nodes are placed on the *principal axes* and their count is
#: raised until each step moves ``u_spot`` by less than this.
_FF_NODES_PER_RAYLEIGH = 8.0
_FF_MAX_NODES = 24001


def _flying_focus_grid(beam, laser, laser_power: float, n_quad: int, n_quad_u: int):
    """Principal-axis nodes for the flying-focus ``(z, w = ct)`` quadrature.

    Returns ``(z, w, weights)``, all 2D and already carrying the area element. The
    quadratic form is evaluated once at the origin to find the principal axes; node counts
    are then raised until the spot-size coordinate ``u_spot = k_z z + beta_ff w`` advances
    by less than a fraction of a Rayleigh range per step, because that — not the Gaussian
    envelope — is what the integrand's structure actually follows.
    """
    pieces = _form_pieces(beam, laser, 0.0, laser_power=laser_power)
    m, g, h = pieces.m, pieces.g, pieces.h
    det_a = m[0, 0] * m[1, 1] - m[0, 1] ** 2

    def quad(p, q):  # p^T A^-1 q for 2-vectors
        return (m[1, 1] * p[0] * q[0] - m[0, 1] * (p[0] * q[1] + p[1] * q[0]) + m[0, 0] * p[1] * q[1]) / det_a

    b = (m[0, 2], m[1, 2])
    g_perp = (g[0], g[1])
    hess = np.array([
        [float(m[2, 2] - quad(b, b)), float(-g[2] + quad(b, g_perp))],
        [float(-g[2] + quad(b, g_perp)), float(h - quad(g_perp, g_perp))],
    ])
    evals, evecs = np.linalg.eigh(hess)
    if np.any(evals <= 0.0):
        raise ValueError("flying-focus overlap: the (z, ct) quadratic form is not positive definite")

    k_hat, _, _ = laser.focusing_axes()
    z_r = min(abs(laser.rayleigh_x()), abs(laser.rayleigh_y()))
    axes = []
    for i, requested in enumerate((n_quad, n_quad_u)):
        extent = 8.0 / math.sqrt(float(evals[i]))
        # how fast this axis moves the spot-size coordinate
        slope = abs(k_hat[2] * evecs[0, i] + laser.beta_ff * evecs[1, i])
        needed = int(2.0 * extent * slope * _FF_NODES_PER_RAYLEIGH / z_r) + 1
        n = min(max(requested, needed, 201), _FF_MAX_NODES)
        axes.append(np.linspace(-extent, extent, n))

    aa, bb = axes[0][:, None], axes[1][None, :]
    z = evecs[0, 0] * aa + evecs[0, 1] * bb
    w = evecs[1, 0] * aa + evecs[1, 1] * bb
    weights = _trapezoid_weights(axes[0])[:, None] * _trapezoid_weights(axes[1])[None, :]
    return z, w, weights


def _reduced_integral_flying_focus(beam, laser, n_quad: int, laser_power: float, n_quad_u: int):
    """``(R, h)`` when ``beta_ff != 0``, where the time integration cannot be done first.

    A flying focus makes the spot-size coordinate ``u_spot = u + beta_ff * ct`` depend on
    time, so the widths depend on **two** independent linear functionals of
    ``(x, y, z, ct)`` — ``z`` and ``u_spot`` — rather than one. Two of the four dimensions
    therefore stay Gaussian and two must be quadratured: here ``(x, y)`` are integrated
    analytically at fixed ``(z, ct)`` and the remaining plane is gridded
    (DER002).

    The result is normalized to the same convention as :func:`_reduced_integral` — the
    caller's prefactor already carries ``sqrt(2 pi / h)``, so that factor is divided out
    here rather than the prefactor being special-cased.
    """
    z, w, weights = _flying_focus_grid(beam, laser, laser_power, n_quad, n_quad_u)
    k_hat, _, _ = laser.focusing_axes()
    total = 0.0
    for start in range(0, z.shape[0], _Z_BLOCK):
        sl = slice(start, start + _Z_BLOCK)
        zb, wb = z[sl], w[sl]
        pieces = _form_pieces(beam, laser, zb, u_shift=laser.beta_ff * wb, laser_power=laser_power)
        m, g, h = pieces.m, pieces.g, pieces.h
        det_a = m[0, 0] * m[1, 1] - m[0, 1] ** 2
        lin = pieces.lin
        v0 = wb * g[0] + lin[0] - m[0, 2] * zb
        v1 = wb * g[1] + lin[1] - m[1, 2] * zb
        quad_v = (m[1, 1] * v0**2 - 2.0 * m[0, 1] * v0 * v1 + m[0, 0] * v1**2) / det_a
        exponent = (
            0.5 * quad_v - 0.5 * m[2, 2] * zb**2 + wb * g[2] * zb - 0.5 * h * wb**2
            + zb * lin[2] + wb * lin[3] - pieces.const
        )
        integrand = np.exp(exponent) / (pieces.transverse_norm(laser_power) * np.sqrt(det_a))
        total += float(np.sum(integrand * weights[sl]))

    h = _form_pieces(beam, laser, 0.0, laser_power=laser_power).h
    return 2.0 * math.pi * total / math.sqrt(2.0 * math.pi / h), h


def _z_grid(beam, laser, n_quad: int, laser_power: float = 1.0) -> np.ndarray:
    beta_0 = beam.beta0()
    D = math.hypot(beam.m("sigma_z"), beta_0 * laser.sigma_ct())
    return _overlap_grid(beam, laser, D / (1.0 + beta_0), n_quad, laser_power)


def _reduced_integral(
    beam, laser, n_quad: int, laser_power: float = 1.0, n_quad_u: int = 1
) -> tuple[float, float]:
    """``(R, h)``: the spatial integral ``Int d^3r exp(-r^T M' r / 2) / (width norm)``.

    Two evaluation modes, and they compute the *same* quantity — see :func:`overlap_yield`
    for which to use when.

    ``n_quad_u == 1`` is the 1D path: the transverse plane is integrated analytically at
    each ``z`` (giving ``2 pi / sqrt(det A)``), which requires treating the spot sizes as
    functions of ``z`` alone.

    ``n_quad_u > 1`` is the exact 2D path. The pulse's widths really depend on
    ``u = k_hat . r``, which with a crossing angle mixes ``z`` with the transverse
    coordinate along the crossing direction. Rotating the transverse plane so that
    ``q1`` lies along that direction, ``u = (k.z) z + sin(theta) q1`` depends on exactly one
    transverse coordinate, so ``q2`` can still be integrated analytically and only
    ``(z, q1)`` are quadratured. Nothing is approximated. As ``theta -> 0`` the widths stop
    depending on ``q1`` and this degenerates *gracefully* into the 1D result — wasted nodes,
    not a singularity, which is why the crossing direction is the right coordinate to keep.
    """
    if laser.beta_ff != 0.0:
        return _reduced_integral_flying_focus(beam, laser, n_quad, laser_power, n_quad_u)
    z = _z_grid(beam, laser, n_quad, laser_power)
    if n_quad_u <= 1:
        pieces = _form_pieces(beam, laser, z, laser_power=laser_power)
        m = pieces.m_prime()
        lin, const = pieces.after_time_integration()
        det_a = m[0, 0] * m[1, 1] - m[0, 1] ** 2
        p0 = lin[0] - m[0, 2] * z
        p1 = lin[1] - m[1, 2] * z
        quad_p = (m[1, 1] * p0**2 - 2.0 * m[0, 1] * p0 * p1 + m[0, 0] * p1**2) / det_a
        exponent = 0.5 * quad_p - 0.5 * m[2, 2] * z**2 + z * lin[2] - const
        integrand = np.exp(exponent) / (pieces.transverse_norm(laser_power) * np.sqrt(det_a))
        return 2.0 * math.pi * float(np.trapezoid(integrand, z)), pieces.h

    k_hat, _, _ = laser.focusing_axes()
    sin_cross = math.hypot(k_hat[0], k_hat[1])
    if sin_cross == 0.0:  # head-on: u depends on z alone, the 1D path is already exact
        return _reduced_integral(beam, laser, n_quad, laser_power, n_quad_u=1)
    cos_a, sin_a = k_hat[0] / sin_cross, k_hat[1] / sin_cross

    # Span for q1, from the *effective* curvature after q2 has been eliminated —
    # ``m11 - m12^2/m22``, not ``m11``. The two differ by however strongly the two
    # transverse directions are correlated, and using the bare ``m11`` silently truncates
    # the q1 integral whenever they are (which is exactly when the 2D mode is worth using).
    probe = _form_pieces(beam, laser, z, laser_power=laser_power).m_prime()
    p11 = cos_a**2 * probe[0, 0] + 2 * cos_a * sin_a * probe[0, 1] + sin_a**2 * probe[1, 1]
    p22 = sin_a**2 * probe[0, 0] - 2 * cos_a * sin_a * probe[0, 1] + cos_a**2 * probe[1, 1]
    p12 = -cos_a * sin_a * probe[0, 0] + (cos_a**2 - sin_a**2) * probe[0, 1] + cos_a * sin_a * probe[1, 1]
    effective = p11 - p12**2 / p22
    usable = effective[np.isfinite(effective) & (effective > 0)]
    span_q = 8.0 / math.sqrt(float(np.min(usable)))
    q1 = np.linspace(-span_q, span_q, n_quad_u)

    # h is z-independent by construction, so take it once rather than from whichever
    # block happened to run last — which would be correct only by accident.
    h_out = _form_pieces(beam, laser, 0.0, laser_power=laser_power).h
    total = 0.0
    for z_block, w_block in _z_blocks(z):
        zz, qq = z_block[:, None], q1[None, :]
        pieces = _form_pieces(beam, laser, zz, u_shift=sin_cross * qq, laser_power=laser_power)
        m = pieces.m_prime()
        m11 = cos_a**2 * m[0, 0] + 2 * cos_a * sin_a * m[0, 1] + sin_a**2 * m[1, 1]
        m22 = sin_a**2 * m[0, 0] - 2 * cos_a * sin_a * m[0, 1] + cos_a**2 * m[1, 1]
        m12 = -cos_a * sin_a * m[0, 0] + (cos_a**2 - sin_a**2) * m[0, 1] + cos_a * sin_a * m[1, 1]
        m13 = cos_a * m[0, 2] + sin_a * m[1, 2]
        m23 = -sin_a * m[0, 2] + cos_a * m[1, 2]
        exponent = -0.5 * (m11 * qq**2 + 2.0 * m13 * qq * zz + m[2, 2] * zz**2) + (
            m12 * qq + m23 * zz
        ) ** 2 / (2.0 * m22)
        integrand = np.sqrt(2.0 * math.pi / m22) * np.exp(exponent) / pieces.transverse_norm(laser_power)
        total += float(np.dot(np.trapezoid(integrand, q1, axis=1), w_block))
    return total, h_out


def overlap_yield(
    beam: GaussianElectronBeam,
    laser: GaussianParaxialLaser,
    N_e: float,
    n_quad: int = 2001,
    n_quad_u: int = 1,
) -> float:
    """Total photon yield from the **general** Gaussian luminosity overlap integral.

    Drops the round-beam and aligned-foci approximations and keeps the collision geometry exactly as
    `gammaforge.io` already describes it — per-axis bunch sizes and emittances, per-axis
    Twiss ``alpha`` (electron waist displacement), per-axis laser waists and Rayleigh
    ranges, astigmatic ``z_fx``/``z_fy`` focal offsets, the ``psi_focus`` rotation between
    the two transverse ellipses, **and a crossing angle** (``theta_xz``/``theta_yz``).

    Derivation in DER001. In outline: the yield is
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
    the round-beam closed form (DER001 §A.4). `tests/test_analytical.py` pins both reductions
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
       emitted *spectrum*: the emission kernel needs a separate analytical treatment from this overlap geometry.
       `AnalyticalEngine` says so on the `Results` when both are in play.
    """
    if n_quad < 11:
        raise ValueError(f"overlap_yield: n_quad must be >= 11 (got {n_quad!r})")

    reduced, h = _reduced_integral(beam, laser, n_quad, 1.0, n_quad_u)
    return (
        SIGMA_T_CGS
        * (1.0 + beam.beta0())
        * N_e
        * laser.n_photons()
        * math.sqrt(2.0 * math.pi / h)
        / ((2.0 * math.pi) ** 3 * beam.m("sigma_z") * laser.sigma_ct())
        * reduced
    )


def overlap_mean_a0_sq(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, n_quad: int = 2001, n_quad_u: int = 1
) -> float:
    """``<a0^2>``, averaged over the collision with the **luminosity** as its weight.

    This is the a0 the bunch actually samples, not the pulse's own maximum: electrons that
    arrive off-focus or off-peak contribute photons at a lower intensity, and this weights
    each by exactly the rate at which it scatters. It is what
    :func:`estimate_spectrum_width`'s nonlinearity term wants, and computing it closes the
    last part of the foci-displacement growth item (RES042).

    No new integral is needed. ``a0^2`` is exactly proportional to the *normalized* photon
    density (`GaussianParaxialLaser._a0_from_density` is a square root of it), so

        <a0^2> = K Int n_e p_L^2 / Int n_e p_L

    and the numerator is the same overlap integral with the laser density entering
    squared — ``laser_power = 2``, which halves every laser width and doubles its
    contribution to ``g``/``h``. Collecting the normalizations,

        <a0^2> = K sqrt(h_1 / h_2) R_2 / (R_1 (2 pi)^{3/2} sigma_lz)

    ``K`` is read off the laser itself (``a0^2 / density`` at one point) rather than
    rebuilt from constants, so this cannot drift from `io.laser`'s own energy->a0 chain.

    Always ``<= laser.a0_peak()**2``, and approaching it only for a collision that is
    pointlike compared with every focal scale.

    **This is exactly the beam-averaged ``ahat``**, and that is not a coincidence. A photon's
    formation length spans the *whole* trajectory, so the physical per-electron quantity is
    ``ahat_i = Int a0^4 dt / Int a0^2 dt`` — one scalar per electron, not a per-timestep
    intensity. Weighting those by luminosity (``L_i ~ Int a0^2 dt``) makes the denominators
    cancel:

        <ahat>_L = sum_i L_i ahat_i / sum_i L_i = Int n_e a0^4 / Int n_e a0^2

    which is what this function evaluates. No trajectory is split anywhere. Checked against
    `engines.xigma.stages.TrajectorySamples.ahat`, which averages each trajectory
    numerically: agreement is within xigma's own particle-sampling noise.

    The **spread** of ``ahat`` across the beam does *not* follow the same way — see
    :data:`NONLINEAR_BROADENING_RANGE` and RES049 for why, and for what is
    reported instead.
    """
    k_const = laser.a0_profile(0.0, 0.0, 0.0, 0.0) ** 2 / laser.photon_density(0.0, 0.0, 0.0, 0.0)
    r1, h1 = _reduced_integral(beam, laser, n_quad, 1.0, n_quad_u)
    r2, h2 = _reduced_integral(beam, laser, n_quad, 2.0, n_quad_u)
    return float(k_const * math.sqrt(h1 / h2) * r2 / (r1 * (2.0 * math.pi) ** 1.5 * laser.sigma_ct()))


def overlap_time_profile(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, N_e: float, t, n_quad: int = 401
):
    """``dN/dt`` (photons per second) at lab times ``t`` — the collision's luminosity history.

    A preview quantity: it costs one quadrature per time point and needs no macroparticles,
    so a GUI can draw the interaction as it unfolds before anything expensive is launched.

    Derivation-wise this is :func:`overlap_yield` with the time integration *not* done, so
    the exponent keeps its explicit ``t`` terms and the transverse plane is what closes
    analytically. By construction ``Int dN/dt dt == overlap_yield(...)``, which
    `tests/test_analytical.py` asserts rather than assumes.

    .. note::

       Not `io.target.OutputKind.TEMPORAL_ENVELOPE`, despite the similar name — that is a
       photon-arrival distribution filled by a full engine run. This is the luminosity rate
       of the collision, a cheap preview with no per-particle content.
    """
    t = np.asarray(t, dtype=float)
    z = _z_grid(beam, laser, n_quad)
    w = (C_CGS * t)[..., None]

    line = np.zeros(t.shape)
    for z_block, weights in _z_blocks(z):
        # A flying focus slides the spot-size evaluation point with time, and here `t` is
        # an explicit axis, so it costs nothing to carry exactly.
        pieces = _form_pieces(beam, laser, z_block, u_shift=laser.beta_ff * w)
        m, g, h = pieces.m, pieces.g, pieces.h
        det_a = m[0, 0] * m[1, 1] - m[0, 1] ** 2
        # Integrating (x, y) at fixed (z, t): the linear term is v = w g_perp - b z.
        lin = pieces.lin
        v0 = w * g[0] + lin[0] - m[0, 2] * z_block
        v1 = w * g[1] + lin[1] - m[1, 2] * z_block
        quad_v = (m[1, 1] * v0**2 - 2.0 * m[0, 1] * v0 * v1 + m[0, 0] * v1**2) / det_a
        exponent = (
            0.5 * quad_v - 0.5 * m[2, 2] * z_block**2 + w * g[2] * z_block - 0.5 * h * w**2
            + z_block * lin[2] + w * lin[3] - pieces.const
        )
        integrand = np.exp(exponent) / (pieces.transverse_norm(1.0) * np.sqrt(det_a))
        line = line + integrand @ weights

    scale = (
        SIGMA_T_CGS * (1.0 + beam.beta0()) * C_CGS * N_e * laser.n_photons()
        * 2.0 * math.pi / ((2.0 * math.pi) ** 3 * beam.m("sigma_z") * laser.sigma_ct())
    )
    return scale * line


def overlap_transverse_profile(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, N_e: float, x, y, n_quad: int = 401
):
    """``dN/dx dy`` (photons per cm^2) on the transverse plane, in the bunch's own frame.

    The other cheap preview: where in the transverse plane the photons are actually being
    produced, which is what shows a user at a glance that their pulse is missing the bunch.
    ``x``/``y`` broadcast against each other, so pass a meshgrid for an image.

    Same integral as :func:`overlap_yield` with the two transverse integrations left
    undone; time still closes analytically. ``Int dN/dx dy dx dy == overlap_yield(...)``,
    asserted in the tests.

    Transverse position is measured from the bunch centroid, which is the lab origin here —
    both distributions are centered there, so "relative to the bunch" and "lab" coincide by
    construction rather than by a shift this function applies.
    """
    if laser.beta_ff != 0.0:
        raise ValueError(
            f"overlap_transverse_profile: a flying focus (beta_ff={laser.beta_ff!r}) makes the "
            "spot size time-dependent, and this profile integrates time out — so the widths "
            "would have to be frozen at a time that no longer exists. Use "
            "overlap_time_profile, which keeps time as an explicit axis, or overlap_yield"
        )
    z = _z_grid(beam, laser, n_quad)
    xb, yb = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    xs, ys = xb[..., None], yb[..., None]

    line = np.zeros(xb.shape)
    h = _form_pieces(beam, laser, 0.0).h  # z-independent; see _reduced_integral
    for z_block, weights in _z_blocks(z):
        pieces = _form_pieces(beam, laser, z_block)
        m = pieces.m_prime()
        lin, const = pieces.after_time_integration()
        exponent = -0.5 * (
            m[0, 0] * xs**2 + m[1, 1] * ys**2 + m[2, 2] * z_block**2
            + 2.0 * (m[0, 1] * xs * ys + m[0, 2] * xs * z_block + m[1, 2] * ys * z_block)
        ) + xs * lin[0] + ys * lin[1] + z_block * lin[2] - const
        line = line + (np.exp(exponent) / pieces.transverse_norm(1.0)) @ weights

    scale = (
        SIGMA_T_CGS * (1.0 + beam.beta0()) * N_e * laser.n_photons()
        * math.sqrt(2.0 * math.pi / h)
        / ((2.0 * math.pi) ** 3 * beam.m("sigma_z") * laser.sigma_ct())
    )
    return scale * line


#: Empirical bracket on ``std(ahat) / <ahat>``, the beam-to-beam spread in nonlinear
#: red-shift as a fraction of the mean shift. **Measured, not derived** — from
#: `engines.xigma.stages.integrate_trajectories` (which averages each trajectory exactly)
#: across thirteen geometries: focus scans, displaced and astigmatic foci, crossing angles,
#: a flying focus, bunch-length and bunch-width scans, and transverse and timing offsets.
#:
#: It is wide because the factor is genuinely scenario-dependent, tracking the transverse
#: size ratio ``sigma_beam / sigma_laser`` almost monotonically: 0.06 for a loose focus
#: (nearly uniform illumination, so almost no spread), 0.39 at the baseline, 0.86 at a
#: tight focus, 1.12 for a bunch ten times wider than the spot. A narrower bracket would be
#: a nicer number and a false one.
NONLINEAR_BROADENING_RANGE = (0.06, 1.12)


@dataclass(frozen=True)
class SpectrumWidthBreakdown:
    """The collimated-spectrum FWHM estimate (units of the Compton edge, dimensionless),
    as four independently-reported components (§4.3: "GUI shows a component table; total
    in quadrature") rather than a single summed float.

    Each field already carries the ``0.5 * 2.355`` FWHM-from-sigma prefactor
    applied to its own term, so :attr:`total` — `math.hypot` of the four fields — squares
    and re-sums them, exactly reproducing the original single
    ``0.5 * 2.355 * sqrt(term1 + term2 + term3 + term4)`` formula. The leading ``0.5`` is
    carried unexplained, as it was in the ported source — not retrofitted with a
    justification the original never had.
    """

    collimation: float  #: from angular collimation, ``(gamma * theta_col)^2``
    emittance: float  #: from angular divergence, ``(gamma * sqrt(div_x * div_y))^2``
    energy_spread: float  #: from beam energy spread, ``sigma_gamma / gamma``
    nonlinearity: float  #: ponderomotive, using the historical implicit ``std = mean``
    #: The nonlinear term is the one quantity here that cannot be pinned exactly — it is
    #: set by the *spread* of ``ahat`` across the beam, not analytically available (RES049),
    #: and is bracketed instead by :data:`NONLINEAR_BROADENING_RANGE`. ``nonlinearity``
    #: above sits at a factor of 1 — near the top of that bracket, so it over-estimates
    #: broadening for most geometries.
    nonlinearity_lo: float = 0.0
    nonlinearity_hi: float = 0.0

    @property
    def total(self) -> float:
        return math.hypot(self.collimation, self.emittance, self.energy_spread, self.nonlinearity)

    @property
    def total_range(self) -> tuple[float, float]:
        """``(lo, hi)`` on the total width, from the nonlinear bracket.

        Often the bracket barely matters — when beam quality dominates, the collimation,
        emittance and energy-spread terms swamp the nonlinear one and ``lo`` and ``hi``
        nearly coincide. That is worth reading off directly rather than assuming.
        """
        others = (self.collimation, self.emittance, self.energy_spread)
        return math.hypot(*others, self.nonlinearity_lo), math.hypot(*others, self.nonlinearity_hi)


def estimate_spectrum_width(
    beam: GaussianElectronBeam,
    laser: GaussianParaxialLaser,
    theta_col: float,
    a0_sq: float | None = None,
) -> SpectrumWidthBreakdown:
    """Collimated-spectrum FWHM estimate, broken into its four components (§4.3).

    ``theta_col``: collimation half-angle (rad) — a single scalar; a caller combining
    `gammaforge.io.target.Target`'s separate ``theta_x_col``/``theta_y_col`` should use
    their geometric mean (RES038), the same x/y-combining convention this
    module already uses for the laser waist (``sigma_lr0``) and the emittance term below.

    ``laser`` is the fitted `GaussianParaxialLaser` (see `gammaforge.io.laser.fit_gaussian_paraxial`);
    ``laser.a0_peak()`` is the pulse's own maximum a0, not the a0 at the electron bunch's
    actual position.

    ``a0_sq`` is the mean square a0 the bunch actually samples. Pass
    :func:`overlap_mean_a0_sq` (RES042), the luminosity-weighted average, rather than the
    pulse's own peak. It **defaults to** ``laser.a0_peak()**2`` deliberately: this
    function also preserves the established peak-a0 estimate, while `AnalyticalEngine`
    passes the overlap-weighted value explicitly.
    """
    gamma0 = beam.gamma0()
    sigma_gamma = beam.sigma_gamma()
    emit_width = math.sqrt(beam.divergence_x() * beam.divergence_y())
    mean_a0_sq = laser.a0_peak() ** 2 if a0_sq is None else a0_sq
    prefactor = 0.5 * 2.355
    # <ahat>: the cycle-averaged intensity (see the note below). Asked of the laser rather
    # than written as 0.5, so this and xigma share one definition of C (RES053/RES054).
    mean_shift = laser.cycle_average_factor() * mean_a0_sq
    lo, hi = NONLINEAR_BROADENING_RANGE
    return SpectrumWidthBreakdown(
        collimation=prefactor * (gamma0 * theta_col) ** 2,
        emittance=prefactor * (gamma0 * emit_width) ** 2,
        energy_spread=prefactor * (sigma_gamma / gamma0),
        nonlinearity=prefactor * mean_shift,
        nonlinearity_lo=prefactor * lo * mean_shift,
        nonlinearity_hi=prefactor * hi * mean_shift,
    )


def angle_integrated_spectrum(
    gamma0: float, sigma_gamma: float, N_e: float, s, n_quad: int = 401, ahat: float = 0.0
):
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
    spectrum by construction against the overlap yield.

    ``n_quad``: quadrature points spanning +-6 ``sigma_gamma`` around ``gamma0`` —
    independent of ``n_particles``, so a generous default costs nothing.

    ``ahat`` is the cycle-averaged normalized intensity, which red-shifts the Compton edge:
    the resonance sits at ``s_res = gamma^2 / (1 + ahat)`` rather than ``gamma^2``, so the
    kinematic variable becomes ``y = s (1 + ahat) / gamma^2`` and the whole spectrum
    compresses towards lower energy. ``ahat = 0`` recovers the linear edge exactly.

    Note ``ahat`` is the **cycle-averaged** intensity, `` <a^2> = C a0^2 `` with ``C = 1/2``
    for linear polarization — *not* ``a0^2`` itself. `a0` is the normalized peak field
    magnitude, so the factor is the cycle average of ``cos^2``; for circular polarization
    the magnitude is constant and ``C = 1``, which is the same factor by which circular
    carries twice the energy density at fixed ``a0``. Passing ``a0^2`` here instead of
    ``<a^2>`` doubles the red-shift.

    Only the *mean* ``ahat`` is applied. Electrons sample different intensities, so the edge
    is also smeared; that spread is not analytically available and is bracketed instead
    (:data:`NONLINEAR_BROADENING_RANGE`, RES049).

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

    if ahat < 0.0:
        raise ValueError(f"angle_integrated_spectrum: ahat must be >= 0 (got {ahat!r})")
    gamma2 = (gamma_grid**2)[:, None]
    y = s_arr[None, :] * (1.0 + ahat) / gamma2
    shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    out = np.sum(quad_weight[:, None] * shape / gamma2, axis=0)

    return out if np.ndim(s) else out[0]

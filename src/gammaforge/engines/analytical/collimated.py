"""Deterministic collimated spectra and the raw DER017 moment channels (DER019 §7, §7.1, §8).

`OutputKind.COLLIMATED_SPECTRUM` is the 3D ``(E, theta_x, theta_y)`` slice — the sole
energy-angle output kind (RES052 deliberately removed the duplicate). This module fills it
without macroparticles, and returns the three raw channels separately rather than only the
reconstructed spectrum, because a reconstructed curve that looks plausible can hide any one
of the three being wrong.

**The delta-resonance inversion.** Integrating over photon energy removes the resonance delta
function, which lets ``gamma`` be inverted *analytically* at each requested ``(s, n)``
(DER019 §7):

    A_R = 1 + Q ahat,   K = D Cbar,   r^2 = (theta_ex - theta_x)^2 + (theta_ey - theta_y)^2
    Gamma^2 = A_R / (K/s - r^2),      support K/s > r^2

so there is **no gamma quadrature at all** — the electron's energy PDF is simply evaluated
at ``Gamma``. That is what makes this tier cheap enough for optimization, and it is why
DER019 prefers it to building a Stage-1 table at all.

**What ``rho0`` is, and is not.** The zeroth channel is a *density in s*, not a
probability distribution on a fixed interval: for a zero-emittance on-axis observer it is
``s^{-7/2}``, which piles up at low energy without bound on ``(0, 1]``. That is correct
physics, not a defect — the finite-normalization quantity is the angle-integrated shape
`nonlinear_spectrum.nonlinear_shape`, and RES036's yield normalization still applies to the
slice as a whole. What the tests pin instead is the *scaling law* and the moment
identities, which are the parts a wrong formula would break.

**Independence.** No gammaforge.xigma import appears anywhere in this module, and none
should: analytical is Xigma's validation oracle (RES095 leaves it one of the remaining
independent legs), so importing the kernel or the reconstruction operator from the
implementation it is meant to check would make the comparison circular. The DER017
reconstruction is therefore re-derived here from its defining operator rather than called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from .fixed_width import KAPPA_G

__all__ = [
    "CollimatedMoments",
    "collimated_moments",
    "reconstruct_second_order",
]

#: `3/(4 pi)`: the DER017 kernel normalization (DER019 §7). Matches
#: `xigma.stages.KERNEL_NORMALIZATION_CONSTANT` by construction rather than by import —
#: see the module docstring on why these two must stay independent.
KERNEL_NORMALIZATION = 3.0 / (4.0 * math.pi)


@dataclass(frozen=True)
class CollimatedMoments:
    """The three raw DER017 channels on a normalized-energy grid, plus the resonance root.

    ``rho1``/``rho2`` are the *unreconstructed* channels. Reporting them alongside
    ``reconstructed`` is the point: DER019 §18.12 asks for ``rho0``, ``rho1`` and ``rho2`` to
    be compared with Xigma separately, since a matching reconstructed spectrum can be
    reached with compensating errors in two of the three.
    """

    s: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray
    #: ``Gamma`` at the resonance root for each ``s``; exposed because the support condition
    #: ``K/s > r^2`` is the model's own validity edge and is useful when diagnosing an
    #: empty spectrum.
    gamma: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        """The second-order finite-line spectrum ``S(s)`` (DER017, via DER019 §7.1).

        ``S = rho0 - (1/s) d[s rho1]/ds + (1/(2s)) d^2[s rho2]/ds^2``

        Differentiated here rather than by calling `xigma.stages.reconstruct_second_order`,
        for the independence reason in the module docstring.
        """
        return reconstruct_second_order(self.s, self.rho0, self.rho1, self.rho2)


def collimated_moments(
    beam: GaussianElectronBeam,
    s,
    theta_ex: float = 0.0,
    theta_ey: float = 0.0,
    ahat: float = 0.0,
    d_factor: float = 1.0,
    q_factor: float = 1.0,
    c_bar: float = 1.0,
    var_q: float | None = None,
    total_yield: float = 1.0,
) -> CollimatedMoments:
    """Raw ``rho0``/``rho1``/``rho2`` for the round head-on zero-emittance tier.

    Implements DER019 §7's master formula with the round head-on on-axis specialization of
    §8, where ``D = Q = 1``, ``r = 0`` and ``Cbar = 1`` for an unchirped pulse:

        A_R = 1 + ahat,  K = Cbar
        Gamma^2 = A_R / (K/s)
        rho0 = Y (3/(4 pi)) (K/s^2) P Gamma^5 / (A_R (1 + Gamma^2 r^2)^2)

    ``P`` is 1 for the azimuthally averaged on-axis kernel, and ``d_factor``/``q_factor``
    carry the direction factors ``D`` and ``Q`` so a caller can at least see the
    dependence; they are 1 for the tier this actually claims.

    ``var_q`` is the within-trajectory DER016 variance of the nonlinear coordinate. It
    defaults to ``KAPPA_G * ahat^2``, the unchirped separable value from DER019 §3.1 — the
    finite-line width of a single trajectory, distinct from the between-trajectory spread in
    `fixed_width`. Passing ``0.0`` reproduces the delta-line limit, in which the three
    channels collapse and the reconstruction returns ``rho0`` unchanged; the tests use that
    to pin the operator independently of the physics.
    """
    if q_factor <= 0.0:
        # DER019 §8.1: Q = 0 means observation along the laser's propagation direction, where
        # the nonlinear coordinate does not shift the resonance. That is a delta-line limit,
        # not something this formula can express.
        raise ValueError(
            "collimated_moments: q_factor = 0 (observation along the laser propagation "
            "direction) does not shift the resonance and is a delta-line limit, not a value "
            "the resonance-inversion formula accepts (DER019 §8.1)"
        )
    if c_bar <= 0.0:
        raise ValueError(f"collimated_moments needs c_bar > 0, got {c_bar!r}")
    if ahat < 0.0:
        raise ValueError(f"collimated_moments needs ahat >= 0, got {ahat!r}")

    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    a_r = 1.0 + q_factor * ahat
    k = d_factor * c_bar

    # r^2 = (theta_ex - theta_x)^2 + (theta_ey - theta_y)^2, zero on axis for this tier.
    r_sq = (theta_ex - 0.0) ** 2 + (theta_ey - 0.0) ** 2

    # Resonance inversion with its own support condition K/s > r^2 (DER019 §7). `s <= 0` is
    # outside the support too — a normalized photon energy is positive by definition — so it
    # is masked here rather than left to produce a divide-by-zero and then a nan that would
    # spread through the moment channels and the reconstruction.
    supported = (s_arr > 0.0) & (k / np.where(s_arr > 0.0, s_arr, 1.0) - r_sq > 0.0)
    safe_s = np.where(supported, s_arr, 1.0)
    inverse_base = k / safe_s - r_sq
    gamma_sq = np.where(supported, a_r / np.where(supported, inverse_base, 1.0), 0.0)
    gamma = np.sqrt(gamma_sq)

    # rho0: the master formula, with P = 1 for the azimuthally averaged on-axis kernel.
    per_s = (
        KERNEL_NORMALIZATION
        * (k / safe_s**2)
        * gamma**5
        / (a_r * (1.0 + gamma_sq * r_sq) ** 2)
    )
    rho0 = np.where(supported, total_yield * per_s, 0.0)

    # The moment channels, at the same root (DER019 §7.1). Unchirped means Cbar = 1,
    # Var(C) = 0 and Cov(q, C) = 0, which reduces both to the single Var(q) term.
    if var_q is None:
        var_q = KAPPA_G * ahat**2
    b_factor = 1.0 + q_factor * ahat + gamma_sq * r_sq
    safe_b = np.where(supported, b_factor, 1.0)
    delta_c = np.where(supported, q_factor**2 * var_q / safe_b**2, 0.0)
    m2 = delta_c.copy()

    return CollimatedMoments(
        s=s_arr,
        rho0=rho0,
        rho1=np.where(supported, rho0 * safe_s * delta_c, 0.0),
        rho2=np.where(supported, rho0 * safe_s**2 * m2, 0.0),
        gamma=gamma,
    )


def _finite_difference(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """``order``-th derivative of ``y`` sampled on a possibly nonuniform grid ``x``.

    Uses `numpy.gradient`, which handles nonuniform spacing and picks the stencil order from
    the number of points. A hand-rolled local polynomial fit was tried first and rejected: the
    monomial Vandermonde system it solves is ill-conditioned enough to return ~1e8 relative
    error on smooth power laws, which then dominated the reconstructed spectrum entirely.
    `numpy.gradient` is exact to roundoff for low degrees and degrades gracefully beyond.

    This is deliberately *not* `xigma.stages.nonuniform_derivative` — the two must be
    separate implementations for the comparison to be a check rather than a tautology.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = y
    for _ in range(order):
        result = np.gradient(result, x, edge_order=2)
    return result


def reconstruct_second_order(
    s: np.ndarray, rho0: np.ndarray, rho1: np.ndarray, rho2: np.ndarray
) -> np.ndarray:
    """The DER017 second-order finite-line reconstruction, re-derived (DER019 §7.1).

        S(s) = rho0 - (1/s) d[s rho1]/ds + (1/(2s)) d^2[s rho2]/ds^2

    Implemented from the defining operator rather than imported from
    `xigma.stages.reconstruct_second_order`, so that comparing this engine's output with
    Xigma's compares two derivations rather than one derivation called twice.
    """
    s = np.asarray(s, dtype=float)
    if s.size < 3:
        raise ValueError("second-order reconstruction requires at least three spectral points")
    if np.any(s <= 0.0):
        raise ValueError("reconstruct_second_order requires positive spectral points")
    first = _finite_difference(s, s * np.asarray(rho1), 1)
    second = _finite_difference(s, s * np.asarray(rho2), 2)
    return np.asarray(rho0) - first / s + 0.5 * second / s

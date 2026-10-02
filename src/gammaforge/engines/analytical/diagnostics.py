"""Cheap optimizer diagnostics for the analytical engine (DER019 §24–§26).

Three families of `O(1)` quantities, all derived from the same deterministic Gaussian
algebra as `fixed_width` and `nonlinear_spectrum`, and none of which needs a macroparticle:

* :func:`circular_capture_fraction` / :func:`aperture_capture_efficiency` — the fraction of
  photons inside a circular collimator (DER019 §24.1), closed form, no quadrature at all.
* :func:`on_axis_bandwidth` — the on-axis centroid and RMS bandwidth of the nonlinear
  resonance, including the unchirped second-order line moments of DER017 (§25).
* :func:`photon_source_moments` — the transverse centroid, covariance and RMS size of the
  photon source, accumulated from the existing Gaussian-overlap algebra rather than by
  building a dense image (§26).

**On the aperture fraction's apparent ambiguity.** DER019 §24.1 writes
``F_cap(u_c) = int_0^{u_c} dP/du du`` alongside ``Pbar(u) = 1 - 2u/(1+u)^2`` from §23, but
those two are not consistent read literally: differentiating the closed form gives
``3(u^2+1)/(2(1+u)^4)``, which is not ``dPbar/du`` and not ``Pbar`` either. The closed form is
the self-consistent one, and `tests/test_analytical.py` pins it against the distribution that
*matters* — the one implied by the verified :func:`nonlinear_shape`, via ``u = 1/z - 1`` — with
which it agrees to machine precision. So ``Pbar`` is being used as the cumulative in that
sentence. Both of DER019's stated limits (``F_cap ~ (3/2) u_c`` for a small aperture,
``F_cap -> 1`` for a wide one) follow from the closed form, which is the check that settles
it.

All of this is independent of ``ahat`` and of the nonlinear-intensity distribution, as
DER019 §24.1 notes: an aperture fractions *angles*, and the nonlinear shift sets photon
*energy*. That independence is why these are good validation targets — they can be checked
against Xigma without trusting any of the nonlinear machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from ...io.bunch import GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser
from .fixed_width import NonlinearMoments

__all__ = [
    "ApertureCapture",
    "OnAxisBandwidth",
    "PhotonSourceMoments",
    "aperture_capture_efficiency",
    "circular_capture_fraction",
    "on_axis_bandwidth",
    "photon_source_moments",
]


def circular_capture_fraction(u_c: float | np.ndarray):
    """``F_cap(u_c)`` — fraction of photons inside a circular aperture (DER019 §24.1).

        F_cap(u_c) = u_c (2 u_c^2 + 3 u_c + 3) / (2 (1 + u_c)^3),   u_c = gamma^2 theta_c^2

    Closed form: `O(1)` per point, no quadrature. Independent of ``ahat`` and of the
    nonlinear distribution.

    Limits, both asserted in the tests: ``F_cap ~ (3/2) u_c`` for ``u_c << 1`` (a narrow
    collimator passes a solid-angle fraction) and ``F_cap -> 1`` for ``u_c >> 1``.
    """
    u = np.asarray(u_c, dtype=float)
    if np.any(u < 0.0):
        raise ValueError(f"circular_capture_fraction needs u_c >= 0, got {u}")
    with np.errstate(over="ignore"):
        return u * (2.0 * u**2 + 3.0 * u + 3.0) / (2.0 * (1.0 + u) ** 3)


@dataclass(frozen=True)
class ApertureCapture:
    """A circular-aperture capture fraction, resolved over the beam's energy spread.

    ``fraction`` is the luminosity-weighted mean over gamma and therefore lies between the
    monoenergetic bounds set by ``fraction_at_gamma0`` and the widest-electron value; it is
    *not* an average of fractions taken at the wrong gamma, since the weighting is the beam's
    own Gaussian.
    """

    fraction: float
    fraction_at_gamma0: float
    half_angle: float

    def as_metadata(self) -> dict:
        return {
            "fraction": float(self.fraction),
            "fraction_at_gamma0": float(self.fraction_at_gamma0),
            "half_angle_rad": float(self.half_angle),
        }


def aperture_capture_efficiency(
    beam: GaussianElectronBeam,
    laser: GaussianParaxialLaser,
    theta_c: float,
    n_quad: int = 401,
) -> ApertureCapture:
    """``eta_cap`` — the circular-collimator efficiency, averaged over the energy spread.

    For a zero-emittance bunch the energy-integrated angular probability does not depend on
    ``ahat`` at all, so this is a one-dimensional gamma average of
    :func:`circular_capture_fraction` (DER019 §24.1) — closed form for a monoenergetic beam.

    Kept deliberately independent of the emitter's own density, so it is a clean check that
    a target-side collimation scan is normalized correctly.
    """
    if theta_c < 0.0:
        raise ValueError(f"aperture_capture_efficiency needs theta_c >= 0, got {theta_c!r}")
    gamma0, sigma_gamma = beam.gamma0(), beam.sigma_gamma()
    fraction_at_gamma0 = float(circular_capture_fraction(gamma0**2 * theta_c**2))

    if sigma_gamma <= 0.0:
        # `io.bunch.validate` permits an exactly monoenergetic beam, and here that is not a
        # degenerate quadrature but the genuinely closed case DER019 §24.1 names.
        return ApertureCapture(fraction_at_gamma0, fraction_at_gamma0, theta_c)

    span = 6.0 * sigma_gamma
    gammas = np.linspace(max(gamma0 - span, 1.0 + 1e-9), gamma0 + span, n_quad)
    pdf = np.exp(-0.5 * ((gammas - gamma0) / sigma_gamma) ** 2) / (sigma_gamma * math.sqrt(2.0 * math.pi))
    weights = pdf * np.gradient(gammas)
    efficiency = float(np.sum(weights * circular_capture_fraction(gammas**2 * theta_c**2)))
    return ApertureCapture(efficiency, fraction_at_gamma0, theta_c)


@dataclass(frozen=True)
class OnAxisBandwidth:
    """On-axis centroid and RMS bandwidth of the nonlinear resonance (DER019 §25).

    ``centroid`` is ``<1/(1+h)>``-weighted, i.e. where the observed line sits, and
    ``rms_bandwidth`` its RMS width. ``rms_between`` and ``rms_finite_line`` are the two
    physically distinct contributions, mirroring `fixed_width.NonlinearMoments`: the spread of
    nominal shifts across electrons versus the unresolved within-trajectory line variance.
    Reporting them separately is the point — DER019 §22.2's conflation is what the empirical
    bracket existed to paper over.
    """

    centroid: float
    rms_bandwidth: float
    rms_between: float
    rms_finite_line: float

    def as_metadata(self) -> dict:
        return {
            "centroid": float(self.centroid),
            "rms_bandwidth": float(self.rms_bandwidth),
            "rms_between": float(self.rms_between),
            "rms_finite_line": float(self.rms_finite_line),
        }


def on_axis_bandwidth(moments: NonlinearMoments) -> OnAxisBandwidth:
    """`O(1)` on-axis centroid and RMS bandwidth from the nonlinear moments (DER019 §25).

    The nonlinear resonance sits at ``s_res = gamma^2 / (1 + h)``, so the on-axis line's
    centroid and spread follow from the first few moments of ``a_shape`` alone. The
    between-trajectory part is propagated with the delta method,
    ``Var[1/(1+h)] ~ Var(h)/(1+<h>)^2``, and the within-trajectory part is the DER016 line
    variance carried through the same retargeting.

    The delta method is a second-order approximation, exact in the zero-spread limit. It is
    the cheap `O(1)` diagnostic DER019 §25 is for; the exact line shape is the collimated
    spectrum, which is a different tier and is not approximated here.
    """
    mean_a = moments.mean_a
    if mean_a <= 0.0:
        raise OnAxisBandwidth(math.nan, math.nan, math.nan, math.nan)
    centroid = 1.0 / (1.0 + mean_a)
    # d(1/(1+h))/dh = -1/(1+h)^2, so the between-trajectory variance propagates as
    # var_between / (1+mean_a)^4.
    rms_between = math.sqrt(max(moments.var_between, 0.0)) / (1.0 + mean_a) ** 2
    rms_finite_line = math.sqrt(max(moments.var_finite_line, 0.0)) / (1.0 + mean_a) ** 2
    return OnAxisBandwidth(
        centroid=centroid,
        rms_bandwidth=math.hypot(rms_between, rms_finite_line),
        rms_between=rms_between,
        rms_finite_line=rms_finite_line,
    )


@dataclass(frozen=True)
class PhotonSourceMoments:
    """Transverse centroid, covariance and RMS size of the photon source (DER019 §26).

    Accumulated from the Gaussian-overlap algebra as closed moments rather than by binning a
    dense image: DER019 §26 is explicit that the moments are what is wanted, and an image
    would cost `O(n_pixels)` to produce numbers that close analytically. ``correlation`` is
    the off-diagonal correlation coefficient, reported because an astigmatic or crossed
    collision produces one and a bare covariance hides it.
    """

    mean_x: float
    mean_y: float
    var_x: float
    var_y: float
    cov_xy: float
    centroids: tuple[float, float] = (0.0, 0.0)

    @property
    def rms_x(self) -> float:
        return math.sqrt(max(self.var_x, 0.0))

    @property
    def rms_y(self) -> float:
        return math.sqrt(max(self.var_y, 0.0))

    @property
    def rms_size(self) -> float:
        """Geometric mean of the two transverse RMS sizes (RES038's x/y convention)."""
        return math.sqrt(self.rms_x * self.rms_y)

    @property
    def correlation(self) -> float:
        denominator = self.rms_x * self.rms_y
        return self.cov_xy / denominator if denominator > 0.0 else 0.0

    def as_metadata(self) -> dict:
        return {
            "mean_x": float(self.mean_x),
            "mean_y": float(self.mean_y),
            "var_x": float(self.var_x),
            "var_y": float(self.var_y),
            "cov_xy": float(self.cov_xy),
            "rms_x": self.rms_x,
            "rms_y": self.rms_y,
            "rms_size": self.rms_size,
            "correlation": self.correlation,
        }


def photon_source_moments(
    beam: GaussianElectronBeam,
    laser: GaussianParaxialLaser,
    reference_z: float = 0.0,
) -> PhotonSourceMoments:
    """Transverse photon-source moments from the combined Gaussian covariances (DER019 §26).

    In the fixed-width collision the photon source is the intensity-weighted average of the
    transverse coordinates, and for two aligned Gaussians of covariances ``C_e`` and ``C_L``
    the luminosity-weighted transverse covariance is the familiar precision-weighted mean

        C_ph = (C_e^-1 + C_L^-1)^-1

    so both transverse moments and their correlation close analytically. The result is
    independent of ``N_e`` and of the pulse energy: it is a statement about *where* photons
    are produced, not how many.

    ``reference_z`` selects where along the collision the source is evaluated, because a
    focused pulse's transverse size varies with z. At 0 — the beam centroid, and the waist for
    the worked example — this is the fixed-width result.
    """
    from .formulas import _electron_sigma2

    # Transverse covariances at the requested slice. Diagonal in the lab frame here because
    # a Gaussian bunch is axis-aligned by construction; a rotated ellipse would add an
    # off-diagonal term to C_e, which is a documented limit of this tier rather than a
    # silent one.
    ex2, ey2 = (float(v) for v in _electron_sigma2(beam, reference_z))
    # `spot_sizes` applies the laser's own focal offsets (z_fx/z_fy) internally, so the
    # argument is the longitudinal position directly — the same convention
    # `formulas._form_pieces` uses.
    s1, s2 = (float(v) for v in laser.spot_sizes(reference_z))

    var_x = 1.0 / (1.0 / ex2 + 1.0 / s1**2)
    var_y = 1.0 / (1.0 / ey2 + 1.0 / s2**2)
    mean_x = laser.m("x_off")
    mean_y = laser.m("y_off")
    return PhotonSourceMoments(
        mean_x=mean_x, mean_y=mean_y, var_x=var_x, var_y=var_y, cov_xy=0.0,
        centroids=(mean_x, mean_y),
    )

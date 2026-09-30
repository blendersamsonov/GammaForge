"""DER023 conservative Gaussian trajectory geometry and luminosity bounds."""

from __future__ import annotations

import math

import numpy as np

from ...io.bunch import Bunch
from ...io.laser import GaussianParaxialLaser, PulseTrainParaxialLaser
from ...io.units import C_CGS


def _temporal_bins(laser):
    envelope = laser.temporal_envelope
    if type(laser) is GaussianParaxialLaser:
        sigma_t, delays = envelope.m("duration"), np.array([0.0])
    else:
        sigma_t, delays = envelope.m("subpulse_duration"), envelope.subpulse_delays()
    standardized = np.r_[-np.inf, np.linspace(-8.0, 8.0, 129), np.inf]
    cdf = np.array([0.0 if x == -np.inf else 1.0 if x == np.inf else
                    0.5 * math.erfc(-x / math.sqrt(2.0)) for x in standardized])
    masses = np.diff(cdf) / len(delays)
    return [(delay + sigma_t * standardized, masses) for delay in delays]


def trajectory_bound_geometry(bunch: Bunch, laser: GaussianParaxialLaser | PulseTrainParaxialLaser):
    """Return ``(F, d², eta_min, B)`` in the circular focal metric of DER023."""
    sigma = laser.m("sigma_x")
    z_r = laser.rayleigh_x()
    if (sigma != laser.m("sigma_y") or laser.m("z_fx") != laser.m("z_fy")
            or laser.beta_ff != 0.0 or z_r * z_r <= 2.0 * sigma * sigma):
        raise ValueError("DER023 bound requires circular coincident-focus paraxial geometry")
    k, f1, f2 = laser.focusing_axes()
    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    e = np.stack((bunch.thx / norm, bunch.thy / norm, 1.0 / norm), axis=1)
    f = 1.0 - e @ k
    if np.any(f <= 64.0 * np.finfo(float).eps):
        raise ValueError("co-propagation is outside the DER023 bound")
    r0 = np.stack((bunch.x - laser.m("x_off"),
                   bunch.y - laser.m("y_off"), bunch.z), axis=1)
    u0 = r0 @ k
    t_eta0 = (laser.m("t_off") + u0 / C_CGS) / f
    a = r0 + C_CGS * e * t_eta0[:, None] - laser.m("z_fx") * k
    b = C_CGS * e / f[:, None]

    def metric(v, w):
        return ((v @ f1) * (w @ f1) + (v @ f2) * (w @ f2)) / (2.0 * sigma**2) + (v @ k) * (w @ k) / z_r**2

    aa, ab, bb = metric(a, a), metric(a, b), metric(b, b)
    eta_min = -ab / bb
    d2 = np.maximum(aa - ab**2 / bb, 0.0)
    return f, d2, eta_min, np.sqrt(bb)


def luminosity_upper_bound(bunch: Bunch, laser: GaussianParaxialLaser | PulseTrainParaxialLaser):
    """Bound ``integral T(eta) S_i(eta) d eta`` without particle trajectory sampling.

    A fixed geometric D bank rounds ``1+d²`` downward. Gaussian bin masses are
    prepared once; the maximum Lorentzian kernel in each bin gives an upper sum.
    """
    _, d2, eta_min, _ = trajectory_bound_geometry(bunch, laser)
    if not len(d2):
        return np.zeros(0)
    d_bank = np.exp2(np.arange(1024, dtype=float))
    indices = np.minimum(np.searchsorted(d_bank, 1.0 + d2, side="right") - 1, 1023)
    d_low = d_bank[indices]
    b0 = C_CGS / (2.0 * laser.rayleigh_x())
    upper = np.zeros_like(d2)
    for edges, masses in _temporal_bins(laser):
        for first in range(0, len(d2), 4096):
            last = min(first + 4096, len(d2))
            distance = np.maximum(np.maximum(edges[:-1][None, :] - eta_min[first:last, None],
                                             eta_min[first:last, None] - edges[1:][None, :]), 0.0)
            upper[first:last] += np.sum(masses[None, :] /
                                         (d_low[first:last, None] + b0**2 * distance**2), axis=1)
    return upper * (1.0 + 1e-12)


def luminosity_lower_bound(bunch: Bunch, laser: GaussianParaxialLaser | PulseTrainParaxialLaser):
    """Lower-bound ``integral T(eta) S_i(eta) d eta`` by finite-bin minima.

    The omitted infinite tails contribute nonnegative luminosity. This bound is
    conservative for the unchirped built-in Gaussian family in DER023's regime.
    """
    f, _, _, _ = trajectory_bound_geometry(bunch, laser)
    k, f1, f2 = laser.focusing_axes()
    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    e = np.stack((bunch.thx / norm, bunch.thy / norm, 1.0 / norm), axis=1)
    r0 = np.stack((bunch.x - laser.m("x_off"),
                   bunch.y - laser.m("y_off"), bunch.z), axis=1)
    t_eta0 = (laser.m("t_off") + (r0 @ k) / C_CGS) / f
    a = r0 + C_CGS * e * t_eta0[:, None] - laser.m("z_fx") * k
    b = C_CGS * e / f[:, None]
    sigma, z_r = laser.m("sigma_x"), laser.rayleigh_x()
    q0, q1 = (a @ k) / z_r, (b @ k) / z_r
    p10, p11 = (a @ f1) / (math.sqrt(2) * sigma), (b @ f1) / (math.sqrt(2) * sigma)
    p20, p21 = (a @ f2) / (math.sqrt(2) * sigma), (b @ f2) / (math.sqrt(2) * sigma)
    lower = np.zeros(bunch.n_particles)
    for edges, masses in _temporal_bins(laser):
        lo, hi = edges[1:-2], edges[2:-1]
        for first in range(0, bunch.n_particles, 4096):
            sl = slice(first, min(first + 4096, bunch.n_particles))
            qlo, qhi = q0[sl, None] + q1[sl, None] * lo, q0[sl, None] + q1[sl, None] * hi
            qmax2 = np.maximum(qlo**2, qhi**2)
            qmin2 = np.where(qlo * qhi <= 0.0, 0.0, np.minimum(qlo**2, qhi**2))
            p_lo = (p10[sl, None] + p11[sl, None] * lo)**2 + (p20[sl, None] + p21[sl, None] * lo)**2
            p_hi = (p10[sl, None] + p11[sl, None] * hi)**2 + (p20[sl, None] + p21[sl, None] * hi)**2
            spatial_floor = np.exp(-np.maximum(p_lo, p_hi) / (1.0 + qmin2)) / (1.0 + qmax2)
            lower[sl] += np.sum(spatial_floor * masses[1:-1], axis=1)
    return lower * (1.0 - 1e-12)

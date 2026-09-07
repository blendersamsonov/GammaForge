"""xigma's pipeline as composable pure functions (GRAND_PLAN.md §4.2).

Stage 0 (:func:`integrate_trajectories`), Stage 1 (:func:`deposit_shape_table`, onto the
a0-independent :attr:`TrajectorySamples.a0_shape` axis), the retarget step
(:func:`retarget_ahat`, a conservative regrid onto the physical, non-uniform ``ahat`` axis
for one specific peak a0 — RES032) and Stage 2 (:func:`spectrum_from_table`,
:func:`angular_spectrum_from_table`, :func:`spectrum_in_angular_range`) all live here.

**Every stage is a pure function.** State lives in the `Collision` facade (Phase 3a), not
here, so validation can call these directly and a stage can be reasoned about without
knowing what cached it.

**The engine calls the laser; it does not model it** (§4.2, P15). Stage 0 samples
``intensity_profile`` — the cycle-averaged ``<a^2>`` — along each trajectory through the
`LaserField` protocol and needs nothing else from it: no envelope formula, no Gaussian
assumption, no spot sizes. The photon density follows by inverting the same
energy→intensity chain the laser used to produce it (:func:`photon_density_scale`), so a
future non-Gaussian `LaserField` drops in with no change here.

**Nothing in this module carries a polarization convention** (RES054).
``<a^2>`` at fixed pulse energy is the same whether the pulse is linear or circular, and
every quantity xigma's Stage 0/1 produce — yield, ``a0_shape``, ``ahat`` — is a functional
of it. Ellipticity enters exactly once, in Stage 2's *angle-resolved* polarization factor
(`io.laser.ELLIPTICITY_IS_NOOP`, §9.2), which is the only place an ellipse is
distinguishable from a line.

**No coordinate normalization** (§2.1, RES015): everything here is lab-frame CGS.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import Bunch, illumination_window, overlap_time_window
from ...io.laser import LaserField, fit_gaussian_paraxial
from ...io.units import C_CGS, E_ESU, HBAR_CGS, ME_CGS, SIGMA_T_CGS
from . import chunking
from .chunking import run_in_chunks

__all__ = [
    "TrajectorySamples",
    "ahat_from_shape",
    "integrate_trajectories",
    "photon_density_scale",
    "RELATIVE_VELOCITY",
    "BYTES_PER_PARTICLE_STEP",
    "ShapeTable",
    "Table",
    "deposit_shape_table",
    "retarget_ahat",
    "spectrum_from_table",
    "angular_spectrum_from_table",
    "spectrum_in_angular_range",
    "angle_integrated_spectrum",
    "KERNEL_NORMALIZATION_CONSTANT",
    "DEFAULT_SHAPE_BINS",
    "DEFAULT_RETARGET_BINS",
    "DEFAULT_AHAT_MIN",
    "DEFAULT_AHAT_MAX",
    "DEFAULT_AHAT_DECADES",
    "SPECTRUM_WORKING_SET_BYTES",
    "SPECTRUM_MAX_ENERGY_CHUNK",
    "stage2_backend",
]

#: Relative-velocity factor for the near-backscattering geometry: electron and photon
#: approach at ``c(1 - v.n0_hat)`` where n0_hat is the laser propagation direction.
#: Head-on (n0_hat = -z_hat, v = beta z_hat) gives ``c(1+beta) -> 2c``.
#: With crossing angles theta_xz, theta_yz: n0_hat = (-sin(theta_xz)cos(theta_yz), sin(theta_yz), -cos(theta_xz)cos(theta_yz))
#: so 1 - v.n0_hat = 1 + beta cos(theta_xz)cos(theta_yz) -> 2 cos^2(alpha/2) where cos(alpha) = cos(theta_xz)cos(theta_yz).
def relative_velocity(
    beta: float, theta_xz: float = 0.0, theta_yz: float = 0.0, *, k_hat: np.ndarray | None = None
) -> float:
    """Relative-velocity factor ``1 - v.n0_hat = 1 + beta cos(theta_xz) cos(theta_yz)``.

    Parameters
    ----------
    beta : float
        Electron velocity / c (typically ~1 for ultra-relativistic beams).
    theta_xz : float
        Crossing angle in x-z plane (radians).
    theta_yz : float
        Crossing angle in y-z plane (radians).
    k_hat : np.ndarray | None
        Unit propagation vector. If provided, ``1 - beta * k_hat[2]`` is used directly.

    Returns
    -------
    float
        The factor multiplying ``c`` for the relative velocity. Head-on limit (theta_xz=theta_yz=0)
        gives ``1 + beta`` (-> 2 for beta=1).
    """
    if k_hat is not None:
        return 1.0 - beta * float(k_hat[2])
    return 1.0 + beta * math.cos(theta_xz) * math.cos(theta_yz)


#: Backward-compatible alias for code/docs that still reference the constant name.
#: ``RELATIVE_VELOCITY = relative_velocity(1.0) == 2.0``
RELATIVE_VELOCITY = 2.0

#: The table-free spectrum has several live particle-by-energy temporaries. Its work is
#: partitioned beneath this fixed ceiling even where the host cannot report available RAM;
#: :mod:`chunking` further shrinks the particle side on constrained hosts.
SPECTRUM_WORKING_SET_BYTES = 64 * 1024**2

#: Energy is also an independent reduction axis. Capping it keeps a very long requested
#: output grid from making even one particle chunk too large.
SPECTRUM_MAX_ENERGY_CHUNK = 2048

#: ``y``, its boolean mask, and the resulting spectral shape are live together. This
#: intentionally overestimates their NumPy temporary footprint.
_SPECTRUM_BYTES_PER_PARTICLE_ENERGY = 64


#: The table-free spectrum has several live particle-by-energy temporaries. Its work is
#: partitioned beneath this fixed ceiling even where the host cannot report available RAM;
#: :mod:`chunking` further shrinks the particle side on constrained hosts.
SPECTRUM_WORKING_SET_BYTES = 64 * 1024**2

#: Energy is also an independent reduction axis. Capping it keeps a very long requested
#: output grid from making even one particle chunk too large.
SPECTRUM_MAX_ENERGY_CHUNK = 2048

#: ``y``, its boolean mask, and the resulting spectral shape are live together. This
#: intentionally overestimates their NumPy temporary footprint.
_SPECTRUM_BYTES_PER_PARTICLE_ENERGY = 64


def polarization_factor(
    gamma: float,
    theta_x: float,
    theta_y: float,
    theta_x_obs: float,
    theta_y_obs: float,
    ellipticity: float,
    psi_pol: float,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> float | np.ndarray:
    """``Tr(U^T Xi U)`` for one electron and lab-frame observation direction.

    The field-free electron velocity is ``beta * (theta_x, theta_y, 1)`` normalized
    in the shared lab frame (DV006). The laser basis is rotated only once for the
    pulse crossing angle; it is never rotated separately for each electron.
    """
    return polarization_factor_vectorized(
        np.asarray(gamma), np.asarray(theta_x), np.asarray(theta_y),
        theta_x_obs, theta_y_obs, ellipticity, psi_pol, theta_xz, theta_yz,
    )


def polarization_factor_vectorized(
    gamma: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    theta_x_obs: float,
    theta_y_obs: float,
    ellipticity: float,
    psi_pol: float,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> np.ndarray:
    """Vectorized :func:`polarization_factor` for per-electron lab velocities.

    ``gamma``, ``theta_x``, and ``theta_y`` broadcast to the table-cell shape. Invalid
    resonance cells can carry ``gamma == 0``; they receive a finite placeholder here and
    are removed by the calling kernel's physical-resonance mask.
    """
    cos_xz = math.cos(theta_xz)
    cos_yz = math.cos(theta_yz)
    sin_xz = math.sin(theta_xz)
    sin_yz = math.sin(theta_yz)
    R = np.array([
        [cos_xz, sin_xz * sin_yz, sin_xz * cos_yz],
        [0.0, cos_yz, -sin_yz],
        [-sin_xz, cos_xz * sin_yz, cos_xz * cos_yz],
    ], dtype=float)

    cp = math.cos(psi_pol)
    sp = math.sin(psi_pol)
    e0 = R @ np.array([cp, sp, 0.0])
    e1 = R @ np.array([-sp, cp, 0.0])
    n = np.array([theta_x_obs, theta_y_obs, 1.0])
    n /= np.linalg.norm(n)

    gamma, theta_x, theta_y = np.broadcast_arrays(
        np.asarray(gamma, dtype=float), np.asarray(theta_x, dtype=float),
        np.asarray(theta_y, dtype=float),
    )
    valid_gamma = gamma >= 1.0
    gamma_safe = np.where(valid_gamma, gamma, 1.0)
    beta = np.sqrt(1.0 - gamma_safe**-2)
    direction_norm = np.sqrt(1.0 + theta_x**2 + theta_y**2)
    vx = beta * theta_x / direction_norm
    vy = beta * theta_y / direction_norm
    vz = beta / direction_norm
    one_minus_vn = 1.0 - (vx * n[0] + vy * n[1] + vz * n[2])

    a0 = float(np.dot(n, e0))
    a1 = float(np.dot(n, e1))
    b0 = vx * e0[0] + vy * e0[1] + vz * e0[2]
    b1 = vx * e1[0] + vy * e1[1] + vz * e1[2]

    eps2 = ellipticity * ellipticity
    xi00 = 1.0 / (1.0 + eps2)
    xi11 = eps2 / (1.0 + eps2)
    gamma2_vn2 = gamma_safe * gamma_safe * one_minus_vn * one_minus_vn
    term0 = 1.0 - a0 * a0 / gamma2_vn2 + 2.0 * a0 * b0 / one_minus_vn
    term1 = 1.0 - a1 * a1 / gamma2_vn2 + 2.0 * a1 * b1 / one_minus_vn
    return xi00 * term0 + xi11 * term1


#: Live bytes per (particle x step) in Stage 0's inner loop, for auto-chunking.
#: Calibrated on the predecessor's equivalent kernel on a GTX 1660 Ti (6 GB): 400,000
#: particles at 64 steps fit, 800,000 did not, with the failing allocation reported at
#: 409,600,000 bytes — about 113 bytes per particle-step, consistent with roughly fourteen
#: concurrently-live float64 temporaries. Rounded up for headroom against other processes
#: on the device.
BYTES_PER_PARTICLE_STEP = 200


def photon_density_scale(laser: LaserField) -> float:
    """Photons per cm^3 per unit cycle-averaged intensity ``<a^2>``.

    The inverse of the laser's energy→intensity chain, and **polarization-agnostic** by
    construction (RES054). With ``<a^2> = (e / m_e c omega0)^2 4 pi U density``
    and the physical photon density ``N_l density = (U / hbar omega0) density``, the pulse
    energy cancels::

        n_photons(r, t) = <a^2>(r, t) * (m_e c)**2 omega0 / (4 pi hbar e**2)

    which is why Stage 0 can take the whole laser through `LaserField.intensity_profile`
    alone. Only ``omega0`` remains, and that comes directly from the laser if exposed,
    or from the descriptive fit (§3.3, RES067).

    Note the ``4 pi``: this scales ``<a^2>``, not the peak ``a0^2`` (which would be
    ``8 pi``, and would carry a polarization convention with it).
    """
    if hasattr(laser, "omega0"):
        omega0 = laser.omega0()
    else:
        omega0 = fit_gaussian_paraxial(laser).omega0()
    return (ME_CGS * C_CGS) ** 2 * omega0 / (4.0 * math.pi * HBAR_CGS * E_ESU**2)


def ahat_from_shape(a0_shape, intensity_peak: float):
    """The paper's ``ahat`` from `TrajectorySamples.a0_shape` and the peak ``<a^2>``.

    The paper's ``ahat = (a0^2 Tr Xi / 2) int|E|^4 / int|E|^2`` groups as
    ``<a^2>_peak * int|E|^4 / int|E|^2`` once ``<a^2> = C a0^2`` is substituted — and
    ``<a^2>`` is polarization-agnostic at fixed pulse energy
    (`io.laser.GaussianParaxialLaser.intensity_profile`), so **no cycle-average factor
    appears here at all** (RES054). `a0_shape` is the shape ratio verbatim; this is a plain
    product.
    """
    return float(intensity_peak) * a0_shape


@dataclass(frozen=True)
class TrajectorySamples:
    """One sample per macroparticle, ready for Stage 1 deposition. CGS.

    ``gamma``/``theta_x``/``theta_y`` pass straight through from the bunch: the pusher is
    ballistic, so a particle's energy and angles do not change along its trajectory.

    ``luminosity`` is the plan's per-particle ``L`` — the photon weight this macroparticle
    deposits, integrated over its passage through the pulse. Summing it is the total yield.

    ``a0_shape`` is **not** the trajectory-averaged effective intensity ``ahat``; it is
    the paper's ``int|E|**4 / int|E|**2`` — ``ahat``'s intensity-independent shape factor,
    computed without reference to any actual pulse strength. That is what lets one Stage 0
    run be retargeted to a different pulse energy without rerunning it (§5's
    `REUSE_INTERMEDIATES` tier). :func:`ahat_from_shape` is the only route from here to
    ``ahat``.

    ``intensity_peak`` is the peak **cycle-averaged** ``<a^2>``, not a peak ``a0``; nothing
    in this dataclass carries a polarization convention, because ``<a^2>`` at fixed pulse
    energy does not depend on one (RES054).

    It is one scalar per particle rather than a per-timestep distribution, and that is
    physics, not an optimization: in this weakly nonlinear regime the photon formation
    length spans the whole trajectory, so — unlike synchrotron radiation — the trajectory
    may **not** be split into independently radiating segments. The predecessor learned
    this the hard way and left a warning against going back to per-timestep a0.
    """

    gamma: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    a0_shape: np.ndarray
    luminosity: np.ndarray
    intensity_peak: float
    n_steps: int

    @property
    def n_particles(self) -> int:
        return int(self.gamma.shape[0])

    def total_yield(self) -> float:
        return float(np.sum(self.luminosity))

    def ahat(self) -> np.ndarray:
        """The trajectory-averaged effective intensity, at this pulse's own strength."""
        return ahat_from_shape(self.a0_shape, self.intensity_peak)

    def retargeted_ahat(self, intensity_peak: float) -> np.ndarray:
        """``ahat`` for a pulse of a different peak ``<a^2>``, no rerun needed."""
        return ahat_from_shape(self.a0_shape, intensity_peak)

    def retargeted_luminosity(self, intensity_peak: float) -> np.ndarray:
        """``luminosity`` for a pulse of a different peak ``<a^2>``, no rerun needed.

        ``luminosity`` integrates the *actual* local intensity (not the normalized ratio
        `a0_shape` does), so unlike `a0_shape` it is not already strength-independent —
        but for the same envelope shape ``<a^2>(t) = intensity_peak * envelope(t)`` is
        exactly *linear* in the peak, so ``luminosity`` scales linearly too (RES028). This is
        what lets both photon count *and* redshift for a different pulse energy come from
        cached Stage 0 samples with no rerun.
        """
        return (intensity_peak / self.intensity_peak) * self.luminosity


def integrate_trajectories(
    bunch: Bunch,
    laser: LaserField,
    n_electrons: float,
    *,
    n_steps: int = 200,
    threshold: float = 1e-3,
    window: str = "active_region",
    backend: str = "numpy",
    chunk: int | None = None,
) -> TrajectorySamples:
    """Stage 0: push every macroparticle through the pulse and sample the overlap.

    Each particle travels a straight line at ``c`` (§2.3) across its own overlap window —
    the closed-form window `gammaforge.io.bunch.overlap_time_window` derives from the
    laser's active region, so a particle is integrated over exactly the interval in which
    it can see the pulse and no longer. The integral is a midpoint sum over ``n_steps``.

    ``n_electrons`` is the interaction's ``N_e``. The bunch's own weights are *relative*
    (§3.2), so absolute photon counts enter here and only here — which is what makes every
    output exactly linear in charge and a charge edit a pure rescale (§5).

    ``threshold`` is the same active-region fraction the prefilter uses. Passing a smaller
    value integrates a longer window; results converge as it shrinks, and the prefilter
    invariance of §7 requires that a particle excluded at this threshold contributes
    nothing at it.

    ``window`` selects where those ``n_steps`` are spent, and it is the cheapest accuracy
    knob here. ``"active_region"`` (the default) uses
    `gammaforge.io.bunch.overlap_time_window`, a conservative *geometric* bound: correct,
    but far wider than the stretch in which a particle is actually illuminated, because the
    active region ignores the ``1 / (s1 s2)`` dimming away from focus. ``"illumination"``
    uses `gammaforge.io.bunch.illumination_window`, which brackets the illuminated stretch
    itself — same particles, same step count, measurably better resolution, because no step
    is spent where the integrand is negligible.

    The default is deliberately the wider one. The illuminated window is an *estimate*
    rather than a bound, so switching changes results (slightly, and towards the converged
    answer) — a deliberate act, not something to inherit silently, and the reason the golden
    references still describe the geometric window.

    Chunking is over particles, whose trajectories are independent, so the partition
    cannot change the answer.
    """
    if n_steps < 1:
        raise ValueError(f"integrate_trajectories: n_steps must be >= 1, got {n_steps}")
    if window not in ("active_region", "illumination"):
        raise ValueError(
            f"integrate_trajectories: window must be 'active_region' or 'illumination', got {window!r}"
        )
    _check_backend(backend)

    if window == "illumination":
        t0, t1 = illumination_window(bunch, laser, threshold)
    else:
        t0, t1 = overlap_time_window(bunch, laser, threshold)
    span = np.maximum(0.0, t1 - t0)
    # A particle that never enters the pulse gets an empty window, which
    # `overlap_time_window` reports as t0 = +inf, t1 = -inf. Its span is zero and it
    # contributes nothing — but `inf + 0 * 0` is still `inf`, and a trajectory evaluated
    # there hands the laser a NaN that propagates into the sum for *every* particle. So
    # empty windows are anchored at a finite time, and the zero span does the rest.
    # (With the prefilter on such particles are already gone; with it off they are not,
    # which is how the §7 prefilter-invariance property found this.)
    start = np.where(span > 0.0, t0, 0.0)
    # Midpoint rule: no sample sits on the window edge, where the integrand is smallest and
    # the window definition is least meaningful.
    offsets = (np.arange(n_steps) + 0.5) / n_steps

    if hasattr(laser, "intensity_peak"):
        intensity_peak = laser.intensity_peak()
    else:
        intensity_peak = fit_gaussian_paraxial(laser).intensity_peak()
    density_scale = photon_density_scale(laser)
    # Absolute photons per macroparticle-second of overlap. The bunch's weights are
    # relative and sum to 1 over the *unfiltered* population, so scaling by N_e here keeps
    # a prefiltered run and a full run identical (§3.2).
    weight = n_electrons * bunch.weight

    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    velocity = (C_CGS * bunch.thx / norm, C_CGS * bunch.thy / norm, C_CGS / norm)
    beta = 1.0  # ultra-relativistic: v = c exactly
    if hasattr(laser, "focusing_axes"):
        k_hat, _, _ = laser.focusing_axes()
        rel_vel = relative_velocity(beta, k_hat=k_hat)
    elif hasattr(laser, "m") and hasattr(laser, "theta_xz") and hasattr(laser, "theta_yz"):
        rel_vel = relative_velocity(beta, laser.m("theta_xz"), laser.m("theta_yz"))
    else:
        metrics = fit_gaussian_paraxial(laser)
        rel_vel = relative_velocity(beta, metrics.m("theta_xz"), metrics.m("theta_yz"))

    def integrate(first_index: int, last_index: int):
        sl = slice(first_index, last_index)
        times = start[sl, None] + offsets[None, :] * span[sl, None]
        # `<a^2>`, not `a0`: everything below is a functional of the cycle-averaged
        # intensity, polarization-agnostic at fixed pulse energy (RES054). Do not form `a0`
        # and square it here — that's where RES053's missing factor of two hid.
        intensity = np.asarray(
            laser.intensity_profile(
                bunch.x[sl, None] + velocity[0][sl, None] * times,
                bunch.y[sl, None] + velocity[1][sl, None] * times,
                bunch.z[sl, None] + velocity[2][sl, None] * times,
                times,
            )
        )
        # The photon density an electron flies through, and the rate it scatters at.
        # In CGS this is simply flux x cross-section x time; the predecessor's k0**2 was
        # the Jacobian of its coordinate normalization and has no counterpart here (RES015).
        dt = span[sl] / n_steps
        rate = rel_vel * density_scale * C_CGS * SIGMA_T_CGS
        luminosity = rate * weight[sl] * dt * np.sum(intensity, axis=1)

        # `ratio` is the local intensity as a fraction of the pulse's peak. a0_shape is its
        # second moment over the trajectory, normalized by its first: the intensity an
        # electron *effectively* experiences, weighted by where it actually radiated. Being
        # a ratio of intensities, it is polarization-agnostic like everything else here.
        ratio = intensity / intensity_peak
        moment_1 = np.sum(ratio, axis=1)
        # A particle with no window saw nothing, whatever the envelope reads at the
        # anchor time above; its effective intensity is zero, not the value at t = 0.
        usable = (moment_1 > 0.0) & (dt > 0.0)
        a0_shape = np.where(
            usable, np.sum(ratio**2, axis=1) / np.maximum(moment_1, 1e-300), 0.0
        )
        return luminosity, a0_shape

    parts = run_in_chunks(
        bunch.n_particles,
        integrate,
        chunk=chunk,
        # No ceiling: the predecessor's measured chunk ceiling was for the *spectrum*
        # path's s-axis, where per-launch overhead amortized by ~8-16. Stage 0 partitions
        # particles, where no such measurement exists, and inventing one would be the
        # cargo-culting the chunking module's own docstring warns the constants against.
        bytes_per_item=BYTES_PER_PARTICLE_STEP * n_steps,
        backend=backend,
    )
    # An empty bunch is reachable, not hypothetical: the prefilter discards every particle
    # for a mistimed pulse or a bunch far wider than the spot. `np.concatenate([])` raises,
    # which would make the prefilter turn a zero yield into an exception — the opposite of
    # the pure optimization §3.2 promises.
    luminosity = np.concatenate([part[0] for part in parts]) if parts else np.zeros(0)
    a0_shape = np.concatenate([part[1] for part in parts]) if parts else np.zeros(0)

    return TrajectorySamples(
        gamma=bunch.gamma,
        theta_x=bunch.thx,
        theta_y=bunch.thy,
        a0_shape=a0_shape,
        luminosity=luminosity,
        intensity_peak=intensity_peak,
        n_steps=n_steps,
    )


#: Default bin counts for :func:`deposit_shape_table`'s four axes, in ``(gamma, theta_x,
#: theta_y, a0_shape)`` order. The ``a0_shape`` axis is deliberately fine — it is bounded
#: and a0-peak-independent, so there is no dynamic-range reason to keep it small the way
#: the old direct-onto-``ahat`` deposit's fourth axis had to be.
DEFAULT_SHAPE_BINS = (48, 48, 48, 96)

#: Defaults for :func:`retarget_ahat`'s fixed, non-uniform target grid, tuned against this
#: repo's scenario bank (RES032) rather than re-derived from the predecessor's
#: ``DEFAULT_A0_MAX``/``retarget_a0`` defaults, which were sized for a different bank.
DEFAULT_RETARGET_BINS = 32
DEFAULT_AHAT_MIN = 0.0
DEFAULT_AHAT_MAX = 0.5
DEFAULT_AHAT_DECADES = 1.0


def _uniform_edges(values: np.ndarray, n_bins: int, margin: float, floor_zero: bool = False) -> np.ndarray:
    """``n_bins + 1`` uniform edges spanning ``values``, padded by ``margin`` of the span.

    A degenerate span (every sample identical — a monoenergetic, zero-divergence beam is
    a real scenario, not a hypothetical one) would otherwise produce a zero-width grid
    that every sample lands exactly on the edge of; padded by ``margin`` of the value's
    own scale instead so the grid always has a real width.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    pad = margin * span if span > 0.0 else margin * max(abs(lo), 1.0)
    lo, hi = lo - pad, hi + pad
    if floor_zero:
        lo = max(lo, 0.0)
    return np.linspace(lo, hi, n_bins + 1)


def _validate_edges_and_shape(edges: tuple[np.ndarray, ...], H: np.ndarray, name: str) -> None:
    """Shared structural check for :class:`ShapeTable` and :class:`Table`: ``H``'s shape
    matches the edge counts, and every axis's edges are strictly increasing. Says nothing
    about *uniform* spacing — `Table`'s ``ahat_edges`` deliberately is not (§4.2)."""
    expected = tuple(e.size - 1 for e in edges)
    if H.shape != expected:
        raise ValueError(f"{name}: H.shape {H.shape} does not match edge counts {expected}")
    for e in edges:
        if np.any(np.diff(e) <= 0.0):
            raise ValueError(f"{name}: edges must be strictly increasing")


@dataclass(frozen=True)
class ShapeTable:
    """Stage 1's output: a 4D photon-weight density over ``(gamma, theta_x, theta_y,
    a0_shape)`` — pulse-strength-agnostic on its axis, since ``a0_shape`` is by
    construction independent of any actual pulse (`TrajectorySamples.a0_shape`).

    Not agnostic in *mass*: ``H`` is deposited with ``samples.luminosity`` at
    ``source_intensity_peak`` (the pulse Stage 0 actually ran), and luminosity scales
    linearly in that peak for the same cached trajectories
    (`TrajectorySamples.retargeted_luminosity`) — the same relation that makes ``ahat``'s
    rescale exact. Querying at a different pulse strength needs :func:`retarget_ahat` to
    rescale ``H``'s total mass, not just relabel the axis.

    Every axis stays uniform (unlike the ``ahat`` axis of the `Table` this feeds into), so
    :attr:`bin_volume` is a single scalar, same as `Table`'s used to be.
    """

    gamma_edges: np.ndarray
    theta_x_edges: np.ndarray
    theta_y_edges: np.ndarray
    a0_shape_edges: np.ndarray
    H: np.ndarray
    total_weight: float
    scheme: str
    source_intensity_peak: float

    def __post_init__(self) -> None:
        _validate_edges_and_shape(
            (self.gamma_edges, self.theta_x_edges, self.theta_y_edges, self.a0_shape_edges), self.H, "ShapeTable"
        )

    @property
    def gamma_centers(self) -> np.ndarray:
        return 0.5 * (self.gamma_edges[:-1] + self.gamma_edges[1:])

    @property
    def theta_x_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_x_edges[:-1] + self.theta_x_edges[1:])

    @property
    def theta_y_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_y_edges[:-1] + self.theta_y_edges[1:])

    @property
    def a0_shape_centers(self) -> np.ndarray:
        return 0.5 * (self.a0_shape_edges[:-1] + self.a0_shape_edges[1:])

    @property
    def bin_volume(self) -> float:
        """Cell volume, constant because every axis here is a uniform grid."""
        return float(
            (self.gamma_edges[-1] - self.gamma_edges[0])
            * (self.theta_x_edges[-1] - self.theta_x_edges[0])
            * (self.theta_y_edges[-1] - self.theta_y_edges[0])
            * (self.a0_shape_edges[-1] - self.a0_shape_edges[0])
            / (self.H.shape[0] * self.H.shape[1] * self.H.shape[2] * self.H.shape[3])
        )


@dataclass(frozen=True)
class Table:
    """Stage 2's input: a 4D photon-weight density over ``(gamma, theta_x, theta_y,
    ahat)``, for one specific peak a0.

    ``H`` is a **density** (weight per unit cell volume). Unlike `ShapeTable`, the ``ahat``
    axis is generally **non-uniform** — :func:`retarget_ahat` builds it dense near
    ``ahat_max`` and coarse toward ``ahat_min`` (RES032), so there is no
    single scalar cell volume; :attr:`ahat_widths` and :attr:`gamma_theta_cell_area` are
    what :func:`spectrum_from_table` actually needs.
    """

    gamma_edges: np.ndarray
    theta_x_edges: np.ndarray
    theta_y_edges: np.ndarray
    ahat_edges: np.ndarray
    H: np.ndarray
    total_weight: float
    scheme: str

    def __post_init__(self) -> None:
        _validate_edges_and_shape(
            (self.gamma_edges, self.theta_x_edges, self.theta_y_edges, self.ahat_edges), self.H, "Table"
        )

    @property
    def gamma_centers(self) -> np.ndarray:
        return 0.5 * (self.gamma_edges[:-1] + self.gamma_edges[1:])

    @property
    def theta_x_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_x_edges[:-1] + self.theta_x_edges[1:])

    @property
    def theta_y_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_y_edges[:-1] + self.theta_y_edges[1:])

    @property
    def ahat_centers(self) -> np.ndarray:
        return 0.5 * (self.ahat_edges[:-1] + self.ahat_edges[1:])

    @property
    def ahat_widths(self) -> np.ndarray:
        """Per-bin ``ahat`` width, shape ``(n_ahat,)`` — non-uniform, unlike every other
        axis here, so this is an array rather than a scalar."""
        return np.diff(self.ahat_edges)

    @property
    def gamma_theta_cell_area(self) -> float:
        """The still-uniform ``theta_x * theta_y`` cell area (gamma is interpolated, not
        integrated over, in :func:`spectrum_from_table` — see :func:`_interp_gamma`)."""
        return float(
            (self.theta_x_edges[-1] - self.theta_x_edges[0]) / self.H.shape[1]
            * (self.theta_y_edges[-1] - self.theta_y_edges[0]) / self.H.shape[2]
        )


def _cell_fractions(values: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Continuous cell coordinate of ``values`` in ``edges``, in units of one bin width."""
    return (values - edges[0]) / (edges[-1] - edges[0]) * n_bins


def _deposit_nearest(coords: tuple[np.ndarray, ...], weight: np.ndarray, n_bins: tuple[int, ...]) -> np.ndarray:
    idx = [np.clip(np.floor(c).astype(np.int64), 0, n - 1) for c, n in zip(coords, n_bins)]
    flat = np.ravel_multi_index(idx, n_bins)
    return np.bincount(flat, weights=weight, minlength=int(np.prod(n_bins))).reshape(n_bins)


def _deposit_cic(coords: tuple[np.ndarray, ...], weight: np.ndarray, n_bins: tuple[int, ...]) -> np.ndarray:
    """Cloud-in-cell: each sample splits its weight over its 16 neighbouring cells.

    Cell-centred convention (predecessor's, §4.2): a sample's continuous coordinate is
    shifted by ``-0.5`` so it interpolates between cell *centres*. ``edge='clamp'``
    always — overflow folds into the boundary cell rather than discarding weight, which
    is what keeps a CIC deposit's total exactly equal to a nearest deposit's for the same
    samples (both conserve weight; only where it lands differs).
    """
    n_axes = len(coords)
    shifted = [c - 0.5 for c in coords]
    low = [np.floor(s).astype(np.int64) for s in shifted]
    frac = [s - lo for s, lo in zip(shifted, low)]

    flat_size = int(np.prod(n_bins))
    H_flat = np.zeros(flat_size, dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=n_axes):
        idx = []
        w = weight
        for axis, bit in enumerate(corner):
            i = np.clip(low[axis] + bit, 0, n_bins[axis] - 1)
            f = frac[axis] if bit else (1.0 - frac[axis])
            idx.append(i)
            w = w * f
        flat = np.ravel_multi_index(idx, n_bins)
        H_flat += np.bincount(flat, weights=w, minlength=flat_size)
    return H_flat.reshape(n_bins)


def deposit_shape_table(
    samples: TrajectorySamples,
    *,
    n_bins: tuple[int, int, int, int] = DEFAULT_SHAPE_BINS,
    scheme: str = "nearest",
    margin: float = 0.02,
) -> ShapeTable:
    """Stage 1: bin Stage 0's per-particle samples into the 4D ``a0_shape`` table ``H``.

    Peak-a0-agnostic: bins directly onto ``samples.a0_shape`` (already independent of any
    actual pulse) with ``samples.luminosity`` as the deposited weight, so one deposit
    serves every peak a0 a caller might later want via :func:`retarget_ahat` — unlike the
    single-stage ``ahat``-axis deposit this replaces, which needed a fresh deposit per
    peak a0 (RES028, superseded by RES032).

    ``scheme`` is ``"nearest"`` (one cell per sample) or ``"cic"`` (cloud-in-cell, 16
    neighbours per sample) — both conserve total weight exactly; CIC trades a discretized
    ``H`` for a smoother one, at 16x the deposition cost.
    """
    if scheme not in ("nearest", "cic"):
        raise ValueError(f"deposit_shape_table: scheme must be 'nearest' or 'cic', got {scheme!r}")

    gamma_edges = _uniform_edges(samples.gamma, n_bins[0], margin)
    theta_x_edges = _uniform_edges(samples.theta_x, n_bins[1], margin)
    theta_y_edges = _uniform_edges(samples.theta_y, n_bins[2], margin)
    a0_shape_edges = _uniform_edges(samples.a0_shape, n_bins[3], margin, floor_zero=True)
    edges = (gamma_edges, theta_x_edges, theta_y_edges, a0_shape_edges)

    coords = tuple(
        _cell_fractions(values, e, n)
        for values, e, n in zip(
            (samples.gamma, samples.theta_x, samples.theta_y, samples.a0_shape), edges, n_bins
        )
    )
    deposit = _deposit_nearest if scheme == "nearest" else _deposit_cic
    H_raw = deposit(coords, samples.luminosity, n_bins)

    bin_volume = float(np.prod([e[-1] - e[0] for e in edges]) / np.prod(n_bins))
    return ShapeTable(
        gamma_edges=gamma_edges,
        theta_x_edges=theta_x_edges,
        theta_y_edges=theta_y_edges,
        a0_shape_edges=a0_shape_edges,
        H=H_raw / bin_volume,
        total_weight=float(H_raw.sum()),
        scheme=scheme,
        source_intensity_peak=samples.intensity_peak,
    )


def _ahat_target_edges(ahat_min: float, ahat_max: float, n_bins: int, decades: float) -> np.ndarray:
    """``n_bins + 1`` non-uniform ``ahat`` edges, log-spaced in distance from the top:
    finest near ``ahat_max`` (where the redshift correction is significant), coarsest near
    ``ahat_min`` (folded floor bin — §4.2, RES032)::

        v_i = (ahat_max - ahat_min) * 10**(-decades * i / n_bins),  i = 0..n_bins
        ahat_i = ahat_max - v_i

    ``i=0`` lands exactly on ``ahat_min`` (``v_0`` is the full span). The raw ``i=n_bins``
    value lands within ``10**-decades`` of ``ahat_max``, not exactly on it; snapped to
    ``ahat_max`` exactly below, so both ends of the grid are exact. **This widens
    the single top bin** — negligibly at ``decades >= 3`` (the widening is a factor of
    ``10**-decades`` of the span), but visibly at the ``decades=1`` this repo's scenario
    bank actually uses (RES032): the top bin ends up wider than its immediate
    neighbour, not narrower. A deliberate, bounded exception to the "finer toward the top"
    trend at the very last bin, not a bug — every other bin still shrinks monotonically.
    """
    if ahat_max <= ahat_min:
        raise ValueError(f"_ahat_target_edges: ahat_max ({ahat_max}) must exceed ahat_min ({ahat_min})")
    if decades <= 0.0:
        raise ValueError(f"_ahat_target_edges: decades must be positive, got {decades}")
    i = np.arange(n_bins + 1)
    v = (ahat_max - ahat_min) * 10.0 ** (-decades * i / n_bins)
    edges = ahat_max - v
    edges[-1] = ahat_max
    return edges


def retarget_ahat(
    shape_table: ShapeTable,
    intensity_peak: float,
    *,
    ahat_min: float = DEFAULT_AHAT_MIN,
    ahat_max: float = DEFAULT_AHAT_MAX,
    n_bins: int = DEFAULT_RETARGET_BINS,
    decades: float = DEFAULT_AHAT_DECADES,
) -> Table:
    """Stage 1.5: conservative (mass-preserving) regrid of a `ShapeTable`'s ``a0_shape``
    axis onto the fixed, non-uniform ``ahat`` axis Stage 2 actually queries, for one
    specific peak cycle-averaged intensity ``<a^2>`` (RES032, supersedes
    RES028; RES054 for why the parameter is an intensity rather than an amplitude).

    Cheap and independent of ``n_particles`` — a ``shape_table.a0_shape_edges.size x
    n_bins``-sized tensordot, not a re-deposit — so a `Collision` can cache the shape
    deposit once and retarget many pulse strengths from it.

    Conservative (mass-preserving) under the same piecewise-uniform-density assumption
    deposition itself makes, onto :func:`_ahat_target_edges`'s non-uniform target law. The
    deposited mass is rescaled by ``intensity_peak / shape_table.source_intensity_peak``
    (:meth:`TrajectorySamples.retargeted_luminosity`'s relation), since the requested
    intensity is generally not the one Stage 0 ran at.
    """
    if ahat_max <= ahat_min:
        raise ValueError(f"retarget_ahat: ahat_max ({ahat_max}) must exceed ahat_min ({ahat_min})")
    intensity_peak = float(intensity_peak)

    # Exact, and the third and last caller of :func:`ahat_from_shape`: a0_shape is
    # strength-independent by construction, so the axis transform is a pure scale.
    source_edges = ahat_from_shape(shape_table.a0_shape_edges, intensity_peak)
    target_edges = _ahat_target_edges(ahat_min, ahat_max, n_bins, decades)

    # Extend both outer target edges to +-inf for overlap purposes only: source mass below
    # ahat_min folds into the floor bin, and — symmetrically — source mass above ahat_max
    # (a pulse strong enough to push the rescaled source past the configured ceiling) folds
    # into the top bin rather than being silently dropped.
    edges_ext = target_edges.copy()
    edges_ext[0] = -np.inf
    edges_ext[-1] = np.inf

    src_lo, src_hi = source_edges[:-1], source_edges[1:]
    src_width = src_hi - src_lo
    tgt_lo, tgt_hi = edges_ext[:-1], edges_ext[1:]

    lo = np.maximum(src_lo[:, None], tgt_lo[None, :])
    hi = np.minimum(src_hi[:, None], tgt_hi[None, :])
    overlap = np.clip(hi - lo, 0.0, None)
    # W[i, j]: fraction of source bin i's mass assigned to target bin j.
    W = overlap / np.clip(src_width, 1e-300, None)[:, None]

    # The source (a0_shape) axis stays uniform in this design — deposit_shape_table only
    # ever builds it via _uniform_edges — so a single scalar width is exact here, unlike
    # the predecessor's identically-shaped `da_source = table.grid.widths[3]`, which was
    # only safe because its source was uniform too (never a general non-uniform case).
    da_source = shape_table.a0_shape_edges[1] - shape_table.a0_shape_edges[0]
    luminosity_rescale = intensity_peak / shape_table.source_intensity_peak

    mass_source = shape_table.H * da_source * luminosity_rescale  # density -> mass, at this strength
    mass_target = np.tensordot(mass_source, W, axes=([3], [0]))
    target_width = np.diff(target_edges)
    H_target = mass_target / target_width

    # Truncate trailing ahat bins the rescaled source never reaches: their mass is exactly
    # zero (W's overlap is exactly zero where no source bin overlaps a target bin), so
    # dropping them changes nothing spectrum_from_table's cell sum would have computed —
    # only how many always-zero terms it evaluates. One-sided: the floor bin (index 0)
    # always catches whatever folded below ahat_min, so only the top can be empty.
    marginal = H_target.sum(axis=(0, 1, 2))
    populated = np.nonzero(marginal > 0.0)[0]
    last = int(populated[-1]) if populated.size else 0
    H_target = H_target[..., : last + 1]
    target_edges = target_edges[: last + 2]

    return Table(
        gamma_edges=shape_table.gamma_edges,
        theta_x_edges=shape_table.theta_x_edges,
        theta_y_edges=shape_table.theta_y_edges,
        ahat_edges=target_edges,
        H=H_target,
        total_weight=shape_table.total_weight * luminosity_rescale,
        scheme=shape_table.scheme,
    )


#: The §9.1 constant (§4.2, RES033): the predecessor's kernel math is pi-free (``coef =
#: 1.5``); ``1/(2 pi)`` is the correction RES026 derived is missing from the paper's
#: cross-section, applied here and nowhere else (`references/delta.py` applies the same
#: correction to its own transcription).
#:
#: **Derived and confirmed, never fitted (P14). Do not adjust it to make a check pass** —
#: if a check disagrees, that is a physics finding to escalate (§0), not a number to tune.
KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2.0 * math.pi)


def _interp_gamma(table: Table, g: np.ndarray) -> np.ndarray:
    """``table.H`` linearly interpolated along gamma at a per-cell query point ``g``.

    ``g`` carries one query value per ``(theta_x, theta_y, ahat)`` cell — the resonance
    condition inverted at that cell's own angle and ahat (§4.2) — so this is not a single
    1D interpolation but ``n_theta_x * n_theta_y * n_ahat`` of them, batched. A query
    outside the tabulated gamma range gets zero: the bunch's gamma distribution simply did
    not populate a resonance there, which is physical, not a boundary artefact to
    extrapolate past.
    """
    gc = table.gamma_centers
    in_range = (g >= gc[0]) & (g <= gc[-1])
    idx = np.clip(np.searchsorted(gc, g) - 1, 0, len(gc) - 2)
    idx_hi = idx + 1
    frac = np.where(in_range, (g - gc[idx]) / (gc[idx_hi] - gc[idx]), 0.0)

    tx_idx, ty_idx, a_idx = np.meshgrid(
        np.arange(table.H.shape[1]), np.arange(table.H.shape[2]), np.arange(table.H.shape[3]), indexing="ij"
    )
    lo = table.H[idx, tx_idx, ty_idx, a_idx]
    hi = table.H[idx_hi, tx_idx, ty_idx, a_idx]
    return np.where(in_range, lo * (1.0 - frac) + hi * frac, 0.0)


def spectrum_from_table(
    table: Table,
    theta_x: float,
    theta_y: float,
    s,
    *,
    psi_pol: float = 0.0,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> np.ndarray:
    """Stage 2: ``d2N / (ds dOmega)`` at one observation direction, over an array of ``s``.

    A direct grid quadrature over Stage 1's table, not the predecessor's GPU importance
    sampler (RES029) — sums the table's own ``(theta_x, theta_y, ahat)`` cells, inverting the
    resonance condition at each cell to find the gamma an electron there would need to
    radiate a photon of energy ``s`` toward ``(theta_x, theta_y)``, and interpolates ``H``
    at that gamma (:func:`_interp_gamma`).

    ``g``/``prefac`` are recomputed inside the ahat loop implicitly — this function never
    factors ahat out of the resonance condition — because the resonance shifts with ahat
    (the nonlinear redshift); an ahat-independent shortcut is wrong, not just slower.

    The polarization factor now includes ellipticity and crossing angle effects per DER006,
    replacing the head-on linear factor ``cos^2 psi``.
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    tx_c = table.theta_x_centers[:, None, None]
    ty_c = table.theta_y_centers[None, :, None]
    a_c = table.ahat_centers[None, None, :]

    r_sq = (tx_c - theta_x) ** 2 + (ty_c - theta_y) ** 2
    theta_cell_area = table.gamma_theta_cell_area
    # ahat is generally non-uniform (§4.2, RES032), so its width is a per-bin array — folded
    # into the sum below rather than factored out as a scalar the way theta's still is.
    ahat_widths = table.ahat_widths[None, None, :]

    out = np.zeros(s_arr.shape[0])
    for k, s_val in enumerate(s_arr):
        # s <= 0 is not a resonance to invert (the formula's own 1/s and 1/s**2 factors
        # are singular there) — zero photon energy is zero photons, and `out` is already
        # zero, so there is nothing to compute.
        if s_val <= 0.0:
            continue
        inv_base = 1.0 / s_val - r_sq
        # A resonance exists only where inv_base > 0 (g_sq would otherwise be negative or
        # infinite); `valid` gates every quantity built from it, including the gamma this
        # cell would query `H` at, so an invalid cell contributes exactly zero rather than
        # a stray extrapolated lookup.
        valid = inv_base > 0.0
        g_sq = (1.0 + a_c) / np.where(valid, inv_base, 1.0)
        g = np.where(valid, np.sqrt(g_sq), 0.0)
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq) ** 2

        # New polarization factor from DER006 (replaces a_fac = 1 - 4*cos^2(psi)*r^2*g^2*gth_sq_inv)
        pol_factor = polarization_factor_vectorized(
            g, tx_c, ty_c, theta_x, theta_y, ellipticity, psi_pol, theta_xz, theta_yz
        )
        prefac = np.where(valid, pol_factor * g**5 * gth_sq_inv / (1.0 + a_c), 0.0)
        H_val = _interp_gamma(table, g)
        out[k] = (
            KERNEL_NORMALIZATION_CONSTANT
            * float(np.sum(H_val * prefac * ahat_widths))
            * theta_cell_area
            / s_val**2
        )
    return out if np.ndim(s) else out[0]


def angular_spectrum_from_table(
    table: Table,
    theta_x_grid,
    theta_y_grid,
    s,
    *,
    psi_pol: float = 0.0,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
    backend: str = "numpy",
    subsampling: int = 32,
) -> np.ndarray:
    """Stage 2: :func:`spectrum_from_table` evaluated over a grid of observation points.

    Feeds `OutputKind.COLLIMATED_SPECTRUM` (§3.4). Shape
    ``(len(theta_x_grid), len(theta_y_grid), len(s))``.

    Dispatches to the CuPy ring/annulus importance-sampling rawkernel when ``backend``
    is ``'cupy'`` or ``'auto'`` (and CuPy + CUDA are available for head-on linear
    polarization). Falls back to the NumPy brute-force grid quadrature otherwise.
    """
    selected_backend = stage2_backend(
        backend, ellipticity=ellipticity, theta_xz=theta_xz, theta_yz=theta_yz,
    )
    if selected_backend == "cupy":
        from .spectrum_sampler import calculate_angular_spectrum_gpu
        return calculate_angular_spectrum_gpu(
            table, theta_x_grid, theta_y_grid, s, psi_pol=psi_pol, subsampling=subsampling
        )

    tx = np.atleast_1d(np.asarray(theta_x_grid, dtype=float))
    ty = np.atleast_1d(np.asarray(theta_y_grid, dtype=float))
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.empty((tx.size, ty.size, s_arr.size))
    for i, x in enumerate(tx):
        for j, y in enumerate(ty):
            out[i, j, :] = spectrum_from_table(
                table, float(x), float(y), s_arr,
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz
            )
    return out


def stage2_backend(
    backend: str,
    *,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> str:
    """Resolve a Stage-2 request to its actual supported compute path."""
    if backend not in ("auto", "cupy", "numpy"):
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'cupy', or 'numpy'.")
    gpu_geometry = ellipticity == 0.0 and theta_xz == 0.0 and theta_yz == 0.0
    if backend == "cupy":
        if not gpu_geometry:
            raise NotImplementedError(
                "angular_spectrum_from_table(backend='cupy'): CuPy importance sampler only supports "
                "head-on linear polarization today; non-zero ellipticity or crossing angles must use backend='numpy'."
            )
        from .spectrum_sampler import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError(
                "angular_spectrum_from_table(backend='cupy') requested but CuPy or a CUDA device is not available."
            )
        return "cupy"
    if backend == "auto" and gpu_geometry:
        from .spectrum_sampler import is_gpu_available
        if is_gpu_available():
            return "cupy"
    return "numpy"


def angle_integrated_spectrum(samples: TrajectorySamples, s) -> np.ndarray:
    """``dN/ds``, angle-integrated in closed form — `Collision.spectrum`'s actual output.

    The same linear-Compton shape as
    `gammaforge.validation.references.delta.single_electron_spectrum`
    (``1.5 * (1 - 2y(1-y))`` for ``y = s / gamma**2``), **deliberately reimplemented here
    rather than imported.** delta exists to check this engine independently (§4.5); if it
    imported its own reference formula back from the engine it checks, or this engine
    imported from `validation`, the check would be circular in the first case and invert
    the package's dependency direction in the second. Matches the predecessor's actual
    production path (`TabulatedEngine.spectrum(s)` /
    ``spectrum_from_particles.angle_integrated_spectrum``), which used this exact table-
    free linear shape rather than Stage 1/2's nonlinear resonance — where ahat matters
    this is a stated approximation, not the full physics (mirrors delta's own caveat).
    """
    s_values = np.atleast_1d(np.asarray(s, dtype=float))
    output_bytes = s_values.size * np.dtype(float).itemsize
    available = chunking.available_ram_bytes()
    if available is not None and output_bytes > available * chunking.SAFETY_FRACTION["numpy"]:
        raise MemoryError(
            "angle_integrated_spectrum: requested output grid needs "
            f"{output_bytes:,} bytes, exceeding the available-memory budget"
        )

    spectrum = np.zeros(s_values.shape, dtype=float)
    for energy_start in range(0, s_values.size, SPECTRUM_MAX_ENERGY_CHUNK):
        energy_stop = min(energy_start + SPECTRUM_MAX_ENERGY_CHUNK, s_values.size)
        energy = s_values[energy_start:energy_stop]
        bytes_per_particle = _SPECTRUM_BYTES_PER_PARTICLE_ENERGY * energy.size
        particle_ceiling = max(1, SPECTRUM_WORKING_SET_BYTES // bytes_per_particle)

        def integrate_particle_chunk(start: int, stop: int) -> np.ndarray:
            gamma_squared = (samples.gamma[start:stop] ** 2)[:, None]
            y = energy[None, :] / gamma_squared
            shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
            return np.sum(samples.luminosity[start:stop, None] * shape / gamma_squared, axis=0)

        partials = run_in_chunks(
            samples.n_particles,
            integrate_particle_chunk,
            bytes_per_item=bytes_per_particle,
            ceiling=particle_ceiling,
        )
        if partials:
            spectrum[energy_start:energy_stop] = np.sum(partials, axis=0)
    return spectrum if np.ndim(s) else spectrum[0]


def spectrum_in_angular_range(
    table: Table,
    theta_x_range: tuple[float, float],
    theta_y_range: tuple[float, float],
    s_edges: np.ndarray,
    *,
    resolution: tuple[int, int] = (33, 33),
    psi_pol: float = 0.0,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
    backend: str = "numpy",
    subsampling: int = 32,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stage 2: the windowed on-demand query the `Collision` facade wraps.

    Builds the observation grid from ``theta_x_range``/``theta_y_range`` at ``resolution``
    and calls :func:`angular_spectrum_from_table`, then marginalizes over angle (for a 1D
    ``dN/ds`` density in the window) and over everything (for the scalar photon count in
    the window). Returns ``(cube, dN_ds, n_photons)``.
    """
    tx = np.linspace(theta_x_range[0], theta_x_range[1], resolution[0])
    ty = np.linspace(theta_y_range[0], theta_y_range[1], resolution[1])
    s_edges = np.asarray(s_edges, dtype=float)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    cube = angular_spectrum_from_table(
        table, tx, ty, s_centers,
        psi_pol=psi_pol, ellipticity=ellipticity,
        theta_xz=theta_xz, theta_yz=theta_yz,
        backend=backend, subsampling=subsampling,
    )
    dN_ds = np.trapezoid(np.trapezoid(cube, ty, axis=1), tx, axis=0)
    n_photons = float(np.trapezoid(dN_ds, s_centers))
    return cube, dN_ds, n_photons


def _check_backend(backend: str) -> None:
    """Stage 0 runs on numpy today; the other two backends land with the kernels (3a).

    Not an oversight, and deliberately an error rather than a silent fallback. P15 puts
    the laser in charge of its own sampling, so a GPU Stage 0 needs
    `LaserField.a0_profile` to be array-module-agnostic — a change to the *protocol's*
    contract that every future implementation inherits, not something Stage 0 can arrange
    on its own by wrapping the call. Accepting ``backend='cupy'`` here and running on the
    host anyway would make the §7 backend-agreement leg pass while comparing numpy with
    numpy, which is worse than not having it.
    """
    if backend != "numpy":
        raise NotImplementedError(
            f"integrate_trajectories(backend={backend!r}): Stage 0 is numpy-only until "
            f"Phase 3a. A GPU path needs LaserField.a0_profile to accept device arrays "
            f"(P15/§3.3), which is a protocol change, not a wrapper here."
        )

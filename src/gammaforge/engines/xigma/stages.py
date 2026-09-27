"""xigma's pipeline as composable pure functions (GRAND_PLAN.md §4.2).

Stage 0 (:func:`integrate_trajectories`), Stage 1 (:func:`deposit_shape_table`, onto the
peak-intensity-independent raw ``a0_shape`` axis), the retarget step
(:func:`retarget_ahat`, a conservative regrid onto the physical, non-uniform
``ahat`` axis for one specific peak intensity — RES032) and Stage 2
(:func:`spectrum_from_table`,
:func:`angular_spectrum_from_table`, :func:`spectrum_in_angular_range`) all live here.

**Every stage is a pure function.** State lives in the `Collision` facade (Phase 3a), not
here, so validation can call these directly and a stage can be reasoned about without
knowing what cached it.

**The engine calls the laser; it does not model it** (§4.2, P15). Stage 0 samples
``intensity_profile`` — the cycle-averaged ``<a^2>`` — and the explicit additional
``carrier_phase_four_gradient`` along each trajectory through the `LaserField` protocol;
it assumes no envelope formula, Gaussian shape, or spot sizes. The photon density follows
by inverting the same
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
from dataclasses import dataclass, field

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from ...io.bunch import Bunch, illumination_window, overlap_time_window
from ...io.laser import LaserField, fit_gaussian_paraxial
from ...io.units import C_CGS, E_ESU, HBAR_CGS, ME_CGS, SIGMA_T_CGS
from . import chunking
from .chunking import run_in_chunks


def _get_array_module(*arrays):
    """Get the array module (numpy or cupy) from input arrays."""
    # Try cupy's get_array_module first (works for both numpy and cupy arrays)
    if _HAS_CUPY:
        try:
            return cp.get_array_module(*arrays)
        except Exception:
            pass
    # Fallback: check if any array is a cupy array
    for arr in arrays:
        if arr is not None:
            if _HAS_CUPY and isinstance(arr, cp.ndarray):
                return cp
            # Check for numpy array
            if isinstance(arr, np.ndarray):
                return np
    return np

__all__ = [
    "TrajectorySamples",
    "TrajectoryDiagnostics",
    "ahat_from_shape",
    "integrate_trajectories",
    "photon_density_scale",
    "RELATIVE_VELOCITY",
    "BYTES_PER_PARTICLE_STEP",
    "ShapeTable",
    "Table",
    "SpectralMoments",
    "deposit_shape_table",
    "retarget_ahat",
    "query_spectral_moments",
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
    "rotated_laser_axes",
    "physical_transverse_axes",
    "compute_stokes_components",
    "stokes_parameters_vectorized",
    "bunch_stokes_parameters",
    "polarization_factor",
    "polarization_factor_vectorized",
    "relative_velocity",
    "doppler_factor_per_particle",
    "direction_doppler_factor",
    "observer_ponderomotive_factor",
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


def _incident_axis(theta_xz: float, theta_yz: float) -> np.ndarray:
    """Laser propagation vector in the pinned two-plane crossing convention."""
    return np.array([-math.sin(theta_xz) * math.cos(theta_yz), math.sin(theta_yz),
                     -math.cos(theta_xz) * math.cos(theta_yz)])


def direction_doppler_factor(theta_x, theta_y, theta_xz=0.0, theta_yz=0.0, *, k_hat=None):
    """Relative Doppler factor (1 - e dot n0)/(1 - n0_z), with beta=1 (DER013).

    The slopes broadcast normally; kernels unpack CGS inputs before calling this helper.
    Exact finite-speed Doppler remains available in ``doppler_factor_per_particle``.
    """
    n0 = _incident_axis(theta_xz, theta_yz) if k_hat is None else np.asarray(k_hat, dtype=float)
    if n0.shape != (3,) or not np.all(np.isfinite(n0)) or not np.isclose(np.dot(n0, n0), 1.0):
        raise ValueError("laser propagation direction must be a finite unit vector")
    nominal = 1.0 - n0[2]
    if nominal <= 0:
        raise ValueError("nominal Doppler factor must be positive; co-propagation is outside xigma's regime")
    xp = _get_array_module(theta_x, theta_y)
    tx, ty = xp.asarray(theta_x), xp.asarray(theta_y)
    encounter = 1.0 - (n0[0] * tx + n0[1] * ty + n0[2]) / xp.sqrt(1.0 + tx**2 + ty**2)
    return xp.maximum(0.0, encounter) / nominal


def observer_ponderomotive_factor(
    theta_e_x,
    theta_e_y,
    theta_x,
    theta_y,
    theta_xz=0.0,
    theta_yz=0.0,
    *,
    k_hat=None,
):
    """Exact coefficient ``(1 - n dot n0) / (1 - e dot n0)`` multiplying ``ahat``.

    ``e`` is the normalized electron direction represented by ``theta_e_x`` and
    ``theta_e_y``; ``n`` is the normalized observation direction represented by
    ``theta_x`` and ``theta_y``. All slope inputs broadcast under NumPy or CuPy. The
    co-propagating denominator is outside xigma's validity domain and is rejected rather
    than clipped.
    """
    n0 = _incident_axis(theta_xz, theta_yz) if k_hat is None else np.asarray(k_hat, dtype=float)
    if n0.shape != (3,) or not np.all(np.isfinite(n0)) or not np.isclose(np.dot(n0, n0), 1.0):
        raise ValueError("laser propagation direction must be a finite unit vector")

    xp = _get_array_module(theta_e_x, theta_e_y, theta_x, theta_y)
    te_x = xp.asarray(theta_e_x)
    te_y = xp.asarray(theta_e_y)
    tn_x = xp.asarray(theta_x)
    tn_y = xp.asarray(theta_y)
    electron_encounter = 1.0 - (
        n0[0] * te_x + n0[1] * te_y + n0[2]
    ) / xp.sqrt(1.0 + te_x**2 + te_y**2)
    singular_tolerance = 64.0 * np.finfo(float).eps
    if xp.any(electron_encounter <= singular_tolerance):
        raise ValueError(
            "electron-laser encounter factor is too small; co-propagation is outside xigma's regime"
        )
    observer_encounter = 1.0 - (
        n0[0] * tn_x + n0[1] * tn_y + n0[2]
    ) / xp.sqrt(1.0 + tn_x**2 + tn_y**2)
    return observer_encounter / electron_encounter


def doppler_factor_per_particle(
    gamma: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
    *,
    k_hat: np.ndarray | None = None,
) -> np.ndarray:
    """Per-particle Doppler factor relative to the nominal-axis approximation.

    This finite-speed diagnostic differs from production's beta=1 direction factor.
    The nominal-axis photon-energy factor is ``cos^2(alpha/2)``
    where ``cos(alpha) = cos(theta_xz) * cos(theta_yz)``. This function computes the
    exact per-particle factor ``(1 - v.n0_hat) / (1 + cos(alpha))`` for each electron,
    where ``v`` is the electron's lab-frame velocity vector and ``n0_hat`` is the
    laser propagation direction.

    Parameters
    ----------
    gamma : np.ndarray
        Per-particle Lorentz factors, shape (n_particles,).
    theta_x : np.ndarray
        Per-particle x divergence angles (rad), shape (n_particles,).
    theta_y : np.ndarray
        Per-particle y divergence angles (rad), shape (n_particles,).
    theta_xz : float
        Laser crossing angle in x-z plane (radians).
    theta_yz : float
        Laser crossing angle in y-z plane (radians).
    k_hat : np.ndarray | None
        Laser propagation unit vector (3,). If provided, used directly instead of
        computing from crossing angles.

    Returns
    -------
    np.ndarray
        Per-particle Doppler factors relative to nominal, shape (n_particles,).
        For head-on (theta_xz=theta_yz=0) and zero divergence, this equals
        ``(1 + beta) / 2`` -> 1 for ultra-relativistic beams.

    Notes
    -----
    The nominal-axis approximation uses the beam-axis velocity (theta_x=theta_y=0)
    and the laser propagation direction. The per-particle version accounts for each
    electron's individual divergence, which matters for beams with significant
    angular spread or large crossing angles.
    """
    gamma = np.asarray(gamma, dtype=float)
    theta_x = np.asarray(theta_x, dtype=float)
    theta_y = np.asarray(theta_y, dtype=float)

    if gamma.ndim != 1 or theta_x.ndim != 1 or theta_y.ndim != 1:
        raise ValueError("gamma, theta_x, theta_y must be 1D arrays")
    if not (gamma.shape == theta_x.shape == theta_y.shape):
        raise ValueError("gamma, theta_x, theta_y must have the same shape")

    # Electron velocity vector (normalized)
    # v = beta * (theta_x, theta_y, 1) / sqrt(1 + theta_x^2 + theta_y^2)
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
    norm = np.sqrt(1.0 + theta_x * theta_x + theta_y * theta_y)
    vx = beta * theta_x / norm
    vy = beta * theta_y / norm
    vz = beta / norm

    # Laser propagation direction n0_hat
    if k_hat is not None:
        n0 = np.asarray(k_hat, dtype=float)
        if n0.shape != (3,):
            raise ValueError("k_hat must have shape (3,)")
    else:
        cos_xz = math.cos(theta_xz)
        cos_yz = math.cos(theta_yz)
        sin_xz = math.sin(theta_xz)
        sin_yz = math.sin(theta_yz)
        n0 = np.array([
            -sin_xz * cos_yz,
            sin_yz,
            -cos_xz * cos_yz,
        ], dtype=float)

    # 1 - v.n0_hat
    v_dot_n0 = vx * n0[0] + vy * n0[1] + vz * n0[2]
    one_minus_v_dot_n0 = 1.0 - v_dot_n0

    # Nominal factor: 1 + cos(alpha) = 1 + cos(theta_xz) * cos(theta_yz)
    if k_hat is not None:
        # For arbitrary k_hat, the nominal factor is 1 - k_hat[2] (head-on beta=1)
        # But we need the general form. The nominal axis is the beam axis (0,0,1)
        # so nominal = 1 - (0,0,1).n0 = 1 - n0[2]
        nominal = 1.0 - n0[2]
    else:
        nominal = 1.0 + math.cos(theta_xz) * math.cos(theta_yz)

    return one_minus_v_dot_n0 / nominal


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
):
    """``Tr(U^T Xi U)`` for one electron and lab-frame observation direction.

    The field-free electron velocity is ``beta * (theta_x, theta_y, 1)`` normalized
    in the shared lab frame (DV006). The laser basis is rotated only once for the
    pulse crossing angle; it is never rotated separately for each electron.
    """
    return polarization_factor_vectorized(
        gamma, theta_x, theta_y,
        theta_x_obs, theta_y_obs, ellipticity, psi_pol, theta_xz, theta_yz,
    )


def rotated_laser_axes(
    psi_pol: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotated laser polarization unit vectors (e0, e1) for arbitrary crossing angles.

    Evaluates the exact 3D rotation ``R_y(theta_xz) @ R_x(theta_yz)`` applied to the
    unrotated laser basis ``[cos(psi_pol), sin(psi_pol), 0]`` and ``[-sin(psi_pol), cos(psi_pol), 0]``
    without small crossing angle approximations (DER005, DER007).
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
    e0 = R @ np.array([cp, sp, 0.0], dtype=float)
    e1 = R @ np.array([-sp, cp, 0.0], dtype=float)
    return e0, e1


def physical_transverse_axes(
    psi_pol: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical transverse dipole radiation unit vectors in the electron frame (DER012).

    Projects the rotated laser polarization basis onto the plane transverse to the
    electron beam axis, reflecting that relativistic longitudinal inertia gamma^3 * m
    suppresses longitudinal acceleration to O(alpha / gamma^2) ~ 10^-11.
    """
    e0, e1 = rotated_laser_axes(psi_pol=psi_pol, theta_xz=theta_xz, theta_yz=theta_yz)
    e0_t = np.array([e0[0], e0[1], 0.0], dtype=float)
    n0 = math.sqrt(e0_t[0] ** 2 + e0_t[1] ** 2)
    if n0 > 1e-12:
        e0_out = e0_t / n0
        # Gram-Schmidt orthonormal minor axis in the transverse plane
        e1_t = np.array([-e0_out[1], e0_out[0], 0.0], dtype=float)
        if e1[0] * e1_t[0] + e1[1] * e1_t[1] < 0.0:
            e1_t = -e1_t
        e1_out = e1_t
    else:
        e0_out = np.zeros(3, dtype=float)
        e1_out = np.zeros(3, dtype=float)

    return e0_out, e1_out


def compute_stokes_components(
    gamma,
    theta_ex,
    theta_ey,
    theta_x: float,
    theta_y: float,
    e0,
    e1,
    ellipticity: float = 0.0,
):
    """Compute Stokes parameters (I, Q, U, V) in the smooth laboratory observer basis (m_x, m_y).

    Evaluates the scattered-photon Stokes parameters in the smooth laboratory observer
    basis (m_x, m_y) defined by parallel transport of the fixed laboratory Cartesian axes
    from z0 to n along the connecting great circle (DER007). Placing the coordinate
    singularity at the backward pole -z0 avoids the on-axis 0/0 singularity and artificial
    2*phi vortex of the spherical meridian basis.

    Single-electron radiation satisfies purity I^2 = Q^2 + U^2 + V^2 identically for
    arbitrary crossing angles and transverse divergence.

    Parameters
    ----------
    gamma : array-like or float
        Electron Lorentz factor(s).
    theta_ex : array-like or float
        Electron transverse divergence x-component(s).
    theta_ey : array-like or float
        Electron transverse divergence y-component(s).
    theta_x : float
        Laboratory observation angle x-component.
    theta_y : float
        Laboratory observation angle y-component.
    e0 : array-like
        Rotated laser polarization unit vector along major axis (shape (3,)).
    e1 : array-like
        Rotated laser polarization unit vector along minor axis (shape (3,)).
    ellipticity : float
        Laser ellipticity in [-1, 1] (0 = linear, +/-1 = circular).

    Returns
    -------
    tuple of arrays
        Stokes parameters (I, Q, U, V) in the smooth laboratory observer basis.
    """
    xp = _get_array_module(gamma, theta_ex, theta_ey)
    no = math.sqrt(1.0 + theta_x**2 + theta_y**2)
    n = xp.array([theta_x, theta_y, 1.0], dtype=float) / no

    denom_m = 1.0 + n[2]
    mx = xp.array([1.0 - n[0]**2 / denom_m, -n[0] * n[1] / denom_m, -n[0]], dtype=float)
    my = xp.array([-n[0] * n[1] / denom_m, 1.0 - n[1]**2 / denom_m, -n[1]], dtype=float)

    gamma, theta_ex, theta_ey = xp.broadcast_arrays(
        xp.asarray(gamma, dtype=float),
        xp.asarray(theta_ex, dtype=float),
        xp.asarray(theta_ey, dtype=float),
    )
    valid_gamma = gamma >= 1.0
    gamma_safe = xp.where(valid_gamma, gamma, 1.0)
    gamma_sq = gamma_safe * gamma_safe
    beta = xp.sqrt(1.0 - 1.0 / gamma_sq)
    delta = 1.0 / (gamma_sq * (1.0 + beta))

    ve = xp.sqrt(1.0 + theta_ex**2 + theta_ey**2)

    delta_z = (
        (theta_ex - theta_x) * (theta_ex + theta_x)
        + (theta_ey - theta_y) * (theta_ey + theta_y)
    ) / (ve * no * (ve + no))
    delta_x = (theta_x - theta_ex) / no + theta_ex * delta_z
    delta_y = (theta_y - theta_ey) / no + theta_ey * delta_z

    d = delta + 0.5 * beta * (delta_x**2 + delta_y**2 + delta_z**2)

    ux = theta_ex / ve
    uy = theta_ey / ve
    uz = 1.0 / ve
    qx = (delta_x + delta * ux) / d
    qy = (delta_y + delta * uy) / d
    qz = (delta_z + delta * uz) / d

    # Local per-electron transverse projection (DER012):
    # Relativistic longitudinal inertia (gamma^3 * m) suppresses longitudinal acceleration
    # to O(alpha / 2*gamma^2) ~ 10^-11. The radiation dipole is transverse to each electron
    # velocity u to order 1/gamma^2.
    u_dot_e0 = ux * e0[0] + uy * e0[1] + uz * e0[2]
    p0x = e0[0] - u_dot_e0 * ux
    p0y = e0[1] - u_dot_e0 * uy
    p0z = e0[2] - u_dot_e0 * uz
    n0 = xp.sqrt(p0x**2 + p0y**2 + p0z**2)
    valid0 = n0 > 1e-12
    n0_safe = xp.where(valid0, n0, 1.0)
    e0px = xp.where(valid0, p0x / n0_safe, 0.0)
    e0py = xp.where(valid0, p0y / n0_safe, 0.0)
    e0pz = xp.where(valid0, p0z / n0_safe, 0.0)

    e1px = uy * e0pz - uz * e0py
    e1py = uz * e0px - ux * e0pz
    e1pz = ux * e0py - uy * e0px
    n1 = xp.sqrt(e1px**2 + e1py**2 + e1pz**2)
    valid1 = n1 > 1e-12
    n1_safe = xp.where(valid1, n1, 1.0)
    e1px = xp.where(valid1, e1px / n1_safe, 0.0)
    e1py = xp.where(valid1, e1py / n1_safe, 0.0)
    e1pz = xp.where(valid1, e1pz / n1_safe, 0.0)
    sign1 = xp.where(e1px * e1[0] + e1py * e1[1] + e1pz * e1[2] < 0.0, -1.0, 1.0)
    e1px *= sign1
    e1py *= sign1
    e1pz *= sign1

    a0 = n[0] * e0px + n[1] * e0py + n[2] * e0pz
    a1 = n[0] * e1px + n[1] * e1py + n[2] * e1pz

    u0x = qx * a0 - e0px
    u0y = qy * a0 - e0py
    u0z = qz * a0 - e0pz
    u1x = qx * a1 - e1px
    u1y = qy * a1 - e1py
    u1z = qz * a1 - e1pz

    U0x = u0x * mx[0] + u0y * mx[1] + u0z * mx[2]
    U0y = u0x * my[0] + u0y * my[1] + u0z * my[2]
    U1x = u1x * mx[0] + u1y * mx[1] + u1z * mx[2]
    U1y = u1x * my[0] + u1y * my[1] + u1z * my[2]

    eps2 = ellipticity * ellipticity
    xi00 = 1.0 / (1.0 + eps2)
    xi11 = eps2 / (1.0 + eps2)

    I = xi00 * (U0x**2 + U0y**2) + xi11 * (U1x**2 + U1y**2)
    Q = xi00 * (U0x**2 - U0y**2) + xi11 * (U1x**2 - U1y**2)
    U = 2.0 * (xi00 * U0x * U0y + xi11 * U1x * U1y)
    V = (-2.0 * ellipticity / (1.0 + eps2)) * (U0x * U1y - U1x * U0y)

    if xp.any(~valid_gamma):
        I = xp.where(valid_gamma, I, 0.0)
        Q = xp.where(valid_gamma, Q, 0.0)
        U = xp.where(valid_gamma, U, 0.0)
        V = xp.where(valid_gamma, V, 0.0)

    return I, Q, U, V


def stokes_parameters_vectorized(
    gamma: np.ndarray | float,
    theta_x: np.ndarray | float,
    theta_y: np.ndarray | float,
    theta_x_obs: float = 0.0,
    theta_y_obs: float = 0.0,
    ellipticity: float = 0.0,
    psi_pol: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Stokes parameters for an ensemble of electrons.

    Signature mirrors :func:`polarization_factor_vectorized` for drop-in convenience.
    """
    e0, e1 = rotated_laser_axes(psi_pol=psi_pol, theta_xz=theta_xz, theta_yz=theta_yz)
    return compute_stokes_components(
        gamma=gamma,
        theta_ex=theta_x,
        theta_ey=theta_y,
        theta_x=theta_x_obs,
        theta_y=theta_y_obs,
        e0=e0,
        e1=e1,
        ellipticity=ellipticity,
    )


def bunch_stokes_parameters(
    samples: TrajectorySamples,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    *,
    e0: np.ndarray | None = None,
    e1: np.ndarray | None = None,
    psi_pol: float = 0.0,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Incoherently sum per-particle Stokes vectors weighted by luminosity.

    ``theta_x`` and ``theta_y`` are the observer angles.  Supplying ``e0`` and
    ``e1`` preserves the explicit polarization basis API; otherwise the basis is
    derived from the polarization and crossing-angle parameters.
    """
    if samples.n_particles == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xp = _get_array_module(samples.gamma, samples.theta_x, samples.theta_y, samples.luminosity)

    if e0 is None or e1 is None:
        e0, e1 = rotated_laser_axes(psi_pol=psi_pol, theta_xz=theta_xz, theta_yz=theta_yz)

    I, Q, U, V = compute_stokes_components(
        samples.gamma,
        samples.theta_x,
        samples.theta_y,
        theta_x,
        theta_y,
        e0,
        e1,
        ellipticity=ellipticity,
    )

    weights = samples.luminosity
    Itot = float(xp.sum(weights * I))
    Qtot = float(xp.sum(weights * Q))
    Utot = float(xp.sum(weights * U))
    Vtot = float(xp.sum(weights * V))

    if Itot > 0.0:
        p_num = math.sqrt(max(0.0, Qtot * Qtot + Utot * Utot + Vtot * Vtot))
        P = min(1.0, p_num / Itot)
        chi = 0.5 * math.atan2(Utot, Qtot)
    else:
        P = 0.0
        chi = 0.0

    return Itot, Qtot, Utot, Vtot, P, chi


def polarization_factor_vectorized(
    gamma,
    theta_x,
    theta_y,
    theta_x_obs: float,
    theta_y_obs: float,
    ellipticity: float,
    psi_pol: float,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
):
    """Vectorized :func:`polarization_factor` for per-electron lab velocities.

    Evaluates the physical transverse dipole projection local to each electron's
    velocity vector (DER012).
    """
    xp = _get_array_module(gamma, theta_x, theta_y)
    e0, e1 = rotated_laser_axes(psi_pol=psi_pol, theta_xz=theta_xz, theta_yz=theta_yz)
    no = math.sqrt(1.0 + theta_x_obs**2 + theta_y_obs**2)
    n = xp.array([theta_x_obs, theta_y_obs, 1.0]) / no

    gamma, theta_x, theta_y = xp.broadcast_arrays(
        xp.asarray(gamma, dtype=float), xp.asarray(theta_x, dtype=float),
        xp.asarray(theta_y, dtype=float),
    )
    valid_gamma = gamma >= 1.0
    gamma_safe = xp.where(valid_gamma, gamma, 1.0)
    gamma_sq = gamma_safe * gamma_safe
    beta = xp.sqrt(1.0 - 1.0 / gamma_sq)
    delta = 1.0 / (gamma_sq * (1.0 + beta))

    ve = xp.sqrt(1.0 + theta_x**2 + theta_y**2)

    # Delta = n - u, with the nearly equal longitudinal component evaluated from
    # the slope norms rather than as ``1/no - 1/ve`` (RES069, RES070).
    delta_z = (
        (theta_x - theta_x_obs) * (theta_x + theta_x_obs)
        + (theta_y - theta_y_obs) * (theta_y + theta_y_obs)
    ) / (ve * no * (ve + no))
    delta_x = (theta_x_obs - theta_x) / no + theta_x * delta_z
    delta_y = (theta_y_obs - theta_y) / no + theta_y * delta_z

    # 1 - beta * u.n = delta + 0.5 * beta * |Delta|^2
    d = delta + 0.5 * beta * (delta_x**2 + delta_y**2 + delta_z**2)

    # n - beta*u = Delta + delta*u, divided by d
    ux = theta_x / ve
    uy = theta_y / ve
    uz = 1.0 / ve
    qx = (delta_x + delta * ux) / d
    qy = (delta_y + delta * uy) / d
    qz = (delta_z + delta * uz) / d

    # Local per-electron transverse projection (DER012):
    u_dot_e0 = ux * e0[0] + uy * e0[1] + uz * e0[2]
    p0x = e0[0] - u_dot_e0 * ux
    p0y = e0[1] - u_dot_e0 * uy
    p0z = e0[2] - u_dot_e0 * uz
    n0 = xp.sqrt(p0x**2 + p0y**2 + p0z**2)
    valid0 = n0 > 1e-12
    n0_safe = xp.where(valid0, n0, 1.0)
    e0px = xp.where(valid0, p0x / n0_safe, 0.0)
    e0py = xp.where(valid0, p0y / n0_safe, 0.0)
    e0pz = xp.where(valid0, p0z / n0_safe, 0.0)

    e1px = uy * e0pz - uz * e0py
    e1py = uz * e0px - ux * e0pz
    e1pz = ux * e0py - uy * e0px
    n1 = xp.sqrt(e1px**2 + e1py**2 + e1pz**2)
    valid1 = n1 > 1e-12
    n1_safe = xp.where(valid1, n1, 1.0)
    e1px = xp.where(valid1, e1px / n1_safe, 0.0)
    e1py = xp.where(valid1, e1py / n1_safe, 0.0)
    e1pz = xp.where(valid1, e1pz / n1_safe, 0.0)
    sign1 = xp.where(e1px * e1[0] + e1py * e1[1] + e1pz * e1[2] < 0.0, -1.0, 1.0)
    e1px *= sign1
    e1py *= sign1
    e1pz *= sign1

    a0 = n[0] * e0px + n[1] * e0py + n[2] * e0pz
    a1 = n[0] * e1px + n[1] * e1py + n[2] * e1pz

    u0x = qx * a0 - e0px
    u0y = qy * a0 - e0py
    u0z = qz * a0 - e0pz
    u1x = qx * a1 - e1px
    u1y = qy * a1 - e1py
    u1z = qz * a1 - e1pz

    norm0_sq = u0x * u0x + u0y * u0y + u0z * u0z
    norm1_sq = u1x * u1x + u1y * u1y + u1z * u1z

    eps2 = ellipticity * ellipticity
    xi00 = 1.0 / (1.0 + eps2)
    xi11 = eps2 / (1.0 + eps2)
    return xi00 * norm0_sq + xi11 * norm1_sq


#: Live bytes per (particle x step) in Stage 0's inner loop, for auto-chunking.
#: Calibrated on a GTX 1660 Ti (6 GB): 400,000
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
class TrajectoryDiagnostics:
    """Histogrammed emission-source overlap in laboratory time (s) and position (cm).

    Densities retain the photon mass inside the supplied edges (RES081).
    """

    t_edges: np.ndarray | None = None
    time_envelope: np.ndarray | None = None
    spatial_edges: tuple[np.ndarray, np.ndarray] | None = None
    spatial_envelope: np.ndarray | None = None


@dataclass(frozen=True)
class TrajectorySamples:
    """One sample per macroparticle, ready for Stage 1 deposition. CGS.

    ``gamma``/``theta_x``/``theta_y`` pass straight through from the bunch: the pusher is
    ballistic, so a particle's energy and angles do not change along its trajectory.

    ``luminosity`` is the plan's per-particle ``L`` — the photon weight this macroparticle
    deposits, integrated over its passage through the pulse. Summing it is the total yield.

    ``a0_shape`` is **not** an amplitude or the trajectory-averaged effective intensity
    ``ahat``. It is the normalized cycle-averaged-intensity shape
    ``integral(C r**2) / integral(C r)``, where ``r = <a**2>/<a**2>_peak``.
    :func:`ahat_from_shape` is the only route from here to ``ahat``.

    ``chirp_mean``, ``var_a_shape``, ``var_chirp``, and ``cov_a_chirp_shape`` are the
    strength-independent carrier/intensity moments defined by the chirped resonance
    model. The variance and covariance of physical ``q`` at another peak intensity follow
    from :meth:`retargeted_var_a` and :meth:`retargeted_cov_a_chirp`.

    ``intensity_peak`` is the peak **cycle-averaged** ``<a^2>``, not a peak ``a0``; nothing
    in this dataclass carries a polarization convention, because ``<a^2>`` at fixed pulse
    energy does not depend on one (RES054).

    It is one scalar per particle rather than a per-timestep distribution, and that is
    physics, not an optimization: in this weakly nonlinear regime the photon formation
    length spans the whole trajectory, so — unlike synchrotron radiation — the trajectory
    may **not** be split into independently radiating segments. Do not replace this with
    per-timestep emission amplitudes.
    """

    gamma: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    a0_shape: np.ndarray
    luminosity: np.ndarray
    intensity_peak: float
    n_steps: int
    chirp_mean: np.ndarray
    var_a_shape: np.ndarray
    var_chirp: np.ndarray
    cov_a_chirp_shape: np.ndarray
    diagnostics: TrajectoryDiagnostics | None = None

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

    def retargeted_var_a(self, intensity_peak: float) -> np.ndarray:
        """Trajectory variance of physical ``q`` at another peak ``<a^2>``."""
        return float(intensity_peak) ** 2 * self.var_a_shape

    def retargeted_cov_a_chirp(self, intensity_peak: float) -> np.ndarray:
        """Trajectory covariance of physical ``q`` and ``C`` at another peak."""
        return float(intensity_peak) * self.cov_a_chirp_shape

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
    t_edges: np.ndarray | None = None,
    spatial_edges: tuple[np.ndarray, np.ndarray] | None = None,
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
    answer) — a deliberate act, not something to inherit silently. Validation therefore
    states explicitly which window it exercises.

    Chunking is over particles, whose trajectories are independent, so the partition
    cannot change the answer.

    Optional ``t_edges`` (seconds) and ``spatial_edges`` (x/y, cm) request overlap
    histograms on fixed grids shared by every chunk. Bin-average densities are returned
    in ``diagnostics``; photons outside those grids are not redistributed.

    ``backend`` selects NumPy, CuPy, or automatic CUDA/CPU selection. Geometry windows
    stay on the host; trajectory/intensity evaluation and histogram reduction execute
    on the selected device. Each chunk returns host arrays before releasing device memory
    (RES083).
    """
    if n_steps < 1:
        raise ValueError(f"integrate_trajectories: n_steps must be >= 1, got {n_steps}")
    if window not in ("active_region", "illumination"):
        raise ValueError(
            f"integrate_trajectories: window must be 'active_region' or 'illumination', got {window!r}"
        )
    backend = _check_backend(backend)
    xp = cp if backend == "cupy" else np
    to_host = cp.asnumpy if backend == "cupy" else np.asarray

    def checked_edges(edges):
        values = np.array(edges, dtype=float, copy=True)
        if (values.ndim != 1 or values.size < 2 or not bool(np.all(np.isfinite(values)))
                or bool(np.any(np.diff(values) <= 0))):
            raise ValueError("diagnostic edges must be finite, strictly increasing 1D arrays")
        return values

    t_edges = None if t_edges is None else checked_edges(t_edges)
    if spatial_edges is not None:
        if len(spatial_edges) != 2:
            raise ValueError("spatial_edges must contain x and y edge arrays")
        spatial_edges = tuple(checked_edges(edges) for edges in spatial_edges)

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
    offsets = (xp.arange(n_steps) + 0.5) / n_steps

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
    if hasattr(laser, "focusing_axes"):
        k_hat, _, _ = laser.focusing_axes()
    elif hasattr(laser, "m") and hasattr(laser, "theta_xz") and hasattr(laser, "theta_yz"):
        k_hat = _incident_axis(laser.m("theta_xz"), laser.m("theta_yz"))
    else:
        metrics = fit_gaussian_paraxial(laser)
        k_hat = _incident_axis(metrics.m("theta_xz"), metrics.m("theta_yz"))
    # Use the same per-electron encounter factor as the resonance (DER013, RES082).
    rel_vel = (1.0 - k_hat[2]) * direction_doppler_factor(bunch.thx, bunch.thy, k_hat=k_hat)
    if np.any(rel_vel <= 64.0 * np.finfo(float).eps):
        raise ValueError(
            "electron-laser encounter factor is too small; co-propagation is outside xigma's regime"
        )
    rate = rel_vel * density_scale * C_CGS * SIGMA_T_CGS
    omega0 = float(laser.omega0())

    def integrate(first_index: int, last_index: int):
        sl = slice(first_index, last_index)
        local_span = xp.asarray(span[sl])
        local_rate = xp.asarray(rate[sl])
        local_encounter = xp.asarray(rel_vel[sl])
        local_weight = xp.asarray(weight[sl])
        times = xp.asarray(start[sl, None]) + offsets[None, :] * local_span[:, None]
        local_velocity = tuple(xp.asarray(speed[sl, None]) for speed in velocity)
        positions = tuple(
            xp.asarray(position[sl, None]) + speed * times
            for position, speed in zip((bunch.x, bunch.y, bunch.z), local_velocity)
        )
        # `<a^2>`, not `a0`: everything below is a functional of the cycle-averaged
        # intensity, polarization-agnostic at fixed pulse energy (RES054). Do not form `a0`
        # and square it here — that's where RES053's missing factor of two hid.
        intensity = xp.asarray(laser.intensity_profile(*positions, times))
        phase_gradient = laser.carrier_phase_four_gradient(*positions, times)
        if not isinstance(phase_gradient, (tuple, list)) or len(phase_gradient) != 4:
            raise TypeError(
                "carrier_phase_four_gradient must return (d_t, d_x, d_y, d_z)"
            )
        d_t, d_x, d_y, d_z = xp.broadcast_arrays(
            *(xp.asarray(component, dtype=float) for component in phase_gradient)
        )
        carrier_ratio = 1.0 + (
            d_t
            + local_velocity[0] * d_x
            + local_velocity[1] * d_y
            + local_velocity[2] * d_z
        ) / (omega0 * local_encounter[:, None])
        # The photon density an electron flies through, and the rate it scatters at.
        # In CGS this is simply flux x cross-section x time; no coordinate-normalization
        # Jacobian belongs here (RES015).
        dt = local_span / n_steps
        contributing = (intensity > 0.0) & (dt[:, None] > 0.0)
        if xp.any(contributing & ((carrier_ratio <= 0.0) | ~xp.isfinite(carrier_ratio))):
            raise ValueError(
                "encountered carrier phase ratio C must be finite and positive over contributing samples"
            )
        carrier_ratio = xp.where(contributing, carrier_ratio, 1.0)
        weighted_intensity = carrier_ratio * intensity
        luminosity = local_rate * local_weight * dt * xp.sum(weighted_intensity, axis=1)

        # `ratio` is the local intensity as a fraction of the pulse's peak. a0_shape is its
        # second moment over the trajectory, normalized by its first: the intensity an
        # electron *effectively* experiences, weighted by where it actually radiated. Being
        # a ratio of intensities, it is polarization-agnostic like everything else here.
        ratio = intensity / intensity_peak
        weighted_ratio = carrier_ratio * ratio
        moment_1 = xp.sum(weighted_ratio, axis=1)
        # A particle with no window saw nothing, whatever the envelope reads at the
        # anchor time above; its effective intensity is zero, not the value at t = 0.
        usable = (moment_1 > 0.0) & (dt > 0.0)
        safe_moment_1 = xp.maximum(moment_1, 1e-300)
        a0_shape = xp.where(
            usable, xp.sum(carrier_ratio * ratio**2, axis=1) / safe_moment_1, 0.0
        )
        chirp_mean = xp.where(
            usable, xp.sum(carrier_ratio**2 * ratio, axis=1) / safe_moment_1, 0.0
        )
        second_a = xp.sum(carrier_ratio * ratio**3, axis=1) / safe_moment_1
        second_chirp = xp.sum(carrier_ratio**3 * ratio, axis=1) / safe_moment_1
        var_a_shape_raw = xp.where(usable, second_a - a0_shape**2, 0.0)
        var_chirp_raw = xp.where(usable, second_chirp - chirp_mean**2, 0.0)

        def checked_variance(raw, second, mean_squared, name):
            scale = xp.maximum(xp.maximum(xp.abs(second), xp.abs(mean_squared)), 1.0)
            tolerance = 64.0 * np.finfo(float).eps * scale
            if xp.any(usable & (raw < -tolerance)):
                raise ValueError(f"{name} is materially negative")
            return xp.where(raw < 0.0, 0.0, raw)

        var_a_shape = checked_variance(
            var_a_shape_raw, second_a, a0_shape**2, "var_a_shape"
        )
        var_chirp = checked_variance(
            var_chirp_raw, second_chirp, chirp_mean**2, "var_chirp"
        )
        cov_a_chirp_shape = xp.where(
            usable,
            xp.sum(carrier_ratio**2 * ratio**2, axis=1) / safe_moment_1
            - a0_shape * chirp_mean,
            0.0,
        )
        time_mass = spatial_mass = None
        if t_edges is not None or spatial_edges is not None:
            contribution = (
                local_rate[:, None]
                * local_weight[:, None]
                * dt[:, None]
                * weighted_intensity
            )
            if t_edges is not None:
                # Direct bin sums avoid subtracting large cumulative masses in dim tails.
                index = xp.searchsorted(xp.asarray(t_edges), times.ravel(), side="right") - 1
                index = xp.where(times.ravel() == t_edges[-1], t_edges.size - 2, index)
                inside = (index >= 0) & (index < t_edges.size - 1)
                time_mass = xp.bincount(index[inside], weights=contribution.ravel()[inside],
                                        minlength=t_edges.size - 1)
            if spatial_edges is not None:
                x, y = positions[:2]
                spatial_mass = xp.histogram2d(x.ravel(), y.ravel(), bins=tuple(xp.asarray(edge) for edge in spatial_edges),
                                             weights=contribution.ravel())[0]
        return tuple(
            None if value is None else to_host(value)
            for value in (
                luminosity,
                a0_shape,
                chirp_mean,
                var_a_shape,
                var_chirp,
                cov_a_chirp_shape,
                time_mass,
                spatial_mass,
            )
        )

    parts = run_in_chunks(
        bunch.n_particles,
        integrate,
        chunk=chunk,
        # No ceiling: the measured chunk ceiling applies to the *spectrum* path's s-axis,
        # where per-launch overhead amortized by ~8-16. Stage 0 partitions
        # particles, where no such measurement exists, and inventing one would be the
        # cargo-culting the chunking module's own docstring warns the constants against.
        bytes_per_item=(BYTES_PER_PARTICLE_STEP + (32 if spatial_edges is not None else
                                                 8 if t_edges is not None else 0)) * n_steps,
        backend=backend,
    )
    # An empty bunch is reachable, not hypothetical: the prefilter discards every particle
    # for a mistimed pulse or a bunch far wider than the spot. `np.concatenate([])` raises,
    # which would make the prefilter turn a zero yield into an exception — the opposite of
    # the pure optimization §3.2 promises.
    if parts:
        luminosity = np.concatenate([part[0] for part in parts])
        a0_shape = np.concatenate([part[1] for part in parts])
        chirp_mean = np.concatenate([part[2] for part in parts])
        var_a_shape = np.concatenate([part[3] for part in parts])
        var_chirp = np.concatenate([part[4] for part in parts])
        cov_a_chirp_shape = np.concatenate([part[5] for part in parts])
    else:
        luminosity = np.zeros(0, dtype=np.float64)
        a0_shape = np.zeros(0, dtype=np.float64)
        chirp_mean = np.zeros(0, dtype=np.float64)
        var_a_shape = np.zeros(0, dtype=np.float64)
        var_chirp = np.zeros(0, dtype=np.float64)
        cov_a_chirp_shape = np.zeros(0, dtype=np.float64)

    diagnostics = None
    if t_edges is not None or spatial_edges is not None:
        time_envelope = spatial_envelope = None
        if t_edges is not None:
            mass = sum((part[6] for part in parts), np.zeros(t_edges.size - 1))
            time_envelope = mass / np.diff(t_edges)
        if spatial_edges is not None:
            x_edges, y_edges = spatial_edges
            mass = sum((part[7] for part in parts),
                       np.zeros((x_edges.size - 1, y_edges.size - 1)))
            spatial_envelope = mass / (np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :])
        diagnostics = TrajectoryDiagnostics(t_edges, time_envelope, spatial_edges, spatial_envelope)

    return TrajectorySamples(
        gamma=bunch.gamma,
        theta_x=bunch.thx,
        theta_y=bunch.thy,
        a0_shape=a0_shape,
        luminosity=luminosity,
        intensity_peak=intensity_peak,
        n_steps=n_steps,
        chirp_mean=chirp_mean,
        var_a_shape=var_a_shape,
        var_chirp=var_chirp,
        cov_a_chirp_shape=cov_a_chirp_shape,
        diagnostics=diagnostics,
    )


#: Default bin counts for :func:`deposit_shape_table`'s five axes, in ``(gamma, theta_x,
#: theta_y, a0_shape, chirp_mean)`` order. The raw-shape axis is deliberately fine — it is
#: bounded and peak-intensity-independent, so there is no dynamic-range reason to keep it
#: small the way the old direct-onto-``ahat`` deposit's fourth axis had to be.
#: Eight carrier-rate bins are an initial numerical setting, not a privileged physical
#: resolution; an exactly constant carrier rate always collapses to one bin.
DEFAULT_SHAPE_BINS = (48, 48, 48, 96, 8)

#: Defaults for :func:`retarget_ahat`'s fixed, non-uniform target grid, tuned against this
#: repo's scenario bank (RES032), independent of defaults tuned for other scenario banks.
DEFAULT_RETARGET_BINS = 32
DEFAULT_AHAT_MIN = 0.0
DEFAULT_AHAT_MAX = 0.5
DEFAULT_AHAT_DECADES = 1.0


def _uniform_edges(values, n_bins: int, margin: float, floor_zero: bool = False):
    """``n_bins + 1`` uniform edges spanning ``values``, padded by ``margin`` of the span.

    A degenerate span (every sample identical — a monoenergetic, zero-divergence beam is
    a real scenario, not a hypothetical one) would otherwise produce a zero-width grid
    that every sample lands exactly on the edge of; padded by ``margin`` of the value's
    own scale instead so the grid always has a real width.
    """
    xp = _get_array_module(values)
    lo, hi = float(xp.min(values)), float(xp.max(values))
    span = hi - lo
    pad = margin * span if span > 0.0 else margin * max(abs(lo), 1.0)
    lo, hi = lo - pad, hi + pad
    if floor_zero:
        lo = max(lo, 0.0)
    return xp.linspace(lo, hi, n_bins + 1)


def _validate_edges_and_shape(edges, H, name: str) -> None:
    """Shared structural check for :class:`ShapeTable` and :class:`Table`: ``H``'s shape
    matches the edge counts, and every axis's edges are strictly increasing. Says nothing
    about *uniform* spacing — `Table`'s ``ahat_edges`` deliberately is not (§4.2)."""
    xp = _get_array_module(*edges, H)
    expected = tuple(e.size - 1 for e in edges)
    if H.shape != expected:
        raise ValueError(f"{name}: H.shape {H.shape} does not match edge counts {expected}")
    for e in edges:
        if xp.any(xp.diff(e) <= 0.0):
            raise ValueError(f"{name}: edges must be strictly increasing")


@dataclass(frozen=True)
class ShapeTable:
    """Stage 1's output: a 5D photon-weight density over ``(gamma, theta_x, theta_y,
    a0_shape, chirp_mean)``. The nonlinear incidence coefficient is observer-dependent
    and therefore is not part of this deposited coordinate.

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
    chirp_edges: np.ndarray
    H: np.ndarray
    H_var_a_shape: np.ndarray
    H_var_chirp: np.ndarray
    H_cov_a_chirp_shape: np.ndarray
    total_weight: float
    scheme: str
    source_intensity_peak: float
    _chirp_eval_points: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        edges = (self.gamma_edges, self.theta_x_edges, self.theta_y_edges,
                 self.a0_shape_edges, self.chirp_edges)
        for name, channel in (
            ("H", self.H),
            ("H_var_a_shape", self.H_var_a_shape),
            ("H_var_chirp", self.H_var_chirp),
            ("H_cov_a_chirp_shape", self.H_cov_a_chirp_shape),
        ):
            _validate_edges_and_shape(edges, channel, f"ShapeTable.{name}")

    @property
    def chirp_centers(self):
        return 0.5 * (self.chirp_edges[:-1] + self.chirp_edges[1:])

    @property
    def chirp_eval_points(self):
        return self.chirp_centers if self._chirp_eval_points is None else self._chirp_eval_points

    @property
    def chirp_widths(self):
        xp = _get_array_module(self.chirp_edges)
        return xp.diff(self.chirp_edges)

    @property
    def bin_volume(self) -> float:
        """Cell volume, constant because every axis here is a uniform grid."""
        xp = _get_array_module(
            self.gamma_edges, self.theta_x_edges, self.theta_y_edges,
            self.a0_shape_edges, self.chirp_edges, self.H,
        )
        return float(
            (self.gamma_edges[-1] - self.gamma_edges[0])
            * (self.theta_x_edges[-1] - self.theta_x_edges[0])
            * (self.theta_y_edges[-1] - self.theta_y_edges[0])
            * (self.a0_shape_edges[-1] - self.a0_shape_edges[0])
            * (self.chirp_edges[-1] - self.chirp_edges[0])
            / math.prod(self.H.shape)
        )


@dataclass(frozen=True)
class Table:
    """Stage 2's input: a 5D photon-weight density over ``(gamma, theta_x, theta_y,
    ahat, chirp_mean)``, for one specific peak intensity.

    The axis stores raw ``ahat``; Stage 2 evaluates the observer-dependent nonlinear
    coefficient at query time. ``H`` is a **density** (weight per unit cell volume).
    Unlike `ShapeTable`, the ``ahat`` axis is generally
    **non-uniform** — :func:`retarget_ahat` builds it dense near
    ``ahat_max`` and coarse toward ``ahat_min`` (RES032), so there is no
    single scalar cell volume; :attr:`ahat_widths` and :attr:`gamma_theta_cell_area` are
    what :func:`spectrum_from_table` actually needs.
    """

    gamma_edges: np.ndarray
    theta_x_edges: np.ndarray
    theta_y_edges: np.ndarray
    ahat_edges: np.ndarray
    chirp_edges: np.ndarray
    H: np.ndarray
    H_var_a: np.ndarray
    H_var_chirp: np.ndarray
    H_cov_a_chirp: np.ndarray
    total_weight: float
    scheme: str
    # Precomputed evaluation points for the ahat axis (zeroth bin at 0 for sub-floor/linear mode).
    # If None, ordinary geometric bin centers are the evaluation points.
    _ahat_eval_points: np.ndarray | None = field(default=None, repr=False)
    _chirp_eval_points: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        edges = (self.gamma_edges, self.theta_x_edges, self.theta_y_edges,
                 self.ahat_edges, self.chirp_edges)
        for name, channel in (
            ("H", self.H),
            ("H_var_a", self.H_var_a),
            ("H_var_chirp", self.H_var_chirp),
            ("H_cov_a_chirp", self.H_cov_a_chirp),
        ):
            _validate_edges_and_shape(edges, channel, f"Table.{name}")

    @property
    def gamma_centers(self):
        xp = _get_array_module(self.gamma_edges)
        return 0.5 * (self.gamma_edges[:-1] + self.gamma_edges[1:])

    @property
    def theta_x_centers(self):
        xp = _get_array_module(self.theta_x_edges)
        return 0.5 * (self.theta_x_edges[:-1] + self.theta_x_edges[1:])

    @property
    def theta_y_centers(self):
        xp = _get_array_module(self.theta_y_edges)
        return 0.5 * (self.theta_y_edges[:-1] + self.theta_y_edges[1:])

    @property
    def ahat_centers(self):
        xp = _get_array_module(self.ahat_edges)
        return 0.5 * (self.ahat_edges[:-1] + self.ahat_edges[1:])

    @property
    def ahat_eval_points(self):
        """Evaluation points for the ahat axis.
        
        If precomputed evaluation points are stored (from retarget_ahat), use those.
        Otherwise, fall back to geometric centers (ahat_centers).
        
        The precomputed points handle:
        - n_bins=1 "ignore nonlinearity" case: single bin evaluated at 0
        - Explicit zeroth bin for sub-floor contributions (ahat_min > 0): first bin evaluated at 0
        """
        # Use precomputed evaluation points if available
        if self._ahat_eval_points is not None:
            return self._ahat_eval_points
        
        # Fallback: use geometric centers (no special handling without precomputed points)
        xp = _get_array_module(self.ahat_edges)
        return 0.5 * (self.ahat_edges[:-1] + self.ahat_edges[1:])

    @property
    def ahat_widths(self):
        """Per-bin ``ahat`` width, shape ``(n_ahat,)`` — non-uniform, unlike every other
        axis here, so this is an array rather than a scalar."""
        xp = _get_array_module(self.ahat_edges)
        return xp.diff(self.ahat_edges)

    @property
    def chirp_centers(self):
        return 0.5 * (self.chirp_edges[:-1] + self.chirp_edges[1:])

    @property
    def chirp_eval_points(self):
        return self.chirp_centers if self._chirp_eval_points is None else self._chirp_eval_points

    @property
    def chirp_widths(self):
        xp = _get_array_module(self.chirp_edges)
        return xp.diff(self.chirp_edges)

    @property
    def gamma_theta_cell_area(self) -> float:
        """The still-uniform ``theta_x * theta_y`` cell area (gamma is interpolated, not
        integrated over, in :func:`spectrum_from_table` — see
        :func:`_interp_gamma_channel`)."""
        xp = _get_array_module(self.theta_x_edges, self.theta_y_edges, self.H)
        return float(
            (self.theta_x_edges[-1] - self.theta_x_edges[0]) / self.H.shape[1]
            * (self.theta_y_edges[-1] - self.theta_y_edges[0]) / self.H.shape[2]
        )


def _cell_fractions(values, edges, n_bins: int):
    """Continuous cell coordinate of ``values`` in ``edges``, in units of one bin width."""
    xp = _get_array_module(values, edges)
    return (values - edges[0]) / (edges[-1] - edges[0]) * n_bins


def _deposit_nearest(coords, weight, n_bins):
    xp = _get_array_module(*coords, weight)
    idx = [xp.clip(xp.floor(c).astype(xp.int64), 0, n - 1) for c, n in zip(coords, n_bins)]
    flat = xp.ravel_multi_index(idx, n_bins)
    return xp.bincount(flat, weights=weight, minlength=math.prod(n_bins)).reshape(n_bins)


def _deposit_cic(coords, weight, n_bins):
    """Cloud-in-cell: each sample splits its weight over its 16 neighbouring cells.

    Cell-centred convention (§4.2): a sample's continuous coordinate is
    shifted by ``-0.5`` so it interpolates between cell *centres*. ``edge='clamp'``
    always — overflow folds into the boundary cell rather than discarding weight, which
    is what keeps a CIC deposit's total exactly equal to a nearest deposit's for the same
    samples (both conserve weight; only where it lands differs).
    """
    xp = _get_array_module(*coords, weight)
    n_axes = len(coords)
    shifted = [c - 0.5 for c in coords]
    low = [xp.floor(s).astype(xp.int64) for s in shifted]
    frac = [s - lo for s, lo in zip(shifted, low)]

    flat_size = math.prod(n_bins)
    H_flat = xp.zeros(flat_size, dtype=xp.float64)
    for corner in itertools.product((0, 1), repeat=n_axes):
        idx = []
        w = weight
        for axis, bit in enumerate(corner):
            i = xp.clip(low[axis] + bit, 0, n_bins[axis] - 1)
            f = frac[axis] if bit else (1.0 - frac[axis])
            idx.append(i)
            w = w * f
        flat = xp.ravel_multi_index(idx, n_bins)
        H_flat += xp.bincount(flat, weights=w, minlength=flat_size)
    return H_flat.reshape(n_bins)


def deposit_shape_table(
    samples: TrajectorySamples,
    *,
    n_bins: tuple[int, int, int, int, int] = DEFAULT_SHAPE_BINS,
    scheme: str = "nearest",
    margin: float = 0.02,
    backend: str = "numpy",
    chunk: int | None = None,
) -> ShapeTable:
    """Stage 1: bin samples into the 5D ``(a0_shape, chirp_mean)`` table channels.

    Incidence and observation geometry are deliberately absent from the nonlinear axis;
    Stage 2 evaluates the exact observer-dependent coefficient for each query.

    ``scheme`` is ``"nearest"`` (one cell per sample) or ``"cic"`` (cloud-in-cell, 32
    neighbours per sample) — both conserve total weight exactly.

    ``backend`` selects deposition independently of the host input arrays. Particle
    chunks share fixed host-derived edges; device deposits return host masses before
    accumulation, retaining the NumPy ShapeTable contract (RES083).
    """
    if scheme not in ("nearest", "cic"):
        raise ValueError(f"deposit_shape_table: scheme must be 'nearest' or 'cic', got {scheme!r}")

    backend = _check_backend(backend)
    xp = cp if backend == "cupy" else np
    to_host = cp.asnumpy if backend == "cupy" else np.asarray

    redshift_shape = np.asarray(samples.a0_shape)
    chirp_mean = np.asarray(samples.chirp_mean)
    if len(n_bins) != 5:
        raise ValueError(f"deposit_shape_table: n_bins must have five entries, got {n_bins!r}")
    chirp_bins = 1 if np.all(chirp_mean == chirp_mean[0]) else n_bins[4]
    actual_bins = (*n_bins[:4], chirp_bins)
    gamma_edges = _uniform_edges(samples.gamma, n_bins[0], margin)
    theta_x_edges = _uniform_edges(samples.theta_x, n_bins[1], margin)
    theta_y_edges = _uniform_edges(samples.theta_y, n_bins[2], margin)
    a0_shape_edges = _uniform_edges(redshift_shape, n_bins[3], margin, floor_zero=True)
    chirp_edges = _uniform_edges(chirp_mean, chirp_bins, margin)
    edges = (gamma_edges, theta_x_edges, theta_y_edges, a0_shape_edges, chirp_edges)

    channels_raw = [np.zeros(actual_bins, dtype=float) for _ in range(4)]
    deposit = _deposit_nearest if scheme == "nearest" else _deposit_cic

    def deposit_chunk(start, stop):
        coords = tuple(_cell_fractions(xp.asarray(values[start:stop]), edge, n)
                       for values, edge, n in zip(
                           (samples.gamma, samples.theta_x, samples.theta_y,
                            redshift_shape, chirp_mean), edges, actual_bins))
        luminosity = xp.asarray(samples.luminosity[start:stop])
        weights = (
            luminosity,
            luminosity * xp.asarray(samples.var_a_shape[start:stop]),
            luminosity * xp.asarray(samples.var_chirp[start:stop]),
            luminosity * xp.asarray(samples.cov_a_chirp_shape[start:stop]),
        )
        masses = [to_host(deposit(coords, weight, actual_bins)) for weight in weights]
        # Commit only after device work/transfer succeeds, so an OOM retry cannot double-count.
        for channel, mass in zip(channels_raw, masses):
            channel += mass

    # Budget particle coordinates, CIC indices/fractions and corner temporaries;
    # fixed table buffers must still fit at the smallest retry size.
    run_in_chunks(samples.n_particles, deposit_chunk, chunk=chunk,
                  bytes_per_item=256, backend=backend)
    bin_volume = math.prod(float(e[-1] - e[0]) for e in edges) / math.prod(actual_bins)
    H_raw, H_var_a_raw, H_var_chirp_raw, H_cov_raw = channels_raw
    return ShapeTable(
        gamma_edges=gamma_edges,
        theta_x_edges=theta_x_edges,
        theta_y_edges=theta_y_edges,
        a0_shape_edges=a0_shape_edges,
        chirp_edges=chirp_edges,
        H=H_raw / bin_volume,
        H_var_a_shape=H_var_a_raw / bin_volume,
        H_var_chirp=H_var_chirp_raw / bin_volume,
        H_cov_a_chirp_shape=H_cov_raw / bin_volume,
        total_weight=float(H_raw.sum()),
        scheme=scheme,
        source_intensity_peak=samples.intensity_peak,
        _chirp_eval_points=(np.asarray([chirp_mean[0]]) if chirp_bins == 1 else None),
    )


def _ahat_target_edges(ahat_min: float, ahat_max: float, n_bins: int, decades: float):
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

    Special case: when ``n_bins == 1``, this returns a single bin spanning ``[0, ahat_max]``
    to support the "ignore nonlinearity" mode where all spectrum calculations evaluate at
    ``ahat = 0``. The evaluation point for this bin is 0, not the midpoint.
    """
    if ahat_max <= ahat_min:
        raise ValueError(f"_ahat_target_edges: ahat_max ({ahat_max}) must exceed ahat_min ({ahat_min})")
    if decades <= 0.0:
        raise ValueError(f"_ahat_target_edges: decades must be positive, got {decades}")
    if n_bins <= 0:
        raise ValueError(f"_ahat_target_edges: n_bins must be positive, got {n_bins}")

    xp = np  # This function creates new arrays, use numpy as base

    # Special case: n_bins == 1 means "ignore nonlinearity" — single bin evaluated at ahat=0
    if n_bins == 1:
        return xp.array([0.0, ahat_max])

    i = xp.arange(n_bins + 1)
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
    """Stage 1.5: conservative regrid of a `ShapeTable`'s raw ``a0_shape`` axis.

    The fixed, non-uniform target coordinate is raw ``ahat`` for one specific peak
    cycle-averaged intensity ``<a^2>`` (RES032, supersedes
    RES028; RES054 for why the parameter is an intensity rather than an amplitude).

    Cheap and independent of ``n_particles`` — a ``shape_table.a0_shape_edges.size x
    n_bins``-sized tensordot, not a re-deposit — so a `Collision` can cache the shape
    deposit once and retarget many pulse strengths from it.

    Conservative (mass-preserving) under the same piecewise-uniform-density assumption
    deposition itself makes, onto :func:`_ahat_target_edges`'s non-uniform target law. The
    deposited mass is rescaled by ``intensity_peak / shape_table.source_intensity_peak``
    (:meth:`TrajectorySamples.retargeted_luminosity`'s relation), since the requested
    intensity is generally not the one Stage 0 ran at.

    When ``n_bins == 1``, the returned table has a single bin spanning ``[0, ahat_max]``
    evaluated at ``ahat = 0`` — this is the "ignore nonlinearity" mode.

    When ``n_bins > 1``, an explicit zeroth bin ``[0, ahat_min]`` is prepended to catch
    sub-floor contributions; this bin is evaluated at ``ahat = 0``. The remaining bins
    follow the standard non-uniform grid from ``ahat_min`` to ``ahat_max``.
    """
    if ahat_max <= ahat_min:
        raise ValueError(f"retarget_ahat: ahat_max ({ahat_max}) must exceed ahat_min ({ahat_min})")
    if n_bins <= 0:
        raise ValueError(f"retarget_ahat: n_bins must be positive, got {n_bins}")
    intensity_peak = float(intensity_peak)

    # Get array module from shape_table arrays
    xp = _get_array_module(
        shape_table.gamma_edges, shape_table.theta_x_edges, shape_table.theta_y_edges,
        shape_table.a0_shape_edges, shape_table.chirp_edges, shape_table.H
    )

    # Exact: a0_shape is strength-independent, so the axis transform is a pure scale.
    source_edges = ahat_from_shape(shape_table.a0_shape_edges, intensity_peak)
    target_edges = _ahat_target_edges(ahat_min, ahat_max, n_bins, decades)

    # Handle the target grid construction:
    # 1. n_bins == 1: target_edges is [0, ahat_max] — single bin evaluated at 0
    #    (ignore nonlinearity mode)
    # 2. n_bins > 1 and ahat_min > 0: prepend explicit zeroth bin [0, ahat_min]
    #    for sub-floor contributions, evaluated at 0
    # 3. n_bins > 1 and ahat_min == 0: use target_edges as-is (first bin starts at 0)
    if n_bins == 1:
        # Single bin [0, ahat_max] — everything maps here, evaluated at 0
        # Extend edges for overlap: [-inf, ahat_max] and [0, +inf]
        edges_ext = target_edges.copy()
        edges_ext[0] = -xp.inf
        edges_ext[-1] = xp.inf
    elif ahat_min > 0.0:
        # Prepend explicit zeroth bin [0, ahat_min] for sub-floor contributions
        # The original target_edges starts at ahat_min; we add 0 at the front
        target_edges = xp.concatenate([xp.array([0.0]), target_edges])
        # Extend edges for overlap: [-inf, ahat_min, ..., ahat_max] and [0, ahat_min, ..., +inf]
        edges_ext = target_edges.copy()
        edges_ext[0] = -xp.inf
        edges_ext[-1] = xp.inf
    else:
        # ahat_min == 0: target_edges already starts at 0, no need to prepend
        # Extend edges for overlap: [-inf, 0, ..., ahat_max] and [0, ..., +inf]
        edges_ext = target_edges.copy()
        edges_ext[0] = -xp.inf
        edges_ext[-1] = xp.inf

    src_lo, src_hi = source_edges[:-1], source_edges[1:]
    src_width = src_hi - src_lo
    tgt_lo, tgt_hi = edges_ext[:-1], edges_ext[1:]

    lo = xp.maximum(src_lo[:, None], tgt_lo[None, :])
    hi = xp.minimum(src_hi[:, None], tgt_hi[None, :])
    overlap = xp.clip(hi - lo, 0.0, None)
    # W[i, j]: fraction of source bin i's mass assigned to target bin j.
    W = overlap / xp.clip(src_width, 1e-300, None)[:, None]

    # The source a0_shape axis stays uniform in this design — deposit_shape_table only
    # ever builds it via _uniform_edges — so a single scalar width is exact here, unlike
    # a scalar width is safe only because the source is uniform; it is not a general
    # non-uniform-axis operation.
    da_source = shape_table.a0_shape_edges[1] - shape_table.a0_shape_edges[0]
    luminosity_rescale = intensity_peak / shape_table.source_intensity_peak

    channel_scales = (
        luminosity_rescale,
        luminosity_rescale * intensity_peak**2,
        luminosity_rescale,
        luminosity_rescale * intensity_peak,
    )
    source_channels = (
        shape_table.H,
        shape_table.H_var_a_shape,
        shape_table.H_var_chirp,
        shape_table.H_cov_a_chirp_shape,
    )
    mass_targets = [
        xp.moveaxis(
            xp.tensordot(channel * da_source * scale, W, axes=([3], [0])), -1, 3
        )
        for channel, scale in zip(source_channels, channel_scales)
    ]
    target_width = xp.diff(target_edges)
    target_channels = [mass / target_width[None, None, None, :, None]
                       for mass in mass_targets]
    H_target, H_var_a_target, H_var_chirp_target, H_cov_target = target_channels

    # Truncate trailing ahat bins the rescaled source never reaches: their mass is exactly
    # zero (W's overlap is exactly zero where no source bin overlaps a target bin), so
    # dropping them changes nothing spectrum_from_table's cell sum would have computed —
    # only how many always-zero terms it evaluates. One-sided: the floor bin (index 0)
    # always catches whatever folded below ahat_min, so only the top can be empty.
    marginal = H_target.sum(axis=(0, 1, 2, 4))
    populated = xp.nonzero(marginal > 0.0)[0]
    last = int(populated[-1]) if populated.size else 0
    H_target = H_target[:, :, :, : last + 1, :]
    H_var_a_target = H_var_a_target[:, :, :, : last + 1, :]
    H_var_chirp_target = H_var_chirp_target[:, :, :, : last + 1, :]
    H_cov_target = H_cov_target[:, :, :, : last + 1, :]
    target_edges = target_edges[: last + 2]

    # Compute evaluation points for the ahat axis
    # For n_bins=1: single bin evaluated at 0
    # For n_bins>1 with explicit zeroth bin (ahat_min > 0): first bin evaluated at 0
    # For n_bins>1 with ahat_min=0: use centers (no explicit zeroth bin)
    if n_bins == 1:
        ahat_eval_points = xp.array([0.0])
    elif ahat_min > 0.0:
        # Explicit zeroth bin was prepended: first evaluation point is 0
        centers = 0.5 * (target_edges[:-1] + target_edges[1:])
        ahat_eval_points = xp.concatenate([xp.array([0.0]), centers[1:]])
    else:
        # No explicit zeroth bin: use centers
        ahat_eval_points = 0.5 * (target_edges[:-1] + target_edges[1:])

    return Table(
        gamma_edges=shape_table.gamma_edges,
        theta_x_edges=shape_table.theta_x_edges,
        theta_y_edges=shape_table.theta_y_edges,
        ahat_edges=target_edges,
        chirp_edges=shape_table.chirp_edges,
        H=H_target,
        H_var_a=H_var_a_target,
        H_var_chirp=H_var_chirp_target,
        H_cov_a_chirp=H_cov_target,
        total_weight=shape_table.total_weight * luminosity_rescale,
        scheme=shape_table.scheme,
        _ahat_eval_points=ahat_eval_points,
        _chirp_eval_points=shape_table._chirp_eval_points,
    )


#: The §9.1 constant (§4.2, RES033): the kernel math is pi-free (``coef = 1.5``);
#: ``1/(2 pi)`` is the correction RES026 derived is missing from the paper's
#: cross-section, applied here and nowhere else (`references/delta.py` applies the same
#: correction to its own transcription).
#:
#: **Derived and confirmed, never fitted (P14). Do not adjust it to make a check pass** —
#: if a check disagrees, that is a physics finding to escalate (§0), not a number to tune.
KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2.0 * math.pi)


@dataclass(frozen=True)
class SpectralMoments:
    """Raw CPU pull-query channels on a common normalized-energy grid."""

    s: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(values) for values in (self.s, self.rho0, self.rho1, self.rho2)
        )
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("SpectralMoments arrays must be one-dimensional")
        if len({values.shape for values in arrays}) != 1:
            raise ValueError("SpectralMoments arrays must have the same shape")


def _inverse_resonance_gamma_sq(A, K, r_sq, s):
    """Analytical inverse ``Gamma**2 = A / (K/s - r**2)`` and its support."""
    xp = _get_array_module(A, K, r_sq)
    inv_base = K / s - r_sq
    valid = inv_base > 0.0
    return A / xp.where(valid, inv_base, 1.0), valid


def _inverse_resonance_jacobian(K, gamma, A, s):
    """Absolute ``dGamma/ds`` at the nominal resonance root."""
    return K * gamma**3 / (2.0 * A * s**2)


def _interp_gamma_channel(table: Table, channel, g):
    """One co-shaped table channel interpolated along gamma at query points ``g``.

    ``g`` carries one query value per ``(theta_x, theta_y, ahat, chirp)`` cell — the resonance
    condition inverted at that cell's own angle and ahat (§4.2) — so this is not a single
    1D interpolation but ``n_theta_x * n_theta_y * n_ahat * n_chirp`` of them, batched. A query
    outside the tabulated gamma range gets zero: the bunch's gamma distribution simply did
    not populate a resonance there, which is physical, not a boundary artefact to
    extrapolate past.
    """
    xp = _get_array_module(
        table.gamma_edges, table.theta_x_edges, table.theta_y_edges,
        table.ahat_edges, table.chirp_edges, channel, g,
    )
    gc = table.gamma_centers
    in_range = (g >= gc[0]) & (g <= gc[-1])
    idx = xp.clip(xp.searchsorted(gc, g) - 1, 0, len(gc) - 2)
    idx_hi = idx + 1
    frac = xp.where(in_range, (g - gc[idx]) / (gc[idx_hi] - gc[idx]), 0.0)

    tx_idx, ty_idx, a_idx, c_idx = xp.meshgrid(
        xp.arange(table.H.shape[1]), xp.arange(table.H.shape[2]),
        xp.arange(table.H.shape[3]), xp.arange(table.H.shape[4]), indexing="ij"
    )
    lo = channel[idx, tx_idx, ty_idx, a_idx, c_idx]
    hi = channel[idx_hi, tx_idx, ty_idx, a_idx, c_idx]
    return xp.where(in_range, lo * (1.0 - frac) + hi * frac, 0.0)


def query_spectral_moments(
    table: Table,
    theta_x: float,
    theta_y: float,
    s,
    *,
    psi_pol: float = 0.0,
    ellipticity: float = 0.0,
    theta_xz: float = 0.0,
    theta_yz: float = 0.0,
) -> SpectralMoments:
    """Return the raw ``rho0``, ``rho1``, and ``rho2`` CPU pull-query channels.

    A direct grid quadrature over Stage 1's table, independent of the GPU importance
    sampler (RES029) — sums the table's own ``(theta_x, theta_y, ahat, chirp)`` cells, inverting the
    resonance condition at each cell to find the gamma an electron there would need to
    radiate a photon of energy ``s`` toward ``(theta_x, theta_y)``, and interpolates ``H``
    at that gamma (:func:`_interp_gamma_channel`).

    ``g``/``prefac`` are recomputed inside the ahat loop implicitly — this function never
    factors ahat out of the resonance condition. The table stores raw ``ahat`` and the
    exact observer-dependent coefficient is evaluated from the same electron and laser
    geometry as the direction Doppler factor.

    The polarization factor now includes ellipticity and crossing angle effects per DER006,
    replacing the head-on linear factor ``cos^2 psi``.
    """
    xp = _get_array_module(
        table.gamma_edges, table.theta_x_edges, table.theta_y_edges,
        table.ahat_edges, table.chirp_edges, table.H,
    )
    s_arr = xp.atleast_1d(xp.asarray(s, dtype=float))
    tx_c = table.theta_x_centers[:, None, None, None]
    ty_c = table.theta_y_centers[None, :, None, None]
    # Use evaluation points (zeroth bin at ahat=0 for sub-floor/linear mode) instead of centers
    a_c = table.ahat_eval_points[None, None, :, None]
    c_c = table.chirp_eval_points[None, None, None, :]
    if bool(xp.any(c_c <= 0.0)):
        raise ValueError("query_spectral_moments requires positive carrier-rate evaluation points")

    r_sq = (tx_c - theta_x) ** 2 + (ty_c - theta_y) ** 2
    theta_cell_area = table.gamma_theta_cell_area
    # ahat is generally non-uniform (§4.2, RES032), so its width is a per-bin array — folded
    # into the sum below rather than factored out as a scalar the way theta's still is.
    cell_widths = (
        table.ahat_widths[None, None, :, None]
        * table.chirp_widths[None, None, None, :]
    )

    # Direction-dependent resonance and Jacobian at beta=1 (DER013, RES082).
    D_rel = direction_doppler_factor(tx_c, ty_c, theta_xz, theta_yz)
    Q = observer_ponderomotive_factor(
        tx_c, ty_c, theta_x, theta_y, theta_xz, theta_yz
    )
    A = 1.0 + Q * a_c
    K = D_rel * c_c

    rho0 = xp.zeros(s_arr.shape[0])
    rho1 = xp.zeros(s_arr.shape[0])
    rho2 = xp.zeros(s_arr.shape[0])
    for k, s_val in enumerate(s_arr):
        # s <= 0 is not a resonance to invert (the formula's own 1/s and 1/s**2 factors
        # are singular there) — zero photon energy is zero photons, and the raw channels
        # are already zero, so there is nothing to compute.
        if s_val <= 0.0:
            continue
        # A resonance exists only where inv_base > 0 (g_sq would otherwise be negative or
        # infinite); `valid` gates every quantity built from it, including the gamma this
        # cell would query `H` at, so an invalid cell contributes exactly zero rather than
        # a stray extrapolated lookup.
        g_sq, valid = _inverse_resonance_gamma_sq(A, K, r_sq, s_val)
        g = xp.where(valid, xp.sqrt(g_sq), 0.0)
        gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq) ** 2
        B = A + g_sq * r_sq

        # New polarization factor from DER006 (replaces a_fac = 1 - 4*cos^2(psi)*r^2*g^2*gth_sq_inv)
        pol_factor = polarization_factor_vectorized(
            g, tx_c, ty_c, theta_x, theta_y, ellipticity, psi_pol, theta_xz, theta_yz
        )
        jacobian = _inverse_resonance_jacobian(K, g, A, s_val)
        prefac = xp.where(
            valid,
            2.0 * KERNEL_NORMALIZATION_CONSTANT
            * pol_factor * g**2 * gth_sq_inv * jacobian,
            0.0,
        )
        H_val = _interp_gamma_channel(table, table.H, g)
        H_var_a = _interp_gamma_channel(table, table.H_var_a, g)
        H_var_chirp = _interp_gamma_channel(table, table.H_var_chirp, g)
        H_cov = _interp_gamma_channel(table, table.H_cov_a_chirp, g)
        W1 = s_val * (
            Q**2 * H_var_a / B**2
            - Q * H_cov / (B * c_c)
        )
        W2 = s_val**2 * (
            H_var_chirp / c_c**2
            + Q**2 * H_var_a / B**2
            - 2.0 * Q * H_cov / (B * c_c)
        )
        scale = theta_cell_area
        rho0[k] = scale * float(xp.sum(H_val * prefac * cell_widths))
        rho1[k] = scale * float(xp.sum(W1 * prefac * cell_widths))
        rho2[k] = scale * float(xp.sum(W2 * prefac * cell_widths))
    return SpectralMoments(
        s=np.asarray(s_arr),
        rho0=np.asarray(rho0),
        rho1=np.asarray(rho1),
        rho2=np.asarray(rho2),
    )


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
):
    """Delta-line Stage-2 spectrum; raw finite-line moments use
    :func:`query_spectral_moments`."""
    moments = query_spectral_moments(
        table, theta_x, theta_y, s,
        psi_pol=psi_pol,
        ellipticity=ellipticity,
        theta_xz=theta_xz,
        theta_yz=theta_yz,
    )
    return moments.rho0[0] if np.ndim(s) == 0 else moments.rho0


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
    rings: int = 32,
    subsampling: int = 32,
) -> np.ndarray:
    """Stage 2: :func:`spectrum_from_table` evaluated over a grid of observation points.

    Feeds `OutputKind.COLLIMATED_SPECTRUM` (§3.4). Shape
    ``(len(theta_x_grid), len(theta_y_grid), len(s))``.

    Dispatches to the CuPy ring/annulus importance-sampling rawkernel when ``backend``
    is ``'cupy'`` or ``'auto'`` (and CuPy + CUDA are available). Falls back to the
    NumPy brute-force grid quadrature otherwise.
    """
    try:
        psi_pol = float(psi_pol)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 psi_pol must be a real scalar") from exc
    if not math.isfinite(psi_pol):
        raise ValueError("Stage-2 psi_pol must be finite")
    selected_backend = stage2_backend(
        backend, ellipticity=ellipticity, theta_xz=theta_xz, theta_yz=theta_yz,
    )
    if selected_backend == "cupy":
        from .spectrum_sampler import calculate_angular_spectrum_gpu
        return calculate_angular_spectrum_gpu(
            table, theta_x_grid, theta_y_grid, s,
            psi_pol=psi_pol, ellipticity=ellipticity,
            theta_xz=theta_xz, theta_yz=theta_yz,
            rings=rings,
            subsampling=subsampling,
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
    try:
        ellipticity = float(ellipticity)
        theta_xz = float(theta_xz)
        theta_yz = float(theta_yz)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 polarization geometry must be real scalars") from exc
    if not all(math.isfinite(value) for value in (ellipticity, theta_xz, theta_yz)):
        raise ValueError("Stage-2 polarization geometry angles and ellipticity must be finite")
    if not 0.0 <= ellipticity <= 1.0:
        raise ValueError("Stage-2 ellipticity must be in [0, 1]")
    if backend == "cupy":
        from .spectrum_sampler import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError(
                "angular_spectrum_from_table(backend='cupy') requested but CuPy or a CUDA device is not available."
            )
        return "cupy"
    if backend == "auto":
        from .spectrum_sampler import is_gpu_available
        if is_gpu_available():
            return "cupy"
    return "numpy"


def angle_integrated_spectrum(
    samples: TrajectorySamples, s, *, theta_xz: float = 0.0, theta_yz: float = 0.0,
) -> np.ndarray:
    """``dN/ds``, angle-integrated in closed form — `Collision.spectrum`'s actual output.

    The same linear-Compton shape as
    `gammaforge.validation.references.delta.single_electron_spectrum`
    (``1.5 * (1 - 2y(1-y))`` with each electron's ``y = s / (D * gamma**2)``), reimplemented here
    rather than imported. delta exists to check this engine independently (§4.5); if it
    imported its own reference formula back from the engine it checks, or this engine
    imported from `validation`, the check would be circular in the first case and invert
    the package's dependency direction in the second. This table-free linear shape is an
    intentional approximation rather than Stage 1/2's nonlinear resonance; where ahat
    matters it is not the full physics (mirrors delta's own caveat).
    The direction-dependent energy scale and its density Jacobian preserve each
    electron's luminosity (DER013, RES082).
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
            doppler = direction_doppler_factor(samples.theta_x[start:stop], samples.theta_y[start:stop],
                                               theta_xz, theta_yz)
            gamma_squared = (doppler * samples.gamma[start:stop] ** 2)[:, None]
            safe_edge = np.where(gamma_squared > 0, gamma_squared, 1.0)
            y = energy[None, :] / safe_edge
            shape = np.where((gamma_squared <= 0) | (y < 0.0) | (y > 1.0),
                             0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
            return np.sum(samples.luminosity[start:stop, None] * shape / safe_edge, axis=0)

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
    rings: int = 32,
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
        rings=rings,
    )
    dN_ds = np.trapezoid(np.trapezoid(cube, ty, axis=1), tx, axis=0)
    n_photons = float(np.trapezoid(dN_ds, s_centers))
    return cube, dN_ds, n_photons


def _check_backend(backend: str) -> str:
    """Resolve Stage-0/1 execution explicitly; public stage arrays stay on the host."""
    if backend not in ("numpy", "cupy", "auto"):
        raise ValueError(f"backend must be 'numpy', 'cupy', or 'auto', got {backend!r}")
    if backend == "numpy":
        return backend
    from .spectrum_sampler import is_gpu_available
    if is_gpu_available():
        return "cupy"
    if backend == "cupy":
        raise RuntimeError("CuPy or a CUDA device is not available for Stage 0/1")
    return "numpy"

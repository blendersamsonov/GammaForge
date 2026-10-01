"""Laser field representation.

Two things live here, and the split is the point:

* :class:`LaserField` — the **sampling contract engines are typed against**. Vectorized,
  lab-frame methods (``intensity_profile``, ``carrier_phase_four_gradient``,
  ``a0_profile``, ``field``, ``active_region``) plus the reference ``omega0``.
  Quasi-monochromatic engines (xigma, delta) consume field sampling alongside physical
  carrier and polarization invariants; analytical explicitly requires `GaussianParaxialLaser`
  (RES067).
* :class:`GaussianParaxialLaser` — today's primary implementation. It owns the four
  geometry angles of §2.2 and an elliptical, astigmatic paraxial-Gaussian model. How it
  gets from its own head-on-frame parameterization to lab coordinates is entirely its own
  business; the protocol says nothing about it.

**Geometry (§2.2, pinned).** ``psi_focus``/``psi_pol`` are defined as if the collision
were exactly head-on (``k0 = -z``), measured from the electron's x-axis; the whole
configuration is then carried into the real 3D geometry by

    R = R_y(theta_xz) @ R_x(theta_yz)

— extrinsic: tilt ``theta_yz`` about the lab x-axis first, then ``theta_xz`` about the
lab y-axis. The roll about k-hat is *determined by that composition order*, not a free
parameter, which is exactly why the order is pinned rather than left to the caller.

**Two quantities, and which one is physics** (RES054). :meth:`intensity_profile`
returns the cycle-averaged ``<a^2>``; :meth:`a0_profile` returns the peak amplitude ``a0``.
Engines want the first. ``<a^2>`` at fixed pulse energy is **the same for every polarization
state** — an elliptical pulse's ``a0`` is smaller by ``sqrt(2C)`` and its cycle average
larger by ``C`` — so nothing that depends on it needs to know how the pulse is polarized.
``a0`` is a *reported* number carrying a convention (the linear-equivalent peak amplitude),
and re-deriving physics from it means round-tripping through that convention.

**Physics deliberately not implemented here (P14c).** One parameter is now fully applied and
one is still partial; both say so out loud rather than passing unremarked:

* ``ellipticity`` is **applied exactly to the photon yield and the mean nonlinear red-shift**
  — by being irrelevant to them, per the invariance above — and to the **angle-resolved**
  kernel (DER004 §1.2, DER006): the polarization factor is
  ``(cos^2 psi + eps^2 sin^2 psi)/(1 + eps^2)`` in the head-on limit, and the full
  DER006 expression with crossing angle. :data:`ELLIPTICITY_IS_NOOP` is flipped to False
  now that this lands.
* ``theta_xz``/``theta_yz`` enter the emission kernel in three places that must land
  together (RES034): (1) the relative-velocity factor ``1 + beta cos(theta_xz) cos(theta_yz)``,
  (2) the resonance frequency and photon energy conversion gain ``cos^2(alpha/2)`` with
  ``cos(alpha) = cos(theta_xz) cos(theta_yz)``, and (3) the polarization structure
  ``u_i.u_j`` gains the ``v.e_i`` terms (DER005 §2.3, DER006). :data:`EMISSION_IS_HEAD_ON`
  is flipped to False now that all three pieces are implemented.
  The rotation is still applied everywhere the pulse is sampled (geometry), and now the
  emission physics downstream matches.
  Xigma's per-electron direction extension uses beta=1 in flux, resonance and Jacobian
  (RES082); the nominal factor remains its shared energy-coordinate conversion.

:func:`validate` no longer warns on these; the markers below remain as the one-line greps
for "the derivation landed".
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol, runtime_checkable

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


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


from .units import (
    C_CGS,
    E_ESU,
    HBAR_CGS,
    ME_CGS,
    Quantity,
    WidthConvention,
    as_canonical_quantity,
)

__all__ = [
    "LaserField",
    "ActiveRegion",
    "GaussianParaxialLaser",
    "PulseTrainParaxialLaser",
    "fit_gaussian_paraxial",
    "rotation_matrix",
    "lab_frame_axes",
    "validate",
    "ELLIPTICITY_IS_NOOP",
    "EMISSION_IS_HEAD_ON",
]

#: ``ellipticity`` is applied to **everything that depends on the cycle-averaged
#: intensity** — the photon yield and the mean nonlinear red-shift (`ahat`) — as of
#: RES054, and to the **angle-resolved** kernel as of DER004 §1.2/DER006:
#: the polarization factor is now
#: ``(cos^2 psi + eps^2 sin^2 psi)/(1 + eps^2)`` in the head-on limit,

#: Live bytes per (particle x step) in Stage 0's inner loop, for auto-chunking.
#: The reason the split is *exactly* here, rather than being an arbitrary staging: the
#: cycle-average factor cancels out of every angle-integrated quantity (see
#: :meth:`GaussianParaxialLaser.intensity_profile`), so those never needed a derivation
#: at all. Only the angle-resolved kernel, which contracts the polarization vectors
#: against an observation direction, can tell an ellipse from a line.
ELLIPTICITY_IS_NOOP = False

#: §9.3 is resolved: the crossing angle enters the emission kernel in three places
#: that must land together (RES034): (1) the relative-velocity factor
#: ``1 + beta cos(theta_xz) cos(theta_yz)``, (2) the resonance frequency
#: and photon energy conversion gain ``cos^2(alpha/2)`` with
#: ``cos(alpha) = cos(theta_xz) cos(theta_yz)``, and (3) the polarization
#: structure ``u_i.u_j`` gains the ``v.e_i`` terms (DER005 §2.3, DER006).
#: This marker is flipped to False now that all three pieces are implemented.
#: The same "flip to False when it lands" grep as :data:`ELLIPTICITY_IS_NOOP`.
EMISSION_IS_HEAD_ON = False


# ---------------------------------------------------------------------------
# The sampling contract
# ---------------------------------------------------------------------------
@runtime_checkable
class LaserField(Protocol):
    """What an engine may assume about a laser (§3.3/P15).

    The sampling methods are **lab-frame** and **array-callable**: pass numpy (or cupy)
    arrays of positions/times and get arrays back, broadcasting normally. Engines call
    these; they never re-implement field physics themselves.

    *Implementation note for future non-analytic sources:* a numba CPU path generally
    cannot jit an arbitrary external callable, so an implementation must be usable either
    vectorized outside a jitted loop or via a lattice it builds once. That is a
    requirement on implementations, not a reason to move sampling back into engines.
    """

    def intensity_profile(self, x, y, z, t):
        """Cycle-averaged normalized intensity ``<a^2>`` at ``(x, y, z, t)``.

        **The method engines should consume.** It is polarization-agnostic (see
        `GaussianParaxialLaser.intensity_profile`), so a consumer of this never needs to
        know or apply a cycle-average factor.
        """
        ...

    def carrier_phase_four_gradient(self, x, y, z, t):
        """Lab-frame derivatives of the additional carrier phase ``delta Phi``.

        The full carrier phase convention is
        ``Phi_L = omega0 * (t - n0 dot r / c) + delta Phi``. Return the four
        ordinary derivatives ``(d_t delta Phi, d_x delta Phi, d_y delta Phi,
        d_z delta Phi)`` in that order, in ``rad/s`` and ``rad/cm`` respectively,
        broadcasting over NumPy or CuPy inputs.
        This is an explicit correction beyond the reference plane-wave carrier;
        it must not implicitly include an implementation's envelope, Gouy, or
        wavefront-curvature phase.
        """
        ...

    def omega0(self) -> float:
        """Reference carrier angular frequency in radians per second."""
        ...

    def a0_profile(self, x, y, z, t):
        """Peak normalized vector-potential envelope at ``(x, y, z, t)``.

        Reported/diagnostic: ``a0`` is convention-dependent (linear-equivalent peak
        amplitude here). Prefer :meth:`intensity_profile` for anything physical.
        """
        ...

    def field(self, x, y, z, t):
        """Period-resolved field at ``(x, y, z, t)``, as lab-frame vector components."""
        ...

    def active_region(self, threshold: float) -> "ActiveRegion":
        """Bounding space-time region where the envelope exceeds ``threshold * a0_peak``."""
        ...


@dataclass(frozen=True)
class ActiveRegion:
    """Conservative bounding region for the §3.2 prefilter.

    A lab-frame point ``(r, t)`` is *possibly* inside the pulse iff it lies within the
    region's local radius of the propagation axis **and** within ``half_length`` of the
    pulse centre measured along that axis in the co-moving sense::

        u        = (r - origin) . axis
        inside  <=>  |r - origin - u*axis| <= radius + radius_slope*|u|
                     and  |u - c*t| <= half_length

    **The region is a cone, not a cylinder**, and that is load-bearing rather than
    decorative. A pulse diverges: far from focus its spot — and with it the transverse
    extent in which ``a0`` clears the threshold — grows without bound. A fixed radius is
    conservative only near focus, so a bunch longer than the Rayleigh range would have had
    particles discarded that the expanded pulse still reaches. ``radius_slope`` bounds that
    growth linearly, which is an upper bound on the hyperbolic truth
    (``sqrt(1 + a^2) <= 1 + |a|``) and therefore still over-inclusive.

    Over-inclusiveness is the whole contract: the prefilter is a pure optimization that
    must never discard a particle which would have contributed (§3.2), so every
    approximation made in deriving this region errs towards keeping particles.

    ``a0_peak`` is carried along because the threshold that produced this region is a
    fraction of it, and callers reporting what was filtered need the absolute scale.
    """

    axis: np.ndarray  # unit vector, lab frame
    origin: np.ndarray  # lab-frame point the pulse centre passes through at t = 0
    radius: float  # cm, the region's half-width where it crosses the focal plane
    radius_slope: float  # cm per cm, how fast that half-width grows away from focus
    half_length: float  # cm
    threshold: float
    a0_peak: float

    def radius_at(self, u):
        """The region's transverse half-width at longitudinal coordinate ``u``."""
        return self.radius + self.radius_slope * np.abs(u)

    def contains(self, x, y, z, t):
        """Boolean mask: which ``(x, y, z, t)`` points fall inside this region."""
        rx = np.asarray(x) - self.origin[0]
        ry = np.asarray(y) - self.origin[1]
        rz = np.asarray(z) - self.origin[2]
        u = rx * self.axis[0] + ry * self.axis[1] + rz * self.axis[2]
        perp2 = np.maximum(rx * rx + ry * ry + rz * rz - u * u, 0.0)
        return (perp2 <= self.radius_at(u) ** 2) & (
            np.abs(u - C_CGS * np.asarray(t)) <= self.half_length
        )


# ---------------------------------------------------------------------------
# Geometry (§2.2)
# ---------------------------------------------------------------------------
def rotation_matrix(theta_xz: float, theta_yz: float) -> np.ndarray:
    """The pinned lab-frame rotation ``R = R_y(theta_xz) @ R_x(theta_yz)`` (§2.2).

    Carries the head-on configuration (``k0 = -z``, focusing/polarization axes measured
    from the electron x-axis) into the actual 3D geometry. Composition order is what fixes
    the roll about k-hat, so it is pinned here and nowhere else.
    """
    cx, sx = math.cos(theta_yz), math.sin(theta_yz)
    cy, sy = math.cos(theta_xz), math.sin(theta_xz)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return r_y @ r_x


def lab_frame_axes(
    theta_xz: float, theta_yz: float, psi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lab-frame ``(k_hat, e1, e2)`` for a transverse-plane rotation ``psi``.

    ``k_hat`` is the propagation direction (head-on ``-z``, rotated by :func:`rotation_matrix`).
    ``e1``/``e2`` are the transverse axes obtained by rotating the transported x-axis by
    ``psi`` in the plane perpendicular to ``k_hat`` — used for both the focusing axes
    (``psi_focus``) and the polarization axes (``psi_pol``), which is why this takes a
    generic ``psi`` rather than being written twice.
    """
    rot = rotation_matrix(theta_xz, theta_yz)
    k_hat = rot @ np.array([0.0, 0.0, -1.0])
    x_hat = rot @ np.array([1.0, 0.0, 0.0])
    y_hat = rot @ np.array([0.0, 1.0, 0.0])
    c, s = math.cos(psi), math.sin(psi)
    return k_hat, c * x_hat + s * y_hat, -s * x_hat + c * y_hat


# ---------------------------------------------------------------------------
# Temporal Envelope Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class TemporalEnvelope(Protocol):
    """Protocol for temporal envelopes in phase time.

    Phase time τ = φ(x,y,z,t) / ω₀ is the optical phase divided by carrier frequency.
    This makes the envelope follow the actual paraxial phase structure including
    Gouy phase and wavefront curvature.
    """

    def envelope(self, phase_time: float | np.ndarray, xp: ModuleType) -> float | np.ndarray:
        """Evaluate envelope at given phase time(s).

        Parameters
        ----------
        phase_time : float or array
            Phase time τ = φ/ω₀ in seconds.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        float or array
            Envelope value(s), normalized such that ∫|envelope|² dτ = 1.
        """
        ...

    def phase_time_width(self) -> float:
        """RMS width of the envelope in phase time (seconds)."""
        ...

    def peak_value(self, xp: ModuleType) -> float:
        """Peak value of the envelope (for active_region bounding)."""
        ...


@dataclass(frozen=True)
class GaussianTemporalEnvelope:
    """Simple Gaussian envelope in phase time.

    This reproduces the current GaussianParaxialLaser behavior when used with
    the full paraxial phase.
    """

    duration: Quantity  # RMS duration in phase time

    UNITS = {"duration": "s"}
    LIGHT_TIME_FIELDS = frozenset({"duration"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "duration",
            as_canonical_quantity(self.duration, "s", "duration", light_time=True),
        )
        if self.m("duration") <= 0.0:
            raise ValueError(f"GaussianTemporalEnvelope: duration must be > 0, got {self.m('duration')!r}")

    def m(self, name: str) -> float:
        return float(getattr(self, name).magnitude)

    def envelope(self, phase_time, xp):
        """Gaussian envelope: exp(-τ²/2σ²) / √(2π)σ"""
        sigma_t = self.m("duration")
        norm = 1.0 / (xp.sqrt(2.0 * xp.pi) * sigma_t)
        return norm * xp.exp(-0.5 * (phase_time / sigma_t) ** 2)

    def phase_time_width(self) -> float:
        return self.m("duration")

    def peak_value(self, xp) -> float:
        sigma_t = self.m("duration")
        return 1.0 / (xp.sqrt(2.0 * xp.pi) * sigma_t)


@dataclass(frozen=True)
class PulseTrainTemporalEnvelope:
    """Train of Gaussian sub-pulses in phase time.

    Reproduces PulseTrainParaxialLaser behavior with phase-aware envelope.
    """

    subpulse_duration: Quantity  # RMS duration of one sub-pulse
    repetition_period: Quantity  # Time between sub-pulses
    n_subpulses: int = 10

    UNITS = {"subpulse_duration": "s", "repetition_period": "s"}
    LIGHT_TIME_FIELDS = frozenset({"subpulse_duration", "repetition_period"})

    def __post_init__(self) -> None:
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(getattr(self, name), unit, name, light_time=True),
            )
        if (
            isinstance(self.n_subpulses, bool)
            or not isinstance(self.n_subpulses, numbers.Integral)
            or self.n_subpulses < 1
        ):
            raise ValueError(f"PulseTrainTemporalEnvelope: n_subpulses must be int >= 1, got {self.n_subpulses!r}")
        object.__setattr__(self, "n_subpulses", int(self.n_subpulses))

        for name in ("subpulse_duration", "repetition_period"):
            if self.m(name) <= 0.0:
                raise ValueError(f"PulseTrainTemporalEnvelope: {name} must be > 0, got {self.m(name)!r}")

    def m(self, name: str) -> float:
        return float(getattr(self, name).magnitude)

    def subpulse_delays(self) -> np.ndarray:
        """Temporal offsets of sub-pulses relative to train center (seconds)."""
        k = np.arange(1, self.n_subpulses + 1, dtype=float)
        return (k - 0.5 * (self.n_subpulses + 1)) * self.m("repetition_period")

    def envelope(self, phase_time, xp):
        """Sum of Gaussian sub-pulses in phase time."""
        sigma_t = self.m("subpulse_duration")
        delays = xp.asarray(self.subpulse_delays())

        # phase_time and delays broadcast: (..., n_subpulses)
        phase_time = xp.asarray(phase_time)
        diff = xp.expand_dims(phase_time, -1) - delays
        longitudinal = xp.sum(xp.exp(-0.5 * (diff / sigma_t) ** 2), axis=-1) / (
            xp.sqrt(2.0 * xp.pi) * sigma_t * self.n_subpulses
        )
        return longitudinal

    def phase_time_width(self) -> float:
        # For a single sub-pulse, the width is the sub-pulse duration.
        # For a train, approximate as n_subpulses * repetition_period.
        if self.n_subpulses == 1:
            return self.m("subpulse_duration")
        return self.m("repetition_period") * self.n_subpulses

    def peak_value(self, xp) -> float:
        # Peak occurs at center of central sub-pulse
        sigma_t = self.m("subpulse_duration")
        return 1.0 / (xp.sqrt(2.0 * xp.pi) * sigma_t)


# ---------------------------------------------------------------------------
# Separable Paraxial Laser Base Class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeparableParaxialLaser:
    """Paraxial beam × phase-aware temporal envelope.

    The photon density factorizes as:
        density(x,y,z,t) = transverse(xi1,xi2,u) × temporal(φ(x,y,z,t)/ω₀)

    where φ is the full paraxial phase including Gouy and curvature.
    This makes the envelope follow the actual optical phase structure.
    """

    # Spatial (paraxial beam)
    pulse_energy: Quantity
    wavelength: Quantity
    sigma_x: Quantity
    sigma_y: Quantity
    z_fx: Quantity = Quantity(0.0, "cm")
    z_fy: Quantity = Quantity(0.0, "cm")
    psi_focus: Quantity = Quantity(0.0, "rad")

    # Temporal envelope (pluggable) - must be set by subclass in __post_init__
    temporal_envelope: TemporalEnvelope = field(default=None, repr=False)

    # Geometry
    x_off: Quantity = Quantity(0.0, "cm")
    y_off: Quantity = Quantity(0.0, "cm")
    t_off: Quantity = Quantity(0.0, "s")
    theta_xz: Quantity = Quantity(0.0, "rad")
    theta_yz: Quantity = Quantity(0.0, "rad")
    psi_pol: Quantity = Quantity(0.0, "rad")
    ellipticity: float = 0.0
    beta_ff: float = 0.0

    width_convention = WidthConvention.SIGMA_INTENSITY_RMS

    UNITS = {
        "pulse_energy": "erg",
        "wavelength": "cm",
        "sigma_x": "cm",
        "sigma_y": "cm",
        "z_fx": "cm",
        "z_fy": "cm",
        "x_off": "cm",
        "y_off": "cm",
        "t_off": "s",
        "theta_xz": "rad",
        "theta_yz": "rad",
        "psi_focus": "rad",
        "psi_pol": "rad",
    }

    LIGHT_TIME_FIELDS = frozenset({"t_off"})

    def __post_init__(self) -> None:
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(
                    getattr(self, name), unit, name, light_time=name in self.LIGHT_TIME_FIELDS
                ),
            )
        # Validate temporal envelope
        if not isinstance(self.temporal_envelope, TemporalEnvelope):
            raise TypeError(f"temporal_envelope must implement TemporalEnvelope protocol, got {type(self.temporal_envelope)}")

        for name in ("ellipticity", "beta_ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                raise ValueError(f"SeparableParaxialLaser: {name} must be a finite scalar, got {value!r}")
        for name in self.UNITS:
            if not math.isfinite(self.m(name)):
                raise ValueError(f"SeparableParaxialLaser: {name} must be finite, got {getattr(self, name)!r}")
        for name in ("pulse_energy", "wavelength", "sigma_x", "sigma_y"):
            if self.m(name) <= 0.0:
                raise ValueError(f"SeparableParaxialLaser: {name} must be > 0, got {self.m(name)!r}")
        if self.beta_ff <= -1.0:
            raise ValueError("SeparableParaxialLaser: beta_ff must be > -1 (Rayleigh range scales as 1 + beta_ff)")
        if not 0.0 <= self.ellipticity <= 1.0:
            raise ValueError("SeparableParaxialLaser: ellipticity must be in [0, 1]")

    def m(self, name: str) -> float:
        """Magnitude of a dimensioned field in its canonical CGS unit."""
        return float(getattr(self, name).magnitude)

    # -- geometry -----------------------------------------------------------
    def focusing_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lab-frame ``(k_hat, f1, f2)``: propagation direction and focusing axes."""
        return lab_frame_axes(self.m("theta_xz"), self.m("theta_yz"), self.m("psi_focus"))

    def polarization_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lab-frame ``(k_hat, p1, p2)``: propagation direction and polarization axes."""
        return lab_frame_axes(self.m("theta_xz"), self.m("theta_yz"), self.m("psi_pol"))

    # -- descriptive scalars ------------------------------------------------
    def omega0(self) -> float:
        """Central angular frequency, rad/s."""
        return 2.0 * math.pi * C_CGS / self.m("wavelength")

    def photon_energy(self) -> float:
        """Central photon energy, erg."""
        return HBAR_CGS * self.omega0()

    def n_photons(self) -> float:
        """Photons in the pulse: pulse energy / central photon energy."""
        return self.m("pulse_energy") / self.photon_energy()

    def rayleigh_x(self) -> float:
        """Rayleigh range of focusing axis 1, cm."""
        return 4.0 * math.pi * self.m("sigma_x") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def rayleigh_y(self) -> float:
        return 4.0 * math.pi * self.m("sigma_y") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def spot_sizes(self, u_spot):
        """RMS intensity spot sizes ``(s1, s2)`` at longitudinal position ``u_spot``."""
        s1 = self.m("sigma_x") * np.sqrt(1.0 + ((u_spot - self.m("z_fx")) / self.rayleigh_x()) ** 2)
        s2 = self.m("sigma_y") * np.sqrt(1.0 + ((u_spot - self.m("z_fy")) / self.rayleigh_y()) ** 2)
        return s1, s2

    # -- phase calculation --------------------------------------------------
    def _paraxial_phase(self, xi1, xi2, u, u_spot, ct, xp):
        """Full paraxial phase φ = k₀(u-ct) - Gouy + curvature."""
        k0 = 2.0 * xp.pi / self.m("wavelength")

        du1 = u_spot - self.m("z_fx")
        du2 = u_spot - self.m("z_fy")
        zr1, zr2 = self.rayleigh_x(), self.rayleigh_y()
        gouy = 0.5 * (xp.arctan2(du1, zr1) + xp.arctan2(du2, zr2))
        inv_r1 = du1 / (du1**2 + zr1**2)
        inv_r2 = du2 / (du2**2 + zr2**2)
        phase = k0 * (u - ct) - gouy + 0.5 * k0 * (xi1**2 * inv_r1 + xi2**2 * inv_r2)
        return phase

    def _phase_time(self, xi1, xi2, u, u_spot, ct, xp):
        """Phase time τ = φ/ω₀."""
        return self._paraxial_phase(xi1, xi2, u, u_spot, ct, xp) / self.omega0()

    # -- local coordinates --------------------------------------------------
    def _local_coordinates(self, x, y, z, t):
        """Lab ``(x, y, z, t)`` → ``(xi1, xi2, u, u_spot, ct)`` in pulse frame."""
        k_hat, f1, f2 = self.focusing_axes()
        xp = _get_array_module(x, y, z, t)
        rx = xp.asarray(x, dtype=float) - self.m("x_off")
        ry = xp.asarray(y, dtype=float) - self.m("y_off")
        rz = xp.asarray(z, dtype=float)
        u = rx * k_hat[0] + ry * k_hat[1] + rz * k_hat[2]
        xi1 = rx * f1[0] + ry * f1[1] + rz * f1[2]
        xi2 = rx * f2[0] + ry * f2[1] + rz * f2[2]
        ct = C_CGS * (xp.asarray(t, dtype=float) - self.m("t_off"))
        u_spot = u + self.beta_ff * ct
        return xi1, xi2, u, u_spot, ct

    # -- photon density (separable) -----------------------------------------
    def photon_density(self, x, y, z, t):
        """Photon density = transverse(xi1,xi2,u) × temporal(phase_time).

        Normalized to integrate to 1 over all space at fixed time.
        """
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        xp = _get_array_module(xi1, xi2, u, u_spot, ct)

        # Transverse Gaussian
        s1, s2 = self.spot_sizes(u_spot)
        transverse = (1.0 / (2.0 * xp.pi * s1 * s2)) * xp.exp(
            -0.5 * ((xi1 / s1) ** 2 + (xi2 / s2) ** 2)
        )

        # Temporal envelope in phase time (normalized to 1 over phase_time).
        # The Jacobian du/d(phase_time) = C_CGS, so we divide by C_CGS to
        # normalize the integral over u to 1.
        phase_time = self._phase_time(xi1, xi2, u, u_spot, ct, xp)
        temporal = self.temporal_envelope.envelope(phase_time, xp) / C_CGS

        return transverse * temporal

    # -- LaserField contract ------------------------------------------------
    def intensity_profile(self, x, y, z, t):
        """Cycle-averaged normalized intensity ``<a^2>``."""
        density = self.photon_density(x, y, z, t)
        xp = _get_array_module(density)
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * xp.pi * self.m("pulse_energy") * density

    def carrier_phase_four_gradient(self, x, y, z, t):
        """Zero additional carrier-phase gradient for the current unchirped fields.

        The paraxial phase used by :meth:`field` and the temporal envelope is
        deliberately excluded from this API.
        """
        xp = _get_array_module(x, y, z, t)
        shape = xp.broadcast_arrays(
            xp.asarray(x), xp.asarray(y), xp.asarray(z), xp.asarray(t)
        )[0].shape
        zero = xp.zeros(shape, dtype=float)
        return zero, zero, zero, zero

    def a0_profile(self, x, y, z, t):
        """Period-averaged normalized vector-potential envelope."""
        return self._a0_from_density(self.photon_density(x, y, z, t))

    def _a0_from_density(self, density):
        xp = _get_array_module(density)
        e0 = xp.sqrt(8.0 * xp.pi * self.m("pulse_energy") * xp.asarray(density, dtype=float))
        return E_ESU * e0 / (ME_CGS * C_CGS * self.omega0())

    def field(self, x, y, z, t):
        """Period-resolved normalized vector potential, lab-frame components."""
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        xp = _get_array_module(xi1, xi2, u, u_spot, ct)
        s1, s2 = self.spot_sizes(u_spot)
        k0 = 2.0 * xp.pi / self.m("wavelength")

        du1 = u_spot - self.m("z_fx")
        du2 = u_spot - self.m("z_fy")
        zr1, zr2 = self.rayleigh_x(), self.rayleigh_y()
        gouy = 0.5 * (xp.arctan2(du1, zr1) + xp.arctan2(du2, zr2))
        inv_r1 = du1 / (du1**2 + zr1**2)
        inv_r2 = du2 / (du2**2 + zr2**2)
        phase = k0 * (u - ct) - gouy + 0.5 * k0 * (xi1**2 * inv_r1 + xi2**2 * inv_r2)

        amplitude = self._a0_from_density(self.photon_density(x, y, z, t)) * xp.cos(phase)
        _, p1, _ = self.polarization_axes()
        return xp.stack([amplitude * p1[0], amplitude * p1[1], amplitude * p1[2]])

    def active_region(self, threshold: float = 1e-3) -> ActiveRegion:
        """Bounding region where envelope >= threshold * peak."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"active_region threshold must be in (0, 1), got {threshold}")

        # Use peak envelope value for bounding
        xp = np  # active_region runs on host
        reach = 2.0 * math.sqrt(math.log(1.0 / threshold))
        half_length = reach * C_CGS * self.temporal_envelope.phase_time_width()

        slide = abs(1.0 + self.beta_ff)
        drift = abs(self.beta_ff) * half_length
        axes = zip(
            (self.m("sigma_x"), self.m("sigma_y")),
            (abs(self.m("z_fx")), abs(self.m("z_fy"))),
            (abs(self.rayleigh_x()), abs(self.rayleigh_y())),
        )
        intercept, slope = 0.0, 0.0
        for sigma, focus_offset, z_r in axes:
            intercept = max(intercept, sigma * (1.0 + (drift + focus_offset) / z_r))
            slope = max(slope, sigma * slide / z_r)
        k_hat, _, _ = self.focusing_axes()
        origin = np.array([self.m("x_off"), self.m("y_off"), 0.0]) - k_hat * (C_CGS * self.m("t_off"))
        return ActiveRegion(
            axis=k_hat,
            origin=origin,
            radius=reach * intercept,
            radius_slope=reach * slope,
            half_length=half_length,
            threshold=threshold,
            a0_peak=self.a0_peak(),
        )

    def a0_peak(self) -> float:
        """Peak period-averaged a0 anywhere in the pulse."""
        return float(self._a0_from_density(self._peak_density()))

    def intensity_peak(self) -> float:
        """Peak cycle-averaged ``<a^2>`` anywhere in the pulse."""
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * np.pi * self.m("pulse_energy") * self._peak_density()

    def _peak_density(self) -> float:
        """Peak normalized photon density."""
        lo, hi = sorted((self.m("z_fx"), self.m("z_fy")))
        u = np.linspace(lo, hi, 257) if hi > lo else np.array([lo])
        s1, s2 = self.spot_sizes(u)
        transverse_max = float(np.max(1.0 / (2.0 * math.pi * s1 * s2)))
        temporal_max = self.temporal_envelope.peak_value(np) / C_CGS
        return transverse_max * temporal_max

    def cycle_average_factor(self) -> float:
        return 0.5 * (1.0 + self.ellipticity**2)


# ---------------------------------------------------------------------------
# GaussianParaxialLaser (now a thin wrapper)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianParaxialLaser(SeparableParaxialLaser):
    """Elliptical, astigmatic paraxial Gaussian pulse with phase-aware envelope.

    This is the phase-aware version of the original GaussianParaxialLaser.
    The temporal envelope is a Gaussian in phase time τ = φ/ω₀.
    """

    duration: Quantity | None = None  # RMS duration in phase time

    # Override UNITS to include duration
    UNITS = {
        "pulse_energy": "erg",
        "wavelength": "cm",
        "sigma_x": "cm",
        "sigma_y": "cm",
        "duration": "s",
        "z_fx": "cm",
        "z_fy": "cm",
        "x_off": "cm",
        "y_off": "cm",
        "t_off": "s",
        "theta_xz": "rad",
        "theta_yz": "rad",
        "psi_focus": "rad",
        "psi_pol": "rad",
    }

    LIGHT_TIME_FIELDS = frozenset({"duration", "t_off"})

    def __post_init__(self) -> None:
        # Create temporal envelope from duration
        if self.duration is None:
            raise ValueError("GaussianParaxialLaser: duration must be provided")
        envelope = GaussianTemporalEnvelope(duration=self.duration)
        object.__setattr__(self, "temporal_envelope", envelope)
        # Call parent __post_init__ (which validates)
        super().__post_init__()

    def m(self, name: str) -> float:
        if name == "duration":
            return self.temporal_envelope.m("duration")
        return super().m(name)

    def sigma_ct(self) -> float:
        """RMS pulse duration expressed as a length, cm."""
        return C_CGS * self.m("duration")

# ---------------------------------------------------------------------------
# The paraxial Gaussian pulse train implementation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PulseTrainParaxialLaser(SeparableParaxialLaser):
    """Train of N_p paraxial Gaussian sub-pulses with phase-aware envelope.

    CGS-Gaussian throughout (P1, RES013, RES054). Conserves total laser energy E_tot across
    configurations, so each sub-pulse carries energy E_tot / N_p.

    The temporal envelope is a train of Gaussians in phase time τ = φ/ω₀.
    When N_p = 1, reproduces GaussianParaxialLaser identically.
    """

    subpulse_duration: Quantity | None = None  # RMS duration of one sub-pulse in phase time
    repetition_period: Quantity | None = None  # Time between sub-pulses
    n_subpulses: int = 10  # Number of sub-pulses (N_p >= 1)

    # Override UNITS to include pulse train parameters
    UNITS = {
        "pulse_energy": "erg",
        "wavelength": "cm",
        "sigma_x": "cm",
        "sigma_y": "cm",
        "subpulse_duration": "s",
        "repetition_period": "s",
        "z_fx": "cm",
        "z_fy": "cm",
        "x_off": "cm",
        "y_off": "cm",
        "t_off": "s",
        "theta_xz": "rad",
        "theta_yz": "rad",
        "psi_focus": "rad",
        "psi_pol": "rad",
    }

    LIGHT_TIME_FIELDS = frozenset({"subpulse_duration", "repetition_period", "t_off"})

    def __post_init__(self) -> None:
        # Create temporal envelope from pulse train parameters
        if self.subpulse_duration is None or self.repetition_period is None:
            raise ValueError("PulseTrainParaxialLaser: subpulse_duration and repetition_period must be provided")
        envelope = PulseTrainTemporalEnvelope(
            subpulse_duration=self.subpulse_duration,
            repetition_period=self.repetition_period,
            n_subpulses=self.n_subpulses,
        )
        object.__setattr__(self, "temporal_envelope", envelope)
        # Call parent __post_init__ (which validates)
        super().__post_init__()

    def m(self, name: str) -> float:
        if name in ("subpulse_duration", "repetition_period"):
            return self.temporal_envelope.m(name)
        return super().m(name)

    def duty_cycle(self) -> float:
        """Duty cycle D = subpulse_duration / repetition_period."""
        return self.m("subpulse_duration") / self.m("repetition_period")

    def subpulse_delays(self) -> np.ndarray:
        """Temporal offsets t_k of individual sub-pulses relative to the train center (seconds)."""
        return self.temporal_envelope.subpulse_delays()

    def subpulse(self, index: int) -> GaussianParaxialLaser:
        """The index-th sub-pulse (0 <= index < n_subpulses) as a GaussianParaxialLaser."""
        if not 0 <= index < self.n_subpulses:
            raise IndexError(f"subpulse index {index} out of range [0, {self.n_subpulses})")
        delays = self.subpulse_delays()
        sub_energy = self.pulse_energy / self.n_subpulses
        return GaussianParaxialLaser(
            pulse_energy=sub_energy,
            wavelength=self.wavelength,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            duration=self.subpulse_duration,
            z_fx=self.z_fx,
            z_fy=self.z_fy,
            x_off=self.x_off,
            y_off=self.y_off,
            t_off=self.t_off + Quantity(float(delays[index]), "s"),
            theta_xz=self.theta_xz,
            theta_yz=self.theta_yz,
            psi_focus=self.psi_focus,
            psi_pol=self.psi_pol,
            ellipticity=self.ellipticity,
            beta_ff=self.beta_ff,
        )

    def subpulses(self) -> list[GaussianParaxialLaser]:
        """Constituent sub-pulses as individual GaussianParaxialLaser instances."""
        return [self.subpulse(i) for i in range(self.n_subpulses)]

    def sigma_ct(self) -> float:
        """RMS sub-pulse duration expressed as a length, cm."""
        return C_CGS * self.m("subpulse_duration")

    def cycle_average_factor(self) -> float:
        """``C`` in ``<a^2> = C a0^2``."""
        return 0.5 * (1.0 + self.ellipticity**2)

    def intensity_profile(self, x, y, z, t):
        """Cycle-averaged normalized intensity ``<a^2>`` at ``(x, y, z, t)``."""
        density = self.photon_density(x, y, z, t)
        xp = _get_array_module(density)
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * xp.pi * self.m("pulse_energy") * density

    def _a0_from_density(self, density):
        """Peak normalized amplitude ``a0`` from a normalized photon-density envelope."""
        xp = _get_array_module(density)
        e0 = xp.sqrt(8.0 * xp.pi * self.m("pulse_energy") * xp.asarray(density, dtype=float))
        return E_ESU * e0 / (ME_CGS * C_CGS * self.omega0())

    def _local_coordinates(self, x, y, z, t):
        """Lab ``(x, y, z, t)`` → ``(xi1, xi2, u, u_spot, ct)`` in the pulse's own frame."""
        k_hat, f1, f2 = self.focusing_axes()
        xp = _get_array_module(x, y, z, t)
        rx = xp.asarray(x, dtype=float) - self.m("x_off")
        ry = xp.asarray(y, dtype=float) - self.m("y_off")
        rz = xp.asarray(z, dtype=float)
        u = rx * k_hat[0] + ry * k_hat[1] + rz * k_hat[2]
        xi1 = rx * f1[0] + ry * f1[1] + rz * f1[2]
        xi2 = rx * f2[0] + ry * f2[1] + rz * f2[2]
        ct = C_CGS * (xp.asarray(t, dtype=float) - self.m("t_off"))
        return xi1, xi2, u, u + self.beta_ff * ct, ct

    def photon_density(self, x, y, z, t):
        """Photon density = transverse(xi1,xi2,u) × temporal(phase_time).

        Normalized to integrate to 1 over all space at fixed time.
        """
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        xp = _get_array_module(xi1, xi2, u, u_spot, ct)

        # Transverse Gaussian
        s1, s2 = self.spot_sizes(u_spot)
        transverse = (1.0 / (2.0 * xp.pi * s1 * s2)) * xp.exp(
            -0.5 * ((xi1 / s1) ** 2 + (xi2 / s2) ** 2)
        )

        # Temporal envelope in phase time (normalized to 1 over phase_time).
        # The Jacobian du/d(phase_time) = C_CGS, so we divide by C_CGS to
        # normalize the integral over u to 1.
        phase_time = self._phase_time(xi1, xi2, u, u_spot, ct, xp)
        temporal = self.temporal_envelope.envelope(phase_time, xp) / C_CGS

        return transverse * temporal

    def a0_profile(self, x, y, z, t):
        """Period-averaged normalized vector-potential envelope."""
        return self._a0_from_density(self.photon_density(x, y, z, t))

    def field(self, x, y, z, t):
        """Period-resolved normalized vector potential, lab-frame components."""
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        xp = _get_array_module(xi1, xi2, u, u_spot, ct)
        s1, s2 = self.spot_sizes(u_spot)
        k0 = 2.0 * xp.pi / self.m("wavelength")

        du1 = u_spot - self.m("z_fx")
        du2 = u_spot - self.m("z_fy")
        zr1, zr2 = self.rayleigh_x(), self.rayleigh_y()
        gouy = 0.5 * (xp.arctan2(du1, zr1) + xp.arctan2(du2, zr2))
        inv_r1 = du1 / (du1**2 + zr1**2)
        inv_r2 = du2 / (du2**2 + zr2**2)
        phase = k0 * (u - ct) - gouy + 0.5 * k0 * (xi1**2 * inv_r1 + xi2**2 * inv_r2)

        amplitude = self._a0_from_density(self.photon_density(x, y, z, t)) * xp.cos(phase)
        _, p1, _ = self.polarization_axes()
        return xp.stack([amplitude * p1[0], amplitude * p1[1], amplitude * p1[2]])

    def active_region(self, threshold: float = 1e-3) -> ActiveRegion:
        """Bounding region where ``a0_profile >= threshold * a0_peak``."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"active_region threshold must be in (0, 1), got {threshold}")
        reach = 2.0 * math.sqrt(math.log(1.0 / threshold))
        burst_c = C_CGS * (self.n_subpulses - 1) * self.m("repetition_period")
        half_length = 0.5 * burst_c + reach * self.sigma_ct()
        slide = abs(1.0 + self.beta_ff)
        drift = abs(self.beta_ff) * half_length
        axes = zip(
            (self.m("sigma_x"), self.m("sigma_y")),
            (abs(self.m("z_fx")), abs(self.m("z_fy"))),
            (abs(self.rayleigh_x()), abs(self.rayleigh_y())),
        )
        intercept, slope = 0.0, 0.0
        for sigma, focus_offset, z_r in axes:
            intercept = max(intercept, sigma * (1.0 + (drift + focus_offset) / z_r))
            slope = max(slope, sigma * slide / z_r)
        k_hat, _, _ = self.focusing_axes()
        origin = np.array([self.m("x_off"), self.m("y_off"), 0.0]) - k_hat * (C_CGS * self.m("t_off"))
        return ActiveRegion(
            axis=k_hat,
            origin=origin,
            radius=reach * intercept,
            radius_slope=reach * slope,
            half_length=half_length,
            threshold=threshold,
            a0_peak=self.a0_peak(),
        )


# ---------------------------------------------------------------------------
# Descriptive fit (§3.3, the laser-side analogue of Bunch.fit_gaussian / P8)
# ---------------------------------------------------------------------------
def fit_gaussian_paraxial(laser: LaserField) -> GaussianParaxialLaser:
    """Descriptive `GaussianParaxialLaser` metrics extracted from any `LaserField`.

    What target autoranging (§3.4), the analytical engine (§4.3) and the GUI sketch panel
    call when they need a rough physical picture — waist, Rayleigh range, effective
    duration, peak a0 — so none of them ever samples a raw field itself. Keeping that in
    one place is what makes those callers correct by construction once a non-Gaussian
    `LaserField` lands (P15).

    For a `GaussianParaxialLaser` the fit is an **identity**: its parameters already are
    the exact answer, and re-deriving them numerically could only add error.

    For a `PulseTrainParaxialLaser` with `n_subpulses == 1`, returns the single constituent
    sub-pulse directly.

    For any other implementation this raises. That is deliberate, not an oversight: no
    such implementation exists yet (`Spectral-FEM-Fields` has no Python bindings), so a
    numerical fit written now would be untestable code speculating about a field
    representation nobody has seen — precisely the speculative abstraction P6 rejects.
    The identity path is what Phase 1 has a consumer and a test for; the numerical path
    lands with the second implementation that needs it, which will also be able to test it.
    """
    if isinstance(laser, GaussianParaxialLaser):
        return laser
    if isinstance(laser, PulseTrainParaxialLaser) and laser.n_subpulses == 1:
        return laser.subpulses()[0]
    raise NotImplementedError(
        f"fit_gaussian_paraxial has no numerical path yet and {type(laser).__name__} is not "
        "a GaussianParaxialLaser. Implement the fit alongside the LaserField "
        "implementation that needs it."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(laser: GaussianParaxialLaser | PulseTrainParaxialLaser) -> list[str]:
    """Hard-fail on impossible values; return warning strings for suspicious ones."""
    if isinstance(laser, PulseTrainParaxialLaser):
        for name in ("pulse_energy", "wavelength", "sigma_x", "sigma_y", "subpulse_duration", "repetition_period"):
            if laser.m(name) <= 0:
                raise ValueError(f"PulseTrainParaxialLaser: {name} must be > 0")
        if laser.n_subpulses < 1:
            raise ValueError("PulseTrainParaxialLaser: n_subpulses must be >= 1")
        if not math.isfinite(laser.m("z_fx")) or not math.isfinite(laser.m("z_fy")):
            raise ValueError("PulseTrainParaxialLaser: focal offsets must be finite")
        if laser.beta_ff <= -1.0:
            raise ValueError("PulseTrainParaxialLaser: beta_ff must be > -1 (Rayleigh range scales as 1 + beta_ff)")
        if not 0.0 <= laser.ellipticity <= 1.0:
            raise ValueError("PulseTrainParaxialLaser: ellipticity must be in [0, 1]")

        warnings: list[str] = []
        if abs(laser.m("z_fx")) > laser.rayleigh_x() or abs(laser.m("z_fy")) > laser.rayleigh_y():
            warnings.append(
                "A focus sits more than a Rayleigh range from the interaction point; the "
                "on-axis a0 there is well below the pulse's peak."
            )
        if laser.m("sigma_x") != laser.m("sigma_y") or laser.m("z_fx") != laser.m("z_fy"):
            warnings.append("Elliptical and/or astigmatic beam — check the focusing axes (psi_focus).")
        if max(laser.m("sigma_x"), laser.m("sigma_y")) < laser.m("wavelength"):
            warnings.append(
                "Spot size is below the wavelength; the paraxial approximation does not hold."
            )
        return warnings

    for name in ("pulse_energy", "wavelength", "sigma_x", "sigma_y", "duration"):
        if laser.m(name) <= 0:
            raise ValueError(f"GaussianParaxialLaser: {name} must be > 0")
    if not math.isfinite(laser.m("z_fx")) or not math.isfinite(laser.m("z_fy")):
        raise ValueError("GaussianParaxialLaser: focal offsets must be finite")
    if laser.beta_ff <= -1.0:
        raise ValueError("GaussianParaxialLaser: beta_ff must be > -1 (Rayleigh range scales as 1 + beta_ff)")
    if not 0.0 <= laser.ellipticity <= 1.0:
        raise ValueError("GaussianParaxialLaser: ellipticity must be in [0, 1]")

    warnings: list[str] = []
    if abs(laser.m("z_fx")) > laser.rayleigh_x() or abs(laser.m("z_fy")) > laser.rayleigh_y():
        warnings.append(
            "A focus sits more than a Rayleigh range from the interaction point; the "
            "on-axis a0 there is well below the pulse's peak."
        )
    if laser.m("sigma_x") != laser.m("sigma_y") or laser.m("z_fx") != laser.m("z_fy"):
        warnings.append("Elliptical and/or astigmatic beam — check the focusing axes (psi_focus).")
    if max(laser.m("sigma_x"), laser.m("sigma_y")) < laser.m("wavelength"):
        warnings.append(
            "Spot size is below the wavelength; the paraxial approximation does not hold."
        )
    return warnings

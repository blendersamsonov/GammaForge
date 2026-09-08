"""Laser field representation (GRAND_PLAN.md §3.3, §2.2; P15, RES067).

Two things live here, and the split is the point:

* :class:`LaserField` — the **sampling contract engines are typed against**. Vectorized,
  lab-frame methods (``intensity_profile``, ``a0_profile``, ``field``, ``active_region``).
  Quasi-monochromatic engines (xigma, kascade) consume field sampling alongside physical
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

:func:`validate` no longer warns on these; the markers below remain as the one-line greps
for "the derivation landed".
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

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

    All three methods are **lab-frame** and **array-callable**: pass numpy (or cupy)
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
# The paraxial Gaussian implementation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianParaxialLaser:
    """Elliptical, astigmatic paraxial Gaussian pulse. CGS-Gaussian throughout (P1).

    ``sigma_x``/``sigma_y`` are the RMS widths of the **intensity** profile at each axis's
    own waist (`WidthConvention.SIGMA_INTENSITY_RMS`), along the focusing axes set by
    ``psi_focus``. ``z_fx``/``z_fy`` place those two waists at (generally different)
    positions along the propagation direction — astigmatism; the round, stigmatic beam is
    the degenerate case ``sigma_x == sigma_y`` and ``z_fx == z_fy``.

    ``duration`` is the RMS intensity duration. ``ellipticity`` is the polarization
    degree — *distinct from spot ellipticity*, which the per-axis waists express.
    It is applied to the angle-resolved kernel (DER004 §1.2, DER006): the polarization
    factor is ``(cos^2 psi + eps^2 sin^2 psi)/(1 + eps^2)`` in the head-on limit,
    and the full DER006 expression with crossing angle. ``beta_ff`` is the flying-focus
    factor, entering only the spot-size term (never the longitudinal envelope), ported
    from the predecessor's xigma formalism.

    Derived quantities (photon energy, peak a0, photon count) are module-level helpers or
    plain methods evaluated at the point of use, never cached properties (P9).
    """

    pulse_energy: Quantity  # energy
    wavelength: Quantity  # length
    sigma_x: Quantity  # length, RMS intensity width along focusing axis 1
    sigma_y: Quantity  # length, RMS intensity width along focusing axis 2
    duration: Quantity  # time, RMS intensity duration
    z_fx: Quantity = Quantity(0.0, "cm")  # focal offset of axis 1 along k_hat
    z_fy: Quantity = Quantity(0.0, "cm")  # focal offset of axis 2 along k_hat
    # Misalignment of the pulse against the bunch, which defines the origin. There is no
    # `z_off` because a longitudinal spatial offset is degenerate with `t_off` **given** `z_fx`/`z_fy`: a rigid shift of the pulse by `Delta`
    # along `k_hat` moves the focus *and* the envelope, so it is exactly
    # `(z_fx += Delta, z_fy += Delta, t_off += Delta/c)`. Focus position and arrival time
    # are genuinely independent — coincident foci still miss if the arrival times differ —
    # and both are present; only the redundant fourth combination is omitted.
    x_off: Quantity = Quantity(0.0, "cm")
    y_off: Quantity = Quantity(0.0, "cm")
    t_off: Quantity = Quantity(0.0, "s")  # pulse centre reaches the origin at t = t_off
    theta_xz: Quantity = Quantity(0.0, "rad")
    theta_yz: Quantity = Quantity(0.0, "rad")
    psi_focus: Quantity = Quantity(0.0, "rad")
    psi_pol: Quantity = Quantity(0.0, "rad")
    ellipticity: float = 0.0
    beta_ff: float = 0.0

    #: The convention every stored width is in — see `WidthConvention` (§2.1).
    width_convention = WidthConvention.SIGMA_INTENSITY_RMS

    #: Canonical CGS unit of each dimensioned field; also what `__post_init__` converts
    #: incoming values into, so `.m` below always yields CGS. Angles are typed too, so a
    #: crossing angle can be given in degrees without a hand-written conversion.
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

    #: Longitudinal extents, which §2.1 allows to be quoted as either a length or a
    #: duration. Only these opt into the `light_time` equivalence — a transverse size
    #: given in femtoseconds is a mistake, not a unit choice.
    LIGHT_TIME_FIELDS = frozenset({"duration", "t_off"})

    def __post_init__(self) -> None:
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(
                    getattr(self, name), unit, name, light_time=name in self.LIGHT_TIME_FIELDS
                ),
            )
        for name in ("ellipticity", "beta_ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                raise ValueError(f"GaussianParaxialLaser: {name} must be a finite scalar, got {value!r}")
        for name in self.UNITS:
            if not math.isfinite(self.m(name)):
                raise ValueError(f"GaussianParaxialLaser: {name} must be finite, got {getattr(self, name)!r}")

    def m(self, name: str) -> float:
        """Magnitude of a dimensioned field in its canonical CGS unit.

        Every method below unpacks through this once at the top, so the vectorized field
        arithmetic underneath is plain numpy — pint never enters a hot path (§2.1).
        """
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
        """Rayleigh range of focusing axis 1, cm.

        **The conversion is the whole content of this method** (RES040). The textbook formula
        ``z_R = pi w0^2 / lambda`` is stated in the **1/e² convention**: ``w0`` is the
        radius at which intensity falls to ``e^-2`` of its on-axis value. This class stores
        widths as **RMS of the photon-density (= intensity) profile**
        (`WidthConvention.SIGMA_INTENSITY_RMS`), which is a different number — at
        ``r = sigma`` the density is down only by ``e^-1/2``, not ``e^-2``.

        Converting first, then applying the standard formula: matching
        ``exp(-r^2 / (2 sigma^2))`` against ``exp(-2 r^2 / w0^2)`` gives ``w0 = 2 sigma``,
        hence ``z_R = 4 pi sigma^2 / lambda``. Anything that reads a Rayleigh range or a
        divergence off a stored ``sigma`` **must** go through that factor of two in the
        radius — skipping it is a factor of 4 in ``z_R`` and in the far-field angle
        ``sigma / z_R`` (RES040).

        The flying-focus factor stretches it by ``(1 + beta_ff)``, the predecessor's xigma
        convention.
        """
        return 4.0 * math.pi * self.m("sigma_x") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def rayleigh_y(self) -> float:
        return 4.0 * math.pi * self.m("sigma_y") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def sigma_ct(self) -> float:
        """RMS pulse duration expressed as a length, cm."""
        return C_CGS * self.m("duration")

    def spot_sizes(self, u_spot):
        """RMS intensity spot sizes ``(s1, s2)`` at longitudinal position ``u_spot``."""
        s1 = self.m("sigma_x") * np.sqrt(1.0 + ((u_spot - self.m("z_fx")) / self.rayleigh_x()) ** 2)
        s2 = self.m("sigma_y") * np.sqrt(1.0 + ((u_spot - self.m("z_fy")) / self.rayleigh_y()) ** 2)
        return s1, s2

    def a0_peak(self) -> float:
        """Peak period-averaged a0 anywhere in the pulse.

        The pulse's own maximum, attained where both spots are smallest and the temporal
        envelope peaks. With astigmatism the two waists are at different ``u``, so the
        joint maximum of ``1 / (s1 s2)`` sits between them and is found numerically over
        the interval they span (a 1D unimodal problem, not worth an optimizer).
        """
        return float(self._a0_from_density(self._peak_density()))

    def intensity_peak(self) -> float:
        """Peak cycle-averaged ``<a^2>`` anywhere in the pulse — :meth:`a0_peak`'s
        polarization-agnostic counterpart, and what engines should key off.

        Same peak-density search as :meth:`a0_peak`, converted through the ``4 pi`` chain
        (:meth:`intensity_profile`) instead of the ``8 pi`` amplitude one.

        Equal to ``cycle_average_factor() * a0_peak()**2`` **only for linear polarization**,
        and the difference is a trap worth naming: :meth:`a0_peak` reports the
        *linear-equivalent* amplitude by convention, so rebuilding the peak intensity as
        ``C * a0_peak()**2`` applies the cycle average without the ``1/sqrt(2C)`` that
        belongs in the amplitude — the result then varies with ``ellipticity`` although the
        physical quantity does not.
        """
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * np.pi * self.m(
            "pulse_energy"
        ) * self._peak_density()

    def _peak_density(self) -> float:
        """Peak normalized photon density: where both spots are smallest and the temporal
        envelope peaks. Shared by :meth:`a0_peak` and :meth:`intensity_peak` so the two
        cannot disagree about *where* the pulse peaks, only about what they report there.
        """
        lo, hi = sorted((self.m("z_fx"), self.m("z_fy")))
        u = np.linspace(lo, hi, 257) if hi > lo else np.array([lo])
        s1, s2 = self.spot_sizes(u)
        return float(np.max(1.0 / ((2.0 * np.pi) ** 1.5 * s1 * s2 * self.sigma_ct())))

    def cycle_average_factor(self) -> float:
        """``C`` in ``<a^2> = C a0^2``, the cycle average of the normalized intensity.

        From the paper's own normalization ``sum_i |eps_i|^2 = 1`` (eq. `field`) with an
        ellipse of axis ratio ``eps = ellipticity``, ``eps_0 = 1/sqrt(1+eps^2)`` and
        ``eps_1 = i eps/sqrt(1+eps^2)``::

            C = (1 + eps^2) / 2

        ``1/2`` for linear (``<cos^2> = 1/2``), ``1`` for circular (constant magnitude),
        and the exact interpolation between — verified numerically against a
        period-resolved ellipse at ``eps = 0, 1/4, 1/2, 1/sqrt(3), 1``.

        **This is a property of the polarization state, not a physical prediction**, and
        it is deliberately *not* how any yield or red-shift is computed — see
        :meth:`intensity_profile` for why those never need it. It exists because ``a0``
        itself is a reported number whose definition depends on the convention.
        """
        return 0.5 * (1.0 + self.ellipticity**2)

    def intensity_profile(self, x, y, z, t):
        """Cycle-averaged normalized intensity ``<a^2>`` at ``(x, y, z, t)``.

        **The polarization-agnostic quantity, and the one physics actually depends on.**
        Every angle-integrated observable — the photon yield, and the mean nonlinear
        red-shift through ``ahat`` — is a functional of ``<a^2>`` along a trajectory, never
        of ``a0`` separately. And ``<a^2>`` does not depend on the polarization state at
        all, at fixed pulse energy::

            <a^2> = C a0^2,   a0^2 = (e / m_e c omega0)^2 * 4 pi U_density / C

        so ``C`` cancels identically, leaving

            <a^2> = (e / m_e c omega0)^2 * 4 pi E_pulse * photon_density

        with no ``C`` and therefore no ``ellipticity`` anywhere in it (RES054's Rationale has
        the numeric check).

        That is why `engines.xigma.stages` integrates this method, not :meth:`a0_profile` —
        forming ``a0`` first and re-applying a polarization factor round-trips through a
        convention-dependent number (RES053/RES054).

        Note the ``4 pi`` rather than ``8 pi``: this is the cycle **average**, whereas
        :meth:`a0_profile` returns the **peak** amplitude of a linearly polarized field.
        """
        density = np.asarray(self.photon_density(x, y, z, t), dtype=float)
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * np.pi * self.m("pulse_energy") * density

    def _a0_from_density(self, density):
        """Peak normalized amplitude ``a0`` from a normalized photon-density envelope.

        The chain in CGS-Gaussian: energy density ``U = E_pulse * density``, intensity
        ``I = c U``, cycle-averaged ``I = c E0^2 / (8 pi)`` for **linear** polarization, so
        ``E0 = sqrt(8 pi E_pulse * density)`` and ``a0 = e E0 / (m_e c omega0)``.

        ``ellipticity`` does **not** enter — ``a0`` is reported as the **linear-equivalent
        peak amplitude**, a stated convention (RES054), so that a number quoted as "a0 = 2"
        means the same field strength regardless of how the pulse is polarized. The
        elliptical peak amplitude, if it is ever wanted, is this divided by ``sqrt(2 C)``
        (:meth:`cycle_average_factor`).

        Physics does not go through here — see :meth:`intensity_profile`.
        """
        e0 = np.sqrt(8.0 * np.pi * self.m("pulse_energy") * np.asarray(density, dtype=float))
        return E_ESU * e0 / (ME_CGS * C_CGS * self.omega0())

    # -- the LaserField contract -------------------------------------------
    def _local_coordinates(self, x, y, z, t):
        """Lab ``(x, y, z, t)`` → ``(xi1, xi2, u, u_spot, ct)`` in the pulse's own frame."""
        k_hat, f1, f2 = self.focusing_axes()
        # Everything is measured from the pulse's own centre, which the misalignment
        # offsets displace from the bunch's. One subtraction here is the whole
        # implementation: every consumer of the field inherits it.
        rx = np.asarray(x, dtype=float) - self.m("x_off")
        ry = np.asarray(y, dtype=float) - self.m("y_off")
        rz = np.asarray(z, dtype=float)
        u = rx * k_hat[0] + ry * k_hat[1] + rz * k_hat[2]
        xi1 = rx * f1[0] + ry * f1[1] + rz * f1[2]
        xi2 = rx * f2[0] + ry * f2[1] + rz * f2[2]
        ct = C_CGS * (np.asarray(t, dtype=float) - self.m("t_off"))
        # Flying focus: the spot-size evaluation point slides with time, while the
        # longitudinal envelope below stays beta_ff-independent (xigma's construction).
        return xi1, xi2, u, u + self.beta_ff * ct, ct

    def photon_density(self, x, y, z, t):
        """Photon-density envelope, normalized to integrate to 1 over space at fixed ``t``.

        Integrating over ``t`` as well would double-count: the pulse translates through
        space, so its photon number is conserved, not accumulated.
        """
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        s1, s2 = self.spot_sizes(u_spot)
        s_ct = self.sigma_ct()
        norm = 1.0 / ((2.0 * np.pi) ** 1.5 * s1 * s2 * s_ct)
        arg = -(xi1**2) / (2.0 * s1**2) - xi2**2 / (2.0 * s2**2) - (u - ct) ** 2 / (2.0 * s_ct**2)
        return norm * np.exp(arg)

    def a0_profile(self, x, y, z, t):
        """Period-averaged normalized vector-potential envelope (the `LaserField` method)."""
        return self._a0_from_density(self.photon_density(x, y, z, t))

    def field(self, x, y, z, t):
        """Period-resolved normalized vector potential, lab-frame components.

        Returns an array of shape ``(3, *broadcast_shape)`` so callers can unpack
        ``ax, ay, az = laser.field(...)``. The natural companion to :meth:`a0_profile`:
        its envelope *is* ``a0_profile``, and consumers derive **E** and **B** from it.

        The carrier phase is the full paraxial one — plane-wave term, per-axis Gouy phase,
        **Elliptical polarization** along `p1` (`psi_pol`), with `ellipticity` entering
        the angle-resolved kernel as the factor ``(cos^2 psi + eps^2 sin^2 psi)/(1 + eps^2)``
        in the head-on limit, and the full DER006 expression with crossing angle.
        """
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        s1, s2 = self.spot_sizes(u_spot)
        k0 = 2.0 * np.pi / self.m("wavelength")

        du1 = u_spot - self.m("z_fx")
        du2 = u_spot - self.m("z_fy")
        zr1, zr2 = self.rayleigh_x(), self.rayleigh_y()
        gouy = 0.5 * (np.arctan2(du1, zr1) + np.arctan2(du2, zr2))
        # Radius of curvature R(u) = u * (1 + (zR/u)^2); written as the reciprocal so
        # 1/R -> 0 smoothly at the waist instead of dividing by zero.
        inv_r1 = du1 / (du1**2 + zr1**2)
        inv_r2 = du2 / (du2**2 + zr2**2)
        phase = k0 * (u - ct) - gouy + 0.5 * k0 * (xi1**2 * inv_r1 + xi2**2 * inv_r2)

        amplitude = self._a0_from_density(self.photon_density(x, y, z, t)) * np.cos(phase)
        _, p1, _ = self.polarization_axes()
        return np.stack([amplitude * p1[0], amplitude * p1[1], amplitude * p1[2]])

    def active_region(self, threshold: float = 1e-3) -> ActiveRegion:
        """Bounding region where ``a0_profile >= threshold * a0_peak`` (§3.2).

        Derived analytically and **conservatively**. Longitudinally, ``a0`` falls as
        ``exp(-(u - ct)^2 / (4 sigma_ct^2))`` (the square root of the intensity Gaussian),
        so the cut is at ``|u - ct| = 2 sigma_ct sqrt(ln(1/threshold))``. Transversely the
        same square-root Gaussian gives ``xi <= reach * s(u)`` with
        ``reach = 2 sqrt(ln(1/threshold))``; the ``1/sqrt(s1 s2)`` amplitude decay away
        from focus is deliberately ignored, which can only make the real region smaller
        than this bound.

        ``s`` **grows with distance from focus**, so the transverse bound is a cone rather
        than a fixed radius. Linearizing the hyperbola,
        ``s_i(v) = sigma_i sqrt(1 + ((v - z_fi)/z_Ri)^2) <= sigma_i (1 + (|v| + |z_fi|)/z_Ri)``,
        gives an intercept and a slope, each maximized over the two focusing axes
        independently — which over-estimates when the two axes disagree, in the safe
        direction. Bounding ``s`` by its value near focus instead (as this did until the
        Phase-2 harness caught it) silently discards particles that a diverged pulse still
        reaches, whenever the bunch is longer than the Rayleigh range.

        **The spot is evaluated at the flying-focus coordinate**, not at ``u``:
        ``v = u + beta_ff * ct`` (see :meth:`_local_coordinates`). Inside the longitudinal
        window ``ct`` is within ``half_length`` of ``u``, so
        ``|v| <= |1 + beta_ff| |u| + |beta_ff| half_length`` — the slide steepens the cone
        by ``|1 + beta_ff|`` and widens its intercept by the drift accumulated across the
        pulse length. The steepening cancels against the ``(1 + beta_ff)`` stretch already
        in :meth:`rayleigh_x`, which is why omitting it makes the cone too *narrow* rather
        than merely inexact — the one direction a conservative bound may not err in.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"active_region threshold must be in (0, 1), got {threshold}")
        reach = 2.0 * math.sqrt(math.log(1.0 / threshold))
        half_length = reach * self.sigma_ct()
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
        # A transverse misalignment moves the region bodily; a timing offset slides the
        # pulse along its own axis, which `overlap_time_window` sees as the region's
        # centre being reached later. Both must be carried or the prefilter silently
        # discards particles that do interact — the one direction §3.2 forbids.
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
# The paraxial Gaussian pulse train implementation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PulseTrainParaxialLaser:
    """Train of N_p paraxial Gaussian sub-pulses with inter-pulse period T_rep.

    CGS-Gaussian throughout (P1, RES013, RES054). Conserves total laser energy E_tot across
    configurations, so each sub-pulse carries energy E_tot / N_p.

    The temporal envelope is parameterized by the duty cycle D = subpulse_duration / repetition_period.
    When N_p = 1, reproduces GaussianParaxialLaser identically (with duration = subpulse_duration).
    """

    pulse_energy: Quantity  # Total energy summed over all sub-pulses
    wavelength: Quantity  # Central carrier wavelength
    sigma_x: Quantity  # length, RMS intensity width along focusing axis 1
    sigma_y: Quantity  # length, RMS intensity width along focusing axis 2
    subpulse_duration: Quantity  # time, RMS intensity duration of one sub-pulse
    repetition_period: Quantity  # time between adjacent sub-pulses (T_rep)
    n_subpulses: int = 10  # Number of sub-pulses (N_p >= 1)
    z_fx: Quantity = Quantity(0.0, "cm")  # focal offset of axis 1 along k_hat
    z_fy: Quantity = Quantity(0.0, "cm")  # focal offset of axis 2 along k_hat
    x_off: Quantity = Quantity(0.0, "cm")
    y_off: Quantity = Quantity(0.0, "cm")
    t_off: Quantity = Quantity(0.0, "s")  # train temporal center reaches origin at t = t_off
    theta_xz: Quantity = Quantity(0.0, "rad")
    theta_yz: Quantity = Quantity(0.0, "rad")
    psi_focus: Quantity = Quantity(0.0, "rad")
    psi_pol: Quantity = Quantity(0.0, "rad")
    ellipticity: float = 0.0
    beta_ff: float = 0.0

    width_convention = WidthConvention.SIGMA_INTENSITY_RMS

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
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(
                    getattr(self, name), unit, name, light_time=name in self.LIGHT_TIME_FIELDS
                ),
            )
        if (
            isinstance(self.n_subpulses, bool)
            or not isinstance(self.n_subpulses, numbers.Integral)
            or self.n_subpulses < 1
        ):
            raise ValueError(
                f"PulseTrainParaxialLaser: n_subpulses must be an integer >= 1, got {self.n_subpulses!r}"
            )
        object.__setattr__(self, "n_subpulses", int(self.n_subpulses))

        for name in ("ellipticity", "beta_ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                raise ValueError(f"PulseTrainParaxialLaser: {name} must be a finite scalar, got {value!r}")
        for name in self.UNITS:
            if not math.isfinite(self.m(name)):
                raise ValueError(f"PulseTrainParaxialLaser: {name} must be finite, got {getattr(self, name)!r}")
        for name in ("pulse_energy", "wavelength", "sigma_x", "sigma_y", "subpulse_duration", "repetition_period"):
            if self.m(name) <= 0.0:
                raise ValueError(f"PulseTrainParaxialLaser: {name} must be > 0, got {self.m(name)!r}")
        if self.beta_ff <= -1.0:
            raise ValueError("PulseTrainParaxialLaser: beta_ff must be > -1 (Rayleigh range scales as 1 + beta_ff)")
        if not 0.0 <= self.ellipticity <= 1.0:
            raise ValueError("PulseTrainParaxialLaser: ellipticity must be in [0, 1]")

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
        """Total photons in the train: pulse energy / central photon energy."""
        return self.m("pulse_energy") / self.photon_energy()

    def rayleigh_x(self) -> float:
        """Rayleigh range of focusing axis 1, cm."""
        return 4.0 * math.pi * self.m("sigma_x") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def rayleigh_y(self) -> float:
        """Rayleigh range of focusing axis 2, cm."""
        return 4.0 * math.pi * self.m("sigma_y") ** 2 / self.m("wavelength") * (1.0 + self.beta_ff)

    def sigma_ct(self) -> float:
        """RMS sub-pulse duration expressed as a length, cm."""
        return C_CGS * self.m("subpulse_duration")

    def duty_cycle(self) -> float:
        """Duty cycle D = subpulse_duration / repetition_period."""
        return self.m("subpulse_duration") / self.m("repetition_period")

    def subpulse_delays(self) -> np.ndarray:
        """Temporal offsets t_k of individual sub-pulses relative to the train center (seconds).

        t_k = (k - (N_p + 1)/2) * T_rep for k in {1, ..., N_p}.
        """
        k = np.arange(1, self.n_subpulses + 1, dtype=float)
        return (k - 0.5 * (self.n_subpulses + 1)) * self.m("repetition_period")

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

    def spot_sizes(self, u_spot):
        """RMS intensity spot sizes ``(s1, s2)`` at longitudinal position ``u_spot``."""
        s1 = self.m("sigma_x") * np.sqrt(1.0 + ((u_spot - self.m("z_fx")) / self.rayleigh_x()) ** 2)
        s2 = self.m("sigma_y") * np.sqrt(1.0 + ((u_spot - self.m("z_fy")) / self.rayleigh_y()) ** 2)
        return s1, s2

    def a0_peak(self) -> float:
        """Peak period-averaged a0 anywhere in the pulse."""
        return float(self._a0_from_density(self._peak_density()))

    def intensity_peak(self) -> float:
        """Peak cycle-averaged ``<a^2>`` anywhere in the pulse."""
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * np.pi * self.m(
            "pulse_energy"
        ) * self._peak_density()

    def _peak_density(self) -> float:
        """Peak normalized photon density."""
        lo, hi = sorted((self.m("z_fx"), self.m("z_fy")))
        u = np.linspace(lo, hi, 257) if hi > lo else np.array([lo])
        s1, s2 = self.spot_sizes(u)
        transverse_max = float(np.max(1.0 / (2.0 * math.pi * s1 * s2)))

        s_ct = self.sigma_ct()
        delays_c = C_CGS * self.subpulse_delays()
        max_c = float(np.max(np.abs(delays_c)))
        grid_zeta = np.linspace(0.0, max_c, 512)
        eval_zeta = np.unique(np.concatenate([[0.0], np.abs(delays_c), grid_zeta]))
        diff = eval_zeta[:, None] - delays_c[None, :]
        long_profile = np.sum(np.exp(-0.5 * (diff / s_ct) ** 2), axis=1) / (
            math.sqrt(2.0 * math.pi) * s_ct * self.n_subpulses
        )
        long_max = float(np.max(long_profile))
        return transverse_max * long_max

    def cycle_average_factor(self) -> float:
        """``C`` in ``<a^2> = C a0^2``."""
        return 0.5 * (1.0 + self.ellipticity**2)

    def intensity_profile(self, x, y, z, t):
        """Cycle-averaged normalized intensity ``<a^2>`` at ``(x, y, z, t)``."""
        density = np.asarray(self.photon_density(x, y, z, t), dtype=float)
        return (E_ESU / (ME_CGS * C_CGS * self.omega0())) ** 2 * 4.0 * np.pi * self.m("pulse_energy") * density

    def _a0_from_density(self, density):
        """Peak normalized amplitude ``a0`` from a normalized photon-density envelope."""
        e0 = np.sqrt(8.0 * np.pi * self.m("pulse_energy") * np.asarray(density, dtype=float))
        return E_ESU * e0 / (ME_CGS * C_CGS * self.omega0())

    def _local_coordinates(self, x, y, z, t):
        """Lab ``(x, y, z, t)`` → ``(xi1, xi2, u, u_spot, ct)`` in the pulse's own frame."""
        k_hat, f1, f2 = self.focusing_axes()
        rx = np.asarray(x, dtype=float) - self.m("x_off")
        ry = np.asarray(y, dtype=float) - self.m("y_off")
        rz = np.asarray(z, dtype=float)
        u = rx * k_hat[0] + ry * k_hat[1] + rz * k_hat[2]
        xi1 = rx * f1[0] + ry * f1[1] + rz * f1[2]
        xi2 = rx * f2[0] + ry * f2[1] + rz * f2[2]
        ct = C_CGS * (np.asarray(t, dtype=float) - self.m("t_off"))
        return xi1, xi2, u, u + self.beta_ff * ct, ct

    def photon_density(self, x, y, z, t):
        """Photon-density envelope, normalized to integrate to 1 over space at fixed ``t``."""
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        s1, s2 = self.spot_sizes(u_spot)
        s_ct = self.sigma_ct()
        zeta = u - ct
        delays_c = C_CGS * self.subpulse_delays()
        diff = np.expand_dims(zeta, -1) - delays_c
        longitudinal = np.sum(np.exp(-0.5 * (diff / s_ct) ** 2), axis=-1) / (
            math.sqrt(2.0 * math.pi) * s_ct * self.n_subpulses
        )
        transverse = (1.0 / (2.0 * math.pi * s1 * s2)) * np.exp(
            -0.5 * ((xi1 / s1) ** 2 + (xi2 / s2) ** 2)
        )
        return transverse * longitudinal

    def a0_profile(self, x, y, z, t):
        """Period-averaged normalized vector-potential envelope."""
        return self._a0_from_density(self.photon_density(x, y, z, t))

    def field(self, x, y, z, t):
        """Period-resolved normalized vector potential, lab-frame components."""
        xi1, xi2, u, u_spot, ct = self._local_coordinates(x, y, z, t)
        s1, s2 = self.spot_sizes(u_spot)
        k0 = 2.0 * math.pi / self.m("wavelength")

        du1 = u_spot - self.m("z_fx")
        du2 = u_spot - self.m("z_fy")
        zr1, zr2 = self.rayleigh_x(), self.rayleigh_y()
        gouy = 0.5 * (np.arctan2(du1, zr1) + np.arctan2(du2, zr2))
        inv_r1 = du1 / (du1**2 + zr1**2)
        inv_r2 = du2 / (du2**2 + zr2**2)
        phase = k0 * (u - ct) - gouy + 0.5 * k0 * (xi1**2 * inv_r1 + xi2**2 * inv_r2)

        amplitude = self._a0_from_density(self.photon_density(x, y, z, t)) * np.cos(phase)
        _, p1, _ = self.polarization_axes()
        return np.stack([amplitude * p1[0], amplitude * p1[1], amplitude * p1[2]])

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
        "implementation that needs it (GRAND_PLAN.md §3.3/P15)."
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

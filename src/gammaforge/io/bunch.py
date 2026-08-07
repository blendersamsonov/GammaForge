"""Electron beam and macroparticle bunch (GRAND_PLAN.md §3.2).

* :class:`GaussianElectronBeam` — the analytic 6D Gaussian description, CGS-Gaussian.
  It is **both** the input description and the output of a fit (P8): there is no
  ``BeamFittedParams`` three-way split, and ``fit_quality`` is simply ``None`` when the
  description came from a user rather than from :func:`fit_gaussian`.
* :class:`Bunch` — raw macroparticle arrays. ``weight`` is **relative**; the physical
  electron count ``N_e`` is a scalar on the interaction (§3.5), never recoverable from
  the bunch, and the predecessor's ``n_electrons`` property is deliberately gone.

Three things here are load-bearing and easy to break:

**Momenta are derived, never sampled.** The sampler draws the physical slice variables
``(x, y, z, thx, thy, gamma)``; :func:`momenta` then gives
``pz = sqrt((gamma^2 - 1) / (1 + thx^2 + thy^2))``, ``px = thx pz``, ``py = thy pz``.
Because ``pz`` is *only* ever obtained that way, ``gamma^2 = 1 + p^2`` holds identically
for every particle — there is no mass-shell "enforcement" step to get wrong.

**Per-variable RNG substreams (§3.2, pinned).** Each sampled variable draws standard
normals from its own substream of ``seed``; the beam's parameters then enter only as an
affine transform on top. This is what makes the §5 cost tiers true rather than
accidental: changing gamma0, energy spread, chirp or dispersion perturbs the gamma draw
and nothing else, so the position/angle arrays Stage 0 consumes stay bit-identical and its
cache legitimately survives.

**Correlations are stored as correlation coefficients**, not as dimensional slopes, and
the marginal energy spread is preserved by construction whatever they are —
:func:`gamma_coefficients` holds that algebra and :func:`validate` reports an inconsistent
set in the user's own terms. :func:`chirp_to_correlation` and
:func:`dispersion_to_correlation` convert from the dimensional forms a user may think in.

The set includes the **angle**-energy correlations ``rho_thx_gamma``/``rho_thy_gamma``
(the dispersion derivative) precisely so that :func:`drift` composes: they are what a
drift needs in order to transport the position-energy correlation, and re-deriving them
from ``alpha`` instead makes a single drift correct while two consecutive drifts silently
disagree with a refit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

from .units import (
    C_CGS,
    E_ESU,
    MEC2_CGS,
    Quantity,
    as_canonical_quantity,
    scale_factor,
)

__all__ = [
    "GaussianElectronBeam",
    "Bunch",
    "validate",
    "sample_gaussian_bunch",
    "momenta",
    "overlap_time_window",
    "prefilter_bunch",
    "drift",
    "propagate",
    "stream",
    "fit_gaussian",
    "evaluate_fit_quality",
    "chirp_to_correlation",
    "gamma_coefficients",
    "dispersion_to_correlation",
    "SAMPLED_VARIABLES",
]

#: Order matters and is part of the reproducibility contract: it fixes which substream of
#: a given seed drives which variable. Reordering or inserting changes every sampled
#: bunch. Append-only if it must ever grow.
SAMPLED_VARIABLES = ("x", "y", "z", "thx", "thy", "gamma")


# ---------------------------------------------------------------------------
# Analytic beam description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianElectronBeam:
    """6D Gaussian electron-beam description. CGS-Gaussian (P1).

    Transverse sizes and the bunch length are RMS values
    (`WidthConvention.SIGMA_INTENSITY_RMS`); the schema converts at the boundary, so
    nothing here is convention-ambiguous.

    ``sigma_z`` is a **length**, which is what the physics uses. Its time view is the
    light-travel time ``sigma_z / c`` — exact to within ``beta0``, and this model is
    ultrarelativistic throughout (§2.3).

    ``alpha_x``/``alpha_y`` are the Twiss tilts: zero when the bunch sits at its own
    waist (always true of a pure analytic input), nonzero for a fit of a bunch that has
    drifted away from it.

    ``rho_*_gamma`` are correlation coefficients in ``(-1, 1)``. The **angle**-energy pair
    ``rho_thx_gamma``/``rho_thy_gamma`` is the dispersion *derivative*, and it is a
    genuinely independent parameter rather than something derivable from ``alpha`` and
    ``rho_x_gamma``: a bunch created at a waist with dispersion but no dispersion
    derivative keeps ``rho_thx_gamma = 0`` however far it later drifts, while its
    ``alpha_x`` and ``rho_x_gamma`` both change. Storing it is what makes :func:`drift`
    composable — see :func:`_drift_plane`.

    Their joint admissibility is not a per-pair bound but one condition on all five at
    once, since they all constrain the same gamma draw; :func:`validate` checks it via
    :func:`gamma_coefficients`.
    """

    bunch_charge: Quantity  # charge
    kinetic_energy: Quantity  # energy
    rel_energy_spread: float  # dimensionless
    sigma_x: Quantity  # length
    sigma_y: Quantity  # length
    emit_x: Quantity  # length * angle, geometric
    emit_y: Quantity  # length * angle, geometric
    sigma_z: Quantity  # length
    rho_x_gamma: float = 0.0  # dispersion, as a correlation coefficient
    rho_y_gamma: float = 0.0
    rho_z_gamma: float = 0.0  # chirp, as a correlation coefficient
    rho_thx_gamma: float = 0.0  # dispersion derivative, as a correlation coefficient
    rho_thy_gamma: float = 0.0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    fit_quality: dict | None = None

    #: Canonical CGS unit of each dimensioned field. Also what `__post_init__` converts
    #: incoming values into, so `.m` below is always CGS.
    UNITS = {
        "bunch_charge": "statC",
        "kinetic_energy": "erg",
        "sigma_x": "cm",
        "sigma_y": "cm",
        "emit_x": "cm * rad",
        "emit_y": "cm * rad",
        "sigma_z": "cm",
    }

    #: Longitudinal extents, which §2.1 allows to be quoted as either a length or a
    #: duration. Only these opt into the `light_time` equivalence — a transverse size
    #: given in femtoseconds is a mistake, not a unit choice.
    LIGHT_TIME_FIELDS = frozenset({"sigma_z"})

    def __post_init__(self) -> None:
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(
                    getattr(self, name), unit, name, light_time=name in self.LIGHT_TIME_FIELDS
                ),
            )

    def m(self, name: str) -> float:
        """Magnitude of a dimensioned field in its canonical CGS unit.

        The unpack every numeric routine here does once at the top, so the arithmetic
        below it is plain floats — pint never reaches a loop or a kernel (§2.1).
        """
        return float(getattr(self, name).magnitude)

    # -- derived scalars (computed at the point of use, never cached — P9) ---
    def gamma0(self) -> float:
        return 1.0 + self.m("kinetic_energy") / MEC2_CGS

    def beta0(self) -> float:
        g = self.gamma0()
        return math.sqrt(1.0 - 1.0 / (g * g))

    def sigma_gamma(self) -> float:
        """Absolute RMS spread in gamma.

        ``rel_energy_spread`` is relative to the **kinetic** energy, matching how such a
        number is quoted and entered; the conversion to a spread in gamma is therefore
        ``sigma_gamma = rel_spread * E_kin / (m c^2)``, not ``rel_spread * gamma0``.
        """
        return self.rel_energy_spread * self.m("kinetic_energy") / MEC2_CGS

    def divergence_x(self) -> float:
        """RMS angular divergence in x, rad, including the Twiss tilt."""
        return self.m("emit_x") * math.sqrt(1.0 + self.alpha_x**2) / self.m("sigma_x")

    def divergence_y(self) -> float:
        return self.m("emit_y") * math.sqrt(1.0 + self.alpha_y**2) / self.m("sigma_y")

    def beta_star_x(self) -> float:
        """Twiss beta at the current slice, cm."""
        return self.m("sigma_x") ** 2 / self.m("emit_x")

    def beta_star_y(self) -> float:
        return self.m("sigma_y") ** 2 / self.m("emit_y")

    def emit_norm_x(self) -> float:
        return self.beta0() * self.gamma0() * self.m("emit_x")

    def emit_norm_y(self) -> float:
        return self.beta0() * self.gamma0() * self.m("emit_y")

    def n_electrons(self) -> float:
        """Physical electron count implied by the bunch charge.

        Lives here, on the *beam description*, because that is where the charge is. The
        interaction carries the resulting ``N_e`` scalar (§3.5); `Bunch` deliberately has
        no such property, since its weights are relative.
        """
        return self.m("bunch_charge") / E_ESU


def _twiss_tilt_correlation(alpha: float) -> tuple[float, float]:
    """Position-angle correlation from a Twiss tilt, and its complement.

    ``rho = -alpha / sqrt(1 + alpha^2)``, with ``sqrt(1 - rho^2) = 1 / sqrt(1 + alpha^2)``
    returned alongside because every caller needs both.
    """
    root = math.sqrt(1.0 + alpha**2)
    return -alpha / root, 1.0 / root


def gamma_coefficients(beam: "GaussianElectronBeam") -> tuple[float, ...]:
    """How gamma is built from the five position/angle deviates, plus its own.

    The sampler draws ``gamma = gamma0 + sigma_gamma * (a_x n_x + a_y n_y + a_z n_z +
    a_thx n_thx + a_thy n_thy + sqrt(residual) n_gamma)``. Since ``x`` is driven by ``n_x``
    alone, ``a_x`` is simply ``rho_x_gamma``; the angle deviates need a correction because
    ``thx`` is driven by *both* ``n_x`` (through the Twiss tilt) and ``n_thx``.

    Returns ``(a_x, a_y, a_z, a_thx, a_thy, residual)``. A negative ``residual`` means the
    requested correlations cannot coexist — :func:`validate` reports that in the user's own
    terms rather than letting a NaN appear inside the sampler.
    """
    rho_tx, s_x = _twiss_tilt_correlation(beam.alpha_x)
    rho_ty, s_y = _twiss_tilt_correlation(beam.alpha_y)
    a_x, a_y, a_z = beam.rho_x_gamma, beam.rho_y_gamma, beam.rho_z_gamma
    a_thx = (beam.rho_thx_gamma - rho_tx * a_x) / s_x
    a_thy = (beam.rho_thy_gamma - rho_ty * a_y) / s_y
    residual = 1.0 - (a_x**2 + a_y**2 + a_z**2 + a_thx**2 + a_thy**2)
    return a_x, a_y, a_z, a_thx, a_thy, residual


def chirp_to_correlation(chirp: float, sigma_z: float, sigma_gamma: float) -> float:
    """Longitudinal chirp ``dgamma/dz`` (1/cm) → the correlation coefficient stored on the beam."""
    if sigma_gamma <= 0.0 or sigma_z <= 0.0:
        return 0.0
    return chirp * sigma_z / sigma_gamma


def dispersion_to_correlation(dispersion: float, sigma_transverse: float, sigma_gamma: float) -> float:
    """Dispersion ``dx/dgamma`` (cm) → the correlation coefficient stored on the beam."""
    if sigma_transverse <= 0.0:
        return 0.0
    return dispersion * sigma_gamma / sigma_transverse


def validate(beam: GaussianElectronBeam) -> list[str]:
    """Hard-fail on impossible values; return warning strings for suspicious ones."""
    for name in ("bunch_charge", "kinetic_energy", "sigma_x", "sigma_y",
                 "emit_x", "emit_y", "sigma_z"):
        if beam.m(name) <= 0:
            raise ValueError(f"GaussianElectronBeam: {name} must be > 0")
    if beam.rel_energy_spread < 0:
        raise ValueError("GaussianElectronBeam: rel_energy_spread must be >= 0")

    for name in ("rho_x_gamma", "rho_y_gamma", "rho_z_gamma", "rho_thx_gamma", "rho_thy_gamma"):
        if not -1.0 < getattr(beam, name) < 1.0:
            raise ValueError(f"GaussianElectronBeam: {name} must be in (-1, 1)")

    *_, residual = gamma_coefficients(beam)
    explained = 1.0 - residual
    if residual <= 0.0:
        raise ValueError(
            f"GaussianElectronBeam: the requested correlations cannot coexist — together "
            f"they would explain {explained:.4g} of the energy variance, which must stay "
            f"below 1. Reduce the dispersion/chirp correlations (rho_x_gamma="
            f"{beam.rho_x_gamma:.3g}, rho_y_gamma={beam.rho_y_gamma:.3g}, rho_z_gamma="
            f"{beam.rho_z_gamma:.3g}, rho_thx_gamma={beam.rho_thx_gamma:.3g}, "
            f"rho_thy_gamma={beam.rho_thy_gamma:.3g}) or the Twiss tilts."
        )

    warnings: list[str] = []
    if beam.rel_energy_spread > 0.1:
        warnings.append("Relative energy spread above 10% — check whether this is intended.")
    if beam.gamma0() < 10.0:
        warnings.append(
            f"gamma0 = {beam.gamma0():.3g} is not ultrarelativistic; xigma's validity "
            f"regime assumes it is (§2.3)."
        )
    if max(beam.divergence_x(), beam.divergence_y()) > 0.1:
        warnings.append("Angular divergence above 100 mrad — the paraxial treatment is questionable.")
    if beam.m("emit_x") > beam.m("sigma_x") or beam.m("emit_y") > beam.m("sigma_y"):
        warnings.append(
            "Geometric emittance exceeds the beam size. Emittance is a length*angle, not "
            "a length — this usually means a units mix-up (mm*mrad entered as cm*rad)."
        )
    if explained >= 0.95:
        warnings.append(
            f"Correlations explain {explained:.3g} of the energy variance; the conditional "
            f"energy spread will be very small."
        )
    return warnings


# ---------------------------------------------------------------------------
# Macroparticle bunch
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bunch:
    """Macroparticle electron bunch: flat CGS arrays, no units, no conventions.

    ``x``/``y``/``z`` are positions (cm) at whatever slice produced them; ``thx``/``thy``
    are the momentum angles ``px/pz`` and ``py/pz`` (rad), **not** positions; ``gamma`` is
    the per-particle Lorentz factor.

    ``weight`` is a per-particle **relative** weight summing to 1 over an unfiltered
    bunch — uniformly ``1/n`` for a sampled bunch, non-uniform for a loaded file. It is
    per-particle rather than scalar so a loaded distribution with unequal weights needs no
    special case, and so :func:`prefilter_bunch` can drop particles without renormalizing
    (§3.2: the prefilter is a pure optimization, and the sum falling below 1 is the honest
    record of what was dropped).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    thx: np.ndarray
    thy: np.ndarray
    gamma: np.ndarray
    weight: np.ndarray
    meta: dict = field(default_factory=dict)
    gaussian_fit: GaussianElectronBeam | None = None

    #: The unit each array is stored in. Declared as data — the same pattern `Axis` uses
    #: for result slices — rather than as pint-wrapped arrays, for two reasons. The
    #: numeric one: `np.cov`, `np.corrcoef` and `np.linalg.slogdet` have no pint
    #: implementation, and `fit_gaussian` needs all three. The structural one: its 6x6
    #: covariance is **dimensionally heterogeneous** (`Sigma[x,x]` is cm², `Sigma[x,gamma]`
    #: is cm, `Sigma[gamma,gamma]` is dimensionless), so it cannot be one `Quantity` at
    #: all — no units library can carry that without a per-element unit matrix.
    UNITS = {
        "x": "cm", "y": "cm", "z": "cm",
        "thx": "rad", "thy": "rad",
        "gamma": "1", "weight": "1",
    }

    def get(self, name: str, unit: str) -> np.ndarray:
        """One array, converted to ``unit`` — the dimensionally checked way in.

        Conversion here is always a pure scale factor (no offset units exist anywhere in
        this project), so it costs a single vectorized multiply, and **nothing at all**
        when the requested unit is the stored one: the array itself is returned, with no
        copy. That is the case for every CGS consumer, so an engine in the core's own unit
        system pays literally zero for asking properly. A dimension mismatch raises in
        `scale_factor` rather than silently scaling by a wrong number.
        """
        if name not in self.UNITS:
            raise KeyError(f"{name!r} is not a dimensioned Bunch array; expected one of {sorted(self.UNITS)}")
        factor = scale_factor(self.UNITS[name], unit)
        values = getattr(self, name)
        return values if factor == 1.0 else values * factor

    @property
    def n_particles(self) -> int:
        return int(np.asarray(self.x).shape[0])

    def arrays(self) -> tuple[np.ndarray, ...]:
        """The six per-particle physical arrays plus weights, in a fixed order."""
        return (self.x, self.y, self.z, self.thx, self.thy, self.gamma, self.weight)

    def select(self, mask) -> "Bunch":
        """A new bunch keeping only ``mask``-selected particles. Weights are untouched."""
        mask = np.asarray(mask)
        return replace(
            self,
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            thx=self.thx[mask],
            thy=self.thy[mask],
            gamma=self.gamma[mask],
            weight=self.weight[mask],
        )


def momenta(bunch: Bunch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized momenta ``(px, py, pz)`` in units of ``m_e c``, derived from the angles.

    ``gamma^2 = 1 + px^2 + py^2 + pz^2`` holds identically by construction — see the
    module docstring.
    """
    pz = np.sqrt((bunch.gamma**2 - 1.0) / (1.0 + bunch.thx**2 + bunch.thy**2))
    return bunch.thx * pz, bunch.thy * pz, pz


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _substream_deviates(seed: int, n_particles: int) -> dict[str, np.ndarray]:
    """Standard-normal deviates, one independent substream per sampled variable.

    ``SeedSequence.spawn`` gives statistically independent streams from one seed, so each
    variable's deviates depend on the seed and the particle count and on **nothing else**
    — the property the §5 ``REUSE_INTERMEDIATES`` tier rests on.
    """
    children = np.random.SeedSequence(seed).spawn(len(SAMPLED_VARIABLES))
    return {
        name: np.random.default_rng(child).standard_normal(n_particles)
        for name, child in zip(SAMPLED_VARIABLES, children)
    }


def sample_gaussian_bunch(beam: GaussianElectronBeam, n_particles: int, seed: int) -> Bunch:
    """Draw ``n_particles`` macroparticles from ``beam`` using ``seed``.

    Positions and angles are drawn from the beam's sizes, emittances and Twiss tilts;
    gamma is drawn *conditionally* on ``(x, y, z)`` through the stored correlation
    coefficients, which keeps its marginal spread exactly ``sigma_gamma`` however strong
    the correlations are, and keeps the position draws independent of every energy-side
    parameter.

    "Same seed" means "same seed + same beam parameters + same ``n_particles`` ⇒ identical
    bunch" (§3.5), not "same bunch regardless of parameters".
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    validate(beam)

    d = _substream_deviates(seed, n_particles)

    # Unpack to plain CGS floats once; everything below is raw numpy (§2.1).
    x = beam.m("sigma_x") * d["x"]
    y = beam.m("sigma_y") * d["y"]
    z = beam.m("sigma_z") * d["z"]

    # Twiss tilt: at a slice with alpha != 0 the position and angle are correlated,
    # rho = -alpha / sqrt(1 + alpha^2), with the angle drawn conditionally on the position
    # so the position stream stays untouched by alpha.
    thx = _tilted_angle(beam.divergence_x(), beam.alpha_x, d["x"], d["thx"])
    thy = _tilted_angle(beam.divergence_y(), beam.alpha_y, d["y"], d["thy"])

    # gamma is drawn conditionally on all five, which keeps its marginal spread exactly
    # sigma_gamma however strong the correlations are (see `gamma_coefficients`).
    a_x, a_y, a_z, a_thx, a_thy, residual = gamma_coefficients(beam)
    gamma = beam.gamma0() + beam.sigma_gamma() * (
        a_x * d["x"]
        + a_y * d["y"]
        + a_z * d["z"]
        + a_thx * d["thx"]
        + a_thy * d["thy"]
        + math.sqrt(residual) * d["gamma"]
    )

    weight = np.full(n_particles, 1.0 / n_particles)
    return Bunch(
        x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma, weight=weight,
        meta={"seed": seed, "n_particles": n_particles},
        gaussian_fit=beam,
    )


def _tilted_angle(divergence: float, alpha: float, position_deviate, angle_deviate):
    rho, complement = _twiss_tilt_correlation(alpha)
    return divergence * (rho * position_deviate + complement * angle_deviate)


# ---------------------------------------------------------------------------
# Laser overlap and the prefilter (§3.2)
# ---------------------------------------------------------------------------
def overlap_time_window(bunch: Bunch, laser, threshold: float = 1e-3):
    """Per-particle time window ``(t0, t1)`` during which a particle can be in the pulse.

    Generalizes the predecessor's head-on-only ``laser_overlap_time_window`` to arbitrary
    geometry by intersecting each particle's straight-line trajectory with the laser's
    own :meth:`~gammaforge.io.laser.GaussianParaxialLaser.active_region` — so a crossing
    angle needs no special case here, it is already in the region's axis.

    Both bounding conditions are cheap and closed-form: the longitudinal one is linear in
    ``t``, the transverse one quadratic. ``t0 > t1`` marks a particle that never enters
    the region at all, which is exactly what :func:`prefilter_bunch` filters on, and the
    same window is what a temporal-envelope autorange (§3.4) needs.

    Returns ``(t0, t1)`` in seconds, in the lab frame.
    """
    region = laser.active_region(threshold)
    axis = np.asarray(region.axis, dtype=float)

    # Ultrarelativistic straight-line motion at speed c (§2.3).
    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    vx, vy, vz = (C_CGS * bunch.thx / norm, C_CGS * bunch.thy / norm, C_CGS / norm)

    dx = bunch.x - region.origin[0]
    dy = bunch.y - region.origin[1]
    dz = bunch.z - region.origin[2]

    # --- longitudinal: |u(t) - c t| <= half_length, linear in t ---
    u0 = dx * axis[0] + dy * axis[1] + dz * axis[2]
    slope = vx * axis[0] + vy * axis[1] + vz * axis[2] - C_CGS
    t_lo, t_hi = _linear_band(u0, slope, region.half_length)

    # --- transverse: |d(t)|^2 - (d(t).axis)^2 <= radius^2, quadratic in t ---
    v_par = vx * axis[0] + vy * axis[1] + vz * axis[2]
    a = (vx**2 + vy**2 + vz**2) - v_par**2
    b = 2.0 * ((dx * vx + dy * vy + dz * vz) - u0 * v_par)
    c = (dx**2 + dy**2 + dz**2) - u0**2 - region.radius**2
    q_lo, q_hi = _quadratic_band(a, b, c)

    return np.maximum(t_lo, q_lo), np.minimum(t_hi, q_hi)


def _linear_band(offset, slope, half_width):
    """Solve ``|offset + slope*t| <= half_width`` for ``t``."""
    degenerate = np.abs(slope) < 1e-30
    safe = np.where(degenerate, 1.0, slope)
    ta = (-half_width - offset) / safe
    tb = (half_width - offset) / safe
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
    # Zero slope: the condition is t-independent — either always or never satisfied.
    always = np.abs(offset) <= half_width
    lo = np.where(degenerate, np.where(always, -np.inf, np.inf), lo)
    hi = np.where(degenerate, np.where(always, np.inf, -np.inf), hi)
    return lo, hi


def _quadratic_band(a, b, c):
    """Solve ``a t^2 + b t + c <= 0`` for ``t``, with ``a >= 0``."""
    # a = c^2 sin^2(angle between velocity and laser axis): it vanishes for exactly
    # (anti)parallel motion, where the transverse distance never changes with time.
    scale = np.maximum(np.abs(a), np.abs(b) + np.abs(c) + 1e-300)
    degenerate = a <= 1e-12 * scale
    safe_a = np.where(degenerate, 1.0, a)
    disc = b * b - 4.0 * safe_a * c
    has_root = disc >= 0.0
    sqrt_disc = np.sqrt(np.where(has_root, disc, 0.0))
    lo = (-b - sqrt_disc) / (2.0 * safe_a)
    hi = (-b + sqrt_disc) / (2.0 * safe_a)
    lo = np.where(has_root, lo, np.inf)
    hi = np.where(has_root, hi, -np.inf)
    always = c <= 0.0
    lo = np.where(degenerate, np.where(always, -np.inf, np.inf), lo)
    hi = np.where(degenerate, np.where(always, np.inf, -np.inf), hi)
    return lo, hi


def prefilter_bunch(bunch: Bunch, laser, threshold: float = 1e-3) -> Bunch:
    """Drop macroparticles whose trajectories never enter the laser's active region.

    A **pure optimization** (§3.2): the discarded particles contribute ``L = 0`` to every
    engine, weights are never renormalized, and ``N_e`` is untouched — so with the same
    seed, results are identical with the prefilter on or off. That invariance is a tested
    property, not an aspiration (§7).
    """
    t0, t1 = overlap_time_window(bunch, laser, threshold)
    return bunch.select(t0 <= t1)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def _drift_plane(
    sigma: float, emit: float, alpha: float, rho_gamma: float, rho_angle_gamma: float, length: float
) -> tuple[float, float, float]:
    """Transport one transverse plane's ``(sigma, alpha, rho_gamma)`` through a drift.

    Emittance is invariant (Liouville), so the standard Twiss drift applies::

        gamma_twiss = (1 + alpha^2) / beta,   beta = sigma^2 / emit
        beta  -> beta - 2 alpha L + gamma_twiss L^2
        alpha -> alpha - gamma_twiss L

    Those two are exact identities on **second moments**, not just statements about a
    Gaussian parent, so they hold to round-off for any bunch.

    The energy correlation transports as ``cov(x, gamma) -> cov(x, gamma) + L cov(x',
    gamma)``, and the angle-energy correlation is unchanged because a drift changes
    neither the angle nor gamma. That is why ``rho_thx_gamma`` has to be a stored
    parameter: an earlier version re-derived ``cov(x', gamma)`` from ``alpha`` at each
    step, on the assumption that gamma couples to the angle only through the position.
    That assumption is true of a freshly sampled bunch and **destroyed by the first
    drift**, so a single drift came out right while two consecutive drifts silently did
    not compose — the bug that `test_drift_composes` now guards.
    """
    beta = sigma**2 / emit
    gamma_twiss = (1.0 + alpha**2) / beta
    beta_new = beta - 2.0 * alpha * length + gamma_twiss * length**2
    sigma_new = math.sqrt(emit * beta_new)
    alpha_new = alpha - gamma_twiss * length
    # cov(x, gamma) / sigma_gamma, before and after; sigma_gamma is unchanged by a drift,
    # and so is the angle spread that scales the angle-energy correlation.
    sigma_angle = emit * math.sqrt(1.0 + alpha**2) / sigma
    cov_new = rho_gamma * sigma + length * rho_angle_gamma * sigma_angle
    return sigma_new, alpha_new, cov_new / sigma_new


def _drift_fit(fit: GaussianElectronBeam, length: float) -> GaussianElectronBeam:
    """Analytically carry a beam description through a field-free drift — no refit."""
    sigma_x, alpha_x, rho_x = _drift_plane(
        fit.m("sigma_x"), fit.m("emit_x"), fit.alpha_x, fit.rho_x_gamma, fit.rho_thx_gamma, length
    )
    sigma_y, alpha_y, rho_y = _drift_plane(
        fit.m("sigma_y"), fit.m("emit_y"), fit.alpha_y, fit.rho_y_gamma, fit.rho_thy_gamma, length
    )
    return replace(
        fit,
        sigma_x=Quantity(sigma_x, "cm"), alpha_x=alpha_x, rho_x_gamma=rho_x,
        sigma_y=Quantity(sigma_y, "cm"), alpha_y=alpha_y, rho_y_gamma=rho_y,
    )


def drift(bunch: Bunch, length) -> Bunch:
    """Ballistically advance by a longitudinal distance ``length`` (cm).

    ``x += thx*L``, ``y += thy*L``; ``z``, angles and gamma unchanged. This is the
    mechanism that produces a Twiss tilt from a bunch sampled at its waist. A scalar
    ``length`` also advances the attached description in lockstep; a per-particle array
    does not, since an ensemble tilt is not defined for it (:func:`propagate` handles
    that case itself).
    """
    length_arr = np.asarray(length, dtype=float)
    new_fit = bunch.gaussian_fit
    if new_fit is not None and length_arr.ndim == 0:
        new_fit = _drift_fit(new_fit, float(length_arr))
    return replace(
        bunch,
        x=bunch.x + bunch.thx * length_arr,
        y=bunch.y + bunch.thy * length_arr,
        gaussian_fit=new_fit,
    )


def propagate(bunch: Bunch, dt) -> Bunch:
    """Advance every macroparticle by a time offset ``dt`` (s, scalar or per-particle).

    Each particle covers ``L_i = vz_i * c * dt`` with ``vz = 1/sqrt(1 + thx^2 + thy^2)``,
    which is :func:`drift`'s own x/y push with a per-particle length, plus the ``z``
    advance that :func:`drift` — being z-parameterized — does not make. The attached
    description is carried by the *ensemble* reference step ``c * mean(dt)``.
    """
    vz = 1.0 / np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    length = vz * C_CGS * np.asarray(dt, dtype=float)

    moved = drift(bunch, length)
    new_fit = moved.gaussian_fit
    if new_fit is not None:
        new_fit = _drift_fit(new_fit, C_CGS * float(np.mean(np.asarray(dt, dtype=float))))
    return replace(moved, z=bunch.z + length, gaussian_fit=new_fit)


def stream(bunch: Bunch, t_grid) -> Iterator[Bunch]:
    """Yield a snapshot of ``bunch`` at each time in ``t_grid`` (s).

    Each snapshot is computed from the original bunch, not from its predecessor, so the
    grid need not be evenly spaced and no error accumulates.
    """
    for t in t_grid:
        yield propagate(bunch, t)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _chi2_6_cdf(x):
    """CDF of a chi-squared distribution with 6 degrees of freedom.

    Closed form for even degrees of freedom: ``1 - e^{-x/2} sum_{j<3} (x/2)^j / j!``.
    Written out rather than pulled from scipy so `gammaforge.io` keeps its dependency
    surface to what `pyproject.toml` already declares.
    """
    h = np.asarray(x, dtype=float) / 2.0
    return 1.0 - np.exp(-h) * (1.0 + h + h * h / 2.0)


def _ks_statistic(samples, cdf) -> float:
    """One-sample Kolmogorov-Smirnov statistic: ``max |ECDF - CDF|``."""
    ordered = np.sort(np.asarray(samples, dtype=float))
    n = ordered.size
    theoretical = cdf(ordered)
    upper = np.arange(1, n + 1) / n - theoretical
    lower = theoretical - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def evaluate_fit_quality(
    data: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, n_synthetic: int = 3, seed: int = 0
) -> dict:
    """Fit-quality metrics with a sampling-noise baseline.

    A KS statistic on its own cannot distinguish "the data isn't Gaussian" from "there
    are only 10⁴ particles". So the same statistics are also computed on synthetic draws
    from the *fitted* Gaussian: metrics that match the synthetic baseline mean the fit is
    noise-limited (good), metrics well above it mean genuine model mismatch.

    ``seed`` is explicit and defaulted rather than left to a fresh entropy draw, so a fit
    is reproducible — the same guarantee `sample_gaussian_bunch` gives (§7).
    """
    centered = data - mu
    try:
        inverse = np.linalg.inv(sigma)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive-definite covariance")
    except np.linalg.LinAlgError:
        regularized = sigma + 1e-12 * np.trace(sigma) / sigma.shape[0] * np.eye(sigma.shape[0])
        inverse = np.linalg.inv(regularized)
        sign, logdet = np.linalg.slogdet(regularized)

    k = data.shape[1]

    def metrics(centered_sample):
        d2 = np.einsum("ni,ij,nj->n", centered_sample, inverse, centered_sample)
        return (
            _ks_statistic(d2, _chi2_6_cdf),
            float(np.mean(d2)),
            float(-0.5 * (k * np.log(2.0 * np.pi) + logdet + np.mean(d2))),
        )

    ks_real, d2_real, loglik_real = metrics(centered)

    rng = np.random.default_rng(seed)
    synthetic = [metrics(rng.multivariate_normal(np.zeros(k), sigma, size=data.shape[0]))
                 for _ in range(n_synthetic)]
    ks_syn = float(np.mean([m[0] for m in synthetic]))

    return {
        "ks_real": ks_real,
        "ks_synthetic": ks_syn,
        "ks_excess": ks_real - ks_syn,
        "mean_d2_real": d2_real,
        "mean_d2_synthetic": float(np.mean([m[1] for m in synthetic])),
        "log_likelihood_real": loglik_real,
        "log_likelihood_synthetic": float(np.mean([m[2] for m in synthetic])),
        "n_synthetic": n_synthetic,
        "seed": seed,
    }


def fit_gaussian(bunch: Bunch, *, bunch_charge: Quantity, seed: int = 0) -> GaussianElectronBeam:
    """Fit a `GaussianElectronBeam` to raw macroparticles (P8: same type as the input).

    Second-moment based: Twiss alpha/beta from the transverse sub-covariances (giving
    waist-referenced sizes and emittances), the longitudinal spread, and the three
    energy correlations. Uses the biased (population) covariance, since these are
    population parameters, not estimates of a parent distribution.

    ``bunch_charge`` is a required argument rather than something read off the bunch:
    weights are relative (§3.2), so the physical charge genuinely is not recoverable from
    a `Bunch` — passing it explicitly is what keeps that honest.
    """
    data = np.stack([bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma], axis=1)
    mu = np.mean(data, axis=0)
    sigma = np.cov(data - mu, rowvar=False, bias=True)

    ix, ixp, iy, iyp, iz, ig = range(6)

    emit_x = math.sqrt(max(sigma[ix, ix] * sigma[ixp, ixp] - sigma[ix, ixp] ** 2, 0.0))
    emit_y = math.sqrt(max(sigma[iy, iy] * sigma[iyp, iyp] - sigma[iy, iyp] ** 2, 0.0))

    sigma_x = math.sqrt(sigma[ix, ix])
    sigma_y = math.sqrt(sigma[iy, iy])
    sigma_z = math.sqrt(sigma[iz, iz])
    sigma_gamma = math.sqrt(sigma[ig, ig])
    gamma0 = float(mu[ig])

    def correlation(i: int, sigma_i: float) -> float:
        if sigma_i <= 0.0 or sigma_gamma <= 0.0:
            return 0.0
        return float(sigma[i, ig] / (sigma_i * sigma_gamma))

    kinetic_energy = (gamma0 - 1.0) * MEC2_CGS
    return GaussianElectronBeam(
        bunch_charge=bunch_charge,
        kinetic_energy=Quantity(kinetic_energy, "erg"),
        rel_energy_spread=sigma_gamma * MEC2_CGS / kinetic_energy if kinetic_energy > 0 else 0.0,
        sigma_x=Quantity(sigma_x, "cm"),
        sigma_y=Quantity(sigma_y, "cm"),
        emit_x=Quantity(emit_x, "cm * rad"),
        emit_y=Quantity(emit_y, "cm * rad"),
        sigma_z=Quantity(sigma_z, "cm"),
        rho_x_gamma=correlation(ix, sigma_x),
        rho_y_gamma=correlation(iy, sigma_y),
        rho_z_gamma=correlation(iz, sigma_z),
        rho_thx_gamma=correlation(ixp, math.sqrt(sigma[ixp, ixp])),
        rho_thy_gamma=correlation(iyp, math.sqrt(sigma[iyp, iyp])),
        alpha_x=-sigma[ix, ixp] / emit_x if emit_x > 0 else 0.0,
        alpha_y=-sigma[iy, iyp] / emit_y if emit_y > 0 else 0.0,
        fit_quality=evaluate_fit_quality(data, mu, sigma, seed=seed),
    )

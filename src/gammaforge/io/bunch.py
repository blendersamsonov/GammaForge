"""Electron beam and macroparticle bunch (GRAND_PLAN.md §3.2).

* :class:`GaussianElectronBeam` — the analytic 6D Gaussian description, CGS-Gaussian.
  It is **both** the input description and the output of a fit (P8): there is no
  ``BeamFittedParams`` three-way split, and ``fit_quality`` is simply ``None`` when the
  description came from a user rather than from :func:`fit_gaussian`.
* :class:`Bunch` — raw macroparticle arrays. ``weight`` is **relative**; the physical
  electron count ``N_e`` is a scalar on the interaction (§3.5), never recoverable from
  the bunch.

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
import numbers
from dataclasses import dataclass, field, replace
from typing import Iterator, Mapping

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
    "luminosity_weights",
    "peak_illumination",
    "illumination_window",
    "prefilter_by_illumination",
    "illumination_report",
    "prefilter_by_luminosity",
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
        scalar_fields = (
            "rel_energy_spread", "rho_x_gamma", "rho_y_gamma", "rho_z_gamma",
            "rho_thx_gamma", "rho_thy_gamma", "alpha_x", "alpha_y",
        )
        for name in scalar_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                raise ValueError(f"GaussianElectronBeam: {name} must be a finite scalar, got {value!r}")
        for name in self.UNITS:
            if not math.isfinite(self.m(name)):
                raise ValueError(f"GaussianElectronBeam: {name} must be finite, got {getattr(self, name)!r}")

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

    def __post_init__(self) -> None:
        """Take ownership of particle vectors and freeze their shape/content.

        A frozen dataclass does not freeze ndarray payloads. Copying at this boundary
        prevents a caller's later mutation from changing an interaction held by a
        runner/cache; read-only flags prevent mutation through the returned Bunch.
        Empty vectors remain valid, which is required by analytical and filtered paths.
        """
        names = ("x", "y", "z", "thx", "thy", "gamma", "weight")
        arrays: dict[str, np.ndarray] = {}
        lengths: dict[str, int] = {}
        for name in names:
            values = np.asarray(getattr(self, name))
            if values.ndim != 1:
                raise ValueError(f"Bunch: {name} must be a one-dimensional particle array")
            copied = np.array(values, dtype=float, copy=True)
            copied.setflags(write=False)
            arrays[name] = copied
            lengths[name] = copied.shape[0]
        if len(set(lengths.values())) != 1:
            detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(f"Bunch: particle arrays must have equal lengths ({detail})")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "meta", dict(self.meta))

    @classmethod
    def _from_owned(
        cls,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        thx: np.ndarray,
        thy: np.ndarray,
        gamma: np.ndarray,
        weight: np.ndarray,
        meta: Mapping | None = None,
        gaussian_fit: GaussianElectronBeam | None = None,
    ) -> "Bunch":
        """Construct from fresh internal arrays without another large defensive copy.

        This is private: public construction always snapshots caller-owned arrays. The
        sampler and pure Bunch transforms use it only after creating fresh arrays, then
        mark those arrays read-only just as the public boundary does.
        """
        arrays = {name: np.asarray(values) for name, values in {
            "x": x, "y": y, "z": z, "thx": thx, "thy": thy, "gamma": gamma, "weight": weight,
        }.items()}
        lengths = {name: values.shape[0] for name, values in arrays.items() if values.ndim == 1}
        if len(lengths) != len(arrays) or len(set(lengths.values())) != 1:
            raise ValueError("Bunch: owned particle arrays must be one-dimensional and equally sized")
        result = object.__new__(cls)
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(result, name, values)
        object.__setattr__(result, "meta", dict(meta or {}))
        object.__setattr__(result, "gaussian_fit", gaussian_fit)
        return result

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
        return type(self)._from_owned(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            thx=self.thx[mask],
            thy=self.thy[mask],
            gamma=self.gamma[mask],
            weight=self.weight[mask],
            meta=self.meta,
            gaussian_fit=self.gaussian_fit,
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
    return Bunch._from_owned(
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

    Handles arbitrary geometry by intersecting each particle's straight-line trajectory with the laser's
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
    v_par = vx * axis[0] + vy * axis[1] + vz * axis[2]
    slope = v_par - C_CGS
    t_lo, t_hi = _linear_band(u0, slope, region.half_length)

    # --- transverse: |d(t)|^2 - (d(t).axis)^2 <= radius^2, quadratic in t ---
    # The region is a cone (§3.2), so its radius depends on where along the axis the
    # encounter happens. Evaluating it at the widest point of *this particle's own*
    # longitudinal window keeps the test conservative and still closed-form — a cone
    # inequality in t is not a single quadratic band, and a global worst-case radius would
    # throw away most of the filter's value on a long bunch.
    radius = region.radius_at(_widest_reach(u0, v_par, t_lo, t_hi))
    a = (vx**2 + vy**2 + vz**2) - v_par**2
    b = 2.0 * ((dx * vx + dy * vy + dz * vz) - u0 * v_par)
    c = (dx**2 + dy**2 + dz**2) - u0**2 - radius**2
    q_lo, q_hi = _quadratic_band(a, b, c)

    return np.maximum(t_lo, q_lo), np.minimum(t_hi, q_hi)


def _widest_reach(u0, v_par, t_lo, t_hi):
    """Largest ``|u|`` a particle can have while inside the longitudinal window.

    ``u`` is linear in ``t``, so its extremes over ``[t_lo, t_hi]`` are at the ends. An
    unbounded window — a particle travelling with the pulse rather than through it —
    yields infinity, which widens the cone to everything and therefore keeps the particle:
    the conservative answer.
    """
    bounded = np.isfinite(t_lo) & np.isfinite(t_hi)
    ends = np.where(bounded, t_lo, 0.0), np.where(bounded, t_hi, 0.0)
    reach = np.maximum(np.abs(u0 + v_par * ends[0]), np.abs(u0 + v_par * ends[1]))
    return np.where(bounded, reach, np.inf)


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


def luminosity_weights(bunch: Bunch, laser, iterations: int = 2) -> np.ndarray:
    """Each macroparticle's expected contribution to the luminosity, in arbitrary units.

    ``w_i = Int dt n_L(r_i + v_i t, t) (c - v_i . k_hat)`` — the rate at which particle
    ``i`` actually produces photons, integrated over its whole trajectory. Freezing the
    spot sizes makes the integrand Gaussian in ``t``, so this closes in **closed form**:
    one pass over the arrays, no time stepping, `O(n_particles)` with a small constant.

    The frozen widths are evaluated at each particle's own closest approach, found by
    iterating the stationary point ``t* = -b/a`` a couple of times — enough for a
    *ranking*, which is all this is for.

    Intended as a relevance measure for :func:`prefilter_by_luminosity`. It is **not** a
    photon count: the normalization is dropped where it is common to all particles, so only
    ratios between weights are meaningful.
    """
    from .laser import fit_gaussian_paraxial

    metrics = fit_gaussian_paraxial(laser)
    k_hat, f1, f2 = metrics.focusing_axes()
    px, py, pz = momenta(bunch)
    vx, vy, vz = (C_CGS * p / bunch.gamma for p in (px, py, pz))

    xi1 = f1[0] * bunch.x + f1[1] * bunch.y + f1[2] * bunch.z
    xi2 = f2[0] * bunch.x + f2[1] * bunch.y + f2[2] * bunch.z
    u0 = k_hat[0] * bunch.x + k_hat[1] * bunch.y + k_hat[2] * bunch.z
    d1 = f1[0] * vx + f1[1] * vy + f1[2] * vz
    d2 = f2[0] * vx + f2[1] * vy + f2[2] * vz
    du = k_hat[0] * vx + k_hat[1] * vy + k_hat[2] * vz
    s_ct = metrics.sigma_ct()

    u_eval = np.zeros_like(bunch.x)
    for _ in range(max(1, iterations)):
        s1, s2 = metrics.spot_sizes(u_eval)
        a = d1**2 / s1**2 + d2**2 / s2**2 + (du - C_CGS) ** 2 / s_ct**2
        b = xi1 * d1 / s1**2 + xi2 * d2 / s2**2 + u0 * (du - C_CGS) / s_ct**2
        c = xi1**2 / s1**2 + xi2**2 / s2**2 + u0**2 / s_ct**2
        t_star = -b / a
        u_eval = u0 + (du + metrics.beta_ff * C_CGS) * t_star
    return (C_CGS - du) / (s1 * s2 * np.sqrt(a)) * np.exp(0.5 * (b**2 / a - c))


def _illumination_quadratic(bunch: Bunch, laser, iterations: int = 2):
    """``(peak_fraction, t_star, curvature)`` for each macroparticle, in closed form.

    Along a straight trajectory with the spot sizes frozen, the photon density is
    ``exp(-(a t^2 + 2 b t + c) / 2)`` times a brightness factor, so everything about a
    particle's encounter with the pulse follows from one quadratic:

    * its peak illumination is at ``t_star = -b / a``,
    * the curvature ``a`` sets how fast it enters and leaves.

    The frozen widths are evaluated at each particle's own closest approach, iterated
    twice. One vectorized pass, no time stepping — `O(n_particles)`.
    """
    from .laser import fit_gaussian_paraxial

    metrics = fit_gaussian_paraxial(laser)
    k_hat, f1, f2 = metrics.focusing_axes()
    px, py, pz = momenta(bunch)
    vx, vy, vz = (C_CGS * p / bunch.gamma for p in (px, py, pz))

    x = bunch.x - metrics.m("x_off")
    y = bunch.y - metrics.m("y_off")
    ct_off = C_CGS * metrics.m("t_off")
    xi1 = f1[0] * x + f1[1] * y + f1[2] * bunch.z
    xi2 = f2[0] * x + f2[1] * y + f2[2] * bunch.z
    u0 = k_hat[0] * x + k_hat[1] * y + k_hat[2] * bunch.z
    d1 = f1[0] * vx + f1[1] * vy + f1[2] * vz
    d2 = f2[0] * vx + f2[1] * vy + f2[2] * vz
    du = k_hat[0] * vx + k_hat[1] * vy + k_hat[2] * vz
    s_ct = metrics.sigma_ct()

    u_eval = np.zeros_like(bunch.x)
    for _ in range(max(1, iterations)):
        s1, s2 = metrics.spot_sizes(u_eval)
        a = d1**2 / s1**2 + d2**2 / s2**2 + (du - C_CGS) ** 2 / s_ct**2
        b = xi1 * d1 / s1**2 + xi2 * d2 / s2**2 + (u0 + ct_off) * (du - C_CGS) / s_ct**2
        c = xi1**2 / s1**2 + xi2**2 / s2**2 + (u0 + ct_off) ** 2 / s_ct**2
        t_star = -b / a
        u_eval = u0 + (du + metrics.beta_ff * C_CGS) * t_star
    # Density relative to the pulse's own maximum: the exponential, times the 1/(s1 s2)
    # amplitude decay that makes the bright region so much smaller than the geometric one.
    brightness = (metrics.m("sigma_x") * metrics.m("sigma_y")) / (s1 * s2)
    return brightness * np.exp(0.5 * (b**2 / a - c)), t_star, a


def peak_illumination(bunch: Bunch, laser, iterations: int = 2) -> np.ndarray:
    """The highest photon density each macroparticle ever meets, as a fraction of the
    pulse's own peak — "how far into the production region does this particle get".

    The region where Compton photons are actually produced is where the pulse is *bright*,
    and that is **not** the region the pulse geometrically occupies. Away from focus the
    spot grows but dims as ``1 / (s1 s2)``, so a particle can sit well inside the diverged
    beam and see almost nothing. `GaussianParaxialLaser.active_region` deliberately ignores
    that decay (its docstring says so) because a bound may only ever err towards keeping
    particles — correct, but it leaves the cone keeping many particles that contribute
    nothing.

    Returns a dimensionless ratio in ``[0, 1]``: 1 for a particle passing exactly through
    the focus at the peak of the pulse, and underflowing to 0 for one that never
    meaningfully meets the pulse at all.
    """
    return _illumination_quadratic(bunch, laser, iterations)[0]


def illumination_window(bunch: Bunch, laser, threshold: float = 1e-6, iterations: int = 2):
    """Per-particle ``(t0, t1)``: when each macroparticle is actually being illuminated.

    The same quadratic that gives :func:`peak_illumination` also says *when* a particle is
    above the threshold, because ``density >= threshold`` is one inequality in ``t``:

        a t^2 + 2 b t + c <= 2 ln(brightness / threshold)

    whose solution is ``t_star +- sqrt(2 ln(peak / threshold) / a)``. So the window and the
    filter are the same computation — a particle is worth keeping exactly when its window
    is non-empty — and both cost one vectorized pass.

    **Why this matters more than the filter.** An engine samples each trajectory with a
    *fixed* number of steps between ``t0`` and ``t1``, so the window's width sets the step
    size. `overlap_time_window` returns the interval during which a particle is inside the
    laser's geometric `~gammaforge.io.laser.ActiveRegion`, which is a conservative bound and
    therefore far wider than the interval where anything actually happens — every step spent
    outside is a step not spent resolving the interaction. This window brackets the
    illuminated stretch itself, so the same step budget lands where the physics is.

    ``t0 > t1`` marks a particle that never reaches the threshold, exactly as
    `overlap_time_window` does, so it drops into the same filtering idiom.

    Times are seconds in the lab frame. Note this is an *estimate*, not a bound: the widths
    are frozen, so a particle's true illuminated stretch can extend slightly past the
    window. That is the same tolerance contract as :func:`prefilter_by_illumination`, and
    the reason `overlap_time_window` remains what `prefilter_bunch`'s exact invariance uses.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"illumination_window: threshold must be in (0, 1), got {threshold}")
    peak, t_star, curvature = _illumination_quadratic(bunch, laser, iterations)
    with np.errstate(divide="ignore", invalid="ignore"):
        half = np.sqrt(2.0 * np.log(peak / threshold) / curvature)
    half = np.where(np.isfinite(half), half, -1.0)  # NaN <=> never above threshold
    return t_star - half, t_star + half


def prefilter_by_illumination(bunch: Bunch, laser, threshold: float = 1e-6) -> Bunch:
    """Drop macroparticles that never reach the region where photons are actually produced.

    Same *shape* of contract as `prefilter_bunch` — a threshold on intensity, a region test,
    over-inclusive by construction — but the region is the pulse's **bright** volume rather
    than a geometric cone around it, so it is far tighter wherever the pulse diverges. A
    particle is kept if the photon density it meets ever exceeds ``threshold`` times the
    pulse's own peak (:func:`peak_illumination`).

    This is the natural filter now that the collision profiles are available analytically:
    it asks the question the cone approximates, and answers it in closed form. Against the
    cone at matched threshold, on a 400 um bunch meeting a 4 um / 1 ps pulse with displaced
    foci, it keeps a small fraction of what the cone does at a comparable induced error.

    .. warning::

       **The threshold does not mean what the same number means for `prefilter_bunch`.**
       That one takes a bound: `1e-3` there is safe by construction. Here `1e-3` induces a
       **44% error** on the scenario this function exists for, because many particles that
       are individually dim still sum to a large contribution. The default is therefore
       `1e-6`, which measures at 3.5e-4. Reasoning by analogy with the cone's threshold is
       the mistake to avoid.

    Not a replacement for `prefilter_bunch`'s exact invariance: like
    :func:`prefilter_by_luminosity` it drops small-but-nonzero contributions, so the answer
    moves by roughly ``threshold``. Unlike that one it thresholds *peak illumination* rather
    than *integrated contribution*, which is what makes it a region test — cheaper to reason
    about, and independent of how long a particle dwells in the beam.
    """
    t0, t1 = illumination_window(bunch, laser, threshold)
    return bunch.select(t0 <= t1)


def illumination_report(bunch: Bunch, laser, threshold: float = 1e-6) -> dict:
    """What `prefilter_by_illumination` would discard, and how much to trust that estimate.

    A **pre-engine diagnostic**, meant to be shown to a user before anything expensive runs
    so that dropping charge is their decision rather than a silent default. It answers two
    questions that have to be read together:

    * ``charge_below`` — the fraction of *charge* (not of particles: `Bunch.weight` is
      relative, and an imported bunch need not be uniformly weighted) that never reaches
      ``threshold`` of the pulse's peak illumination, and would therefore be dropped.
    * ``ks_excess`` — how far the bunch is from the Gaussian description the estimate rests
      on, above sampling noise, taken from `GaussianElectronBeam.fit_quality`. ``None`` when
      the beam is analytic, which is a *stronger* statement than ignorance: such a bunch is
      Gaussian by construction, so the estimate carries no model error at all.

    The pairing is the point. Illumination is computed from a Gaussian picture of the pulse,
    so a bunch that fits a Gaussian badly is exactly the case where the tails — the charge
    this would discard — are least well described. "10% of the charge is below threshold" is
    a different decision at ``ks_excess`` of 0.005 than at 0.2, and neither number alone
    says which.

    Unlike the analytical engine's own quantities this is `O(n_particles)`: it inspects real
    macroparticles, which is the only way to answer "how much of *this* bunch". That cost
    belongs to a diagnostic, not to any engine's estimate path.
    """
    fraction = peak_illumination(bunch, laser)
    below = fraction < threshold
    total_weight = float(np.sum(bunch.weight))
    quality = getattr(bunch.gaussian_fit, "fit_quality", None) if bunch.gaussian_fit else None
    return {
        "threshold": threshold,
        "charge_below": float(np.sum(bunch.weight[below]) / total_weight) if total_weight > 0 else 0.0,
        "particles_below": int(np.count_nonzero(below)),
        "n_particles": bunch.n_particles,
        "median_illumination": float(np.median(fraction)),
        "max_illumination": float(np.max(fraction)) if fraction.size else 0.0,
        "ks_excess": None if quality is None else float(quality["ks_excess"]),
        "gaussian_by_construction": quality is None,
    }


def prefilter_by_luminosity(bunch: Bunch, laser, epsilon: float = 1e-4) -> Bunch:
    """Keep the macroparticles carrying all but ``epsilon`` of the total luminosity weight.

    **A different contract from `prefilter_bunch`, deliberately.** That one drops only
    particles a geometric bound proves contribute exactly zero, so results are bit-identical
    with it on or off — a tested invariance. This one drops particles that contribute a
    little, so it *does* move the answer, and is a tolerance rather than an optimization.
    Both exist because they are good at different things; this is not a replacement.

    The contract is deliberately stated in terms of the weights, not the answer: it drops
    the particles whose **frozen-width weight** sums to less than ``epsilon`` of the total.
    The induced error on a yield is of the same order but is not guaranteed to equal
    ``epsilon`` — measured at 1.2e-3 for ``epsilon = 1e-4`` on a wide-bunch scenario.

    Worth it exactly when the geometric cone is loose, which is when the collision is
    mismatched or the foci are displaced — the cone must widen conservatively there, while
    a relevance measure does not. Measured on a 400 um bunch against a 4 um / 1 ps pulse
    with displaced foci: the cone keeps 94% of particles, this keeps **31%** at 1.3e-4
    induced error. On a well-matched collision there is no headroom at all — the cone
    already keeps everything, and so should this.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"prefilter_by_luminosity: epsilon must be in (0, 1), got {epsilon}")
    weights = luminosity_weights(bunch, laser)
    total = float(np.sum(weights))
    if total <= 0.0:
        return bunch
    order = np.argsort(weights)[::-1]
    keep_count = int(np.searchsorted(np.cumsum(weights[order]) / total, 1.0 - epsilon)) + 1
    mask = np.zeros(bunch.n_particles, dtype=bool)
    mask[order[:keep_count]] = True
    return bunch.select(mask)


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
    return Bunch._from_owned(
        x=bunch.x + bunch.thx * length_arr,
        y=bunch.y + bunch.thy * length_arr,
        z=bunch.z,
        thx=bunch.thx,
        thy=bunch.thy,
        gamma=bunch.gamma,
        weight=bunch.weight,
        meta=bunch.meta,
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
    return Bunch._from_owned(
        x=moved.x,
        y=moved.y,
        z=bunch.z + length,
        thx=moved.thx,
        thy=moved.thy,
        gamma=moved.gamma,
        weight=moved.weight,
        meta=moved.meta,
        gaussian_fit=new_fit,
    )


def stream(bunch: Bunch, t_grid) -> Iterator[Bunch]:
    """Yield a snapshot of ``bunch`` at each time in ``t_grid`` (s).

    Each snapshot is computed from the original bunch, not from the prior snapshot, so the
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

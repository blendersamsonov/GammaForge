"""CGS-Gaussian constants, the pint registry, and the width-convention vocabulary.

**One unit system in the shared core: CGS-Gaussian** (GRAND_PLAN.md §2.1/P1) — cm, s, g,
erg, statC, gauss/statV·cm⁻¹.

**Canonical values, dimensioned types.** The two are separate decisions and this project
makes both. Every physics dataclass *stores* canonical CGS (P1 stays literally true), and
every dimensioned field is *typed* as a pint ``Quantity`` (§2.1). The typing is what makes
a unit mistake at an engine boundary an exception rather than a silent factor-of-100 error
in the answer — and that boundary is real and permanent, since kascade is SI internally
(§4.4). P1's single-unit-system rule removes the *multiplicity* of systems; it does not
reach the one conversion each engine must still perform, and this does.

**pint stops at the engine boundary.** An engine unpacks once, at ``run()``, and everything
below is plain floats: kernels never see a ``Quantity`` (§2.1), and neither does any loop
here. Bulk per-particle arrays are not wrapped at all — they declare their unit as data
(`Bunch.UNITS`, the same pattern `Axis` uses for result slices) and convert through a
checked scale factor, because a 6D beam covariance is dimensionally heterogeneous
(``Σ[x,x]`` is cm², ``Σ[x,γ]`` cm, ``Σ[γ,γ]`` dimensionless) and cannot be one ``Quantity``
in any units library.

**Why the charge conversion is hand-coded.** CGS-Gaussian is not a unit rescaling of SI
for electromagnetic quantities — charge has different dimensionality in the two systems
(``[M]^½ [L]^(3/2) [T]^-1`` vs ``[I][T]``), so pint refuses ``elementary_charge →
franklin`` outright. The exact textbook conversion ``1 C = c[cm/s] / 10 statC`` is
applied by hand in :data:`E_ESU`; every other constant is a straight pint conversion into
a named CGS unit and therefore self-verifying against CODATA.

**Conventions.** A transverse size or a longitudinal extent is a fuzzy physical width
that a single number only pins down once you say *which* definition was used: RMS of the
intensity profile? FWHM? the 1/e² radius? :class:`WidthConvention` and
:class:`TimeConvention` name those choices, and :func:`convert_width` / :func:`convert_time`
move between them. Per §2.1 this vocabulary applies to **width-type parameters only** —
energies, charges, emittances, angles, a0 and the like are plain units with no convention
slot at all, and there is deliberately no ``NoConvention`` sentinel to bolt onto them.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pint

__all__ = [
    "ureg",
    "Quantity",
    "LIGHT_TIME_CONTEXT",
    "GAUSSIAN_CHARGE_CONTEXT",
    "BOUNDARY_CONTEXTS",
    "C_CGS",
    "E_ESU",
    "STATC_PER_COULOMB",
    "ME_CGS",
    "MEC2_CGS",
    "HBAR_CGS",
    "ALPHA",
    "R_E_CGS",
    "SIGMA_T_CGS",
    "EV_CGS",
    "WidthConvention",
    "TimeConvention",
    "convert_width",
    "convert_time",
    "to_canonical",
    "from_canonical",
    "scale_factor",
    "as_canonical_quantity",
    "UnknownConversionError",
]

ureg = pint.UnitRegistry()
Quantity = ureg.Quantity


# ---------------------------------------------------------------------------
# Boundary contexts: the two conversions pint cannot do unaided
# ---------------------------------------------------------------------------
LIGHT_TIME_CONTEXT = "light_time"
GAUSSIAN_CHARGE_CONTEXT = "gaussian_charge"

# Defined through the DSL (`Context.from_lines`) rather than the plain-Python
# `Context.add_transformation` API. The latter registers its dimensionality keys as
# `ParserHelper` instances, which don't compare equal to the `UnitsContainer` instances
# `Quantity.to()` looks them up with -- a silent DimensionalityError despite the context
# being "registered" (observed in pint 0.25). The DSL form has no
# such mismatch.
ureg.add_context(
    pint.Context.from_lines(
        [
            f"@context {LIGHT_TIME_CONTEXT}",
            "    [length] -> [time]: value / speed_of_light",
            "    [time] -> [length]: value * speed_of_light",
        ]
    )
)


# ---------------------------------------------------------------------------
# Physical constants, CGS-Gaussian, as plain floats
# ---------------------------------------------------------------------------
def _cgs(name: str, unit: str) -> float:
    """A CODATA constant from pint's own table as a plain float in a named CGS unit.

    Self-verifying (no hand-typed digit can drift from the real value) and usable
    directly in numpy/cupy/numba arithmetic, which a pint ``Quantity`` is not.
    """
    return float(Quantity(1, name).to(unit).magnitude)


C_CGS = _cgs("speed_of_light", "cm / s")
#: statC per coulomb — the exact textbook EM-unit conversion, ``1 C = c[cm/s] / 10 statC``.
STATC_PER_COULOMB = C_CGS / 10.0
ME_CGS = _cgs("electron_mass", "g")
HBAR_CGS = _cgs("hbar", "erg * s")
ALPHA = _cgs("fine_structure_constant", "dimensionless")
R_E_CGS = _cgs("classical_electron_radius", "cm")
SIGMA_T_CGS = _cgs("thomson_cross_section", "cm ** 2")
EV_CGS = _cgs("electron_volt", "erg")

MEC2_CGS = ME_CGS * C_CGS**2

# The one hand-applied conversion (see module docstring): 1 C = c[cm/s] / 10 statC.
E_ESU = _cgs("elementary_charge", "C") * STATC_PER_COULOMB

# Registered after the constants because its factor is derived from `C_CGS`, not typed in.
# This context is what lets the schema accept a bunch charge in pC and store it in statC:
# pint refuses that outright, since charge genuinely has different dimensionality in the
# two systems. Confining the exception to a named context keeps it from firing anywhere
# a caller did not ask for it -- ordinary conversions are unaffected.
ureg.add_context(
    pint.Context.from_lines(
        [
            f"@context {GAUSSIAN_CHARGE_CONTEXT}",
            "    [current] * [time] -> [mass] ** 0.5 * [length] ** 1.5 / [time]: "
            f"value * {STATC_PER_COULOMB!r} * statC / coulomb",
            "    [mass] ** 0.5 * [length] ** 1.5 / [time] -> [current] * [time]: "
            f"value / ({STATC_PER_COULOMB!r} * statC / coulomb)",
        ]
    )
)

#: Applied at every boundary. `gaussian_charge` is unconditional because charge has only
#: one meaning — a value either is a charge or is not, and the SI/Gaussian split is a
#: notational accident rather than a physical ambiguity.
#:
#: `light_time` is deliberately **not** here. It equates a length with a duration, which is
#: right for the one pairing §2.1 introduces it for — a longitudinal extent that may be
#: quoted either way — and wrong everywhere else: applied globally it would let a
#: *transverse* beam size be given in femtoseconds, which is not a unit choice but a
#: different physical quantity. Callers opt in per field via `light_time=True`.
BOUNDARY_CONTEXTS = (GAUSSIAN_CHARGE_CONTEXT,)


# ---------------------------------------------------------------------------
# Width / duration convention vocabulary
# ---------------------------------------------------------------------------
class WidthConvention(Enum):
    """Definitions of a transverse Gaussian width (laser spot or electron-beam size).

    ``SIGMA_INTENSITY_RMS`` is canonical: every shared dataclass stores that one, and
    :func:`convert_width` routes through it.
    """

    SIGMA_INTENSITY_RMS = "sigma_intensity_rms"  # I(r) ~ exp(-r^2 / (2 sigma^2))
    SIGMA_FIELD_RMS = "sigma_field_rms"  # E(r) ~ exp(-r^2 / (2 s^2)); s = sigma*sqrt(2)
    FWHM_INTENSITY = "fwhm_intensity"  # full width at half peak intensity
    W0_1E2 = "w0_1e2"  # 1/e^2 intensity radius (the "waist" convention)


class TimeConvention(Enum):
    """Definitions of a longitudinal/temporal Gaussian extent.

    Same algebra as :class:`WidthConvention` minus ``W0_1E2`` (no standard "waist"
    analogue in time). Kept as a separate enum rather than reusing ``WidthConvention`` so
    a transverse width can never be silently accepted where a duration was expected.
    """

    SIGMA_INTENSITY_RMS = "sigma_intensity_rms"
    SIGMA_FIELD_RMS = "sigma_field_rms"
    FWHM_INTENSITY = "fwhm_intensity"


_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_SQRT2 = np.sqrt(2.0)

# value -> canonical (sigma of the intensity profile), and back.
_TO_SIGMA: dict[Enum, float] = {
    WidthConvention.SIGMA_INTENSITY_RMS: 1.0,
    WidthConvention.SIGMA_FIELD_RMS: 1.0 / _SQRT2,
    WidthConvention.FWHM_INTENSITY: 1.0 / _FWHM_OVER_SIGMA,
    WidthConvention.W0_1E2: 0.5,
    TimeConvention.SIGMA_INTENSITY_RMS: 1.0,
    TimeConvention.SIGMA_FIELD_RMS: 1.0 / _SQRT2,
    TimeConvention.FWHM_INTENSITY: 1.0 / _FWHM_OVER_SIGMA,
}


class UnknownConversionError(ValueError):
    """Raised when a convention has no registered conversion in its family."""


def _convert(value: float, from_conv: Enum, to_conv: Enum, family: type[Enum]) -> float:
    if not isinstance(from_conv, family) or not isinstance(to_conv, family):
        raise UnknownConversionError(
            f"{family.__name__} conversion needs two {family.__name__} members, "
            f"got {from_conv!r} -> {to_conv!r}"
        )
    return value * _TO_SIGMA[from_conv] / _TO_SIGMA[to_conv]


def convert_width(value: float, from_conv: WidthConvention, to_conv: WidthConvention) -> float:
    """Reinterpret a transverse width from one convention to another."""
    return _convert(value, from_conv, to_conv, WidthConvention)


def convert_time(value: float, from_conv: TimeConvention, to_conv: TimeConvention) -> float:
    """Reinterpret a longitudinal/temporal extent from one convention to another."""
    return _convert(value, from_conv, to_conv, TimeConvention)


# ---------------------------------------------------------------------------
# Boundary unit conversion
# ---------------------------------------------------------------------------
def _contexts(light_time: bool) -> tuple[str, ...]:
    return (*BOUNDARY_CONTEXTS, LIGHT_TIME_CONTEXT) if light_time else BOUNDARY_CONTEXTS


def to_canonical(magnitude: float, unit: str, canonical_unit: str, *, light_time: bool = False) -> float:
    """Convert a user-facing magnitude into the core's canonical CGS unit.

    ``light_time`` opts this conversion into the length ↔ duration equivalence, for the
    longitudinal extents §2.1 introduces it for (a bunch length quoted in picoseconds, a
    pulse duration quoted in microns). Leave it off for everything else, so a transverse
    size cannot be given as a time.
    """
    return float(Quantity(magnitude, unit).to(canonical_unit, *_contexts(light_time)).magnitude)


def from_canonical(magnitude: float, canonical_unit: str, unit: str, *, light_time: bool = False) -> float:
    """Inverse of :func:`to_canonical`: canonical CGS magnitude → a display unit."""
    return float(Quantity(magnitude, canonical_unit).to(unit, *_contexts(light_time)).magnitude)


def scale_factor(from_unit: str, to_unit: str) -> float:
    """The pure scale factor taking a magnitude in ``from_unit`` to ``to_unit``.

    Every conversion in this project is a pure scaling — there are no offset units
    (nothing like °C) anywhere — so a conversion can always be reduced to one number and
    *folded into arithmetic a caller is doing anyway*, instead of making an extra pass over
    an array. That is what lets bulk per-particle data stay as raw ndarrays while still
    converting through a dimensionally checked factor (§3.2).

    A mismatch raises here, at the point the factor is computed, rather than silently
    producing a wrong number downstream.
    """
    return from_canonical(1.0, from_unit, to_unit)


def as_canonical_quantity(value, canonical_unit: str, name: str, *, light_time: bool = False) -> Quantity:
    """Validate that ``value`` is a `Quantity` of the right dimension; return it in CGS.

    The two halves matter for different reasons. **Requiring** a `Quantity` is what makes
    a unit mistake at an engine boundary an exception rather than a silent factor-of-100
    error — the failure mode P1 exists to prevent, at the one boundary P1's
    single-unit-system rule does not reach (kascade is SI internally, §4.4). **Converting**
    to canonical CGS keeps P1 literally true: what is stored really is a canonical
    CGS-Gaussian value, and the unit tag is there so no consumer can misread it.

    A bare float is rejected rather than assumed to be canonical: "assume it means what we
    happen to store" is exactly the convention this replaces.
    """
    if not isinstance(value, Quantity):
        raise TypeError(
            f"{name} must be a pint Quantity carrying its unit, e.g. "
            f"Quantity(20, 'um') or 20 * ureg.um — got {type(value).__name__} {value!r}. "
            f"A bare number is refused on purpose: it would have to be assumed to already "
            f"be in {canonical_unit!r}, which is the silent-mismatch this typing removes."
        )
    try:
        return value.to(canonical_unit, *_contexts(light_time))
    except pint.DimensionalityError as exc:
        raise TypeError(f"{name}: expected something convertible to {canonical_unit!r} — {exc}") from exc

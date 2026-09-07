"""Typed parameter schema (GRAND_PLAN.md §3.1).

The replacement for the predecessor's ``(label, default, key)`` triples, ``Job.extra``
stringly dicts, and bare ``StringVar`` parsing (P5). A :class:`FieldSpec` declares one
parameter — what it means, what canonical CGS unit the core stores it in, which display
units a GUI may offer, and what makes a value valid. A :class:`Parameters` object binds a
set of specs to validated values and is what engines receive; the same object with
defaults is what an engine publishes as its ``schema``.

Two invariants make this worth having:

* **Values inside a `Parameters` are always canonical CGS floats.** Unit strings and
  width conventions are *boundary* concepts: :meth:`Parameters.set_display` converts in,
  :meth:`Parameters.display` converts out, and nothing in between ever sees a unit.
* **Validation is centralized.** Range, choice membership and convention/kind agreement
  are checked once, at construction, rather than at each engine's boundary.

`FieldSpec` deliberately carries **no** editable/grey-out flag: that state is derived
solely from an engine's declared ``recompute_costs`` (§3.1, §5).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .units import (
    TimeConvention,
    WidthConvention,
    convert_time,
    convert_width,
    from_canonical,
    to_canonical,
)

__all__ = [
    "FieldKind",
    "FieldSpec",
    "Parameters",
    "SchemaError",
    "DIMENSIONLESS",
]

#: Canonical unit string for a dimensionless field. Chosen over ``""`` so a missing unit
#: is a bug rather than an accidentally-valid dimensionless declaration.
DIMENSIONLESS = "1"


class SchemaError(ValueError):
    """Raised when a schema declaration or a value assigned to it is invalid."""


class FieldKind(Enum):
    """What sort of parameter a :class:`FieldSpec` describes.

    The distinction the core actually acts on is *convention semantics*: ``WIDTH`` and
    ``DURATION`` carry a convention (RMS vs FWHM vs 1/e²), ``SCALAR`` has no such
    ambiguity to resolve, and ``CHOICE`` is a string from a closed set rather than a
    number. Integer-ness and ranges are orthogonal to this and live in their own
    :class:`FieldSpec` slots.
    """

    SCALAR = "scalar"
    WIDTH = "width"
    DURATION = "duration"
    CHOICE = "choice"


_CONVENTION_FAMILY: dict[FieldKind, type[Enum] | None] = {
    FieldKind.SCALAR: None,
    FieldKind.WIDTH: WidthConvention,
    FieldKind.DURATION: TimeConvention,
    FieldKind.CHOICE: None,
}


@dataclass(frozen=True)
class FieldSpec:
    """Declaration of a single parameter.

    ``unit`` is the **canonical core unit** the value is stored in — CGS-Gaussian
    (``cm``, ``s``, ``erg``, ``statC``, ``rad``, ...) or :data:`DIMENSIONLESS`.
    ``display_units`` are the pint-convertible alternatives a GUI may offer; the
    canonical unit is always usable whether or not it appears there.

    ``convention`` is the convention the **stored** value is in, and is required for
    ``WIDTH``/``DURATION`` and forbidden otherwise. In practice it is always the
    canonical ``SIGMA_INTENSITY_RMS``; the slot exists so a stored value is never
    convention-ambiguous, not so different fields can disagree.

    ``choices`` applies to ``CHOICE`` fields only; ``integer`` marks a numeric field
    whose value must be integral (particle counts, bin counts, step counts).
    """

    key: str
    label: str
    kind: FieldKind
    unit: str
    default: float | str
    display_units: tuple[str, ...] = ()
    convention: WidthConvention | TimeConvention | None = None
    value_range: tuple[float, float] | None = None
    choices: tuple[str, ...] = ()
    integer: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            raise SchemaError("FieldSpec.key must be non-empty")

        family = _CONVENTION_FAMILY[self.kind]
        if family is None and self.convention is not None:
            raise SchemaError(
                f"{self.key}: {self.kind.name} fields carry no convention "
                f"(got {self.convention!r}) — see GRAND_PLAN.md §2.1, there is no "
                f"NoConvention sentinel"
            )
        if family is not None and not isinstance(self.convention, family):
            raise SchemaError(
                f"{self.key}: {self.kind.name} fields require a {family.__name__}, "
                f"got {self.convention!r}"
            )

        if self.kind is FieldKind.CHOICE:
            if not self.choices:
                raise SchemaError(f"{self.key}: CHOICE fields need a non-empty `choices`")
            if self.unit != DIMENSIONLESS:
                raise SchemaError(f"{self.key}: CHOICE fields must declare unit {DIMENSIONLESS!r}")
            if self.integer or self.value_range is not None:
                raise SchemaError(f"{self.key}: CHOICE fields take no numeric constraints")
        elif self.choices:
            raise SchemaError(f"{self.key}: `choices` is meaningful only for CHOICE fields")

        if self.value_range is not None:
            low, high = self.value_range
            if not low <= high:
                raise SchemaError(f"{self.key}: value_range {self.value_range} is inverted")

        # Fail at declaration time, not on the first GUI interaction, if a unit string is
        # not something pint can parse or convert into the canonical one. pint reads "1"
        # as dimensionless, so DIMENSIONLESS needs no special case here.
        for unit in (self.unit, *self.display_units):
            try:
                to_canonical(1.0, unit, self.unit, light_time=self.kind is FieldKind.DURATION)
            except Exception as exc:  # pint raises several unrelated exception types
                raise SchemaError(f"{self.key}: cannot convert {unit!r} to {self.unit!r}: {exc}") from exc

        self.validate(self.default)

    # -- validation ---------------------------------------------------------
    def validate(self, value: float | str) -> float | str:
        """Return ``value`` if it is admissible for this spec, else raise `SchemaError`."""
        if self.kind is FieldKind.CHOICE:
            if value not in self.choices:
                raise SchemaError(f"{self.key}: {value!r} is not one of {self.choices}")
            return value

        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise SchemaError(f"{self.key}: expected a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise SchemaError(f"{self.key}: value must be finite, got {value!r}")
        if self.integer and value != int(value):
            raise SchemaError(f"{self.key}: value must be integral, got {value!r}")
        if self.value_range is not None:
            low, high = self.value_range
            if not low <= value <= high:
                raise SchemaError(f"{self.key}: {value!r} outside allowed range [{low}, {high}]")
        return value

    # -- boundary conversion ------------------------------------------------
    def to_core(
        self,
        magnitude: float,
        unit: str | None = None,
        convention: WidthConvention | TimeConvention | None = None,
    ) -> float:
        """Convert a user-facing ``(magnitude, unit, convention)`` into the core value.

        Unit conversion and convention reinterpretation are independent steps: the unit
        moves the number between length/time scales, the convention reinterprets *what
        width* the number was measuring. Omitting either means "already canonical".
        """
        if self.kind is FieldKind.CHOICE:
            raise SchemaError(f"{self.key}: CHOICE fields have no unit conversion")
        value = (
            magnitude
            if unit is None or unit == self.unit
            else to_canonical(magnitude, unit, self.unit, light_time=self.kind is FieldKind.DURATION)
        )
        return self.validate(self._reinterpret(value, convention, self.convention))

    def to_display(
        self,
        value: float,
        unit: str | None = None,
        convention: WidthConvention | TimeConvention | None = None,
    ) -> float:
        """Inverse of :meth:`to_core`: a core value in the requested unit/convention."""
        if self.kind is FieldKind.CHOICE:
            raise SchemaError(f"{self.key}: CHOICE fields have no unit conversion")
        value = self._reinterpret(value, self.convention, convention)
        if unit is None or unit == self.unit:
            return float(value)
        return from_canonical(value, self.unit, unit, light_time=self.kind is FieldKind.DURATION)

    def _reinterpret(
        self,
        value: float,
        from_conv: WidthConvention | TimeConvention | None,
        to_conv: WidthConvention | TimeConvention | None,
    ) -> float:
        if from_conv is None or to_conv is None or from_conv is to_conv:
            return float(value)
        if self.kind is FieldKind.WIDTH:
            return convert_width(value, from_conv, to_conv)
        return convert_time(value, from_conv, to_conv)


@dataclass(frozen=True)
class Parameters:
    """A validated set of parameter values bound to their specs.

    Engines receive this; an engine's published ``schema`` is the same object holding
    each spec's default. Values are canonical CGS floats (or ``CHOICE`` strings) — never
    a stringly dict, never a mutable config object (P11).
    """

    specs: tuple[FieldSpec, ...]
    values: Mapping[str, float | str]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        seen: dict[str, FieldSpec] = {}
        for spec in specs:
            if spec.key in seen:
                raise SchemaError(f"duplicate field key {spec.key!r}")
            seen[spec.key] = spec

        unknown = set(self.values) - set(seen)
        if unknown:
            raise SchemaError(f"unknown parameter keys: {sorted(unknown)}")

        validated = {key: spec.validate(self.values.get(key, spec.default)) for key, spec in seen.items()}
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "values", MappingProxyType(validated))

    # -- construction -------------------------------------------------------
    @classmethod
    def from_specs(cls, specs: tuple[FieldSpec, ...] | list[FieldSpec], **values: float | str) -> "Parameters":
        """Build from specs, taking each field's default unless overridden."""
        return cls(specs=tuple(specs), values=values)

    def with_values(self, **overrides: float | str) -> "Parameters":
        """A new `Parameters` with ``overrides`` applied and re-validated."""
        return replace(self, values={**self.values, **overrides})

    # -- access -------------------------------------------------------------
    def spec(self, key: str) -> FieldSpec:
        for spec in self.specs:
            if spec.key == key:
                return spec
        raise SchemaError(f"no such parameter: {key!r}")

    def __getitem__(self, key: str) -> float | str:
        try:
            return self.values[key]
        except KeyError:
            raise SchemaError(f"no such parameter: {key!r}") from None

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def get_float(self, key: str) -> float:
        """Value as a float — the accessor kernels use, so a stray string surfaces here."""
        value = self[key]
        if isinstance(value, str):
            raise SchemaError(f"{key!r} is a CHOICE field, not numeric")
        return float(value)

    def get_int(self, key: str) -> int:
        return int(self.get_float(key))

    def get_choice(self, key: str) -> str:
        value = self[key]
        if not isinstance(value, str):
            raise SchemaError(f"{key!r} is not a CHOICE field")
        return value

    # -- boundary helpers ---------------------------------------------------
    def set_display(
        self,
        key: str,
        magnitude: float,
        unit: str | None = None,
        convention: WidthConvention | TimeConvention | None = None,
    ) -> "Parameters":
        """Set one field from a user-facing magnitude/unit/convention."""
        return self.with_values(**{key: self.spec(key).to_core(magnitude, unit, convention)})

    def display(
        self,
        key: str,
        unit: str | None = None,
        convention: WidthConvention | TimeConvention | None = None,
    ) -> float:
        """Read one field in a user-facing unit/convention."""
        return self.spec(key).to_display(self.get_float(key), unit, convention)

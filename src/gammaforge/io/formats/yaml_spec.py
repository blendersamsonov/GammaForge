"""Beam / laser / sampling YAML files.

**Units are explicit at the file boundary and converted to CGS on load.** A parameter is
written as a mapping::

    sigma_x: {value: 20.0, unit: um, convention: sigma_intensity_rms}
    rel_energy_spread: 0.01

A bare number means "already in this field's canonical core unit", which is what makes
dimensionless fields read naturally; anything with a physical unit should carry it.

The whole module is driven by the `FieldSpec` sets in `gammaforge.io.fields` — it never
names an individual parameter. Adding, renaming or re-uniting a parameter there is
therefore automatically reflected here, which is the point of declaring the field sets
once (§3.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..fields import BEAM_FIELDS, LASER_FIELDS, SAMPLING_FIELDS
from ..schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters, SchemaError
from ..units import TimeConvention, WidthConvention

__all__ = [
    "SPEC_VERSION",
    "GROUPS",
    "parameters_to_yaml_dict",
    "parameters_from_yaml_dict",
    "save_spec",
    "load_spec",
]

#: Bumped only when the file layout changes incompatibly. A file without it is rejected
#: rather than guessed at.
SPEC_VERSION = 1

#: The named parameter groups a spec file may contain.
GROUPS: dict[str, tuple[FieldSpec, ...]] = {
    "beam": BEAM_FIELDS,
    "laser": LASER_FIELDS,
    "sampling": SAMPLING_FIELDS,
}

_CONVENTIONS: dict[FieldKind, type] = {
    FieldKind.WIDTH: WidthConvention,
    FieldKind.DURATION: TimeConvention,
}


def _preferred_unit(spec: FieldSpec) -> str:
    """The unit a value is written in: the first declared display unit, else canonical.

    Writing in a display unit keeps files readable — ``100 pC`` rather than
    ``0.299792458 statC`` — while the unit tag keeps them unambiguous.
    """
    return spec.display_units[0] if spec.display_units else spec.unit


#: Significant digits a value is written with. Converting 100 pC into canonical statC and
#: back lands on 99.99999999999999, which is correct and looks broken — and these files are
#: meant to be read and hand-edited. Twelve digits is far more precision than any of these
#: parameters is known to, while keeping the round-trip well inside the 1e-11 the tests
#: require.
_WRITTEN_DIGITS = 12


def _round(value: float) -> float:
    return float(f"%.{_WRITTEN_DIGITS}g" % value)


def parameters_to_yaml_dict(params: Parameters) -> dict[str, Any]:
    """Serialize one parameter group to plain YAML-safe types."""
    out: dict[str, Any] = {}
    for spec in params.specs:
        if spec.kind is FieldKind.CHOICE:
            out[spec.key] = params.get_choice(spec.key)
            continue
        unit = _preferred_unit(spec)
        entry: dict[str, Any] = {"value": _round(spec.to_display(params.get_float(spec.key), unit))}
        if unit != DIMENSIONLESS:
            entry["unit"] = unit
        if spec.convention is not None:
            entry["convention"] = spec.convention.value
        out[spec.key] = entry["value"] if len(entry) == 1 else entry
    return out


def parameters_from_yaml_dict(specs: tuple[FieldSpec, ...], data: Mapping[str, Any]) -> Parameters:
    """Read one parameter group, converting every entry into the canonical core unit."""
    unknown = set(data) - {spec.key for spec in specs}
    if unknown:
        raise SchemaError(f"unknown parameters in spec file: {sorted(unknown)}")

    values: dict[str, float | str] = {}
    for spec in specs:
        if spec.key not in data:
            continue  # the spec's own default applies
        entry = data[spec.key]
        if spec.kind is FieldKind.CHOICE:
            values[spec.key] = entry
        elif isinstance(entry, Mapping):
            values[spec.key] = spec.to_core(
                float(entry["value"]),
                entry.get("unit"),
                _parse_convention(spec, entry.get("convention")),
            )
        else:
            # A bare number is canonical by definition -- see the module docstring.
            values[spec.key] = float(entry)
    return Parameters(specs=specs, values=values)


def _parse_convention(spec: FieldSpec, name: str | None):
    if name is None:
        return None
    family = _CONVENTIONS.get(spec.kind)
    if family is None:
        raise SchemaError(f"{spec.key}: a {spec.kind.name} field carries no convention")
    try:
        return family(name)
    except ValueError:
        raise SchemaError(
            f"{spec.key}: {name!r} is not a {family.__name__} "
            f"(expected one of {[member.value for member in family]})"
        ) from None


def save_spec(path: str | Path, **groups: Parameters) -> None:
    """Write one or more named parameter groups to a YAML spec file.

    ``save_spec(path, beam=..., laser=...)``. Group names must be keys of :data:`GROUPS`,
    so a typo fails here rather than producing a file that silently loads as empty.
    """
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SchemaError(f"unknown parameter group(s) {sorted(unknown)}; expected {sorted(GROUPS)}")
    document = {"version": SPEC_VERSION}
    document.update({name: parameters_to_yaml_dict(params) for name, params in groups.items()})
    Path(path).write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False))


def load_spec(path: str | Path) -> dict[str, Parameters]:
    """Read a YAML spec file into validated `Parameters`, one per group present."""
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, Mapping):
        raise SchemaError(f"{path}: spec file must be a mapping")
    version = document.get("version")
    if version != SPEC_VERSION:
        raise SchemaError(
            f"{path}: spec version {version!r} is not supported (this build reads "
            f"version {SPEC_VERSION})"
        )
    unknown = set(document) - set(GROUPS) - {"version"}
    if unknown:
        raise SchemaError(f"{path}: unknown group(s) {sorted(unknown)}")
    return {
        name: parameters_from_yaml_dict(specs, document[name])
        for name, specs in GROUPS.items()
        if name in document
    }

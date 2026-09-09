"""Input draft state for the browser UI.

The browser keeps values in :class:`Parameters`, exactly as YAML and engines do.  Widget
preferences are deliberately separate: changing a displayed unit must never change the
physical calculation draft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..io.fields import (
    BEAM_FIELDS,
    LASER_FIELDS,
    SAMPLING_FIELDS,
    TARGET_FIELDS,
    beam_from_parameters,
    laser_from_parameters,
    sampling_from_parameters,
)
from ..io.schema import FieldKind, Parameters, SchemaError
from ..io.target import OutputKind, OutputRequest, SLICE_AXES, Target
from ..io.units import Quantity

if TYPE_CHECKING:
    from ..engines.base import Engine


@dataclass(frozen=True)
class DisplayPreference:
    """A field's presentation choice; it is not part of a calculation request."""

    unit: str | None = None
    convention: Any | None = None


@dataclass
class InputState:
    """Mutable browser draft with immutable validated parameter groups beneath it."""

    engines: dict[str, "Engine"]
    groups: dict[str, Parameters] = field(init=False)
    selected: set[str] = field(init=False)
    requested: dict[OutputKind, tuple[int, ...]] = field(
        default_factory=lambda: {OutputKind.TOTAL_YIELD: (), OutputKind.SPECTRUM: (64,)}
    )
    errors: dict[str, str] = field(default_factory=dict)
    display: dict[tuple[str, str], DisplayPreference] = field(default_factory=dict)
    raw: dict[tuple[str, str], str] = field(default_factory=dict)
    output_raw: dict[OutputKind, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.groups = {
            "beam": Parameters.from_specs(BEAM_FIELDS),
            "laser": Parameters.from_specs(LASER_FIELDS),
            "sampling": Parameters.from_specs(SAMPLING_FIELDS),
            "target": Parameters.from_specs(TARGET_FIELDS),
            **{f"engine:{name}": engine.schema for name, engine in self.engines.items()},
        }
        # The first concrete engine is the normal/default path. Validation engines remain
        # visible but opt-in, without teaching the GUI any engine-specific names.
        first_engine = next(iter(self.engines), None)
        self.selected = {first_engine} if first_engine is not None else set()

    @staticmethod
    def error_key(group: str, key: str) -> str:
        return f"{group}.{key}"

    def preference(self, group: str, key: str) -> DisplayPreference:
        spec = self.groups[group].spec(key)
        return self.display.get(
            (group, key), DisplayPreference(spec.display_units[0] if spec.display_units else spec.unit, spec.convention)
        )

    def set_display(self, group: str, key: str, *, unit: str | None = None, convention: Any | None = None) -> None:
        """Change only the displayed representation, retaining the canonical value."""
        old = self.preference(group, key)
        self.display[(group, key)] = DisplayPreference(old.unit if unit is None else unit, old.convention if convention is None else convention)

    def set_value(self, group: str, key: str, value: str | float, *, unit: str | None = None, convention: Any | None = None) -> bool:
        """Validate and apply a physical edit.  Invalid text remains an explicit error."""
        error_key = self.error_key(group, key)
        self.raw[(group, key)] = str(value)
        params = self.groups[group]
        spec = params.spec(key)
        try:
            if spec.kind is FieldKind.CHOICE:
                if not isinstance(value, str):
                    raise SchemaError(f"{key}: expected a choice")
                updated = params.with_values(**{key: value})
            else:
                magnitude = float(value)
                pref = self.preference(group, key)
                updated = params.set_display(key, magnitude, unit if unit is not None else pref.unit,
                                             convention if convention is not None else pref.convention)
        except (TypeError, ValueError, SchemaError) as exc:
            self.errors[error_key] = str(exc)
            return False
        self.groups[group] = updated
        self.errors.pop(error_key, None)
        valid = self._validate_physical_group(group)
        if valid:
            self.errors.pop(self.error_key("request", "physical"), None)
        return valid

    def _validate_physical_group(self, group: str) -> bool:
        """Catch dataclass constraints that span several individually-valid fields."""
        constructors = {
            "beam": beam_from_parameters,
            "laser": laser_from_parameters,
            "sampling": sampling_from_parameters,
        }
        group_error = self.error_key(group, "physical")
        try:
            if group == "target":
                Target(
                    Quantity(self.groups[group].get_float("theta_x_col"), "rad"),
                    Quantity(self.groups[group].get_float("theta_y_col"), "rad"),
                )
            elif group in constructors:
                constructors[group](self.groups[group])
        except (TypeError, ValueError, SchemaError) as exc:
            self.errors[group_error] = str(exc)
            return False
        self.errors.pop(group_error, None)
        return True

    def supports(self, kind: OutputKind) -> bool:
        return any(kind in self.engines[name].supported_outputs for name in self.selected)

    def set_requested(self, kind: OutputKind, enabled: bool, resolution: tuple[int | str, ...] | None = None) -> bool:
        """Update an output request, refusing an output no selected engine can make."""
        error_key = self.error_key("outputs", kind.name)
        if kind is OutputKind.TOTAL_YIELD:
            self.requested[kind] = ()
            self.errors.pop(error_key, None)
            return True
        if not enabled:
            self.requested.pop(kind, None)
            self.errors.pop(error_key, None)
            return True
        try:
            if not self.supports(kind):
                raise ValueError(f"No selected engine supports {kind.name}")
            raw_resolution = resolution if resolution is not None else self.requested.get(kind, (64,) * len(SLICE_AXES[kind] or ()))
            self.output_raw[kind] = tuple(str(value) for value in raw_resolution)
            request = OutputRequest(
                kind,
                tuple(int(value) for value in raw_resolution),
            )
        except (TypeError, ValueError) as exc:
            self.errors[error_key] = str(exc)
            return False
        self.requested[kind] = request.resolution
        self.errors.pop(error_key, None)
        return True

    def request(self):
        """Create the widget-free execution snapshot, or fail with visible draft errors."""
        if self.errors:
            raise ValueError("Input errors must be corrected before Calculate")
        unsupported = [kind.name for kind in self.requested if not self.supports(kind)]
        if self.selected and unsupported:
            raise ValueError(f"No selected engine supports requested outputs: {', '.join(unsupported)}")
        outputs = tuple(OutputRequest(kind, resolution) for kind, resolution in self.requested.items())
        target_params = self.groups["target"]
        try:
            beam = beam_from_parameters(self.groups["beam"])
            laser = laser_from_parameters(self.groups["laser"])
            sampling = sampling_from_parameters(self.groups["sampling"])
            target = Target(
                Quantity(target_params.get_float("theta_x_col"), "rad"),
                Quantity(target_params.get_float("theta_y_col"), "rad"),
                outputs,
            )
        except (TypeError, ValueError, SchemaError) as exc:
            self.errors[self.error_key("request", "physical")] = str(exc)
            raise ValueError("Input errors must be corrected before Calculate") from exc
        # This module intentionally imports the runner contract only at the boundary so
        # state-only uses remain importable while the execution feature is installed.
        from ..io.calculation import CalculationRequest

        return CalculationRequest(
            beam=beam,
            laser=laser,
            target=target,
            sampling=sampling,
            engine_params={
                name: self.groups[f"engine:{name}"]
                for name in self.engines
                if name in self.selected
            },
        )

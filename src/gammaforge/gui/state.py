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
from ..io.plotting import display_scale
from ..io.results import Axis
from ..io.target import OutputKind, OutputRequest, SLICE_AXES, Target, auto_ranges
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
    selected_engine: str | None = field(init=False, default=None)
    requested: dict[OutputKind, tuple[int, ...]] = field(
        default_factory=lambda: {OutputKind.TOTAL_YIELD: (), OutputKind.SPECTRUM: (64,)}
    )
    errors: dict[str, str] = field(default_factory=dict)
    display: dict[tuple[str, str], DisplayPreference] = field(default_factory=dict)
    raw: dict[tuple[str, str], str] = field(default_factory=dict)
    output_raw: dict[OutputKind, tuple[str, ...]] = field(default_factory=dict)
    manual_ranges: dict[OutputKind, dict[Axis, tuple[float, float]]] = field(default_factory=dict)
    manual_axes: set[tuple[OutputKind, Axis]] = field(default_factory=set)
    range_raw: dict[tuple[OutputKind, Axis], tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.groups = {
            "beam": Parameters.from_specs(BEAM_FIELDS),
            "laser": Parameters.from_specs(LASER_FIELDS),
            "sampling": Parameters.from_specs(SAMPLING_FIELDS),
            "target": Parameters.from_specs(TARGET_FIELDS),
            **{f"engine:{name}": engine.schema for name, engine in self.engines.items()},
        }
        # Default to first engine
        first_engine = next(iter(self.engines), None)
        self.selected_engine = first_engine

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
        if self.selected_engine is None:
            return False
        return kind in self.engines[self.selected_engine].supported_outputs

    def set_selected_engine(self, engine_name: str | None) -> bool:
        """Set the selected engine for calculation."""
        if engine_name is not None and engine_name not in self.engines:
            return False
        self.selected_engine = engine_name
        return True

    def set_requested(self, kind: OutputKind, enabled: bool, resolution: tuple[int | str, ...] | None = None) -> bool:
        """Update an output request, refusing an output no selected engine can make."""
        error_key = self.error_key("outputs", kind.name)
        if kind is OutputKind.TOTAL_YIELD:
            self.requested[kind] = ()
            self.errors.pop(error_key, None)
            return True
        if not enabled:
            self.requested.pop(kind, None)
            for key in tuple(self.errors):
                if key == error_key or key.startswith(f"outputs.{kind.name}."):
                    self.errors.pop(key, None)
            return True
        try:
            if not self.supports(kind):
                raise ValueError(f"Selected engine does not support {kind.name}")
            raw_resolution = resolution if resolution is not None else self.requested.get(kind, (64,) * len(SLICE_AXES[kind] or ()))
            self.output_raw[kind] = tuple(str(value) for value in raw_resolution)
            request = OutputRequest(
                kind,
                tuple(int(value) for value in raw_resolution),
            )
        except (TypeError, ValueError) as exc:
            self.errors[error_key] = str(exc)
            return False
        ranges_valid = True
        for output_kind, axis in tuple(self.manual_axes):
            if output_kind is not kind:
                continue
            raw = self.range_raw.get((kind, axis))
            if raw is not None:
                ranges_valid = self.set_output_range(kind, axis, *raw) and ranges_valid
            elif axis not in self.manual_ranges.get(kind, {}):
                self.errors[self.error_key("outputs", f"{kind.name}.{axis.key}.range")] = (
                    f"{kind.name}: enter finite, increasing {axis.name} limits"
                )
                ranges_valid = False
        if not ranges_valid:
            return False
        self.requested[kind] = request.resolution
        self.errors.pop(error_key, None)
        return True

    def set_output_range(
        self,
        kind: OutputKind,
        axis: Axis,
        low: str | float,
        high: str | float,
    ) -> bool:
        """Set one manual axis range from the GUI's display units."""
        error_key = self.error_key("outputs", f"{kind.name}.{axis.key}.range")
        self.manual_axes.add((kind, axis))
        self.range_raw[(kind, axis)] = (str(low), str(high))
        axes = SLICE_AXES[kind]
        try:
            if axes is None or axis not in axes:
                raise ValueError(f"{axis.name} is not an axis of {kind.name}")
            scale = display_scale(axis)
            bounds = (float(low) / scale, float(high) / scale)
            OutputRequest(kind, self.requested.get(kind, (64,) * len(axes)), {axis: bounds})
        except (TypeError, ValueError) as exc:
            self.errors[error_key] = str(exc)
            return False
        self.manual_ranges.setdefault(kind, {})[axis] = bounds
        self.errors.pop(error_key, None)
        return True

    def set_output_auto(self, kind: OutputKind, axis: Axis, enabled: bool) -> None:
        """Select derived or manual bounds for one output axis."""
        if not enabled:
            self.manual_axes.add((kind, axis))
            return
        self.manual_axes.discard((kind, axis))
        ranges = self.manual_ranges.get(kind)
        if ranges is not None:
            ranges.pop(axis, None)
            if not ranges:
                self.manual_ranges.pop(kind, None)
        self.errors.pop(self.error_key("outputs", f"{kind.name}.{axis.key}.range"), None)

    def auto_range(self, kind: OutputKind) -> dict[Axis, tuple[float, float]] | None:
        """Return current derived bounds, or ``None`` when sampling is required."""
        axes = SLICE_AXES[kind]
        if axes is None:
            return {}
        output = OutputRequest(kind, self.requested.get(kind, (64,) * len(axes)))
        target_params = self.groups["target"]
        target = Target(
            Quantity(target_params.get_float("theta_x_col"), "rad"),
            Quantity(target_params.get_float("theta_y_col"), "rad"),
            (output,),
        )
        try:
            return auto_ranges(
                target,
                beam_from_parameters(self.groups["beam"]),
                laser_from_parameters(self.groups["laser"]),
            ).get(kind, {})
        except ValueError as exc:
            if kind is OutputKind.TEMPORAL_ENVELOPE and "needs the bunch" in str(exc):
                return None
            raise

    def request(self):
        """Create the widget-free execution snapshot, or fail with visible draft errors."""
        if self.errors:
            raise ValueError("Input errors must be corrected before Calculate")
        unsupported = [kind.name for kind in self.requested if not self.supports(kind)]
        if self.selected_engine and unsupported:
            raise ValueError(f"Selected engine does not support requested outputs: {', '.join(unsupported)}")
        if not self.selected_engine:
            raise ValueError("Select an engine for calculation.")
        outputs = tuple(
            OutputRequest(kind, resolution, self.manual_ranges.get(kind) or None)
            for kind, resolution in self.requested.items()
        )
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
                self.selected_engine: self.groups[f"engine:{self.selected_engine}"]
            },
        )

    def estimate_request(self):
        """Build the analytical preview from only the fields its formulas consume."""
        if any(key.startswith(("beam.", "laser.", "target.")) for key in self.errors):
            raise ValueError("Correct the beam, laser, or collimation inputs to update estimates")
        from ..io.calculation import CalculationRequest
        from ..io.interaction import SamplingSpec

        target_params = self.groups["target"]
        target = Target(
            Quantity(target_params.get_float("theta_x_col"), "rad"),
            Quantity(target_params.get_float("theta_y_col"), "rad"),
            (OutputRequest(OutputKind.TOTAL_YIELD),),
        )
        return CalculationRequest(
            beam=beam_from_parameters(self.groups["beam"]),
            laser=laser_from_parameters(self.groups["laser"]),
            target=target,
            sampling=SamplingSpec(),
            engine_params={},
        )

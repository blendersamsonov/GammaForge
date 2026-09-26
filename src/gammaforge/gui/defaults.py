"""Persistent, per-panel defaults for the local browser GUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml

from ..io.formats.yaml_spec import parameters_from_yaml_dict, parameters_to_yaml_dict
from ..io.results import Axis
from ..io.schema import Parameters, SchemaError
from ..io.target import OutputKind, OutputRequest

DEFAULTS_VERSION = 1
GEOMETRY_KEYS = ("theta_xz", "theta_yz", "psi_focus", "psi_pol")

_SECTION_GROUPS = {
    "electrons": "beam",
    "sampling": "sampling",
    "laser": "laser",
    "geometry": "laser",
    "target": "target",
}


def default_path() -> Path:
    """Return the user-local GUI defaults file without adding a runtime dependency."""
    configured = os.environ.get("GAMMAFORGE_CONFIG_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "gammaforge" / "gui-defaults.yaml"


class GuiDefaultsStore:
    """Read and atomically update schema-validated GUI panel defaults."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = default_path() if path is None else Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": DEFAULTS_VERSION}
        try:
            document = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not read GUI defaults from {self.path}: {exc}") from exc
        if not isinstance(document, dict) or document.get("version") != DEFAULTS_VERSION:
            raise ValueError(f"Unsupported GUI defaults file at {self.path}")
        return document

    def _write(self, document: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Could not save GUI defaults to {self.path}: {exc}") from exc

    def save_parameters(
        self,
        section: str,
        params: Parameters,
        *,
        keys: Iterable[str] | None = None,
    ) -> None:
        if section not in _SECTION_GROUPS:
            raise ValueError(f"Unknown defaults section {section!r}")
        selected = set(params) if keys is None else set(keys)
        unknown = selected - set(params)
        if unknown:
            raise ValueError(f"Unknown parameter keys for {section}: {sorted(unknown)}")
        values = parameters_to_yaml_dict(params)
        document = self._read()
        document[section] = {key: values[key] for key in params if key in selected}
        self._write(document)

    def save_engine(self, name: str, params: Parameters) -> None:
        document = self._read()
        engines = document.setdefault("engines", {})
        if not isinstance(engines, dict):
            raise ValueError(f"Invalid engines section in {self.path}")
        engines[name] = parameters_to_yaml_dict(params)
        self._write(document)

    def save_outputs(self, state) -> None:
        outputs = []
        for kind, resolution in state.requested.items():
            manual = state.manual_ranges.get(kind, {})
            outputs.append({
                "kind": kind.value,
                "resolution": list(resolution),
                "manual_ranges": {
                    axis.key: list(bounds) for axis, bounds in manual.items()
                } or None,
            })
        document = self._read()
        document["outputs"] = outputs
        self._write(document)

    def apply(self, state) -> None:
        """Apply every saved section only after the complete file validates."""
        document = self._read()
        group_updates: dict[str, Parameters] = {}
        try:
            for section, group in _SECTION_GROUPS.items():
                data = document.get(section)
                if data is None:
                    continue
                if not isinstance(data, dict):
                    raise ValueError(f"{section} defaults must be a mapping")
                current = group_updates.get(group, state.groups[group])
                allowed = set(current)
                if section == "geometry":
                    allowed = set(GEOMETRY_KEYS)
                elif section == "laser":
                    allowed -= set(GEOMETRY_KEYS)
                misplaced = set(data) - allowed
                if misplaced:
                    raise ValueError(
                        f"{section} defaults contain fields from another panel: {sorted(misplaced)}"
                    )
                loaded = parameters_from_yaml_dict(current.specs, data)
                group_updates[group] = current.with_values(
                    **{key: loaded.values[key] for key in data}
                )

            engine_updates: dict[str, Parameters] = {}
            engines = document.get("engines", {})
            if not isinstance(engines, dict):
                raise ValueError("engine defaults must be a mapping")
            for name, data in engines.items():
                group = f"engine:{name}"
                if group not in state.groups:
                    continue
                engine_updates[group] = parameters_from_yaml_dict(state.groups[group].specs, data)

            requested = None
            manual_ranges = None
            if "outputs" in document:
                if not isinstance(document["outputs"], list):
                    raise ValueError("output defaults must be a list")
                requested = {}
                manual_ranges = {}
                for entry in document["outputs"]:
                    kind = OutputKind(entry["kind"])
                    if kind in requested:
                        raise ValueError(f"duplicate output default: {kind.value}")
                    manual = {
                        Axis.from_key(key): tuple(bounds)
                        for key, bounds in (entry.get("manual_ranges") or {}).items()
                    }
                    output = OutputRequest(kind, tuple(entry["resolution"]), manual or None)
                    requested[kind] = output.resolution
                    if output.manual_ranges:
                        manual_ranges[kind] = dict(output.manual_ranges)
                if OutputKind.TOTAL_YIELD not in requested:
                    raise ValueError("output defaults must include total yield")
        except (KeyError, TypeError, ValueError, SchemaError) as exc:
            raise ValueError(f"Invalid GUI defaults in {self.path}: {exc}") from exc

        state.groups.update(group_updates)
        state.groups.update(engine_updates)
        if requested is not None:
            state.requested = requested
            state.manual_ranges = manual_ranges or {}
            state.manual_axes = {
                (kind, axis)
                for kind, ranges in state.manual_ranges.items()
                for axis in ranges
            }

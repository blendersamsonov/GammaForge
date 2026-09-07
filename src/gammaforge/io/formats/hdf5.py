"""Results serialization: HDF5 slices plus a YAML parameter sidecar (GRAND_PLAN.md §8).

Layout::

    /slices/<output_kind>/distr           the density array
    /slices/<output_kind>/axes/<axis_key> that axis's values, with a `unit` attribute
    /slices/<output_kind>/widths/<axis_key> optional histogram cell widths
      attrs: axis_order = [axis keys, in the slice's own order]
    /photons/<field>                      MC photon macroparticles, when present

Axis order is stored explicitly rather than inferred from HDF5's own ordering of group
members, which is alphabetical and would silently transpose ``(energy, theta_x, theta_y)``
slices on the way back in.

Units are **not** converted here: everything on disk is canonical CGS, exactly as in
memory. The ``unit`` attributes are there so a file is self-describing to a reader that is
not this package, not because anything is stored in anything else.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import json
import os
import tempfile

import h5py
import numpy as np
import yaml

from ..calculation import CalculationRequest
from ..bunch import Bunch
from ..fields import BEAM_FIELDS, beam_from_parameters, beam_to_parameters
from ..results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from ..schema import Parameters
from ..target import OutputKind
from .calculation import request_from_dict, request_to_dict
from .yaml_spec import save_spec

__all__ = ["save_results", "load_results", "load_request", "sidecar_path", "RESULTS_VERSION"]

RESULTS_VERSION = 1

_PHOTON_FIELDS = ("energy", "theta_x", "theta_y", "x", "y", "z", "t", "weight", "order")
_ELECTRON_FIELDS = ("x", "y", "z", "thx", "thy", "gamma", "weight")


def sidecar_path(path: str | Path) -> Path:
    """The YAML parameter file that accompanies an HDF5 results file."""
    return Path(path).with_suffix(".yaml")


def save_results(
    results: Results, path: str | Path, *, request: CalculationRequest | None = None,
    **groups: Parameters,
) -> None:
    """Write ``results`` to HDF5, and any given parameter groups to the YAML sidecar.

    ``save_results(results, "run.h5", beam=..., laser=...)`` produces ``run.h5`` and
    ``run.yaml``. The parameters live in YAML rather than in HDF5 attributes so they stay
    readable and diffable, and so one serialization path (`yaml_spec`) covers both a spec
    file a user wrote and the record of what a run actually used.

    Warnings and numerical metadata are preserved. Dataclass metadata loads as field
    mappings, without importing engine classes. An optional ``request`` is embedded
    in this same file and restored by ``load_request``. It must be the submitted
    snapshot that actually produced these results. Existing spec sidecars remain
    supported, but are not a complete run record.
    """
    path = Path(path)
    metadata = json.dumps(_encode_metadata(results.model_specific), allow_nan=False)
    electron_metadata = None
    if results.electrons is not None:
        if not isinstance(results.electrons, Bunch):
            raise TypeError("electron persistence requires Bunch; refusing to discard electrons")
        fit = results.electrons.gaussian_fit
        electron_metadata = json.dumps(_encode_metadata({
            "meta": results.electrons.meta,
            "fit": None if fit is None else dict(beam_to_parameters(fit).values),
            "fit_quality": None if fit is None else fit.fit_quality,
        }), allow_nan=False)
    request_yaml = None if request is None else yaml.safe_dump(request_to_dict(request), sort_keys=False)
    # Validate before touching an existing result and publish only a complete HDF5 file.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        _write_results(results, temporary_path, metadata, request_yaml, electron_metadata)
        if groups:
            save_spec(sidecar_path(path), **groups)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_results(results: Results, path: str | Path, metadata: str, request_yaml: str | None, electron_metadata: str | None) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["gammaforge_results_version"] = RESULTS_VERSION
        handle.create_dataset("model_specific_json", data=metadata)
        if request_yaml is not None:
            handle.create_dataset("request_yaml", data=request_yaml)
        if results.electrons is not None:
            electrons = handle.create_group("electrons")
            electrons.create_dataset("metadata_json", data=electron_metadata)
            for name in _ELECTRON_FIELDS:
                dataset = electrons.create_dataset(name, data=getattr(results.electrons, name))
                dataset.attrs["unit"] = results.electrons.UNITS[name]
        slices = handle.create_group("slices")
        for kind, slice_ in results.photon_slices.items():
            name = kind.value if hasattr(kind, "value") else str(kind)
            group = slices.create_group(name)
            group.create_dataset("distr", data=slice_.distr)
            axes = group.create_group("axes")
            for axis, values in slice_.axes.items():
                dataset = axes.create_dataset(axis.key, data=values)
                dataset.attrs["unit"] = axis.unit
            group.attrs["axis_order"] = [axis.key for axis in slice_.axis_order]
            if slice_.widths:
                widths = group.create_group("widths")
                for axis, values in slice_.widths.items():
                    widths.create_dataset(axis.key, data=values)

        if results.photons is not None:
            photons = handle.create_group("photons")
            for name in _PHOTON_FIELDS:
                values = getattr(results.photons, name)
                if values is not None:
                    photons.create_dataset(name, data=np.asarray(values))

def load_results(path: str | Path, kind_from_name=None) -> Results:
    """Read back what :func:`save_results` wrote.

    Known output names load as ``OutputKind`` members by default; unknown names remain
    strings. Pass ``str`` for the legacy string-key behavior, or a custom converter.
    Legacy files without metadata return an empty metadata mapping; their provenance
    cannot be inferred. Unsupported future versions raise.
    """
    path = Path(path)
    photon_slices = {}
    photons = None
    electrons = None
    with h5py.File(path, "r") as handle:
        _check_version(handle)
        metadata = _decode_metadata(json.loads(handle["model_specific_json"].asstr()[()])) if "model_specific_json" in handle else {}
        for name, group in handle["slices"].items():
            order = [key.decode() if isinstance(key, bytes) else str(key)
                     for key in group.attrs["axis_order"]]
            axes = {Axis.from_key(key): np.asarray(group["axes"][key]) for key in order}
            key = kind_from_name(name) if kind_from_name is not None else _output_key(name)
            widths = (
                {Axis.from_key(name): np.asarray(values) for name, values in group["widths"].items()}
                if "widths" in group else None
            )
            photon_slices[key] = PhasespaceSlice(
                axes=axes, distr=np.asarray(group["distr"]), widths=widths
            )

        if "photons" in handle:
            stored = {name: np.asarray(handle["photons"][name])
                      for name in _PHOTON_FIELDS if name in handle["photons"]}
            photons = PhotonMacroparticles(**stored)

        if "electrons" in handle:
            group = handle["electrons"]
            electron_metadata = _decode_metadata(json.loads(group["metadata_json"].asstr()[()]))
            fit = electron_metadata["fit"]
            if fit is not None:
                fit = replace(beam_from_parameters(Parameters(BEAM_FIELDS, fit)), fit_quality=electron_metadata["fit_quality"])
            electrons = Bunch(
                **{name: np.asarray(group[name]) for name in _ELECTRON_FIELDS},
                meta=electron_metadata["meta"], gaussian_fit=fit,
            )

    return Results(photon_slices=photon_slices, photons=photons, electrons=electrons, model_specific=metadata)


def load_request(path: str | Path, engine_schemas: Mapping[str, Parameters]) -> CalculationRequest:
    """Load the embedded submitted request; missing provenance is an explicit error."""
    with h5py.File(path, "r") as handle:
        _check_version(handle)
        if "request_yaml" not in handle:
            raise ValueError("results file has no recorded calculation request (unknown provenance)")
        return request_from_dict(yaml.safe_load(handle["request_yaml"].asstr()[()]), engine_schemas)


def _check_version(handle) -> None:
    version = handle.attrs.get("gammaforge_results_version", 0)
    if version not in (0, RESULTS_VERSION):
        raise ValueError(f"results version {version!r} is not supported")


def _output_key(name: str):
    try:
        return OutputKind(name)
    except ValueError:
        return name


def _encode_metadata(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        return ["mapping", {key: _encode_metadata(item) for key, item in value.items()}]
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufU":
            raise TypeError(f"unsupported metadata array dtype {value.dtype}")
        return ["array", value.dtype.str, list(value.shape), value.tolist()]
    if isinstance(value, (list, tuple)):
        return ["tuple" if isinstance(value, tuple) else "list", [_encode_metadata(item) for item in value]]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return ["scalar", value]
    raise TypeError(f"unsupported metadata type {type(value).__name__}")


def _decode_metadata(value):
    kind = value[0]
    if kind == "mapping":
        return {key: _decode_metadata(item) for key, item in value[1].items()}
    if kind == "array":
        dtype = np.dtype(value[1])
        if dtype.kind not in "biufU":
            raise ValueError(f"unsupported metadata array dtype {dtype}")
        return np.asarray(value[3], dtype=dtype).reshape(value[2])
    if kind in ("tuple", "list"):
        items = [_decode_metadata(item) for item in value[1]]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        return value[1]
    raise ValueError(f"unsupported metadata encoding {kind!r}")

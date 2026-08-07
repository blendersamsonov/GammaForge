"""Results serialization: HDF5 slices plus a YAML parameter sidecar (GRAND_PLAN.md §8).

Layout::

    /slices/<output_kind>/distr           the density array
    /slices/<output_kind>/axes/<axis_key> that axis's values, with a `unit` attribute
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

import h5py
import numpy as np

from ..results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from ..schema import Parameters
from .yaml_spec import save_spec

__all__ = ["save_results", "load_results", "sidecar_path"]

_PHOTON_FIELDS = ("energy", "theta_x", "theta_y", "x", "y", "z", "t", "weight", "order")


def sidecar_path(path: str | Path) -> Path:
    """The YAML parameter file that accompanies an HDF5 results file."""
    return Path(path).with_suffix(".yaml")


def save_results(results: Results, path: str | Path, **groups: Parameters) -> None:
    """Write ``results`` to HDF5, and any given parameter groups to the YAML sidecar.

    ``save_results(results, "run.h5", beam=..., laser=...)`` produces ``run.h5`` and
    ``run.yaml``. The parameters live in YAML rather than in HDF5 attributes so they stay
    readable and diffable, and so one serialization path (`yaml_spec`) covers both a spec
    file a user wrote and the record of what a run actually used.
    """
    path = Path(path)
    with h5py.File(path, "w") as handle:
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

        if results.photons is not None:
            photons = handle.create_group("photons")
            for name in _PHOTON_FIELDS:
                values = getattr(results.photons, name)
                if values is not None:
                    photons.create_dataset(name, data=np.asarray(values))

    if groups:
        save_spec(sidecar_path(path), **groups)


def load_results(path: str | Path, kind_from_name=None) -> Results:
    """Read back what :func:`save_results` wrote.

    ``kind_from_name`` maps a stored slice name to whatever key the caller wants back —
    pass ``OutputKind`` to recover the enum members. It is a parameter rather than a hard
    import so this module stays independent of the output vocabulary it is merely storing.
    """
    path = Path(path)
    photon_slices = {}
    photons = None
    with h5py.File(path, "r") as handle:
        for name, group in handle["slices"].items():
            order = [key.decode() if isinstance(key, bytes) else str(key)
                     for key in group.attrs["axis_order"]]
            axes = {Axis.from_key(key): np.asarray(group["axes"][key]) for key in order}
            key = kind_from_name(name) if kind_from_name is not None else name
            photon_slices[key] = PhasespaceSlice(axes=axes, distr=np.asarray(group["distr"]))

        if "photons" in handle:
            stored = {name: np.asarray(handle["photons"][name])
                      for name in _PHOTON_FIELDS if name in handle["photons"]}
            photons = PhotonMacroparticles(**stored)

    return Results(photon_slices=photon_slices, photons=photons)

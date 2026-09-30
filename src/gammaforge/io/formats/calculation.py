"""Safe YAML representation of a Gaussian calculation request (RES063)."""

from __future__ import annotations

from collections.abc import Mapping

from ..calculation import CalculationRequest
from ..fields import (
    BEAM_FIELDS, LASER_FIELDS, SAMPLING_FIELDS,
    beam_from_parameters, beam_to_parameters,
    laser_from_parameters, laser_to_parameters,
    sampling_from_parameters, sampling_to_parameters,
)
from ..laser import GaussianParaxialLaser
from ..results import Axis
from ..schema import Parameters
from ..target import OutputKind, OutputRequest, Target
from ..units import Quantity

__all__ = ["request_to_dict", "request_from_dict"]


def request_to_dict(request: CalculationRequest) -> dict:
    """Record all inputs in canonical CGS without rounding numerical settings.

    Arbitrary field implementations need their own agreed representation; they cannot
    be reconstructed from an approximate Gaussian fit.
    """
    if type(request.laser) is not GaussianParaxialLaser:
        raise TypeError("calculation export supports GaussianParaxialLaser only")
    return {
        "version": 1,
        "units": "CGS-Gaussian",
        "beam": dict(beam_to_parameters(request.beam).values),
        "laser": dict(laser_to_parameters(request.laser).values),
        "sampling": dict(sampling_to_parameters(request.sampling).values),
        "target": {
            "theta_x_col": request.target.m("theta_x_col"),
            "theta_y_col": request.target.m("theta_y_col"),
            "outputs": [
                {
                    "kind": output.kind.value,
                    "resolution": [int(n) for n in output.resolution],
                    "manual_ranges": None if output.manual_ranges is None else {
                        axis.key: [float(v) for v in bounds]
                        for axis, bounds in output.manual_ranges.items()
                    },
                }
                for output in request.target.outputs
            ],
        },
        "engine_params": {name: dict(params.values) for name, params in request.engine_params.items()},
    }


def request_from_dict(document: dict, engine_schemas: Mapping[str, Parameters] | None = None) -> CalculationRequest:
    """Restore using the engine catalog's schemas unless the caller supplies its own.

    No import is selected by file content: the recorded engine names are looked up in a
    schema mapping, never resolved by reaching into an implementation module (RES095).
    """
    if engine_schemas is None:
        # Deferred: `gammaforge.io` is the shared layer and does not import `engines`.
        from ...engines.catalog import engine_schemas as catalog_schemas

        engine_schemas = catalog_schemas()
    if document.get("version") != 1 or document.get("units") != "CGS-Gaussian":
        raise ValueError("unsupported calculation version or unit system")
    engine_values = document["engine_params"]
    unknown = set(engine_values) - set(engine_schemas)
    if unknown:
        raise ValueError(f"missing engine schemas: {sorted(unknown)}")
    target = document["target"]
    outputs = tuple(
        OutputRequest(
            OutputKind(output["kind"]), tuple(output["resolution"]),
            None if output["manual_ranges"] is None else {
                Axis.from_key(key): tuple(bounds) for key, bounds in output["manual_ranges"].items()
            },
        )
        for output in target["outputs"]
    )
    return CalculationRequest(
        beam=beam_from_parameters(Parameters(BEAM_FIELDS, document["beam"])),
        laser=laser_from_parameters(Parameters(LASER_FIELDS, document["laser"])),
        sampling=sampling_from_parameters(Parameters(SAMPLING_FIELDS, document["sampling"])),
        target=Target(Quantity(target["theta_x_col"], "rad"), Quantity(target["theta_y_col"], "rad"), outputs),
        engine_params={name: Parameters(engine_schemas[name].specs, values) for name, values in engine_values.items()},
    )

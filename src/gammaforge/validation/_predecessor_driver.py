"""Runs inside the **predecessor** repo's interpreter. Not part of this package's API.

`make_references` executes this file as a script under an interpreter whose ``sys.path``
points at the old repo, because both repos install a package literally named
``gammaforge`` — the two cannot coexist in one process, and no import trick makes that
safe. The subprocess boundary is the design, not a workaround.

Contract, deliberately narrow so this file needs no shared code with either side:

* **in** — one JSON file, the path given as ``argv[1]``: an output directory, a list of
  scenarios described in SI (the old repo's unit system), and which models to run;
* **out** — one ``.npz`` per (scenario, model) holding every slice the model produced,
  plus a ``manifest.json`` describing what ran and what the old repo's own HEAD was.

Nothing here converts units, renames a quantity, or interprets a result: it reports what
the old code produced, in the old code's own terms. The translation into this repo's
vocabulary happens on the other side of the boundary, in `make_references`, where it is
versioned with the format it targets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def build_beam(spec):
    from gammaforge.io.bunch import GaussianElectronBeam
    from gammaforge.io.units import PhysicalMeaning, PhysicalQuantity, TimeConvention, WidthConvention

    def quantity(value, unit, meaning, convention=None):
        if convention is None:
            return PhysicalQuantity(value, unit, meaning)
        return PhysicalQuantity(value, unit, meaning, convention)

    return GaussianElectronBeam(
        bunch_charge_C=quantity(spec["bunch_charge_C"], "coulomb", PhysicalMeaning.BUNCH_CHARGE),
        kinetic_energy_eV=quantity(spec["kinetic_energy_eV"], "electron_volt", PhysicalMeaning.BEAM_ENERGY),
        rel_energy_spread_rms=spec["rel_energy_spread"],
        sigma_pz=spec["rel_energy_spread"],
        sigma_x_m=quantity(spec["sigma_x_m"], "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE,
                           WidthConvention.SIGMA_INTENSITY_RMS),
        sigma_y_m=quantity(spec["sigma_y_m"], "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE,
                           WidthConvention.SIGMA_INTENSITY_RMS),
        emit_geom_x_m=quantity(spec["emit_geom_x_m"], "meter", PhysicalMeaning.EMITTANCE),
        emit_geom_y_m=quantity(spec["emit_geom_y_m"], "meter", PhysicalMeaning.EMITTANCE),
        sigma_t_s=quantity(spec["sigma_t_s"], "second", PhysicalMeaning.BUNCH_LENGTH,
                           TimeConvention.SIGMA_INTENSITY_RMS),
    )


def build_laser(spec):
    from gammaforge.io.laser import GaussianParaxialLaser
    from gammaforge.io.units import NoConvention, PhysicalMeaning, PhysicalQuantity, TimeConvention, WidthConvention

    return GaussianParaxialLaser(
        pulse_energy_J=PhysicalQuantity(spec["pulse_energy_J"], "joule",
                                        PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
        wavelength_m=PhysicalQuantity(spec["wavelength_m"], "meter",
                                      PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
        waist_rms_x_m=PhysicalQuantity(spec["sigma_x_m"], "meter",
                                       PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        waist_rms_y_m=PhysicalQuantity(spec["sigma_y_m"], "meter",
                                       PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
        duration_rms_s=PhysicalQuantity(spec["duration_s"], "second",
                                        PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
    )


def build_job(scenario, model):
    """The old repo's `Job`, with its bunch sampled by the old repo's own sampler.

    The two samplers are not bit-compatible (`make_references` explains why), so the old
    model sees a statistically equivalent bunch, not the same one. That is the dominant
    term in every tolerance a golden is compared under.
    """
    from gammaforge.io.bunch import sample_gaussian_bunch
    from gammaforge.io.interaction import InteractionParameters
    from gammaforge.models.api import AXIS_ENERGY, Job, OutputSpec, SliceRequest

    beam = build_beam(scenario["beam"])
    laser = build_laser(scenario["laser"])
    bunch = sample_gaussian_bunch(beam, n_particles=scenario["n_particles"],
                                  rng=np.random.default_rng(scenario["seed"]))
    output = OutputSpec(slices=(SliceRequest((AXIS_ENERGY,), (scenario["n_energy_bins"],)),))
    # CPU on purpose: a golden should be the float64 answer, not whatever a particular
    # GPU produced in float32, and it should not stop being reproducible because a CUDA
    # toolkit moved. The old adapters ignore this key when they have no device knob.
    extra = {"theta_col_rad": scenario["theta_col_rad"], "a0_max": scenario["a0_max"],
             "device_preference": "cpu"}
    return Job(interaction=InteractionParameters(laser=laser, electrons=bunch),
               output=output, seed=scenario["seed"], extra=extra)


#: Which adapter each model name means. Adding a model here is the whole change needed to
#: start snapshotting it — the payload and the writer below are model-agnostic.
def adapter_for(model):
    if model == "analytical":
        from gammaforge.models.analytical import Adapter
        return Adapter()
    if model == "xigma":
        from gammaforge.models.xigma_i.adapter import XigmaAdapter
        return XigmaAdapter()
    if model == "delta":
        from gammaforge.models.xigma_i.adapter import DirectAdapter
        return DirectAdapter()
    raise SystemExit(f"_predecessor_driver: no adapter known for model {model!r}")


def write_result(out_dir, scenario_name, model, results):
    """One ``.npz`` per (scenario, model): every slice flattened into named arrays.

    ``slice{i}_axes`` is a JSON list of the old repo's axis names, in the slice's own
    order, since a ``.npz`` holds arrays and nothing else.
    """
    arrays = {}
    index = []
    for i, slice_ in enumerate(results.photon_slices):
        axis_names = list(slice_.axes)
        arrays[f"slice{i}_distr"] = np.asarray(slice_.distr)
        for name in axis_names:
            arrays[f"slice{i}_axis_{name}"] = np.asarray(slice_.axes[name])
        index.append({"index": i, "axes": axis_names})

    path = Path(out_dir) / f"{scenario_name}__{model}.npz"
    np.savez_compressed(
        path,
        index=np.array(json.dumps(index)),
        scalars=np.array(json.dumps({k: float(v) for k, v in (results.model_specific or {}).items()
                                     if np.isscalar(v)})),
        **arrays,
    )
    return path.name


def head_state(repo):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    commit, dirty = head_state(payload["repo"])
    entries = []
    for scenario in payload["scenarios"]:
        for model in payload["models"]:
            job = build_job(scenario, model)
            results = adapter_for(model).run(job)
            entries.append({
                "scenario": scenario["name"],
                "model": model,
                "file": write_result(out_dir, scenario["name"], model, results),
            })

    (out_dir / "manifest.json").write_text(json.dumps({
        "commit": commit,
        "dirty": dirty,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "entries": entries,
    }, indent=2))


if __name__ == "__main__":
    main()

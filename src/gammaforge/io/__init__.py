"""Shared CGS-Gaussian physics core: schema, beam, laser, target, interaction, results.

Depends on nothing else in the repo (GRAND_PLAN.md §3). Package name `io` is retained
from the predecessor for continuity (P7: no `core` package).

Modules, in dependency order:

* `units` — CGS-Gaussian constants, the pint boundary contexts, width conventions (§2.1)
* `schema` — `FieldSpec` / `Parameters`: typed, validated, unit-converting (§3.1)
* `bunch` — `Bunch`, `GaussianElectronBeam`, sampling, prefilter, propagation, fit (§3.2)
* `laser` — the `LaserField` protocol and `GaussianParaxialLaser` (§3.3, P15)
* `results` — `Axis`, `PhasespaceSlice`, `Results` (§3.6)
* `target` — `OutputKind`, `Target`, auto-ranging (§3.4)
* `interaction` — `SamplingSpec`, `InteractionParameters` (§3.5)
* `fields` — the shared beam/laser/sampling `FieldSpec` sets (§3.1)
* `formats` — YAML specs, elegant `.ele`, HDF5 results (§8)

The names re-exported below are the library surface a notebook or an engine should use;
anything else is reachable through its module but is not part of the contract.
"""

from .bunch import Bunch, GaussianElectronBeam, sample_gaussian_bunch
from .interaction import InteractionParameters, SamplingSpec, build_interaction
from .laser import GaussianParaxialLaser, LaserField, PulseTrainParaxialLaser, fit_gaussian_paraxial
from .results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from .schema import FieldKind, FieldSpec, Parameters
from .target import OutputKind, OutputRequest, Target
from .units import TimeConvention, WidthConvention

__all__ = [
    "Axis",
    "Bunch",
    "FieldKind",
    "FieldSpec",
    "GaussianElectronBeam",
    "GaussianParaxialLaser",
    "InteractionParameters",
    "LaserField",
    "OutputKind",
    "PulseTrainParaxialLaser",
    "OutputRequest",
    "Parameters",
    "PhasespaceSlice",
    "PhotonMacroparticles",
    "Results",
    "SamplingSpec",
    "Target",
    "TimeConvention",
    "WidthConvention",
    "build_interaction",
    "fit_gaussian_paraxial",
    "sample_gaussian_bunch",
]

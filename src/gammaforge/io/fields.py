"""Shared field-set declarations (GRAND_PLAN.md §3.1, §8).

The electron, laser and sampling parameter sets are declared **once, here**, and shared by
the GUI, the engines and YAML I/O — the "single source of truth for parameter semantics"
§3.1 asks for. Declaring them in one place is also what lets serialization be generic:
`gammaforge.io.formats.yaml_spec` writes and reads any of these sets without knowing what
they mean, because each `FieldSpec` already says.

Every ``key`` here matches the corresponding dataclass field name exactly, which is what
makes :func:`to_parameters` / :func:`from_parameters` mechanical rather than a hand-written
mapping table that would rot the first time a field is renamed.
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from typing import TypeVar

from .bunch import GaussianElectronBeam
from .interaction import SamplingSpec
from .laser import GaussianParaxialLaser, PulseTrainParaxialLaser
from .schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters
from .units import Quantity, TimeConvention, WidthConvention

__all__ = [
    "BEAM_FIELDS",
    "LASER_FIELDS",
    "SAMPLING_FIELDS",
    "TARGET_FIELDS",
    "to_parameters",
    "from_parameters",
]

_RMS = WidthConvention.SIGMA_INTENSITY_RMS
_RMS_T = TimeConvention.SIGMA_INTENSITY_RMS
_POSITIVE = (0.0, math.inf)
_CORRELATION = (-1.0, 1.0)

#: Display units, ordered so the **first** entry is what a person would actually write —
#: it is both what a GUI dropdown opens on and what a saved YAML file is written in. So a
#: charge leads with pC rather than the canonical statC, and a beam energy with MeV rather
#: than erg: canonical units are for storing values, not for reading them.
_SIZE_UNITS = ("um", "nm", "mm", "cm", "m")
_WAVELENGTH_UNITS = ("nm", "um", "cm", "m")
_ANGLE_UNITS = ("rad", "mrad", "urad", "degree")
#: A longitudinal extent may be given as either a length or a duration (§2.1's
#: ``light_time`` pairing), so both appear — but the two fields that use this lead with
#: different halves, because a bunch *length* reads naturally in microns and a pulse
#: *duration* in femtoseconds, and the leading entry is what gets written to file.
_BUNCH_LENGTH_UNITS = ("um", "mm", "cm", "m", "fs", "ps", "ns", "s")
_PULSE_DURATION_UNITS = ("fs", "ps", "ns", "s", "um", "mm", "cm", "m")

# Laser type specific field keys (used for round-trip conversion)
_GAUSSIAN_LASER_KEYS = ("duration",)
_PULSE_TRAIN_LASER_KEYS = ("subpulse_duration", "repetition_period", "n_subpulses")

#: Each field's canonical unit is read from the dataclass that stores it, so the unit is
#: stated in exactly one place. (Nothing breaks if they differ — the boundary converts
#: either way — but there is no reason for them to, and one statement is one fewer thing
#: to keep in step.)
_BEAM = GaussianElectronBeam.UNITS
_LASER = GaussianParaxialLaser.UNITS
_LASER_PULSE_TRAIN = PulseTrainParaxialLaser.UNITS


BEAM_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("bunch_charge", "Bunch charge", FieldKind.SCALAR, _BEAM["bunch_charge"], 100e-12 * 2.99792458e9,
              display_units=("pC", "nC", "C", "statC"), value_range=_POSITIVE),
    FieldSpec("kinetic_energy", "Kinetic energy", FieldKind.SCALAR, _BEAM["kinetic_energy"], 100e6 * 1.602176634e-12,
              display_units=("MeV", "GeV", "keV", "eV", "erg"), value_range=_POSITIVE),
    FieldSpec("rel_energy_spread", "Relative energy spread", FieldKind.SCALAR, DIMENSIONLESS, 0.01,
              display_units=(DIMENSIONLESS, "percent"), value_range=(0.0, 1.0)),
    FieldSpec("sigma_x", "Beam size (x)", FieldKind.WIDTH, _BEAM["sigma_x"], 20e-4,
              display_units=_SIZE_UNITS, convention=_RMS, value_range=_POSITIVE),
    FieldSpec("sigma_y", "Beam size (y)", FieldKind.WIDTH, _BEAM["sigma_y"], 20e-4,
              display_units=_SIZE_UNITS, convention=_RMS, value_range=_POSITIVE),
    FieldSpec("emit_x", "Geometric emittance (x)", FieldKind.SCALAR, _BEAM["emit_x"], 1e-7,
              display_units=("mm * mrad", "um * rad", "cm * rad", "m * rad"), value_range=_POSITIVE),
    FieldSpec("emit_y", "Geometric emittance (y)", FieldKind.SCALAR, _BEAM["emit_y"], 1e-7,
              display_units=("mm * mrad", "um * rad", "cm * rad", "m * rad"), value_range=_POSITIVE),
    FieldSpec("sigma_z", "Bunch length", FieldKind.DURATION, _BEAM["sigma_z"], 1e-2,
              display_units=_BUNCH_LENGTH_UNITS, convention=_RMS_T, value_range=_POSITIVE),
    FieldSpec("rho_x_gamma", "Dispersion correlation (x-E)", FieldKind.SCALAR, DIMENSIONLESS, 0.0,
              value_range=_CORRELATION),
    FieldSpec("rho_y_gamma", "Dispersion correlation (y-E)", FieldKind.SCALAR, DIMENSIONLESS, 0.0,
              value_range=_CORRELATION),
    FieldSpec("rho_z_gamma", "Chirp correlation (z-E)", FieldKind.SCALAR, DIMENSIONLESS, 0.0,
              value_range=_CORRELATION),
    FieldSpec("rho_thx_gamma", "Dispersion-derivative correlation (x'-E)", FieldKind.SCALAR,
              DIMENSIONLESS, 0.0, value_range=_CORRELATION),
    FieldSpec("rho_thy_gamma", "Dispersion-derivative correlation (y'-E)", FieldKind.SCALAR,
              DIMENSIONLESS, 0.0, value_range=_CORRELATION),
    FieldSpec("alpha_x", "Twiss alpha (x)", FieldKind.SCALAR, DIMENSIONLESS, 0.0),
    FieldSpec("alpha_y", "Twiss alpha (y)", FieldKind.SCALAR, DIMENSIONLESS, 0.0),
)


LASER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("pulse_energy", "Pulse energy", FieldKind.SCALAR, _LASER["pulse_energy"], 1e7,
              display_units=("J", "mJ", "erg"), value_range=_POSITIVE),
    FieldSpec("wavelength", "Central wavelength", FieldKind.SCALAR, _LASER["wavelength"], 0.8e-4,
              display_units=_WAVELENGTH_UNITS, value_range=_POSITIVE),
    FieldSpec("laser_type", "Laser type", FieldKind.CHOICE, DIMENSIONLESS, "gaussian",
              choices=("gaussian", "pulse_train")),
    FieldSpec("sigma_x", "Spot size (focusing axis 1)", FieldKind.WIDTH, _LASER["sigma_x"], 10e-4,
              display_units=_SIZE_UNITS, convention=_RMS, value_range=_POSITIVE),
    FieldSpec("sigma_y", "Spot size (focusing axis 2)", FieldKind.WIDTH, _LASER["sigma_y"], 10e-4,
              display_units=_SIZE_UNITS, convention=_RMS, value_range=_POSITIVE),
    FieldSpec("duration", "Pulse duration", FieldKind.DURATION, _LASER["duration"], 30e-15,
              display_units=_PULSE_DURATION_UNITS, convention=_RMS_T, value_range=_POSITIVE),
    FieldSpec("subpulse_duration", "Sub-pulse duration", FieldKind.DURATION, _LASER_PULSE_TRAIN["subpulse_duration"], 30e-15,
              display_units=_PULSE_DURATION_UNITS, convention=_RMS_T, value_range=_POSITIVE),
    FieldSpec("repetition_period", "Repetition period", FieldKind.DURATION, _LASER_PULSE_TRAIN["repetition_period"], 100e-15,
              display_units=_PULSE_DURATION_UNITS, convention=_RMS_T, value_range=_POSITIVE),
    FieldSpec("n_subpulses", "Number of sub-pulses", FieldKind.SCALAR, DIMENSIONLESS, 10,
              value_range=(1, 1000), integer=True),
    FieldSpec("z_fx", "Focal offset (axis 1)", FieldKind.SCALAR, _LASER["z_fx"], 0.0, display_units=_SIZE_UNITS),
    FieldSpec("z_fy", "Focal offset (axis 2)", FieldKind.SCALAR, _LASER["z_fy"], 0.0, display_units=_SIZE_UNITS),
    # Misalignment of the pulse against the bunch. No `z_off`: a rigid longitudinal shift
    # by `Delta` moves the focus *and* the envelope, so it already reads as
    # `(z_fx += Delta, z_fy += Delta, t_off += Delta/c)`. Focus position and arrival time
    # stay independent knobs — coincident foci still miss if the timing differs.
    FieldSpec("x_off", "Transverse misalignment (x)", FieldKind.SCALAR, _LASER["x_off"], 0.0,
              display_units=_SIZE_UNITS),
    FieldSpec("y_off", "Transverse misalignment (y)", FieldKind.SCALAR, _LASER["y_off"], 0.0,
              display_units=_SIZE_UNITS),
    FieldSpec("t_off", "Timing offset", FieldKind.SCALAR, _LASER["t_off"], 0.0,
              display_units=("fs", "ps", "ns", "s")),
    FieldSpec("theta_xz", "Crossing angle in xz", FieldKind.SCALAR, "rad", 0.0,
              display_units=_ANGLE_UNITS, value_range=(-math.pi, math.pi)),
    FieldSpec("theta_yz", "Crossing angle in yz", FieldKind.SCALAR, "rad", 0.0,
              display_units=_ANGLE_UNITS, value_range=(-math.pi, math.pi)),
    FieldSpec("psi_focus", "Focusing-axes rotation", FieldKind.SCALAR, "rad", 0.0,
              display_units=_ANGLE_UNITS, value_range=(-math.pi, math.pi)),
    FieldSpec("psi_pol", "Polarization-axes rotation", FieldKind.SCALAR, "rad", 0.0,
              display_units=_ANGLE_UNITS, value_range=(-math.pi, math.pi)),
    FieldSpec("ellipticity", "Polarization ellipticity", FieldKind.SCALAR, DIMENSIONLESS, 0.0,
              value_range=(0.0, 1.0)),
    FieldSpec("beta_ff", "Flying-focus factor", FieldKind.SCALAR, DIMENSIONLESS, 0.0,
              value_range=(-0.999, math.inf)),
)


SAMPLING_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("n_particles", "Macroparticles", FieldKind.SCALAR, DIMENSIONLESS, 100_000,
              value_range=(1, 1e9), integer=True),
    # First-class and user-editable (§3.5/§6): reproducibility is a guarantee, not
    # incidental internal state.
    FieldSpec("seed", "Random seed", FieldKind.SCALAR, DIMENSIONLESS, 0,
              value_range=(0, 2**31 - 1), integer=True),
    FieldSpec("prefilter", "Prefilter threshold (fraction of peak a0)", FieldKind.SCALAR,
              DIMENSIONLESS, 1e-3, value_range=(0.0, 0.999)),
    # IID stays the default until the adaptive strategy's benchmarks justify promoting it;
    # see `gammaforge.io.adaptive_sampling`. A CHOICE rather than a boolean so adding a
    # third strategy later cannot silently reinterpret a saved request.
    FieldSpec("strategy", "Sampling strategy", FieldKind.CHOICE, DIMENSIONLESS, "iid",
              choices=("iid", "adaptive")),
)


TARGET_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "theta_x_col", "Collimation half-angle (x)", FieldKind.SCALAR, "rad", 1e-3,
        display_units=("mrad", "urad", "rad", "degree"), value_range=(0.0, math.inf),
    ),
    FieldSpec(
        "theta_y_col", "Collimation half-angle (y)", FieldKind.SCALAR, "rad", 1e-3,
        display_units=("mrad", "urad", "rad", "degree"), value_range=(0.0, math.inf),
    ),
)


_T = TypeVar("_T")


def _magnitude(value, spec: FieldSpec):
    """A dataclass field as the plain canonical number `Parameters` stores.

    `Parameters` is the *boundary* layer — GUI entry, YAML, engine knobs — and it already
    knows each field's canonical unit from its own `FieldSpec`, so re-carrying the unit
    inside it would duplicate that declaration rather than protect anything. Dimensional
    typing earns its keep on the physics dataclasses, where a value is read directly by an
    engine that may use a different unit system.
    """
    return value.m_as(spec.unit) if isinstance(value, Quantity) else value


def to_parameters(obj, specs: tuple[FieldSpec, ...]) -> Parameters:
    """Read a dataclass's values into a validated `Parameters` over ``specs``."""
    return Parameters.from_specs(
        specs, **{spec.key: _magnitude(getattr(obj, spec.key), spec) for spec in specs}
    )


def from_parameters(cls: type[_T], params: Parameters) -> _T:
    """Build a dataclass of type ``cls`` from ``params``.

    Only keys ``cls`` actually declares are passed, so a parameter set may legitimately
    carry extras (an engine merging its own knobs into a shared group) without breaking
    construction. Integer-declared fields are handed over as `int`, since a dataclass
    field like ``n_particles`` should not silently become a float.
    """
    accepted = {f.name for f in dataclass_fields(cls)}
    dimensioned = getattr(cls, "UNITS", {})
    values = {}
    for spec in params.specs:
        if spec.key not in accepted:
            continue
        if spec.key in dimensioned:
            # Re-attach the unit the target dataclass declares, not the one the spec
            # stores in, so the two can differ without this silently mismatching.
            values[spec.key] = Quantity(params.get_float(spec.key), spec.unit)
        elif spec.integer:
            values[spec.key] = params.get_int(spec.key)
        else:
            values[spec.key] = params[spec.key]
    return cls(**values)


#: Convenience aliases with the concrete types spelled out, for call-site readability.
def beam_to_parameters(beam: GaussianElectronBeam) -> Parameters:
    return to_parameters(beam, BEAM_FIELDS)


def beam_from_parameters(params: Parameters) -> GaussianElectronBeam:
    return from_parameters(GaussianElectronBeam, params)


def laser_to_parameters(laser: GaussianParaxialLaser | PulseTrainParaxialLaser) -> Parameters:
    """Convert a laser (Gaussian or PulseTrain) to Parameters."""
    # Determine laser_type from the actual object type
    laser_type = "pulse_train" if isinstance(laser, PulseTrainParaxialLaser) else "gaussian"
    
    # Select relevant field specs based on laser type
    if laser_type == "pulse_train":
        relevant_specs = tuple(s for s in LASER_FIELDS if s.key not in _GAUSSIAN_LASER_KEYS)
    else:
        relevant_specs = tuple(s for s in LASER_FIELDS if s.key not in _PULSE_TRAIN_LASER_KEYS)
    
    # Create a dict with all field values, including laser_type
    values = {spec.key: _magnitude(getattr(laser, spec.key), spec) for spec in relevant_specs if spec.key != "laser_type"}
    values["laser_type"] = laser_type
    return Parameters.from_specs(LASER_FIELDS, **values)


def laser_from_parameters(params: Parameters) -> GaussianParaxialLaser | PulseTrainParaxialLaser:
    """Build a laser from Parameters, selecting type based on laser_type field."""
    laser_type = params.get_choice("laser_type")
    if laser_type == "pulse_train":
        return from_parameters(PulseTrainParaxialLaser, params)
    return from_parameters(GaussianParaxialLaser, params)


def sampling_to_parameters(sampling: SamplingSpec) -> Parameters:
    return to_parameters(sampling, SAMPLING_FIELDS)


def sampling_from_parameters(params: Parameters) -> SamplingSpec:
    return from_parameters(SamplingSpec, params)


__all__ += [
    "beam_to_parameters",
    "beam_from_parameters",
    "laser_to_parameters",
    "laser_from_parameters",
    "sampling_to_parameters",
    "sampling_from_parameters",
]

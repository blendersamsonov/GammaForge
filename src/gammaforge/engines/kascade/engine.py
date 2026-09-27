"""Minimal kascade Monte-Carlo behind GammaForge's uniform `Engine` protocol.

The event generator is a pure-array emission chain in `solver.py`. This wrapper is the
single unit and model boundary: shared inputs arrive in canonical CGS, laser fields are
sampled through `LaserField`, and the solver receives plain SI/dimensionless arrays. It
has no mutable engine-side configuration or automatic file output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ...io.bunch import Bunch, overlap_time_window
from ...io.interaction import InteractionParameters
from ...io.laser import LaserField, fit_gaussian_paraxial
from ...io.results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest, auto_ranges, slice_axis_values, slice_axis_widths
from ...io.units import C_CGS, E_ESU, HBAR_CGS, MEC2_CGS, ME_CGS, SIGMA_T_CGS
from ..base import RecomputeCost
from .schema import default_parameters
from .solver import (
    C_LIGHT_SI,
    ElectronChunk,
    TrajectoryGrid,
    beta_of,
    doppler_factor,
    kn_sigma_ratio,
    simulate_chunk,
)

__all__ = ["KascadeEngine"]

_M_TO_CM = 1e2
_ERG_TO_JOULE = 1e-7
_JOULE_TO_ERG = 1e7
_CM3_TO_M3_DENSITY = 1e6
_SIGMA_T_SI = SIGMA_T_CGS * 1e-4
_MEC2_JOULE = MEC2_CGS * _ERG_TO_JOULE

SUPPORTED_OUTPUTS: tuple[OutputKind, ...] = (
    OutputKind.TOTAL_YIELD,
    OutputKind.SPECTRUM,
    OutputKind.TEMPORAL_ENVELOPE,
    OutputKind.SPATIAL_DISTRIBUTION,
    OutputKind.ANGULAR_DISTRIBUTION,
    OutputKind.COLLIMATED_SPECTRUM,
    OutputKind.MACROPARTICLE_DUMP,
)

RECOMPUTE_COSTS: dict[str, RecomputeCost] = {"n_e": RecomputeCost.QUERY_ONLY}


@dataclass(frozen=True)
class _SIElectrons:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    gamma: np.ndarray
    electron_weight: np.ndarray

    @property
    def n_particles(self) -> int:
        return int(self.gamma.size)

    def chunk(self, start: int, stop: int) -> ElectronChunk:
        section = slice(start, stop)
        return ElectronChunk(
            x=self.x[section],
            y=self.y[section],
            z=self.z[section],
            theta_x=self.theta_x[section],
            theta_y=self.theta_y[section],
            gamma=self.gamma[section],
        )


def _bunch_to_si(bunch: Bunch, n_electrons: float) -> _SIElectrons:
    """Checked conversion at the engine boundary; weights become absolute here."""
    return _SIElectrons(
        x=bunch.get("x", "m"),
        y=bunch.get("y", "m"),
        z=bunch.get("z", "m"),
        theta_x=bunch.get("thx", "rad"),
        theta_y=bunch.get("thy", "rad"),
        gamma=bunch.get("gamma", "1"),
        electron_weight=n_electrons * bunch.weight,
    )


@dataclass(frozen=True)
class _RawResults:
    gamma_final: np.ndarray
    theta_x_final: np.ndarray
    theta_y_final: np.ndarray
    time_last_emit: np.ndarray
    lambda_total: np.ndarray
    n_photons: np.ndarray
    parent: np.ndarray
    generation: np.ndarray
    energy_joule: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: np.ndarray
    truncated: int


def _photon_density_scale_m3(laser: LaserField) -> float:
    """Photons/m^3 per unit cycle-averaged ``<a^2>`` (RES054)."""
    if hasattr(laser, "omega0"):
        omega = laser.omega0()
    else:
        omega = fit_gaussian_paraxial(laser).omega0()
    per_cm3 = (ME_CGS * C_CGS) ** 2 * omega / (4.0 * math.pi * HBAR_CGS * E_ESU**2)
    return per_cm3 * _CM3_TO_M3_DENSITY


def _trajectory_grid(
    bunch: Bunch,
    laser: LaserField,
    electrons: ElectronChunk,
    *,
    n_time: int,
    threshold: float,
    cos_collision: float,
    photon_energy_over_mec2: float,
    quantum: bool,
    reference_gamma: float,
) -> TrajectoryGrid:
    """Sample `LaserField` in CGS once, then construct kascade's SI optical depth."""
    t0, t1 = overlap_time_window(bunch, laser, threshold)
    span = np.maximum(0.0, t1 - t0)
    start = np.where(span > 0.0, t0, 0.0)
    fraction = np.linspace(0.0, 1.0, n_time)
    time = start[:, None] + span[:, None] * fraction[None, :]

    norm = np.sqrt(1.0 + electrons.theta_x**2 + electrons.theta_y**2)
    velocity_x = C_LIGHT_SI * electrons.theta_x / norm
    velocity_y = C_LIGHT_SI * electrons.theta_y / norm
    velocity_z = C_LIGHT_SI / norm

    intensity = np.asarray(
        laser.intensity_profile(
            bunch.x[:, None] + (_M_TO_CM * velocity_x)[:, None] * time,
            bunch.y[:, None] + (_M_TO_CM * velocity_y)[:, None] * time,
            bunch.z[:, None] + (_M_TO_CM * velocity_z)[:, None] * time,
            time,
        ),
        dtype=float,
    )

    beta0 = float(beta_of(reference_gamma))
    flux = 1.0 - beta0 * cos_collision
    sigma_scale = 1.0
    if quantum:
        recoil0 = float(
            doppler_factor(reference_gamma, cos_collision)
            * photon_energy_over_mec2
            / reference_gamma
        )
        sigma_scale = float(kn_sigma_ratio(0.5 * recoil0))
    rate = (
        intensity
        * _photon_density_scale_m3(laser)
        * _SIGMA_T_SI
        * C_LIGHT_SI
        * flux
        * sigma_scale
    )
    dt = np.diff(time, axis=1)
    increments = 0.5 * (rate[:, 1:] + rate[:, :-1]) * dt
    cumulative = np.concatenate(
        [np.zeros((electrons.gamma.size, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return TrajectoryGrid(
        time=time,
        intensity=intensity,
        cumulative=cumulative,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
    )


def _concatenate(parts, name: str, dtype=float):
    arrays = [getattr(part, name) for part in parts]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)


def _run_raw(
    interaction: InteractionParameters, params: Parameters
) -> tuple[_RawResults, _SIElectrons]:
    si = _bunch_to_si(interaction.bunch, interaction.N_e)
    laser = interaction.laser
    if hasattr(laser, "focusing_axes") and hasattr(laser, "photon_energy"):
        k_hat, _, _ = laser.focusing_axes()
        photon_energy = laser.photon_energy()
    else:
        metrics = fit_gaussian_paraxial(laser)
        k_hat, _, _ = metrics.focusing_axes()
        photon_energy = metrics.photon_energy()
    cos_collision = float(k_hat[2])
    photon_energy_over_mec2 = photon_energy / MEC2_CGS
    reference_gamma = interaction.beam.gamma0()
    quantum = params.get_choice("quantum") == "klein-nishina"
    rng = np.random.default_rng(interaction.sampling.seed)
    chunk_size = params.get_int("chunk")
    parts = []

    for start in range(0, si.n_particles, chunk_size):
        stop = min(start + chunk_size, si.n_particles)
        chunk = si.chunk(start, stop)
        bunch = interaction.bunch.select(np.arange(start, stop))
        grid = _trajectory_grid(
            bunch,
            interaction.laser,
            chunk,
            n_time=params.get_int("n_time"),
            threshold=params.get_float("threshold"),
            cos_collision=cos_collision,
            photon_energy_over_mec2=photon_energy_over_mec2,
            quantum=quantum,
            reference_gamma=reference_gamma,
        )
        part = simulate_chunk(
            chunk,
            grid,
            photon_energy_over_mec2=photon_energy_over_mec2,
            electron_rest_energy_joule=_MEC2_JOULE,
            cos_collision=cos_collision,
            quantum=quantum,
            max_photons=params.get_int("max_photons"),
            rng=rng,
        )
        if part.parent.size:
            part = replace(part, parent=part.parent + start)
        parts.append(part)

    raw = _RawResults(
        gamma_final=_concatenate(parts, "gamma_final"),
        theta_x_final=_concatenate(parts, "theta_x_final"),
        theta_y_final=_concatenate(parts, "theta_y_final"),
        time_last_emit=_concatenate(parts, "time_last_emit"),
        lambda_total=_concatenate(parts, "lambda_total"),
        n_photons=_concatenate(parts, "n_photons", int),
        parent=_concatenate(parts, "parent", int),
        generation=_concatenate(parts, "generation", int),
        energy_joule=_concatenate(parts, "energy_joule"),
        theta_x=_concatenate(parts, "theta_x"),
        theta_y=_concatenate(parts, "theta_y"),
        x=_concatenate(parts, "x"),
        y=_concatenate(parts, "y"),
        z=_concatenate(parts, "z"),
        time=_concatenate(parts, "time"),
        truncated=sum(part.truncated for part in parts),
    )
    return raw, si


def _histogram_slice(
    request: OutputRequest,
    ranges: dict[Axis, tuple[float, float]],
    samples: dict[Axis, np.ndarray],
    weights: np.ndarray,
) -> PhasespaceSlice:
    values = slice_axis_values(request, ranges)
    widths = slice_axis_widths(request, ranges)
    axes = tuple(values)
    bins = request.resolution
    ordered_ranges = [ranges[axis] for axis in axes]
    points = np.column_stack([samples[axis] for axis in axes])
    counts, _ = np.histogramdd(points, bins=bins, range=ordered_ranges, weights=weights)
    bin_volume = math.prod(
        (high - low) / count for (low, high), count in zip(ordered_ranges, bins)
    )
    return PhasespaceSlice(axes=values, distr=counts / bin_volume, widths=widths)


class KascadeEngine:
    """Sequential multi-photon Monte Carlo, retained only as a validation engine."""

    name = "kascade"
    schema: Parameters = default_parameters()
    supported_outputs: tuple[OutputKind, ...] = SUPPORTED_OUTPUTS
    recompute_costs: dict[str, RecomputeCost] = RECOMPUTE_COSTS

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        raw, si = _run_raw(interaction, params)
        photon_weight = si.electron_weight[raw.parent]
        total_yield = float(np.dot(si.electron_weight, raw.lambda_total))
        ranges = auto_ranges(
            interaction.target,
            interaction.beam,
            interaction.laser,
            interaction.bunch,
        )
        samples = {
            Axis.ENERGY: raw.energy_joule * _JOULE_TO_ERG,
            Axis.TIME: raw.time,
            Axis.X: raw.x * _M_TO_CM,
            Axis.Y: raw.y * _M_TO_CM,
            Axis.THETA_X: raw.theta_x,
            Axis.THETA_Y: raw.theta_y,
        }
        slices = {}
        wants_dump = False
        for request in interaction.target.outputs:
            if request.kind is OutputKind.TOTAL_YIELD:
                slices[request.kind] = PhasespaceSlice(axes={}, distr=np.asarray(total_yield))
            elif request.kind is OutputKind.MACROPARTICLE_DUMP:
                wants_dump = True
            elif request.kind in SUPPORTED_OUTPUTS:
                slices[request.kind] = _histogram_slice(
                    request, ranges[request.kind], samples, photon_weight
                )

        photons = None
        electrons = None
        if wants_dump:
            photons = PhotonMacroparticles(
                energy=samples[Axis.ENERGY],
                theta_x=raw.theta_x,
                theta_y=raw.theta_y,
                x=samples[Axis.X],
                y=samples[Axis.Y],
                z=raw.z * _M_TO_CM,
                t=raw.time,
                weight=photon_weight,
                order=raw.generation,
            )
            norm = np.sqrt(1.0 + si.theta_x**2 + si.theta_y**2)
            t_final = np.where(raw.n_photons > 0, raw.time_last_emit, 0.0)
            electrons = Bunch(
                x=(si.x + t_final * C_LIGHT_SI * si.theta_x / norm) * _M_TO_CM,
                y=(si.y + t_final * C_LIGHT_SI * si.theta_y / norm) * _M_TO_CM,
                z=(si.z + t_final * C_LIGHT_SI / norm) * _M_TO_CM,
                thx=raw.theta_x_final,
                thy=raw.theta_y_final,
                gamma=raw.gamma_final,
                weight=interaction.bunch.weight,
                meta={"time_last_emit": raw.time_last_emit, "n_photons": raw.n_photons},
            )

        warnings = []
        laser = interaction.laser
        ellipticity = getattr(laser, "ellipticity", 0.0)
        psi_pol = laser.m("psi_pol") if hasattr(laser, "m") and hasattr(laser, "psi_pol") else 0.0
        theta_xz = laser.m("theta_xz") if hasattr(laser, "m") and hasattr(laser, "theta_xz") else 0.0
        theta_yz = laser.m("theta_yz") if hasattr(laser, "m") and hasattr(laser, "theta_yz") else 0.0
        if (
            ellipticity != 0.0
            or psi_pol != 0.0
            or theta_xz != 0.0
            or theta_yz != 0.0
        ):
            warnings.append(
                "kascade uses a linear lab-x polarization kernel; "
                "only overlap, relative velocity, and resonance energy use the configured geometry"
            )
        return Results(
            photon_slices=slices,
            electrons=electrons,
            photons=photons,
            model_specific={
                "mean_photons_per_electron": total_yield / interaction.N_e,
                "measured_macrophotons": int(raw.energy_joule.size),
                "truncated_electrons": raw.truncated,
                "quantum": params.get_choice("quantum") == "klein-nishina",
                "warnings": tuple(warnings),
            },
        )

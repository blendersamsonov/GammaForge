"""Pure-array sequential Compton event generator used by `KascadeEngine`.

This is the predecessor's kascade emission chain reduced to the part Phase 5 needs:
optical-depth inversion, Thomson/Klein--Nishina angle sampling, sequential recoil, and
per-photon/per-electron output. Inputs are already plain SI or dimensionless arrays; the
engine boundary owns all unit conversion and `LaserField` sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "C_LIGHT_SI",
    "ElectronChunk",
    "TrajectoryGrid",
    "ChunkResult",
    "beta_of",
    "doppler_factor",
    "kn_sigma_ratio",
    "simulate_chunk",
]

C_LIGHT_SI = 299_792_458.0


@dataclass(frozen=True)
class ElectronChunk:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class TrajectoryGrid:
    """One chunk's lab times, cycle-averaged intensity and cumulative optical depth."""

    time: np.ndarray
    intensity: np.ndarray
    cumulative: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_z: np.ndarray


@dataclass(frozen=True)
class ChunkResult:
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


def beta_of(gamma):
    values = np.clip(np.asarray(gamma, dtype=float), 1.0 + 1e-12, None)
    return np.sqrt(np.clip(1.0 - 1.0 / values**2, 0.0, 1.0))


def doppler_factor(gamma, cos_collision: float):
    beta = beta_of(gamma)
    return (1.0 - beta * cos_collision) / (1.0 - beta)


def recoil_parameter(gamma, photon_energy_over_mec2: float, cos_collision: float):
    values = np.asarray(gamma, dtype=float)
    return doppler_factor(values, cos_collision) * photon_energy_over_mec2 / np.clip(
        values, 1e-30, None
    )


def kn_sigma_ratio(k):
    """Total Klein--Nishina cross section divided by the Thomson cross section."""
    k = np.asarray(k, dtype=float)
    small = k < 1e-3
    safe = np.where(small, 1e-3, k)
    term = (
        (1.0 + safe)
        / safe**3
        * (
            2.0 * safe * (1.0 + safe) / (1.0 + 2.0 * safe)
            - np.log1p(2.0 * safe)
        )
        + np.log1p(2.0 * safe) / (2.0 * safe)
        - (1.0 + 3.0 * safe) / (1.0 + 2.0 * safe) ** 2
    )
    closed = 0.75 * term
    return np.where(small, 1.0 - 2.0 * k + 5.2 * k**2, closed)


def _kn_over_thomson(u2, recoil):
    ratio = (1.0 + u2) / (1.0 + u2 + recoil)
    k = 0.5 * recoil
    inverse = (1.0 / ratio - 1.0) / np.clip(k, 1e-30, None)
    cos_theta = 1.0 - inverse
    sin2 = np.clip(1.0 - cos_theta**2, 0.0, 1.0)
    weight = ratio**2 * (ratio + 1.0 / ratio - sin2) / (1.0 + cos_theta**2)
    return np.clip(weight, 0.0, 1.0)


def _sample_emission_angle(gamma: np.ndarray, rng: np.random.Generator, recoil=None):
    """Sample the predecessor's linearly-polarized angular kernel."""
    theta_x = np.empty(gamma.size)
    theta_y = np.empty(gamma.size)
    remaining = np.arange(gamma.size)
    while remaining.size:
        values = gamma[remaining]
        radial_u = rng.random(remaining.size)
        radius = np.sqrt(radial_u / (1.0 - radial_u)) / values
        azimuth = 2.0 * np.pi * rng.random(remaining.size)
        tx = radius * np.cos(azimuth)
        ty = radius * np.sin(azimuth)
        u2 = (values * radius) ** 2
        weight = 1.0 - 4.0 * (values * tx) ** 2 / (1.0 + u2) ** 2
        if recoil is not None:
            weight *= _kn_over_thomson(u2, recoil[remaining])
        accepted = rng.random(remaining.size) < weight
        selected = remaining[accepted]
        theta_x[selected] = tx[accepted]
        theta_y[selected] = ty[accepted]
        remaining = remaining[~accepted]
    return theta_x, theta_y


def _invert_cumulative(cumulative: np.ndarray, time: np.ndarray, threshold: np.ndarray):
    """Invert each row's monotone optical-depth table by linear interpolation."""
    total = cumulative[:, -1]
    valid = threshold <= total
    clipped = np.clip(threshold, 0.0, total)
    index = np.sum(cumulative <= clipped[:, None], axis=1) - 1
    index = np.clip(index, 0, cumulative.shape[1] - 2)
    rows = np.arange(cumulative.shape[0])
    low = cumulative[rows, index]
    high = cumulative[rows, index + 1]
    fraction = np.where(high > low, (clipped - low) / np.maximum(high - low, 1e-300), 0.0)
    emit_time = time[rows, index] + fraction * (time[rows, index + 1] - time[rows, index])
    return emit_time, valid, index, fraction


def simulate_chunk(
    electrons: ElectronChunk,
    grid: TrajectoryGrid,
    *,
    photon_energy_over_mec2: float,
    electron_rest_energy_joule: float,
    cos_collision: float,
    quantum: bool,
    max_photons: int,
    rng: np.random.Generator,
) -> ChunkResult:
    """Run one independent chunk of the sequential emission chain."""
    n_electrons = electrons.gamma.size
    gamma = electrons.gamma.copy()
    theta_x = electrons.theta_x.copy()
    theta_y = electrons.theta_y.copy()
    previous = np.zeros(n_electrons)
    active = np.ones(n_electrons, dtype=bool)

    parents = []
    generations = []
    energies = []
    photon_theta_x = []
    photon_theta_y = []
    photon_x = []
    photon_y = []
    photon_z = []
    photon_time = []

    for generation in range(1, max_photons + 1):
        active_index = np.flatnonzero(active)
        if active_index.size == 0:
            break
        threshold = previous[active_index] - np.log1p(-rng.random(active_index.size))
        emit_time, valid, time_index, fraction = _invert_cumulative(
            grid.cumulative[active_index], grid.time[active_index], threshold
        )
        active[active_index[~valid]] = False
        emitted = active_index[valid]
        if emitted.size == 0:
            continue

        local_index = time_index[valid]
        local_fraction = fraction[valid]
        local_rows = np.arange(active_index.size)[valid]
        intensity = grid.intensity[active_index][local_rows, local_index]
        intensity += local_fraction * (
            grid.intensity[active_index][local_rows, local_index + 1] - intensity
        )

        gamma_before = gamma[emitted]
        doppler = doppler_factor(gamma_before, cos_collision)
        recoil = doppler * photon_energy_over_mec2 / gamma_before
        sampled_recoil = recoil if quantum else None
        relative_x, relative_y = _sample_emission_angle(gamma_before, rng, sampled_recoil)
        theta2 = relative_x**2 + relative_y**2
        recoil_term = recoil if quantum else 0.0
        energy_over_mec2 = doppler * photon_energy_over_mec2 / (
            1.0 + gamma_before**2 * theta2 + intensity + recoil_term
        )
        gamma_after = np.clip(gamma_before - energy_over_mec2, 1.0 + 1e-12, None)

        lab_x = theta_x[emitted] + relative_x
        lab_y = theta_y[emitted] + relative_y
        theta_x[emitted] -= (energy_over_mec2 / gamma_after) * np.sin(relative_x)
        theta_y[emitted] -= (energy_over_mec2 / gamma_after) * np.sin(relative_y)
        gamma[emitted] = gamma_after
        previous[emitted] = threshold[valid]

        parents.append(emitted.copy())
        generations.append(np.full(emitted.size, generation, dtype=int))
        energies.append(energy_over_mec2 * electron_rest_energy_joule)
        photon_theta_x.append(lab_x)
        photon_theta_y.append(lab_y)
        photon_time.append(emit_time[valid])
        photon_x.append(electrons.x[emitted] + grid.velocity_x[emitted] * emit_time[valid])
        photon_y.append(electrons.y[emitted] + grid.velocity_y[emitted] * emit_time[valid])
        photon_z.append(electrons.z[emitted] + grid.velocity_z[emitted] * emit_time[valid])

    parent = np.concatenate(parents) if parents else np.empty(0, dtype=int)
    time = np.concatenate(photon_time) if photon_time else np.empty(0)
    time_last = np.zeros(n_electrons)
    if parent.size:
        np.maximum.at(time_last, parent, time)
    n_photons = (
        np.bincount(parent, minlength=n_electrons)
        if parent.size
        else np.zeros(n_electrons, dtype=int)
    )

    def concatenate_or_empty(values, dtype=float):
        return np.concatenate(values) if values else np.empty(0, dtype=dtype)

    return ChunkResult(
        gamma_final=gamma,
        theta_x_final=theta_x,
        theta_y_final=theta_y,
        time_last_emit=time_last,
        lambda_total=grid.cumulative[:, -1],
        n_photons=n_photons,
        parent=parent,
        generation=concatenate_or_empty(generations, int),
        energy_joule=concatenate_or_empty(energies),
        theta_x=concatenate_or_empty(photon_theta_x),
        theta_y=concatenate_or_empty(photon_theta_y),
        x=concatenate_or_empty(photon_x),
        y=concatenate_or_empty(photon_y),
        z=concatenate_or_empty(photon_z),
        time=time,
        truncated=int(active.sum()),
    )

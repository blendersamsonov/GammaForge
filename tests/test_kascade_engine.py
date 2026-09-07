"""Minimal kascade port: boundary, engine contract, and Thomson sanity anchor."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.base import Engine
from gammaforge.engines.analytical.formulas import overlap_yield
from gammaforge.engines.kascade.engine import KascadeEngine, _bunch_to_si, _histogram_slice
from gammaforge.engines.kascade.solver import (
    C_LIGHT_SI,
    ElectronChunk,
    TrajectoryGrid,
    simulate_chunk,
)
from gammaforge.io.bunch import Bunch, GaussianElectronBeam
from gammaforge.io.interaction import InteractionParameters, SamplingSpec
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest, Target
from gammaforge.io.units import SIGMA_T_CGS, Quantity as Q
from gammaforge.validation import scenarios


def _params(**overrides):
    return KascadeEngine.schema.with_values(**{"n_time": 81, "chunk": 257, **overrides})


def _interaction(n_particles=4_000, outputs=()):
    scenario = replace(
        scenarios.BASELINE,
        sampling=replace(
            scenarios.BASELINE.sampling,
            n_particles=n_particles,
            prefilter=0.0,
        ),
        target=replace(scenarios.BASELINE.target, outputs=outputs),
    )
    return scenarios.build(scenario)


def test_kascade_engine_conforms_to_the_engine_protocol():
    assert isinstance(KascadeEngine(), Engine)


def test_bunch_is_converted_once_to_si_and_absolute_electron_weights():
    bunch = Bunch(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        z=np.array([5.0, 6.0]),
        thx=np.array([0.1, 0.2]),
        thy=np.array([0.3, 0.4]),
        gamma=np.array([100.0, 200.0]),
        weight=np.array([0.25, 0.75]),
    )
    converted = _bunch_to_si(bunch, n_electrons=40.0)

    assert np.array_equal(converted.x, np.array([0.01, 0.02]))
    assert np.array_equal(converted.y, np.array([0.03, 0.04]))
    assert np.array_equal(converted.z, np.array([0.05, 0.06]))
    assert converted.theta_x is bunch.thx
    assert converted.theta_y is bunch.thy
    assert converted.gamma is bunch.gamma
    assert np.array_equal(converted.electron_weight, np.array([10.0, 30.0]))


def test_histogram_slice_integrates_to_the_captured_photon_weights():
    request = OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, (2, 2))
    ranges = {Axis.THETA_X: (0.0, 2.0), Axis.THETA_Y: (0.0, 4.0)}
    samples = {
        Axis.THETA_X: np.array([0.2, 1.8, 3.0]),
        Axis.THETA_Y: np.array([0.2, 3.8, 1.0]),
    }
    weights = np.array([2.0, 3.0, 5.0])
    histogram = _histogram_slice(request, ranges, samples, weights)
    assert histogram.integrate() == pytest.approx(5.0)


def test_supported_outputs_are_filled_and_macroparticles_are_cgs():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(24,)),
        OutputRequest(OutputKind.TEMPORAL_ENVELOPE, resolution=(12,)),
        OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, resolution=(8, 7)),
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(9, 8)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(12, 5, 5)),
        OutputRequest(OutputKind.MACROPARTICLE_DUMP),
    )
    interaction = _interaction(outputs=requests)
    results = KascadeEngine().run(interaction, _params())

    assert set(results.photon_slices) == set(KascadeEngine.supported_outputs) - {
        OutputKind.MACROPARTICLE_DUMP
    }
    assert results.photons is not None
    assert results.electrons is not None
    assert results.photons.n_macroparticles > 0
    assert results.electrons.n_particles == interaction.bunch.n_particles
    assert np.array_equal(results.electrons.weight, interaction.bunch.weight)
    assert np.all(results.photons.energy > 0.0)
    assert np.max(np.abs(results.photons.x)) < 1.0  # cm, not the predecessor's metres
    assert results.photon_slices[OutputKind.SPECTRUM].axis_order == (Axis.ENERGY,)

    scaled = results.scaled(2.0)
    assert np.array_equal(scaled.photons.energy, results.photons.energy)
    assert np.array_equal(scaled.photons.weight, 2.0 * results.photons.weight)
    assert scaled.electrons is results.electrons


def test_same_interaction_and_seed_reproduce_the_mc_exactly():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(32,)),
        OutputRequest(OutputKind.MACROPARTICLE_DUMP),
    )
    interaction = _interaction(outputs=requests)
    first = KascadeEngine().run(interaction, _params())
    again = KascadeEngine().run(interaction, _params())

    assert np.array_equal(first.photons.energy, again.photons.energy)
    assert np.array_equal(first.photons.theta_x, again.photons.theta_x)
    assert np.array_equal(
        first.photon_slices[OutputKind.SPECTRUM].distr,
        again.photon_slices[OutputKind.SPECTRUM].distr,
    )


def test_klein_nishina_mode_reduces_the_yield_and_returns_photons():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.MACROPARTICLE_DUMP),
    )
    interaction = _interaction(outputs=requests)
    thomson = KascadeEngine().run(interaction, _params())
    quantum = KascadeEngine().run(interaction, _params(quantum="klein-nishina"))

    assert quantum.photon_slices[OutputKind.TOTAL_YIELD].integrate() < (
        thomson.photon_slices[OutputKind.TOTAL_YIELD].integrate()
    )
    assert quantum.photons is not None
    assert quantum.photons.n_macroparticles > 0
    assert np.all(quantum.photons.energy > 0.0)


def test_head_on_thomson_optical_depth_matches_the_closed_form():
    """An on-axis electron crossing a broad, effectively unfocused Gaussian pulse.

    Integrating the photon column gives ``N_L / (2 pi sigma_x sigma_y)``. Multiplying
    by ``sigma_T`` is an absolute normalization check independent of every other engine.
    """
    beam = GaussianElectronBeam(
        bunch_charge=Q(1.0, "statC"),
        kinetic_energy=Q(200.0, "MeV"),
        rel_energy_spread=0.0,
        sigma_x=Q(100.0, "um"),
        sigma_y=Q(100.0, "um"),
        emit_x=Q(1.0, "um * rad"),
        emit_y=Q(1.0, "um * rad"),
        sigma_z=Q(0.03, "um"),
    )
    laser = GaussianParaxialLaser(
        pulse_energy=Q(0.05, "J"),
        wavelength=Q(0.8, "um"),
        sigma_x=Q(100.0, "um"),
        sigma_y=Q(100.0, "um"),
        duration=Q(0.1, "fs"),
    )
    bunch = Bunch(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        thx=np.zeros(1), thy=np.zeros(1), gamma=np.array([beam.gamma0()]),
        weight=np.ones(1), gaussian_fit=beam,
    )
    target = Target(
        theta_x_col=Q(1.0, "mrad"),
        theta_y_col=Q(1.0, "mrad"),
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD),),
    )
    interaction = InteractionParameters(
        beam=beam,
        laser=laser,
        bunch=bunch,
        target=target,
        N_e=1.0,
        sampling=SamplingSpec(n_particles=1, seed=17, prefilter=0.0),
    )

    results = KascadeEngine().run(
        interaction,
        KascadeEngine.schema.with_values(n_time=801, threshold=1e-10),
    )
    actual = results.photon_slices[OutputKind.TOTAL_YIELD].integrate()
    expected = (
        laser.n_photons()
        * SIGMA_T_CGS
        / (2.0 * math.pi * laser.m("sigma_x") * laser.m("sigma_y"))
    )
    assert actual == pytest.approx(expected, rel=2e-4)


def test_thomson_yield_agrees_with_the_independent_analytical_engine_across_the_bank():
    """Kascade independently samples trajectories; analytical integrates Gaussians."""
    for scenario in scenarios.SCENARIOS:
        sampled = replace(
            scenario,
            sampling=replace(scenario.sampling, n_particles=20_000, prefilter=0.0),
            target=replace(
                scenario.target,
                outputs=(OutputRequest(OutputKind.TOTAL_YIELD),),
            ),
        )
        interaction = scenarios.build(sampled)
        result = KascadeEngine().run(interaction, _params(n_time=201))
        actual = result.photon_slices[OutputKind.TOTAL_YIELD].integrate()
        expected = overlap_yield(interaction.beam, interaction.laser, interaction.N_e)

        assert actual == pytest.approx(expected, rel=0.015), scenario.name


def test_negative_emission_times_are_preserved_in_last_emission_reduction():
    """Audit A06 regression: all emissions at t < 0 must retain their true last-emission time."""
    n_electrons = 64
    n_time = 50
    time = np.linspace(-2e-12, -1e-12, n_time)
    time_grid = np.tile(time, (n_electrons, 1))
    cum_rate = np.linspace(0.0, 100.0, n_time)
    cum_grid = np.tile(cum_rate, (n_electrons, 1))

    electrons = ElectronChunk(
        x=np.zeros(n_electrons),
        y=np.zeros(n_electrons),
        z=np.zeros(n_electrons),
        theta_x=np.zeros(n_electrons),
        theta_y=np.zeros(n_electrons),
        gamma=np.full(n_electrons, 1000.0),
    )
    grid = TrajectoryGrid(
        time=time_grid,
        intensity=np.zeros_like(time_grid),
        cumulative=cum_grid,
        velocity_x=np.zeros(n_electrons),
        velocity_y=np.zeros(n_electrons),
        velocity_z=np.full(n_electrons, C_LIGHT_SI),
    )
    result = simulate_chunk(
        electrons,
        grid,
        photon_energy_over_mec2=1e-5,
        electron_rest_energy_joule=8.187e-14,
        cos_collision=-1.0,
        quantum=False,
        max_photons=1,
        rng=np.random.default_rng(42),
    )

    assert result.n_photons.sum() == n_electrons
    assert np.all(result.time < 0.0)
    assert np.all(result.time_last_emit < 0.0)
    assert np.array_equal(result.time_last_emit, result.time)


@pytest.mark.parametrize(
    "time_range",
    [
        (-3e-12, -1e-12),  # all-negative
        (-2e-12, 2e-12),   # mixed-sign
        (1e-12, 3e-12),    # all-positive
    ],
)
def test_time_last_emit_matches_parent_maximum_across_histories(time_range):
    """Each emitting parent's last time must equal the maximum of its own photon times."""
    t0, t1 = time_range
    n_electrons = 24
    n_time = 60
    time = np.linspace(t0, t1, n_time)
    time_grid = np.tile(time, (n_electrons, 1))
    cum_grid = np.tile(np.linspace(0.0, 20.0, n_time), (n_electrons, 1))

    electrons = ElectronChunk(
        x=np.zeros(n_electrons),
        y=np.zeros(n_electrons),
        z=np.zeros(n_electrons),
        theta_x=np.zeros(n_electrons),
        theta_y=np.zeros(n_electrons),
        gamma=np.full(n_electrons, 1000.0),
    )
    grid = TrajectoryGrid(
        time=time_grid,
        intensity=np.zeros_like(time_grid),
        cumulative=cum_grid,
        velocity_x=np.zeros(n_electrons),
        velocity_y=np.zeros(n_electrons),
        velocity_z=np.full(n_electrons, C_LIGHT_SI),
    )
    result = simulate_chunk(
        electrons,
        grid,
        photon_energy_over_mec2=1e-5,
        electron_rest_energy_joule=8.187e-14,
        cos_collision=-1.0,
        quantum=False,
        max_photons=4,  # repeated-parent histories
        rng=np.random.default_rng(7),
    )

    assert result.parent.size > 0
    assert np.any(result.n_photons > 1)
    for p in range(n_electrons):
        mask = result.parent == p
        if np.any(mask):
            assert result.time_last_emit[p] == np.max(result.time[mask])
        else:
            assert result.time_last_emit[p] == 0.0


def test_time_last_emit_handles_nonemitters_zero_photons_and_empty_input():
    # 1. Partial emission: even electrons emit, odd electrons do not
    n_electrons = 10
    n_time = 30
    time_grid = np.tile(np.linspace(-2e-12, -1e-12, n_time), (n_electrons, 1))
    cum_grid = np.zeros((n_electrons, n_time))
    cum_grid[::2, :] = np.linspace(0.0, 50.0, n_time)

    electrons = ElectronChunk(
        x=np.zeros(n_electrons),
        y=np.zeros(n_electrons),
        z=np.zeros(n_electrons),
        theta_x=np.zeros(n_electrons),
        theta_y=np.zeros(n_electrons),
        gamma=np.full(n_electrons, 1000.0),
    )
    grid = TrajectoryGrid(
        time=time_grid,
        intensity=np.zeros_like(time_grid),
        cumulative=cum_grid,
        velocity_x=np.zeros(n_electrons),
        velocity_y=np.zeros(n_electrons),
        velocity_z=np.full(n_electrons, C_LIGHT_SI),
    )
    result = simulate_chunk(
        electrons,
        grid,
        photon_energy_over_mec2=1e-5,
        electron_rest_energy_joule=8.187e-14,
        cos_collision=-1.0,
        quantum=False,
        max_photons=2,
        rng=np.random.default_rng(11),
    )
    assert np.all(result.n_photons[1::2] == 0)
    assert np.all(result.time_last_emit[1::2] == 0.0)
    assert np.all(result.n_photons[::2] > 0)
    assert np.all(result.time_last_emit[::2] < 0.0)

    # 2. Zero photons overall
    cum_zero = np.zeros((n_electrons, n_time))
    grid_zero = TrajectoryGrid(
        time=time_grid,
        intensity=np.zeros_like(time_grid),
        cumulative=cum_zero,
        velocity_x=np.zeros(n_electrons),
        velocity_y=np.zeros(n_electrons),
        velocity_z=np.full(n_electrons, C_LIGHT_SI),
    )
    res_zero = simulate_chunk(
        electrons,
        grid_zero,
        photon_energy_over_mec2=1e-5,
        electron_rest_energy_joule=8.187e-14,
        cos_collision=-1.0,
        quantum=False,
        max_photons=2,
        rng=np.random.default_rng(12),
    )
    assert res_zero.parent.size == 0
    assert np.all(res_zero.n_photons == 0)
    assert np.all(res_zero.time_last_emit == 0.0)

    # 3. Empty input
    empty_e = ElectronChunk(
        x=np.empty(0),
        y=np.empty(0),
        z=np.empty(0),
        theta_x=np.empty(0),
        theta_y=np.empty(0),
        gamma=np.empty(0),
    )
    empty_grid = TrajectoryGrid(
        time=np.empty((0, 10)),
        intensity=np.empty((0, 10)),
        cumulative=np.empty((0, 10)),
        velocity_x=np.empty(0),
        velocity_y=np.empty(0),
        velocity_z=np.empty(0),
    )
    res_empty = simulate_chunk(
        empty_e,
        empty_grid,
        photon_energy_over_mec2=1e-5,
        electron_rest_energy_joule=8.187e-14,
        cos_collision=-1.0,
        quantum=False,
        max_photons=2,
        rng=np.random.default_rng(13),
    )
    assert res_empty.time_last_emit.size == 0
    assert res_empty.n_photons.size == 0


def test_final_electron_position_reconstruction_preserves_nonemitters_and_avoids_sentinels():
    interaction = _interaction(
        n_particles=500,
        outputs=(OutputRequest(OutputKind.MACROPARTICLE_DUMP),),
    )
    results = KascadeEngine().run(interaction, _params())
    electrons = results.electrons
    assert electrons is not None

    t_last = electrons.meta["time_last_emit"]
    n_photons = electrons.meta["n_photons"]

    # All exported arrays and metadata must be finite (no -inf or NaN sentinels)
    assert np.all(np.isfinite(electrons.x))
    assert np.all(np.isfinite(electrons.y))
    assert np.all(np.isfinite(electrons.z))
    assert np.all(np.isfinite(t_last))

    # Non-emitters stay at the focus (t = 0.0, identical to initial bunch positions)
    non_emitters = n_photons == 0
    assert np.any(non_emitters)
    assert np.all(t_last[non_emitters] == 0.0)
    assert np.allclose(electrons.x[non_emitters], interaction.bunch.x[non_emitters])
    assert np.allclose(electrons.y[non_emitters], interaction.bunch.y[non_emitters])
    assert np.allclose(electrons.z[non_emitters], interaction.bunch.z[non_emitters])

    # Emitters have recorded last emission times and drift accordingly
    emitters = n_photons > 0
    assert np.any(emitters)
    assert np.any(t_last[emitters] < 0.0)
    assert np.any(t_last[emitters] > 0.0)
    norm = np.sqrt(1.0 + interaction.bunch.thx[emitters] ** 2 + interaction.bunch.thy[emitters] ** 2)
    expected_x = (
        interaction.bunch.x[emitters] * 1e-2
        + t_last[emitters] * C_LIGHT_SI * interaction.bunch.thx[emitters] / norm
    ) * 1e2
    assert np.allclose(electrons.x[emitters], expected_x)

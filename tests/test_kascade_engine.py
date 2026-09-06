"""Minimal kascade port: boundary, engine contract, and Thomson sanity anchor."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.base import Engine
from gammaforge.engines.analytical.formulas import overlap_yield
from gammaforge.engines.kascade.engine import KascadeEngine, _bunch_to_si
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

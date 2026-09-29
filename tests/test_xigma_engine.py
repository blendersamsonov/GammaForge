"""End-to-end numerical invariants of the public xigma engine."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier2]

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import sample_gaussian_bunch
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest, auto_ranges
from gammaforge.io.units import Quantity as Q
from gammaforge.validation import scenarios


def _interaction(n_particles=4000):
    scenario = replace(
        scenarios.BASELINE,
        sampling=replace(scenarios.BASELINE.sampling, n_particles=n_particles),
    )
    return scenarios.build(scenario)


def _engine_params(**overrides):
    return XigmaEngine.schema.with_values(
        n_bins_gamma=10,
        n_bins_theta_x=10,
        n_bins_theta_y=10,
        n_bins_a0_shape=16,
        n_bins_ahat=4,
        **overrides,
    )


def test_total_yield_and_spectrum_integral_converge_to_the_same_number():
    interaction = _interaction(n_particles=3000)
    errors = []
    for n in (60, 600):
        target = replace(
            interaction.target,
            outputs=(
                OutputRequest(OutputKind.TOTAL_YIELD),
                OutputRequest(OutputKind.SPECTRUM, resolution=(n,)),
            ),
        )
        results = XigmaEngine().run(replace(interaction, target=target), _engine_params())
        total = results.photon_slices[OutputKind.TOTAL_YIELD].integrate()
        spectrum = results.photon_slices[OutputKind.SPECTRUM].integrate()
        errors.append(abs(spectrum - total) / total)

    assert errors[1] < errors[0]
    assert errors[1] < 0.01


def test_angle_resolved_and_angle_integrated_normalizations_agree():
    interaction = _interaction()
    wide = 4.6 / interaction.beam.gamma0()
    target = replace(
        interaction.target,
        theta_x_col=Q(wide, "rad"),
        theta_y_col=Q(wide, "rad"),
        outputs=(
            OutputRequest(OutputKind.SPECTRUM, resolution=(80,)),
            OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(80, 25, 25)),
        ),
    )
    params = XigmaEngine.schema.with_values(
        n_bins_gamma=32,
        n_bins_theta_x=24,
        n_bins_theta_y=24,
        n_bins_a0_shape=64,
        n_bins_ahat=8,
        scheme="cic",
        backend="numpy",
    )
    results = XigmaEngine().run(replace(interaction, target=target), params)
    spectrum = results.photon_slices[OutputKind.SPECTRUM]
    cube = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    theta_x = cube.axes[Axis.THETA_X]
    theta_y = cube.axes[Axis.THETA_Y]
    d_n_d_e = np.trapezoid(np.trapezoid(cube.distr, theta_y, axis=2), theta_x, axis=1)
    ratio = np.trapezoid(d_n_d_e, spectrum.axes[Axis.ENERGY]) / spectrum.integrate()

    assert 0.75 < ratio < 1.05


def test_chirped_angular_integration_and_auto_energy_range_capture_the_line():
    interaction = _interaction(n_particles=96)

    class ShiftedCarrier:
        def __init__(self, laser):
            self.laser = laser

        def __getattr__(self, name):
            return getattr(self.laser, name)

        def carrier_phase_four_gradient(self, x, y, z, t):
            return (1.2 * self.laser.omega0(), 0.0, 0.0, 0.0)

    requests = (
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, (1, 1), {
            Axis.THETA_X: (-1e-5, 1e-5), Axis.THETA_Y: (-1e-5, 1e-5),
        }),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (128, 1, 1)),
    )
    interaction = replace(
        interaction,
        laser=ShiftedCarrier(interaction.laser),
        target=replace(interaction.target, outputs=requests),
    )
    params = _engine_params(
        backend="numpy", line_model="delta", n_steps=32, n_bins_chirp=4,
    ).with_values(n_bins_ahat=1)
    collision = Collision(interaction, params)
    samples = collision.build_overlap()
    assert np.min(samples.chirp_mean[samples.luminosity > 0]) > 1.5
    s = collision._energy_quadrature_grid()
    assert s[-1] > 1.5 * np.max(samples.gamma**2)

    results = collision.run(requests)
    assert results.photon_slices[OutputKind.ANGULAR_DISTRIBUTION].distr[0, 0] > 0.0
    collimated = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    nominal = auto_ranges(interaction.target, interaction.beam, interaction.laser,
                          interaction.bunch)[OutputKind.COLLIMATED_SPECTRUM][Axis.ENERGY]
    assert collimated.axes[Axis.ENERGY][-1] > nominal[1]
    assert np.any(collimated.distr > 0.0)


def test_illumination_window_improves_coarse_stage0_accuracy():
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 4_000, 0)
    n_e = beam.n_electrons()
    reference = integrate_trajectories(
        bunch, laser, n_e, n_steps=4000, threshold=1e-9
    ).total_yield()

    def error(window, n_steps):
        value = integrate_trajectories(
            bunch,
            laser,
            n_e,
            n_steps=n_steps,
            threshold=1e-6,
            window=window,
        ).total_yield()
        return abs(value / reference - 1.0)

    assert error("illumination", 50) < 0.2 * error("active_region", 50)
    assert error("illumination", 20) < error("active_region", 20)

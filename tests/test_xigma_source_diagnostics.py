"""Source diagnostics preserve mass, requested grids, and Stage-0 chunk invariance."""
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import overlap_time_window
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest, auto_ranges
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios

pytestmark = [pytest.mark.tier1, pytest.mark.fast]


@pytest.fixture
def interaction():
    scenario = replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=48))
    return scenarios.build(scenario)


def integrate(interaction, **kwargs):
    return integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e,
                                  n_steps=32, **kwargs)


def diagnostic_mass(diagnostics):
    return (np.sum(diagnostics.time_envelope * np.diff(diagnostics.t_edges)),
            np.sum(diagnostics.spatial_envelope * np.diff(diagnostics.spatial_edges[0])[:, None]
                   * np.diff(diagnostics.spatial_edges[1])[None, :]))


@pytest.mark.parametrize('chunk', [1, 7, 48])
def test_fixed_nonuniform_grids_conserve_mass_across_chunks(interaction, chunk):
    t0, t1 = overlap_time_window(interaction.bunch, interaction.laser)
    low, high = t0.min(), t1.max()
    t_edges = low + (high - low) * np.linspace(0, 1, 13)**2
    spatial_edges = (np.array([-1., -.001, 0., .001, 1.]), np.array([-1., 0., .002, 1.]))
    plain = integrate(interaction)
    whole = integrate(interaction, t_edges=t_edges, spatial_edges=spatial_edges)
    split = integrate(interaction, t_edges=t_edges, spatial_edges=spatial_edges, chunk=chunk)
    np.testing.assert_array_equal(split.diagnostics.t_edges, t_edges)
    np.testing.assert_array_equal(split.diagnostics.spatial_edges[0], spatial_edges[0])
    np.testing.assert_allclose(split.diagnostics.time_envelope, whole.diagnostics.time_envelope, rtol=2e-12)
    np.testing.assert_allclose(split.diagnostics.spatial_envelope, whole.diagnostics.spatial_envelope, rtol=2e-12)
    np.testing.assert_array_equal(split.luminosity, plain.luminosity)
    np.testing.assert_array_equal(split.a0_shape, plain.a0_shape)
    np.testing.assert_allclose(diagnostic_mass(split.diagnostics), plain.total_yield(), rtol=2e-12)


def test_histogram_retry_does_not_double_count(interaction, monkeypatch):
    edges = np.linspace(-1e-9, 1e-9, 9)
    expected = integrate(interaction, t_edges=edges)
    histogram = np.bincount
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError('forced histogram allocation failure')
        return histogram(*args, **kwargs)

    monkeypatch.setattr(np, 'bincount', fail_once)
    actual = integrate(interaction, t_edges=edges, chunk=48)
    assert calls == 3
    np.testing.assert_allclose(actual.diagnostics.time_envelope, expected.diagnostics.time_envelope, rtol=2e-12)


def requests(nx=7, ny=3, nt=11, half=1.):
    return (OutputRequest(OutputKind.TOTAL_YIELD),
            OutputRequest(OutputKind.TEMPORAL_ENVELOPE, (nt,)),
            OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (nx, ny),
                          {Axis.X: (-half, half), Axis.Y: (-half, half)}))


def test_outputs_use_requested_grid_and_explicit_measure(interaction):
    target = replace(interaction.target, outputs=requests())
    interaction = replace(interaction, target=target)
    result = XigmaEngine().run(interaction, XigmaEngine.schema.with_values(n_steps=32))
    temporal = result.photon_slices[OutputKind.TEMPORAL_ENVELOPE]
    spatial = result.photon_slices[OutputKind.SPATIAL_DISTRIBUTION]
    total = result.photon_slices[OutputKind.TOTAL_YIELD].integrate()
    assert temporal.distr.shape == (11,)
    assert spatial.distr.shape == (7, 3)
    for axis, n in ((Axis.X, 7), (Axis.Y, 3)):
        np.testing.assert_allclose(spatial.axes[axis], -1 + (np.arange(n) + .5) * 2 / n, atol=1e-15)
    for histogram in (temporal, spatial):
        assert set(histogram.widths) == set(histogram.axes)
        assert histogram.integrate() == pytest.approx(total, rel=2e-12)
    assert result.scaled(2).photon_slices[OutputKind.TEMPORAL_ENVELOPE].integrate() == pytest.approx(2*total)


def test_clipping_is_reported_without_renormalization(interaction):
    collision = Collision(interaction, XigmaEngine.schema.with_values(n_steps=32))
    full = collision.run(requests())
    clipped = collision.run(requests(half=1e-5))
    total = full.photon_slices[OutputKind.TOTAL_YIELD].integrate()
    captured = clipped.photon_slices[OutputKind.SPATIAL_DISTRIBUTION].integrate()
    assert captured < .1 * total
    accounting = clipped.model_specific['stage0_diagnostics'][OutputKind.SPATIAL_DISTRIBUTION.value]
    assert accounting['captured_fraction'] == pytest.approx(captured / total)
    assert accounting['outside_fraction'] == pytest.approx(1 - captured / total)
    assert clipped.scaled(2).model_specific['stage0_diagnostics'] == clipped.model_specific['stage0_diagnostics']
    assert any('exclude' in warning for warning in clipped.model_specific['warnings'])


@pytest.mark.parametrize('empty', [False, True])
def test_no_overlap_has_zero_histograms_with_finite_display_axes(interaction, empty):
    bunch = interaction.bunch.select(np.zeros(48, dtype=bool)) if empty else interaction.bunch
    laser = replace(interaction.laser, x_off=Quantity(1e3, 'cm'))
    interaction = replace(interaction, bunch=bunch, laser=laser)
    collision = Collision(interaction, XigmaEngine.schema.with_values(n_steps=32))
    result = collision.run(requests(nx=1, ny=1, nt=1))
    for histogram in result.photon_slices.values():
        assert histogram.integrate() == 0
        assert all(np.all(np.isfinite(axis)) for axis in histogram.axes.values())
    assert any('display interval' in warning for warning in result.model_specific['warnings'])


def test_new_grid_recomputes_once_and_cached_diagnostics_are_read_only(interaction, monkeypatch):
    import gammaforge.engines.xigma.collision as module
    original = module.integrate_trajectories
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'integrate_trajectories', counted)
    collision = Collision(interaction, XigmaEngine.schema.with_values(n_steps=32))
    collision.run(requests())
    collision.run(requests())
    assert calls == 1
    collision.run(requests(nx=4, nt=17))
    assert calls == 2
    assert collision.build_overlap().diagnostics.time_envelope.size == 17
    assert calls == 2
    with pytest.raises(ValueError, match='read-only'):
        collision.build_overlap().diagnostics.t_edges[0] = 0


@pytest.mark.parametrize('edges', [[0], [0, 0], [1, 0], [0, np.inf], [[0, 1]]])
def test_invalid_diagnostic_edges_fail_at_boundary(interaction, edges):
    with pytest.raises(ValueError, match='diagnostic edges'):
        integrate(interaction, t_edges=edges)


def test_spatial_coordinates_follow_displaced_emission_events(interaction):
    bunch = replace(interaction.bunch, x=np.full(48, .003), y=np.full(48, -.002),
                    thx=np.zeros(48), thy=np.zeros(48))
    laser = replace(interaction.laser, x_off=Quantity(.003, 'cm'), y_off=Quantity(-.002, 'cm'))
    interaction = replace(interaction, bunch=bunch, laser=laser)
    samples = integrate(interaction, spatial_edges=(np.array([-.01, 0., .01]),
                                                    np.array([-.01, 0., .01])))
    mass = samples.diagnostics.spatial_envelope * .01**2
    assert mass[1, 0] == pytest.approx(samples.total_yield(), rel=1e-12)
    assert np.count_nonzero(mass) == 1


def test_crossed_pulse_train_diagnostics_conserve_overlap(interaction):
    from gammaforge.io.laser import PulseTrainParaxialLaser
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(.01, 'J'), wavelength=Quantity(800, 'nm'),
        sigma_x=Quantity(20, 'um'), sigma_y=Quantity(30, 'um'),
        subpulse_duration=Quantity(30, 'fs'), repetition_period=Quantity(200, 'fs'),
        n_subpulses=3, theta_xz=Quantity(.2, 'rad'), theta_yz=Quantity(-.1, 'rad'))
    interaction = replace(interaction, laser=laser)
    result = Collision(interaction, XigmaEngine.schema.with_values(n_steps=256)).run(requests())
    total = result.photon_slices[OutputKind.TOTAL_YIELD].integrate()
    assert total > 0
    for kind in (OutputKind.TEMPORAL_ENVELOPE, OutputKind.SPATIAL_DISTRIBUTION):
        assert result.photon_slices[kind].integrate() == pytest.approx(total, rel=2e-12)

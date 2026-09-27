"""Numerical measures, physical output ranges, and reproducible interaction assembly."""

import numpy as np
import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.interaction import PREFILTER_OFF, SamplingSpec, build_interaction
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.plotting import project_slice
from gammaforge.io.results import Axis, PhasespaceSlice, Results
from gammaforge.io.target import (
    RANGE_HEADROOM,
    OutputKind,
    OutputRequest,
    Target,
    auto_ranges,
    compton_edge_energy,
)
from gammaforge.io.units import Quantity as Q


def make_beam(**overrides):
    values = dict(
        bunch_charge=Q(100, "pC"),
        kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.01,
        sigma_x=Q(20, "um"),
        sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"),
        emit_y=Q(2e-7, "cm * rad"),
        sigma_z=Q(100, "um"),
    )
    return GaussianElectronBeam(**{**values, **overrides})


def make_laser(**overrides):
    values = dict(
        pulse_energy=Q(1, "J"),
        wavelength=Q(800, "nm"),
        sigma_x=Q(10, "um"),
        sigma_y=Q(10, "um"),
        duration=Q(30, "fs"),
    )
    return GaussianParaxialLaser(**{**values, **overrides})


def make_target(*outputs):
    return Target(Q(1, "mrad"), Q(2, "mrad"), outputs=tuple(outputs))


def test_nonuniform_histogram_projection_preserves_mass():
    axes = {
        Axis.ENERGY: np.array([0.1, 0.7]),
        Axis.THETA_X: np.array([-0.4, 0.3]),
        Axis.THETA_Y: np.array([-0.5, 0.2]),
    }
    widths = {
        Axis.ENERGY: np.array([0.2, 0.8]),
        Axis.THETA_X: np.array([0.3, 0.7]),
        Axis.THETA_Y: np.array([0.4, 0.6]),
    }
    density = np.arange(1.0, 9.0).reshape(2, 2, 2)
    source = PhasespaceSlice(axes, density, widths=widths)
    projected = project_slice(source, (Axis.ENERGY,))
    expected = np.sum(
        density
        * widths[Axis.THETA_X][None, :, None]
        * widths[Axis.THETA_Y][None, None, :],
        axis=(1, 2),
    )

    np.testing.assert_allclose(projected.distr, expected)
    assert projected.integrate() == pytest.approx(source.integrate())


def test_histogram_and_smooth_measures_integrate_correctly():
    one_bin = PhasespaceSlice(
        {Axis.ENERGY: np.array([2.0])},
        np.array([3.0]),
        widths={Axis.ENERGY: np.array([0.25])},
    )
    assert one_bin.integrate() == pytest.approx(0.75)

    energy = np.array([0.0, 0.5, 2.0])
    density = energy**2 + 1.0
    smooth = PhasespaceSlice({Axis.ENERGY: energy}, density)
    assert smooth.integrate() == pytest.approx(np.trapezoid(density, energy))


def test_three_dimensional_integral_equals_energy_marginal():
    energy = np.linspace(1.0, 4.0, 33)
    theta_x = np.linspace(-1.0, 1.0, 17)
    theta_y = np.linspace(-1.0, 1.0, 21)
    density = (
        np.exp(-energy[:, None, None])
        * np.exp(-theta_x[None, :, None] ** 2)
        * np.exp(-theta_y[None, None, :] ** 2)
    )
    cube = PhasespaceSlice(
        {Axis.ENERGY: energy, Axis.THETA_X: theta_x, Axis.THETA_Y: theta_y},
        density,
    )
    marginal = PhasespaceSlice(
        {Axis.ENERGY: energy},
        np.trapezoid(np.trapezoid(density, theta_y, axis=2), theta_x, axis=1),
    )

    assert cube.integrate() == pytest.approx(marginal.integrate(), rel=1e-12)


def test_result_rescaling_is_exactly_linear():
    results = Results(
        {
            OutputKind.TOTAL_YIELD: PhasespaceSlice({}, np.asarray(10.0)),
            OutputKind.SPECTRUM: PhasespaceSlice(
                {Axis.ENERGY: np.arange(4.0)}, np.ones(4)
            ),
        }
    )
    doubled = results.scaled(2.0)

    for kind in results.photon_slices:
        assert doubled.photon_slices[kind].integrate() == pytest.approx(
            2.0 * results.photon_slices[kind].integrate()
        )


def test_compton_edge_has_thomson_limit_and_recoil_reduction():
    laser = make_laser()
    low = make_beam(kinetic_energy=Q(100, "MeV"))
    high = make_beam(kinetic_energy=Q(100, "GeV"))
    photon = laser.photon_energy()

    low_thomson = 4.0 * low.gamma0() ** 2 * photon
    assert compton_edge_energy(low, photon) == pytest.approx(low_thomson, rel=5e-3)
    assert compton_edge_energy(low, photon) < low_thomson
    assert compton_edge_energy(high, photon) / (4.0 * high.gamma0() ** 2 * photon) < 0.5


def test_auto_ranges_cover_the_physical_energy_and_angular_support():
    beam, laser = make_beam(), make_laser()
    target = make_target(
        OutputRequest(OutputKind.SPECTRUM, (64,)),
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, (16, 16)),
    )
    ranges = auto_ranges(target, beam, laser)
    energy = ranges[OutputKind.SPECTRUM][Axis.ENERGY]
    angle = ranges[OutputKind.ANGULAR_DISTRIBUTION][Axis.THETA_X]

    assert energy[0] == 0.0
    assert energy[1] == pytest.approx(
        RANGE_HEADROOM * compton_edge_energy(beam, laser.photon_energy())
    )
    assert angle[0] == pytest.approx(-angle[1])
    assert angle[1] > 1.0 / beam.gamma0()

    hotter = make_beam(kinetic_energy=Q(400, "MeV"))
    hotter_angle = auto_ranges(target, hotter, laser)[OutputKind.ANGULAR_DISTRIBUTION][
        Axis.THETA_X
    ]
    assert hotter_angle[1] < angle[1]


def test_interaction_sampling_is_reproducible_and_prefilter_conserves_weight():
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    filtered = build_interaction(
        make_beam(), make_laser(), target, SamplingSpec(5000, 8, 1e-3)
    )
    repeated = build_interaction(
        make_beam(), make_laser(), target, SamplingSpec(5000, 8, 1e-3)
    )
    unfiltered = build_interaction(
        make_beam(), make_laser(), target, SamplingSpec(5000, 8, PREFILTER_OFF)
    )

    for first, second in zip(filtered.bunch.arrays(), repeated.bunch.arrays()):
        assert np.array_equal(first, second)
    assert filtered.N_e == unfiltered.N_e
    assert filtered.bunch.n_particles < unfiltered.bunch.n_particles
    assert np.allclose(filtered.bunch.weight, 1.0 / 5000)


def test_charge_retarget_changes_only_the_electron_count():
    interaction = build_interaction(
        make_beam(),
        make_laser(),
        make_target(OutputRequest(OutputKind.SPECTRUM, (32,))),
        SamplingSpec(5000, 8),
    )
    recharged = interaction.with_charge(2.0 * interaction.beam.bunch_charge)

    assert recharged.N_e == pytest.approx(2.0 * interaction.N_e)
    assert recharged.bunch is interaction.bunch

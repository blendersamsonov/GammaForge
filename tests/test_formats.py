"""Representative scientific-data serialization round trips."""

import numpy as np
import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.fields import (
    beam_from_parameters,
    beam_to_parameters,
    laser_from_parameters,
    laser_to_parameters,
)
from gammaforge.io.formats.hdf5 import load_results, save_results
from gammaforge.io.formats.sdds import load_elegant_ele, save_elegant_ele
from gammaforge.io.formats.yaml_spec import SPEC_VERSION, load_spec, save_spec
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from gammaforge.io.target import OutputKind
from gammaforge.io.units import C_CGS, EV_CGS, Quantity as Q


def make_beam():
    return GaussianElectronBeam(
        bunch_charge=Q(100, "pC"),
        kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.012,
        sigma_x=Q(20, "um"),
        sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"),
        emit_y=Q(2e-7, "cm * rad"),
        sigma_z=Q(100, "um"),
        rho_x_gamma=0.15,
        rho_z_gamma=0.3,
        alpha_x=0.4,
    )


def make_laser():
    return GaussianParaxialLaser(
        pulse_energy=Q(1, "J"),
        wavelength=Q(800, "nm"),
        sigma_x=Q(10, "um"),
        sigma_y=Q(15, "um"),
        duration=Q(30, "fs"),
        z_fx=Q(10, "um"),
        z_fy=Q(-20, "um"),
        theta_xz=Q(0.05, "rad"),
        theta_yz=Q(-0.02, "rad"),
        psi_focus=Q(0.3, "rad"),
        psi_pol=Q(0.7, "rad"),
        ellipticity=0.0,
        beta_ff=0.1,
    )


def test_yaml_round_trip_preserves_dimensioned_inputs(tmp_path):
    path = tmp_path / "scenario.yaml"
    beam = make_beam()
    laser = make_laser()
    save_spec(
        path,
        beam=beam_to_parameters(beam),
        laser=laser_to_parameters(laser),
    )
    restored = load_spec(path)
    beam_back = beam_from_parameters(restored["beam"])
    laser_back = laser_from_parameters(restored["laser"])

    for name in beam.UNITS:
        assert beam_back.m(name) == pytest.approx(beam.m(name), rel=1e-11, abs=1e-30)
    for name in laser.UNITS:
        assert laser_back.m(name) == pytest.approx(laser.m(name), rel=1e-11, abs=1e-30)


def test_yaml_accepts_human_units_and_width_time_conventions(tmp_path):
    path = tmp_path / "hand.yaml"
    path.write_text(
        f"version: {SPEC_VERSION}\n"
        "beam:\n"
        "  bunch_charge: {value: 100, unit: pC}\n"
        "  kinetic_energy: {value: 250, unit: MeV}\n"
        "  sigma_x: {value: 47.09640090061899, unit: um, convention: fwhm_intensity}\n"
        "  sigma_z: {value: 1.0, unit: ps}\n"
        "  rel_energy_spread: 0.01\n"
    )
    beam = beam_from_parameters(load_spec(path)["beam"])

    assert beam.m("bunch_charge") == pytest.approx(100e-12 * C_CGS / 10.0)
    assert beam.m("kinetic_energy") == pytest.approx(250e6 * EV_CGS)
    assert beam.m("sigma_x") == pytest.approx(20e-4, rel=1e-9)
    assert beam.m("sigma_z") == pytest.approx(1e-12 * C_CGS)


def test_elegant_round_trip_preserves_distribution_and_si_boundary(tmp_path):
    bunch = sample_gaussian_bunch(make_beam(), 3000, seed=5)
    path = tmp_path / "bunch.ele"
    save_elegant_ele(bunch, path)
    restored = load_elegant_ele(path)

    for name in ("x", "y", "z", "thx", "thy", "gamma"):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(bunch, name), rtol=1e-8, atol=1e-12
        )
    assert restored.weight.sum() == pytest.approx(1.0)
    sigma_x_line = next(
        line for line in path.read_text().splitlines() if "name=SigmaX " in line
    )
    written_sigma_x = float(sigma_x_line.split("fix=")[1].split()[0])
    assert written_sigma_x == pytest.approx(float(np.std(bunch.x)) / 100.0, rel=1e-5)


def make_results():
    energy = np.linspace(1e-8, 5e-7, 24)
    theta_x = np.linspace(-1e-3, 1e-3, 9)
    theta_y = np.linspace(-2e-3, 2e-3, 7)
    rng = np.random.default_rng(0)
    return Results(
        photon_slices={
            OutputKind.TOTAL_YIELD: PhasespaceSlice({}, np.asarray(1.234e9)),
            OutputKind.COLLIMATED_SPECTRUM: PhasespaceSlice(
                {
                    Axis.ENERGY: energy,
                    Axis.THETA_X: theta_x,
                    Axis.THETA_Y: theta_y,
                },
                rng.random((24, 9, 7)),
            ),
        },
        photons=PhotonMacroparticles(
            energy=rng.random(50),
            theta_x=rng.random(50),
            theta_y=rng.random(50),
            x=rng.random(50),
            y=rng.random(50),
            z=rng.random(50),
            t=rng.random(50),
            weight=np.full(50, 0.02),
        ),
    )


def test_hdf5_round_trip_preserves_axes_integrals_and_particles(tmp_path):
    results = make_results()
    path = tmp_path / "run.h5"
    save_results(results, path)
    restored = load_results(path, kind_from_name=OutputKind)

    for kind, original in results.photon_slices.items():
        loaded = restored.photon_slices[kind]
        assert loaded.axis_order == original.axis_order
        np.testing.assert_array_equal(loaded.distr, original.distr)
        assert loaded.integrate() == pytest.approx(original.integrate(), rel=1e-12)
    assert restored.photons is not None
    np.testing.assert_array_equal(restored.photons.energy, results.photons.energy)
    np.testing.assert_array_equal(restored.photons.weight, results.photons.weight)


def test_hdf5_round_trip_preserves_nonuniform_histogram_measure(tmp_path):
    histogram = PhasespaceSlice(
        {Axis.ENERGY: np.array([0.5, 2.0])},
        np.array([3.0, 5.0]),
        widths={Axis.ENERGY: np.array([1.0, 2.0])},
    )
    path = tmp_path / "histogram.h5"
    save_results(Results({OutputKind.SPECTRUM: histogram}), path)
    restored = load_results(path, kind_from_name=OutputKind).photon_slices[
        OutputKind.SPECTRUM
    ]

    np.testing.assert_array_equal(
        restored.widths[Axis.ENERGY], histogram.widths[Axis.ENERGY]
    )
    assert restored.integrate() == pytest.approx(histogram.integrate())

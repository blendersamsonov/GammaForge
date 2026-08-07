"""Serialization round-trips: YAML specs, elegant `.ele`, HDF5 results (GRAND_PLAN.md §8)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from gammaforge.io.bunch import Bunch, GaussianElectronBeam, fit_gaussian, sample_gaussian_bunch
from gammaforge.io.fields import (
    BEAM_FIELDS,
    LASER_FIELDS,
    SAMPLING_FIELDS,
    beam_from_parameters,
    beam_to_parameters,
    laser_from_parameters,
    laser_to_parameters,
    sampling_from_parameters,
    sampling_to_parameters,
)
from gammaforge.io.formats.hdf5 import load_results, save_results, sidecar_path
from gammaforge.io.formats.sdds import load_elegant_ele, save_elegant_ele
from gammaforge.io.formats.yaml_spec import SPEC_VERSION, load_spec, save_spec
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from gammaforge.io.schema import Parameters, SchemaError
from gammaforge.io.target import OutputKind
from gammaforge.io.units import C_CGS, EV_CGS, MEC2_CGS, Quantity as Q


def make_beam() -> GaussianElectronBeam:
    return GaussianElectronBeam(
        bunch_charge=Q(100, "pC"), kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.012, sigma_x=Q(20, "um"), sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"), emit_y=Q(2e-7, "cm * rad"), sigma_z=Q(100, "um"),
        rho_x_gamma=0.15, rho_z_gamma=0.3, alpha_x=0.4,
    )


def make_laser() -> GaussianParaxialLaser:
    return GaussianParaxialLaser(
        pulse_energy=Q(1, "J"), wavelength=Q(800, "nm"), sigma_x=Q(10, "um"),
        sigma_y=Q(15, "um"), duration=Q(30, "fs"), z_fx=Q(10, "um"), z_fy=Q(-20, "um"),
        theta_xz=Q(0.05, "rad"), theta_yz=Q(-0.02, "rad"), psi_focus=Q(0.3, "rad"),
        psi_pol=Q(0.7, "rad"), ellipticity=0.0, beta_ff=0.1,
    )


# ---------------------------------------------------------------------------
# Field declarations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "specs, cls",
    [(BEAM_FIELDS, GaussianElectronBeam), (LASER_FIELDS, GaussianParaxialLaser),
     (SAMPLING_FIELDS, SamplingSpec)],
)
def test_every_declared_field_matches_a_dataclass_field(specs, cls):
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    assert {spec.key for spec in specs} <= names


def test_dataclass_parameter_bridges_round_trip():
    for obj, to_params, from_params in [
        (make_beam(), beam_to_parameters, beam_from_parameters),
        (make_laser(), laser_to_parameters, laser_from_parameters),
        (SamplingSpec(n_particles=4096, seed=17, prefilter=5e-4), sampling_to_parameters,
         sampling_from_parameters),
    ]:
        assert from_params(to_params(obj)) == obj


def test_sampling_integer_fields_stay_integers():
    spec = sampling_from_parameters(Parameters.from_specs(SAMPLING_FIELDS, n_particles=1234, seed=9))
    assert isinstance(spec.n_particles, int) and isinstance(spec.seed, int)


# ---------------------------------------------------------------------------
# YAML specs
# ---------------------------------------------------------------------------
def test_yaml_spec_round_trips_through_display_units(tmp_path):
    path = tmp_path / "scenario.yaml"
    save_spec(path, beam=beam_to_parameters(make_beam()), laser=laser_to_parameters(make_laser()))
    loaded = load_spec(path)
    beam_back = beam_from_parameters(loaded["beam"])
    laser_back = laser_from_parameters(loaded["laser"])
    def value(obj, name):
        return obj.m(name) if name in obj.UNITS else getattr(obj, name)

    for name in (spec.key for spec in BEAM_FIELDS):
        assert value(beam_back, name) == pytest.approx(value(make_beam(), name), rel=1e-11, abs=1e-30), name
    for name in (spec.key for spec in LASER_FIELDS):
        assert value(laser_back, name) == pytest.approx(value(make_laser(), name), rel=1e-11, abs=1e-30), name


def test_yaml_file_states_its_units_explicitly(tmp_path):
    path = tmp_path / "scenario.yaml"
    save_spec(path, beam=beam_to_parameters(make_beam()))
    document = yaml.safe_load(path.read_text())
    assert document["version"] == SPEC_VERSION
    # A dimensional field carries its unit; a dimensionless one is a bare number.
    assert document["beam"]["bunch_charge"]["unit"] == "pC"
    assert document["beam"]["sigma_x"]["unit"] == "um"
    assert document["beam"]["sigma_x"]["convention"] == "sigma_intensity_rms"
    assert isinstance(document["beam"]["rho_x_gamma"], float)


def test_yaml_accepts_hand_written_units_and_conventions(tmp_path):
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
    assert beam.m("sigma_x") == pytest.approx(20e-4, rel=1e-9)  # the FWHM above is 20 um RMS
    assert beam.m("sigma_z") == pytest.approx(1e-12 * C_CGS)  # a duration entered for a length
    # Unspecified fields fall back to their declared defaults.
    assert beam.m("sigma_y") == pytest.approx(20e-4)


def test_yaml_rejects_an_unknown_parameter(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(f"version: {SPEC_VERSION}\nbeam:\n  nonexistent: 1.0\n")
    with pytest.raises(SchemaError, match="unknown parameters"):
        load_spec(path)


def test_yaml_rejects_an_unknown_group_and_a_wrong_version(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(f"version: {SPEC_VERSION}\nplasma:\n  density: 1.0\n")
    with pytest.raises(SchemaError, match="unknown group"):
        load_spec(path)
    path.write_text("version: 99\nbeam: {}\n")
    with pytest.raises(SchemaError, match="not supported"):
        load_spec(path)


def test_yaml_rejects_an_unknown_convention(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"version: {SPEC_VERSION}\nbeam:\n  sigma_x: {{value: 1, unit: um, convention: hwhm}}\n"
    )
    with pytest.raises(SchemaError, match="not a WidthConvention"):
        load_spec(path)


def test_save_spec_rejects_an_unknown_group(tmp_path):
    with pytest.raises(SchemaError, match="unknown parameter group"):
        save_spec(tmp_path / "x.yaml", plasma=beam_to_parameters(make_beam()))


# ---------------------------------------------------------------------------
# elegant .ele
# ---------------------------------------------------------------------------
def test_ele_round_trip_preserves_the_distribution(tmp_path):
    beam = make_beam()
    bunch = sample_gaussian_bunch(beam, 3000, seed=5)
    path = tmp_path / "bunch.ele"
    save_elegant_ele(bunch, path)
    loaded = load_elegant_ele(path)

    assert loaded.n_particles == bunch.n_particles
    for name in ("x", "y", "z", "thx", "thy"):
        assert np.allclose(getattr(loaded, name), getattr(bunch, name), rtol=1e-8, atol=1e-12)
    assert np.allclose(loaded.gamma, bunch.gamma, rtol=1e-8)


def test_ele_load_normalizes_weights_and_carries_no_charge(tmp_path):
    bunch = sample_gaussian_bunch(make_beam(), 500, seed=5)
    path = tmp_path / "bunch.ele"
    save_elegant_ele(bunch, path)
    loaded = load_elegant_ele(path)
    # §3.2/§8: weights come back relative, and N_e is not recoverable from the file.
    assert np.allclose(loaded.weight, 1.0 / 500)
    assert loaded.weight.sum() == pytest.approx(1.0)
    assert loaded.meta["charge_from_file"] is False
    assert loaded.gaussian_fit is None


def test_ele_file_is_written_in_si_metres(tmp_path):
    bunch = sample_gaussian_bunch(make_beam(), 200, seed=5)
    path = tmp_path / "bunch.ele"
    save_elegant_ele(bunch, path)
    text = path.read_text()
    assert "units=m " in text and "units=GeV" in text
    sigma_x_line = next(line for line in text.splitlines() if "name=SigmaX " in line)
    written = float(sigma_x_line.split("fix=")[1].split()[0])
    assert written == pytest.approx(float(np.std(bunch.x)) / 100.0, rel=1e-5)


def test_ele_fit_recovers_the_original_beam(tmp_path):
    beam = make_beam()
    path = tmp_path / "bunch.ele"
    save_elegant_ele(sample_gaussian_bunch(beam, 100_000, seed=5), path)
    fitted = fit_gaussian(load_elegant_ele(path), bunch_charge=beam.bunch_charge)
    for name in ("sigma_x", "sigma_y", "sigma_z", "emit_x", "emit_y"):
        assert fitted.m(name) == pytest.approx(beam.m(name), rel=0.03), name


def test_ele_reference_gamma_is_honoured(tmp_path):
    bunch = sample_gaussian_bunch(make_beam(), 500, seed=5)
    path = tmp_path / "bunch.ele"
    save_elegant_ele(bunch, path, reference_gamma=300.0)
    text = path.read_text()
    energy_line = next(line for line in text.splitlines() if "name=Energy " in line)
    written = float(energy_line.split("fix=")[1].split()[0])
    assert written == pytest.approx(300.0 * MEC2_CGS / EV_CGS / 1e9, rel=1e-5)


def test_ele_refuses_to_silently_drop_non_uniform_weights(tmp_path):
    bunch = sample_gaussian_bunch(make_beam(), 100, seed=5)
    uneven = Bunch(
        x=bunch.x, y=bunch.y, z=bunch.z, thx=bunch.thx, thy=bunch.thy, gamma=bunch.gamma,
        weight=np.linspace(0.5, 1.5, 100) / 100,
    )
    with pytest.raises(ValueError, match="non-uniform weights"):
        save_elegant_ele(uneven, tmp_path / "x.ele")


def test_ele_rejects_a_non_sdds_file(tmp_path):
    path = tmp_path / "nope.ele"
    path.write_text("hello\n")
    with pytest.raises(ValueError, match="not an SDDS file"):
        load_elegant_ele(path)


def test_ele_rejects_a_file_without_an_energy_header(tmp_path):
    path = tmp_path / "noenergy.ele"
    path.write_text(
        "SDDS1\n"
        + "".join(f"&column name={c} type=double &\n" for c in ("x", "xp", "y", "yp", "z", "dP"))
        + "&data mode=ascii\n 0 0 0 0 0 0\n&end\n"
    )
    with pytest.raises(ValueError, match="mean beam energy"):
        load_elegant_ele(path)


# ---------------------------------------------------------------------------
# HDF5 results
# ---------------------------------------------------------------------------
def make_results() -> Results:
    energy = np.linspace(1e-8, 5e-7, 24)
    theta_x = np.linspace(-1e-3, 1e-3, 9)
    theta_y = np.linspace(-2e-3, 2e-3, 7)
    rng = np.random.default_rng(0)
    return Results(
        photon_slices={
            OutputKind.TOTAL_YIELD: PhasespaceSlice(axes={}, distr=np.asarray(1.234e9)),
            OutputKind.SPECTRUM: PhasespaceSlice(
                axes={Axis.ENERGY: energy}, distr=rng.random(24)
            ),
            OutputKind.COLLIMATED_SPECTRUM: PhasespaceSlice(
                axes={Axis.ENERGY: energy, Axis.THETA_X: theta_x, Axis.THETA_Y: theta_y},
                distr=rng.random((24, 9, 7)),
            ),
        },
        photons=PhotonMacroparticles(
            energy=rng.random(50), theta_x=rng.random(50), theta_y=rng.random(50),
            x=rng.random(50), y=rng.random(50), z=rng.random(50), t=rng.random(50),
            weight=np.full(50, 0.02),
        ),
    )


def test_hdf5_results_round_trip(tmp_path):
    results = make_results()
    path = tmp_path / "run.h5"
    save_results(results, path)
    loaded = load_results(path, kind_from_name=OutputKind)

    assert set(loaded.photon_slices) == set(results.photon_slices)
    for kind, original in results.photon_slices.items():
        restored = loaded.photon_slices[kind]
        # Axis *order* must survive, not just axis membership: HDF5 groups iterate
        # alphabetically, which would transpose a (energy, theta_x, theta_y) slice.
        assert restored.axis_order == original.axis_order
        assert np.array_equal(restored.distr, original.distr)
        for axis, values in original.axes.items():
            assert np.array_equal(restored.axes[axis], values)


def test_hdf5_preserves_the_axis_order_of_an_asymmetric_slice(tmp_path):
    results = make_results()
    path = tmp_path / "run.h5"
    save_results(results, path)
    restored = load_results(path, kind_from_name=OutputKind).photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    assert restored.axis_order == (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y)
    assert restored.distr.shape == (24, 9, 7)


def test_hdf5_round_trips_photon_macroparticles(tmp_path):
    results = make_results()
    path = tmp_path / "run.h5"
    save_results(results, path)
    loaded = load_results(path, kind_from_name=OutputKind)
    assert loaded.photons is not None
    assert loaded.photons.n_macroparticles == 50
    assert np.array_equal(loaded.photons.energy, results.photons.energy)
    assert loaded.photons.order is None


def test_hdf5_writes_a_yaml_sidecar_with_the_parameters(tmp_path):
    path = tmp_path / "run.h5"
    save_results(make_results(), path, beam=beam_to_parameters(make_beam()),
                 laser=laser_to_parameters(make_laser()))
    sidecar = sidecar_path(path)
    assert sidecar.exists()
    groups = load_spec(sidecar)
    assert beam_from_parameters(groups["beam"]).m("sigma_x") == pytest.approx(20e-4, rel=1e-11)


def test_hdf5_slices_carry_their_units(tmp_path):
    import h5py

    path = tmp_path / "run.h5"
    save_results(make_results(), path)
    with h5py.File(path, "r") as handle:
        assert handle["slices/spectrum/axes/energy"].attrs["unit"] == "erg"
        assert handle["slices/collimated_spectrum/axes/theta_x"].attrs["unit"] == "rad"


def test_integrals_survive_the_round_trip(tmp_path):
    results = make_results()
    path = tmp_path / "run.h5"
    save_results(results, path)
    loaded = load_results(path, kind_from_name=OutputKind)
    for kind in (OutputKind.SPECTRUM, OutputKind.COLLIMATED_SPECTRUM):
        assert loaded.photon_slices[kind].integrate() == pytest.approx(
            results.photon_slices[kind].integrate(), rel=1e-12
        )


def test_field_units_are_declared_by_the_dataclass_that_stores_them():
    # `fields.py` reads each canonical unit from the dataclass rather than restating it,
    # so the unit is written down once. Nothing would break if they differed — the
    # boundary converts either way — but there is no reason for them to.
    import dataclasses

    for specs, cls in [(BEAM_FIELDS, GaussianElectronBeam), (LASER_FIELDS, GaussianParaxialLaser)]:
        for spec in specs:
            if spec.key in cls.UNITS:
                assert spec.unit == cls.UNITS[spec.key], spec.key


@pytest.mark.parametrize(
    "specs, key, expected",
    [
        # A longitudinal extent may be given either way, so which half leads matters:
        # a pulse duration reads as fs, a bunch length as um.
        (LASER_FIELDS, "duration", "fs"),
        (BEAM_FIELDS, "sigma_z", "um"),
        # Canonical units are for storing, not reading: nobody writes a bunch charge in
        # statC or a beam energy in erg.
        (BEAM_FIELDS, "bunch_charge", "pC"),
        (BEAM_FIELDS, "kinetic_energy", "MeV"),
        (BEAM_FIELDS, "emit_x", "mm * mrad"),
        (LASER_FIELDS, "pulse_energy", "J"),
        (LASER_FIELDS, "wavelength", "nm"),
    ],
)
def test_files_are_written_in_the_unit_a_person_would_write(specs, key, expected):
    spec = next(spec for spec in specs if spec.key == key)
    assert spec.display_units[0] == expected

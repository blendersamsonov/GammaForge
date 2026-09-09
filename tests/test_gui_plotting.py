from importlib.util import find_spec

import numpy as np
import pytest


if find_spec("plotly") is None:
    pytest.skip("requires the gui extra (plotly)", allow_module_level=True)
import plotly.graph_objects  # noqa: F401 -- fail on a broken declared GUI installation

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.io.bunch import Bunch, GaussianElectronBeam
from gammaforge.io.drawing import geometry_model
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.plotting import collimated_projections, display_scale, export_overlay, export_plot, plot_slice, project_slice
from gammaforge.io.results import Axis, PhasespaceSlice, PhotonMacroparticles, Results
from gammaforge.gui.outputs import particle_summary
from gammaforge.io.formats.hdf5 import load_results, save_results
from gammaforge.io.target import OutputKind
from gammaforge.io.units import Quantity as Q


def _collimated():
    # Deliberately asymmetric/nonuniform: a raw sum would pass neither identity below.
    axes = {
        Axis.ENERGY: np.array([2.0, 3.0, 7.0]),
        Axis.THETA_X: np.array([-2.0, -0.25, 1.0]),
        Axis.THETA_Y: np.array([-1.5, 0.1, 3.0]),
    }
    return PhasespaceSlice(axes, np.arange(27.0).reshape(3, 3, 3) + 1.0)


def test_collimated_projections_use_the_slice_quadrature():
    source = _collimated()
    projections = collimated_projections(source)
    assert np.isclose(projections["spectrum"].integrate(), source.integrate())
    assert np.isclose(projections["energy_theta_x"].integrate(), source.integrate())
    assert np.isclose(projections["energy_theta_y"].integrate(), source.integrate())


def test_projection_preserves_a_nonuniform_energy_marginal():
    source = _collimated()
    projected = project_slice(source, (Axis.ENERGY,))
    expected = np.trapezoid(np.trapezoid(source.distr, source.axes[Axis.THETA_Y], axis=2),
                            source.axes[Axis.THETA_X], axis=1)
    np.testing.assert_allclose(projected.distr, expected)


def test_display_density_jacobian_preserves_integral():
    source = PhasespaceSlice({Axis.ENERGY: np.array([1.0, 2.0, 4.0])}, np.array([3.0, 5.0, 7.0]))
    scale = display_scale(Axis.ENERGY)
    display_density = source.distr / scale
    assert np.isclose(np.trapezoid(display_density, source.axes[Axis.ENERGY] * scale), source.integrate())


def test_geometry_model_uses_the_laser_rotation_not_a_second_convention():
    beam = GaussianElectronBeam(Q(100, "pC"), Q(100, "MeV"), .01, Q(20, "um"), Q(30, "um"),
                                Q(1e-7, "cm * rad"), Q(2e-7, "cm * rad"), Q(100, "um"))
    laser = GaussianParaxialLaser(Q(1, "J"), Q(800, "nm"), Q(10, "um"), Q(15, "um"), Q(30, "fs"),
                                  theta_xz=Q(.3, "rad"), theta_yz=Q(-.2, "rad"), psi_focus=Q(.4, "rad"))
    model = geometry_model(beam, laser)
    expected = laser.focusing_axes()
    np.testing.assert_allclose(model["k_hat"], expected[0])
    np.testing.assert_allclose(model["focus_axes"][0], expected[1])


def test_headless_plot_exports_and_hdf5_download_payload_round_trip(tmp_path):
    spectrum = PhasespaceSlice({Axis.ENERGY: np.array([1.0, 2.0, 4.0])}, np.array([3.0, 5.0, 7.0]))
    png = tmp_path / "spectrum.png"
    pdf = tmp_path / "overlay.pdf"
    export_plot(spectrum, png)
    export_overlay({"xigma": spectrum, "analytical": spectrum.scaled(.5)}, pdf)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")

    h5 = tmp_path / "download.h5"
    results = Results({OutputKind.SPECTRUM: spectrum})
    save_results(results, h5)
    restored = load_results(h5, OutputKind)
    np.testing.assert_allclose(restored.photon_slices[OutputKind.SPECTRUM].distr, spectrum.distr)


def test_2d_plot_keeps_axis_order_and_one_bin_projection_is_explicitly_rejected():
    spatial = PhasespaceSlice(
        {Axis.X: np.array([1.0, 4.0]), Axis.Y: np.array([2.0, 3.0, 9.0])},
        np.arange(6.0).reshape(2, 3),
    )
    figure = plot_slice(spatial)
    np.testing.assert_allclose(figure.data[0].x, spatial.axes[Axis.X] * display_scale(Axis.X))
    np.testing.assert_allclose(figure.data[0].y, spatial.axes[Axis.Y] * display_scale(Axis.Y))
    np.testing.assert_allclose(figure.data[0].z, spatial.distr.T / (display_scale(Axis.X) * display_scale(Axis.Y)))

    insufficient = PhasespaceSlice(
        {Axis.ENERGY: np.array([1.0, 2.0]), Axis.THETA_X: np.array([0.0]), Axis.THETA_Y: np.array([-.1, .1])},
        np.ones((2, 1, 2)),
    )
    try:
        collimated_projections(insufficient)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("a one-bin angular projection must not invent a width")


def test_zero_angle_section_keeps_held_angle_in_its_density_units():
    source = PhasespaceSlice(
        {Axis.ENERGY: np.array([1.0, 2.0]), Axis.THETA_X: np.array([-.1, .1]), Axis.THETA_Y: np.array([-.2, .2])},
        np.ones((2, 2, 2)),
    )
    section = collimated_projections(source)["energy_at_theta_x_zero"]
    figure = plot_slice(section, density_axes=source.axis_order)
    expected = 1.0 / (display_scale(Axis.ENERGY) * display_scale(Axis.THETA_X) * display_scale(Axis.THETA_Y))
    np.testing.assert_allclose(figure.data[0].z, np.full((2, 2), expected))
    assert figure.data[0].colorbar.title.text == "Photons / eV / mrad / mrad"


def test_particle_summary_is_generic_over_results_contract():
    values = np.arange(3.0)
    result = Results(
        {},
        electrons=Bunch(values, values, values, values, values, values + 2.0,
                        np.full(3, 1 / 3)),
        photons=PhotonMacroparticles(
            energy=values + 1.0, theta_x=values, theta_y=values,
            x=values, y=values, z=values, t=values,
            weight=np.array([2.0, 3.0, 4.0]),
        ),
        model_specific={"truncated_electrons": 2, "warnings": ("limited",)},
    )

    assert particle_summary(result) == (
        ("Photon macroparticles", "3"),
        ("Weighted photons", "9"),
        ("Final electrons", "3"),
        ("Truncated Electrons", "2"),
    )

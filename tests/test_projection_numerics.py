"""Integral-preserving projections and geometry used by plotting/export code."""

from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("plotly") is None:
    pytest.skip("requires the plotting extra", allow_module_level=True)

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.drawing import geometry_model
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.plotting import collimated_projections, display_scale, project_slice
from gammaforge.io.results import Axis, PhasespaceSlice
from gammaforge.io.units import Quantity as Q


def _collimated():
    axes = {
        Axis.ENERGY: np.array([2.0, 3.0, 7.0]),
        Axis.THETA_X: np.array([-2.0, -0.25, 1.0]),
        Axis.THETA_Y: np.array([-1.5, 0.1, 3.0]),
    }
    return PhasespaceSlice(axes, np.arange(27.0).reshape(3, 3, 3) + 1.0)


def test_collimated_projections_preserve_nonuniform_measure():
    source = _collimated()
    projections = collimated_projections(source)

    for name in ("spectrum", "energy_theta_x", "energy_theta_y"):
        assert projections[name].integrate() == pytest.approx(source.integrate())

    expected = np.trapezoid(
        np.trapezoid(source.distr, source.axes[Axis.THETA_Y], axis=2),
        source.axes[Axis.THETA_X],
        axis=1,
    )
    np.testing.assert_allclose(project_slice(source, (Axis.ENERGY,)).distr, expected)


def test_display_density_jacobian_preserves_integral():
    source = PhasespaceSlice(
        {Axis.ENERGY: np.array([1.0, 2.0, 4.0])},
        np.array([3.0, 5.0, 7.0]),
    )
    scale = display_scale(Axis.ENERGY)
    display_integral = np.trapezoid(
        source.distr / scale,
        source.axes[Axis.ENERGY] * scale,
    )
    assert display_integral == pytest.approx(source.integrate())


def test_geometry_model_uses_the_laser_rotation():
    beam = GaussianElectronBeam(
        Q(100, "pC"),
        Q(100, "MeV"),
        0.01,
        Q(20, "um"),
        Q(30, "um"),
        Q(1e-7, "cm * rad"),
        Q(2e-7, "cm * rad"),
        Q(100, "um"),
    )
    laser = GaussianParaxialLaser(
        Q(1, "J"),
        Q(800, "nm"),
        Q(10, "um"),
        Q(15, "um"),
        duration=Q(30, "fs"),
        theta_xz=Q(0.3, "rad"),
        theta_yz=Q(-0.2, "rad"),
        psi_focus=Q(0.4, "rad"),
    )
    model = geometry_model(beam, laser)
    expected = laser.focusing_axes()

    np.testing.assert_allclose(model["k_hat"], expected[0])
    np.testing.assert_allclose(model["focus_axes"][0], expected[1])

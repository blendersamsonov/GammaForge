"""Unit and convention conversions at the dimensioned engine boundary."""

import math

import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.io.schema import FieldKind, FieldSpec, Parameters
from gammaforge.io.units import TimeConvention, WidthConvention
from gammaforge.engines.xigma.schema import default_parameters as xigma_parameters


SIGMA_X = FieldSpec(
    key="sigma_x",
    label="Laser spot size (x)",
    kind=FieldKind.WIDTH,
    unit="cm",
    default=1e-3,
    display_units=("um", "mm", "cm"),
    convention=WidthConvention.SIGMA_INTENSITY_RMS,
    value_range=(0.0, math.inf),
)
DURATION = FieldSpec(
    key="duration",
    label="Pulse duration",
    kind=FieldKind.DURATION,
    unit="s",
    default=3e-14,
    display_units=("fs", "ps", "s"),
    convention=TimeConvention.SIGMA_INTENSITY_RMS,
)


def test_units_and_width_conventions_convert_to_canonical_cgs():
    assert SIGMA_X.to_core(10.0, "um") == pytest.approx(1e-3)
    assert DURATION.to_core(30.0, "fs") == pytest.approx(3e-14)

    rms = SIGMA_X.to_core(10.0, "um", WidthConvention.SIGMA_INTENSITY_RMS)
    fwhm = SIGMA_X.to_core(10.0, "um", WidthConvention.FWHM_INTENSITY)
    assert fwhm == pytest.approx(rms / (2.0 * math.sqrt(2.0 * math.log(2.0))))


def test_display_conversions_round_trip_without_changing_physics():
    for unit, convention in (
        ("um", None),
        ("mm", WidthConvention.FWHM_INTENSITY),
        ("cm", WidthConvention.W0_1E2),
    ):
        shown = SIGMA_X.to_display(1e-3, unit, convention)
        assert SIGMA_X.to_core(shown, unit, convention) == pytest.approx(1e-3)

    params = Parameters.from_specs((SIGMA_X,)).set_display(
        "sigma_x", 25.0, "um", WidthConvention.FWHM_INTENSITY
    )
    assert params.display(
        "sigma_x", "um", WidthConvention.FWHM_INTENSITY
    ) == pytest.approx(25.0)


def test_duration_can_be_entered_as_a_length_via_light_time():
    assert DURATION.to_core(3e-4, "cm") == pytest.approx(3e-4 / 29979245800.0)


def test_xigma_reduced_dimension_axes_accept_one_bin():
    params = xigma_parameters().with_values(
        n_bins_a0_shape=1,
        n_bins_chirp=1,
        n_bins_ahat=1,
    )
    assert params.get_int("n_bins_a0_shape") == 1
    assert params.get_int("n_bins_chirp") == 1
    assert params.get_int("n_bins_ahat") == 1

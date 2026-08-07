"""Parameter-schema validation and boundary conversion (GRAND_PLAN.md §3.1, Phase 1 exit)."""

from __future__ import annotations

import math

import pytest

from gammaforge.io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters, SchemaError
from gammaforge.io.units import TimeConvention, WidthConvention

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
N_STEPS = FieldSpec(
    key="n_steps",
    label="Trajectory steps",
    kind=FieldKind.SCALAR,
    unit=DIMENSIONLESS,
    default=64,
    value_range=(1, 1e6),
    integer=True,
)
DEVICE = FieldSpec(
    key="device",
    label="Backend",
    kind=FieldKind.CHOICE,
    unit=DIMENSIONLESS,
    default="numpy",
    choices=("numpy", "cupy", "numba"),
)

SPECS = (SIGMA_X, DURATION, N_STEPS, DEVICE)


# -- FieldSpec declaration ---------------------------------------------------
def test_width_field_requires_a_width_convention():
    with pytest.raises(SchemaError, match="require a WidthConvention"):
        FieldSpec(key="w", label="w", kind=FieldKind.WIDTH, unit="cm", default=1.0)
    with pytest.raises(SchemaError, match="require a WidthConvention"):
        FieldSpec(
            key="w", label="w", kind=FieldKind.WIDTH, unit="cm", default=1.0,
            convention=TimeConvention.SIGMA_INTENSITY_RMS,
        )


def test_scalar_field_rejects_a_convention():
    # P1/§2.1: unambiguous fields have no convention slot at all -- no NoConvention.
    with pytest.raises(SchemaError, match="carry no convention"):
        FieldSpec(
            key="a0", label="a0", kind=FieldKind.SCALAR, unit=DIMENSIONLESS, default=0.1,
            convention=WidthConvention.SIGMA_INTENSITY_RMS,
        )


def test_choice_field_needs_choices_and_rejects_numeric_constraints():
    with pytest.raises(SchemaError, match="non-empty `choices`"):
        FieldSpec(key="d", label="d", kind=FieldKind.CHOICE, unit=DIMENSIONLESS, default="a")
    with pytest.raises(SchemaError, match="numeric constraints"):
        FieldSpec(
            key="d", label="d", kind=FieldKind.CHOICE, unit=DIMENSIONLESS, default="a",
            choices=("a", "b"), integer=True,
        )


def test_unparseable_or_incompatible_units_fail_at_declaration():
    with pytest.raises(SchemaError, match="cannot convert"):
        FieldSpec(key="s", label="s", kind=FieldKind.SCALAR, unit="cm", default=1.0,
                  display_units=("not_a_unit",))
    with pytest.raises(SchemaError, match="cannot convert"):
        FieldSpec(key="s", label="s", kind=FieldKind.SCALAR, unit="cm", default=1.0,
                  display_units=("erg",))


def test_default_is_validated_at_declaration():
    with pytest.raises(SchemaError, match="outside allowed range"):
        FieldSpec(key="s", label="s", kind=FieldKind.SCALAR, unit=DIMENSIONLESS,
                  default=-1.0, value_range=(0.0, 1.0))
    with pytest.raises(SchemaError, match="must be integral"):
        FieldSpec(key="n", label="n", kind=FieldKind.SCALAR, unit=DIMENSIONLESS,
                  default=1.5, integer=True)


# -- boundary conversion -----------------------------------------------------
def test_unit_conversion_into_the_core():
    assert SIGMA_X.to_core(10.0, "um") == pytest.approx(1e-3)
    assert DURATION.to_core(30.0, "fs") == pytest.approx(3e-14)


def test_convention_reinterpretation_is_independent_of_the_unit():
    # 10 um FWHM is a smaller RMS sigma than 10 um already-RMS.
    as_rms = SIGMA_X.to_core(10.0, "um", WidthConvention.SIGMA_INTENSITY_RMS)
    as_fwhm = SIGMA_X.to_core(10.0, "um", WidthConvention.FWHM_INTENSITY)
    assert as_fwhm < as_rms
    assert as_fwhm == pytest.approx(as_rms / (2.0 * math.sqrt(2.0 * math.log(2.0))))


def test_display_round_trip():
    for unit, convention in [("um", None), ("mm", WidthConvention.FWHM_INTENSITY),
                             ("cm", WidthConvention.W0_1E2)]:
        shown = SIGMA_X.to_display(1e-3, unit, convention)
        assert SIGMA_X.to_core(shown, unit, convention) == pytest.approx(1e-3)


def test_duration_can_be_entered_as_a_length_via_light_time():
    # The pulse-duration <-> pulse-length pairing the schema needs (GRAND_PLAN.md §2.1).
    assert DURATION.to_core(3e-4, "cm") == pytest.approx(3e-4 / 29979245800.0)


def test_choice_field_has_no_unit_conversion():
    with pytest.raises(SchemaError, match="no unit conversion"):
        DEVICE.to_core(1.0, "cm")


# -- Parameters --------------------------------------------------------------
def test_defaults_are_used_for_unspecified_fields():
    params = Parameters.from_specs(SPECS)
    assert params["sigma_x"] == pytest.approx(1e-3)
    assert params.get_choice("device") == "numpy"
    assert params.get_int("n_steps") == 64


def test_values_are_validated_on_construction_and_override():
    with pytest.raises(SchemaError, match="not one of"):
        Parameters.from_specs(SPECS, device="opencl")
    with pytest.raises(SchemaError, match="unknown parameter keys"):
        Parameters.from_specs(SPECS, nonexistent=1.0)
    with pytest.raises(SchemaError, match="must be integral"):
        Parameters.from_specs(SPECS).with_values(n_steps=12.5)


def test_parameters_are_immutable():
    params = Parameters.from_specs(SPECS)
    with pytest.raises(TypeError):
        params.values["sigma_x"] = 1.0  # type: ignore[index]
    updated = params.with_values(sigma_x=2e-3)
    assert params["sigma_x"] == pytest.approx(1e-3)
    assert updated["sigma_x"] == pytest.approx(2e-3)


def test_set_display_and_display_round_trip_through_parameters():
    params = Parameters.from_specs(SPECS).set_display("sigma_x", 25.0, "um", WidthConvention.FWHM_INTENSITY)
    assert params.display("sigma_x", "um", WidthConvention.FWHM_INTENSITY) == pytest.approx(25.0)
    assert params["sigma_x"] < 25e-4  # stored as an RMS sigma in cm, narrower than the FWHM


def test_duplicate_keys_are_rejected():
    with pytest.raises(SchemaError, match="duplicate field key"):
        Parameters.from_specs((SIGMA_X, SIGMA_X))


def test_accessors_reject_type_confusion():
    params = Parameters.from_specs(SPECS)
    with pytest.raises(SchemaError, match="CHOICE field, not numeric"):
        params.get_float("device")
    with pytest.raises(SchemaError, match="not a CHOICE field"):
        params.get_choice("n_steps")
    with pytest.raises(SchemaError, match="no such parameter"):
        params["nope"]

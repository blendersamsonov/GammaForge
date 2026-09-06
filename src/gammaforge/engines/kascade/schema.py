"""Typed numeric knobs for the minimal kascade Monte-Carlo port (P11)."""

from __future__ import annotations

from ...io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters

__all__ = ["KASCADE_SPECS", "default_parameters"]


KASCADE_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="quantum",
        label="Cross section",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="thomson",
        choices=("thomson", "klein-nishina"),
    ),
    FieldSpec(
        key="n_time",
        label="Optical-depth time samples",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=201,
        integer=True,
        value_range=(3, 100_001),
    ),
    FieldSpec(
        key="threshold",
        label="Active-region threshold",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=1e-6,
        value_range=(1e-12, 1.0),
    ),
    FieldSpec(
        key="max_photons",
        label="Maximum emissions per electron",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=30,
        integer=True,
        value_range=(1, 10_000),
    ),
    FieldSpec(
        key="chunk",
        label="Electrons per chunk",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=5_000,
        integer=True,
        value_range=(1, 10_000_000),
    ),
)


def default_parameters() -> Parameters:
    return Parameters.from_specs(KASCADE_SPECS)

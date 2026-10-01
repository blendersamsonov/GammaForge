"""xigma's own numeric knobs, as a typed `Parameters` schema.

Everything here is a **numerics** field in §5's sense; sampler controls affect only Stage 2.
The public engine still defaults to `FULL_RERUN` because cross-run reuse is not implemented.
What *is* cheap for
xigma — bunch charge, the collimation window — lives in `InteractionParameters`/`Target`,
not here (P9: this schema does not duplicate fields another module already owns).
"""

from __future__ import annotations

from ...io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters

__all__ = ["XIGMA_SPECS", "default_parameters"]

XIGMA_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="n_steps",
        label="Trajectory steps",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=200,
        integer=True,
        value_range=(1, 100_000),
    ),
    FieldSpec(
        key="gaussian_order",
        label="Gaussian temporal quadrature order",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=24,
        integer=True,
        value_range=(1, 256),
    ),
    FieldSpec(
        key="stage0_quadrature",
        label="Stage-0 quadrature",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        # Midpoint remains the production default (RES096).
        default="midpoint",
        choices=("midpoint", "auto"),
    ),
    FieldSpec(
        key="discard_tolerance",
        label="Gaussian Stage-0 discard fraction",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=0.0,
        value_range=(0.0, 0.999),
    ),
    FieldSpec(
        key="threshold",
        label="Active-region threshold",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=1e-3,
        value_range=(1e-12, 1.0),
    ),
    FieldSpec(
        key="scheme",
        label="Deposition scheme",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="nearest",
        choices=("nearest", "cic"),
    ),
    FieldSpec(
        key="n_bins_gamma",
        label="Table bins: gamma",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=48,
        integer=True,
        value_range=(2, 2048),
    ),
    FieldSpec(
        key="n_bins_theta_x",
        label="Table bins: theta_x",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=48,
        integer=True,
        value_range=(2, 2048),
    ),
    FieldSpec(
        key="n_bins_theta_y",
        label="Table bins: theta_y",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=48,
        integer=True,
        value_range=(2, 2048),
    ),
    FieldSpec(
        key="n_bins_a0_shape",
        label="Shape table bins: a0_shape",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=96,
        integer=True,
        value_range=(1, 512),
    ),
    FieldSpec(
        key="n_bins_chirp",
        label="Shape table bins: carrier rate",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=8,
        integer=True,
        value_range=(1, 512),
    ),
    FieldSpec(
        key="n_bins_ahat",
        label="Retarget grid bins: ahat",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=256,
        integer=True,
        value_range=(1, 512),
    ),
    FieldSpec(
        key="ahat_min",
        label="Retarget grid: ahat floor (everything below folds here)",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=0.0,
        value_range=(0.0, 10.0),
    ),
    FieldSpec(
        key="ahat_max",
        label="Retarget grid: ahat ceiling",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=0.5,
        value_range=(1e-6, 100.0),
    ),
    FieldSpec(
        key="line_model",
        label="Single-electron line model",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="moment2",
        choices=("delta", "moment2"),
    ),
    FieldSpec(
        key="backend",
        label="Compute backend",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="cupy",
        choices=("auto", "cupy", "numpy"),
    ),
    FieldSpec(
        key="sampler_rings",
        label="CuPy sampler rings",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=32,
        integer=True,
        value_range=(8, 64),
    ),
    FieldSpec(
        key="sampler_subsampling",
        label="CuPy sampler subsampling",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=32,
        integer=True,
        value_range=(1, 16_777_215),
    ),
)


def default_parameters() -> Parameters:
    """xigma's published ``schema`` — every field at its default (P5)."""
    return Parameters.from_specs(XIGMA_SPECS)

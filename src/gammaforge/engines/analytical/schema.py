"""analytical's own numeric knob, as a typed `Parameters` schema (GRAND_PLAN.md §3.1/§4.3).

Only `n_quad` lives here. Everything else the formulas need — the collimation half-angle,
beam, laser, `N_e` — comes through `InteractionParameters`/`Target`, which already own
that state (`gammaforge.io.target.Target.theta_x_col`/`theta_y_col`); duplicating it on
this schema is exactly what `xigma/schema.py`'s own docstring argues against (P9).
"""

from __future__ import annotations

from ...io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters

__all__ = ["ANALYTICAL_SPECS", "default_parameters"]

ANALYTICAL_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="n_quad",
        label="Energy-spread quadrature points",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=401,
        integer=True,
        value_range=(11, 100_001),
    ),
    # 1 = the fast 1D path (spot sizes sampled along z). >1 turns on the exact 2D
    # quadrature over (z, q1) that a crossing angle strictly requires — a deliberate
    # semi-analytical mode, ~40 ms at 51 nodes against ~2 ms for the 1D path, so it is
    # opt-in rather than the default (RES043). No effect head-on, where 1D is already exact.
    FieldSpec(
        key="n_quad_u",
        label="Exact-transverse quadrature nodes (1 = fast approximation)",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=1,
        integer=True,
        value_range=(1, 4001),
    ),
    # A separate knob from `n_quad`, deliberately: this is the longitudinal overlap
    # integral, whose integrand carries the four hourglass/Rayleigh scales and the focal
    # offsets, and it converges on its own terms. Sharing one field would tie the
    # spectrum's energy-spread quadrature to a count chosen for a different integral.
    FieldSpec(
        key="n_quad_overlap",
        label="Overlap-integral quadrature points",
        kind=FieldKind.SCALAR,
        unit=DIMENSIONLESS,
        default=2001,
        integer=True,
        value_range=(11, 1_000_001),
    ),
)


def default_parameters() -> Parameters:
    """analytical's published ``schema`` — its one field at its default (P5)."""
    return Parameters.from_specs(ANALYTICAL_SPECS)

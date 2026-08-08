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
)


def default_parameters() -> Parameters:
    """analytical's published ``schema`` — its one field at its default (P5)."""
    return Parameters.from_specs(ANALYTICAL_SPECS)

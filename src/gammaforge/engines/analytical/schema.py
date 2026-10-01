"""analytical's own numeric knobs and model selection, as a typed `Parameters` schema.

Only quadrature counts and model selection live here. Everything else the formulas need —
the collimation half-angle, beam, laser, `N_e` — comes through
`InteractionParameters`/`Target`, which already own that state
(`gammaforge.io.target.Target.theta_x_col`/`theta_y_col`); duplicating it on this schema is
exactly what `xigma/schema.py`'s own docstring argues against (P9).

`model_mode` and `model_pin` are `Parameters` fields rather than an
`AnalyticalEngine(mode=...)` constructor argument on purpose: the `Engine` protocol is
`schema` + `run`, so run-time controls on the schema are what keeps the GUI's schema-driven
inputs, request persistence, and request replay working (RES018). Both are `CHOICE` fields
for the same reason `xigma/schema.py` uses them — the set of valid strings is closed and
should be validated, not free text.

`model_pin` deliberately lists `"auto"` alongside the model names. "Not pinned" is a real,
frequently wanted state, and modelling it as the absence of a value would make the replayed
request differ from the submitted one.
"""

from __future__ import annotations

from ...io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters
from .models import MODEL_MODES

__all__ = ["ANALYTICAL_SPECS", "default_parameters"]

#: Expert pinning is offered for every registered model plus `"auto"`. Kept as a plain tuple
#: rather than imported from `models` so the schema (what users see) and the registry (what
#: exists) cannot silently disagree: `test_analytical.py` asserts they match.
_PIN_CHOICES: tuple[str, ...] = ("auto", "overlap_der001_mean_ahat")

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
    # Which model the planner picks *per observable* (DER019 §14, §17). `auto` is the
    # cheapest model judged reliable for each requested output; `reference` is the highest
    # fidelity deterministic model the geometry supports. This is a fidelity request, not a
    # performance request — the GUI needs no engine-specific branch to offer it.
    FieldSpec(
        key="model_mode",
        label="Model fidelity",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="auto",
        choices=MODEL_MODES,
    ),
    # Expert pin, for controlled comparisons and validation runs. `"auto"` is a real choice
    # rather than an empty value so a submitted request replays identically.
    FieldSpec(
        key="model_pin",
        label="Model (expert)",
        kind=FieldKind.CHOICE,
        unit=DIMENSIONLESS,
        default="auto",
        choices=_PIN_CHOICES,
    ),
)


def default_parameters() -> Parameters:
    """analytical's published ``schema`` — its one field at its default (P5)."""
    return Parameters.from_specs(ANALYTICAL_SPECS)

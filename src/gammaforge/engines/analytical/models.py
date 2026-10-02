"""Per-observable model selection for `AnalyticalEngine` (DER019 §14, §16, §17).

DER019 derives a *hierarchy* of deterministic Gaussian source models: a closed round
reduction, low-dimensional fixed-width tiers, and focused/flying-focus source maps that
retain the real paraxial spot evolution. Exposing that hierarchy as a flat list of user
choices would push internal model complexity onto ordinary users and break the promise that
`mode="auto"` is a sensible default, so selection lives here instead: this module names each
model, says which outputs it can produce, when it applies, what it costs, and picks one
**per requested observable** (DER019 §17).

Three deliberate properties, each from a requirement rather than taste:

* **Selection is per observable, not per run.** A 0D `TOTAL_YIELD` and a 1D `SPECTRUM` can
  legitimately be served by different tiers; one global approximation for a whole run is the
  thing the planner exists to prevent.
* **A rejected cheaper model records why.** Silent fallback outside a model's assumptions is
  the failure mode this repo rejects repeatedly, so a promotion from `auto` is always
  reported (`ModelChoice.rejected`).
* **Nothing here invents a validity threshold.** `auto` accepts only models whose acceptance
  policy says they are exact for the geometry at hand. Approximate tiers stay pinned-only
  until validation evidence exists (DER019 §16, §17.8), so this module's `auto` path cannot
  quietly promote a guessed constant into physics.

This module deliberately contains **no physics**. Each model's executor is supplied by
`engine.py`; the planner only ranks and reports. That split is what lets the planner be
exercised and proven inert before any formula changes (see the "planner selects the existing
model" regression in `tests/test_analytical.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from ...io.target import OutputKind

__all__ = [
    "MODEL_MODES",
    "AnalyticalModel",
    "ModelChoice",
    "ModelInputs",
    "ModelRejection",
    "ModelSelector",
]

#: Public model-selection semantics, in increasing fidelity. These are the values of the
#: `model_mode` schema field; the ordering here is the one `reference` walks.
MODEL_MODES: tuple[str, ...] = ("auto", "fast", "reference")


@dataclass(frozen=True)
class ModelInputs:
    """The geometry facts a model may condition its applicability on.

    Passed whole to each :attr:`AnalyticalModel.applies` predicate rather than letting models
    reach back into `InteractionParameters`: a model that silently reads state the planner
    never showed it is exactly the invisible-coupling failure the provenance record is meant
    to rule out.
    """

    beam: object
    laser: object
    #: Emitted by the caller for ``TOTAL_YIELD``, which has no axes of its own.
    photon_energy: float
    #: Crossing angle, radians. ``0.0`` head-on.
    crossing_angle: float = 0.0
    #: Flying-focus strength from `GaussianParaxialLaser.beta_ff`; ``0.0`` means a
    #: stationary focus. A non-zero value disqualifies the fixed-width tiers outright
    #: (DER019 §16: the 1D frozen-width path is unsafe for a flying focus).
    beta_ff: float = 0.0


@dataclass(frozen=True)
class ModelRejection:
    """Why one candidate model was not selected — recorded rather than discarded.

    ``structurally_inapplicable`` distinguishes "this geometry is outside the model" from
    "the model is fine here but its validity has not been established", which are different
    statements about the user's setup and only one of which is actionable by changing inputs.
    """

    name: str
    reason: str
    structurally_inapplicable: bool = False


@dataclass(frozen=True)
class AnalyticalModel:
    """One entry in the hierarchy: what it can do, when, and at what price.

    ``cost_rank`` orders candidates cheapest-first and is a *relative* integer within one
    observable, not a wall-clock estimate — the real relative cost of the higher tiers is an
    open numerical question (DER019 §17.8) and a made-up seconds-per-run figure would be a
    guess wearing a measurement's clothes.

    ``exact`` means the model reproduces its target observable exactly **under its own stated
    assumptions**, not that it is assumption-free: the exact focused tiers still assume an
    unchirped carrier. ``acceptance`` is the predicate that decides whether `auto` may use the
    model; when it is ``None`` the model is pinned-only.
    """

    name: str
    outputs: tuple[OutputKind, ...]
    applies: Callable[[ModelInputs], bool]
    #: Accepting predicate for automatic selection, or ``None`` for pinned-only.
    acceptance: Callable[[ModelInputs], bool] | None
    #: Higher is more faithful, lower is cheaper; compared against the mode's preference.
    fidelity_rank: int
    #: Lower is cheaper. Compared within one observable.
    cost_rank: int
    exact: bool
    #: Human-readable assumptions, surfaced verbatim in the provenance record.
    assumptions: tuple[str, ...]
    #: Outer deterministic quadrature dimension (0 = closed form).
    outer_dimension: int = 0
    #: Whether evaluating this model runs a per-node trajectory integration.
    trajectory_quadrature: bool = False


@dataclass(frozen=True)
class ModelChoice:
    """The planner's decision for one observable, in serializable form.

    This is what `AnalyticalEngine` writes into `Results.model_specific` under
    ``models`` (RES063: result metadata is persisted, so every field here is plain data —
    no dataclasses, enums, or numpy scalars that ``_encode_metadata`` cannot round-trip).
    """

    observable: str
    model: str
    requested_mode: str
    exact: bool
    assumptions: tuple[str, ...]
    outer_dimension: int
    trajectory_quadrature: bool
    rejected: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def as_metadata(self) -> dict:
        """A ``model_specific``-safe mapping (JSON-serializable under RES063)."""
        return {
            "observable": self.observable,
            "model": self.model,
            "requested_mode": self.requested_mode,
            "exact": self.exact,
            "assumptions": list(self.assumptions),
            "outer_dimension": int(self.outer_dimension),
            "trajectory_quadrature": bool(self.trajectory_quadrature),
            "rejected": [dict(item) for item in self.rejected],
        }


class ModelSelector:
    """Picks one model per observable from the registered hierarchy.

    Registered models are supplied by `engine.py` rather than enumerated here, so this module
    stays a planner and never becomes a second source of truth about which models exist.
    """

    def __init__(self, models: tuple[AnalyticalModel, ...]):
        self._models = tuple(models)

    @property
    def models(self) -> tuple[AnalyticalModel, ...]:
        return self._models

    def names(self) -> tuple[str, ...]:
        """Every registered model name, for the expert-pin schema field."""
        return tuple(model.name for model in self._models)

    def select(
        self, inputs: ModelInputs, kind: OutputKind, mode: str = "auto", pin: str | None = None
    ) -> ModelChoice:
        """Choose a model for ``kind``.

        ``pin`` bypasses automatic choice but still rejects structural inapplicability: an
        expert asking for a focused model in a geometry that cannot support it gets an
        explicit error naming the reason, never a quiet substitution (DER019 §17.8).
        """
        if mode not in MODEL_MODES:
            raise ValueError(f"model_mode must be one of {MODEL_MODES}, got {mode!r}")

        candidates = [model for model in self._models if kind in model.outputs]
        if not candidates:
            supported = ", ".join(sorted({k.value for m in self._models for k in m.outputs}))
            raise ValueError(
                f"no analytical model produces {kind.value!r} "
                f"(models available here: {sorted(m.name for m in self._models)}; "
                f"outputs covered: {supported})"
            )

        rejected: list[ModelRejection] = []

        if pin is not None:
            pinned = next((m for m in self._models if m.name == pin), None)
            if pinned is None:
                raise ValueError(f"unknown analytical model {pin!r}; known models: {list(self.names())}")
            if kind not in pinned.outputs:
                raise ValueError(
                    f"analytical model {pin!r} does not produce {kind.value!r} "
                    f"(it produces: {[k.value for k in pinned.outputs]})"
                )
            if not pinned.applies(inputs):
                raise ValueError(
                    f"analytical model {pin!r} is structurally invalid for this geometry: "
                    f"{_applicability_note(pinned, inputs)} Pinning cannot change applicability — "
                    "it bypasses cost and validity ranking, not the model's own assumptions."
                )
            for model in candidates:
                if model.name != pin:
                    rejected.append(
                        ModelRejection(
                            model.name,
                            f"expert pin {pin!r} selected {pinned.name!r} instead",
                            structurally_inapplicable=False,
                        )
                    )
            return _choice(kind, pinned, mode, rejected)

        applicable: list[AnalyticalModel] = []
        for model in candidates:
            if not model.applies(inputs):
                rejected.append(
                    ModelRejection(model.name, _applicability_note(model, inputs), True)
                )
                continue
            if model.acceptance is None:
                rejected.append(
                    ModelRejection(
                        model.name,
                        "approximate model; automatic acceptance needs validation evidence, so it "
                        "is pinned-only (DER019 §17.8)",
                        structurally_inapplicable=False,
                    )
                )
                continue
            if not model.acceptance(inputs):
                rejected.append(
                    ModelRejection(
                        model.name, "validity diagnostic outside the accepted range", False
                    )
                )
                continue
            applicable.append(model)

        if not applicable:
            raise ValueError(
                f"no accepted analytical model for {kind.value!r}. Rejected: "
                + "; ".join(f"{r.name}: {r.reason}" for r in rejected)
                + ". Pin a model explicitly with model_pin to run outside automatic selection."
            )

        selected = _pick(applicable, mode)
        for model in applicable:
            if model.name != selected.name:
                rejected.append(
                    ModelRejection(model.name, f"higher cost than {selected.name!r} in mode {mode!r}")
                )
        rejected.sort(key=lambda item: item.name)
        return _choice(kind, selected, mode, rejected)


def _pick(candidates: list[AnalyticalModel], mode: str) -> AnalyticalModel:
    """The cheapest accepted model for ``auto``/``fast``, the most faithful for ``reference``."""
    if mode == "reference":
        return max(candidates, key=lambda m: (m.fidelity_rank, -m.cost_rank))
    return min(candidates, key=lambda m: (m.cost_rank, -m.fidelity_rank))


def _choice(
    kind: OutputKind, model: AnalyticalModel, mode: str, rejected: list[ModelRejection]
) -> ModelChoice:
    rejected.sort(key=lambda item: item.name)
    return ModelChoice(
        observable=kind.value,
        model=model.name,
        requested_mode=mode,
        exact=model.exact,
        assumptions=model.assumptions,
        outer_dimension=model.outer_dimension,
        trajectory_quadrature=model.trajectory_quadrature,
        rejected=tuple(
            {"model": r.name, "reason": r.reason, "structural": r.structurally_inapplicable}
            for r in rejected
        ),
    )


def _applicability_note(model: AnalyticalModel, inputs: ModelInputs) -> str:
    """A concrete reason for structural inapplicability, not a restatement of the verdict."""
    if inputs.beta_ff != 0.0 and model.outer_dimension <= 1:
        return (
            f"a flying focus (beta_ff={inputs.beta_ff:g}) makes the spot size time-dependent, "
            "which the 1D frozen-width reduction cannot represent (DER019 §16)"
        )
    return "this geometry is outside the model's stated domain"

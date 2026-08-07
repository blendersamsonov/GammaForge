"""The uniform engine interface (GRAND_PLAN.md §4.1) and the recompute-cost tiers (§5).

Two declarations, both small, both needed before any engine exists: the validation
harness of §7 runs *an engine* against a scenario, and it cannot express that without a
name for what an engine is. Everything an engine needs is already in `gammaforge.io`;
this module adds only the calling convention.

**Nothing here is a registry.** `ENGINES` (§4.1) arrives with the first engine that has
something to register — a lazy import table over an empty set of engines would be exactly
the speculative machinery P10 rejects.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from ..io.interaction import InteractionParameters
from ..io.results import Results
from ..io.schema import Parameters
from ..io.target import OutputKind

__all__ = ["RecomputeCost", "Engine"]


class RecomputeCost(Enum):
    """How expensive re-Calculating is after one field changes (§5).

    Generic and **engine-declared**: the predecessor's global ``ParamGroup`` enum was tied
    to xigma's three stages and had no meaning for a single-stage Monte Carlo. Each engine
    publishes its own ``field key -> RecomputeCost`` mapping instead.

    The tiers describe how cheap a re-Calculate is — **not** how live a field is. Nothing
    but the analytical estimates panel is real-time (§5), so even `QUERY_ONLY` still waits
    for Calculate. The one exception is bunch charge, whose linearity lets the GUI rescale
    existing `Results` with no engine call at all.
    """

    QUERY_ONLY = "query_only"
    REUSE_INTERMEDIATES = "reuse_intermediates"
    FULL_RERUN = "full_rerun"


@runtime_checkable
class Engine(Protocol):
    """What every engine exposes, and all the harness or the GUI may rely on.

    Deliberately four attributes and one method:

    * ``name`` — the key it is reported and cached under.
    * ``schema`` — its numeric knobs as a validated `Parameters` (P5). No mutable
      ``Config`` object, no ``model_params()`` triples, no adapter attributes.
    * ``supported_outputs`` — plain declarative data in the `OutputKind` vocabulary
      (P10), never a capability registry or a negotiation protocol.
    * ``recompute_costs`` — field key to `RecomputeCost`, driving grey-out and cheap
      requery (§5). A field absent from the mapping is `FULL_RERUN` by default, so an
      engine only declares what it can genuinely do cheaply.
    * ``run`` — the whole calling convention: an interaction plus its own parameters in,
      `Results` out. Outputs it cannot produce are simply omitted.

    `runtime_checkable` makes ``isinstance`` check attribute *presence* only, which is
    what the harness wants: a helpful error naming the missing attribute, rather than an
    `AttributeError` from somewhere deep in a run.
    """

    name: str
    schema: Parameters
    supported_outputs: tuple[OutputKind, ...]
    recompute_costs: dict[str, RecomputeCost]

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results: ...

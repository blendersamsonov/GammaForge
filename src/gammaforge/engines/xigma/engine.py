"""`XigmaEngine`: the thin `Engine` wrapper the GUI and validation harness call (§4.2).

Opaque by contract (P3) — it builds one `Collision` per `run()` and returns `Results`.
Nothing here is stateful across calls; `Collision`'s own memoization (`collision.py`)
is what makes a `run()` requesting several outputs share Stage 0/1 work.
"""

from __future__ import annotations

from ...io.interaction import InteractionParameters
from ...io.results import Results
from ...io.schema import Parameters
from ...io.target import OutputKind
from ..base import RecomputeCost
from .collision import Collision
from .schema import default_parameters

__all__ = ["XigmaEngine"]

#: Matches `Collision`'s `_SUPPORTED` set (§4.2) — the outputs this engine can fill today.
SUPPORTED_OUTPUTS: tuple[OutputKind, ...] = (
    OutputKind.TOTAL_YIELD,
    OutputKind.SPECTRUM,
    OutputKind.ANGULAR_DISTRIBUTION,
    OutputKind.COLLIMATED_SPECTRUM,
)

#: §5's mapping, restricted to what is honestly wired **today**. Bunch charge is exactly
#: linear in N_e (§3.5) for every engine and is handled at the `io` level
#: (`InteractionParameters.with_charge`/`Results.scaled`) without an engine run at all, so
#: it costs nothing regardless of which engine is active. Everything else defaults to
#: `FULL_RERUN` (base.py) — including pulse energy, not claimed cheap despite being
#: analytically a pure rescale (D030).
RECOMPUTE_COSTS: dict[str, RecomputeCost] = {
    "n_e": RecomputeCost.QUERY_ONLY,
}


class XigmaEngine:
    """The tabulated-overlap engine, behind the uniform `Engine` protocol (`base.py`)."""

    name = "xigma"
    schema: Parameters = default_parameters()
    supported_outputs: tuple[OutputKind, ...] = SUPPORTED_OUTPUTS
    recompute_costs: dict[str, RecomputeCost] = RECOMPUTE_COSTS

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        return Collision(interaction=interaction, params=params).run(interaction.target.outputs)

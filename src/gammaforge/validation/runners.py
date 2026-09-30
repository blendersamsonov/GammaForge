"""Running one engine on one scenario.

Thin on purpose. An engine already takes exactly what a scenario already holds, so a
runner has one job: sample the bunch once, hand the same `InteractionParameters` to the
engine, and label what came back. It does not add per-model configuration builders,
boundary dictionaries, or a commit-keyed result cache:

* every engine takes `InteractionParameters` (§4.1), so there is nothing per-model to build;
* `io` samples and prefilters (§3.5), so no engine has its own sampler to keep in step;
* results are not cached to disk. The engine facade caches its *own* intermediates by
  exact input hash (§5), which is both finer-grained and always valid. A second,
  coarser, commit-keyed cache on top would be the speculative machinery P6 warns about —
  reinstate it if and only if a real suite run is measured to be too slow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..engines.base import Engine
from ..io.interaction import InteractionParameters, SamplingSpec
from ..io.results import Results
from ..io.schema import Parameters
from .scenarios import Scenario, build

__all__ = ["Run", "run_engine", "run_bank"]


@dataclass(frozen=True)
class Run:
    """One engine's output plus the interaction it was actually given.

    The interaction is kept because most validation legs need it: the temporal autorange
    needs the bunch, an invariance check needs to know which bunch it compared, and a
    report that cannot say what was run is not a report.
    """

    engine: str
    scenario: str
    interaction: InteractionParameters
    results: Results


def run_engine(
    engine: Engine,
    scenario: Scenario,
    params: Parameters | None = None,
    sampling: SamplingSpec | None = None,
) -> Run:
    """Sample the scenario and run one engine against it.

    ``params`` defaults to the engine's own schema defaults — engine knobs belong to the
    engine (P5), which is why they are an argument here rather than a scenario field.
    """
    if not isinstance(engine, Engine):
        raise TypeError(
            f"{engine!r} does not satisfy the Engine protocol: it "
            f"needs name, schema, supported_outputs, recompute_costs and run()"
        )
    interaction = build(scenario, sampling)
    results = engine.run(interaction, params if params is not None else engine.schema)
    return Run(
        engine=engine.name,
        scenario=scenario.name,
        interaction=interaction,
        results=results,
    )


def run_bank(
    engine: Engine,
    scenarios: Iterable[Scenario],
    params: Parameters | None = None,
) -> list[Run]:
    return [run_engine(engine, scenario, params) for scenario in scenarios]

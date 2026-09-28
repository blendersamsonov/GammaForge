"""The compiled interaction every engine run consumes (GRAND_PLAN.md §3.5).

`InteractionParameters` is the single bundle an engine receives: an analytic beam
description, a laser typed against the **`LaserField` protocol** (never the concrete
`GaussianParaxialLaser` — that is what makes a future non-Gaussian laser a
zero-engine-change swap, P15), the already-sampled and already-prefiltered bunch, the
target, and the electron count.

**`io` samples and prefilters; engines consume.** Every engine in a run is given the
*same* bunch, which is why `SamplingSpec` lives at the interaction level rather than in
any engine's parameters — a per-engine ``n_mc`` would be meaningless.

**N_e is a separate scalar.** The bunch carries only relative weights (§3.2), and every
output is exactly linear in ``N_e`` — no space charge in the model. That is what lets a
charge edit rescale displayed results instantly (`QUERY_ONLY`, §5) instead of re-running
anything, and why the prefilter never touches ``N_e``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numbers
from typing import TYPE_CHECKING

from .bunch import Bunch, GaussianElectronBeam, prefilter_bunch, sample_gaussian_bunch
from .laser import LaserField
from .target import Target
from .units import Quantity

if TYPE_CHECKING:  # imported for the type only; the module is not needed to build a bunch
    from .adaptive_sampling import AdaptiveSamplingPlan

__all__ = [
    "SamplingSpec",
    "InteractionParameters",
    "build_interaction",
    "PREFILTER_OFF",
    "SAMPLING_STRATEGIES",
    "IID",
    "ADAPTIVE",
]

#: A prefilter threshold of zero keeps every particle — the "off" setting, expressed as a
#: value of the same numeric field rather than as a separate boolean, so the GUI renders
#: one control and the invariance test (§7) sweeps one axis.
PREFILTER_OFF = 0.0

#: The two ways a bunch can be drawn. `IID` is the historical sampler and stays the
#: default: the adaptive strategy is opt-in until its benchmarks justify making it the
#: default, and until then an unexpected regime must keep the behaviour users already rely
#: on. `ADAPTIVE` is the luminosity-aware stratifier (RES092) — the *same* Gaussian bunch,
#: represented with non-uniform weights, at a lower cost for the same Stage-1/Stage-2
#: accuracy. Both produce an ordinary `Bunch` with relative per-particle weights summing to
#: one, so no engine can tell them apart except by inspecting `sampling.strategy`.
IID = "iid"
ADAPTIVE = "adaptive"
SAMPLING_STRATEGIES = (IID, ADAPTIVE)


@dataclass(frozen=True)
class SamplingSpec:
    """How the bunch is drawn.

    ``seed`` is a **first-class, GUI-displayed and editable field** (§6), not incidental
    internal state: reproducibility is a user-facing guarantee.

    ``prefilter`` is the active-region threshold as a fraction of peak a0 (§3.2), with
    :data:`PREFILTER_OFF` disabling it. Because the prefilter is a pure optimization,
    changing it must not change results — a tested invariant, not an intention (§7).

    ``strategy`` selects between the IID sampler and the luminosity-aware adaptive
    stratifier. It is a level-level choice like ``prefilter``, not an engine knob: both
    strategies are interaction-level (§3.5) because every engine in a run is given the
    *same* bunch, and a per-engine sampling strategy would be meaningless for the same
    reason a per-engine ``n_mc`` is.

    The adaptive strategy's tuning constants are deliberately **not** here. They live in
    `gammaforge.io.adaptive_sampling.PilotConfig` as module defaults until a benchmark
    defends a different value; surfacing a dozen provisional numbers in the GUI would
    advertise a precision the numbers do not have (§22).
    """

    n_particles: int = 100_000
    seed: int = 0
    prefilter: float = 1e-3
    strategy: str = IID

    def __post_init__(self) -> None:
        if isinstance(self.n_particles, bool) or not isinstance(self.n_particles, numbers.Integral):
            raise ValueError(f"SamplingSpec: n_particles must be an integer, got {self.n_particles!r}")
        if self.n_particles < 1:
            raise ValueError(f"SamplingSpec: n_particles must be >= 1, got {self.n_particles}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise ValueError(f"SamplingSpec: seed must be an integer in [0, 2**31 - 1], got {self.seed!r}")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError(f"SamplingSpec: seed must be in [0, 2**31 - 1], got {self.seed}")
        if isinstance(self.prefilter, bool) or not isinstance(self.prefilter, numbers.Real) or not math.isfinite(float(self.prefilter)):
            raise ValueError(f"SamplingSpec: prefilter must be finite, got {self.prefilter!r}")
        if not 0.0 <= self.prefilter < 1.0:
            raise ValueError(
                f"SamplingSpec: prefilter must be in [0, 1) — a fraction of peak a0, with "
                f"0 meaning off — got {self.prefilter}"
            )
        if self.strategy not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"SamplingSpec: strategy must be one of {SAMPLING_STRATEGIES}, got {self.strategy!r}"
            )


@dataclass(frozen=True)
class InteractionParameters:
    """Everything an engine run needs, and nothing it does not (P9).

    Build it with :func:`build_interaction` rather than by hand: that is what guarantees
    the bunch actually corresponds to the beam, the sampling spec and the seed.
    """

    beam: GaussianElectronBeam
    laser: LaserField
    bunch: Bunch
    target: Target
    N_e: float
    sampling: SamplingSpec

    def __post_init__(self) -> None:
        if isinstance(self.N_e, bool) or not isinstance(self.N_e, numbers.Real) or not math.isfinite(float(self.N_e)):
            raise ValueError(f"InteractionParameters: N_e must be a finite scalar, got {self.N_e!r}")
        if self.N_e <= 0:
            raise ValueError(f"InteractionParameters: N_e must be > 0, got {self.N_e}")

    def with_charge(self, bunch_charge: Quantity) -> "InteractionParameters":
        """A copy with a new bunch charge — the one edit that reuses the bunch verbatim.

        Charge sets ``N_e`` alone; it does not enter the sampled distribution, so the
        bunch is *not* resampled and existing results can simply be rescaled by the ratio
        of the two ``N_e`` values (§5). Contrast the resample rule below.
        """
        beam = replace(self.beam, bunch_charge=bunch_charge)
        return replace(self, beam=beam, N_e=beam.n_electrons())


def build_interaction(
    beam: GaussianElectronBeam,
    laser: LaserField,
    target: Target,
    sampling: SamplingSpec | None = None,
    *,
    plan: "AdaptiveSamplingPlan | None" = None,
) -> InteractionParameters:
    """Sample and prefilter the bunch, and bundle everything an engine needs.

    **Bunch resample rule (§3.5).** Calling this always draws a fresh bunch from the
    current ``seed``. So "same seed" means *"same seed + same beam/laser parameters + same
    n_particles ⇒ identical bunch"*, not "same bunch regardless of parameters" — any
    beam or laser physical edit is expected to come back through here. Target and charge
    edits must not: they do not affect the sampled distribution, and rebuilding for them
    would needlessly invalidate every engine cache keyed on the bunch. Use
    :meth:`InteractionParameters.with_charge` and `dataclasses.replace` for those.

    **The prefilter is applied identically to both strategies and renormalizes neither.**
    For the adaptive strategy that matters more than it looks: the regional weights
    ``P_m / n_m`` already sum to exactly one, and renormalizing after a discard would
    quietly convert "the fraction of the population this represents" into "one",
    inflating every result by the discarded fraction (§3.2, RES092).

    ``plan`` is the adaptive strategy's reuse hook. A plan depends only on
    ``(beam, laser, seed)``, so passing one back skips the pilot entirely — which is what
    makes a particle-count sweep affordable, and what lets a caller hold the plan that
    :func:`build_interaction` used. It is ignored by the IID strategy, which has no plan.
    """
    sampling = sampling or SamplingSpec()
    if sampling.strategy == ADAPTIVE:
        from .adaptive_sampling import build_adaptive_bunch

        bunch, _plan = build_adaptive_bunch(
            beam, laser, sampling.n_particles, sampling.seed, plan=plan
        )
    else:
        bunch = sample_gaussian_bunch(beam, sampling.n_particles, sampling.seed)
    if sampling.prefilter > PREFILTER_OFF:
        bunch = prefilter_bunch(bunch, laser, sampling.prefilter)
    return InteractionParameters(
        beam=beam,
        laser=laser,
        bunch=bunch,
        target=target,
        N_e=beam.n_electrons(),
        sampling=sampling,
    )

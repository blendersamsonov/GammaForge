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

from .bunch import Bunch, GaussianElectronBeam, prefilter_bunch, sample_gaussian_bunch
from .laser import LaserField
from .target import Target
from .units import Quantity

__all__ = ["SamplingSpec", "InteractionParameters", "build_interaction", "PREFILTER_OFF"]

#: A prefilter threshold of zero keeps every particle — the "off" setting, expressed as a
#: value of the same numeric field rather than as a separate boolean, so the GUI renders
#: one control and the invariance test (§7) sweeps one axis.
PREFILTER_OFF = 0.0


@dataclass(frozen=True)
class SamplingSpec:
    """How the bunch is drawn.

    ``seed`` is a **first-class, GUI-displayed and editable field** (§6), not incidental
    internal state: reproducibility is a user-facing guarantee.

    ``prefilter`` is the active-region threshold as a fraction of peak a0 (§3.2), with
    :data:`PREFILTER_OFF` disabling it. Because the prefilter is a pure optimization,
    changing it must not change results — a tested invariant, not an intention (§7).
    """

    n_particles: int = 100_000
    seed: int = 0
    prefilter: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_particles < 1:
            raise ValueError(f"SamplingSpec: n_particles must be >= 1, got {self.n_particles}")
        if not 0.0 <= self.prefilter < 1.0:
            raise ValueError(
                f"SamplingSpec: prefilter must be in [0, 1) — a fraction of peak a0, with "
                f"0 meaning off — got {self.prefilter}"
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
) -> InteractionParameters:
    """Sample and prefilter the bunch, and bundle everything an engine needs.

    **Bunch resample rule (§3.5).** Calling this always draws a fresh bunch from the
    current ``seed``. So "same seed" means *"same seed + same beam/laser parameters + same
    n_particles ⇒ identical bunch"*, not "same bunch regardless of parameters" — any
    beam or laser physical edit is expected to come back through here. Target and charge
    edits must not: they do not affect the sampled distribution, and rebuilding for them
    would needlessly invalidate every engine cache keyed on the bunch. Use
    :meth:`InteractionParameters.with_charge` and `dataclasses.replace` for those.
    """
    sampling = sampling or SamplingSpec()
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

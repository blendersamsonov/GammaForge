# RES051 — discarding charge is reported to the user, never decided silently

Status: implemented
Type: feature

## Problem

RES047's measurements make whether to discard low-illumination charge a scenario-dependent
choice — on a well-matched collision there is nothing to discard, and on a mismatched one
there is a lot — so the code cannot pick a default that is right in both, but a user still
needs a way to see how much is discardable before running anything expensive.

## Decision

`io.bunch.illumination_report` returns the fraction of **charge** below an illumination
threshold together with the bunch's Gaussian `ks_excess`, as a pre-engine diagnostic. No
filter is applied automatically anywhere.

## Alternatives considered

**Auto-select a filter from the report.** Changes results silently on a judgement the user
is better placed to make, and the plan already rejects that class of hidden behavior.

**Report the particle fraction.** Wrong for exactly the bunches — imported, non-uniformly
weighted — where the question is most likely to be asked.

**Fold it into `prefilter_by_illumination`.** A function that filters should filter; a
function that informs should inform.

## Rationale

It is not a *blind* choice: `io.bunch.peak_illumination` is one pass and its distribution
says exactly how much charge is discardable, before anything expensive runs. The two
numbers must be read together, which is why one function returns both. The illumination
estimate rests on a Gaussian picture of the collision, so a bunch that fits a Gaussian
badly is exactly the case where its *tails* — the charge a filter would drop — are least
well described. "10% of the charge is below threshold" is a different decision at a
`ks_excess` of 0.005 than at 0.2, and neither number alone says which. `ks_excess` is
`None` for an analytic beam, which is reported as `gaussian_by_construction` rather than as
missing data: that beam has no model error, which is a stronger statement than ignorance.

Charge rather than particle count, because `Bunch.weight` is relative and an imported
bunch need not be uniformly weighted — the two coincide only for a freshly sampled one.

## Consequences

This is `O(n_particles)` and that is correct: it inspects real macroparticles, which is the
only way to answer "how much of *this* bunch". The cost belongs to a diagnostic, not to any
engine's estimate path (RES049).

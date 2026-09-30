# RES048 — the illumination window is the primary result; the filter is a corollary of it

Status: implemented
Type: feature

## Problem

RES047's `peak_illumination` and a per-particle illuminated-time window come from the same
underlying quadratic, and computing only the peak-based filter leaves the more directly
useful half — the window an engine could sample against — on the floor.

## Decision

`io.bunch.illumination_window` returns per-particle `(t0, t1)` — when each macroparticle is
actually above the illumination threshold — and `io.bunch.prefilter_by_illumination` is now
defined as "keep particles whose window is non-empty", mirroring how `io.bunch.
prefilter_bunch` is defined through `io.bunch.overlap_time_window`.

## Alternatives considered

**Return only the half-width.** The window is not symmetric about `t = 0` — it is centred
on each particle's own closest approach `t_star`, which is the whole point.

**Have `prefilter_by_illumination` keep its own independent threshold test.** Two code
paths for one inequality, free to disagree.

**Widen `overlap_time_window` to serve both.** It is a bound with a tested invariance
depending on it; an estimate must not share its name.

## Rationale

Both come from one quadratic. With frozen widths the density along a straight trajectory
is `exp(-(a t^2 + 2 b t + c) / 2)` times a brightness factor, so `density >= threshold` is
a single inequality in `t` whose solution is `t_star +- sqrt(2 ln(peak / threshold) / a)`.
The filter asks whether that interval exists; the window is the interval.

**The window is the more valuable half.** An engine samples each trajectory with a *fixed*
number of steps between `t0` and `t1`, so the window's width sets the step size, and every
step spent where nothing happens is a step not spent resolving the interaction.
`overlap_time_window` brackets the particle's time inside the laser's geometric
`ActiveRegion` — a conservative bound, hence far wider than the illuminated stretch. On the
baseline scenario the illuminated window is 1.5x narrower and, with the same particles and
the same 33 steps, integrates 7x more accurately.

The tradeoff is favorable in a way worth stating, because it answers "how much margin":
width grows as `sqrt(ln(1 / threshold))` while the truncation floor tracks the threshold
roughly decade for decade. Nine decades of threshold — 1e-3 down to 1e-12 — cost about a
doubling of the window and buy ten orders of magnitude of floor (baseline: 101 ps / 9.2e-5
against 218 ps / 4.1e-15). Being generous is cheap; being stingy is not.

## Consequences

Two properties recorded because they are easy to get backwards. The window errs **wide**:
away from closest approach the true spot is larger than the frozen value, so the real
intensity falls faster than the model and the edges come out about 100x below threshold.
For a window that is the right way to be wrong — a too-narrow one truncates the interaction
and no step budget recovers it. And unlike `overlap_time_window` it is an *estimate*, not a
bound, which is why `prefilter_bunch`'s exact invariance still rests on the geometric one.

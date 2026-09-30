# RES050 — xigma can sample trajectories over the illuminated window, opt-in

Status: implemented
Type: feature

## Problem

Stage 0 spends a *fixed* `n_steps` between `t0` and `t1`, so the window's width sets the
step size and any step outside the illuminated stretch is wasted resolution. RES048 gives
`io.bunch.illumination_window`, a narrower bracket than the geometric `overlap_time_window`
Stage 0 uses today — whether to switch, and whether to make it the default, needs weighing.

## Decision

`engines.xigma.stages.integrate_trajectories` gains `window="active_region" |
"illumination"`, defaulting to `"active_region"` — the existing `io.bunch.
overlap_time_window` behavior, unchanged.

## Alternatives considered

**Switch the default.** Would change every golden reference and xigma result for a
scenario-dependent gain, while the parallel Phase 3b session is live in the same files.

**Choose the window automatically from the geometry.** Requires a reliable predictor of
which regime a scenario is in; the measurements below say that predictor is not obvious,
and an engine silently changing its integration bounds is the harder thing to debug.

**Expose it as a xigma `FieldSpec`.** Reasonable eventually, but it selects an algorithm
rather than setting a numeric knob, and §3.1's schema is for the latter.

## Rationale

`io.bunch.illumination_window` brackets the illuminated stretch itself, and on the baseline
scenario it converges 17x faster at 50 steps.

The default stays the wider window for two reasons, both measured rather than assumed. The
illuminated window truncates at its threshold, so it has an accuracy *floor* while the
geometric one keeps converging — past ~100 steps on the baseline the geometric window
wins. And the advantage is **scenario-dependent, not uniform**: on a tight-focus pulse the
geometric window is already better by 20 steps, because the spot varies so much across the
window that the frozen-width estimate mis-sizes it. A change that helps 17x in one place
and hurts in another does not belong in a default that golden references and a parallel
session both depend on.

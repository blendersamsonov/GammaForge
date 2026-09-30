# RES036 — `SPECTRUM`'s grid integral is rescaled to `estimate_yield`'s total by construction, not as a discrepancy patch

Status: implemented
Type: architecture

## Problem

`angle_integrated_spectrum`'s raw output integrates to `InteractionParameters.N_e`, not a
photon count — the per-gamma kinematic shape integrates to 1 over its kinematic domain and
the Gaussian energy PDF integrates to 1 over gamma, so the raw quadrature is "one
scattering attempt per electron." `estimate_yield` supplies the actual per-electron
scattering probability the kinematic shape has no way to know, and the two need to be
combined into `SPECTRUM` in a way that is exact rather than tolerance-matched — the
predecessor's own `# QUICK FIX, FLAGGED FOR FUTURE INVESTIGATION` comment was doing this
without naming it as the definition.

## Decision

`engines.analytical.engine.AnalyticalEngine._fill`'s `SPECTRUM` branch scales
`angle_integrated_spectrum`'s raw shape (evaluated at `InteractionParameters.N_e` = 1) by
`total_yield / raw_integral`, where `raw_integral` is the **discrete trapezoid integral
over the actual emitted energy grid** — not the shape's analytic infinite-domain value of 1
— so that `PhasespaceSlice.integrate()` reproduces `total_yield` to float precision.
`SPECTRUM` is therefore defined as `total_yield x (normalized shape)` — not two
independently-estimated quantities forced into agreement after the fact.

## Alternatives considered

**Leave `SPECTRUM` and `TOTAL_YIELD` as two independent estimates, tolerance-compared in
tests.** Matches §7's treatment of xigma/delta/kascade cross-validation, but analytical's
own two formulas do not claim to be independent measurements of the same thing — one is a
probability, the other a normalized shape of where that probability lands in energy — so
treating their disagreement as a tolerance to converge is a category error, not a
cross-validation.

**Normalize by the shape's analytic value of 1 instead of the discrete grid integral.**
Simpler, but leaves the identity only approximately true (to whatever the auto-range's
trapezoid discretization loses), which is exactly the exact-vs-tolerance distinction §7
draws.

## Rationale

§7 asks for `integral spectrum = total_yield` as an **exact identity**, not a tolerance —
normalizing against the grid's own discrete integral (rather than the analytic value) is
what makes it exact at any resolution, the same move RES026/§9.1 made for the kernel
normalization: fix it at the point it is produced, not with a permanent tolerance band.
`test_spectrum_grid_integral_correction_factor_is_near_one` (`tests/test_analytical.py`)
guards that the correction factor this applies stays close to 1 — i.e. the auto-derived
energy range is not silently masking real spectral weight by truncating the grid.

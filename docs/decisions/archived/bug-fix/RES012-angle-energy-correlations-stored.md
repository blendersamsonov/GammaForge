# RES012 — Angle-energy correlations are stored, so `drift` composes

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

RES005's original design stored only position-energy correlations and re-derived
`cov(x', gamma)` from `alpha` at each drift step, on the assumption that gamma couples to
the angle only through position. That assumption is true of a freshly sampled bunch and
destroyed by the first drift, and the resulting transport error was subtle enough to
survive the original test suite.

## Decision

`GaussianElectronBeam` carries `rho_thx_gamma` and `rho_thy_gamma` — the angle-energy
correlations, i.e. the dispersion derivative — alongside the position-energy ones.
`_drift_plane` transports `cov(x, gamma) -> cov(x, gamma) + L cov(x', gamma)` and leaves the
angle-energy correlation alone, since a drift changes neither the angle nor gamma.
`gamma_coefficients` holds the resulting sampler algebra and the joint admissibility
condition.

## Alternatives considered

**The original RES005 design, storing only the position-energy correlations and re-deriving
`cov(x', gamma)` from `alpha` at each drift step, on the model assumption that gamma
couples to the angle *only* through the position.** Rejected — see Rationale.

## Rationale

That assumption is true of a freshly sampled bunch and **destroyed by the first drift**.
The symptom was subtle enough to survive the original test suite: a single `drift` agreed
with a refit of the drifted macroparticles to sampling noise, so the transport looked
right, while two consecutive drifts silently disagreed with one drift of the combined
length (`rho_x_gamma` 0.238 vs 0.186 for 25 + 25 cm against 50 cm). It was found by asking
the transport to *compose* and to *reverse*, neither of which the earlier test did.

The stored form is also the more faithful model, independently of the bug: a bunch created
at a waist with dispersion but no dispersion derivative keeps `rho_thx_gamma = 0` however
far it drifts, while `alpha_x` and `rho_x_gamma` both change — so the two genuinely are
independent parameters, and the old formula silently imposed a relation between them.
`fit_gaussian` now extracts them too, rather than discarding that part of the covariance.

## Consequences

The resulting transport is exact rather than approximate: `drift` composes and reverses to
round-off (~1e-16), which is what `test_drift_composes` and `test_drift_is_reversible` pin.

# D038 — the spectrum-width breakdown's `theta_col` is the geometric mean of `Target`'s x/y collimation half-angles

Status: implemented
Class: architecture

## Problem

`estimate_spectrum_width` needs a single scalar collimation half-angle, but `io.target.
Target` already owns `theta_x_col`/`theta_y_col` separately (§3.4) — the question is
whether to combine them at the call site or duplicate a `theta_col` field on analytical's
own schema.

## Decision

`engines.analytical.engine.AnalyticalEngine.run` computes `theta_col = sqrt(target.
m("theta_x_col") * target.m("theta_y_col"))` and passes that single scalar to
`estimate_spectrum_width`, rather than adding a `theta_col` field to `engines.analytical.
schema.ANALYTICAL_SPECS`.

## Alternatives considered

**Take `min(theta_x_col, theta_y_col)` (the tighter, more conservative aperture).**
Defensible, but inconsistent with every other x/y-combining choice this module already
makes, all of which use the geometric mean.

## Rationale

A second copy on analytical's own schema would duplicate state another module owns, the
same rule `xigma/schema.py`'s own docstring states for why xigma's schema excludes the
collimation window. The geometric mean matches the x/y-combining convention this same port
already uses elsewhere for elliptical inputs — the laser waist (`sigma_lr0` in
`estimate_yield`) and the angular-divergence term (`emit_width` in
`estimate_spectrum_width`) — rather than introducing a new one.

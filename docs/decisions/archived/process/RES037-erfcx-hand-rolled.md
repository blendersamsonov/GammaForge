# RES037 — `erfcx` is hand-rolled from `math.erfc`, not a new `scipy` dependency

Status: implemented
Class: process
Archived: 2026-09-08

## Problem

`engines.analytical.formulas` needs `erfcx` (`exp(nu**2) * erfc(nu)`), which the
predecessor got from `scipy.special.erfcx`, but `pyproject.toml` declares only `numpy`
today.

## Decision

`engines.analytical.formulas._erfcx` computes `exp(nu**2) * erfc(nu)` directly via
`math.erfc` below `nu = 25`, and a standard asymptotic series above it, instead of calling
`scipy.special.erfcx`.

## Alternatives considered

**Add `scipy` as a dependency.** Technically simpler and matches the predecessor exactly,
but breaks a stated dependency-surface precedent (`_chi2_6_cdf`'s own docstring) for a
regime this port never actually reaches — no operational benefit to offset the new
dependency.

## Rationale

`io.bunch._chi2_6_cdf` already sets the precedent of writing out a closed-form special
function rather than adding `scipy`, "so `gammaforge.io` keeps its dependency surface to
what `pyproject.toml` already declares." The realistic argument range was checked by hand,
not assumed: the baseline scenario (`validation.scenarios.BASELINE`) gives `nu ~ 0.06`,
and the predecessor's own worked example gives `nu ~ 0.48` — both far below the `nu ~ 25`
regime where `erfcx`'s overflow protection actually matters, so the direct-then-asymptotic
hand-rolled version is safe and proportionate, not a numerically fragile shortcut.

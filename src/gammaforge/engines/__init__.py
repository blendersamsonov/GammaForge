"""Engine implementations behind the uniform `Engine` interface (GRAND_PLAN.md §4).

`base` defines the interface itself — the `Engine` protocol (§4.1) and the
`RecomputeCost` tiers (§5). It exists ahead of the engines because the validation harness
runs *an engine* and needs a name for what that is (D018); the `ENGINES`
registry does not, and arrives with the first engine to register.

Will hold `xigma` (first-class), `analytical` (first-class), `kascade` (minimal port).
`delta` lives in `gammaforge.validation.references`, not here (§4.5) — it is a
validation-only reference, never a registered engine.
"""

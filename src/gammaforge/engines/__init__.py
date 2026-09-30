"""Engine implementations behind the uniform `Engine` interface.

`base` defines the `Engine` protocol and `RecomputeCost` tiers. It exists ahead
of the engines because the validation harness
runs *an engine* and needs a name for what that is (RES018); the `ENGINES`
registry does not, and arrives with the first engine to register.

Will hold `xigma` (first-class), `analytical` (first-class), `kascade` (minimal port).
`delta` lives in `gammaforge.validation.references`, not here — it is a
validation-only reference, never a registered engine.
"""

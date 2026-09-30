"""Engine implementations behind the uniform `Engine` interface.

`base` defines the `Engine` protocol and `RecomputeCost` tiers (RES018). `catalog` is the
one public enumeration of the engines that ship here: `LocalRunner`, calculation
serialization and the GUI all resolve engines through it, so none of them imports an
engine implementation module (RES095).

An engine's role — a selectable calculation, the analytical estimate overlay, or an
internal/validation-only reference — is data on its catalog entry, not a name check in a
frontend.
"""

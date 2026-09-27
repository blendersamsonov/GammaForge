"""Validation suite: scenarios, runners, and independent references (GRAND_PLAN.md §7).

Guiding principle: **tests assert physics, not implementation.** Modules, in dependency
order:

* `scenarios` — the shared scenario bank every leg runs against
* `metrics` — how two results are compared (window-integrated, statistical tolerances)
* `runners` — running one engine on one scenario
* `invariance` — the chunk / prefilter / backend / seed properties of §7
* `run` — the entry point: ``python -m gammaforge.validation.run``

`references/` holds delta's validation-only reference implementations, never registered
engines.
"""

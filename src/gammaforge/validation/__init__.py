"""Validation suite: scenarios, runners, golden references (GRAND_PLAN.md §7).

Guiding principle: **tests assert physics, not implementation.** Modules, in dependency
order:

* `scenarios` — the shared scenario bank every leg runs against
* `metrics` — how two results are compared (window-integrated, statistical tolerances)
* `golden` — committed reference snapshots: format, provenance, comparison
* `runners` — running one engine on one scenario
* `invariance` — the chunk / prefilter / backend / seed properties of §7
* `make_references` — regenerating goldens from the predecessor; the *only* place the old
  repo is referenced, and it runs it in a subprocess (`DECISIONS.md` D019)
* `run` — the entry point: ``python -m gammaforge.validation.run``

`references/` holds the committed snapshots under ``data/``, and will hold `delta` (§4.5)
— a validation-only reference implementation, never a registered engine.
"""

# RES031 — `XigmaEngine` is not passed to `run_suite()` by `validation.run.main()`

Status: implemented
Class: testing

## Problem

`run_suite(engines=[...])` already works — `invariance.engine_checks` and
`golden.compare_to_golden` are engine-generic (Phase 2) — but wiring `XigmaEngine` into the
routine suite needs to reckon with the cost of `run_engine` running an engine against the
scenario's own `Target.outputs`, sized for the predecessor's GPU importance sampler.

## Decision

The engine exists (`engines/xigma/engine.py`) and is tested (`tests/test_xigma_engine.py`,
`tests/test_stage1_stage2.py`), but `python -m gammaforge.validation.run` does not exercise
it: `main()` still calls `run_suite()` with no engines, same as before this phase.

`run.py::identity_checks` gained a fourth leg instead (RES029): Stage 2's table kernel
against `delta` at one point, at the identity harness's existing 2000-particle, small-table
scale — the thing that actually exercises `stages.deposit_shape_table`/
`stages.retarget_ahat` (RES032)/`angular_spectrum_from_table` in the routine suite.

## Alternatives considered

**Wire the engine in at a reduced, suite-only resolution.** Would need a second, ad-hoc
`Target` variant that no golden or scenario elsewhere uses, and still would not be testing
what the scenario bank's actual outputs cost.

**Reduce `_DEFAULT_OUTPUTS`'s resolution to something numpy can afford.** Would silently
change what every future engine (including a real GPU xigma) is validated against, for a
limitation of this phase's kernel alone. Revisit once cupy/numba land (RES029) or Phase 7's
"full scenario bank" exit criterion is actually being worked.

## Rationale

`COLLIMATED_SPECTRUM` at `(64, 16, 16)` and `ANGULAR_DISTRIBUTION` at `(64, 64)` are sized
for the predecessor's GPU importance sampler (§12's own risk row: "measured 27s @ 64 energy
bins on CPU... linear in n_energy"). This phase's numpy kernel is also linear in the
*angular* grid size (a plain Python loop over observation points, §4.2/RES029), so
`COLLIMATED_SPECTRUM` alone measured 36.6s at that resolution and 100k particles — times
`engine_checks`' four full runs per scenario, times three scenarios, `python -m
gammaforge.validation.run` would go from ~3s to tens of minutes. That is real, expected
Calculate cost for a deliberate query (§12: "expected, not a defect"), not a defect in a
suite meant to be run routinely.

# 06 — Honest snapshots, validated boundaries and fixed-input caches

Read [coordination rules](README.md). Covers A13, remaining A14 and the review part
of A16. Land result changes after 02 and Collision changes after 03.

## Evidence and entry points

`src/gammaforge/io/{calculation,interaction,bunch,laser,schema,target,results}.py`,
`engines/runner.py`, `engines/xigma/collision.py`; tests for calculation runner,
GUI controller, bunch, laser, schema, target/results and xigma. Read RES030, P2/P4,
and the independent RNG-substream/resampling contract in the plan.

`CalculationRequest` is frozen but its `engine_params` dict can be cleared.
Collision publicly permits replacing `interaction` or `params` after building its
cache; the earlier probe doubled N_e this way and still obtained the old cached
yield. Mapping proxies are shallow and particle/result arrays may alias callers.

Constructor probes accepted `SamplingSpec(n_particles=1.5)`, negative seeds,
non-finite `rel_energy_spread`/`beta_ff`, and unequal Bunch array lengths. A rich
GUI FieldSpec validator does not protect direct notebook/dataclass construction.

## Ready scope

1. State ownership at the calculation/runner and fixed-Collision boundaries.
   Snapshot nested parameter mappings; prevent replacement of fixed cache inputs
   or invalidate exactly and explicitly. Keep mutable memoization internal.
   Demonstrate the policy against original-container mutation and returned data
   mutation, not merely assignment to a frozen attribute.
2. Choose the minimum safe array ownership/copy policy. Read-only views alone do
   not protect against a caller that still owns a writable alias. Conversely,
   blindly deep-copying every large array at every stage defeats the memory fix.
   Isolate borrowed mutable data at the boundary that actually requires a stable
   snapshot, and document that contract.
3. Validate finite physical scalars, integral counts, the supported seed domain,
   and consistent one-dimensional particle-array lengths at construction/use
   boundaries. Keep zero/empty cases that are meaningful. Coordinate output
   resolutions, axes and range validation with 02 rather than duplicating it.
4. Preserve exact charge rescaling, relative weights, unit conversions, seed
   determinism and independent RNG substreams. Arbitrary LaserField objects may
   be mutable: do not silently claim they are hashable immutable cache keys.

## Review only: cross-run reuse

LocalRunner reuses sampled interactions, but each XigmaEngine.run constructs a new
Collision. Produce a concrete stage/input invalidation table and proposed tests
for target, output, charge, laser intensity, sampling, geometry and energy edits.
Identify which declarations are honest today. **Do not build a general cache or
promise QUERY_ONLY/REUSE_INTERMEDIATES before it exists.** Broader cross-run stage
reuse needs a separately agreed implementation scope, especially after 07's
LaserField review. The private multi-intensity cache is not proof of a working
public retarget workflow.

## Acceptance

- Mutating the original request dict/nested values after submission cannot change
  an in-flight calculation. Cached results are not silently corrupted through
  array aliases or replacement of supposedly fixed inputs.
- Invalid constructor inputs fail early with useful field names; valid NumPy
  integer inputs and explicitly supported empty arrays still work.
- Same seed/parameters reproduces the bunch; charge changes preserve samples and
  scale photon results exactly. Cache tests use stage-call counts, not timings.
- Existing notebook workflows have an explicit migration note if public
  mutability changes. Publish the snapshot contract to 05. No GUI engine branches,
  Config class, global cache registry, or speculative serialization hooks.

Starting check: `.venv/bin/pytest -q tests/test_calculation_runner.py tests/test_gui_controller.py tests/test_bunch.py tests/test_laser.py tests/test_schema.py tests/test_target_results_interaction.py tests/test_xigma_engine.py`.

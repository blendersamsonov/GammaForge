# Alpha inputs handoff

Integrated into main. Collision replacement and returned-sample mutation are now
guarded by the sampler integration; arbitrary mutable fields remain outside alpha.

Worktree: `/tmp/gammaforge-alpha-inputs`
Branch: `work/alpha-inputs` (based on `main`)

## Status

Milestone 1 complete. The input-boundary patch is implemented in `gammaforge.io`:

- `CalculationRequest` snapshots its nested `engine_params` mapping and requires
  `Parameters` values.
- `SamplingSpec` accepts NumPy integer scalars, rejects non-integral counts, and enforces
  the supported seed domain `[0, 2**31 - 1]`; non-finite prefilters fail early.
- `InteractionParameters` rejects non-finite/non-positive `N_e` values on direct
  construction.
- Beam and Gaussian-laser scalar/quantity fields reject non-finite values while existing
  deferred physical-range validation remains in `validate(...)`.
- `Bunch` takes independent, read-only copies of all 1-D particle arrays, checks equal
  lengths, shallow-copies the top-level metadata mapping, and preserves valid empty
  vectors. Pure internal transforms use a private fresh-array path to avoid repeating
  defensive copies; public construction remains copy-on-entry.
- `FieldSpec` accepts NumPy real scalar values; `Target` rejects non-finite collimation
  angles.

Milestone 2 complete. `OutputRequest` snapshots manual-range mappings and normalizes
resolutions; `Target` snapshots an output sequence. Decision record:
`docs/decisions/implemented/architecture/RES064-input-boundary-snapshots-and-array-ownership.md`.

Focused verification: `213 passed`:

```sh
cd /tmp/gammaforge-alpha-inputs
PYTHONPATH=src /home/alexander/Work/Code/GammaForge/.venv/bin/pytest -q \
  tests/test_schema.py tests/test_bunch.py tests/test_laser.py \
  tests/test_target_results_interaction.py tests/test_calculation_runner.py
```

`tests/test_doc_staleness.py` passes. The integration added the `RES064` index row;
the decision-format guard now passes as well.

## Resume commands

```sh
cd /tmp/gammaforge-alpha-inputs
git status --short
PYTHONPATH=src /home/alexander/Work/Code/GammaForge/.venv/bin/pytest -q
git diff --check
```

## Scope/deferred items

- No edits to `engines/xigma/collision.py` or `engines/runner.py`.
- No GUI or kascade work.
- The request snapshots the mapping and immutable `Parameters`; arbitrary mutable
  `LaserField` implementations are intentionally not claimed to be hashable immutable
  cache keys. This remains a runner/cache ownership concern for root to decide.
- Collision fixed-input replacement/invalidation remains with root’s sampler/collision
  work; this branch only reports observations.

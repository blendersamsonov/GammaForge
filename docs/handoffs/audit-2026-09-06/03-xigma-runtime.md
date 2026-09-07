# 03 — Bounded xigma execution and empty-input behavior

Read [coordination rules](README.md). Covers A04, A07, the warning part of A17 and
small production-declaration duplication in A18. No emission-formula changes.

## Entry points

`src/gammaforge/engines/xigma/{stages,collision,engine,chunking}.py`, and
`src/gammaforge/io/target.py` (read/coordinate
with 02). Tests: `test_xigma_engine.py`, `test_stage1_stage2.py`,
`test_stage0_delta.py`. Read RES029, RES030, RES032, RES054.

`angle_integrated_spectrum` creates full particle × energy arrays. A million
particles with 2,048 energy samples consumes 16.384 GB per double-precision
temporary, before other arrays. Stage 0's chunking does not bound this query.

An explicit empty selected Bunch returns zero yield/spectrum but angular and
collimated requests reach reductions over empty arrays. `Collision.run` also
autoranges unsupported requests before omitting them, so an unsupported temporal
output can raise on a nonexistent overlap window.

## Work

1. Bound the table-free spectrum's temporary allocations using the existing
   chunking machinery where appropriate. Preserve scalar/vector API behavior and
   numeric output; consider both particle and energy dimensions. Check output
   allocation budgets before attempting an enormous requested grid. No arbitrary
   GUI resolution hardcaps or separate auto-chunk framework.
2. Define zero-result assembly for empty interactions for every supported output.
   Filter unsupported requests before computing their ranges or prerequisites.
   Exercise the actual prefilter-to-empty path as well as explicit empty input.
3. Expose the already documented linear/no-nonlinear-redshift spectrum limitation
   through Results warnings/metadata, as analytical does for its approximations.
   Coordinate persistence of warnings with 05. Do not implement new spectrum
   physics or declare the polarization issue from 01 solved.
4. Consolidate duplicated supported-output declarations/defaults only where they
   express the same existing contract and are in files already owned here.
   Leave reference formulas, multi-intensity cache design and scalar/vector
   polarization consolidation to their assigned owners.

## Acceptance

- Chunk-size sweeps match a small unchunked reference within a justified floating
  reduction tolerance, including nonuniform weights and scalar energy queries.
- A deterministic allocation/chunk spy proves no full N-particle × N-energy
  temporary is attempted for an oversized case. Do not deliberately OOM the host.
- Every supported output on an empty bunch returns a valid zero result using 02's
  contract. An unsupported request cannot trigger its own autorange failure.
- Existing approximation warnings reach Results and the GUI without engine-type
  branches in GUI code. No scientific constants or tuned defaults change.

Serialize edits to `stages.py` with 01's eventual approved physics work, and land
before 06 changes Collision ownership. Invalid-resonance warnings near zero gamma
may reveal the physics path's masking bug: give 01 a reproducer instead of globally
suppressing warnings or guessing a physical denominator.

Starting check: `.venv/bin/pytest -q tests/test_xigma_engine.py tests/test_stage1_stage2.py tests/test_stage0_delta.py`.

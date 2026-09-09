# CuPy numerical promotion work

Status: implementation, final real-CUDA gate, and full-suite verification pass
after the concurrent Stokes update. NumPy polarization stabilization
landed separately as RES070; do not modify that reference implementation in this
workstream.

## Scope and ownership

Root owns research, review, real-CUDA acceptance runs, and documentation. Cheaper
agents implement independent bounded pieces, with no independent commits:

- `cupy_accuracy_controls`: sampler, schema, Collision, angular-dispatch plumbing;
  new sampler-control tests. Checkpoint: /tmp/gammaforge_cupy_controls_handoff.md.
- `cupy_geometry_bench` (resumed): numerical convergence module and its tests.
  Checkpoint: /tmp/gammaforge_cupy_convergence_handoff.md.
- `cupy_geometry_tests` (resumed): standalone release runner and end-to-end tests.
  Checkpoint: /tmp/gammaforge_cupy_release_handoff.md.

No GUI, Kascade optimization, outgoing GPU Stokes, new physics, or Stage-0/1 GPU
implementation.

The two resumed geometry agents hit their usage limits after writing initial code.
The remaining cheaper `cupy_accuracy_controls` agent implemented the review fixes
across all three pieces. Source files are authoritative over older checkpoints.
Concurrent laser pulse-train and Stokes changes landed through commit 549c801;
preserve them. RES072 records this workstream.

## Recent polarization compatibility

RES073 factors out the incident-axis rotation and adds a separate CPU Stokes query.
CuPy still evaluates the scalar intensity factor and now accepts signed ellipticity
in [-1, 1], matching intensity's even dependence on handedness. CPU guard/symmetry
regressions and CUDA comparisons against Stokes I were added without replacing the
independent extended-precision Eq. udef tests. The sampler formula is unchanged.
The release runner fingerprints actual imported source modules before/after a run
and fails if those files changed during validation.

## Acceptance plan

1. Preserve the default 32-ring/32-subsampling behavior. Add bounded independent
   ring and sampling controls through the typed schema and result provenance.
   Above-default ring refinement must be executable on the available GPU.
2. Compare the same interpolated input H on independently refined CPU integration
   grids. Vary GPU rings separately from subsampling, keeping output grids fixed.
   Measure integral, integrated absolute density error, and spectral centroid.
3. Exercise the shared scenario bank and supplementary crossed/high-energy/off-axis
   cases. A nonconverged reference is a failed/inconclusive gate, never a pass.
4. Require actual CUDA in the release command; record environment, settings, and
   failures in a machine-readable report. Retain the former crossed-overflow case
   as an actual public-engine regression. CPU-only ordinary tests may skip CUDA.
5. Review measured results before changing support claims. Backend numerical
   agreement does not close independent arbitrary-angle emission validation.

Existing CPU/GPU density thresholds are not to be loosened to obtain a green run.
The existing 3% integral / 5% integrated-density acceptance is a finite-resolution
numerical check, not a uniform production accuracy guarantee or the plan's
roundoff-level backend-invariance target.

## Resuming

Read this file, PROGRESS.md, the three checkpoints, and the current diff. Agent
changes share the workspace and survive usage limits. Run GPU jobs sequentially;
sandbox device absence is not proof that CUDA is unavailable on the host.
Use the .venv interpreter; real-CUDA commands may need sandbox escalation.

## Verification checkpoint

- Final complete actual-CUDA suite: 673 passed, one skipped, 25 CuPy experimental
  JIT warnings in 305.58 seconds. This includes the signed-ellipticity/Stokes I
  regressions, public crossed-engine runs, 32-/64-capacity controls, and CDF tests.
- Minimum Python 3.12/no-CuPy focused run: 38 passed, 24 CUDA skips. The standalone
  runner exits 1 with an explicit CuPy-import failure report; it cannot claim an
  all-skipped success.
- Initial six-case numerical sweep passed five cases. High gamma required more
  reference resolution: 8x to 16x changed L1 by 9.13%; 16x to 32x by 2.30%;
  32x to 64x by 0.80%. The final crossed/circular high-gamma cases use 32x/64x,
  without loosening tolerances. Initial sweep used uniform ahat synthetic bins;
  the final passing report independently reran the matrix with nonuniform bins.
- A short benchmark execution check (three warmups, five repeats) still gives
  about 92 ms resident on the resolved 32-ahat table versus 138 ms for the
  older RES068 baseline. This is not an exclusive final timing comparison.
- Final gate after the latest Stokes changes passed all 80 numerical checks across
  eight cases plus both public-engine CUDA smoke runs. Source fingerprints stayed
  unchanged. Report: docs/validation/cupy-release-2026-09-08.json. Worst default
  finite-window integral difference 0.747%, integrated density L1 0.799%.
- Reproduce the final gate (implementation frozen during acceptance):
  `.venv/bin/python scripts/validate_cupy_release.py --output /tmp/gammaforge-cupy-release-final-2026-09-08.json`.
- Documentation guards pass: 18 tests. Headless alpha validation passes on the
  current tree. The code graph was refreshed with AST-only extraction; JSON report
  contents are not indexed, so read the saved report directly for measurements.
- No remaining implementation step in this workstream. Acceptance results describe
  the CuPy promotion commit (see git history). Keep CuPy experimental pending independent arbitrary-angle emission
  validation; passing this matrix is not a universal accuracy guarantee.

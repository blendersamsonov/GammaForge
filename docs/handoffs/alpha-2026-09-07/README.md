# Script alpha integration checkpoint

## Scope and deliverables

Prepared locally as 0.1.0a1; no registry publication, remote push or release tag.
The sampler, input-boundary, persistence/validation and example branches are merged
into main. Start with `docs/ALPHA.md`, not the earlier per-agent checkpoints.
GUI and kascade remain development-only, outside alpha support.

The original plan was: integrate the sampler without losing audited runtime fixes;
close high-priority script input/persistence/tooling handoffs; reproduce the original
XIGMA explicit-input workflow; check analytical/xigma agreement and a clean install.
Those script-alpha deliverables are complete. GPU promotion is explicitly deferred
because its stronger distribution comparison fails; see `docs/ALPHA_GPU_VALIDATION.md`.

## Verification on 2026-09-07

- Main working-tree development environment, Python 3.14: 567 passed, 3 skipped,
  1 strict expected failure in 187.47 s. This includes the pre-existing local kascade
  changes; those changes are not part of the alpha integration commits.
- Minimum editable environment, Python 3.12.13 with the exact runtime constraints:
  546 passed, 14 optional tests skipped in 122.02 s. Dependency consistency passes.
- Clean wheel environment, no GUI or CUDA dependency: alpha validation passes;
  the complete guide calculation, HDF5 save/load and submitted-request reload pass.
- The crossing-angle script was run from that installed wheel, outside the checkout.
  Committed figure/data use NumPy 2.0.0, Pint 0.25 and Matplotlib 3.8.4.
  Mean/max absolute yield differences are 0.0342% / 0.09193%.
- Full production validation: every executed numerical check passes, but exit status
  remains 1 for three coverage blockers. It must not be described as fully certified.
- Documentation/input/tooling checks after final edits: 25 passed, 1 optional skip.
- AST graph refreshed. Documentation semantic extraction was not rerun; the tool also
  reported that the provenance JSON produces no AST nodes.

## Resume without repeating completed work

The smaller-model agents reached their usage limits; their branches and checkpoints
were retained and integration was completed by the parent agent. Existing worktrees:

- `/tmp/gammaforge-alpha-integration`: `release/alpha-script`, integrated sampler.
- `/tmp/gammaforge-alpha-inputs`: `work/alpha-inputs`, input-boundary work.
- `/tmp/gammaforge-alpha-example`: `work/alpha-example`, figure script.
- `/tmp/gammaforge-alpha-io`: `work/alpha-io`, persistence and validation. Contains
  redundant uncommitted dependency-floor edits; main is authoritative.

Temporary test environments are `/tmp/gammaforge-alpha-minimum` and
`/tmp/gammaforge-alpha-wheel`; the latter has an installed wheel, not an editable checkout.
The built local artifact is `dist/gammaforge-0.1.0a1-py3-none-any.whl`.
Do not reset the main worktree to clean up: unrelated author changes and audit files remain.

The meaningful next GPU task is convergence diagnosis of both quadrature and sampler,
not a tolerance adjustment. Start with:

```sh
.venv/bin/pytest -q tests/test_xigma_gpu_sampler.py --runxfail
```

On this host CUDA requires execution outside the sandbox; a sandbox no-device error
does not establish absence of the GTX 1660 Ti. The open physics-author decisions in
`PROGRESS.md` and independent arbitrary-angle validation remain unchanged.

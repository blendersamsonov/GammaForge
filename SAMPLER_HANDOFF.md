# Sampler integration handoff

## State

- Worktree: `/tmp/gammaforge-alpha-integration`
- Branch: `release/alpha-script`, initially `54e9132` (`main`).
- Requested merge source: `feature/xigma-importance-sampler` (`b74c210`).
- Scope owner: xigma code/tests, sampler decision and `DER008`, and merge-conflict docs.

## Preserved main work to integrate

The uncommitted main runtime audit changes are in `stages.py`, `collision.py`,
`engine.py`, `test_stage1_stage2.py`, and `test_xigma_engine.py`. They add bounded
memory for table-free spectra, empty-bunch results, and a public supported-output
contract. They must be applied after resolving the sampler merge.

## Commands and results so far

```sh
git -C /tmp/gammaforge-alpha-integration status --short --branch
# ## release/alpha-script
git -C /home/alexander/Work/Code/GammaForge diff -- <runtime-audit paths>
```

`graphify query "xigma importance sampler Stage 2 collision integration"` confirmed
that `Collision` owns the Stage-2 query and the previous CPU quadrature was RES029.

The merge was started and conflicts are resolved in the working tree. `GRAND_PLAN.md`
uses main's authoritative version; `INDEX.md` retains main `RES061` and adds the sampler
as `RES062`. The sampler decision filename and its `DER008` pointer have been renamed.

Main's runtime audit diff has been reconciled into the sampler work: bounded table-free
spectrum working sets, empty-bunch zero slices, supported-request autoranging, and the
results warning are present. The imported head-on GPU kernel initially retained an older
relative-angle polarization shortcut, so it has been replaced with the existing RES060
head-on linear specialization using sampled electron lab velocity and observer direction.
`Collision` is now frozen at its public input boundary while its private memoization uses
`object.__setattr__`; replacement after Stage 0 is regression-tested.

Focused results:

```text
tests/test_xigma_engine.py: 16 passed, 1 warning in 17.43s
five new/runtime/route tests in test_stage1_stage2.py: 5 passed in 1.12s
test_xigma_gpu_sampler.py: 6 skipped (CuPy 14.2.0 installed; cudaErrorNoDevice)
tests/test_decision_format.py tests/test_derivation_format.py tests/test_doc_staleness.py:
18 passed, 2 skipped in 4.27s
python -m gammaforge.validation.run: ALL CHECKS PASS
```

The full pytest invocation exceeded the command runner's 30-second foreground window
and was terminated without a final result; no test process remained afterward.

The final routing contract is explicit: `auto` selects CuPy only with an available CUDA
device, zero ellipticity, and zero crossing angles. Other geometries use NumPy; an
explicit unsupported CuPy request raises. CuPy uses a deterministic golden-ratio sequence
with fixed `samples_total=256` and `subsampling=32` (no RNG seed). GPU output records
those settings and every angular/collimated result records its selected Stage-2 backend.

## Remaining

1. Stage the latest test/checkpoint edits and commit the integration.
2. A CUDA-equipped host must run the GPU-versus-NumPy comparison in
   `tests/test_xigma_gpu_sampler.py`; this host has CuPy but no CUDA device.

## Completed commit

- Merge commit: `74615e1 Integrate xigma importance sampler for alpha`
- Parents: alpha base `54e9132` and `feature/xigma-importance-sampler` `b74c210`.

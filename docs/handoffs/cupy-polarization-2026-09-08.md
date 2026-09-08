# CuPy incident polarization and crossing-angle extension

Status: implementation and verification complete. The cheaper agents reached their limits, but their last queued
test-adapter edit landed successfully. No further library implementation was needed
from root. The completed head-on baseline for comparisons is commit 6420781.

## Scope and ownership

User requested root research/orchestration with cheaper-model implementation.
Root owns physics review, acceptance checks, integration, decisions and reporting.
Implementation agent owns sampler and dispatch; test agent owns new regressions.
NumPy remains default; GUI, kascade, outgoing Stokes and spatially varying polarization
are excluded. Do not modify unrelated `.claude/` or `docs/2-2016.pdf`.

## Physics contract

Authority: manuscript at /home/alexander/Work/Papers/2026/Compton-Numerics/xigma.tex,
Eq. udef (lines 296–304), plus DER004–DER006 and author-approved RES060.
Use field-free electron and observer directions in one lab frame; rotate the laser
basis once with R_y(theta_xz) R_x(theta_yz). The intensity factor is

    (|U0|^2 + epsilon^2 |U1|^2) / (1 + epsilon^2)
    Ui = (n - beta*u) (n.ei) / d - ei,  d = 1 - beta*u.n

The original expanded norm has severe float32 cancellation when the basis has a
longitudinal component. Root's collinear probe at crossing 0.3 rad gives 1.0 at
gamma 2000 and 0.0 at gamma 10000, versus cos(0.3)^2 = 0.91266780745.
Use the vector norm instead, with stable evaluation of both differences:

    delta = 1/[gamma^2 (1+beta)]
    Delta = n-u
    d = delta + beta |Delta|^2/2
    n-beta*u = Delta + delta*u

For slopes xe,ye and xo,yo, norms ve=sqrt(1+xe^2+ye^2), no=sqrt(1+xo^2+yo^2):

    Delta_z = [(xe-xo)(xe+xo)+(ye-yo)(ye+yo)]/[ve*no*(ve+no)]
    Delta_x = (xo-xe)/no + xe*Delta_z
    Delta_y = (yo-ye)/no + ye*Delta_z

This avoids subtracting rounded near-one longitudinal directions. It is algebraic
rearrangement of Eq. udef, not new emission physics. No clipping negative intensity
or normalization tuning. Existing Stage-0 flux and Collision energy conversion
already carry crossing angle; do not duplicate them in the sampler.

## Acceptance and resumption

1. Direct production-device factor vs independently computed high-precision Eq. udef,
   including collinear gamma 2000/10000, two-plane tilts and off-axis electrons.
2. Linear/elliptical/circular limits, circular basis-rotation invariance, zero-angle
   continuity; existing CDF/support/head-on tests stay green.
3. Refine INPUT angular integration of the same H (not output grid) for CPU/GPU
   mass and absolute-density checks. Include actual crossed deposited tables and a
   nonuniform ahat case; retain 3% mass / 5% integrated-density limits unless root
   establishes a distinct convergence issue.
4. End-to-end energy/density Jacobian and backend metadata/routing tests. Missing CUDA
   still errors for explicit cupy and falls back for auto.
5. Warm resident and host timings relative to 6420781; final full suite and CPU-only
   optional-dependency collection. Record results before updating current support docs.

Agents must leave concrete paths, completed changes, failing commands and next steps
in /tmp handoffs before stopping; no independent commits or unrelated edits.

## Usage-limit recovery

Both implementation and test agents hit their usage limits on 2026-09-08. Their
shared-workspace edits remained intact; root continued read-only numerical checks,
benchmark execution, review and documentation. Older /tmp checkpoints may still
describe the deleted temporary host launcher; the final test uses a local rawkernel
probe of the private production device helper.

Written implementation:

- Private device helper: `_polarization_factor_device(gamma, xe, ye, xo, yo,
  e0x, e0y, e0z, e1x, e1y, e1z, xi00, xi11)` in the sampler module.
- Host `_polarization_parameters(psi_pol, ellipticity, theta_xz, theta_yz)` returns
  the eight basis/weight scalars above; the normal sampler converts them to float32.
- GPU wrapper accepts all geometry keywords; dispatch restriction removed;
  experimental warning retained. NumPy emission formulas unchanged.
- A larger crossing benchmark exposed intermediate overflow: baseline H maximum
  8.77e20, polarization factor about 3.40e5, and an unscaled weighted term about
  1.10e41. Its energy-scaled value is only 1.22e28. The implementation agent moved
  the existing `1/s^2` into the prefactor before H multiplication and removed the
  late division in the atomic add. This preserves the normalization exactly in
  algebra; rerun numerical checks after this final source edit.

Acceptance disposition:

1. The private-helper adapter is complete and all 12 direct CUDA precision/limit
   checks pass. No public test-only launcher was added to the library.
2. Crossed deposited and synthetic nonuniform-ahat refined-CPU comparisons pass.
   No failed density tolerances were loosened.
3. The executable benchmark reproduces and now passes the former overflow case at
   crossing (0.3, 0.2), both ahat grids, observer axes ±3e-4 and s from 0.65 to 0.99
   mean(gamma)^2. It raises on non-finite output. Additional manual actual-engine
   runs at (0.02, -0.015) and (0.3, 0.2) with ellipticity 0.4 are finite/nonnegative/
   nonzero and report CuPy. The controlled Jacobian test checks scaling separately.
4. Final exclusive warmed benchmark completed after the overflow correction; use
   `docs/ALPHA_GPU_VALIDATION.md`, not the earlier incomplete benchmark checkpoint.
5. Full suite passes (604 passed, one skipped). Graph refreshed; root owns scoped
   commit. Preserve unrelated `.claude/` and `docs/2-2016.pdf`.

Agent checkpoints:

- /tmp/gammaforge_cupy_geometry_impl_handoff.md
- /tmp/gammaforge_cupy_geometry_tests_handoff.md
- /tmp/gammaforge_cupy_geometry_bench_handoff.md

Root's read-only reference check evaluated the proposed float32 algebra against
direct extended-precision Eq. udef for 1,600 basis evaluations: maximum
abs(error)/max(1,reference) was about 8.0e-7. This is not a substitute for the
production-device tests, which now pass independently.

## Root verification after the final arithmetic reorder

- `.venv/bin/python -m gammaforge.validation.run --alpha`: all checks pass (the
  unchanged analytical/NumPy alpha workflow, run before the final GPU-only reorder).
- `.venv/bin/python -m pytest -q tests/test_xigma_gpu_sampler.py
  tests/test_xigma_sampler_cdf.py tests/test_xigma_sampler_regressions.py
  tests/test_stage1_stage2.py`: 53 passed on CUDA after the reorder.
- `.venv/bin/python scripts/benchmark_xigma_polarization.py --warmup 1 --repeats 1`:
  all ten old/new/table/geometry combinations complete, including crossing (0.3, 0.2)
  on both ahat grids. This ran alongside tests and is an execution smoke check only,
  not an exclusive warmed timing result.
- Decision/staleness/derivation-format tests: 18 passed. `git diff --check`: clean.
- New polarization/scenario checks: 19 passed on CUDA after the reorder.
- Minimum Python 3.12 environment, focused geometry/sampler/Stage-1/2 tests:
  31 passed, 25 CUDA tests skipped cleanly.
- Exclusive final benchmark (20 warmups, 15 repetitions): all cases complete. New
  head-on resident times 3.50 / 92.57 ms versus baseline 5.04 / 138.28 ms on the
  one-/32-ahat-bin tables. Full geometry timing table is in the validation record.
- Final full suite: 604 passed, one skipped, 15 CuPy experimental-JIT warnings.

## Follow-up outside this CuPy implementation

Root's final high-gamma exact-limit probe found an existing NumPy cancellation issue:
`polarization_factor(10000., 0., 0., 0., 0., 0., 0., 0.3, 0.)` returns 1.3886977,
whereas the exact collinear cosine-squared value is 0.9126678. The CUDA vector helper
passes this limit. Stabilizing NumPy is a separate numerical repair, now tracked in
PROGRESS and the validation record; it is not an independent-emission physics claim
or a change to total-yield overlap.

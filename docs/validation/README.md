# What has been checked

GammaForge has two separate validation questions:

1. **Numerical agreement:** do two implementations give similar answers for the
   measured inputs, and do those answers settle as calculation grids are refined?
2. **Scientific accuracy:** does the model describe the intended physics across the
   full range of inputs?

The dated files here record particular measurements, including their settings and
limits. They are evidence for decisions and derivations, not instructions a user
must read before running GammaForge.

## Current result

| Question | What the measurements show | Record |
|---|---|---|
| Does production xigma agree with an independently coded delta-emission reference? | Yes, for the measured fixed observation directions after one bounded grid refinement. The run passed 388 numerical checks. Two scientific coverage blockers remain. | [Production comparison](delta-production-2026-09-29.md) |
| Does CUDA agree numerically with a refined CPU calculation? | Yes, for the nine measured cases; all 90 convergence checks passed. This does not establish independent physics accuracy. | [CUDA release gate](cupy-release-2026-09-28.md) |

Neither result establishes accuracy for every geometry or observer. In particular,
particle sampling, trajectories, angular apertures and additional independent
physics comparisons still need evidence. The production command reports these gaps
as blockers; [issue #1](https://github.com/blendersamsonov/GammaForge/issues/1)
and [issue #10](https://github.com/blendersamsonov/GammaForge/issues/10) track
the remaining work.

## Why the other records remain

| Record | Purpose |
|---|---|
| [Early GPU sampler](alpha-gpu-sampler-2026-09.md) | Historical measurements behind RES068, RES069 and RES072; use the newer CUDA gate for current acceptance. |
| [CUDA particle stages](cupy-stages01-2026-09-13.md) | Shows that Stages 0 and 1 actually ran on CUDA and agreed with NumPy for the measured cases. |
| [CUDA delta reference](delta-cupy-2026-09-14.md) | Checks the independent GPU emission reference against extended-precision CPU lines. |
| [Early delta convergence](delta-convergence-2026-09-10.md) | Shows how an earlier off-axis discrepancy shrank with finer grids; superseded as an acceptance gate by the current production comparison. |
| [Initial production comparison](delta-production-2026-09-14.md) | Records the six low-intensity refinement failures that motivated the bounded retry in the current report. |
| [Direction-dependent Doppler](direction-doppler-2026-09-12.md) | Checks a specific crossing-angle implementation change; it is narrower than scientific acceptance. |
| [Gamma proposal](gamma-proposal-2026-09-13.md) | Measures the accuracy and runtime of a CUDA sampling change; it is a numerical efficiency check. |
| [Gauss-Hermite and the Stage-1 table](adaptive-sampling-cubature-2026-10-02.md) | Preliminary. Records a negative result: tensor Gauss-Hermite integrates smooth totals well but cannot fill the Stage-1 table, because its product weights leave ~10³ effective points regardless of how many nodes are spent. Small budgets and a coarse reference, so the convergence rates in it are not usable. |

Only two historical JSON packets remain committed because RES076 and RES077 cite
their exact measurements: `delta-doppler-2026-09-09.json` and
`delta-numpy-pilot-2026-09-09.json`. Other generated packets and example figures
were removed after their relevant results were summarized in the records above.
Run new measurements into ignored `output/validation/`; commit a concise result
only when it supports a current claim, with its command, settings, result and limits.

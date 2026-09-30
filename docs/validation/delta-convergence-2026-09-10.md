# Fixed-direction delta convergence checkpoint — 2026-09-10

This is the durable measurement summary from the completed arbitrary-angle delta
implementation handoff. RES076–RES079 describe the implementation and numerical
interpretation. It is an exploratory checkpoint, superseded as an acceptance gate
by [the production report](delta-production-2026-09-29.md). The original raw
refinement packets were saved under `/tmp` and are not committed; the tables below
are the surviving measurements. They do not establish independent Stage-0 physics
or angular-aperture agreement.

The early transverse and refined pilot packets were removed from the current tree
after the measurements below were retained. Their grids are historical diagnostics;
use the later production gate for current acceptance.

The bounded baseline probe held 16,000 particles, 64 Stage-0 steps, seed 20260721,
24 common physical energy bins, gamma/shape-`ahat` bins at 32, retarget bins at
256, and q32/q64 energy integration. The same particles and reporting edges were
used within each geometry/observer comparison. Head-on and small crossed geometry
were measured on and off axis.

| Geometry | Observer | 16 angular bins | 32 angular bins | 64 angular bins |
|---|---|---:|---:|---:|
| Crossed | off axis | 14.1% L1 | 2.16% | 0.77% |
| Head-on | off axis | 13.0% | 1.81% | 0.77% |
| Crossed | on axis | 1.81% | 1.22% | 1.17% |
| Head-on | on axis | 1.26% | 1.26% | 1.17% |

Count error stayed below 0.2%; q32/q64 changes passed the local refinement
threshold. With 64 angular bins, changing gamma bins from 16 to 64 reduced
crossed off-axis L1 from 2.56% to 0.91%. With gamma 32 and angular 64, retarget
128/256/512 reduced crossed off-axis L1 from 4.63% to 1.17% to 0.65%.
Head-on off-axis followed approximately 4.6% to 1.2% to 0.65%. This localized
the large initial off-axis discrepancy to table and retarget resolution rather
than a normalization correction.

At the finer baseline configuration (gamma 32, angular 64, retarget 512), five
seeds at 16,000 particles gave crossed off-axis L1 from 0.46% to 1.02%; a
fixed-seed 8,000–64,000 particle sweep gave 0.41% to 0.70%. Count errors stayed
below 0.04%. Stage-0 steps 64→128 changed L1 by less than 0.001%; 128→256
was indistinguishable at reported precision. Switching active-region versus
illumination windows changed L1 by less than 0.01%.

A matched crossed off-axis CPU/CUDA probe at that configuration measured:

| Backend | Count error | Spectral L1 | Centroid error |
|---|---:|---:|---:|
| NumPy | -0.002% | 0.73% | 0.004% |
| CuPy | 0.15% | 0.81% | 0.005% |

A separate eight-case actual-CUDA release comparison reported all numerical
checks passing, with worst cited L1 0.85% in the crossed case. These are
numerical backend and fixed-direction reference checks, not acceptance of the
shared trajectory, luminosity, reduced emission model, or unmeasured apertures.
The current [CUDA release record](cupy-release-2026-09-28.md) has the newer
nine-case gate, including finite-line reconstruction.

Reproduce current checks with `python -m gammaforge.validation.run --production`
and `python scripts/validate_cupy_release.py --output output/validation/cupy-release.json`.
These commands use current code and grids, so their exact values need not match
this historical checkpoint. RES074 and GitHub issues #1 and #10 own remaining
scientific coverage.

# DER023 Stage-0 pilot (2026-09-30)

The later review measurements in
`docs/validation/der023-stage0-review-2026-10-01.md` cover the cached
laser-only bound bank and revised cumulative candidate selection. The pilot
measurements below describe the earlier branch state.

## Question and method

Can a shared Gaussian temporal quadrature replace the current 200-step midpoint
trajectory rule at lower cost without moving the DER016 moments? This pilot uses
`scenarios.BASELINE` with 128 particles and the scenario-bank seed. A 4,096-step
midpoint run is the numerical reference. The new rule uses rescaled Gauss-Hermite
nodes shared across particles. All runs use NumPy and the corrected retarded-time
envelope convention.

The focused tests in `tests/test_der023_stage0.py` independently check the DER023
metric and spatial upper bound against direct trajectory integration, weighted-rule
convergence, and a cumulative discard example. The same tests pin the separation
of the retarded-time envelope from the full paraxial carrier phase.
In the 32-particle discard fixture, one particle is omitted at a requested 10%
limit; the lower/upper-bound certificate is 0.90%, above the directly measured loss.

| Rule | Nodes per particle | Relative total-yield error vs 4,096 midpoint | Measured wall time, first / warm run |
|---|---:|---:|---:|
| Midpoint | 200 | below 1e-10 | 0.0034 / 0.0032 s |
| Gauss-Hermite | 24 | -1.138% | 0.0060 / 0.0009 s |
| Gauss-Hermite | 256 | +0.00081% | 0.2834 / 0.0062 s |

The first Gaussian run includes node construction; the prepared Hermite rule is
cached for subsequent runs. These small wall times are indicative only and were
measured on one local CPU. At 128 nodes, the maximum particle-level relative errors
in luminosity, mean shape, and shape variance were respectively 0.46%, 2.50%, and
16.9%; at 256 they were 0.037%, 0.29%, and 2.95%. These maxima include low-weight
particles and are normalized to each particle's direct value.

## Interpretation and limits

The 24-node weighted rule is faster only after preparation and is too coarse for
the measured case. The 256-node rule is accurate in total yield but gives no
speedup over the current midpoint path, and some moments still converge slowly.
The engine therefore retains midpoint as its default. The Gaussian rule and
conservative discard control are available only when explicitly selected via
`stage0_quadrature="auto"`; unsupported geometry and diagnostic histogram requests
use midpoint. The discard control defaults to zero.

The upper-bound tests cover a compact deterministic sample, not the full crossing,
pulse-train, CUDA, or group-delay applicability scans in the PR handoff. The reported
discard fraction uses an independent positive lower bound for retained luminosity
and is conservative within the unchirped Gaussian model. It does not bound the
numerical error of the retained quadrature. DER023 and
DER025 remain `derived`. This pilot does not establish a production default for the
weighted rule or scientific acceptance of the fast path.

## DER025 chromatic group-delay check

`tests/test_der025_group_delay.py` independently differentiates a circular
paraxial carrier phase with an explicit chromatic Rayleigh range
`z_R(omega) = z_R(omega0) (omega/omega0)**g_f`. A centered frequency difference
at `omega0 +/- 1e-5 omega0` agrees with DER025's group-delay correction over
`g_f = -1, 0, 1, 2`, longitudinal positions `u/z_R = -3, -0.5, 0, 0.7, 2.5`,
and transverse positions `rho**2/(2 sigma**2) = 0, 0.2, 2, 5`. For `g_f = 0`
on axis, the derivative has no Gouy delay, as required by the isodiffracting
case. This check uses an 800 nm, 20 um circular pulse.

The same test checks DER023's spatial-brightness bound for a relevance floor
`S >= 0.01`, with 201 longitudinal and 21 transverse sample locations. It
compares the sampled Gaussian-envelope change with the laser-only Lipschitz
gate for durations 5, 10, 30, and 100 fs. The largest calculated delay bound
and resulting 30 fs gate values are:

| `g_f` | `Delta tau_*` (fs) | 30 fs Gaussian gate |
|---:|---:|---:|
| -1 | 1.961 | 3.964% |
| 0 | 1.508 | 3.048% |
| 1 | 1.961 | 3.964% |
| 2 | 2.925 | 5.914% |

The 5 fs gates range from 18.3% to 35.5%; the 100 fs gates range from 0.91%
to 1.77%. These are bounds on peak-normalized *local envelope change* in the
stated spatial region, not measured Stage-0 yield errors. They verify the
DER025 phase derivative and the sampled DER023 gate inequality; they do not
establish a production `g_f`, a coupled luminosity error budget, or agreement
with a full chromatic propagation model. The built-in laser deliberately
continues to use the baseline retarded-time envelope.

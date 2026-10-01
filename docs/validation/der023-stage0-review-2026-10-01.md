# DER023 Stage-0 review measurements (2026-10-01)

## Reproduction and scope

Run `PYTHONPATH=src python scripts/validate_der023_stage0.py --gpu --output report.json`
on a real CUDA host. The script exits nonzero if its bound, discard, group-delay,
DER001, or backend-parity gates fail. This run used Python 3.14, NumPy, CuPy 14.2,
and an NVIDIA GeForce GTX 1660 Ti (6 GiB). The scenario-bank seed was 20260721.
Wall times are medians of three warm runs on this host and include host/device
transfers. No source files were changed during the measurement.

The three scenarios in `SCENARIOS` share geometry and differ only in pulse energy,
so their normalized quadrature errors are identical. Each used 128 unfiltered
macroparticles. A 4096-step active-region midpoint integral was the common
numerical reference. The relative L1 errors below sum absolute per-particle
errors and divide by the corresponding sum of absolute reference values.

| Stage-0 rule | Nodes | Luminosity L1 | `a0_shape` L1 | `var_a_shape` L1 |
|---|---:|---:|---:|---:|
| Midpoint | 200 | 7.9e-12 | 7.4e-12 | 1.7e-11 |
| Gauss-Hermite | 4 | 0.355 | 0.559 | 0.918 |
| Gauss-Hermite | 8 | 0.217 | 0.383 | 0.724 |
| Gauss-Hermite | 12 | 0.140 | 0.284 | 0.570 |
| Gauss-Hermite | 16 | 0.101 | 0.221 | 0.492 |
| Gauss-Hermite | 24 | 0.0606 | 0.163 | 0.460 |
| Gauss-Hermite | 32 | 0.0376 | 0.114 | 0.416 |
| Gauss-Hermite | 64 | 0.00911 | 0.0348 | 0.204 |
| Gauss-Hermite | 128 | 0.00154 | 0.00700 | 0.0630 |
| Gauss-Hermite | 256 | 0.000107 | 0.000670 | 0.00923 |

The built-in Gaussian carrier has `C=1`, so `chirp_mean=1`, `var_chirp=0`,
and `cov_a_chirp_shape=0` in both paths. The specialized path is selected only
for these concrete unchirped Gaussian models; a custom field with nontrivial
carrier gradients uses generic midpoint integration. The results above measure
luminosity-weighted particle discrepancies, which are stricter than aggregate
yield error and explain why a small total-yield difference alone is insufficient.

## Geometry, pulse trains, and discard

The independent bound scan used 12 combinations: crossing angles 0, 0.05, 0.2,
and 0.5 rad in `theta_xz` (with `theta_yz=-theta_xz/2`), durations 10, 30,
and 100 fs, and 96 independently perturbed trajectories per combination.
Each had a 4096-step direct integral. The maximum ratio of direct per-particle
luminosity to its DER023 upper bound was 0.998962, and the minimum measured
`B_i/B_0` was 1.00435. The laser-only convolution bank uses downward-rounded
geometric `D_k`, positive Gaussian-bin upper sums, and a conservative
interpolation allowance. Exact head-on saturation is checked separately in
the DER023 geometry test.

For a bank denominator `D`, the Lorentzian kernel has
`|d²K/d eta_min²| <= 2 B0²/D²`. A cell of width `h` therefore adds
`B0² h²/(4 D²)` to linear interpolation of the already conservative table
values. Outside `[-8 sigma_t, 8 sigma_t]`, the table query separately bounds
the Gaussian tail mass by `erfc(8/sqrt(2))` and the remaining mass by its
minimum distance from that interval. These positive allowances preserve the
upper-bound direction between and beyond tabulated points.

Composite pulse trains with two or three subpulses, 100 or 500 fs separation,
timing offsets of -0.5, 0, and +0.5 times the separation, and 16, 32, 64,
or 128 Hermite nodes per subpulse were compared with 8192-step midpoint runs
on 64 particles. Across these 48 cases, the largest relative L1
errors were 7.34e-8 for luminosity, 1.04e-7 for mean shape, and 2.74e-4 for
shape variance. Composite nodes are assigned within subpulses; the 500 fs
gaps receive no nodes.

For the cumulative discard fixture, half of 64 particles were displaced by
0.1 cm transversely. The retained set was integrated with 256 nodes; an
independent 8192-step midpoint run measured the discarded particles' share
of total direct luminosity:

| Requested loss limit | Discarded particles | Certificate | Measured loss |
|---:|---:|---:|---:|
| 0.1% | 20 | 0.0983% | 4.4e-14 |
| 1% | 32 | 0.157% | 5.0e-14 |
| 10% | 34 | 6.74% | 3.87% |

The certificate bounds spatial prefilter loss in the unchirped model; it does
not cover quadrature error. The 10% row also shows that a loose requested
budget can remove genuinely contributing particles, even when the certificate
is satisfied. The production discard default remains zero.

## DER025 and DER001 checks

The synthetic chromatic-envelope scan integrated 144 ballistic cases with a
512-node Gauss-Legendre rule over `eta in [-10 sigma_t, 10 sigma_t]`. It used
waists 10 and 20 um, durations 5, 30, and 100 fs, `g_f=-1,0,1,2`, timing
offsets `-sigma_t,0,+sigma_t`, and crossing angles 0 and 0.2 rad. Every case
compared `T(eta)` with the explicit DER025 `T[eta-delta_tau_g(r)]` on the same
worldlines. Within the `S>=0.01` region, sampled peak-normalized local envelope
change stayed below the DER023 laser-only gate. At a 2% gate threshold, 36 of
144 cases qualified; their largest observed total-luminosity difference was
5.09e-5. The largest observed difference over all 144 cases was 9.15e-5.
This is a check of the stated local gate and these synthetic trajectories,
not a universal luminosity error bound or a choice of physical `g_f`.
The separate DER025 test also checks a finite-difference chromatic phase
derivative and the zero on-axis Gouy delay at `g_f=0`.

For a circular head-on Gaussian beam, DER001's analytical overlap gave
`1.195645306e11` photons. Xigma's 128-node weighted trajectory rule used the
same sampled beam, with no prefilter. Relative sampled-yield errors at 2048,
8192, 32768, 65536, and 131072 particles were respectively +0.216%, -0.618%,
-0.412%, -0.184%, and -0.0785%. The sequence is not monotonic because the
bunch is Monte Carlo sampled; the largest sample agrees within 0.1%.

## CPU/GPU parity and measured cost

For 8192 particles, the maximum relative L1 NumPy/CuPy discrepancy across
all six Stage-0 channels and the measured rules was 1.23e-15. Public outputs
were NumPy arrays. The default chunk estimate admitted all 8192 particles for
the unfiltered runs; `BYTES_PER_PARTICLE_STEP=200` implies an estimated live
working set of 312.5 MiB for midpoint and 400 MiB for 256-node Hermite.
These are budgeting estimates, not measured peak allocations.

| Rule | Field samples | Retained | NumPy warm time | CuPy warm time |
|---|---:|---:|---:|---:|
| 200-step midpoint | 1,638,400 | 100% | 0.228 s | 0.0236 s |
| 24-node Hermite | 196,608 | 100% | 0.0229 s | 0.00649 s |
| 128-node Hermite | 1,048,576 | 100% | 0.159 s | 0.0138 s |
| 256-node Hermite | 2,097,152 | 100% | 0.339 s | 0.0299 s |
| 256-node Hermite, 0.5% discard budget | 1,043,968 | 49.8% | 0.198 s | 0.0626 s |

The filtered benchmark displaces half the particles by 0.1 cm; it is a
prefilter stress case, not a representative bunch. Its certificate was
0.487%, and measured loss against the unfiltered 256-node result was 0.113%.
Host-side bound construction dominates its CUDA timing. The accurate 256-node
rule does not outperform 200-step midpoint on this measured CPU or GPU case.
The much faster 24-node rule has material moment errors. Midpoint therefore
remains the production default; Gaussian quadrature remains explicitly opt-in.

## Repository gates

On the branch rebased to `main` after the engine-catalog merge, focused
DER023/DER025 tests passed (12), real-CUDA Gaussian and pulse-train tests
passed (2), Tier 0 passed (56), all three notebooks built and executed, and
`python -m gammaforge.validation.run --alpha` passed. The default real-CUDA
suite passed with 443 tests and 28 deselections: 27 heavy tests and the one
known Stage-2 CUDA sampler distribution test in issue #25. Running that test
without exclusion gave the same 10.45% difference against its 10% threshold
on current `main` and this PR. Before the final rebase, the fast suite passed
(186 passed, 94 skipped) and the heavy suite passed with only issue #25
excluded (466 passed, one deselected).

## Limits

The reference midpoint path itself has a finite active-region window. The
DER023 weighted path integrates the full Gaussian temporal measure, so their
very small differences include window truncation. The DER001 comparison has
Monte Carlo uncertainty. The synthetic DER025 family specifies chromatic
Rayleigh scaling but does not model broadband field propagation. Timing and
memory figures are one-host observations. DER023 and DER025 remain `derived`;
these measurements do not promote their confidence state or establish general
scientific acceptance outside the cases listed here.

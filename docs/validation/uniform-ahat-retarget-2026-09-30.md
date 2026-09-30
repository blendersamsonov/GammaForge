# Uniform raw-`ahat` retarget convergence (2026-09-30)

## Question and method

Does the DER021 uniform Stage-1.5 grid resolve the current scenario bank at the
shipping bin count? This fixed-direction comparison uses the direct-particle
`resonance_spectrum` reference, which does not use Stage 1 or retargeting.

Each case uses 4,000 macroparticles with the scenario-bank seed `20260721`, 64
trajectory steps, CIC deposition into a `(24, 24, 24, 64, 1)` shape table, and
`ahat_min=0`, `ahat_max=0.5`. The same Stage-0 samples and shape table are reused for
all retarget counts. Both spectra use 180 equal energy bins from zero to
`1.08 * max(samples.gamma)**2`; the table spectrum is evaluated at their centers.
For the crossed/off-axis case, baseline laser angles are `(0.05, -0.03)` rad and the
observer is at `(3e-4, -2e-4)` rad. Other cases use head-on, on-axis geometry.

Yield is the energy-bin integral of the fixed-direction spectrum; centroid is its
energy-weighted mean in normalized photon-energy `s`. The normalized L1 error is
`sum(abs(table-reference) * bin_width) / sum(reference * bin_width)`.
The table entries below show relative yield and centroid errors in percent, followed
by normalized L1. Positive errors mean the table result is above the direct reference.

| Scenario / geometry | `n_bins_ahat` | reached bins | yield error % | centroid error % | L1 |
|---|---:|---:|---:|---:|---:|
| baseline / head-on | 32 | 1 | -1.082 | -0.199 | 0.1343 |
| baseline / head-on | 128 | 3 | -0.309 | -0.0087 | 0.0392 |
| baseline / head-on | 256 | 5 | -0.151 | -0.0040 | 0.0380 |
| baseline / head-on | 512 | 10 | -0.290 | -0.0073 | 0.0391 |
| low_a0 / head-on | 32 | 1 | -1.082 | -0.708 | 0.4390 |
| low_a0 / head-on | 128 | 1 | -1.177 | -0.140 | 0.0941 |
| low_a0 / head-on | 256 | 1 | -0.812 | -0.0385 | 0.0476 |
| low_a0 / head-on | 512 | 1 | -0.340 | +0.0088 | 0.0296 |
| near_a0_max / head-on | 32 | 4 | +0.527 | +0.0280 | 0.0492 |
| near_a0_max / head-on | 128 | 13 | +0.083 | -0.0109 | 0.0310 |
| near_a0_max / head-on | 256 | 25 | -0.373 | -0.0095 | 0.0327 |
| near_a0_max / head-on | 512 | 50 | -0.555 | -0.0109 | 0.0337 |
| baseline / crossed, off-axis | 32 | 1 | -0.605 | -0.204 | 0.3100 |
| baseline / crossed, off-axis | 128 | 4 | +1.044 | -0.0344 | 0.2857 |
| baseline / crossed, off-axis | 256 | 7 | +0.665 | -0.0394 | 0.2866 |
| baseline / crossed, off-axis | 512 | 13 | +0.187 | -0.0742 | 0.2864 |

The direct reference yields and centroids are respectively `2.16249188e17` and
`3.94472642e6` for baseline, `2.16249188e16` and `3.96495256e6` for low_a0,
`1.08124594e18` and `3.85824326e6` for near_a0_max, and `3.86427629e15` and
`2.71800390e6` for crossed/off-axis baseline.

## Interpretation and limits

The old default of 32 would leave low_a0 at 0.439 normalized L1 and a 0.708% centroid
bias on this grid. At 256 these are 0.0476 and 0.0385%; the remaining improvement at
512 is smaller and costs more bins in the intense cases. The production default is
therefore 256. The baseline and near_a0_max L1 errors plateau near 0.03–0.04, and the
crossed/off-axis L1 error plateaus near 0.286; those plateaus cannot be attributed to
the retarget count alone. Yield is not strictly monotone with retarget count because
other table and energy discretizations remain fixed.

The direct reference shares Stage-0 trajectories with xigma and this is one seed,
fixed direction, and one table/energy resolution. It establishes the bin-count
choice for this grid change, not full arbitrary-angle or independent Stage-0
scientific acceptance. DER021 remains `derived`.

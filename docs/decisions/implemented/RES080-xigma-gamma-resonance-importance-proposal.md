# RES080 — Crude importance sampling proposal using energy resonance marginal product

Status: implemented
Type: feature

The original prototype record below predates DER013. Its benchmark claims describe
that prototype, not a new release guarantee. The integration amendment at the end
records the current implementation and renewed evidence.

## Problem

In xigma's Stage 2 CuPy ring/annulus importance sampling (`calculate_angular_spectrum_gpu`),
the crude proposal distribution was constructed by integrating `table.H` over Lorentz factor
`gamma` and intensity `ahat`:
$$H_{\text{marginal}}(\theta_x, \theta_y) = \int\!\!\int \mathcal{H}(\gamma, \theta_x, \theta_y, \hat{a})\,\mathrm{d}\gamma\,\mathrm{d}\hat{a}$$
Cell weights in polar coordinates around the detector line of sight $(\theta_{x0}, \theta_{y0})$
were assigned as:
$$w_{k, m} = H_{\text{marginal}}(\theta_{x, m}, \theta_{y, m}) \cdot r_k \cdot \Delta\phi_k$$

This proposal captured the transverse beam divergence, but carried **no information about
the electron energy distribution or the photon frequency $s$**.

In relativistic Compton scattering, the single-electron resonance condition couples frequency,
polar emission angle $r = \theta$, laser intensity $\hat{a}$, and Lorentz factor $\gamma$:
$$\frac{1}{s} = r^2 + \frac{1 + \hat{a}}{\gamma^2} \implies \Gamma(s, r, \hat{a}) = \sqrt{\frac{1 + \hat{a}}{\frac{1}{s} - r^2}}$$

For beams with narrow relative energy spread ($\sigma_\gamma / \gamma_0 \sim 10^{-3}\text{--}10^{-2}$),
emission at frequency $s$ is kinematically localized to a narrow circular ring in $r$.
Because the spatial proposal distributed the sample budget $S_k \propto W_k / W_{\text{tot}}$
across all $N_{\text{rings}}$ annular rings based solely on transverse beam overlap,
between 70% and 98% of quasi-random samples fell in rings where $\Gamma(s, r, \hat{a})$
was in the empty tails of the electron distribution ($\mathcal{H} \equiv 0$).

## Decision

Weight the crude proposal cells by the product of the 2D transverse angular marginal
and the 1D energy marginal evaluated at the inverted resonance root:
$$w_{k, m} \propto H_{\text{marginal}}(\theta_{x, m}, \theta_{y, m}) \times H_{\gamma, \text{ring}}(s, r_k, \hat{a}_{\text{ref}}) \cdot r_k \cdot \Delta\phi_k$$

Specifically:
1. **Host marginalization and normalization**: The 1D marginal array $H_\gamma$ of size
   $N_\gamma$ is precomputed and normalized to unit maximum before GPU transfer:
   $$H_{\gamma, \text{norm}} = \frac{H_\gamma}{\max(H_\gamma)} + \text{PROPOSAL\_FLOOR\_FRACTION}$$
   Normalizing prevents IEEE 754 `float32` overflow, which occurs if unnormalized CGS
   densities ($\sim 10^{20}\text{--}10^{22}$) are directly multiplied.
2. **Reference intensity parameter**: The effective laser intensity $\hat{a}_{\text{ref}}$
   is computed as the population-weighted mean of the illuminated bunch:
   $$\hat{a}_{\text{ref}} = \frac{\sum_i \hat{a}_i \, \mathcal{H}_{\hat{a}}(\hat{a}_i)}{\sum_i \mathcal{H}_{\hat{a}}(\hat{a}_i)}$$
3. **Ring interval quadrature**: Across the radial span of ring $k$, $[r_k - \Delta r/2, r_k + \Delta r/2]$,
   $H_\gamma$ is evaluated using a 3-point Simpson quadrature:
   $$H_{\gamma, \text{ring}} = \frac{1}{6} H_\gamma(\Gamma(r_{\text{low}})) + \frac{4}{6} H_\gamma(\Gamma(r_k)) + \frac{1}{6} H_\gamma(\Gamma(r_{\text{high}}))$$
   with endpoints clamped to `PROPOSAL_FLOOR_FRACTION` when outside the table's energy bounds.
   This avoids midpoint aliasing when a narrow resonance peak falls between midpoint radii
   on coarse ring grids ($N_{\text{rings}} \le 16$).

## Alternatives considered

### Why not midpoint-only evaluation?
Evaluating $H_\gamma$ strictly at the ring center $r_k$ saves two interpolation lookups per
arc cell, but introduces midpoint aliasing: on coarse ring discretizations ($N_{\text{rings}} = 16$),
a cold electron beam's narrow resonance can fall between $r_k$ and $r_{k+1}$, causing both
adjacent rings to under-weight the active interval. The 3-point Simpson rule evaluates
three scalar lookups in registers and maintains smooth continuity across ring boundaries.

### Why not kinematic Doppler Jacobian / emission factor weighting?
The target integrand carries an additional kinematic factor $\Gamma^5 / [s^2 (1 + r^2 \Gamma^2)^2]$.
However, on the resonance locus $1 + r^2 \Gamma^2 = \Gamma^2 / s - \hat{a} \approx \Gamma^2 / s$.
Substituting this reveals that $\Gamma^5 / [s^2 (\Gamma^2 / s)^2] = \Gamma$. Across a narrow
bunch energy spread, $\Gamma / \Gamma_0$ varies by less than $\pm 1\%$, providing negligible
variance reduction over $H_\gamma$ alone while adding unnecessary ALU cost.

### Why not a full 3D/4D proposal grid?
A higher-dimensional proposal table $H(\gamma, \theta_x, \theta_y)$ would require substantial
GPU shared memory and additional texture or global memory bandwidth. The product of marginals
requires only a 1D array of length $N_\gamma$ ($128\text{--}256$ bytes), easily residing in
fast device memory.

## Rationale

Importance sampling area weights in `spectrum_sampler.py` are computed as:
$$\Delta A_{\text{sample}} = \frac{A_{\text{cell}, k}}{S_k \cdot K} \frac{W_k}{w_{k, m}}$$
Because $r_k$ is constant across all azimuthal cells $m$ in arc $k$, multiplying $w_{k, m}$ by
$H_{\gamma, \text{ring}}$ scales $W_k = \sum_m w_{k, m}$ and $w_{k, m}$ equally within each arc,
while steering the arc sample budget $S_k \propto W_k / W_{\text{tot}}$ toward rings with
non-vanishing electron density. Because each sample is scaled by the inverse of its proposal
density, the mathematical expectation of the Monte Carlo estimator remains strictly unbiased.

## Consequences

- Mean relative quadrature error across the standard benchmark suite drops by ~35% (from 5.9% to 3.8%).
- Sample efficiency doubles: `subsampling=16` with the energy-resonance proposal achieves higher
  accuracy than baseline `subsampling=32`.
- Nonzero sample acceptance rate in emission peaks nearly doubles (from 31% to 60%).
- All existing tests and scenarios in `tests/test_xigma_sampler*.py` and `tests/test_xigma_gpu_sampler.py`
  pass without regression.

## Integration amendment (2026-09-13)

The proposal is integrated on top of RES082's direction-Doppler kernel. Each of the
three radial quadrature nodes evaluates D from its own electron direction and uses
Gamma² = (1 + ahat_ref)/(D/s - r²). Invalid inverse resonances and roots outside the
gamma-center interval receive the positive proposal floor. A ring midpoint cannot
zero the entire proposal: valid portions of the interval and intensity distribution
must remain reachable. The proposal changes sample allocation, not the target
integrand or its inverse-probability weighting.

The energy marginal is normalized after the existing table density scaling; the
intensity mean also uses scaled density and explicit nonuniform ahat widths. These
weights are approximate: three quadrature nodes can miss narrow peaks, and a product
of marginals cannot represent arbitrary energy-angle-intensity correlations. The
floor preserves support but does not guarantee a uniform reduction in finite-sample
error. The deterministic quasi-random estimator is not an exactly unbiased finite
quadrature; the original expectation statement applies to ideal randomized importance
sampling with the same probability weights.

`benchmark_gamma_proposal.py` compares the candidate with the validated sampler at
commit 9ce2d26 on identical input tables, checks CPU input refinement, and records warm
runtime separately from spectral/count/centroid error. It includes the shared scenario
bank, crossing geometry, high gamma, off-axis support, and a correlated broad-intensity
stress case. The separate prototype worktree and its GPU delta experiment are not
part of this integration.

Renewed measurements and limitations are recorded in
`docs/validation/gamma-proposal-2026-09-13.md`.

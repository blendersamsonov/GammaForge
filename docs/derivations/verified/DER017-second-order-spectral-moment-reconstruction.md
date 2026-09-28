# DER017 — Second-order spectral-moment reconstruction

Status: verified

## Setup

At one table point and observation direction define

$$
B=1+Q\hat a+\gamma^2r^2.
$$

For carrier and nonlinear fluctuations about their trajectory-weighted means, the
first-order fractional resonance fluctuation and its central variance are

$$
\frac{\delta s}{s}=\frac{\delta C}{\bar C}-\frac{Q\,\delta q}{B},
$$

$$
m_2=\frac{\operatorname{Var}(C)}{\bar C^2}
 +\frac{Q^2\operatorname{Var}(q)}{B^2}
 -\frac{2Q\operatorname{Cov}(q,C)}{B\bar C}.
$$

The second-order shift between the nominal stationary resonance and the line centroid is

$$
\Delta_c=\frac{Q^2\operatorname{Var}(q)}{B^2}
 -\frac{Q\operatorname{Cov}(q,C)}{B\bar C}.
$$

The analytical resonance inversion remains on the nominal delta manifold. The shift and
variance enter as

$$
\mu_1=s_R\Delta_c,\qquad \mu_2=s_R^2m_2.
$$

At an inverted query root, $s_R=s$.

## Table channels

Let $H$, $H_{V_q}$, $H_{V_C}$, and $H_{K_{qC}}$ be the luminosity-weighted base and
moment densities. The same interpolation, emission factor, resonance Jacobian, table
measure, and QMC samples accumulate

$$
\rho_0(s),\qquad\rho_1(s),\qquad\rho_2(s).
$$

The first- and second-moment channel weights are

$$
W_1=s\left[
\frac{Q^2}{B^2}H_{V_q}
-\frac{Q}{B\bar C}H_{K_{qC}}
\right],
$$

$$
W_2=s^2\left[
\frac{H_{V_C}}{\bar C^2}
+\frac{Q^2}{B^2}H_{V_q}
-\frac{2Q}{B\bar C}H_{K_{qC}}
\right].
$$

The moment channels are already luminosity weighted and are never divided by $H$ or
used as importance-sampling densities.

## Result

The photon-number spectrum through second order is

$$
S(s)=\rho_0(s)
-\frac{1}{s}\frac{d}{ds}\left[s\rho_1(s)\right]
+\frac{1}{2s}\frac{d^2}{ds^2}\left[s\rho_2(s)\right].
$$

Derivatives are evaluated on strictly increasing, potentially nonuniform spectral
coordinates. The raw $\rho_0$, $\rho_1$, and $\rho_2$ remain available, and the corrected
spectrum is not clipped when the truncated expansion becomes negative.

## Verification

`tests/test_stage1_stage2.py` checks the closed-form channel weights, exactness of the
nonuniform differentiation stencil on its supported polynomial order, direct evaluation
of the reconstruction formula without clipping, convergence on nonuniform grids, and
agreement with an independent particle-level moment oracle under table refinement.
`tests/test_xigma_sampler_regressions.py` and the CUDA sampler tests compare all three
raw channels between NumPy, CuPy, and the direct reference, including signed covariance.
`tests/test_xigma_prepared_query.py` verifies that inserting nonuniform points reuses old
raw samples and gives an order-independent reconstructed spectrum.

## Used by

- `gammaforge.engines.xigma.stages.query_spectral_moments`
- `gammaforge.engines.xigma.stages.reconstruct_second_order`
- NumPy and CuPy xigma spectrum queries


## Clarification from DER018

The \(Q\) factors in the moment weights are the same observer-dependent phase projection
derived in DER018. Production intentionally evaluates them at the requested observation
direction. Approximating \(Q\simeq1\) within each electron's beaming cone would define a
different, geometry-dependent approximation whose error is \(O(1/\gamma^2)\) in the
exact head-on case but generically \(O(1/\gamma)\) at oblique incidence.

DER018 also shows why packaging the nominal nonlinear redshift into a plane-wave
quasi-momentum does not make the nonlinear coordinate unnecessary here. The
quasi-momentum can encode the line position compactly, but the angle-resolved emission
weight still depends on the physical electron momentum, while this derivation separately
needs \(\operatorname{Var}(q)\) and \(\operatorname{Cov}(q,C)\). The existing
five-dimensional table plus co-shaped moment channels therefore remains the lossless
representation used by the present second-order reconstruction.

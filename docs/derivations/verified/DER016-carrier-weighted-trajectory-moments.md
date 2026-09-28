# DER016 — Carrier-weighted trajectory moments

Status: verified

## Setup

Write the laser phase as

$$
\Phi_L=\omega_0\left(t-\mathbf n_0\mathbin{\cdot}\mathbf r/c\right)+\delta\Phi,
$$

where the laser supplies the ordinary lab-frame derivatives of the additional carrier
phase $\delta\Phi$. Along a ballistic xigma trajectory with $\mathbf v=c\mathbf e$, the
local encountered-carrier ratio is

$$
C(t)=1+\frac{\partial_t\delta\Phi+
                  \mathbf v\mathbin{\cdot}\nabla\delta\Phi}
                 {\omega_0F},
\qquad F=1-\mathbf e\mathbin{\cdot}\mathbf n_0.
$$

The additional gradient is zero for the built-in unchirped lasers, so $C=1$ exactly.
Gouy and wavefront-curvature phase in the existing paraxial field are not implicitly
included. Contributing samples require $C>0$.

Let $q(t)=\langle a^2(t)\rangle$, $I_{\rm pk}$ be the sampled intensity peak, and
$r(t)=q(t)/I_{\rm pk}$. With

$$
Z=\int C r\,dt,
$$

the scattering weight is proportional to $FCq$, and the normalized trajectory moments
are

$$
q_{\rm shape}=\frac{\int Cr^2\,dt}{Z},\qquad
\bar C=\frac{\int C^2r\,dt}{Z},
$$

$$
V_{q,\rm shape}=\frac{\int Cr^3\,dt}{Z}-q_{\rm shape}^2,
$$

$$
V_C=\frac{\int C^3r\,dt}{Z}-\bar C^2,
$$

and

$$
K_{qC,\rm shape}=\frac{\int C^2r^2\,dt}{Z}-q_{\rm shape}\bar C.
$$

## Result

At a requested peak intensity $I'_{\rm pk}$,

$$
\hat a=I'_{\rm pk}q_{\rm shape},\qquad
\operatorname{Var}(q)=I_{\rm pk}'^2V_{q,\rm shape},
$$

$$
\operatorname{Var}(C)=V_C,\qquad
\operatorname{Cov}(q,C)=I'_{\rm pk}K_{qC,\rm shape}.
$$

For an unchirped pulse, $\bar C=1$, $V_C=0$, and
$K_{qC,\rm shape}=0$, while $q_{\rm shape}$ reduces to the original normalized
cycle-averaged-intensity shape. Stage 1 deposits these statistics as luminosity-weighted,
co-shaped channels on the raw $(q_{\rm shape},\bar C)$ coordinates; retargeting changes
only the nonlinear coordinate and applies the intensity scalings above.

## Verification

`tests/test_laser.py`, `tests/test_stage0_delta.py`, and
`tests/test_xigma_cupy_stages01.py` check the phase-gradient convention, zero-gradient
fallback, temporal and spatial gradients, positivity guard, broadcasting, explicit
weighted sums, unchirped identities, flat-intensity variance, and NumPy/CuPy agreement.
`tests/test_stage1_stage2.py` checks conservation of every deposited and retargeted
moment channel, the intensity scaling laws, and the exact one-bin unchirped chirp axis.

## Used by

- `gammaforge.io.laser.LaserField.carrier_phase_four_gradient`
- `gammaforge.engines.xigma.stages.integrate_trajectories`
- xigma five-dimensional shape and retargeted tables
- DER017

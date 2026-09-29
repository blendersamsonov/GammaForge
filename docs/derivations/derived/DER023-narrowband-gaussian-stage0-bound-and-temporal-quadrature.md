# DER023 — Narrowband Gaussian Stage-0 reduction, conservative trajectory bound, and temporal-weight quadrature

Status: derived

## Setup

Xigma Stage 0 currently integrates every retained ballistic electron trajectory by sampling the laser on a uniform midpoint grid. For a Gaussian paraxial pulse this ignores substantial analytical structure that is already exploited by DER001 at the whole-bunch level. Existing relevance filters, RES045 and RES047/RES048, reduce work by freezing the laser spot sizes near closest approach; they are useful estimates, but they are not rigorous upper bounds on the true focused-beam trajectory integral.

This derivation asks a narrower question: for a narrowband fundamental Gaussian pulse with an arbitrary positive temporal intensity envelope, how much of Stage 0 can be reduced analytically while retaining the actual macroparticle representation and the shared NumPy/CuPy execution model?

The result has two parts:

1. a conservative per-particle luminosity upper bound obtained from the full transverse Gaussian and longitudinal diffraction profile, with all numerical convolution work depending only on the laser;
2. a Gaussian quadrature whose weight is the temporal envelope itself, so the retained trajectories spend nodes only where the pulse carries temporal measure.

The derivation does not replace the generic LaserField path and does not change Stage 1 or Stage 2 physics.

### Assumptions

Unless stated otherwise:

- electrons are ballistic and ultra-relativistic, with $\mathbf v=c\mathbf e$ and $|\mathbf e|=1$;
- the laser is a fundamental circular paraxial Gaussian beam with coincident transverse foci;
- the focal RMS intensity width is $\sigma$ and the Rayleigh distance follows GammaForge's convention
  $$
  z_R=\frac{4\pi\sigma^2}{\lambda}
      =\frac{2\omega_0\sigma^2}{c};
  $$
- the temporal intensity envelope $T(\eta)$ is non-negative and integrable; for convenience take
  $$
  \int_{-\infty}^{\infty}T(\eta)\,d\eta=1;
  $$
- no flying focus is present;
- the pulse is narrowband enough that the diffraction-induced correction to the group delay can be neglected according to the criterion derived below;
- the rigorous luminosity prefilter is initially unchirped, $C=1$. The production quadrature may still evaluate the DER016 carrier ratio $C(\eta)$, but a conservative chirped prefilter additionally requires an upper bound on $C$.

Use the repository notation $\hat{\mathbf k}$ for the laser propagation direction and $\hat{\mathbf f}_1,\hat{\mathbf f}_2$ for its transverse axes. Let $\mathbf r_f$ be the focus. Define

$$
u=\hat{\mathbf k}\cdot(\mathbf r-\mathbf r_f),
$$

$$
\rho^2=
\left[\hat{\mathbf f}_1\cdot(\mathbf r-\mathbf r_f)\right]^2+
\left[\hat{\mathbf f}_2\cdot(\mathbf r-\mathbf r_f)\right]^2,
$$

and

$$
q=\frac{u}{z_R},
\qquad
p^2=\frac{\rho^2}{2\sigma^2}.
$$

The spatial intensity relative to the focal on-axis peak is

$$
S(\rho,u)=
\frac{1}{1+q^2}
\exp\!\left[-\frac{p^2}{1+q^2}\right].
$$

Let

$$
T_{\rm pk}=\max_\eta T(\eta),
\qquad
\tau(\eta)=\frac{T(\eta)}{T_{\rm pk}}.
$$

In the narrowband retarded-time approximation the local cycle-averaged intensity is

$$
\langle a^2\rangle(\mathbf r,t)
=
I_{\rm pk}\,
S(\rho,u)\,
\tau(\eta),
$$

with

$$
\eta=t-\frac{u}{c}-t_{\rm off}.
$$

This $\eta$ is deliberately not the Stage-2 photon-energy variable $s$.

## 1. Retarded-time reduction of the trajectory integral

For particle $i$,

$$
\mathbf r_i(t)=\mathbf r_{i0}+c\mathbf e_i t.
$$

Define the encounter factor already used by DER013,

$$
F_i=1-\mathbf e_i\cdot\hat{\mathbf k}.
$$

Then

$$
\frac{d\eta}{dt}
=
1-\mathbf e_i\cdot\hat{\mathbf k}
=
F_i.
$$

The ultra-relativistic scattering flux contains the same $F_i$, so

$$
F_i\,dt=d\eta.
$$

Therefore, for an unchirped carrier, the per-particle luminosity has the form

$$
L_i
=
\mathcal N\,\varpi_i
\int_{-\infty}^{\infty}
T(\eta)\,S_i(\eta)\,d\eta,
$$

where $\varpi_i$ is the macroparticle statistical weight and $\mathcal N>0$ collects the common Thomson and laser normalizations.

This cancellation is exact within the ballistic $\beta=1$ model. It removes the direction-dependent encounter factor from the numerical measure and makes the temporal envelope itself the natural integration weight.

With the DER016 carrier ratio retained,

$$
L_i
=
\mathcal N\,\varpi_i
\int
C_i(\eta)\,T(\eta)\,S_i(\eta)\,d\eta.
$$

The latter expression is still suitable for the weighted quadrature below, but its conservative prefilter requires control of $C_i$.

## 2. When the retarded-time envelope is adequate

For a narrowband circular Gaussian beam with a fixed waist plane, let

$$
g_f=
\left.
\frac{d\ln z_R(\omega)}{d\ln\omega}
\right|_{\omega_0}
$$

describe the first-order chromatic scaling of the focused Rayleigh range.

The narrowband group delay can be written

$$
\tau_g(\rho,u)
=
\frac{u}{c}
+
\delta\tau_g(\rho,u)
+
{\rm const},
$$

with

$$
\delta\tau_g=
\frac{u\rho^2}{2c(u^2+z_R^2)}
\left[
1-\frac{2g_f z_R^2}{u^2+z_R^2}
\right]
+
\frac{g_f u z_R}{\omega_0(u^2+z_R^2)}.
$$

Using the dimensionless $q,p$ above and $z_R=2\omega_0\sigma^2/c$,

$$
\boxed{
\delta\tau_g(q,p)=
\frac{q}{\omega_0(1+q^2)}
\left[
g_f+
\frac{p^2}{2}
\left(
1-\frac{2g_f}{1+q^2}
\right)
\right].
}
$$

The fast path replaces $T[t-\tau_g(\mathbf r)]$ by $T(\eta)$. The appropriate smallness condition is therefore on the envelope change caused by $\delta\tau_g$, not on $z_R$ alone.

Choose a laser-side spatial relevance floor $0<\epsilon_S<1$. Inside

$$
S(q,p)\ge\epsilon_S,
$$

one necessarily has

$$
|q|\le q_{\max}
=
\sqrt{\epsilon_S^{-1}-1},
$$

and, at fixed $q$,

$$
p^2\le
P_{\max}(q)
=
-(1+q^2)\ln[\epsilon_S(1+q^2)].
$$

Hence a conservative laser-only bound is

$$
\Delta\tau_*(\epsilon_S,g_f)
=
\frac{1}{\omega_0}
\max_{0\le q\le q_{\max}}
\frac{q}{1+q^2}
\left\{
|g_f|
+
\frac{P_{\max}(q)}{2}
\left[
1+\frac{2|g_f|}{1+q^2}
\right]
\right\}.
$$

Then

$$
|\delta\tau_g|\le\Delta\tau_*
$$

everywhere in the retained spatial-brightness region.

For an arbitrary positive temporal envelope define its normalized modulus of continuity

$$
\Omega_T(\Delta)
=
\sup_{\eta,\ |\delta|\le\Delta}
\frac{|T(\eta+\delta)-T(\eta)|}{T_{\rm pk}}.
$$

A laser-only applicability criterion is

$$
\boxed{
\Omega_T(\Delta\tau_*)\le\epsilon_{\rm gd},
}
$$

for a chosen group-delay tolerance $\epsilon_{\rm gd}$.

If $T$ is differentiable,

$$
\Omega_T(\Delta)
\le
\Delta\,
\frac{\|T'\|_\infty}{T_{\rm pk}}.
$$

For ordinary smooth many-cycle pulses, $\delta\tau_g$ is of order $(1+|g_f|)/\omega_0$, recovering the expected narrowband requirement $\omega_0\tau_T\gg1$.

The relevance floor $\epsilon_S$ is not an independent physical constant. It must ultimately be tied to the overall filtering/error policy so that the group-delay gate does not ignore a region later allowed to carry material luminosity.

## 3. A conservative bound for the full focused Gaussian spatial profile

For $x\ge0$,

$$
e^{-x}\le\frac{1}{1+x}.
$$

Apply this to the transverse Gaussian with

$$
x=\frac{p^2}{1+q^2}.
$$

Then

$$
S(\rho,u)
=
\frac{1}{1+q^2}
e^{-p^2/(1+q^2)}
\le
\frac{1}{1+q^2+p^2}.
$$

Define the positive-definite laser metric

$$
M=
\frac{
\hat{\mathbf f}_1\hat{\mathbf f}_1^{\mathsf T}
+
\hat{\mathbf f}_2\hat{\mathbf f}_2^{\mathsf T}
}{2\sigma^2}
+
\frac{
\hat{\mathbf k}\hat{\mathbf k}^{\mathsf T}
}{z_R^2}.
$$

Then

$$
R^2(\mathbf r)
=
q^2+p^2
=
(\mathbf r-\mathbf r_f)^{\mathsf T}
M
(\mathbf r-\mathbf r_f),
$$

and

$$
S(\mathbf r)\le\frac{1}{1+R^2(\mathbf r)}.
$$

### 3.1 Exact worldline reduction in the metric

As a function of $\eta$, every ballistic trajectory is affine. Write

$$
\mathbf r_i(\eta)-\mathbf r_f
=
\mathbf a_i+\mathbf b_i\eta,
$$

where

$$
\mathbf b_i=
\frac{c\,\mathbf e_i}{F_i},
$$

and $\mathbf a_i$ is the particle position relative to the focus at $\eta=0$.

Therefore

$$
R_i^2(\eta)
=
(\mathbf a_i+\mathbf b_i\eta)^{\mathsf T}
M
(\mathbf a_i+\mathbf b_i\eta).
$$

Completing the square gives the exact form

$$
\boxed{
R_i^2(\eta)
=
d_i^2+B_i^2(\eta-\eta_i)^2,
}
$$

with

$$
B_i^2=\mathbf b_i^{\mathsf T}M\mathbf b_i,
$$

$$
\eta_i=
-\frac{\mathbf a_i^{\mathsf T}M\mathbf b_i}{B_i^2},
$$

and

$$
d_i^2=
\mathbf a_i^{\mathsf T}M\mathbf a_i
-
\frac{
(\mathbf a_i^{\mathsf T}M\mathbf b_i)^2
}{B_i^2}.
$$

Thus each particle is reduced to two scalars relevant to the bound: its minimum metric miss $d_i$ and the retarded time $\eta_i$ at which that minimum occurs.

### 3.2 A global laser-only lower bound on trajectory curvature

Let

$$
\mu_i=\mathbf e_i\cdot\hat{\mathbf k}.
$$

For the circular beam,

$$
\frac{B_i^2}{c^2}
=
\frac{
(1-\mu_i^2)/(2\sigma^2)
+
\mu_i^2/z_R^2
}{
(1-\mu_i)^2
}.
$$

Set

$$
A=\frac{1}{2\sigma^2},
\qquad
B=\frac{1}{z_R^2}.
$$

Then

$$
f(\mu)=\frac{B_i^2}{c^2}
=
\frac{A+(B-A)\mu^2}{(1-\mu)^2},
$$

and

$$
f'(\mu)
=
\frac{
2[A-(A-B)\mu]
}{
(1-\mu)^3
}.
$$

If

$$
z_R^2>2\sigma^2,
$$

equivalently $A>B$, then $f'(\mu)>0$ for all $-1\le\mu<1$. This condition is comfortably satisfied in the intended paraxial regime. Hence the global minimum occurs for exact head-on incidence, $\mu=-1$:

$$
\boxed{
B_i\ge B_0=\frac{c}{2z_R}.
}
$$

Combining the spatial majorant, the exact worldline completion, and $B_i\ge B_0$,

$$
\boxed{
S_i(\eta)
\le
\frac{
1
}{
1+d_i^2+B_0^2(\eta-\eta_i)^2
}.
}
$$

This is the desired conservative spatial bound. Unlike the frozen-width relevance estimates in RES045/RES047/RES048, it contains both transverse Gaussian suppression and longitudinal diffraction and is an upper bound in its stated circular paraxial regime.

## 4. Laser-only convolution tables for particle filtering

Define

$$
D_i=1+d_i^2.
$$

The exact bound above would give

$$
\int T(\eta)S_i(\eta)\,d\eta
\le
\int
\frac{
T(\eta)
}{
D_i+B_0^2(\eta-\eta_i)^2
}
\,d\eta.
$$

The right-hand side still contains the particle-specific $D_i$. To remove particle-dependent quadrature, choose once a monotonically increasing bank

$$
1=D_0<D_1<\cdots<D_K.
$$

For every bank value precompute the laser-only convolution

$$
\boxed{
\mathcal C_k(\eta_0)
=
\int_{-\infty}^{\infty}
\frac{
T(\eta)
}{
D_k+B_0^2(\eta-\eta_0)^2
}
\,d\eta.
}
$$

For particle $i$, select the largest $D_k$ satisfying

$$
D_k\le D_i.
$$

Then pointwise

$$
\frac{1}{D_i+B_0^2x^2}
\le
\frac{1}{D_k+B_0^2x^2},
$$

so

$$
\boxed{
\int T(\eta)S_i(\eta)\,d\eta
\le
U_i
\equiv
\mathcal C_{k(i)}(\eta_i).
}
$$

Every numerical convolution in $\mathcal C_k$ depends only on the laser and the fixed bank. The particle path requires only closed-form geometry and table interpolation.

A geometric bank, for example $D_{k+1}/D_k=r$, gives a direct accuracy/cost knob. The ratio $r$ is a numerical choice to benchmark, not a derived constant.

For a Gaussian temporal envelope, $\mathcal C_k$ is a Gaussian-Lorentzian convolution and can also be expressed with standard Voigt/Faddeeva/erfcx functions. A precomputed numerical table is more general because the same construction supports arbitrary positive envelopes and pulse trains.

### 4.1 Conservative cumulative discard control

For the unchirped model,

$$
L_i\le
\mathcal N\,\varpi_i U_i.
$$

For a discarded set $\mathcal D$ define

$$
\mathcal U_{\mathcal D}
=
\mathcal N
\sum_{i\in\mathcal D}
\varpi_i U_i.
$$

Then

$$
L_{\mathcal D}
\le
\mathcal U_{\mathcal D}.
$$

After the retained set $\mathcal K$ has been integrated, let its true luminosity be $L_{\mathcal K}$. The true discarded fraction satisfies

$$
\frac{
L_{\mathcal D}
}{
L_{\mathcal K}+L_{\mathcal D}
}
\le
\boxed{
\frac{
\mathcal U_{\mathcal D}
}{
L_{\mathcal K}+\mathcal U_{\mathcal D}
}.
}
$$

Therefore the filter can be made self-certifying: choose a candidate discarded set from the cheap bounds, integrate the retained particles, and check the inequality. If the requested tolerance is not met, restore the largest-bound discarded particles and integrate them.

If the retained numerical integration has a certified lower bound $L_{\mathcal K}^{-}$ rather than an exact value, replace $L_{\mathcal K}$ by $L_{\mathcal K}^{-}$ in the certificate.

This resolves the main weakness of a simple per-particle intensity threshold: many individually small positive contributions cannot silently add up beyond the stated discard budget.

## 5. Temporal-envelope-weighted Gaussian quadrature

The same point-electron reduction gives the natural production quadrature.

Define the positive measure

$$
d\mu(\eta)=T(\eta)\,d\eta.
$$

For any sufficiently smooth function $f$,

$$
\int T(\eta)f(\eta)\,d\eta
=
\int f\,d\mu
\approx
\sum_{j=1}^{n}
W_j f(\eta_j),
$$

where $\{\eta_j,W_j\}$ are the $n$-point Gaussian quadrature nodes and weights for the measure $\mu$.

The nodes and weights depend only on the temporal laser envelope, so they are built once and reused for every particle.

This is preferable to a uniform midpoint grid or a global Gauss-Legendre rule because the nodes are automatically concentrated where $T$ carries measure. Long empty intervals, including gaps in a pulse train, need not consume trajectory samples.

### 5.1 Gaussian envelope

For

$$
T(\eta)
=
\frac{1}{\sqrt{2\pi}\sigma_t}
\exp\!\left[
-\frac{\eta^2}{2\sigma_t^2}
\right],
$$

set

$$
x=\frac{\eta}{\sqrt{2}\sigma_t}.
$$

Then

$$
T(\eta)\,d\eta
=
\frac{1}{\sqrt{\pi}}e^{-x^2}\,dx.
$$

If $x_j,h_j$ are ordinary Gauss-Hermite nodes and weights for the weight $e^{-x^2}$, then

$$
\boxed{
\eta_j=\sqrt{2}\sigma_t x_j,
\qquad
W_j=\frac{h_j}{\sqrt{\pi}}.
}
$$

Thus the common Gaussian-pulse case requires no new quadrature construction.

### 5.2 Pulse trains and arbitrary positive envelopes

If

$$
T(\eta)=\sum_m \alpha_m T_m(\eta),
\qquad
\alpha_m\ge0,
\qquad
\sum_m\alpha_m=1,
$$

a composite rule may be formed from the Gaussian rule of each subpulse, multiplying its weights by $\alpha_m$.

For a general positive envelope one may construct the corresponding Gauss-Christoffel rule numerically from the measure $T(\eta)d\eta$. If a robust construction is not available, the generic Gauss-Legendre trajectory integrator remains a valid fallback.

## 6. Rewriting the DER016 Stage-0 moments on the common temporal measure

Let

$$
g_i(\eta)=S_i(\eta),
\qquad
\tau(\eta)=\frac{T(\eta)}{T_{\rm pk}},
$$

and let $C_i(\eta)$ be the encountered carrier ratio of DER016.

Define

$$
D_i
=
\int
C_i(\eta)\,
g_i(\eta)\,
T(\eta)\,d\eta.
$$

Apart from the common physical normalization, $D_i$ is the luminosity integral.

The current DER016 trajectory moments become

$$
a_{{\rm shape},i}
=
\frac{
\int
C_i g_i^2 \tau\,T\,d\eta
}{
D_i
},
$$

$$
\overline C_i
=
\frac{
\int
C_i^2 g_i\,T\,d\eta
}{
D_i
},
$$

$$
V_{a,i}
=
\frac{
\int
C_i g_i^3 \tau^2\,T\,d\eta
}{
D_i
}
-
a_{{\rm shape},i}^2,
$$

$$
V_{C,i}
=
\frac{
\int
C_i^3 g_i\,T\,d\eta
}{
D_i
}
-
\overline C_i^2,
$$

and

$$
K_{aC,i}
=
\frac{
\int
C_i^2 g_i^2 \tau\,T\,d\eta
}{
D_i
}
-
a_{{\rm shape},i}\overline C_i.
$$

Therefore one common weighted quadrature gives

$$
D_i
\approx
\sum_j W_j C_{ij}g_{ij},
$$

and the remaining moments follow from the same nodes by inserting the corresponding powers of $C_{ij}$, $g_{ij}$, and $\tau_j$.

No new Stage-0 moment is introduced, no Stage-1 coordinate is added, and DER016/DER017 remain unchanged. This is a numerical reformulation of the same trajectory statistics.

## Result

Within the circular paraxial, ballistic, narrowband retarded-time model:

1. the encounter factor cancels exactly against the change from lab time to
   $$
   \eta=t-u/c-t_{\rm off};
   $$
2. the full focused Gaussian spatial profile obeys
   $$
   S_i(\eta)
   \le
   \frac{
   1
   }{
   1+d_i^2+B_0^2(\eta-\eta_i)^2
   },
   \qquad
   B_0=\frac{c}{2z_R},
   $$
   provided $z_R^2>2\sigma^2$;
3. a fixed bank of laser-only convolutions $\mathcal C_k$ gives a conservative upper bound $U_i$ for every particle without any particle-dependent quadrature;
4. the cumulative discarded luminosity can be certified after integrating the retained set;
5. the retained trajectory integrals are naturally evaluated by Gaussian quadrature for the positive measure $T(\eta)d\eta$;
6. the same weighted nodes reproduce all current DER016 Stage-0 moments, so Stage 1 and Stage 2 require no physics change.

The only physical approximation specific to this fast path is the replacement of the full narrowband group-delay envelope by the retarded-time envelope. Its applicability can be tested with the laser-only bound $\Omega_T(\Delta\tau_*)$ derived above. Finite-order quadrature adds an ordinary numerical convergence error.

The conservative prefilter result is rigorously unchirped unless a qualifying laser model supplies an upper bound for its positive encountered carrier factor $C$.

## Limiting cases and consistency checks

### Head-on trajectory

For

$$
\mathbf e=-\hat{\mathbf k},
$$

one has

$$
F=2,
\qquad
B=B_0=\frac{c}{2z_R}.
$$

The trajectory that is most weakly localized by the spatial beam therefore saturates the global curvature bound.

### Through-focus trajectory

If the metric miss vanishes,

$$
d_i=0,
$$

then $D_i=1$ and the bound uses the narrowest member of the convolution bank.

### Large transverse miss

For

$$
d_i\gg1,
$$

the denominator $D_i=1+d_i^2$ strongly suppresses the luminosity bound. This is the part absent from a purely longitudinal diffraction envelope.

### No-diffraction limit

As

$$
z_R\to\infty,
$$

$$
B_0\to0.
$$

The global spatial majorant loses longitudinal localization, as it must for a non-diffracting infinite beam. The temporal envelope alone then localizes the interaction.

### Gaussian temporal envelope

The temporal-measure quadrature reduces exactly to a rescaled Gauss-Hermite rule.

### Pulse train

A positive sum of subpulse envelopes gives a positive sum of their quadrature measures. A composite rule resolves separated pulses without sampling the empty gaps between them.

### Few-cycle or strongly chromatic pulse

If

$$
\Omega_T(\Delta\tau_*)
$$

is not small, the retarded-time approximation is not controlled. The fast path must fall back to a trajectory integrator using the full laser model.

### Co-propagation

As

$$
F=1-\mathbf e\cdot\hat{\mathbf k}\to0,
$$

the $\eta$ parameterization becomes singular. This is the same co-propagating regime already excluded by Xigma's encounter-factor guard, not a new restriction introduced by the quadrature.

## Relation to existing derivations and decisions

DER001 is the closest analytical ancestor. It integrates the Gaussian bunch-laser overlap by closing Gaussian dimensions analytically and leaving only the irreducible low-dimensional quadrature. DER023 takes the point-electron limit relevant to Xigma and uses the same principle to move the arbitrary temporal envelope into a common one-dimensional measure.

DER013 supplies the direction-dependent encounter factor $F$. Its cancellation against $d\eta/dt$ is why the temporal measure is common to all particles.

DER016 defines the carrier-weighted Stage-0 moments. Section 6 is an algebraic rewriting of those same moments on the temporal measure; it does not alter their physical meaning.

DER019 derives deterministic Gaussian source reductions for the semi-analytical/AnalyticalEngine side and, in some fixed-width Gaussian limits, can eliminate macroparticles entirely. DER023 serves a different purpose: it retains the real particle bunch and accelerates Xigma Stage 0 for a supported Gaussian laser model. It does not compete with or supersede DER019.

RES045, RES047, and RES048 introduce luminosity and illumination relevance estimates by freezing spot sizes around closest approach. DER023 provides a genuinely conservative upper bound in its narrower circular paraxial regime, so it can support an explicit luminosity-loss certificate rather than only an empirically calibrated tolerance.

RES083 establishes explicit shared NumPy/CuPy Stage-0/1 execution with particle chunking and host stage boundaries. The weighted-node formulation preserves that architecture; it does not motivate a separate CUDA physics implementation.

RES067 keeps Xigma typed against the generic LaserField contract. The Gaussian optimization should therefore be an optional model capability with the current generic trajectory sampling path as fallback.

## Implementation implications

The current Stage-0 implementation on main uses $n_{\rm steps}=200$ midpoint samples over a per-particle window and evaluates intensity and DER016 carrier moments on particle-by-step arrays. A narrowband Gaussian fast path can replace only the node construction and weighted reductions while retaining the same particle chunking and NumPy/CuPy array-module code.

A likely structure is:

- a laser-side or prepared Gaussian capability exposes the retarded temporal envelope, focal metric, and precomputed quadrature/bound tables;
- per-particle host/vectorized geometry computes $F_i$, $d_i$, and $\eta_i$;
- the conservative bank optionally removes particles subject to a requested luminosity-loss certificate;
- retained particles evaluate the common $\eta_j$ nodes and map them analytically back to lab time and position;
- all current Stage-0 moments are accumulated with $W_j$;
- Stage 1 is unchanged;
- arbitrary LaserField implementations, flying-focus fields, unsupported astigmatic geometry, or pulses failing the group-delay gate use the generic trajectory integrator.

A custom CUDA kernel is not implied by this derivation. The main purpose is to reduce the number and cost of field evaluations while preserving one backend-independent vectorized implementation.

The production diagnostics deserve a separate path or separate convergence study. Temporal and spatial histogram bins introduce discontinuous indicator functions, so a quadrature optimized for smooth integrated moments need not reproduce histogram shapes at the same low order.

### Current laser-envelope semantic discrepancy

Current SeparableParaxialLaser evaluates its temporal envelope at the monochromatic phase time $\phi/\omega_0$, including Gouy and wavefront-curvature phase. For a physical narrowband pulse, the translated envelope is controlled instead by the spectral group delay $\partial_\omega\Phi|_{\omega_0}$.

DER023 therefore must not be implemented by silently replacing the current laser's envelope argument. Before this fast path becomes a production specialization of SeparableParaxialLaser, the laser-envelope convention must be resolved and validated. Until then, the derivation can be implemented only for a laser model whose retarded-time envelope semantics are explicit, or used as an independent validation/optimization model.

## Validation ideas

The strongest checks are scientific rather than purely structural:

1. Directly integrate the simplified circular Gaussian pulse on a very fine lab-time grid and verify
   $$
   L_i\le \mathcal N\varpi_i U_i
   $$
   for a large trajectory ensemble, including exact head-on paths, through-focus paths, large misses, and oblique crossings.

2. Verify the analytical ingredients independently:
   $$
   e^{-x}\le(1+x)^{-1},
   $$
   the completed-square formulas for $d_i,\eta_i,B_i$, and the global minimum
   $$
   B_0=c/(2z_R)
   $$
   when $z_R^2>2\sigma^2$.

3. For a Gaussian temporal envelope, compare rescaled Gauss-Hermite orders such as
   $n=4,8,12,16,24,32$ against an over-resolved direct integral, and compare equal-node Gauss-Hermite and Gauss-Legendre convergence.

4. Compare every Stage-0 quantity, not only luminosity: $a_{\rm shape}$, $\overline C$, $V_a$, $V_C$, and $K_{aC}$ against the existing midpoint implementation in the common model.

5. For pulse trains, scan subpulse separation and number of subpulses. Verify that a composite temporal-measure rule converges without requiring nodes in long empty gaps.

6. Validate the cumulative discard certificate. After integrating the retained set, directly integrate the discarded set in test scenarios and verify that its actual luminosity fraction never exceeds the reported upper bound.

7. Validate the group-delay applicability gate against the full narrowband model using
   $T[t-\tau_g(\rho,u)]$. Scan pulse duration, $\sigma$, $z_R$, $g_f$, timing, and transverse position. The chosen $\epsilon_{\rm gd}$ must bound the measured fast-path error in its accepted domain.

8. Re-run NumPy/CuPy Stage-0 agreement using identical quadrature nodes and weights. Benchmark only after the numerical accuracy target is fixed.

9. Cross-check integrated Gaussian-bunch luminosity against DER001 in common circular/head-on cases. The per-particle fast path averaged over a sufficiently dense sampled Gaussian bunch must converge to the analytical overlap.

## Open questions

1. **Laser-envelope semantics.** The current $\phi/\omega_0$ envelope convention and the physical narrowband group-delay convention are not identical. This is the main physics/API question that must be resolved before production use on SeparableParaxialLaser.

2. **Elliptical and astigmatic beams.** GammaForge supports $\sigma_x\ne\sigma_y$, $z_{Rx}\ne z_{Ry}$, and displaced foci. A comparably tight global bound should be derived for that full model instead of applying the circular result heuristically.

3. **Carrier chirp.** The weighted quadrature handles $C(\eta)$ directly, but the rigorous prefilter needs a conservative positive upper bound on the encountered carrier factor. Decide whether qualifying laser models should expose such a bound or whether the conservative prefilter remains unchirped-only.

4. **Generic positive-envelope quadrature.** Gauss-Christoffel quadrature is the natural mathematical construction, but the most stable numerical representation for arbitrary user envelopes remains to be selected. Gaussian and Gaussian-pulse-train envelopes have straightforward specialized rules.

5. **Coupling of tolerances.** The spatial relevance floor $\epsilon_S$, group-delay tolerance $\epsilon_{\rm gd}$, quadrature tolerance, and luminosity discard tolerance should form one documented error budget. Their production defaults should come from validation rather than from the derivation.

6. **Capability boundary.** The optimization should not duplicate Gaussian beam physics inside Xigma. The exact laser-side interface for supplying the retarded envelope, metric, weighted quadrature, and bound tables remains an architectural decision rather than a physics result.

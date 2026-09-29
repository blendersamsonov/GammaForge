# DER024 — Luminosity-aware adaptive sampling of Gaussian electron bunches

Status: derived

## Setup

GammaForge currently represents an analytic six-dimensional Gaussian electron bunch by
drawing IID macroparticles and running Xigma Stage 0 for every retained particle. This is
physically correct, but it is not computationally efficient when the Stage-0 trajectory
integration dominates the cost: increasing the sample count adds many new particles in
the dense Gaussian core, while low-probability regions that can still matter to the
Stage-1 table converge slowly.

This derivation constructs a weighted, stratified representation of the **same** Gaussian
electron measure. The expensive trajectory budget is distributed more uniformly across
phase space and can additionally be biased toward regions with larger expected scattering
luminosity. The statistical weights remain exact, so the luminosity pilot changes
efficiency rather than physics.

The result is intended for the analytic `GaussianElectronBeam` path. Replacing an
arbitrary loaded particle bunch by a fitted Gaussian is a separate approximation and is
not derived here.

The derivation is observer-independent. Current Xigma Stage 1 stores
`(gamma, theta_x, theta_y, a0_shape, chirp_mean)` plus the DER016 variance/covariance
channels, while the exact observer-dependent ponderomotive factor `Q` remains a Stage-2
query-time quantity according to DER015.

### Existing GammaForge Gaussian coordinates

The current Gaussian bunch sampler is most naturally expressed in six independent latent
normal coordinates

$$
d=
(d_x,d_y,d_z,d_{\theta x},d_{\theta y},d_\gamma),
\qquad
d_j\sim\mathcal N(0,1)
$$

independently.

The physical bunch is a deterministic linear map

$$
X=T_{\rm beam}(d).
$$

In the current implementation,

$$
x=\sigma_x d_x,
\qquad
y=\sigma_y d_y,
\qquad
z=\sigma_z d_z.
$$

For the x plane define

$$
\rho_x=-\frac{\alpha_x}{\sqrt{1+\alpha_x^2}},
\qquad
c_x=\frac{1}{\sqrt{1+\alpha_x^2}},
$$

so that

$$
\theta_x=
\sigma_{\theta x}
(\rho_xd_x+c_xd_{\theta x}),
$$

and analogously for y.

Let `gamma_coefficients(beam)` return

$$
(a_x,a_y,a_z,a_{\theta x},a_{\theta y},R).
$$

Then

$$
\gamma=
\gamma_0+
\sigma_\gamma
\left(
a_xd_x+a_yd_y+a_zd_z+
a_{\theta x}d_{\theta x}+a_{\theta y}d_{\theta y}
+\sqrt R,d_\gamma
\right).
$$

This existing latent-to-physical map defines the target Gaussian for the present
derivation. A separate covariance convention is neither required nor desirable.

The target latent density is

$$
p(d)=
(2\pi)^{-3}\exp(-|d|^2/2).
$$

## 1. Why direct IID Gaussian sampling is inefficient here

IID sampling allocates local sample count in proportion to probability mass. In a
high-dimensional Gaussian this places many particles in the core. For an ordinary scalar
Monte Carlo expectation that can be appropriate, but GammaForge performs an expensive
trajectory transform first and only then compresses the result into a multidimensional
Stage-1 table.

Consequently, nearby core particles often become nearly redundant after the Stage-0 to
Stage-1 map, whereas lower-density source regions receive few representatives and their
table contribution converges slowly.

The desired representation is therefore a space-filling quadrature of the Gaussian
measure rather than another IID realization: the numerical particles may be distributed
more broadly, provided their weights reproduce the original Gaussian measure exactly.

## 2. Broad reference Gaussian as a coverage coordinate system

Introduce an isotropically broader Gaussian in the independent latent coordinates,

$$
q_s(d)=\mathcal N(0,s^2I_6),
\qquad s\ge1.
$$

The reference distribution is **not** a replacement physical beam. It is used only to
construct the geometry of the strata.

Define coordinatewise

$$
u_j=\Phi(d_j/s),
$$

where $\Phi$ is the standard-normal CDF. Under $q_s$,

$$
u\sim U([0,1]^6).
$$

Thus the complete unbounded six-dimensional latent space is mapped to a unit cube.

A practical starting value is

$$
s=\sqrt2,
$$

which halves the Gaussian exponent relative to the target and spreads the coverage
substantially farther into the tails. This value is a numerical starting point, not a
new physical constant.

Partition the cube into disjoint axis-aligned boxes

$$
R_m=\prod_{j=1}^{6}[a_{mj},b_{mj}],
$$

satisfying

$$
R_m\cap R_n=\varnothing
\quad(m\ne n),
\qquad
\bigcup_mR_m=[0,1]^6.
$$

The reference probability of a box equals its Euclidean volume,

$$
B_m=
\prod_j(b_{mj}-a_{mj}).
$$

No finite phase-space cutoff is introduced.

## 3. Exact target mass of every stratum

The reference-cube box corresponds to latent bounds

$$
\ell_{mj}=s\Phi^{-1}(a_{mj}),
\qquad
h_{mj}=s\Phi^{-1}(b_{mj}).
$$

Because the **target** distribution factorizes in the latent coordinates, the exact target
Gaussian mass of region $m$ is

$$
\boxed{
P_m=
\prod_{j=1}^{6}
\left[
\Phi(h_{mj})-\Phi(\ell_{mj})
\right].
}
$$

At cube boundaries the corresponding latent bounds are $-\infty$ or $+\infty$.

Since the regions form a partition of the complete cube,

$$
\boxed{
\sum_mP_m=1.
}
$$

Thus every stratum carries a known exact fraction of the physical electron bunch.

This is the key simplification relative to ordinary pointwise importance sampling: there
is no need to assign a separate $p/q_s$ factor to every production particle.

## 4. Exact target-conditional sampling inside a stratum

For coordinate $j$, define

$$
A_{mj}=\Phi(\ell_{mj}),
\qquad
H_{mj}=\Phi(h_{mj}).
$$

For a uniform or low-discrepancy coordinate $v_j\in(0,1)$ set

$$
\boxed{
d_j=
\Phi^{-1}
\left[
A_{mj}
+
v_j(H_{mj}-A_{mj})
\right].
}
$$

Because both the target Gaussian and the region factorize over the latent coordinates,
this produces samples from

$$
p(d\mid R_m)
$$

exactly.

The physical particle is then obtained through the existing map $T_{\rm beam}$.

If region $m$ is represented by $n_m$ production particles, every one receives the same
relative electron weight

$$
\boxed{
w_{mi}=\frac{P_m}{n_m}.
}
$$

Therefore

$$
\sum_iw_i
=
\sum_mn_m\frac{P_m}{n_m}
=
\sum_mP_m
=
1
$$

before any existing geometric prefilter.

The weights need no self-normalization. Changing $n_m$ changes only numerical resolution,
not the represented physical probability.

## 5. Low-discrepancy placement

Inside each region, an extensible low-discrepancy sequence is preferable to IID random
points because the numerical problem has a fixed sample budget and benefits from uniform
coverage.

A simple first implementation may use a six-dimensional Halton sequence with bases

$$
(2,3,5,7,11,13),
$$

combined with deterministic per-region Cranley--Patterson shifts derived from the global
seed and a stable region identifier.

The essential requirement is prefix stability: requesting more points in an existing
region should extend its local sequence rather than regenerate the original points.

The low-discrepancy coordinates are transformed by the target-conditional inverse-CDF
formula above, so the production samples remain exact quadrature points for the target
conditional measure.

## 6. The physically relevant luminosity pilot

The adaptive allocation should not use a binary in/out interaction test. It should use a
continuous estimate of how much scattering a particle is expected to contribute.

For an electron with slopes $(\theta_x,\theta_y)$ define

$$
\mathbf e=
\frac{(\theta_x,\theta_y,1)}
{\sqrt{1+\theta_x^2+\theta_y^2}},
$$

and let $\mathbf n_0$ be the laser propagation direction. The current Xigma encounter
factor is

$$
\boxed{
F=1-\mathbf e\cdot\mathbf n_0.
}
$$

Along the ballistic trajectory

$$
\mathbf r(t)=\mathbf r_0+c\mathbf e,t,
$$

current Stage 0 also evaluates the encountered carrier ratio

$$
\boxed{
C(t)=
1+
\frac{
\partial_t\delta\Phi+
\mathbf v\cdot\nabla\delta\Phi
}{
\omega_0F
},
}
$$

with $\mathbf v=c\mathbf e$ in the production approximation.

Stage 0 weights the local cycle-averaged intensity

$$
I(t)=\langle a^2\rangle
$$

by $C(t)$. Apart from constants common to all pilot particles, the corresponding
single-electron luminosity score is therefore

$$
\boxed{
\ell(X)
=
F
\int
C(t)
I(\mathbf r_0+c\mathbf e t,t)
\,dt.
}
$$

For an unchirped laser $C(t)=1$ and this reduces to $F\int I\,dt$.

The pilot should estimate this same quantity. The current
`illumination_window()` is not a suitable correctness boundary because it is an
approximate relevance window and can discard weak real contributions. A conservative
outer interval such as the existing `overlap_time_window()`, combined with a cheap
high-order quadrature or a suitable specialization of the mature Gaussian overlap
theory, is appropriate.

The pilot does not enter the statistical weights. Therefore an imperfect pilot can
reduce efficiency but cannot bias the represented electron distribution, provided no
stratum is removed.

## 7. Regional pilot moments

Sample pilot points from the target conditional distribution in each region and evaluate
$\ell$.

Define

$$
\mu_m=E[\ell\mid R_m],
$$

$$
M_{2,m}=E[\ell^2\mid R_m],
$$

and

$$
\sigma_m^2=M_{2,m}-\mu_m^2.
$$

The estimated physical photon contribution of the complete region is

$$
\boxed{
C_m=P_m\mu_m.
}
$$

If only the total yield were required, classical stratified Neyman allocation would
assign additional samples according to

$$
n_m\propto P_m\sigma_m.
$$

GammaForge instead needs the multidimensional Stage-1 distribution. A region with almost
constant luminosity may still need several particles if its source points map to different
Stage-1 cells.

A conservative first production score is therefore

$$
\boxed{
A_m=P_m\sqrt{M_{2,m}}.
}
$$

This remains nonzero for luminous but internally smooth regions and also responds to
luminosity variation.

Normalize it as

$$
L_m=
\frac{A_m}{\sum_jA_j}.
$$

This is a practical first allocation law, not a proof of optimality for the full
five-dimensional table.

## 8. Mixture of broad coverage and luminosity allocation

To prevent a noisy pilot from starving weak or tail regions, mix the luminosity proposal
with the broad-reference region mass:

$$
\boxed{
Q_m=
(1-\lambda)B_m+
\lambda L_m.
}
$$

Here $Q_m$ is the desired fraction of the production sample budget assigned to region $m$.

The limits are useful:

- $\lambda=0$: pure broad-reference space-filling coverage;
- $\lambda\to1$: increasingly luminosity-driven allocation.

A practical initial value is $\lambda\simeq0.75$, to be tuned by convergence studies.

If every pilot luminosity is numerically zero, set $Q_m=B_m$.

The mixture also supplies a useful bound. For $s\ge1$,

$$
\frac{p(d)}{q_s(d)}
=
s^6
\exp\left[
-\frac12
\left(1-\frac1{s^2}\right)
|d|^2
\right]
\le s^6.
$$

Hence

$$
\frac{P_m}{B_m}\le s^6.
$$

Since $Q_m\ge(1-\lambda)B_m$,

$$
\boxed{
\frac{P_m}{Q_m}
\le
\frac{s^6}{1-\lambda}.
}
$$

For $s=\sqrt2$ and $\lambda=0.75$ the bound is $32$.

## 9. Arbitrary integer production budget

Let the final partition contain $M$ regions and let the expensive Stage-0 budget be $N$.

Require

$$
N\ge M
$$

and reserve at least one production particle per region.

Allocate the remaining $N-M$ particles according to $Q_m$ using a standard
largest-remainder apportionment. The resulting integer counts satisfy

$$
n_m\ge1,
\qquad
\sum_mn_m=N.
$$

The exact weight remains

$$
w_{mi}=P_m/n_m.
$$

Thus changing $N$ does not require recomputing the pilot; only the integer allocation
changes.

For an especially convenient multiplicative refinement by an integer $K$, one may take

$$
n'_m=K n_m
$$

and extend the prefix-stable low-discrepancy sequence in every region.

## 10. Adaptive refinement of the coarse partition

A large pilot should not be placed uniformly everywhere from the start. Instead, begin
with a coarse balanced partition, evaluate several cheap pilot points per leaf, and split
only regions that remain poorly resolved.

For a scalar-yield objective, the natural unresolved-variance scale is

$$
\boxed{
R_m=P_m\sigma_m.
}
$$

This quantity can be used as the first refinement priority.

A deterministic binary k-d-style partition is sufficient:

1. start from the complete reference cube;
2. split a selected box at its midpoint along its longest side;
3. break axis ties deterministically;
4. keep all leaves;
5. repeat until the requested pilot-region budget is reached.

The leaves remain disjoint and cover the complete cube after every refinement.

A limitation is important: $P_m\sigma_m$ measures luminosity variation only. It does not
know whether a region spans multiple Stage-1 bins even when its luminosity is almost
constant. The broad baseline allocation protects against this failure mode in the first
implementation.

A later refinement metric may also use the pilot-predicted spread in

$$
(\gamma,\theta_x,\theta_y,a0_{\rm shape},\bar C).
$$

## 11. Relation to current Stage-0 and Stage-1 moments

Current Stage 0 defines the normalized intensity

$$
r(t)=I(t)/I_{\rm peak}.
$$

The same pilot quadrature that estimates luminosity can cheaply estimate

$$
a0_{\rm shape}
=
\frac{\int C r^2dt}
{\int C rdt},
$$

and

$$
\bar C=
\frac{\int C^2r\,dt}
{\int Cr\,dt},
$$

together with the DER016 second moments and covariance.

These quantities are not required for the first allocation law, but they provide a
direct route to a future Stage-1-aware refinement criterion.

The sampler itself does not need the observer-dependent $Q$ from DER015. Keeping $Q$
in Stage 2 preserves the current reusable-table architecture.

## 12. Limiting cases and consistency checks

### 12.1 $s=1$

The reference cube becomes the target-Gaussian CDF cube. The method reduces to ordinary
stratified/QMC sampling of the target Gaussian with exact regional masses.

### 12.2 $\lambda=0$

The luminosity pilot does not affect the production allocation. This isolates the gain
from broad phase-space stratification alone.

### 12.3 Unchirped pulse

For $C(t)=1$,

$$
\ell=F\int I\,dt,
$$

and the pilot becomes the usual encounter-factor-weighted integrated intensity.

### 12.4 No interaction

If all pilot luminosities vanish, the allocation falls back to $Q_m=B_m$. The method
remains a valid weighted representation of the electron beam and Stage 0 returns
negligible or zero yield.

### 12.5 Large $N$

For a fixed partition and convergent low-discrepancy sequences, the weighted regional
quadrature converges to expectations under the original Gaussian bunch.

### 12.6 $N<M$

The requested production budget is insufficient to maintain the one-particle-per-region
coverage guarantee. The implementation should reduce $M$ or reject the configuration,
rather than silently dropping regions.

## 13. Relation to existing derivations

DER019 derives deterministic Gaussian source reductions for semi-analytical Compton
spectra. In special Gaussian limits it can remove macroparticles entirely. DER024 serves
a different role: it keeps the general Xigma particle/trajectory pipeline but constructs
a more efficient weighted representation of an analytic Gaussian bunch before Stage 0.

DER023 derives a conservative Gaussian-laser Stage-0 bound and temporal-weight quadrature
for supported narrowband laser models. DER023 reduces the cost **per retained particle**;
DER024 reduces the **number of particles** that require Stage-0 integration. The two
optimizations are complementary.

DER016 defines the carrier-weighted trajectory moments used by current Stage 0 and Stage
1. The DER024 pilot uses the same carrier-weighted luminosity and may reuse the same
trajectory samples to predict those moments.

DER015 establishes that the nonlinear incidence coefficient $Q$ is observer-dependent
and belongs in Stage 2. DER024 therefore deliberately keeps the adaptive source sampler
observer-independent.

DER008 concerns importance sampling of the Stage-2 emission integral after the Stage-1
table already exists. DER024 operates much earlier, on the electron source before Stage
0, and does not replace DER008.

## Result

For an analytic GammaForge Gaussian bunch, a full-support adaptive weighted
representation can be built as follows:

1. Work in the six independent latent standard-normal coordinates that already define
   `sample_gaussian_bunch()`.
2. Use a broader reference Gaussian only to map the full latent space to a convenient
   uniform cube and construct disjoint strata.
3. Compute each stratum's exact target mass
   $$
   P_m=
   \prod_j
   [\Phi(h_{mj})-\Phi(\ell_{mj})].
   $$
4. Sample production points from the **target conditional Gaussian** inside each stratum.
5. If a region receives $n_m$ production samples, give each one the exact constant weight
   $$
   w=P_m/n_m.
   $$
6. Estimate regional relevance with the current Stage-0-compatible pilot
   $$
   \ell(X)=F\int C(t)I(t)\,dt.
   $$
7. Allocate the production budget from a mixture of broad-reference coverage and
   luminosity importance,
   $$
   Q_m=(1-\lambda)B_m+
   \lambda
   \frac{P_m\sqrt{M_{2,m}}}
   {\sum_jP_j\sqrt{M_{2,j}}}.
   $$
8. Preserve at least one production point in every final region so pilot errors change
   efficiency, not support.

This changes only the numerical representation of the known source measure. It does not
change Xigma's Stage-0, Stage-1, or Stage-2 physics.

## Implementation implications

The first implementation should:

- refactor the existing latent-normal to physical-`Bunch` transform so IID and adaptive
  sampling share exactly the same Twiss/gamma-correlation mapping;
- add a focused adaptive-sampling component at the interaction/io layer;
- keep the existing IID sampler and its pinned RNG-substream reproducibility unchanged;
- expose adaptive sampling as an opt-in sampling strategy initially;
- build deterministic non-overlapping reference-cube regions;
- compute exact $P_m$ and sample the target conditional distribution in each region;
- use an extensible low-discrepancy sequence inside regions;
- implement a cheap carrier-aware single-electron luminosity predictor consistent with
  current Stage 0;
- cache the completed region/pilot plan so arbitrary new production budgets can be
  materialized without rerunning the pilot;
- preserve the existing geometric prefilter semantics: an unfiltered adaptive bunch has
  weight sum one, while the retained weight after prefiltering is not renormalized;
- audit engines other than Xigma before enabling the strategy globally, because
  nonuniform `Bunch.weight` must be respected downstream.

The first implementation should not automatically resample an arbitrary imported bunch.
The current `fit_gaussian()` path uses unweighted moments and is not yet a general
weighted, centroid-preserving six-dimensional replacement model.

## Validation ideas

The strongest checks are:

1. **Exact partition invariants.** Verify the reference boxes are disjoint, cover the
   complete cube, satisfy $\sum B_m=1$, and have exact target masses satisfying
   $\sum P_m=1$.

2. **Target-distribution moments.** The weighted adaptive ensemble must reproduce the
   configured Gaussian means, covariances, Twiss tilts, and all gamma correlations.

3. **Tail convergence.** Compare the weighted distribution of
   $r^2=|d|^2$ with the exact $\chi^2_6$ law for IID sampling, broad stratification
   with $\lambda=0$, and the full adaptive scheme at equal $N$.

4. **Pilot convergence.** Compare the cheap luminosity predictor with a deliberately
   over-resolved current Stage 0 for head-on, displaced, timing-offset, tight-focus,
   wide-bunch, crossing-angle, astigmatic, and carrier-chirped cases. Compare successive
   pilot quadrature orders.

5. **Five-dimensional Stage-1 convergence.** At fixed expensive particle budgets compare
   IID and adaptive sampling against a high-statistics reference using total yield,
   normalized `H`, `H_var_a_shape`, `H_var_chirp`,
   `H_cov_a_chirp_shape`, and informative marginals.

6. **Stage-2 convergence.** Compare raw spectral moments and reconstructed spectra for
   several observation directions. The adaptive plan itself must remain
   observer-independent.

7. **Performance.** Measure pilot cost, bunch-construction cost, Stage-0 cost, Stage-1
   cost, and total runtime. The main metric is how many expensive Stage-0 particles are
   needed to reach a fixed Stage-1/Stage-2 error target.

8. **Complementarity with DER023.** After DER024 is independently validated, benchmark
   it both with the existing generic Stage-0 integrator and, when available, with a
   DER023-compatible fast path. The gains should multiply rather than change each
   other's physics.

## Open questions

1. The initial choices $s=\sqrt2$ and $\lambda\simeq0.75$ are principled numerical
   starting points but must be tuned against the scenario bank.

2. The first refinement priority $P_m\sigma_m$ is optimal only for a scalar yield. The
   best criterion for the full five-dimensional Stage-1 table remains open.

3. The optimal number of initial regions, maximum leaves, and pilot points per leaf
   should be selected empirically from total-cost convergence, not treated as physics
   constants.

4. The cleanest implementation of the single-electron pilot remains an architectural
   choice: direct high-order trajectory quadrature through `LaserField` is general,
   while exposing a specialization of the analytical Gaussian overlap machinery may be
   faster and avoid duplicated Gaussian physics.

5. Support for arbitrary loaded weighted bunches requires a separate full weighted
   six-dimensional Gaussian fit, centroid preservation, and an explicit fit-quality
   policy.

6. After DER024 reduces the Stage-0 particle count, the dominant bottleneck may shift to
   trajectory quadrature, GPU memory traffic, or Stage 1. Re-profile before pursuing the
   next optimization.

# GammaForge Coding-Agent Handoff
## Luminosity-Aware Adaptive Sampling Before Xigma Stage 0

**Status:** implementation handoff  
**Repository:** `https://github.com/blendersamsonov/GammaForge`  
**Target branch:** `main`  
**Repository state inspected:** `4540929e5d1d567f085fd392c9959d54e873a04d` — *Document unified xigma chirp and incidence model*

The physics and sampling mathematics in this document are the implementation specification. The coding agent should implement, test, benchmark, and integrate them. Do **not** redesign the statistical method unless a genuine contradiction with the current repository is found.

---

# 1. Objective

GammaForge currently samples an analytic 6D Gaussian electron bunch with IID Monte Carlo particles and then runs Xigma Stage 0 on every retained macroparticle.

That is correct but inefficient for large runs. IID Gaussian sampling puts many particles in the dense core, where neighboring samples often become nearly redundant once Stage 1 bins the Stage-0 result. Low-density tails receive relatively few samples and converge slowly.

The new feature should represent the **same physical Gaussian bunch** using:

1. a broad, space-filling partition of the six independent Gaussian latent coordinates;
2. exact target probability mass for every partition cell;
3. target-conditional low-discrepancy sampling inside each cell;
4. nonuniform macroparticle weights `P_m / n_m`;
5. a cheap Stage-0-like luminosity pilot to allocate more expensive trajectories to cells that matter most;
6. adaptive refinement of the coarse partition;
7. no pilot-based false negatives.

The desired result is:

> For a fixed expensive Stage-0 trajectory budget, achieve substantially faster convergence of the current Stage-1/Stage-2 outputs than the IID Gaussian sampler.

The existing IID path must remain available and initially remain the default.

---

# 2. Current repository state that the implementation must respect

The repository has evolved beyond the older 4D Xigma design.

## 2.1 Xigma Stage 0

Current `integrate_trajectories()` in `src/gammaforge/engines/xigma/stages.py`:

- uses straight ultrarelativistic ballistic trajectories;
- defaults to `n_steps = 200` through Xigma schema;
- defaults to the conservative `"active_region"` time window;
- evaluates cycle-averaged intensity `I = <a^2>`;
- evaluates `laser.carrier_phase_four_gradient`;
- constructs the encountered carrier ratio `C(t)`;
- uses `weighted_intensity = C * I`;
- includes the electron/laser encounter factor;
- multiplies by `N_e * bunch.weight`;
- returns, per macroparticle:
  - `gamma`,
  - `theta_x`,
  - `theta_y`,
  - `luminosity`,
  - `a0_shape`,
  - `chirp_mean`,
  - `var_a_shape`,
  - `var_chirp`,
  - `cov_a_chirp_shape`.

The adaptive pilot must be consistent with this current definition of luminosity.

## 2.2 Xigma Stage 1

Current Stage 1 deposits a five-dimensional `ShapeTable` over

\[
(\gamma,\theta_x,\theta_y,a0_{\rm shape},\bar C),
\]

where `chirp_mean` is the stored carrier coordinate.

The default bin counts are currently

\[
(48,48,48,96,8).
\]

Co-shaped channels are:

- `H`,
- `H_var_a_shape`,
- `H_var_chirp`,
- `H_cov_a_chirp_shape`.

With nearest deposition, one sample contributes to one 5D cell. With CIC it contributes to 32 neighbors.

Validation of the adaptive sampler must target this current 5D representation, not an older 4D table.

## 2.3 Stage 1.5 and Stage 2

Stage 1.5 retargets the raw nonlinear shape coordinate to physical `ahat`.

Stage 2 applies the exact observer-dependent ponderomotive incidence coefficient `Q` at query time.

Therefore the adaptive electron sampler must remain observer-independent. Do not put observation direction, `Q`, or any Stage-2 query geometry into the reusable adaptive sampling plan.

## 2.4 Existing Gaussian sampler reproducibility

`sample_gaussian_bunch()` uses one pinned RNG substream per sampled latent variable via `SeedSequence.spawn`.

This supports GammaForge's reuse/caching semantics.

Do not break or reinterpret the existing IID reproducibility contract.

The new adaptive sampler can have its own deterministic low-discrepancy construction, but the old IID sampler must remain bit-for-bit unchanged for the same inputs.

## 2.5 Analytical overlap machinery

The analytical engine is more mature than older handoffs assumed.

Current `src/gammaforge/engines/analytical/formulas.py` includes:

- general Gaussian beam/laser overlap;
- exact 2D crossing-angle reduction when `n_quad_u > 1`;
- graceful head-on reduction;
- a dedicated flying-focus reduction;
- overlap-weighted nonlinear moments.

This machinery is useful as validation/reference and may later be factored into a direct single-electron predictor.

For the first implementation, the adaptive sampler may use a small direct trajectory quadrature against the live `LaserField` API, provided that it matches current Stage-0 definitions.

---

# 3. Non-goals

Do not combine this feature with unrelated changes.

Specifically, do **not**:

- change Xigma radiation physics;
- change the current five-dimensional Stage-1 coordinates;
- move observer-dependent `Q` out of Stage 2;
- replace Stage 0 with the analytical engine;
- make `illumination_window()` the correctness boundary;
- prune cells because the pilot says they are unimportant;
- redesign the loaded-bunch Gaussian fit in the first PR;
- make the adaptive sampler the default before benchmarks pass;
- change the old IID sampler's RNG behavior;
- optimize the Stage-0 midpoint quadrature in the same PR unless required for the pilot implementation itself.

---

# 4. Exact target Gaussian already defined by GammaForge

The implementation must work in the six independent latent Gaussian deviates already used by `sample_gaussian_bunch()`.

Define

\[
d=(d_x,d_y,d_z,d_{\theta x},d_{\theta y},d_\gamma),
\]

with all components independently distributed as

\[
d_j\sim\mathcal N(0,1).
\]

The physical bunch is a deterministic linear map

\[
X=T_{\rm beam}(d).
\]

Do not reconstruct this map from an independently assembled covariance matrix.

## 4.1 Positions

\[
x=\sigma_x d_x,\qquad
y=\sigma_y d_y,\qquad
z=\sigma_z d_z.
\]

## 4.2 Twiss-correlated angles

For x,

\[
\rho_x=-\frac{\alpha_x}{\sqrt{1+\alpha_x^2}},
\qquad
c_x=\frac1{\sqrt{1+\alpha_x^2}},
\]

and

\[
\theta_x=
\sigma_{\theta x}
(\rho_xd_x+c_xd_{\theta x}).
\]

Similarly for y.

## 4.3 Gamma correlations

Let `gamma_coefficients(beam)` return

\[
(a_x,a_y,a_z,a_{\theta x},a_{\theta y},R).
\]

Then

\[
\gamma=
\gamma_0+
\sigma_\gamma
(a_xd_x+a_yd_y+a_zd_z+a_{\theta x}d_{\theta x}+a_{\theta y}d_{\theta y}+\sqrt R\,d_\gamma).
\]

This includes all current stored energy correlations and the Twiss corrections already implemented in `gamma_coefficients()`.

## 4.4 Required refactor

Before implementing adaptive sampling, factor the existing latent-deviate-to-physical-bunch map into one private reusable helper.

Conceptually:

```python
def _bunch_from_standard_deviates(
    beam: GaussianElectronBeam,
    deviates: Mapping[str, np.ndarray],
    weight: np.ndarray,
    *,
    meta: Mapping | None = None,
) -> Bunch:
    ...
```

Exact naming/layout is an implementation choice.

Both `sample_gaussian_bunch()` and the new adaptive sampler must call the same helper.

This is required to prevent the two sampling paths from drifting in their treatment of Twiss tilt or gamma correlations.

---

# 5. Broad reference Gaussian for coverage geometry

The target latent density is

\[
p(d)=(2\pi)^{-3}\exp(-|d|^2/2).
\]

To obtain broader phase-space coverage, introduce a reference Gaussian

\[
q_s(d)=\mathcal N(0,s^2I_6),\qquad s\ge1.
\]

Use the initial default

\[
\boxed{s=\sqrt2}.
\]

This broad Gaussian is **not** the physical beam. It is used only to define a convenient coordinate system and a broad partition.

Define

\[
u_j=\Phi(d_j/s),
\]

where `Phi` is the standard-normal CDF.

Under `q_s`,

\[
u\sim U([0,1]^6).
\]

This maps the six-dimensional problem to an ordinary unit cube.

---

# 6. Non-overlapping regions

Partition the complete cube `[0,1]^6` into axis-aligned boxes

\[
R_m=\prod_{j=1}^{6}[a_{mj},b_{mj}].
\]

The partition must satisfy

\[
R_m\cap R_n=\varnothing\quad(m\ne n),
\]

and

\[
\bigcup_mR_m=[0,1]^6.
\]

No finite Gaussian cutoff is introduced.

The reference-Gaussian probability of a region is simply its cube volume:

\[
\boxed{B_m=\prod_j(b_{mj}-a_{mj}).}
\]

This quantity is used for the broad-coverage allocation component.

---

# 7. Exact target probability mass of each region

A reference-cube box maps to an axis-aligned box in the independent latent coordinates.

For each bound,

\[
\ell_{mj}=s\Phi^{-1}(a_{mj}),
\qquad
h_{mj}=s\Phi^{-1}(b_{mj}).
\]

Under the **target** Gaussian \(p(d)=N(0,I_6)\), the exact physical probability of region `m` is

\[
\boxed{
P_m=
\prod_{j=1}^{6}
[\Phi(h_{mj})-\Phi(\ell_{mj})].
}
\]

At `u=0` or `u=1`, use latent bounds `-inf` and `+inf`.

Required invariant:

\[
\boxed{\sum_mP_m=1}
\]

up to floating-point roundoff.

This `P_m` is the actual fraction of the electron bunch represented by the region.

---

# 8. Sample the true target conditional inside each region

Do **not** sample production particles from the broad reference Gaussian and then use pointwise `p/q`.

Instead, sample directly from the target Gaussian conditioned on the region.

For one coordinate, define

\[
A_{mj}=\Phi(\ell_{mj}),
\qquad
H_{mj}=\Phi(h_{mj}).
\]

Given a low-discrepancy coordinate `v_j` in `(0,1)`, set

\[
\boxed{
d_j=
\Phi^{-1}
[A_{mj}+v_j(H_{mj}-A_{mj})].
}
\]

Because the target latent Gaussian factorizes and the region is axis-aligned in latent space, this exactly samples

\[
p(d\mid R_m).
\]

Map the resulting `d` through the shared beam transform.

---

# 9. Exact constant macroparticle weights

If region `m` receives `n_m` production particles, every particle in that region gets

\[
\boxed{w_{mi}=P_m/n_m.}
\]

This is exact for the stratified estimator.

The total relative bunch weight before the existing prefilter is automatically

\[
\sum_iw_i=
\sum_mn_m\frac{P_m}{n_m}
=
\sum_mP_m
=
1.
\]

Therefore:

- no pointwise `p/q_s` weights are needed;
- no post-hoc self-normalization is needed;
- all particles in one region have equal weight;
- a change in `n_m` changes only numerical resolution, not represented physical mass.

This supersedes the older pointwise-importance-weight formulation.

---

# 10. Low-discrepancy points inside each region

Use an extensible low-discrepancy sequence instead of IID RNG inside each region.

A minimal first implementation can use Halton bases `(2,3,5,7,11,13)`.

Requirements:

- start from index 1, not the all-zero Halton point;
- use deterministic Cranley-Patterson shifts;
- shift must depend on global seed, stable region ID, and stream tag (`pilot` vs `production`);
- region prefixes must be stable when more points are requested;
- map local `v` points through the truncated-target inverse-CDF construction above;
- avoid exact 0/1 before inverse-normal evaluation.

Do not add SciPy solely for QMC.

Implement a vectorized normal inverse CDF using a standard high-accuracy rational approximation and validate it against `statistics.NormalDist().inv_cdf`.

---

# 11. Correct current-main luminosity predictor

The pilot must estimate the same physical relevance that current Stage 0 uses.

For an electron with slopes `(theta_x, theta_y)`, define

\[
\mathbf e=
\frac{(\theta_x,\theta_y,1)}
{\sqrt{1+\theta_x^2+\theta_y^2}}.
\]

Let `n0` be the laser propagation direction.

Define

\[
\boxed{F=1-\mathbf e\cdot\mathbf n_0.}
\]

Current main also includes the carrier correction.

Along the ballistic trajectory

\[
\mathbf r(t)=\mathbf r_0+c\mathbf e\,t,
\]

the encountered carrier ratio is

\[
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
\]

with `v = c e`.

The pilot relevance score should therefore be

\[
\boxed{
\ell(X)
=
F\int C(t)I(\mathbf r(t),t)\,dt.
}
\]

Here `I=<a^2>` is the same cycle-averaged intensity used by Stage 0.

For built-in unchirped lasers, `C(t)=1`, so the predictor reduces to

\[
\ell=F\int I\,dt.
\]

The older intensity-only pilot is not sufficient for current-main chirped pulses.

---

# 12. Predictor implementation

Implement the pilot as a separate pure/vectorized helper.

Suggested API:

```python
def trajectory_luminosity_predictor(
    bunch: Bunch,
    laser: LaserField,
    *,
    n_quad: int = 64,
    threshold: float = 1e-8,
) -> np.ndarray:
    ...
```

It should return a nonnegative per-particle score proportional to the expected Stage-0 luminosity **without** `bunch.weight` and `N_e`.

Required procedure:

1. Compute the same normalized electron direction as Stage 0.
2. Compute the same encounter factor `F`.
3. Use `overlap_time_window()` to obtain a conservative finite trajectory interval.
4. Use Gauss-Legendre quadrature on each nonempty interval.
5. Evaluate `laser.intensity_profile`.
6. Evaluate `laser.carrier_phase_four_gradient`.
7. Construct `C(t)` with the same convention and positivity checks as Stage 0.
8. Integrate `C * I`.
9. Multiply by `F`.
10. Return zero for genuinely empty conservative windows.

Do not use `illumination_window()` as the sole integration domain. Its current contract is an estimate, not a conservative bound.

The pilot is allowed to be approximate. Its approximation may affect efficiency, but never the represented target distribution.

---

# 13. Relationship to the mature analytical engine

The analytical engine now has mature Gaussian overlap reductions, including exact 2D crossing-angle handling and a dedicated flying-focus path.

Use it as a validation reference where applicable.

Do **not** tightly couple the new sampler to `AnalyticalEngine` classes.

The long-term clean design may expose a shared single-electron Gaussian-overlap primitive.

For the first implementation, a small direct Gauss-Legendre trajectory integral through the `LaserField` protocol is acceptable and more general.

---

# 14. Pilot can cheaply estimate current Stage-0 table coordinates too

While evaluating the pilot trajectory, define

\[
r(t)=I(t)/I_{\rm peak}.
\]

Current main uses

\[
\boxed{
a0_{\rm shape}
=
\frac{\int Cr^2dt}{\int Crdt}.
}
\]

Current carrier mean is

\[
\boxed{
\bar C=
\frac{\int C^2r\,dt}{\int Cr\,dt}.
}
\]

The current second moments are similarly available from the same quadrature.

The first implementation does **not** need these values for allocation.

However, if easy to expose, retain them as pilot diagnostics because they enable a future explicitly Stage-1-aware refinement metric.

---

# 15. Pilot statistics inside a region

Pilot particles in a region are drawn from the target conditional distribution `p(d|R_m)`.

For pilot luminosities `ell_i`, estimate

\[
\mu_m=E[\ell\mid R_m],
\]

\[
M_{2,m}=E[\ell^2\mid R_m],
\]

\[
\sigma_m=
\sqrt{\max(M_{2,m}-\mu_m^2,0)}.
\]

The expected physical photon contribution of the region is

\[
\boxed{C_m=P_m\mu_m.}
\]

---

# 16. Scalar optimality and the practical production score

If the only goal were minimizing variance of the **total yield**, ordinary stratified Neyman allocation would give

\[
n_m\propto P_m\sigma_m.
\]

GammaForge does not only need total yield. It needs the distribution deposited into the 5D Stage-1 table.

A region with nearly constant luminosity can still need multiple particles because its samples may map into different `(gamma, theta_x, theta_y, a0_shape, chirp_mean)` cells.

Therefore use the more conservative initial production importance

\[
\boxed{A_m=P_m\sqrt{M_{2,m}}.}
\]

Normalize

\[
L_m=A_m/\sum_jA_j.
\]

This is not claimed to be mathematically optimal for the complete Stage-1 table. It is the first implementation target.

A later improvement may use pilot-predicted Stage-1 coordinate variation directly.

---

# 17. Mix luminosity allocation with broad coverage

Do not allocate the complete sample budget from `L_m`.

Keep an explicit broad-coverage component:

\[
\boxed{
Q_m=(1-\lambda)B_m+\lambda L_m.
}
\]

Initial default:

\[
\boxed{\lambda=0.75.}
\]

Interpretation:

- `lambda = 0`: purely broad space-filling coverage;
- `lambda = 1`: purely luminosity-driven allocation;
- intermediate values protect weak/tail phase-space regions.

If all luminosity scores vanish numerically, use `Q_m=B_m`.

For `s>=1`,

\[
p(d)/q_s(d)\le s^6,
\]

so

\[
P_m/B_m\le s^6.
\]

Since `Q_m >= (1-lambda) B_m`,

\[
\boxed{
P_m/Q_m\le s^6/(1-\lambda).
}
\]

For `s=sqrt(2)` and `lambda=0.75`, the bound is 32.

This is a useful reason to retain the broad baseline mixture.

---

# 18. Arbitrary integer production budget

Let the final region count be `M`.

Require `N >= M`.

Every region receives at least one production particle.

Define

\[
N_{\rm extra}=N-M.
\]

Allocate `N_extra` according to `Q_m` using largest-remainder apportionment:

1. `r_m = N_extra * Q_m`;
2. `a_m = floor(r_m)`;
3. distribute the remaining particles to the largest fractional parts;
4. `n_m = 1 + a_m`.

Required invariants:

\[
n_m\ge1,
\qquad
\sum_mn_m=N.
\]

Particle weights remain `P_m/n_m`.

No pilot needs to be rerun just because the requested `N` changes.

---

# 19. Multiplicative refinement and plan reuse

Create a reusable `AdaptiveSamplingPlan`.

The plan should contain:

- reference scale `s`;
- region tree;
- region bounds;
- `B_m`;
- `P_m`;
- pilot moments;
- final `Q_m`;
- deterministic region IDs;
- seed;
- pilot configuration;
- optional previous production allocation.

For an exact multiplicative refinement, support

\[
n'_m=K n_m
\]

and extend the per-region low-discrepancy sequence prefix.

No pilot reevaluation is needed.

For an arbitrary new `N`, simply recompute integer allocation from the existing `Q_m`.

Rebuild the plan only when a beam/laser property that affects the proposal changes.

---

# 20. Adaptive pilot refinement

The pilot itself should be adaptive.

Suggested starting defaults to benchmark:

```text
initial_regions = 64
max_regions = 256
pilot_points_per_region = 8
pilot_quad_nodes = 64
pilot_window_threshold = 1e-8
proposal_scale = sqrt(2)
luminosity_fraction = 0.75
```

These are numerical starting values, not physics constants.

## 20.1 Initial partition

Build a deterministic balanced binary partition of the reference cube.

For an arbitrary requested leaf count:

- a node assigned `m` descendants splits into:
  - `m_left = floor(m/2)`
  - `m_right = m - m_left`;
- split the current longest cube side;
- if tied, use a deterministic axis order;
- place the cut at fraction `m_left/m`.

This produces arbitrary non-power-of-two region counts without overlap.

## 20.2 Refinement priority

For a scalar-yield objective, the unresolved contribution scales with

\[
P_m\sigma_m.
\]

Use

\[
\boxed{R_m=P_m\sigma_m}
\]

as the first refinement priority.

Repeatedly:

1. evaluate pilot statistics in current leaves;
2. rank by `R_m`;
3. split the highest-priority leaves;
4. sample fresh pilot points in new children;
5. stop at `max_regions`.

Do not remove unsplit leaves.

## 20.3 Important limitation

`P_m sigma_m` only measures luminosity variation. It does not know whether a region spreads over many current Stage-1 cells.

Keep this as an explicitly documented limitation and future improvement.

A later refinement criterion should combine luminosity variation with predicted spread in

\[
\gamma,\theta_x,\theta_y,a0_{\rm shape},\bar C.
\]

---

# 21. Proposed code organization

Create a focused module such as

```text
src/gammaforge/io/adaptive_sampling.py
```

or a similarly appropriate `io`-level location.

Do not bury the algorithm inside Xigma Stage 0.

The sampler produces a `Bunch`, and the interaction layer already owns bunch construction.

Suggested internal types:

```python
@dataclass(frozen=True)
class SamplingRegion:
    id: int
    lo: np.ndarray
    hi: np.ndarray
    reference_mass: float
    target_mass: float
    pilot_mean: float
    pilot_second_moment: float
    pilot_std: float
    allocation_probability: float
```

```python
@dataclass(frozen=True)
class AdaptiveSamplingPlan:
    seed: int
    proposal_scale: float
    luminosity_fraction: float
    regions: tuple[SamplingRegion, ...]
    pilot_config: ...
    diagnostics: ...
```

Exact public/private split is an implementation decision.

---

# 22. `SamplingSpec` integration

Current `SamplingSpec` contains:

- `n_particles`;
- `seed`;
- `prefilter`.

Add the smallest public option needed to select the new strategy.

For example:

```python
strategy: str = "iid"  # "iid" | "adaptive"
```

Do not expose all tuning constants to the GUI in the first implementation.

Use module-level defaults initially.

In `build_interaction()`:

### IID

Keep current code exactly unchanged.

### Adaptive

1. validate `GaussianElectronBeam`;
2. build/reuse adaptive plan;
3. materialize exactly `n_particles`;
4. create an ordinary weighted `Bunch`;
5. apply the existing geometric prefilter exactly as today;
6. construct `InteractionParameters`.

---

# 23. Preserve existing prefilter semantics

Current `Bunch.weight` represents relative physical population.

For an unfiltered adaptive bunch,

\[
\sum_iw_i=1.
\]

The existing geometric prefilter deliberately does **not** renormalize after dropping particles.

Keep that behavior.

Therefore after filtering,

\[
\sum_iw_i\le1
\]

honestly records the fraction of the original population retained.

Do not renormalize adaptive weights after `prefilter_bunch()`.

Do not make the adaptive luminosity pilot itself a filter.

---

# 24. Engine compatibility audit

Xigma Stage 0 already uses

```python
weight = n_electrons * bunch.weight
```

so it should naturally support adaptive weights.

Before making adaptive sampling available to every engine, inspect each engine that consumes `Bunch`.

Required outcome:

- either confirm it respects per-particle `Bunch.weight`;
- or explicitly restrict adaptive sampling for that engine;
- or hard-fail instead of silently assuming equal weights.

Do not alter `N_e` semantics.

---

# 25. Metadata and diagnostics

Adaptive bunch/plan metadata should be sufficient to reproduce and debug a run.

At minimum record:

- sampling strategy;
- seed;
- requested production `N`;
- `proposal_scale`;
- `luminosity_fraction`;
- initial region count;
- final region count;
- pilot points per region;
- pilot quadrature nodes;
- pilot threshold;
- regional production counts;
- min/max/quantiles of `P_m/n_m`;
- effective sample size:
  \[
  N_{\rm eff}=(\sum_iw_i)^2/\sum_iw_i^2;
  \]
- pilot build time if practical;
- estimated regional luminosity fractions.

Do not dump all pilot particles into metadata.

---

# 26. Tests: exact Gaussian representation

These should be fast and independent of Xigma physics.

## 26.1 Region partition

For many region counts:

- every bound is inside `[0,1]`;
- boxes are non-overlapping;
- union covers the cube;
- `sum(B_m) == 1`.

## 26.2 Target masses

Check

\[
P_m\ge0,
\]

and

\[
\sum_mP_m=1
\]

to floating-point precision.

For selected simple boxes, compare `P_m` to an independent normal-CDF product.

## 26.3 Conditional samples

For individual boxes, verify transformed samples stay within the correct latent bounds.

## 26.4 Exact total weight

Before prefilter,

\[
\sum_iw_i=1
\]

to machine precision.

No normalization pass should be needed.

## 26.5 Exact budget

For many `N>=M`,

\[
\sum_mn_m=N,
\qquad
n_m\ge1.
\]

## 26.6 Beam moments

With sufficiently large adaptive `N`, weighted moments must converge to the same beam distribution as the analytic target.

Include nonzero:

- `alpha_x`;
- `alpha_y`;
- `rho_x_gamma`;
- `rho_y_gamma`;
- `rho_z_gamma`;
- `rho_thx_gamma`;
- `rho_thy_gamma`.

This is a critical regression test.

---

# 27. Tests: tail convergence

Recover the latent `d` values or test directly in latent coordinates.

Define

\[
r^2=d\cdot d.
\]

For the target,

\[
r^2\sim\chi^2_6.
\]

At equal `N`, compare:

1. existing IID sampling;
2. broad stratified sampling with `lambda=0`;
3. full luminosity-adaptive sampling where appropriate.

Measure:

- global weighted CDF error;
- high-quantile error;
- tail probability error.

Do not compare the **unweighted** adaptive samples to the Gaussian target; their spatial density is deliberately different.

---

# 28. Tests: pilot convergence

Compare `trajectory_luminosity_predictor()` to deliberately over-resolved current Stage 0.

Cases should include:

- head-on centered collision;
- transverse offset;
- timing offset;
- angular offset;
- crossing angle;
- tight focus;
- wide electron bunch;
- astigmatic laser;
- nonzero Twiss alpha;
- nonzero beam/energy correlations;
- intrinsic carrier chirp;
- supported flying-focus cases if the live `LaserField` supports them.

Compare:

- 32 vs 64 pilot nodes;
- 64 vs 128 pilot nodes;
- pilot vs high-resolution Stage 0.

The pilot does not need machine-precision equality.

What matters is that its regional ranking and allocation are stable enough to improve convergence.

Also add a deliberately low-quality predictor test proving that the final weighted electron distribution remains correct. This pins the architectural guarantee that pilot quality affects efficiency, not sampling correctness.

---

# 29. Tests: current five-dimensional Stage 1

Build high-statistics IID references for representative scenarios.

At multiple smaller budgets compare IID vs adaptive.

Compare:

- total yield;
- normalized 5D `ShapeTable.H`;
- `H_var_a_shape`;
- `H_var_chirp`;
- `H_cov_a_chirp_shape`;
- 1D marginals over each axis;
- useful 2D marginals where informative.

Use both nearest and, at least in a smaller case, CIC if adaptive sampling is intended to support both.

Do not use only total yield as the acceptance metric.

---

# 30. Tests: Stage 2

For representative tables, compare adaptive and IID/reference results for several observation directions.

At minimum compare:

- raw spectral moment outputs;
- reconstructed spectrum;
- centroid;
- integrated yield;
- representative angular spectra.

The adaptive plan itself is observer-independent, but the complete pipeline must still converge after current Stage-2 `Q` is applied.

---

# 31. Benchmarks

Report separately:

- plan construction;
- pilot evaluation;
- production bunch construction;
- Stage 0;
- Stage 1;
- Stage 2 where relevant;
- total runtime.

The key benchmark is:

> For a fixed Stage-1/Stage-2 accuracy target, how many expensive Stage-0 particles are required by IID vs adaptive sampling?

Do not judge the feature only at equal `N`.

Do not hard-code an expected order-of-magnitude speedup before measuring it.

---

# 32. Loaded or imported bunches are a later feature

The first implementation should work only for an analytic `GaussianElectronBeam`.

Do not automatically Gaussian-resample arbitrary loaded particle data yet.

Current `fit_gaussian()`:

- uses unweighted means/covariances;
- does not preserve an arbitrary nonzero 6D centroid as a reconstructable analytic beam;
- is therefore not sufficient as a universal loaded-data replacement path.

Future path:

1. weighted full-6D Gaussian fit;
2. fit-quality evaluation;
3. explicit user/config acceptance of the Gaussian approximation;
4. adaptive resampling of the fitted distribution.

Keep this outside the first implementation.

---

# 33. Stage-0 quadrature optimization is a separate follow-up

The adaptive sampler reduces the number of particles.

A separate optimization can reduce the number of field evaluations per particle.

Current Stage 0 still uses fixed midpoint sampling.

After adaptive sampling is implemented and benchmarked, investigate:

- Gauss-Legendre trajectory quadrature;
- adaptive verified quadrature;
- narrower-but-verified integration support;
- fused GPU trajectory integration.

Do not mix these changes into the adaptive-sampling PR unless necessary for the pilot helper.

Re-profile after adaptive sampling before deciding where the next bottleneck is.

---

# 34. Recommended implementation sequence

1. Refactor the existing latent Gaussian-to-physical-bunch transform.
2. Preserve and regression-test old IID bitwise reproducibility.
3. Add vectorized normal CDF/inverse-CDF utilities as needed.
4. Add extensible 6D low-discrepancy local sequences.
5. Implement deterministic reference-cube partitioning.
6. Implement exact `B_m`.
7. Implement exact target mass `P_m`.
8. Implement target-conditional sampling.
9. Implement constant regional weights `P_m/n_m`.
10. Validate the `lambda=0` broad-stratified sampler before adding laser physics.
11. Implement current-main carrier-aware trajectory luminosity predictor.
12. Implement regional pilot moments.
13. Implement `Q_m`.
14. Implement arbitrary integer allocation.
15. Implement adaptive cell splitting.
16. Implement `AdaptiveSamplingPlan` reuse.
17. Integrate opt-in strategy into `SamplingSpec` and `build_interaction()`.
18. Audit non-Xigma engine weight handling.
19. Add Stage-1 current-5D convergence tests.
20. Add representative Stage-2 convergence tests.
21. Add performance benchmarks.
22. Tune only numerical defaults supported by benchmark evidence.

---

# 35. Settled constraints — do not reinterpret

The following are fixed for this implementation:

- The physical target is exactly the existing `GaussianElectronBeam`.
- The latent variables are the six existing independent standard normals.
- The broad Gaussian is only a coverage geometry.
- Regions live in the broad-reference CDF cube.
- Regions must be disjoint and cover the complete cube.
- Every region has exact target mass `P_m`.
- Production points are sampled from the target Gaussian conditional on the region.
- Every production particle in region `m` has weight `P_m/n_m`.
- No pilot estimate may delete a region.
- The pilot luminosity for current main is
  \[
  F\int C(t)I(t)\,dt.
  \]
- Observer-dependent `Q` stays in Stage 2.
- Current Stage 1 is 5D and carries the three moment channels.
- Existing prefilter semantics remain unchanged.
- Existing IID RNG behavior remains unchanged.
- The new method is opt-in during validation.
- Loaded-bunch Gaussian replacement is deferred.

---

# 36. Definition of done

The implementation is complete when all of the following are true:

1. The old IID sampler remains bit-for-bit compatible.
2. Adaptive mode constructs exactly `N` production particles.
3. The adaptive regions cover the complete latent target support through the reference cube.
4. Exact regional target masses sum to one.
5. Production weights sum to one before prefiltering without self-normalization.
6. Weighted beam moments reproduce the target Gaussian and all current correlations.
7. The carrier-aware pilot matches current Stage-0 luminosity well enough for stable allocation.
8. No pilot-based false-negative region deletion exists.
9. Xigma Stage 0 consumes adaptive particles without a physics special case.
10. Current 5D Stage-1 tables and moment channels converge to the high-statistics reference.
11. Representative Stage-2 outputs converge to the same reference.
12. Benchmarks demonstrate a material reduction in expensive Stage-0 work at fixed output accuracy.
13. The adaptive plan can be reused for arbitrary `N`, and local low-discrepancy prefixes can be extended for multiplicative refinement.
14. The implementation documents measured pilot overhead, regional weight distribution, effective sample size, and convergence improvement.

---

# 37. Final note to the coding agent

Treat this as an implementation task.

Do not spend time re-deriving the sampling method.

If a repository detail conflicts with this handoff, verify `main` and adapt the software interface while preserving the mathematical invariants above.
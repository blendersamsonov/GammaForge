# GammaForge Coding-Agent Handoff
## Phase II — Deterministic Gaussian Cubature and Smooth Push-Forward

**Status:** revised implementation/research handoff  
**Issue:** #20  
**Pull request:** #21  
**Branch:** `feature/luminosity-aware-adaptive-sampling`  
**Authority:** DER024 for the exact Gaussian representation; RES093 and `docs/validation/adaptive-sampling-2026-09-29.md` for the completed Phase-I implementation and measured evidence.

This is a **material revision of the original handoff after validation**. The original handoff remains preserved in branch history. Do not rewrite the Phase-I result as though it never happened.

The task now is to determine whether GammaForge can exploit the smooth analytic six-dimensional Gaussian source as a **deterministic cubature / push-forward problem**, rather than continuing to tune the luminosity-allocation heuristic.

The agent should implement the experiments and the minimal supporting code needed to answer that question. Do not promote a new production default merely because an experiment is promising; report the evidence first.

---

# 1. Phase-I conclusion that motivates this revision

Phase I established three different facts.

1. **The exact weighted Gaussian representation is sound.**  
   The region partition tiles the reference cube, exact target masses sum to one, and regional weights `P_m / n_m` preserve the configured Gaussian measure. Pilot errors affect efficiency rather than correctness.

2. **Replacing IID clustering by deterministic/stratified sampling is the robust win.**  
   In the 10-scenario validation bank, plain target stratification (`s=1, lambda=0`) beat IID on resolved spectrum error in 9 of 10 scenarios.

3. **The additional luminosity-aware machinery is not a robust general-purpose win.**
   - `s=sqrt(2)` was worse than `s=1` in 9 of 10 scenarios without allocation.
   - luminosity allocation helped concentrated interactions by roughly 12–15%, but was neutral or harmful elsewhere;
   - `wide_bunch` produced a severe weight collapse;
   - adaptive region splitting had no measurable benefit;
   - the attempted cell-aware rule requires much richer pilot information than the current pilot supplies.

Therefore do **not** spend this phase tuning `lambda`, proposal scale, or splitting further.

The new question is more fundamental:

> How many expensive Stage-0 evaluations are actually required to integrate a smooth known 6D Gaussian source when the source integral is treated as cubature rather than as a Monte-Carlo particle cloud?

The target of a 10–100x reduction in Stage-0 trajectories is **not ruled out** by Phase I. The previous benchmark was dominated by the discontinuous Stage-1 deposition and was not a clean test of smooth 6D integration.

---

# 2. Central hypothesis

Let

[
d=(d_x,d_y,d_z,d_{	heta x},d_{	heta y},d_gamma)
sim mathcal N(0,I_6)
]

be the six independent latent coordinates already used by `GaussianElectronBeam`.

Stage 0 defines a deterministic map

[
d longmapsto
Y(d)
=
(gamma,	heta_x,	heta_y,a0_{m shape},ar C,
operatorname{Var}(a0_{m shape}),
operatorname{Var}(C),
operatorname{Cov}(a0_{m shape},C),
L),
]

where (L) is the current carrier-weighted luminosity.

For a Gaussian laser and ballistic trajectories, this map is expected to be smooth over most of the source measure.

For any smooth observable (f(Y(d))),

[
mathbb E[f]
=
int_{mathbb R^6}
p(d) f(Y(d)),d^6d
]

should therefore be amenable to deterministic high-order cubature and low-discrepancy integration.

Ordinary IID Monte Carlo converges only statistically, approximately as (N^{-1/2}). The purpose of this phase is to **measure the convergence law** of deterministic methods instead of assuming the final binned Stage-1 spectrum has the same convergence behavior.

---

# 3. Methods to compare

Implement the following as experiment-level samplers/integrators. Reuse existing latent-to-physical mapping and Stage-0 code.

## 3.1 IID control

The existing `sample_gaussian_bunch()` path.

This is the baseline only. Do not modify it.

## 3.2 Global Gaussian QMC

No regions, no luminosity pilot, no nonuniform allocation.

Generate a single extensible low-discrepancy sequence

[
u_iin(0,1)^6
]

and map directly to target latent normals

[
d_{ij}=Phi^{-1}(u_{ij}).
]

Use equal weights

[
w_i=1/N.
]

Start by reusing the existing deterministic shifted Halton implementation.

If `scipy.stats.qmc.Sobol` is already available in the experiment environment, it may be added as an **experiment-only comparison** with scrambling, but do not add SciPy as a GammaForge runtime dependency solely for this work.

Required property: increasing (N) should preserve or naturally extend the existing sequence where the chosen QMC construction permits it.

## 3.3 Existing fixed target stratification

Keep the current `s=1, lambda=0`, no-splitting arm as a comparator.

This tells us whether explicit regional stratification adds anything beyond global low-discrepancy sampling.

## 3.4 Tensor Gauss-Hermite cubature

This is the most important new experiment.

For the standard-normal expectation,

[
mathbb E[f(d)]
=
rac{1}{pi^3}
int_{mathbb R^6}
e^{-|x|^2}
f(sqrt2 x),d^6x.
]

Let ((x_k,w_k)) be the ordinary one-dimensional Gauss-Hermite rule for weight (e^{-x^2}), available from `numpy.polynomial.hermite.hermgauss(n)`.

The six-dimensional tensor rule is

[
d_{i_1dots i_6}
=
sqrt2
(x_{i_1},ldots,x_{i_6}),
]

with normalized positive weight

[
W_{i_1dots i_6}
=
rac{
w_{i_1}cdots w_{i_6}
}{
pi^3
}.
]

All weights are positive and should sum to one to roundoff.

Test at least orders

[
n=3,4,5,6,7,8,
]

and use (n=9) or (10) where computationally practical.

The corresponding trajectory counts are:

| 1D order | 6D nodes |
|---:|---:|
| 3 | 729 |
| 4 | 4,096 |
| 5 | 15,625 |
| 6 | 46,656 |
| 7 | 117,649 |
| 8 | 262,144 |
| 9 | 531,441 |
| 10 | 1,000,000 |

This directly probes the user's original expectation: can a smooth six-dimensional Gaussian source be represented accurately with (10^4)–(10^5) carefully chosen Stage-0 trajectories instead of millions of random particles?

The Gauss-Hermite path should use the existing shared latent-to-physical transformation and produce positive `Bunch.weight = W` when exercising the existing pipeline.

Do **not** reject Gauss-Hermite because its weight-based Monte-Carlo effective sample size is low. (N_{m eff}) is a Monte-Carlo diagnostic and is not an accuracy criterion for deterministic Gaussian quadrature.

## 3.5 Sparse-grid / Smolyak probe

Only after the tensor Gauss-Hermite baseline is working.

Implement an **experiment-only** Smolyak-type Gaussian cubature using one-dimensional Gauss-Hermite rules if feasible without a large new dependency.

Purpose:

- determine whether the smooth-observable convergence of tensor Gauss-Hermite can be reached with substantially fewer than (n^6) evaluations;
- estimate the effective dimension/anisotropy of the Stage-0 map.

Important limitation: ordinary Smolyak combination rules may have signed weights.

Signed cubature weights are acceptable for **smooth-observable experiments** but are not automatically acceptable as production `Bunch.weight`, Stage-1 histogram mass, or variance channels.

Do not wire a signed sparse-grid rule into the production Xigma table path in this phase.

If a reliable sparse-grid implementation would become a project by itself, stop after the tensor Gauss-Hermite and global-QMC experiments and report that rather than building a fragile rule.

---

# 4. First experiment: remove Stage-1 discontinuity entirely

Before judging any method on the histogrammed spectrum, test the smooth Stage-0 map directly.

For every sampling/cubature method, compute a fixed vector of luminosity-weighted smooth observables.

At minimum include:

[
M_0 = int L(d),p(d),dd,
]

[
M_{y_i} = int L(d)y_i(d),p(d),dd,
]

[
M_{y_i y_j}
=
int L(d)y_i(d)y_j(d),p(d),dd,
]

for

[
y=
(gamma,	heta_x,	heta_y,a0_{m shape},ar C).
]

Also include several nonlinear but smooth observables, for example:

[
int L a0_{m shape}^3,p,dd,
]

[
int Lexp[-(	heta_x/	heta_*)^2-(	heta_y/	heta_*)^2],p,dd,
]

and several low-frequency characteristic/Fourier probes of the push-forward:

[
Phi(k)
=
int
L(d)
exp[i,kcdot 	ilde y(d)]
p(d),dd,
]

where (	ilde y) is a dimensionless standardized version of the five Stage-1 coordinates.

Choose a small fixed set of (k) vectors spanning individual axes and mixed directions.

These characteristic-function probes are important: they test the **shape** of the pushed-forward distribution while remaining smooth, unlike histogram cell indicators.

---

# 5. Reference for smooth observables

Do not use a noisy 4M IID result as the only authority if deterministic methods themselves can establish convergence.

Build the reference hierarchically.

Suggested procedure:

1. run tensor Gauss-Hermite at increasing order;
2. run global QMC at large (N) with several deterministic scrambles/shifts;
3. compare both to the large IID reference already available;
4. declare a reference quantity resolved only when two independent numerical constructions agree within the requested tolerance.

For each observable, record the difference between successive Gauss-Hermite orders.

If orders 8, 9 and 10 agree closely while IID still fluctuates, use the converged deterministic value as the stronger reference for that smooth observable.

---

# 6. Measure convergence rate, not only equal-N ratios

For each observable and method, measure error versus expensive Stage-0 evaluation count (N).

Fit an empirical power law over the resolved range:

[
epsilon(N)propto N^{-alpha}.
]

The critical comparison is the exponent and the particle count required for target errors.

Report at least:

- error at fixed (N);
- fitted (alpha);
- (N) required to reach (10^{-2}), (10^{-3}), and where possible (10^{-4}) relative error.

The central research question is:

> Does global QMC or Gaussian cubature show substantially faster-than-(N^{-1/2}) convergence on smooth Stage-0 observables?

If yes, quantify whether the resulting trajectory reduction is 10x, 30x, 100x, or more for realistic accuracy targets.

---

# 7. Second experiment: locate where the high-order advantage is lost

Only after the smooth-observable experiment.

Pass IID, global QMC, fixed target stratification, and tensor Gauss-Hermite through the existing Stage-1/Stage-2 pipeline.

Compare **two deposition schemes**:

1. nearest;
2. CIC.

Hypothesis:

- nearest deposition introduces discontinuous cell-indicator functions and may destroy much of the high-order cubature convergence;
- CIC is continuous piecewise-linear in the deposited coordinates and may preserve more of the QMC/cubature advantage.

Use the same Stage-0 evaluations for both deposition schemes where possible.

Compare:

- 5D table error/marginals;
- spectrum L1;
- centroid;
- several observation directions;
- convergence exponent versus (N).

If Gauss-Hermite/QMC has a large advantage for smooth observables but the advantage collapses under nearest deposition and partially returns under CIC, that is strong evidence that **Stage 1, not Stage 0 source integration, is the remaining bottleneck**.

---

# 8. Optional third experiment: smooth push-forward instead of a hard histogram

Do this only if section 7 demonstrates that deposition is the bottleneck.

Add an experiment-only smooth deposition/kernel, not a production redesign.

Possible choices:

- triangular/B-spline deposition wider than one cell;
- Gaussian kernel deposition with bandwidth tied to table spacing.

The purpose is diagnostic:

> Does smoothing the push-forward restore the deterministic convergence seen in the Stage-0 smooth observables?

Do not tune a production kernel or change Stage-1 semantics in this phase.

If the answer is yes, report it as evidence for a separate future Stage-1 representation task.

---

# 9. Scenario selection

Do not immediately repeat the complete expensive seven-arm Phase-I bank.

Use a small but informative scenario set:

1. `baseline`;
2. `tight_focus`;
3. `wide_bunch`;
4. `crossing`;
5. `twiss_corr` if cost permits.

These span smooth/general, concentrated illumination, severe beam/laser scale mismatch, oblique geometry, and correlated source coordinates.

The tensor Gauss-Hermite rule should be tested first on `baseline` and `tight_focus`; expand only after the method is numerically sound.

---

# 10. Production-bin check

The Phase-I validation used reduced Stage-1 bins.

Once the method comparison has been narrowed to the leading one or two deterministic methods, repeat a smaller subset at the current production Stage-1 resolution.

Do not retain all replicate tables in memory.

Change the experiment harness to stream/fold the required statistics so memory scales weakly with replicate count.

The goal is to check whether the observed convergence ordering survives production binning, not to repeat every earlier arm.

---

# 11. Expected interpretations

The experiments should distinguish these cases.

## Case A — smooth observables improve by 10–100x, histogram spectrum does not

Conclusion:

- the Gaussian source and Stage-0 map are highly compressible;
- the current Stage-1 histogram/deposition is destroying the high-order integration advantage;
- stop tuning source sampling;
- open a separate Stage-1 push-forward/representation task.

## Case B — tensor Gauss-Hermite and QMC also improve the final spectrum strongly

Conclusion:

- adopt the simplest robust deterministic source rule;
- compare global QMC against Gauss-Hermite on arbitrary-(N), reuse, GPU compatibility, and measured accuracy;
- regional luminosity allocation becomes optional/secondary.

## Case C — smooth observables themselves converge only modestly better than IID

Conclusion:

- the Stage-0 map has stronger effective non-smoothness/high-dimensionality than assumed;
- the original expectation of 10–100x source compression is not supported;
- retain the validated simpler stratified/QMC improvement and stop escalating the cubature approach.

## Case D — Gauss-Hermite converges rapidly only in some scenarios

Conclusion:

- identify which physical geometry causes the loss of regularity;
- report the boundary;
- do not auto-select a method until there is a deterministic pre-run diagnostic.

---

# 12. What not to do in this phase

Do not:

- further tune `lambda`, `s`, or adaptive splitting as the main task;
- implement the Phase-I proposed `N_eff` guard as a production policy yet;
- replace IID as the default;
- redesign Stage-1 table semantics before the diagnostic experiments;
- use signed sparse-grid weights as production macroparticle weights without a separate design review;
- change DER015/DER016/DER024 physics;
- modify Stage-2 observer-dependent physics;
- optimize Stage-0 trajectory quadrature simultaneously with source cubature;
- claim speedup from the cheap source-generation time: the relevant cost is expensive Stage-0 evaluations at fixed final accuracy.

---

# 13. Reuse current branch work

Do not throw away Phase-I implementation.

Reuse:

- `_bunch_from_standard_deviates`;
- the validated `norm_ppf`;
- Halton/shift utilities where useful;
- experiment scenario definitions;
- paired-replicate statistics;
- Stage-0 chunk controls;
- validation result machinery.

The existing regional adaptive sampler should remain available during the experiments as a comparator.

Do not delete it until the new evidence supports a simplification decision.

---

# 14. Required deliverables

1. Experiment-level implementation of:
   - global target-Gaussian QMC;
   - tensor Gauss-Hermite cubature;
   - optional sparse-grid probe if straightforward.
2. A smooth-observable convergence benchmark with machine-readable output.
3. Error-vs-(N) plots/tables and fitted convergence exponents.
4. A deposition comparison: nearest vs CIC for the leading deterministic methods.
5. At least one production-bin confirmation after the candidate set is narrowed.
6. An updated validation record that clearly separates:
   - Phase-I conclusions;
   - smooth Stage-0 cubature results;
   - Stage-1 deposition results.
7. A short recommendation choosing among Cases A–D above.
8. Focused regression tests only for reusable numerical utilities that survive the experiment.

Do not turn every experimental branch into permanent public API.

---

# 15. Acceptance criteria

This phase is complete when the report can answer all of the following quantitatively:

1. Does global target QMC outperform the existing fixed `s=1` regional stratifier, match it, or lose to it?
2. How fast does tensor Gauss-Hermite converge on smooth Stage-0 observables as one-dimensional order increases?
3. For representative scenarios, how many Stage-0 evaluations are required for (10^{-2}), (10^{-3}), and where resolvable (10^{-4}) error?
4. Is the convergence substantially faster than IID's effective (N^{-1/2}) behavior?
5. Does nearest deposition destroy that advantage?
6. Does CIC retain more of it?
7. Does the ordering survive at production Stage-1 resolution?
8. Is a 10–100x reduction in expensive Stage-0 trajectories supported for any scientifically relevant accuracy target?
9. If the answer is no, what numerical stage is actually limiting convergence?

Stop after producing this evidence and recommendation. Do **not** automatically redesign Stage 1 or promote a new production default without review.

---

# 16. Completion / merge note

This handoff remains temporary execution context and must be removed from the branch before PR #21 is ready to merge.

Durable conclusions belong in:

- DER024 only where they concern the already-derived exact Gaussian sampling mathematics;
- RES093 / a follow-up decision for lasting software-design choices;
- `docs/validation/` for the numerical evidence;
- issue #20 and PR #21 for the work history.

Do not promote DER024 automatically as a consequence of numerical performance results.

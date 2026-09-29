# Luminosity-Aware Adaptive Sampling — Implementation Report

**Branch:** `feature/luminosity-aware-adaptive-sampling` (worktree `../GammaForge-adaptive-sampling`)
**Commit:** `66cb3bd` — *Add opt-in luminosity-aware adaptive sampling*
**Base:** `main` @ `4540929`
**Implements:** `docs/handoffs/luminosity-aware-adaptive-sampling.md` (RES092)

> **Read this for the gain, not the method.** The sampling method is implemented as specified
> and works. The *measured* benefit is much smaller than the handoff anticipated, and two of
> the handoff's three headline claims did not survive measurement. Section 3 has the numbers;
> section 6 has what I think should change in the design.

---

## 1. What was built

`src/gammaforge/io/adaptive_sampling.py` (1045 lines) plus an opt-in switch.

| Piece | What it does |
|---|---|
| `PilotConfig` | Numerical knobs: regions, pilot points/nodes/panels, `proposal_scale`, `luminosity_fraction`. Module defaults, not GUI-exposed. |
| `SamplingRegion` | One reference-cube box: bounds, `B_m`, exact `P_m`, pilot moments, `Q_m`. |
| `AdaptiveSamplingPlan` | Region tree, masses, pilot stats, `Q_m`, seed. Reusable for **any** particle count; `allocate()` re-derives the apportionment without re-piloting. |
| `norm_ppf` | Halley iteration on libm's `erfc`. 2.8e-14 vs `statistics.NormalDist().inv_cdf`. |
| `trajectory_luminosity_predictor` | `F ∫C(t)I(t)dt` by composite Gauss-Legendre, matching Stage 0's definition. |
| `build_adaptive_bunch` | Materializes exactly `N` particles, weights `P_m/n_m`. |

Wired in as `SamplingSpec.strategy` (`"iid" | "adaptive"`, **default `"iid"`**), a `CHOICE`
field so saved requests round-trip and a future third strategy can't silently reinterpret
them. Both strategies return an ordinary `Bunch` with relative weights summing to one, so no
engine distinguishes them.

**Verification:** 485 passed (default sweep), 512 passed (`--run-heavy`), `validation.run`
exit 0. The IID sampler is verified **bit-for-bit** unchanged against a reproduction of its
pre-refactor arithmetic. `scripts/benchmark_adaptive_sampling.py` produces every number below.

---

## 2. The method is sound — and that is not the same as it paying off

The structural guarantee holds exactly, and is what makes the design safe:

- The reference-cube boxes **tile** the cube, so `∑P_m = 1` identically, not approximately.
- Weights are `P_m/n_m`, so `∑w = ∑P_m = 1` **analytically**, with no renormalization pass.
- `Q_m ≥ (1-λ)B_m` and `B_m > 0`, so no region can be starved.

Therefore a **deliberately broken pilot cannot make the result wrong** — only less efficient.
This is pinned by `test_a_deliberately_bad_pilot_still_represents_the_beam_exactly`, which
builds a real plan with a 4-node quadrature, 1 point/region and a 50% window, then holds it
to the same weight-sum and moment standards as a good pilot.

Measured, adaptive is exactly correct in the weighted distribution: per-coordinate latent
marginals converge up to **~10× faster** than IID, with the error genuinely shrinking in `N`.

**But "correct and better-converging" is not the same as "cheaper at fixed accuracy" for the
outputs that matter.** That is where the measurement disagrees with the handoff.

---

## 3. The gain, measured

Baseline scenario, Stage 0 at 200 steps, 5D `ShapeTable` at `(24,16,16,32,8)`, 600-point
spectrum grid. Errors are relative to a 200k-particle IID reference, averaged over 2 sampler
seeds × 3 reference seeds.

**Reference resolution is the first thing to establish**, because it bounds every claim:
two independent 200k references differ by **1.35e-4 in yield** and **9.7e-3 in spectral L1**.
Any absolute tolerance finer than that measures the reference, not the sampler. All
comparisons below are therefore comparative (same budget, both strategies, same reference).

### Total yield — the one clear win

| N | IID | adaptive | ratio |
|---|---|---|---|
| 2 500 | 4.15e-3 | 5.30e-4 | **0.13** |
| 5 000 | 1.92e-3 | 4.05e-4 | **0.21** |
| 10 000 | 3.51e-3 | 3.19e-4 | **0.09** |
| 20 000 | 3.46e-3 | 1.23e-6 | **0.0004** |
| 40 000 | 1.75e-3 | 4.28e-6 | **0.002** |
| 80 000 | 1.37e-3 | 1.65e-4 | **0.12** |

Consistent 5–10× at small and mid budgets. But look at the absolute values: **IID never gets
below 1.4e-3, and adaptive's own error flattens at ~5e-4 — which is the reference noise
floor.** Past `N = 20 000` the measurement has bottomed out, not the estimator.

### Converted to "particles needed for a target error"

| target | IID | adaptive | reduction |
|---|---|---|---|
| yield 1e-2 | 2 500 | 2 500 | 1.0× |
| yield 3e-3 | 3 348 | 2 500 | 1.3× |
| yield 1e-3 | not reached | 2 500 | — |
| spectrum 1e-2 | 74 419 | not reached | — |

The honest headline: **adaptive reaches a 1e-3 yield target at ~2 500 particles, where IID
does not reach it anywhere in the measured range.** But IID's *own* error is only ever 1.4e-3
or worse, so the comparison is "adaptive is much better than a mediocre baseline", not
"adaptive delivers N× more accuracy".

### Resolved spectrum — parity, and that is the important negative

| N | IID spectral L1 | adaptive | ratio |
|---|---|---|---|
| 2 500 | 5.38e-2 | 5.90e-2 | 1.10 (worse) |
| 5 000 | 2.99e-2 | 4.69e-2 | 1.57 (worse) |
| 10 000 | 2.98e-2 | 2.77e-2 | 0.93 |
| 20 000 | 1.52e-2 | 2.08e-2 | 1.37 (worse) |
| 40 000 | 1.63e-2 | 1.52e-2 | 0.93 |
| 80 000 | 9.45e-3 | 1.10e-2 | 1.16 (worse) |

**No improvement.** This is the handoff's own §20.3 limitation arriving as a measurement: the
refinement criterion `R_m = P_m σ_m` measures *luminosity variation* and knows nothing about
how widely a region spreads across Stage-1 cells. Spending particles where luminosity varies
does not reduce the Poisson noise per Stage-1 cell, which is what the spectral residual
actually is.

Worse: at 4 of 6 budgets adaptive is **worse**, because deliberately uneven particle
allocation increases the variance of cell-population counts. A strategy that helps the
integral can hurt the density, and here it does both.

### Radial latent distribution — no gain either

`r² = d·d` is a χ²₆ functional, and a cell whose corner reaches `|d| = ∞` spans the entire
range of `r²`. At the shipped defaults **all 256 regions have at least one unbounded latent
face**, so no box isolates the radial tail and stratification cannot help it. The handoff's
§27 tail claim does not hold for the radial functional.

Per-coordinate marginals **do** improve (up to ~10×), because a box genuinely is a product
of coordinate intervals there.

### Cost

Plan build 0.60 s for 256 regions, paid **once** and reused across every budget — this is what
makes the accuracy-versus-cost question answerable at all. (The benchmark builds one plan per
sampler seed; the two plans together took 1.20 s.) Adaptive runs are ~1.3–1.8× slower in
wall-clock at equal `N`, from the `erfc`-based quantile function. Both are dwarfed by Stage 0
at 200 steps.

Reproduce: `python scripts/benchmark_adaptive_sampling.py --json out.json`.

---

## 4. Why the gain is bounded — the diagnosis

Three structural reasons, in order of importance.

**1. Stage 0's per-particle cost is uniform, and so is the information.** The expensive work
is a trajectory integral. Every particle costs the same 200 field evaluations whether it
carries luminosity or not. Stratification helps only when cheap particles can be *dropped*;
here they can't — Stage 0 must integrate all of them regardless.

**2. The dominant error is per-cell Poisson, not variance in the integral.** Stage 1 bins
`N` particles into a 5D table. A cell's mass is a sum of a few particles, so its relative
error is ~`1/√n_cell`. Reallocating particles between *regions* does not change how many land
in a given *cell*. The allocation optimizes the wrong variance.

**3. The luminosity is far more concentrated than the yield's error budget cares about.**
`A_m = P_m √M_{2,m}` concentrates particles aggressively (weight spread ~5×, `N_eff` 16k/20k).
But if Stage 1 is already resolving the table well, moving particles between regions mostly
trades noise for noise.

**Consequence for the design:** the stratifier helps the quantity it was designed around
(∫ℓ dμ) and is neutral-to-harmful for the quantity Phase 3b actually needs (a converged 5D
density). To get a real Stage-1 win, the allocation would have to become **cell-aware** — the
pilot already computes `a0_shape` and `chirp_mean`, and §14 of the handoff says so. That is
a substantially larger change, not a tuning exercise.

---

## 5. Two things I fixed in the handoff's design

**The pilot needed composite quadrature, not more nodes.** The integration domain is
`overlap_time_window`, a deliberately over-wide conservative bound, so a single `n`-node rule
spreads nodes over a region the pulse occupies only a small part of, and under-resolves the
peak that dominates the answer. Measured rank agreement with over-resolved Stage 0 among the
*bright* particles — which is what the allocation keys on:

| configuration | mean over 5 geometries |
|---|---|
| 64 nodes, 1 panel | 0.35 |
| **64 nodes, 2 panels** | **0.84** |
| 128 nodes, 2 panels | 0.94 |
| 64 nodes, 4 panels | 0.69 (worse — nodes/panel too few) |

Splitting the same budget into 2 panels is ~2.4× better at identical cost, because the domain
stays a bound and truncation remains impossible. Narrowing the window instead would have been
cheaper still, but that trades the correctness boundary for accuracy.

**`norm_ppf` had to be iteration, not a published rational approximation.** AS 241 and Acklam
were both implemented first and both failed. The dangerous property: a *wrong but monotone*
quantile function still satisfies `∑P_m = 1`, so the obvious invariant cannot detect it — it
surfaces later as a negative regional mass. Halley iteration on `erfc` has no coefficients to
mistype and measures 2.8e-14 across `[1e-300, 1-1e-300]`.

---

## 6. Three bugs, each invisible to the obvious invariant

Worth recording because each is the kind that ships silently.

1. **Conditional drawn in the wrong CDF coordinates.** Boxes are axis-aligned in the
   *reference* coordinate `u = Φ(d/s)`, so their edges in the target's CDF are
   `Φ(sΦ⁻¹(a))` — equal to `a` only at `s = 1`. Drawing with `a`/`b` samples the *reference*
   conditional: `∑P_m = 1` and `∑w = 1` both still hold, and only *where* the represented mass
   sits is wrong. Signature: error flat in `N`. Caught by
   `test_adaptive_representation_has_no_floor_as_n_grows`.

2. **`_normal_interval_mass` cancelling to exactly zero** for bounds deep in the lower tail
   (`(-38, -36)` → 0 instead of 4e-284), because it used the `Q` pair there. Only reachable
   with a fine enough lower-tail partition, which is why the mass sum hid it.

3. **`norm_ppf` coefficient order / constant term** — as above.

---

## 7. What I'd change in the design

1. **Re-scope the claim to the yield, or make the allocation Stage-1-aware.** As built, the
   method earns its keep on ∫ℓ dμ and not on the 5D density. Either say so, or replace
   `R_m = P_m σ_m` with a criterion that includes predicted cell spread. I'd do the former
   for now: the latter is a research task, and the honest version of the feature is "cheaper
   total-yield runs" rather than "faster Stage-1 convergence".

2. **Keep `"iid"` as the default — the measurement supports that clearly.** A 5–10× yield gain
   at a specific accuracy, with spectral parity, does not justify changing a default every
   existing run depends on. Promotion needs a cell-aware criterion first.

3. **Tune `λ` against the real objective.** `λ = 0.75` was the handoff's starting point, not a
   measured optimum. Given the spectral regression at 3 of 5 budgets, a **lower** `λ` (broader
   allocation) may well dominate — a broad allocation is closer to IID's cell occupancies and
   should lose less on the spectrum. I did not sweep this; it is the cheapest remaining
   experiment and the one most likely to recover the spectrum.

4. **Reconsider `s = √2`.** The reference scale is what makes a cell's corner reach `|d| = ∞`,
   which is precisely what defeats the radial claim. A larger `s` makes boxes more compact in
   `r²` at the cost of a worse weight bound `P_m/Q_m ≤ s⁶/(1-λ)`.

---

## 8. Scope not done, deliberately

- **Loaded/imported bunches** — the current `fit_gaussian` is unweighted and cannot
  reconstruct an arbitrary centroid, so it is not a universal replacement path.
- **Stage-1-aware refinement** — see above.
- **Wider scenario bank** — all numbers above are the baseline scenario only. The yield gain is
  plausibly geometry-dependent (a tight-focus or crossing-angle case, where luminosity is far
  more concentrated, should favour adaptive *more*), and I have not measured that.

## 9. Two pre-existing issues found, not fixed

- **`.gitignore` ignores `scripts/`, but `tests/test_report_figures.py` imports from it.** The
  test cannot run in *any* worktree, only in the main checkout. Pre-existing; excluded with
  `--ignore`. Worth fixing separately — probably by force-adding the two modules that test
  needs, or moving them out of `scripts/`.
- **Adding `propagation_direction()` to the `LaserField` protocol broke a conforming
  implementation** in `tests/test_laser.py`. That is the protocol doing its job (a second
  implementation must state its propagation direction), but it is a reminder that extending a
  `@runtime_checkable` Protocol is a breaking change for external implementers. RES092
  records it; a future non-Gaussian laser will need to supply it.

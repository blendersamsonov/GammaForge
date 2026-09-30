# RES093 — Luminosity-aware adaptive sampling as an opt-in bunch strategy

Status: implemented
Class: feature

## Problem

`sample_gaussian_bunch` draws macroparticles IID from the beam's six latent standard normals,
and every engine then runs Xigma Stage 0 — one trajectory integration per particle, at
`n_steps` field evaluations each. Stage 0 dominates a run's cost, and IID sampling is a poor
way to spend that budget twice over: it crowds the dense core, where neighbouring samples
often land in the same Stage-1 cell and become near-redundant once deposited, and it starves
the tails, which carry real target mass (`P_m`) and much of the spectrum's shape.

The plan for a better sampler is
`docs/handoffs/luminosity-aware-adaptive-sampling.md`: represent the *same* physical Gaussian
by a stratified estimator — a broad reference Gaussian supplying a coverage geometry, the
unit cube partitioned into disjoint boxes, each box carrying an **exact** target mass, points
drawn from the **target conditional** with a low-discrepancy sequence, and constant per-box
weights `P_m / n_m`. A cheap Stage-0-like pilot then decides how to split the budget between
boxes.

## Decision

`gammaforge.io.adaptive_sampling` implements it as a second sampling strategy, selected by
`SamplingSpec.strategy` (`"iid" | "adaptive"`, default `"iid"`). It produces an ordinary
`Bunch` with relative per-particle weights summing to one, so no engine distinguishes the two
except by inspecting `sampling.strategy`.

Four choices inside the handoff's design were settled by measurement rather than by following
the document, and each is load-bearing:

- **The target conditional is formed in the target's own CDF coordinates**
  (`Phi(s Phi^-1(a))`), not in the cube's. A box is axis-aligned in the *reference* coordinate
  `u = Phi(d/s)`, so these coincide only at `s == 1`. Formed the wrong way, the sampler draws
  from the *reference* conditional: `sum(P_m) == 1` and `sum(w) == 1` both still hold, and
  only *where* the represented mass sits is wrong. The signature is an error that stops
  shrinking as `N` grows. `test_adaptive_representation_has_no_floor_as_n_grows` exists
  specifically to catch it.
- **The quantile function is inverted by Halley iteration on libm's `erfc`**, not by a
  published rational approximation. AS 241 and Acklam were both implemented first and both
  failed here: coefficient-order and constant-term transcription slips make the map
  non-monotone, and `sum(P_m) == 1` cannot detect that (the partition-of-unity identity needs
  only monotonicity, not correctness). Iteration has no coefficients to mistype and measures
  2.8e-14 against `statistics.NormalDist().inv_cdf` over `p in [1e-300, 1-1e-300]`.
- **The pilot uses a *composite* Gauss-Legendre rule over two panels**, not one high-order
  rule. The integration domain is `overlap_time_window`, a deliberately over-wide
  conservative bound, so a single rule spreads its nodes over a region the pulse occupies only
  a small part of and under-resolves the peak that dominates the answer. Splitting the same
  node budget into two panels roughly doubles the rank agreement among the *bright* particles
  — which is what the allocation keys on — at no extra cost, and keeps the domain a bound, so
  truncation stays impossible.
- **`pilot_quad_nodes` defaults to 128, not 64**, because the pilot only ever evaluates
  `regions x points_per_region` particles (~2e3). The extra nodes cost ~0.1 s against Stage
  0's per-particle cost.

The adaptive strategy stays **opt-in and off by default**. A new sampling path is not promoted
on the strength of one scenario's benchmark, and an unexpected regime must keep the behaviour
users already rely on. `SamplingSpec.strategy` is a `CHOICE` field so a saved request
round-trips through YAML/HDF5 and a third strategy cannot silently reinterpret it.

## Alternatives considered

### Why not importance-sample from the broad Gaussian with pointwise `p/q` weights?

The handoff's older formulation: draw production particles from `q_s` and reweight each by
`p(d)/q_s(d)`. Rejected, and superseded by the stratified form: because the boxes tile the
cube, `P_m` is exact, and sampling from the target conditional gives every particle in a box
the single constant weight `P_m / n_m` with **no** contribution from the density ratio at all.
The `p/q` form pays variance for the density ratio on every particle to end up with the same
estimator, and its weights are neither constant nor exactly summable to one.

### Why not narrow the pilot's window to buy resolution?

Tightening the pilot's active-region threshold does improve rank agreement monotonically, and
is cheaper than composite panels. Rejected because the window is the correctness boundary: a
threshold that clips real luminosity biases the pilot, and the bias is asymmetric in exactly
the region that matters. Composite panels improve resolution with the domain unchanged, so a
conservative bound stays conservative.

### Why not `illumination_window()` as the integration domain?

Explicitly excluded by the handoff, and for a good reason: it is an *estimate* of where
illumination happens, with the widths frozen. Integrating over it biases the result towards
whichever particles the approximation happens to bracket well. `overlap_time_window` is a
genuine bound — nodes outside the pulse simply evaluate to zero intensity.

### Why not change the default to `"adaptive"` now?

The measured gain is a consistent win on the total yield (roughly 1.3x–5x fewer Stage-0
particles for a given yield target; adaptive reaches 1e-3 relative yield at a budget where IID
does not reach it inside the measured range) and **parity** on the resolved spectrum. A gain
that real and that narrow is not yet a mandate to change a default every existing run depends
on. Promotion needs the wider scenario bank and the `spectrum` question resolved — see below.

## Rationale

The stratifier's guarantee is structural rather than statistical, and that is the reason to
trust it: `P_m` is exact, the boxes tile the cube, and the weights are `P_m / n_m`, so
`sum_i w_i = sum_m P_m = 1` *analytically*, with no renormalization pass. A deliberately bad
pilot therefore cannot make the represented distribution wrong — only less efficient.
`test_a_deliberately_bad_pilot_still_represents_the_beam_exactly` holds a 4-node,
1-point-per-region, 50%-threshold plan to exactly the same standard as a good one, which is
what makes running an approximate `O(regions x points x nodes)` pilot against an
`O(N x steps)` Stage 0 safe.

The pilot is observer-independent, so one plan serves every observation direction and the
exact observer-dependent `Q` stays in Stage 2 (RES090). A plan depends only on
`(beam, laser, seed)` and is reusable for any budget, which is what makes the
accuracy-versus-cost question answerable at all: the pilot is paid once, not once per
sampler seed times once per budget.

## Consequences

- `io.bunch._bunch_from_standard_deviates` now holds the single latent→physical map, called by
  both samplers. Two copies of the Twiss/gamma algebra could drift apart silently, and a
  sampler that disagreed about `alpha` would still produce a plausible-looking bunch. The
  IID path is verified **bit-for-bit** unchanged
  (`test_iid_sampler_is_unchanged_by_the_shared_transform_refactor`).
- `LaserField` gains a `propagation_direction()` method, with `io.laser.laser_propagation_direction`
  as the one place that question is answered. The pilot needs the encounter factor
  `1 - v.n0_hat`, and a second copy of the §2.2 two-plane rotation convention is a
  silent-disagreement risk. xigma's three-branch `hasattr` chain collapsed onto it.
- **Honest limits, measured rather than assumed.** Adaptive wins on the total yield and is at
  **parity** on the resolved spectrum, because the spectral residual is dominated by how many
  particles land in each Stage-1 cell and reallocation does not change that. This is the
  handoff's own §20.3 limitation — the refinement criterion sees luminosity variation, not
  Stage-1 cell spread — surfacing as a measurement. The radial (chi-squared-6) latent
  distribution also does **not** improve: a cell whose corner reaches `|d| = inf` spans the
  whole range of `r^2`, so no box isolates the radial tail. The per-coordinate marginals,
  where a box is nearly a product interval, improve up to ~10x.
- No engine may normalize by particle count; `test_engines_do_not_normalize_by_particle_count`
  enforces it. xigma consumes per-particle weights (verified by a single-particle weight
  perturbation), kascade does too via `n_electrons * bunch.weight`, and `AnalyticalEngine`
  never reads the bunch.
- The prefilter still does not renormalize, so the retained weight sum honestly records the
  dropped fraction (§3.2) for the adaptive path as well.
- The `norm_ppf` clamp is asymmetric by necessity: the lower tail reaches `1e-300` (quantile
  ~ -37) but `1 - 1e-300` is not representable, so the upper tail stops near 8.2 sigma.
  Nothing depends on the deeper half; unbounded box *edges* are handled in latent space where
  `+/-inf` goes to `erfc` directly and the mass is exact.
- **Not done, and deliberately:** loaded/imported bunches are out of scope (the current
  `fit_gaussian` is unweighted and cannot reconstruct an arbitrary centroid, so it is not a
  universal replacement path); and the refinement criterion remains luminosity-only, where the
  pilot already computes the Stage-1 coordinates (`a0_shape`, `chirp_mean`) and a later
  criterion should use them.

## Amendments

> **2026-09-30 — the "parity on the resolved spectrum" claim above is superseded; the
> allocation is regime-dependent, and `proposal_scale = sqrt(2)` is a net negative.** The
> ten-scenario bank with 12 paired replicates per arm
> (`docs/validation/adaptive-sampling-2026-09-29.md`, §5) measures three separate things, and
> the decision's own defaults survive none of them cleanly. Measured as paired differences
> against plain stratified QMC (`s=1`, `λ=0`) on spectrum L1 at `N = 40 000`:
>
> - **Stratification is a settled win.** IID is resolved worse in 9 of 10 scenarios,
>   consistently 1.5–1.8×. This is the load-bearing result and it is stronger than the
>   entry's "measured gain" framing suggests.
> - **`proposal_scale = sqrt(2)` is a net negative.** `s=√2` with no allocation is resolved
>   worse than `s=1` in 9 of 10. The shipped default is wrong on this evidence, independent of
>   the luminosity term.
> - **The luminosity allocation's value depends on the regime, and interacts with `s`.** The
>   same `λ=0.75` helps at `s=√2` (3 better / 1 worse / 6 noise) and *hurts* at `s=1`
>   (2 better / 5 worse / 3 noise). Its wins are the concentrated scenarios — `tight_focus`,
>   `focus_3um`, `twiss_corr` — at a consistent 12–15% error reduction, which is the handoff's
>   own prediction arriving as a measurement. The one large regression is `wide_bunch`
>   (+39%, resolved), driven by a weight collapse to `N_eff/N = 0.045` against 0.836–0.992
>   everywhere else.
>
> So "parity on the resolved spectrum" was true only of the two scenarios the first
> measurement happened to use (`baseline` and `wide_bunch`), which are the two most *diffuse*
> in the bank, and true for neither of them by the better measurement above. The refusal to
> promote the default stands, but the reason is now sharper: **no single `(s, λ)` dominates the
> bank**, which is a different problem from "the gain is narrow". Whether that is a
> `s × λ` interaction with a physical reading — an over-broad reference needing the luminosity
> term to correct its own over-dispersion — or an artifact is unresolved, and it is the
> question that decides whether this can be a fixed default at all.
>
> Region splitting (`initial_regions=64 → max_regions=256`) has **no measurable effect**: the
> verdicts agree in 8 of 10 scenarios against the same arm without it. On this evidence it is
> not earning its keep.
>
> A Stage-1-cell-aware criterion does **not** resolve the regime dependence: it reproduces the
> same split (helps `tight_focus`/`focus_3um`, hurts `baseline`, collapses to `N_eff = 155` on
> `wide_bunch`) and the pilot cannot supply it in any case, seeing 5.9–7.7 cells per region
> against the oracle's 46–63. See RES094 for the `N_eff` guard this motivates.

> **2026-09-30 — the luminosity allocation was inert from first commit until `970e0f9`.**
> Region pilot moments were computed and then not stored on `SamplingRegion`, so `Q_m` fell
> back to `B_m` for *every* `luminosity_fraction`, and only luminosity-driven splitting was
> live. The shipped defaults consequently produced numbers that looked like a working
> `λ=0.75` experiment while measuring uniform allocation. Every structural invariant in the
> entry above still held — `∑P_m = 1`, `∑w = 1`, `Q_m ≥ (1-λ)B_m` — the run was fast, the
> tables were well-formed, and nothing warned. The decision above is therefore sound in what it
> specifies and was, for its first two commits, not what the code did; the measurements it
> cites from that period should be read as characterising the `λ=0` path. Fixed with
> regression tests asserting the moments are populated and that `λ = 0`, `0.5`, `1.0` produce
> different allocations. `preflight.py` in the experiment harness now refuses to run against
> pre-fix code, because this failure is indistinguishable from a working run at the table level.

> **2026-09-30 — two measurement-methodology corrections that invalidate first-pass numbers.**
> The first benchmark compared arms by absolute error against a reference, which is not a
> significance test, and its numbers are not comparable to the current ones. (i) The
> reference floor — the spread of the reference across its own seeds — bounds the *reference*,
> not an arm; a 20–40k arm's seed-to-seed spread is several times larger, so "above the floor"
> does not mean distinguishable. (ii) Arms reused the reference's seeds (`SEEDS =
> REF_SEEDS[:2]`), so an IID arm at seed 3 drew the same particle stream as the reference at
> seed 3. Both are corrected in the harness; the entry's "parity" figure above came from that
> regime. Concretely, the `baseline` spectrum ratio of 0.64× read as a large win is 9.26e-3
> vs 9.06e-3 at 12 paired replicates — nominally the other way. It was seed noise.

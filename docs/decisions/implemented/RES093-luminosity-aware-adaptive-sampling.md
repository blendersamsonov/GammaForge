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

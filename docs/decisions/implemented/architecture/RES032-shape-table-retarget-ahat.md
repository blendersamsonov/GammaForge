# RES032 — Stage 1 deposits onto `a0_shape`, not `ahat`; a conservative regrid (`retarget_ahat`) onto a fixed, non-uniform `ahat` axis replaces the direct deposit — supersedes RES028

Status: implemented
Class: architecture

## Problem

Discussion with the project's author (a physicist) surfaced that RES028's direct-onto-`ahat`
deposit was wrong for where the resonance physics actually needs resolution. A grid whose
bin density follows wherever the sampled data happens to span (RES028's `_uniform_edges` on
the raw `ahat` array) puts no more resolution near the peak than anywhere else, even
though the redshift correction only matters where `ahat` is comparable to 1.

## Decision

**What changed, concretely:**

- **`stages.ShapeTable`** (new type, not a `kind` string flag — matches this codebase's
  existing aversion to stringly-typed discriminators, `Axis`/`OutputKind` enums,
  `io/schema.py`'s "never a stringly dict") replaces direct deposit onto `ahat`.
  `stages.deposit_shape_table` bins Stage 0's `TrajectorySamples.a0_shape` (already
  peak-independent, already bounded — no dynamic-range problem) onto a fine, uniform,
  linear grid (`DEFAULT_SHAPE_BINS = (48, 48, 48, 96)`), with native `luminosity` as the
  deposited weight and the run's own `a0_peak` recorded as `source_a0_peak`. One deposit
  per `TrajectorySamples`, reusable for any peak a0.
- **`stages.retarget_ahat`** (adapted from the predecessor's `retarget_a0`, verbatim
  overlap-weighted-mass-transfer structure — `deposition.py:423-511` in the old repo) turns
  a `ShapeTable` plus one peak a0 into the `Table` (unchanged name, now exclusively the
  `ahat`-axis, Stage-2-ready result) Stage 2 queries. Two rescales, both exact: the source
  edges scale by `a0_peak**2` (`ahat = a0_peak**2 * a0_shape`), and the deposited mass
  rescales by `(a0_peak / source_a0_peak)**2` — the same relation `retargeted_luminosity`
  (RES028) already established, needed here because, unlike the predecessor's `retarget_a0`
  (which only ever retargeted onto its own run's laser), this one is meant to serve a
  genuinely different peak a0 than the one Stage 0 ran at.
- **Target grid law** (`stages._ahat_target_edges`), log-spaced in distance from the top —
  bin width shrinking monotonically toward `ahat_max`, the opposite of what a plain
  logarithmic axis gives (constant *relative* width means bins widen in absolute terms
  toward the top; this repo wants the reverse, dense absolute resolution at the top):
  ```
  v_i = (ahat_max - ahat_min) * 10**(-decades * i / n_bins),  i = 0..n_bins
  ahat_i = ahat_max - v_i                                     (edges[-1] snapped to ahat_max)
  ```
  The snap widens the single top bin — negligible at `decades >= 3`, visible at the
  `decades=1` default below (the top bin ends up *wider* than its neighbour, a deliberate,
  bounded exception `tests/test_stage1_stage2.py::
  test_ahat_target_edges_bin_widths_shrink_toward_the_top` checks explicitly, not a defect).
- **Defaults, tuned against the actual scenario bank**, not re-derived from the
  predecessor's *DEFAULT_A0_MAX*/`retarget_a0` defaults, which were sized for a different
  bank: `ahat_max = 0.5` (kept — the predecessor's value, generous headroom over this
  bank's measured `ahat` maxima: `baseline` 0.019, `low_a0` 0.0019, `near_a0_max` 0.095),
  `ahat_min = 0.0`, `n_bins_ahat = 32` (kept). **`ahat_decades = 1.0`, down from an initial
  `decades=3` (matching the predecessor's implicit concentration) that was checked
  numerically and rejected**: at `decades=3`, bin 0 alone spans `[0, 0.097]` — wider than
  `near_a0_max`'s entire `ahat` range — so every scenario in the bank collapsed into the
  floor bin, resolving nothing and defeating the entire point of the change. Swept
  `decades` against the bank's actual mean/max `ahat` (measured this session) before
  picking `1.0`: `near_a0_max` spreads across bins 1-2 (real structure), while `low_a0`
  correctly stays concentrated near the floor — its `ahat` (max 0.0019) genuinely is small
  enough that the redshift correction barely matters there, which is squashing working as
  intended, not a resolution failure.
- **Truncation.** Target `ahat` bins the rescaled source data's true maximum never reaches
  carry **exactly zero** mass (the overlap-weighted transfer cannot place mass where no
  source bin overlaps) — `retarget_ahat` sums the regridded mass over the other three axes,
  finds the last populated `ahat` bin, and slices the mass array/`ahat_edges` down to it
  before returning (keeping >= 1 bin in the degenerate all-zero case). Free:
  `spectrum_from_table`'s cell sum is unaffected in value, only in how many always-zero
  terms it evaluates — `tests/test_stage1_stage2.py::
  test_retarget_ahat_truncation_does_not_change_the_kernel_output` checks the truncated and
  explicitly-zero-padded tables agree exactly.
- **`Table`'s old `bin_volume` property removed** (no longer generally correct once
  `ahat_edges` is non-uniform; confirmed unused by production code). `Table` gained
  `ahat_widths` (per-bin array, `np.diff(ahat_edges)`) and `gamma_theta_cell_area` (the
  still-uniform 2-axis scalar) in its place; `spectrum_from_table` folds `ahat_widths` into
  the cell sum itself rather than multiplying one global scalar in afterward.
- **`Collision`** gained a `_shape_table` singleton cache (mirrors `build_overlap`'s
  pattern) alongside the existing `_tables` dict; `_table(a0_peak)` now calls
  `retarget_ahat(self._shape(), a0_peak, ...)`. A genuine improvement over RES028's
  "re-deposit from scratch per `a0_peak`": `retarget_ahat`'s cost is independent of
  `n_particles`, so multiple `_table()` calls on one `Collision` (e.g. a future
  pulse-energy scan) now pay the `n_particles`-scale deposit cost once, not per `a0_peak`
  — exactly the win RES028 said nothing currently needed. Cache-key staleness (a grid-shape
  parameter changing without a new `a0_peak` on a long-lived `Collision`) falls under
  RES030's already-documented scope ("cheap to reuse, not smart about being replaced")
  without new reasoning — the new `n_bins_a0_shape`/`ahat_min`/`ahat_max`/`ahat_decades`
  parameters are not a new gap, just more instances of the same one.
- **`schema.py`** gained `n_bins_a0_shape` (default 96), `ahat_min` (0.0), `ahat_max` (0.5),
  `ahat_decades` (1.0); `n_bins_ahat`'s existing key is kept but its default moves 12 -> 32
  and its meaning shifts from "direct-deposit bin count" to "retarget target-grid bin
  count." The cross-field constraint `ahat_max > ahat_min` has no home in `FieldSpec` (no
  cross-field hook exists in `io/schema.py`) and is enforced inside `retarget_ahat`/
  `_ahat_target_edges` at call time instead, matching where the predecessor put the
  identical check — not worth building general cross-field schema validation for one case.

Supersedes RES028: its "nothing in this repo needs [the regrid]" conclusion held for
cross-run comparability (the predecessor's own stated reason) but did not consider
concentrating resolution, which is what this decision is for.

## Alternatives considered

**Rescale the existing direct-deposit table's edges in place for a new peak a0, without a
full regrid** (RES028's own closing suggestion, since binning is invariant under a uniform
positive rescale of both data and target edges by the same factor). Would have been
cheaper still, but only works when the target grid *is* just a scaled copy of the source —
which stops being true the moment the target law is a fixed, non-uniform shape rather than
"wherever this particular a0_peak's data happens to land." The whole point of this
decision is wanting that fixed, physically-motivated shape, so the pure-rescale trick is
no longer available regardless of its cost.

## Rationale

`ahat` enters `s_res = gamma**2/(1+ahat+gamma**2*r**2)`; the redshift only matters where
`ahat` is comparable to 1, near a pulse's peak, while the bulk of a bunch's trajectories
sit at much smaller `ahat` where the correction barely perturbs anything. A **fixed**,
non-uniform target grid — dense near `ahat_max`, coarse toward `ahat_min`, everything below
`ahat_min` folded into one floor bin — is the right shape, and reaching it over the whole
a0-independent population (not tied to one pulse) is exactly what the predecessor's
`retarget_a0` conservative regrid was for.

## Consequences

Measured effect: the fourth `run.py::identity_checks` leg (RES029) and the
`XigmaEngine.run()` smoke test both got noticeably faster, since most configurations
populate only a handful of the 32 target bins. Cost: `n_bins_ahat`'s default moving 12 ->
32, and its meaning shifting from direct-deposit bin count to retarget target-grid bin
count, is a behavior change for any existing config relying on the old default.

## Amendments

> **2026-09-23 — Shape coordinate extended by RES088.** The peak-independent fourth
> coordinate is now $P a_{0,\mathrm{shape}}$, with the per-trajectory incidence factor
> deposited before retargeting. The cache/regrid architecture and mass-preserving scale
> remain unchanged; the resulting Table coordinate now represents $P\hat a$ rather than
> raw $\hat a$.

> **2026-09-28 — RES088 superseded by RES090.** Stage 1 again stores raw
> $a_{0,\mathrm{shape}}$, and the retargeted coordinate again means raw $\hat a$.
> Observation-dependent $Q$ is evaluated only in Stage 2. The table is now
> five-dimensional through a separate $\bar C$ coordinate with co-shaped moment channels.

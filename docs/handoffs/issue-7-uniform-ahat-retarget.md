# Issue #7 handoff — uniform raw-`ahat` production retarget

## Goal

Implement issue #7 on this branch: make xigma Stage 1.5 use uniformly spaced bins in the
physical raw `ahat` coordinate in production, while preserving the existing Stage-1
shape-table and Stage-1.5 conservative-retarget architecture.

The branch starts from `main` commit
`66e876d278f7613fdc50ed2c08218f0100da8551`.

This handoff is the execution specification for this implementation branch. The durable
task record remains GitHub issue #7.

## Authority

Follow these repository artifacts in this order:

- **Issue #7** owns scope, acceptance criteria, and the requested implementation outcome.
- **DER021** —
  `docs/derivations/derived/DER021-uniform-raw-ahat-retargeting-from-resonance-sensitivity.md`
  — owns the numerical-physics rationale for uniform raw-`ahat` spacing.
- **DER015** owns observer-dependent `Q` and the requirement that the reusable table store
  raw observer-independent `ahat`.
- **DER016** owns the trajectory moments and the intensity scalings of the retargeted
  nonlinear coordinate and moment channels.
- **DER017** owns the second-order finite-line moment channels consumed by Stage 2.
- **RES032** records the existing shape-table/retarget architecture and the historical
  nonuniform grid. Its architecture remains useful; DER021 supersedes only the grid-spacing
  rationale.

Do not re-derive the physics in implementation code. DER021 is intentionally still
`derived`; this implementation must not promote it.

## Current state on `main`

The repository already has the two-stage nonlinear representation that should be kept:

1. Stage 1 deposits a fine, uniform, peak-intensity-independent shape table in
   `a0_shape` / nonlinear shape, together with the co-shaped DER016 moment channels.
2. Stage 1.5 rescales that shape coordinate to physical raw `ahat` for a requested peak
   intensity and conservatively transfers mass and all moment channels into a Stage-2
   `Table`.
3. Stage 2 evaluates observer-dependent `Q`, the resonance, inverse root, Jacobian,
   polarization, and finite-line moments.

The active production grid is still the RES032 nonuniform law:

- `src/gammaforge/engines/xigma/schema.py` exposes `ahat_decades` with default `1.0`;
- `src/gammaforge/engines/xigma/stages.py::_ahat_target_edges` constructs a grid
  logarithmic in distance from `ahat_max`;
- `retarget_ahat(..., decades=...)` accepts the grid-shape parameter;
- `src/gammaforge/engines/xigma/collision.py` forwards the configured
  `ahat_decades`;
- `Table` and Stage 2 already support per-bin `ahat_widths`, so no Stage-2 redesign is
  required;
- retargeting already preserves total weight and the co-shaped moment channels, supports
  the `n_bins == 1` ignore-nonlinearity mode, supports a special floor bin when
  `ahat_min > 0`, and truncates trailing unpopulated bins.

The existing test suite contains several calls that still pass `decades=1.0`, and a
specific regression test that pins the old production under-resolution. Those tests need
to be reconciled with DER021 rather than mechanically preserved.

## Required work

### Production grid

Change the ordinary Stage-1.5 target grid to uniform spacing in raw `ahat`.

For `n_bins > 1`, the ordinary target interval should be generated as

```python
np.linspace(ahat_min, ahat_max, n_bins + 1)
```

subject to the existing floor-bin semantics described below.

There must be one production grid policy. Do not add a runtime choice between uniform and
nonuniform spacing.

### Remove the active `ahat_decades` knob

Remove `ahat_decades` from:

- xigma's active parameter schema;
- `Collision` production plumbing;
- `retarget_ahat`'s production-facing signature;
- active defaults/exports such as `DEFAULT_AHAT_DECADES`;
- current examples/config fixtures and active documentation where it is described as a
  live production control.

Backward compatibility with persisted historical configs is only worth adding if the
existing configuration loader already has a normal migration/deprecation mechanism. If
such a shim is needed, it may accept-and-ignore the legacy key, but the key must not
affect the production grid.

Git history is sufficient to recover the old law. A private legacy helper is optional for
validation comparison only; it must be unreachable from the normal engine/schema path.

### Preserve special cases

Keep the current behavior for:

- `n_bins == 1`: one bin spanning the nonlinear interval, evaluated at `ahat = 0`, as
  the explicit ignore-nonlinearity mode;
- `ahat_min > 0`: the explicit `[0, ahat_min]` catch/floor bin evaluated at zero, if
  the current semantics are retained; the ordinary bins above that floor should be
  uniform over `[ahat_min, ahat_max]`;
- trailing empty-bin truncation, provided it remains spectrum-invariant.

Do not interpret the special floor bin as a return to a generally nonuniform production
coordinate.

### Preserve the conservative retarget

Do not replace Stage 1.5 with a re-deposit from particles.

The existing overlap-based conservative transfer must continue to preserve:

- `H`;
- `H_var_a`;
- `H_var_chirp`;
- `H_cov_a_chirp`, including sign;
- the current intensity/luminosity rescaling rules from DER016;
- reuse of the cached Stage-1 shape table across requested peak intensities.

### Keep Stage 2 general

Do not refactor Stage 2 merely because production `ahat` bins become equal-width.

It is acceptable, and preferable for this issue, to keep:

- `Table.ahat_widths` as an array;
- the general nonuniform-width integration code;
- existing interpolation and sampler interfaces.

The branch should change the production target-grid policy, not the generic internal table
representation.

### Documentation / decision history

Update durable documentation that currently presents the RES032 nonuniform grid as active
production behavior.

Preserve RES032 as historical design context rather than rewriting history. Add or update
the appropriate decision record so the durable repository state says:

- the shape-table + conservative-retarget architecture remains;
- DER021 supplies the rationale for uniform raw-`ahat` spacing;
- `ahat_decades` is retired from production.

Do not edit DER021 except for a genuine discovered error that must first be surfaced to
the author.

## Invariants

The implementation must not change the following:

- raw `ahat` is observer independent;
- `Q` remains a Stage-2 query-time quantity from DER015;
- Stage-0 trajectory sampling and DER016 moments;
- Stage-1 shape-table semantics;
- physical resonance, inverse resonance, Jacobian, polarization, and line-model formulas;
- delta versus `moment2` semantics;
- NumPy/CuPy scientific behavior outside the changed discretization;
- public output definitions and normalization;
- the ability to retarget one cached shape table to multiple peak intensities.

Any spectral change should be attributable to the changed numerical discretization of
`ahat`, not to a physics-model change.

## Validation and acceptance criteria

Keep validation focused on the grid change. Do not turn this issue into the paper-wide
validation project.

### Structural tests

Add/update tests showing that:

1. for `n_bins > 1`, ordinary target edges are exactly uniformly spaced over the
   requested interval and have exact endpoints;
2. `n_bins == 1` preserves the current ignore-nonlinearity semantics;
3. with `ahat_min > 0`, the special floor bin retains its intended evaluation at zero
   and the ordinary bins above it are uniform;
4. the public `Collision` / engine path produces the uniform grid;
5. no active `ahat_decades` value can alter production behavior.

Remove or rewrite tests that intentionally pin the old RES032 grid or its known
under-resolution. Do not replace them with goldens that merely freeze one new spectrum.

### Conservation tests

Retain and, where needed, adapt the existing checks that conservative retargeting preserves
the expected intensity-scaled integrals of all four table channels.

Retain the truncation-invariance test.

### Scientific convergence check

Against the current direct-particle reduced-model reference, scan `n_bins_ahat` for at
least:

- `baseline`;
- `low_a0`;
- `near_a0_max`;
- one crossed/off-axis case where table discretization is visibly relevant.

Record at minimum:

- total yield;
- spectral centroid;
- normalized integrated L1 spectral error.

If a legacy nonuniform helper is kept temporarily, compare it at equal bin count as a
diagnostic only.

Do not reintroduce a nonlinear spacing parameter if 32 bins are insufficient. Report the
uniform-grid convergence and change the default `n_bins_ahat` only if the evidence
requires it.

### Test-suite scope

Reuse the existing Stage-1/Stage-2 and validation infrastructure. Add only the focused
regression coverage needed for this change.

No performance threshold belongs in CI.

## Deliverables

The implementation branch should ultimately contain:

- production uniform raw-`ahat` retargeting;
- removal/retirement of the active `ahat_decades` knob;
- focused unit/regression tests for uniform spacing and preserved retarget semantics;
- current-scenario convergence evidence sufficient to justify the retained or adjusted
  `n_bins_ahat` default;
- updated durable decision/documentation records reflecting DER021;
- any necessary config/example migration updates.

Before the PR is ready to merge, move lasting numerical evidence into the normal
validation/documentation location and delete this handoff file from the branch.

## Out of scope

Do not:

- change DER013, DER015, DER016, DER017, or Stage-2 physics;
- promote DER021;
- replace the uniform grid with an adaptive, logarithmic, or observer-dependent grid;
- redesign the Stage-1 table;
- redesign the query sampler;
- implement the broader paper-validation issue;
- edit the manuscript;
- merge the PR as part of this handoff workflow.

## Open questions and blockers

There is no blocking physics question.

The only non-blocking implementation result to determine is whether the current default
`n_bins_ahat = 32` is sufficiently converged once the production grid is uniform. Treat
that as measured numerical evidence, not as a reason to invent another grid-shape knob.

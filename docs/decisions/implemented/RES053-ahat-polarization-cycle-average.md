# RES053 — `ahat` carries the polarization cycle average; the paper was right and the code was short a factor 1/2

Status: implemented
Type: bug-fix

## Problem

`a0` is by definition the normalized **peak** magnitude of the electric field, but the
cycle-averaged normalized intensity that the resonance denominator `ahat` physically needs
is `<a**2>`, not `a0**2` — and `TrajectorySamples.a0_shape` was an amplitude-squared
trajectory average with no cycle factor applied anywhere on the path from it to `ahat`.
Raised as a §0 BLOCKING code/paper discrepancy (P14: *if code and paper disagree, stop and
flag it*).

## Decision

`engines.xigma.stages.ahat_from_shape` is now the single route from `TrajectorySamples.
a0_shape` to the paper's `ahat`, and it applies `io.laser.CYCLE_AVERAGE_FACTOR = 0.5`
(superseded by RES054, which replaced the module constant with `io.laser.
GaussianParaxialLaser.cycle_average_factor` and removed the need for it on this path
entirely). Every `ahat` this repo computes was previously **twice** the paper's, so both
live consumers — `validation.references.delta`'s `s_res = gamma**2 / (1 + ahat + gamma**2
r**2)` and xigma's Stage 2 kernel, which inverts the same resonance per `ahat` cell — were
overstating the nonlinear red-shift by 2x.

**Where the factor goes, and — equally load-bearing — where it does not.**

- **It goes on `ahat`, exactly once.** `ahat` is a ratio of a 4th to a 2nd envelope moment,
  so substituting `<a**2> = C a0_env**2` throughout leaves exactly one power of *C*, not
  two.
- **It does not go on the photon count.** `stages.photon_density_scale` inverts
  `_a0_from_density` *exactly* — the pulse energy cancels and the same amplitude
  convention appears on both sides — so its `a0**2 -> photons/cm**3` conversion is
  self-consistent whichever convention `a0_profile` uses. Applying *C* there as well would
  be a genuine double count, and would break §9.1's closed-form yield identity (`anchor
  ratio 1.000002`), which is what makes this checkable rather than a matter of taste.
  Confirmed: total yield is bit-for-bit unchanged by this entry.
- **It does not go inside `a0_shape`.** `a0_shape` stays the paper's `int|E|**4 / int|E|**2`
  verbatim, so the two can still be compared by eye, and the paper's own grouping `(a0**2 /
  2) int|E|**4 / int|E|**2` stays readable in the code.

## Alternatives considered

**Fold the `1/2` into `a0_shape` at Stage 0.** One site instead of three thin call sites,
and it survives the retarget for free. Rejected because the synthetic fixtures in
`tests/test_stage1_stage2.py` set `a0_shape` directly: they would have stayed green with
docstrings silently claiming an `ahat` they no longer produce — the same
quietly-wrong-comment failure mode as the ratio blindness RES033 closed. Routing through
`ahat_from_shape` instead made those annotations break loudly and forced the bin placements
to be re-derived (`0.0045/0.36` in bins 0/17 became `0.00225/0.18` in bins 0/6). It also
keeps `a0_shape` equal to the paper's `int|E|**4 / int|E|**2` verbatim.

**Apply *C* at each consumer (`delta.resonance_spectrum` and Stage 2) instead.** Two sites
today, and the next consumer inherits the bug by default. The whole point of a named helper
is that "where does the cycle average live" has one answer.

**Make *CYCLE_AVERAGE_FACTOR* a function of `ellipticity` now.** That is §9.2, and it is
still an open derivation — `ELLIPTICITY_IS_NOOP` stays `True`. What this entry buys §9.2 is
a *concrete* hook: `C = 1/2 -> 1` from linear to circular is exactly what `ellipticity`
should interpolate, and it enters in two known places (this constant and
`_a0_from_density`'s `sqrt(8 pi u)`), so §9.2 is no longer "does ellipticity matter" but
"apply a known factor in two known places".

## Rationale

`a0` is by definition the normalized **peak** magnitude of the electric field, so the
cycle-averaged normalized intensity is `<a**2> = C a0**2` with `C = 1/2` for linear
polarization (`<cos**2> = 1/2`) and `C = 1` for circular (constant magnitude).
`GaussianParaxialLaser._a0_from_density` implements the linear chain explicitly (`E0 =
sqrt(8 pi u)`), so `a0_profile` is the linear **peak** amplitude envelope and `C = 1/2`
applies. `TrajectorySamples.a0_shape` is the amplitude-squared trajectory average with no
cycle factor, so `ahat_code = 2 ahat_paper`. There is no second convention under which both
are right: it is a missing factor.

**The §0 question this closes.** The open worry recorded in DER003 was
whether a compensating 2 already sat somewhere else — `stages.relative_velocity(1.0) = 2.0`,
the `E = 4 hbar omega0 s` energy convention, or `stages.KERNEL_NORMALIZATION_CONSTANT`. It
does not, and the reason is structural rather than numerical: all three live in the
photon-count / normalization path, whose absolute value §9.1 pins against an elementary
`flux x cross-section x time` total (RES033), while `ahat` lives in the resonance denominator
and moves photons along `s` without changing how many there are. The two quantities cannot
cancel each other because no check that constrains one is sensitive to the other.

**The cross-validation gap this exposed (§7).** xigma and `delta` **share**
`TrajectorySamples.ahat()`, so this error was common-mode and cancelled in every
xigma-vs-`delta` comparison. §7's cross-check machinery — whose stated purpose is catching
exactly this class of mistake — was blind to it, and so was every test in this repo: the
full suite stayed green through the fix, all 293 of it. Two independent reasons: (1)
**shared inputs** — a quantity both paths read from the same object cannot be checked by
comparing the paths; the analytical engine is the leg that exposed it, because it computes
`ahat` from its own overlap integral rather than from `TrajectorySamples`. (2) **integrated
observables** — even with independent inputs, `validation.run`'s fourth identity leg
compares a **sum over `s`**, and the red-shift moves photons along `s` while conserving that
sum. Measured directly, by scaling `ahat` over an 8x range (`x0.5` to `x4`) on all three
scenarios at the production grid: the leg stays inside `0.9987`–`0.9998`, a 0.11% spread,
while the same range moves the spectrum's centroid by many percent. The leg is not merely
insensitive to a factor of two — it is structurally blind to the whole quantity.
`tests/test_stage1_stage2.py::
test_the_kernel_and_delta_agree_on_where_the_redshift_puts_the_photons` is the new check
that watches the spectrum's **centroid** instead; a 1.2x kernel-side `ahat` bias moves it
0.53% against a 0.009% clean residual.

## Consequences

**Measured effect.** Yield, `a0_peak`, and every golden scalar: unchanged (the goldens
compare `a0_peak`/`gamma0`/`n_electrons`/`n_photons`, none of which touch `ahat`;
distribution goldens are still the unexercised comparison RES033 flagged for Phase 5/7).
`validation.run`'s fourth leg moved `0.9848 -> 1.0245` on `near_a0_max` and not at all on
the other two, well inside its `0.1` tolerance. Scenario-bank `ahat`, luminosity-weighted
mean: `baseline` `0.0114 -> 0.0057`, `low_a0` `0.00114 -> 0.00057`, `near_a0_max` `0.0571 ->
0.0286`.

**A consequence for RES032's grid, recorded and deliberately not acted on.** RES032 tuned
`retarget_ahat`'s target-grid defaults (`ahat_max = 0.5`, `n_bins = 32`, `decades = 1.0`)
against measured bank `ahat` values that were all 2x too large, and halving them pushes the
whole bank further into the grid's coarse floor region — `near_a0_max` drops from four
resolved bins to two. Measured centroid bias against `delta` at production defaults, at
`theta = 0`: `baseline` `-0.59% -> -1.14%`, `low_a0` `-1.59% -> -1.65%`, `near_a0_max`
`+0.08% -> -0.04%`. The bias is dominated by the floor bin standing in at its own centre
(`0.0174`) for every population below `0.035`, which is **pre-existing** and not created
here. A one-parameter change — `decades = 1.0 -> 0.3`, same `n_bins`, same `ahat_max`
headroom — removes most of it (`near_a0_max` goes to five bins, kernel/`delta` agreement at
the spectral peak `0.60 -> 0.99`). That is left to the author: it is a production default
they tuned with stated physics reasoning ("concentrate resolution near `ahat_max`, where
the red-shift matters"), and the measurement now says the bank lives near `ahat_min`
instead — which is a judgement about what the defaults are *for*, not an arithmetic error
to correct on their behalf.

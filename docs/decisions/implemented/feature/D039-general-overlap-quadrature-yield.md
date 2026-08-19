# D039 — the analytical yield is a general overlap quadrature, not the round-beam closed form

Status: implemented
Class: feature

## Problem

D035 landed Phase 4 with the predecessor's round-beam, no-displacement approximations
unchanged, naming non-round beams and foci displacement as open growth items for the total
yield. Closing them needs either a new derivation or a demonstration that the general case
reduces to a Gaussian integral no new physics is required for.

## Decision

`engines.analytical.formulas.overlap_yield` evaluates the Gaussian luminosity overlap
integral in its general form — per-axis bunch sizes and emittances, per-axis Twiss `alpha`,
per-axis laser waists and Rayleigh ranges, astigmatic `z_fx`/`z_fy` focal offsets, and the
`psi_focus` rotation between the two transverse ellipses — leaving exactly one longitudinal
quadrature. `AnalyticalEngine` calls it.
`engines.analytical.formulas.estimate_yield`'s round-beam closed form stays in the module
but is no longer what the engine ships. The derivation is `docs/DERIVATIONS.md` §A.

`overlap_yield` raises on a crossing angle (`theta_xz`/`theta_yz`) and on a flying focus
(`beta_ff`). The first is `GRAND_PLAN.md` §9.3's open derivation; the second makes the
spot-size evaluation point time-dependent, which breaks the analytic time integration the
result rests on. Refusing is the P14c-compliant move — the failure mode P14c names is
returning a plausible number whose derivation does not apply.

**Scope, precisely.** This closes two of D035's three growth items **for the total yield
only** — non-round beams and foci displacement. It does *not* close them for
`estimate_spectrum_width`, whose nonlinearity term still uses `laser.a0_peak()`, the
pulse's own maximum, rather than the a0 the bunch actually samples; that needs an
overlap-weighted `<a0^2>` and stays open. Displacing the foci therefore moves the yield
correctly while leaving that width component unmoved. The third item, collimated-spectrum
construction, is untouched.

Also a real behavioral narrowing: because `AnalyticalEngine` calls `overlap_yield`
unguarded, the engine **raises** on a flying-focus laser where it previously returned a
(wrong) number. No current caller does that — `validation.scenarios` leaves `beta_ff` at
zero — but Phase 6 must decide what the GUI's estimates panel shows for such a
configuration rather than propagating an exception. (This paragraph originally covered the
crossing angle too; D041 supersedes that half — a crossing angle is now handled, not
refused.)

## Alternatives considered

**Keep `estimate_yield` as the engine's yield and expose the general form as an opt-in.**
Leaves the engine shipping approximations that the collision geometry in `io` already has
the data to avoid, and (see D040) a known convention error as well.

**Generalize the closed form's `nu` instead of integrating numerically.** There is no
closed form to generalize to: `det(C_e + C_l)` is a quartic in `z` once the ellipses are
unequal and rotated, and `Gaussian / sqrt(quartic)` is not elementary.

**Add explicit waist-position fields for the bunch.** Duplicates what `alpha_x`/`alpha_y`
already encode, and would immediately desync from `io.bunch.drift` (P9).

## Rationale

The generalization invents no physics: every step is a Gaussian integral or a standard
identity, and the transverse part collapses to a single determinant
(`engines.analytical.formulas.overlap_det`) that reduces to the round-beam `1/(2 pi
sigma_0^2)` only when both ellipses are circular. Two properties make it safe to prefer
over the closed form. It reduces to that closed form *analytically* in the round, aligned,
`alpha = 0` limit, so the old formula becomes a regression anchor rather than being
discarded — `test_overlap_yield_reduces_to_the_round_beam_closed_form` pins the reduction
against an independently-coded evaluation. And it needs **no new schema state**: the
electron waist displacement is already `GaussianElectronBeam.alpha_x`/`alpha_y` (verified
against `io.bunch._drift_fit` to machine precision, which pins the sign that no symmetric
test can catch), and the laser side is already `z_fx`/`z_fy`/`psi_focus` — P9 applies here
exactly as it did in D038.

The integrand is strictly positive and smooth, so the cost is a fixed grid, not a Monte
Carlo. Only its *resolution* needs care: the hourglass and Rayleigh scales can be far
shorter than the longitudinal weight and displaced foci move that structure off the origin,
so `_overlap_grid` unions a grid over the Gaussian support with one refined window per
waist. `n_quad_overlap` is a separate `FieldSpec` from `n_quad` because it governs a
different integral with different convergence behavior.

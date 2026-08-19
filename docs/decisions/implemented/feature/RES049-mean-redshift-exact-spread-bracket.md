# RES049 — the mean red-shift is exact; its spread is reported as a measured bracket

Status: implemented
Class: feature

## Problem

The analytical engine had no way to apply the nonlinear red-shift at all — computing it
needs a per-electron `ahat`, and a photon's formation length spans the whole trajectory, so
the physical quantity is one scalar per electron, `ahat_i = Int a0^4 dt / Int a0^2 dt`, that
cannot be built from locally-constant pieces. Whether both the mean and the spread of that
quantity across the bunch are computable in closed form needed determining before either
could be applied or reported.

## Decision

`engines.analytical.formulas.overlap_mean_a0_sq` gives the beam-averaged `ahat` exactly,
and `AnalyticalEngine` now **applies** the resulting red-shift: `angle_integrated_spectrum`
takes `ahat` and puts the resonance at `gamma^2 / (1 + ahat)`. The broadening it causes is
**not** computed — it is bracketed by `engines.analytical.formulas.
NONLINEAR_BROADENING_RANGE`, an empirical `(0.06, 1.12)` on `std(ahat) / <ahat>`, surfaced
through `SpectrumWidthBreakdown`'s `nonlinearity_lo`/`_hi` and `total_range`. §0's factor of
two is **resolved** (author, 2026-08-10): not a convention but the polarization cycle
average. `a0` is the normalized *peak* field magnitude, so the cycle-averaged normalized
intensity is `<a^2> = C a0^2` with `C = 1/2` for linear polarization (the average of
`cos^2`) and `C = 1` for circular, where the magnitude is constant — the same factor by
which circular carries twice the cycle-averaged energy density at fixed `a0`. `io.laser`
builds `a0` through the linear chain explicitly, so `C = 1/2` here and the engine passes
`0.5 * <a0^2>`. See DER003.

## Alternatives considered

**Report `std` from the joint distribution.** Wrong quantity, and dangerous precisely
because it looks like the right one. An earlier version of this entry claimed this was
computable; that was wrong. Taking moments of the instantaneous `a0^2` over all (particle,
time) pairs mixes the **within-trajectory** variation into the answer, and that variation
is already averaged away inside `ahat_i` — it must not broaden anything. The tell: with
every electron sharing one `ahat` but `a0^2` varying along each trajectory, the true beam
spread is zero and the joint formula returns a positive number. Measured against xigma, the
joint version over-states by ~1.5x (0.70 against a true 0.39 at the baseline).

**A per-particle trajectory quadrature for `ahat_i`.** Correct and affordable in isolation,
but `O(n_particles)` in the one place that must not have it. The correct quantity has no
clean closed form: `ahat_i` is a *ratio* of trajectory integrals, so `<ahat^2>` needs
`(Int a0^4)^2 / (Int a0^2)` per particle — a reciprocal of a Gaussian integral inside a
bunch integral. The one shortcut that would have rescued it, `ahat_i = A_i / sqrt(2)` —
exact if the profile along a trajectory were Gaussian, with the peak available in closed
form from `io.bunch.peak_illumination` — fails in practice: median ratio 0.92 at the
baseline but 0.33 at a tight focus, because the spot varies too much across the encounter.
Scenario-dependent, so no fixed correction rescues it either.

**Use the 0.4-0.9 bracket as proposed.** Five of thirteen measured geometries fall outside
it, one above.

## Rationale

The *mean* survives the locally-constant-pieces constraint exactly, because luminosity
weighting (`L_i ~ Int a0^2 dt`) cancels `ahat_i`'s denominator:

    <ahat>_L = sum_i L_i ahat_i / sum_i L_i = Int n_e a0^4 / Int n_e a0^2

which is precisely what the overlap integral evaluates, splitting nothing. Verified
against `engines.xigma.stages.TrajectorySamples.ahat`, which averages each trajectory
numerically.

Hence a bracket, and hence *no* per-particle path in the semi-analytical engine: a
trajectory quadrature per macroparticle was considered and rejected, because the engine's
defining property is that its cost is `O(n_quad)` and never `O(n_particles)`.

**The bracket is measured, and wider than first proposed.** Thirteen geometries through
xigma — focus scans, displaced and astigmatic foci, crossing angles, a flying focus, bunch
length and width scans, transverse and timing offsets — give 0.06 to 1.12, not the 0.4-0.9
first suggested. It tracks `sigma_beam / sigma_laser` almost monotonically: 0.06 for a loose
focus (nearly uniform illumination, so almost no spread), 0.39 at the baseline, 0.86 at a
tight focus, 1.12 for a bunch ten times wider than the spot. A narrower bracket would read
better and be false.

Its width is usually tolerable for the reason the range exists at all: when beam quality
dominates, the collimation, emittance and energy-spread terms swamp the nonlinear one and
`total_range` nearly collapses — asserted as a test, not assumed. xigma computes the exact
value when it matters.

## Consequences

Recorded because it is easy to misread: the legacy scalar `nonlinearity` corresponds to a
factor of **1**, near the *top* of the bracket, so the predecessor's formula over-estimates
the nonlinear broadening for most geometries.

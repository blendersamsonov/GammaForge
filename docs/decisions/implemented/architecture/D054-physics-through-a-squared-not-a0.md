# D054 — physics goes through `<a^2>`, not `a0`: the polarization convention is removed from the yield/red-shift path rather than documented

Status: implemented
Class: architecture

## Problem

D053 fixed a missing factor of two by adding a cycle-average constant that had to be
applied at exactly the right place and nowhere else — a round trip through a
convention-dependent number (`a0`) that hid the bug through the entire suite. Every
angle-integrated observable (the photon yield, the mean nonlinear red-shift) is actually a
functional of the cycle-averaged `<a^2>` along a trajectory, which raised the question of
whether forming `a0` and re-applying the cycle factor is the right shape at all, or whether
the polarization convention should be removed from the path entirely.

## Decision

`LaserField` gains `intensity_profile` — the cycle-averaged normalized intensity `<a^2>` —
and that is what xigma's Stage 0 integrates. `GaussianParaxialLaser` gains `intensity_peak`
and `cycle_average_factor`; *TrajectorySamples.a0_peak* becomes `TrajectorySamples.
intensity_peak`, and `retarget_ahat`/`Collision._table` take a peak intensity instead of a
peak amplitude. The module constant *CYCLE_AVERAGE_FACTOR* (D053) is **removed**: on the
yield/red-shift path there is nothing left for it to multiply.

**Why this is a structural change and not a rename.** D053 fixed a missing factor of two by
adding one; this removes the place where such a factor can go missing. Forming `a0` and
then re-applying *C* is a round trip through a convention-dependent number, and it hid
D053's bug through the entire suite. `ahat_from_shape` is now a plain product of the shape
factor and the peak intensity, with no constant to omit.

The change also caught its own instance of the bug it prevents. The first version of Stage
0's line read `intensity_peak = metrics.cycle_average_factor() * metrics.a0_peak()**2` —
but `a0_peak()` is the *linear-equivalent* amplitude by convention, so multiplying by this
pulse's own *C* applies the cycle average without the compensating `1/sqrt(2C)`, and the
supposed invariant varied with `ellipticity` (0.0164 → 0.0328 from linear to circular).
`test_stage_0_is_bit_identical_under_any_polarization` failed immediately.
`GaussianParaxialLaser.intensity_peak` exists so that no caller has to reconstruct it.

**What `ellipticity` now does, precisely.** It is applied — exactly, not approximately — to
the total yield and the mean red-shift, by being irrelevant to both. `ELLIPTICITY_IS_NOOP`
stays `True` for one remaining consumer: xigma's **angle-resolved** kernel still uses the
linear polarization factor `cos^2 psi` instead of `(cos^2 psi + eps^2 sin^2 psi)/(1 +
eps^2)` (`docs/DERIVATIONS.md` §1.2). That is the only place a contraction against an
observation direction can distinguish an ellipse from a line, which is why the split falls
exactly there. `validate()`'s warning was rewritten to say this rather than the previous,
now-false "results are those of a linearly polarized pulse".

## Alternatives considered

**Keep `a0_profile` as the integrand and apply *C* in the consumers (the D053 shape).**
Correct arithmetic, and it is what D053 shipped. Rejected because it makes the red-shift
*look* polarization-dependent when it is not, and because every consumer is then one
omission away from D053's bug — which is exactly how that bug arose and survived.

**Drop `a0_peak`/`a0_profile` entirely.** `a0` is what people quote, what the goldens
compare (`a0_peak` to 6.6e-11), and what the paper's own `ahat` is written in terms of. It
stays as a **reported** quantity with a stated convention (linear-equivalent peak
amplitude), and `LaserField` documents which of the two an engine should reach for.

**Make `cycle_average_factor` return 1 and fold *C* into `a0_peak` instead.** Would make
`a0_peak` polarization-dependent, breaking the golden comparisons and the convention that a
quoted `a0` means one field strength.

## Rationale

At fixed pulse energy `<a^2>` does not depend on the polarization state. Writing `<a^2> = C
a0^2` with `C = (1 + eps^2)/2`, holding the energy fixed forces `a0^2 = (e/m_e c omega)^2
4 pi U / C`, so

    <a^2> = (e / m_e c omega)^2 * 4 pi * E_pulse * photon_density

with no *C* in it. A circular pulse of the same energy has `a0` smaller by `sqrt(2)` and
carries twice the cycle-averaged intensity per unit `a0^2`; the two offset exactly.
Measured on `BASELINE`: linear `a0 = 0.18117`, circular `a0 = 0.12811`, and `<a^2> =
0.0164113` for both, equal to the `4 pi` chain evaluated with no polarization input at all.
Every angle-integrated observable is a functional of `<a^2>` along a trajectory — the
photon yield through `photon_density_scale`, and the mean nonlinear red-shift through
`ahat`, which is a *ratio* of intensity moments. So **none of them ever needed a
polarization derivation**, and §9.2 was never blocking them.

## Consequences

**Testing.** The claim is asserted as an **invariance**, not as the value of a constant:
`tests/test_stage0_delta.py::test_stage_0_is_bit_identical_under_any_polarization` runs
Stage 0 at `ellipticity` 0/0.3/1.0 and requires `luminosity`, `a0_shape` and `ahat` to be
bit-identical. That fails if anyone reintroduces a polarization factor anywhere on the
path, in either direction, without needing to know where they put it.
`tests/test_laser.py::test_the_cycle_averaged_intensity_is_polarization_agnostic` pins the
other side — `C = (1+eps^2)/2` checked against a period-resolved ellipse, and `a0_peak`
genuinely *does* move — so the invariance is a real cancellation rather than both sides
being constant.

**Numerically inert.** All 424 tests pass and `validation.run` is unchanged: for the
linear pulses in the scenario bank `<a^2> = a0^2/2` exactly, so every number this repo
produced before the change it produces after. What changed is which quantities can be
expressed at all.

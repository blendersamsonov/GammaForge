# RES047 — the illumination filter tests the *bright* region, which is what the cone approximates

Status: implemented
Class: feature

## Problem

The region where photons are actually produced is where the pulse is *bright*, and that is
not the region it geometrically occupies. Away from focus the spot grows but dims as `1 /
(s1 s2)`, and `GaussianParaxialLaser.active_region` deliberately ignores that decay —
correctly, since a bound may only err towards keeping particles — but the consequence is
that the cone keeps a great many particles that contribute nothing. On a wide bunch meeting
a small displaced pulse, the cone keeps 94% of the bunch while the median particle it keeps
is illuminated at below `1e-6` of the peak.

## Decision

`io.bunch.peak_illumination` gives the highest photon density each macroparticle ever
meets, as a fraction of the pulse's own peak, in closed form; `io.bunch.
prefilter_by_illumination` keeps particles above a threshold on it. It joins `io.bunch.
prefilter_bunch` (unchanged, still the default) and `io.bunch.prefilter_by_luminosity`
(RES045).

Computing the real quantity is cheap for the same reason `luminosity_weights` is: frozen
widths make the density quadratic in `t` along a straight trajectory, so its maximum is
closed-form at the stationary point. It is the *same* machinery read at its peak instead of
integrated.

## Alternatives considered

**Tighten `active_region` itself to include the amplitude decay.** Tempting, since that is
where the looseness lives — but the region is a **bound** whose over-inclusiveness
`prefilter_bunch`'s exact invariance depends on (§3.2), and the bright region is not a
bound. It would convert a guarantee into a tolerance without changing any signature.

**Threshold the region once and reuse `ActiveRegion.contains`.** The bright region is not
a cone: its transverse reach *shrinks* with distance from focus and vanishes entirely
beyond it, which `radius_at`'s linear form cannot express.

## Rationale

This is the better-shaped of the two tolerance filters. It is a **region test** — the same
contract shape as the cone, thresholding an intensity — which makes it a drop-in whose
behavior a reader already understands, and it is independent of how long a particle dwells
in the beam. `prefilter_by_luminosity` thresholds *integrated contribution* instead, which
is the more directly meaningful quantity when what a user wants to control is "how much
luminosity am I discarding". They measure comparably: on the wide/displaced scenario,
illumination at `1e-6` keeps 28.4% for 3.5e-4 induced error, weight at `1e-4` keeps 31.3%
for 1.3e-4. Both are kept because they answer different questions; neither replaces the
cone.

Worth recording: the illumination threshold has to be set far *lower* than intuition
suggests (`1e-6`, not `1e-3`) to reach a small induced error, because many weakly
illuminated particles sum to a non-negligible contribution even though none matters
individually. That is the price of a peak-based criterion, and it is why the integrated
variant exists alongside it.

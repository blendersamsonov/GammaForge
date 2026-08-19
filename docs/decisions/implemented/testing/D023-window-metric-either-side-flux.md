# D023 — The window metric counts a window if *either* side has flux in it

Status: implemented
Class: testing

## Problem

The predecessor's `significant = win_flux_ref > floor` restricts both reported comparison
numbers to the reference's own support — which is invisible in every test one would think
to write and blinds the metric to the exact failure the module documents itself as
catching.

## Decision

`window_integrated_deviation` includes a window in `max_window` when the candidate has flux
above the floor even if the reference has none, and the floor is a fraction of the
**total** reference flux (1e-3) rather than of the mean window flux. `weighted_l1` stays
reference-weighted. An all-zero reference is scaled by the candidate's own total instead of
returning a perfect match.

## Alternatives considered

**The predecessor's `significant = win_flux_ref > floor`, which restricts both reported
numbers to the reference's own support.** Rejected — see Rationale.

## Rationale

That restriction is invisible in every test one would think to write and it blinds the
metric to the exact failure the module documents itself as catching. Photons appearing
where the reference has none — a Compton edge in the wrong place — contribute nothing to a
reference-*weighted* average, by construction, and were then dropped from the maximum as
well: a candidate carrying 0.12% of its yield in a region the reference sets hard to zero
scored `weighted_l1 = 0.0, max_window = 0.0` and passed a 2% golden tolerance. The floor
moved to a fraction of the total for a related reason: as a fraction of the *mean window*
it shrinks when the binning is refined, so the same spurious flux reports a different
deviation at 64 bins and at 256. Against the total, a spurious peak is measured in units of
a thousandth of the yield — a number a tolerance can be set against, and one that does not
move when the grid does.

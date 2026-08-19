# RES046 — misalignment lives on the laser as `x_off`/`y_off`/`t_off`, and there is deliberately no `z_off`

Status: implemented
Class: feature

## Problem

Transverse and timing misalignment between pulse and bunch is the knob that actually
determines yield in a real experiment, and until now the data model could not express it
at all — both distributions were centred on the origin by construction, so the
derivation's generality over foci displacement (RES039) stopped at the longitudinal
direction.

## Decision

`io.laser.GaussianParaxialLaser` gains `x_off`, `y_off` and `t_off`, applied once in
`_local_coordinates`, exposed in `io.fields.LASER_FIELDS`, and carried through the
analytical overlap integral as a linear term (DER001 §A.11).
`GaussianParaxialLaser.active_region` shifts its origin to match.

**There is no `z_off` on purpose — but not for the reason it first appears.** Focus
position and arrival time are *not* the same thing: the focal plane is fixed in space while
the envelope sweeps through it at `c`, so two pulses with exactly coincident foci still
miss if they arrive at different times. That configuration is real and representable
(`z_fx = z_fy = 0`, `t_off != 0`), and `test_coincident_foci_can_still_miss_in_time` pins
it. What makes `z_off` redundant is that a *rigid* longitudinal shift moves the focus
**and** the envelope, so it already reads as a shift of `z_fx` and `z_fy` together with the
matching shift of `t_off`. The longitudinal degrees of freedom are `z_fx`, `z_fy`, `t_off`;
a `z_off` would be a linear combination of those three, and having it would let two knobs
silently cancel.

`active_region` had to move in the same change, not after: it is a *bound* the prefilter
relies on never being too small (§3.2), and a region left at the origin while the pulse
moved would discard particles that do interact — the one direction that contract forbids.

## Alternatives considered

**Put the offsets on `InteractionParameters` as a relative displacement.** Arguably more
symmetric, but it would split the geometry across two objects and leave `LaserField`
consumers unable to see it, so `photon_density` would return the aligned field.

**Offset the bunch instead.** Same physics, but the bunch is the reference frame
everything else is measured against, and `Bunch` arrays are per-particle — shifting them
would make the offset a property of a sample rather than of the configuration.

**A full 3D `r_off` plus `t_off`.** Over-parametrized by exactly one, as above.

## Rationale

Putting it on the laser matches where §2.2 already pins every other geometric degree of
freedom (crossing angles, `psi_focus`, `z_fx`/`z_fy`); the bunch defines the origin.
Applying it in `_local_coordinates` rather than at each call site is what makes every
consumer — `photon_density`, `a0_profile`, `field`, and so xigma as well as analytical —
inherit it from one subtraction.

Analytically a misalignment adds exactly one linear term to the quadratic form, which is
why it needed no new derivation — but it must be carried through *both* completions of the
square, and a dropped piece shifts the answer rather than making it diverge. Hence two
guards: `test_zero_offset_is_bit_identical_to_no_offset_at_all`, and an exact `exp(-d^T
(C_e + C_l)^-1 d / 2)` falloff in the no-hourglass limit checked to 1e-13 with an
off-diagonal displacement, since an on-axis test passes with a wrong inverse.

# RES034 — A crossing angle warns that only the geometry is applied (§9.3's half of P14c)

Status: implemented
Type: bug-fix

## Problem

§9.2 and §9.3 are both open derivations the paper does not contain, and P14c requires both
to be visible rather than silently approximated — but only §9.2 was. The asymmetry was easy
to miss precisely because the crossing angle is *partly* implemented: `io.laser.
rotation_matrix` is applied everywhere the pulse is sampled, so overlap, timing and the a0
an electron actually sees all respond correctly to a tilt. What does not respond is the
emission — `engines.xigma.stages.relative_velocity()` supplies the encounter factor, and the Stage-2 kernel
measures angles from the collinear axis. A caller who tilts the beam and watches the yield
change has every reason to conclude the physics followed — a worse failure mode than a
parameter that visibly does nothing, and the asymmetry with §9.2 (which had a marker and a
warning) was itself a P14c violation.

## Decision

`io.laser.EMISSION_IS_HEAD_ON` joins `io.laser.ELLIPTICITY_IS_NOOP` as a module-level
marker, and `io.laser.validate` warns when `theta_xz` or `theta_yz` is nonzero.

## Alternatives considered

**Reject a nonzero crossing angle outright.** The geometry half is real and useful (§2.2
pins the rotation convention for exactly this reason), and near-backscattering is where the
derivation is valid — a small tilt is a legitimate configuration, not an error.

**Put the marker in `engines/xigma/stages.py`, next to `relative_velocity()`.** That is where
the limitation physically lives, and the constant's comment now points both ways. But the
warning has to reach whoever configures the laser, `io` may not import `engines`, and
`validate` is already the warning surface for exactly this class of statement — so the
marker sits with the warning and the engine-side comment cross-references it.

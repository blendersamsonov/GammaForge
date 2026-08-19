# D015 — No coordinate normalization: xigma works in CGS directly

Status: implemented
Class: architecture

## Problem

`GRAND_PLAN.md` open question 7 previously assumed the predecessor's `k0_las`-normalized
coordinate system would carry over into xigma, "purely internal to xigma." An audit
(author-directed) of the predecessor's pipeline needed to confirm whether the
normalization buys anything in the rebuilt architecture.

## Decision

The predecessor's `k0_las`-normalized coordinate system is not carried over. Stage 0 will
integrate trajectories, sample `LaserField`, and bin diagnostics in CGS.

## Alternatives considered

**Porting the normalization, as `GRAND_PLAN.md` open question 7 previously assumed ("stays
purely internal to xigma").** Rejected — see Rationale.

## Rationale

An audit of the predecessor's pipeline found the normalization buys nothing. Its `k0**2`
in the overlap integrand is precisely the Jacobian of the normalization — with `u = k0 r`
and `tau = k0 c t`, `n_phys * c * dt_phys` becomes `n_u * k0**2 * dt_u`, so `k0**2` is what
*replaces* `c`, not independent physics. `a0_shape` is built from a ratio of envelopes and
is k0-free by construction; the H-table's axes (gamma, theta_x, theta_y, a0) are all
dimensionless; the Stage-2 kernel contains no `omega`, `hbar` or energy scale at all,
working in a dimensionless `s` that the adapter converts with `4 hbar omega0` at its
boundary. Every remaining appearance is `/k0_las` or `/omega_las` *un*-normalizing the
spatial and temporal diagnostics.

So in the rebuilt architecture — where the laser owns lab-frame CGS sampling and the bunch
owns CGS trajectories — normalizing would mean scaling coordinates on the way into
`LaserField` and unscaling on the way out, for no gain. It would also fight **P15**
directly: an arbitrary `Spectral-FEM-Fields`-backed pulse knows nothing about `k0_las`, and
a crossing angle makes the bookkeeping worse. The one remaining argument for it would be
float32 conditioning on the GPU path, which is a Phase-2.5 measurement rather than a reason
to normalize now: in CGS the integrand's factors sit near 1e8 cm⁻³ and 1e-13 s, comfortably
inside float32's range.

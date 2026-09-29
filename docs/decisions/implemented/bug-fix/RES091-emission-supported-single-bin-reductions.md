# RES091 — Emission-supported single-bin reductions

Status: implemented
Class: bug-fix

## Problem

A configured one-bin carrier axis used the first particle's carrier mean as its
evaluation point. Reordering otherwise identical trajectory samples therefore changed
the spectrum. Particles with zero luminosity could also set table ranges even though
they contributed no photons; a nonempty, non-overlapping bunch produced an invalid
zero carrier evaluation point. Separately, the one-bin `ahat` mode evaluated the
nonlinear coordinate at zero but retained its nonlinear fluctuation channels, so the
`moment2` spectrum still contained nonlinear corrections.

## Decision

`deposit_shape_table` determines axis support from particles with positive luminosity.
When the carrier axis has one bin, its evaluation point is their luminosity-weighted
mean carrier rate. With no emitting particles, it uses a positive neutral carrier rate
and deposits zero mass. `retarget_ahat` sets the nonlinear variance and nonlinear-carrier
covariance channels to zero when `n_bins_ahat=1`; it retains carrier variance.

## Alternatives considered

- Keep the first particle's carrier rate: rejected because array order has no physical
  meaning and can move the spectrum without changing deposited mass.
- Use the unweighted mean of all particles: rejected because dark particles would shift
  the result and each emitting particle contributes a different luminosity.
- Keep nonlinear fluctuation channels in the linear `ahat` mode: rejected because a
  zero nonlinear evaluation point would then coexist with a nonlinear finite-line
  correction, contrary to the mode's stated meaning.

## Rationale

Luminosity is the table's deposited measure, so its weighted carrier mean is the
representative value of a collapsed carrier axis. Dark particles cannot affect a
weighted photon distribution. A linear nonlinear-coordinate limit removes both terms
containing its fluctuation while preserving a carrier-only finite-line width.

## Consequences

Single-bin spectra are invariant under particle permutation and adding zero-weight
particles. A nonempty bunch with zero overlap returns zero spectral moments. The
one-bin `ahat` mode now gives a true linear nonlinear-coordinate limit for both `delta`
and `moment2`; chirp broadening remains available.

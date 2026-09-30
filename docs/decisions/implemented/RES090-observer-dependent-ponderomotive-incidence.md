# RES090 — Ponderomotive incidence is observer-dependent in Stage 2

Status: implemented
Type: bug-fix

## Problem

RES088 replaced the exact observation direction in the nonlinear incidence coefficient
with the electron direction and deposited the resulting factor in Stage 1. That
beaming-cone substitution omitted the denominator $1-\mathbf e\cdot\mathbf n_0$ from the
required coefficient and made an observer-dependent resonance term part of an
observer-independent table coordinate.

## Decision

This decision supersedes RES088. Stage 1 stores raw $a_{0,\mathrm{shape}}$, and retargeting
produces raw $\hat a$. Stage 2 evaluates

$$
Q=\frac{1-\mathbf n\cdot\mathbf n_0}
        {1-\mathbf e\cdot\mathbf n_0}
$$

from the requested observation direction and the same normalized electron and laser
directions used by the resonance. The resonance, inverse, and Jacobian use $Q$ together
with the trajectory-averaged carrier correction $\bar C$ as recorded in DER015.

The table is five-dimensional over
$(\gamma,\theta_{ex},\theta_{ey},\hat a,\bar C)$ and carries three co-shaped,
luminosity-weighted moment channels. Raw spectral moments are queryable independently;
the `moment2` line model applies DER017's second-order reconstruction on an arbitrary
strictly increasing spectral grid. Prepared queries cache individual spectral samples
and backend state, so adaptive insertion evaluates only new points.

## Alternatives considered

- Retain RES088's electron-direction approximation: rejected because it omits the
  observer dependence required by the supplied resonance and puts query geometry into
  Stage 1.
- Deposit $Q\hat a$: impossible for a reusable table because $Q$ depends on the requested
  observation direction.
- Shift the analytical resonance root by the finite-line centroid correction: rejected
  because it destroys the closed-form inverse; the shift belongs in the first spectral
  moment.
- Combine $D$, $Q$, $\bar C$, and $\hat a$ into an opaque effective-gamma coordinate:
  rejected because it hides distinct physics and sacrifices the reusable correlated
  table representation.

## Rationale

Keeping raw nonlinear strength and carrier mean as separate table coordinates preserves
their correlation with electron direction while leaving all observation geometry in the
query stage. Parallel moment channels retain the information needed for finite-line
centroid and variance corrections without expanding the table with additional moment
axes or using signed quantities as sampling probabilities.

## Consequences

Head-on unchirped calculations retain $Q=\bar C=1$. Crossed-incidence nonlinear spectra
deliberately differ from RES088 when the observation and electron directions do not
coincide. Built-in lasers remain exactly unchirped, while any conforming `LaserField` can
supply an additional carrier-phase four-gradient without xigma-specific pulse modelling.
NumPy and CuPy share the same five-dimensional physics and raw moment interface.

The manuscript still needs to be synchronized with DER015–DER017; independent
arbitrary-angle scientific acceptance remains open and is not implied by numerical
backend agreement.

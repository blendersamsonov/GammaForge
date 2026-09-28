# RES088 — Ponderomotive incidence uses the electron beaming direction

Status: implemented
Class: bug-fix
Archived: 2026-09-28

**Superseded by RES090** (2026-09-28): the nonlinear incidence coefficient is now the
exact observer-dependent $Q$ evaluated in Stage 2, while Stage 1 again stores raw
nonlinear shape.

## Problem

The crossed-angle resonance used a unit coefficient on `ahat`. The unspecialized
retarded phase instead contains the observation-dependent numerator
$(1-\mathbf n\cdot\mathbf n_0)/2$. Leaving the head-on coefficient in place therefore
overstated nonlinear redshift at large incidence angles, even though the incident
direction-Doppler factor was already applied consistently by RES082.

## Decision

Use the author-selected beaming-cone substitution $\mathbf n\simeq\mathbf e$ for this
coefficient. Every electron's unchanged `ahat` is multiplied in the resonance denominator
by

$$P=(1-\mathbf e\cdot\mathbf n_0)/2.$$

The same beta=1 unit electron direction drives RES082's production Doppler factor, so
$P=(F_0/2)D$. Stage 1 deposits $P a_{0,\mathrm{shape}}$ per trajectory; retargeting then
produces a Table coordinate equal to $P\hat a$. NumPy, CuPy, the gamma-resonance proposal,
and support bounds consume that corrected coordinate directly and never reconstruct $P$
from angular cell centres. Direct delta references, which have no table, apply $P$ per
particle (DER014). Diagnostic choices between a nominal, finite-beta, or beta=1 resonance
numerator do not change $P$.

This partially extends RES082's nonlinear denominator; it does not replace that
decision's encounter-flux, Jacobian, or linear-spectrum behavior.

## Alternatives considered

- Retain the exact observation-dependent ratio from the 2026-09-23 handoff: not selected
  for production. The useful emission from a particular electron is confined to its
  $O(1/\gamma)$ cone, so the author chose its velocity direction for this coefficient.
- Keep the unit coefficient: rejected because it discards the general-incidence decrease
  while retaining other crossed-angle effects.
- Multiply $a_0$ before constructing `ahat`: rejected because `ahat` already represents
  an intensity-like cycle average; applying the factor to amplitude would square it and
  would also corrupt the trajectory quantity shared by other calculations.
- Apply $P$ only while querying the deposited table: rejected after author review because
  it replaces the per-trajectory correlation by a value evaluated at angular cell centres.
  The nonlinear table coordinate must already contain the incidence factor.

## Rationale

The substitution preserves the physically important incidence dependence without adding
an observer-dependent nonlinear coefficient to every Stage-2 query. Depositing it per
trajectory preserves its correlation with electron angle through the table approximation;
query-time placement would discard information unnecessarily. Head-on geometry and all
linear spectra are unchanged.

## Consequences

Nonzero-`ahat` spectra at crossed incidence have less ponderomotive redshift than under
the former unit coefficient. Viewing direction still changes the explicit angular term
and hence the observed line, but it does not independently change $P$ within this
beaming-cone model. The model is not a claim of exact unrestricted-angle radiation, and
the existing independent arbitrary-angle acceptance work remains open. The manuscript
source must be updated to record the author-selected closure before paper and code are
again textually synchronized.

## Amendments

> **2026-09-23 — Placement corrected to table deposition.** The first implementation
> multiplied raw `ahat` inside the NumPy and CUDA spectrum queries. The author clarified
> that the per-trajectory product belongs in deposition. Stage 1 now deposits
> $P a_{0,\mathrm{shape}}$, the Table axis represents $P\hat a$, and Stage 2 uses that
> axis without another factor.

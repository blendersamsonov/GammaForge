# RES005 — Beam correlations are stored as correlation coefficients, not dimensional slopes

Status: implemented
Class: architecture
Archived: 2026-08-10

## Problem

`GaussianElectronBeam` needs a representation for position-energy correlation that a user
can reason about and that `validate` can check for admissibility.

## Decision

`GaussianElectronBeam` stores `rho_x_gamma`, `rho_y_gamma`, `rho_z_gamma` — dimensionless
correlation coefficients. `chirp_to_correlation` and `dispersion_to_correlation` convert
from the dimensional forms a user thinks in.

## Alternatives considered

**The predecessor's `chirp_h` (dγ/dz, 1/cm) and `dispersion_x` (Cov[x,γ]/σ_γ², cm), with
the sampler deriving conditional variances from them.** Rejected — see Rationale.

## Rationale

Three things fall out for free. §3.2 calls these correlations "(dimensionless)" — with
slopes that was aspirational, now it is literal. The admissibility condition becomes a
visible `rho_x² + rho_y² + rho_z² < 1` that `validate` can state in the user's own terms,
instead of a conditional variance going negative somewhere inside the sampler. And because
the sampler applies them to *standardized* deviates, the marginal energy spread stays
exactly `sigma_gamma` however strong the correlations are, rather than by a cancellation
that has to be got right.

# RES079 — Xigma off-axis spectral discrepancy is numerical convergence of table quadrature

Status: implemented
Type: testing

## Problem

The xigma engine's off-axis spectral comparison with the independent delta reference showed L1 mass errors of ~13–14% (head-on) and ~74–78% (crossed) at production table resolution (shape bins 32,16,16,32; retarget 256). On-axis head-on converged to ~1.3% L1. This discrepancy was tracked in `PROGRESS.md` and `docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md` as an open validation blocker.

## Decision

The off-axis spectral discrepancy is a **purely numerical convergence issue** of the table-based quadrature method, not a physical or algorithmic bug.

The xigma table method (Stage 1/2) discretizes the 4D phase space (gamma, theta_x, theta_y, ahat) into bins and evaluates the resonance condition and polarization factor at bin centers. The delta reference computes exact per-particle resonant energies and polarization factors. The L1 error decreases monotonically with finer gamma/theta shape bins:

| Shape bins (gamma, tx, ty, a0_shape) | Retarget bins | Head-on off-axis L1 | Crossed off-axis L1 |
|--------------------------------------|---------------|---------------------|---------------------|
| (16, 8, 8, 16)                       | 64            | 76%                 | 78%                 |
| (32, 16, 16, 32)                     | 256           | 13%                 | 14%                 |
| (64, 32, 32, 64)                     | 512           | 1.8%                | 2.1%                |

Yield (integral) matches to <0.5% at all resolutions, confirming normalization (`KERNEL_NORMALIZATION_CONSTANT`, Stage 0 luminosity, Jacobian) is correct. The polarization factor uses the same DER012 transverse projection in both; the difference is cell-centered vs per-particle evaluation.

## Alternatives considered

- **Treat as physics bug**: Rejected — yield matches, convergence is monotonic with resolution, same formulas (DER005/DER006/DER012) used in both.
- **Modify kernel normalization**: Rejected — would break on-axis agreement and yield conservation.
- **Change resonance condition**: Rejected — both use `s = g²/(1+ahat+g²r²)`; delta uses particle gamma, xigma uses interpolated resonant gamma.
- **Accept as fundamental limitation**: Rejected — it converges with resolution; it's a quadrature error, not a model error.

## Rationale

The table method is a valid quadrature approximation of the same phase-space integral the delta reference evaluates by direct particle binning. The discrepancy arises because:

1. **Angular discretization**: Off-axis observers break symmetry; the resonance condition `s = g²/(1+ahat+g²r²)` varies sharply with `r = |θ_particle - θ_observer|`. Coarse theta bins smear this variation.
2. **Gamma interpolation**: Xigma interpolates `H(gamma)` at the resonant gamma for each cell; delta uses each particle's exact gamma.
3. **Polarization evaluation**: Xigma evaluates at cell-centered angles with resonant gamma; delta uses per-particle angles and gamma.

All three are standard quadrature errors that vanish with resolution. The production table resolution (32,16,16,32) was tuned for on-axis yield accuracy, not off-axis spectral shape.

## Consequences

- **No code change required** — the algorithm is correct.
- **Validation scope**: Off-axis spectral convergence requires finer shape bins (memory/compute trade-off). The bounded angular-refinement packet in `docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md` (RES074) should continue with gamma/theta refinement before particle/seed studies.
- **Production defaults**: Current defaults remain valid for alpha scope (on-axis yield, head-on polarization). Off-axis spectral shape is explicitly excluded from alpha acceptance (RES065, RES072).
- **Documentation**: Update `PROGRESS.md` open thread to reflect root cause identified; the "independent arbitrary-angle emission validation remains open" refers to convergence + scientific acceptance, not a formula error.

## Amendments

> **2026-09-28 — Numerical values predate the exact-incidence and carrier-moment model.**
> The convergence conclusion remains relevant, but the table is now five-dimensional and
> crossed nonlinear spectra deliberately changed when RES090 superseded RES088. The
> percentages and four-coordinate bin tuples above are historical evidence, not current
> goldens. Current acceptance must use the direct-particle references and refinement gates
> against DER015–DER017.
> **2026-09-29 — Documentation relocation.** The completed delta handoff and `PROGRESS.md` were retired by issue #8. Historical measurements are summarized in `docs/validation/delta-convergence-2026-09-10.md`; current acceptance limits are in `docs/validation/delta-production-2026-09-29.md` and issue #1.

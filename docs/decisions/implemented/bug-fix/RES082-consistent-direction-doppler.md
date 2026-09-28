# RES082 — Consistent per-electron direction Doppler in xigma

Status: implemented
Class: bug-fix

## Problem

The initial per-particle Doppler commit changed only the NumPy resonance root. Its
Jacobian still assumed the nominal axis, the CUDA kernel retained nominal resonance
and support, and Stage 0 retained nominal encounter flux. The finite-speed diagnostic
helper also differed from the beta=1 production root. Matching old convergence reports
could not establish agreement between these different conventions.

## Decision

Retain xigma's ultra-relativistic ballistic speed beta=1, and use each electron's
normalized direction consistently. The recommended approximation was presented to the
author, followed by the instruction to continue on 2026-09-12. This records the adopted
approximation, not an independent author review of DER013's algebra.

`direction_doppler_factor` computes the relative factor D = (1 - e dot n0)/(1 - n0_z).
Stage 0 applies the actual encounter factor once to each particle's overlap flux.
NumPy and CUDA use D in both the resonance root and its Jacobian (DER013). CUDA's
radial support encloses the factor over the entire electron-angle rectangle and is
padded outward for device precision. Its proposal still uses the angular marginal;
this correction does not adopt the separate gamma-proposal worktree experiment.

`angle_integrated_spectrum` applies the same energy/density rescaling per electron,
while retaining its stated linear, table-free shape. The shared nominal photon-energy
conversion remains a coordinate conversion, not a second application of physical
Doppler shift. The angular-output energy quadrature covers the corrected particle edges.

The independent `emission_lines` adds a direction mode evaluated from its own vectors.
The matched-bin pilot explicitly selects that mode and records it with the source
fingerprints, including the GPU kernel. Historical nominal and exact finite-speed modes
remain available for diagnostics; their meanings do not change.

This extends the nominal encounter/energy conventions in DER005 and RES069. It does
not supersede their geometry or polarization work, or close RES074's full scientific
acceptance requirements.

## Alternatives considered

- Change only CUDA's root to match NumPy: rejected because the missing Jacobian changes
  photon count, and nominal radial bounds can clip valid shifted emission entirely.
- Preserve nominal Stage-0 encounter flux: rejected in favour of applying the same
  direction convention to flux and resonance.
- Retain exact finite speed in the production Doppler factor: deferred. It makes D
  depend on gamma and requires a different inverse resonance and derivative. Applying
  an exact-speed helper only after a beta=1 inversion would still be inconsistent.
- Use the observer direction for GPU support: rejected because support must include
  every possible emitting electron direction inside the integration rectangle.

## Rationale

The manuscript's general Eq. wR supplies the direction-dependent resonance, while its
head-on table reduction must be extended explicitly. DER013 differentiates that
extension and separates it from a claim of unrestricted-angle or exact finite-gamma
physics. Independent gamma quadrature tests pin photon mass and spectral centroid;
a CUDA stress test checks nonzero emission above the old nominal support edge.

## Consequences

Direction-sensitive photon yields and spectral energies change. Exact-speed diagnostics
can still differ by the finite-gamma correction intentionally omitted from production.
Earlier numerical promotion records describe their original source versions; renewed
CPU/GPU and matched-bin evidence is required. Independent scientific acceptance and
author review of the extended derivation remain separate from numerical test success.

## Author verification (2026-09-13)

A. Samsonov explicitly confirmed DER013 as verified. Its status now combines author
verification with the existing symbolic and numerical implementation checks. The
author-review thread recorded above is closed; broader arbitrary-angle scientific
acceptance remains open.

## Amendments

> **2026-09-23 — Nonlinear denominator extended by RES088.** RES082's Doppler factor,
> encounter flux, Jacobian placement, and linear spectrum remain current. RES088 uses
> the same electron direction to add the correlated ponderomotive coefficient
> $P=(1-\mathbf e\cdot\mathbf n_0)/2$ multiplying `ahat`.

> **2026-09-28 — RES088 superseded by RES090.** The Doppler factor, encounter flux,
> Jacobian placement, and linear spectrum remain current. The nonlinear coefficient is
> now the exact observer-dependent ratio $Q$ evaluated in Stage 2; chirped resonances use
> $D\bar C$ without changing RES082's definition of $D$.

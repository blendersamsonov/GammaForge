# RES059 — minimal kascade ports the emission chain, not the predecessor's framework

Status: implemented
Type: feature

*(Numbering note: this entry was written as RES058 and renumbered to RES059 on 2026-09-06
to resolve a collision with parallel NiceGUI work.)*

## Problem

Phase 5 needs kascade as an independent Monte-Carlo validation leg, but its historical
implementation is coupled to the predecessor's SI configuration, concrete Gaussian laser,
result container, and automatic particle-file output. Carrying those pieces across would
create a second input model and bypass the shared `LaserField` boundary. Reimplementing a
more general Monte Carlo would exceed kascade's deliberately limited role.

## Decision

`KascadeEngine` retains the historical sequential optical-depth inversion, polarized
Thomson angle sampler, optional Klein--Nishina rejection, and per-emission recoil in a
pure-array solver. The wrapper converts `Bunch` arrays from canonical CGS to SI once,
reconstitutes absolute electron weights as `N_e * Bunch.weight`, samples cycle-averaged
intensity through `LaserField.intensity_profile`, and converts `Results` back to CGS.

The engine writes no files and owns no mutable configuration. Its numerical controls are
validated `Parameters`, and macroparticle output uses `PhotonMacroparticles` plus `Bunch`.
The shared `Results.scaled` charge-only path scales photon macroparticle weights as well as
photon slices; final-electron relative weights remain unchanged.
The reported total yield is the weighted sum of each electron's integrated optical depth;
photon slices and dumps remain the seeded event sample. This makes the normalization anchor
deterministic while preserving the Monte-Carlo distribution needed for validation.

## Alternatives considered

**Copy the historical module and add a thin adapter.** This would preserve more lines
verbatim, but it would also retain the obsolete configuration/result types, a concrete
laser dependency, round-beam field reconstruction, scalar particle weights, and implicit
file output. Those violate the current engine and unit boundaries.

**Build the planned replacement Monte Carlo now.** A new nonlinear or arbitrary-field
algorithm would be a first-class engine, not the minimal cross-check Phase 5 calls for. Its
physics and interface should be decided with the parallel replacement effort rather than
inferred from kascade.

**Use the random emitted-photon count as total yield.** That is faithful to a literal event
count but makes the Thomson normalization check and cross-engine yield comparison noisier
without adding independent physics. The event count remains available in model-specific
results and in the weighted macroparticle sample.

## Rationale

The emission chain is the independently useful part of kascade. Sampling the actual shared
field keeps its overlap calculation independent of xigma's Stage 0 while honoring P15, and
the one checked CGS-to-SI conversion keeps the solver's established equations in their
native units. An on-axis closed-form photon-column test checks the complete normalization;
a separate bank-wide comparison checks the sampled overlap against the analytical Gaussian
integral.

## Consequences

Kascade now produces every current output kind and an optional final-electron/photon dump.
It appears in the engine-generic NiceGUI tabs but remains off by default. The particle
results panel reports both populations; existing HDF5 serialization preserves photon
macroparticles, while final-electron export remains the Phase 5 format/typing item.
Its angular kernel is still the historical linear-lab-x approximation. Non-default
polarization or crossing geometry is
therefore reported in `Results.model_specific` while the already-valid overlap, flux, and
energy changes are applied; this port does not claim to close DER004 or DER005.

The deterministic total and stochastic histogram from one run can differ by Monte-Carlo
counting noise. Consumers comparing distribution integrals must use statistical tolerances,
as required for MC legs by the validation plan.

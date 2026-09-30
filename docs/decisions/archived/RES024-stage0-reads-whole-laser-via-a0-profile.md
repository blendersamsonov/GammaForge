# RES024 — Stage 0 reads the whole laser through `a0_profile`

Status: implemented
Type: architecture
Archived: 2026-08-10

**Superseded by RES054** (2026-08-10). Stage 0 now reads *intensity_profile* — the
cycle-averaged `<a^2>` — and the constant is `4 pi`, not `8 pi`. **The decision below is
unchanged in substance**: Stage 0 still takes the whole laser through one sampling method
plus `active_region`, and the pulse energy still cancels. Only *which* method, because
`<a^2>` is polarization-agnostic where `a0` carries a convention.

## Problem

`integrate_trajectories` needs a laser-sampling method to compute the photon density each
trajectory sees, and the question is which quantity to sample and whether the `LaserField`
protocol needs widening to supply it.

## Decision

`integrate_trajectories` needs exactly two things from a `LaserField`: `a0_profile`
sampled along each trajectory, and `active_region` (through `overlap_time_window`) to
bound the integration. The photon density it needs comes from inverting the laser's own
energy→a0 chain — `photon_density_scale` — in which the pulse energy cancels, leaving
`n_photons(r,t) = a0(r,t)**2 (m_e c)**2 omega0 / (8 pi hbar e**2)`.

The remaining input, `omega0`, comes from `fit_gaussian_paraxial` — the descriptive-fit
route §3.4 already established for autoranging, not an attribute read off a concrete type.

## Alternatives considered

**Extend the `LaserField` protocol with a `photon_density` method.** Would work, but it
widens the protocol every future implementation must satisfy in order to supply something
already derivable from a method it must supply anyway.

**Have Stage 0 evaluate a Gaussian envelope itself, as the predecessor's `push_and_sample`
did by calling its own `pulse_envelope` with hand-passed `sigma_lr0`/`sigma_lz`/`beta_ff`
scalars.** This is what P15 exists to prevent — an engine that unpacks a laser into five
floats and re-derives its shape has hardcoded a Gaussian, whatever the type annotation
says, and a non-Gaussian `LaserField` cannot be substituted into it.

## Rationale

The cancellation of the pulse energy is what makes the chosen approach work and is worth
stating: two quantities that look independent (how bright the pulse is, how many photons
are in it) are one, because `a0` was defined from the same energy.

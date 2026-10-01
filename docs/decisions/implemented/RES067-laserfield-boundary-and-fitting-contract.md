# RES067 — LaserField boundary and fitting contract

Status: implemented
Type: architecture

## Problem

The architecture in `GRAND_PLAN.md` §3.3 and design principle P15 stated that engines
depended strictly on the `LaserField` protocol (`intensity_profile`, `a0_profile`,
`field`, `active_region`), with the expectation that arbitrary pulse representations
(such as from *Spectral-FEM-Fields*) could slot in with "zero changes to any engine."

An audit of the codebase revealed this contract was non-functional and structurally
incomplete:

1. **Total Execution Failure on Conforming Wrappers**: Any object implementing
   `LaserField` that was not a concrete `GaussianParaxialLaser` passed
   `isinstance(..., LaserField)` at runtime, but immediately raised `NotImplementedError`
   across all engines (`AnalyticalEngine`, `XigmaEngine`, `KascadeEngine`),
   `auto_ranges`, bunch illumination diagnostics, and drawing utilities.
2. **Hidden Invariant Dependence on `fit_gaussian_paraxial`**: The function
   `fit_gaussian_paraxial` implements an identity path only (RES010). While
   `GRAND_PLAN.md` §3.3 described this function as providing "descriptive metrics" (waist
   size, Rayleigh range, effective duration) for autoranging and visual sketches,
   downstream code actually consumed it for foundational physical invariants and
   kinematics:
   - `omega0` / `photon_energy`: carrier frequency and incident photon energy
     $\hbar\omega_0$, consumed by Xigma Stage 0 (`photon_density_scale`) to convert
     cycle-averaged $\langle a^2 \rangle$ to physical photon density $n_\gamma$, by
     Kascade for photon density in $\text{m}^{-3}$, by Xigma Stage 2 for emission
     kinematics and Compton edge scaling, and by `auto_ranges` for energy range
     boundaries.
   - `intensity_peak`: peak cycle-averaged intensity, used by Xigma Stage 0 to compute
     the envelope shape ratio $a_0(\mathbf{r}, t) / a_{0,\text{peak}}$.
   - `theta_xz`, `theta_yz`: lab-frame laser tilt/crossing angles, consumed by Stage 0
     for relative velocity $1 + \beta\cos\alpha$ and Stage 2 for the incident photon
     boost $\cos^2(\alpha/2)$.
   - `psi_pol`, `ellipticity`: global polarization orientation and ellipticity,
     consumed directly by Stage 2's emission kernel (DER004, DER005, DER006, RES060).
3. **Mismatched Physics Approximations Across Engines**:
   - `AnalyticalEngine` computes a closed-form bivariate Gaussian overlap integral
     (`overlap_yield`) and cannot compute exact yields for an arbitrary pulse by
     definition.
   - Xigma Stage 2 and Kascade assume a strictly monochromatic plane-wave-like incident
     field with uniform polarization $(\psi_{\text{pol}}, \varepsilon)$ and uniform
     propagation vector $\hat{\mathbf{k}}$.
   - Stage 0 field sampling via `intensity_profile` was the only engine layer truly
     capable of ingesting arbitrary spatio-temporal fields.

The claim of "zero engine changes" conflated Stage 0 trajectory sampling with the
monochromatic, uniformly polarized plane-wave kinematics hardcoded into Stage 2,
Kascade, and Analytical.

## Decision

1. **Explicit Engine Boundaries**:
   `AnalyticalEngine.run` explicitly checks `isinstance(interaction.laser, GaussianParaxialLaser)`
   and raises `TypeError` with an informative message if given any other `LaserField`,
   stating that its closed-form overlap integrals mathematically require an astigmatic
   paraxial Gaussian pulse.
2. **Decoupled Physical Invariants**:
   Engines and autoranging query physical invariants directly from the laser if exposed,
   rather than routing through `fit_gaussian_paraxial`:
   - `omega0` / `photon_energy` for carrier frequency and photon density scaling in Stage 0
     (`photon_density_scale`), Kascade, and `auto_ranges`.
   - `intensity_peak` for trajectory shape normalization in Stage 0 (`sample_trajectories`).
   - `polarization_axes` / `focusing_axes` for relative velocity in Stage 0 and crossing
     angle / polarization in Stage 2 and Kascade.
   - `ellipticity` for angle-resolved emission in Stage 2 and Kascade.
3. **Decoupled Spatial Autoranging**:
   `auto_ranges` inspects `laser.m("sigma_x")`, `laser.m("sigma_y")` when available, and
   falls back to `laser.active_region(1e-3).radius` for general `LaserField` sources.
4. **Physics Scope Boundaries**:
   - Broadband / polychromatic pulses are explicitly out of scope for now: the paper's
     numerical analysis shows that for realistic interaction scenarios, the spectral
     bandwidth of the laser does not significantly affect emitted photon observables;
     quasi-monochromatic carrier treatment is physically sufficient.
   - `fit_gaussian_paraxial` remains an identity check on `GaussianParaxialLaser` (RES010)
     raising `NotImplementedError` rather than guessing a numerical spatial fit.

## Alternatives considered

- **Build a numerical moment-based Gaussian fitter**:
  Sampling `intensity_profile` on a 3D grid and fitting Gaussian moments was considered
  in RES010 and rejected. Re-evaluating it confirms that rejection: even a perfect
  numerical spatial fitter cannot recover carrier frequency $\omega_0$ or polarization
  $(\psi_{\text{pol}}, \varepsilon)$ from cycle-averaged scalar intensity $\langle a^2 \rangle$.
  Moreover, routing arbitrary pulses through a Gaussian fitter produces results for the
  fitted Gaussian, not the actual field, concealing physical inaccuracies in Stage 2.
- **Leave `LaserField` unchanged and allow runtime crashes**:
  Leaving the 4-method protocol as the advertised engine boundary while every engine
  crashes at `fit_gaussian_paraxial` produces an illusion of modularity. Developers
  attempting to integrate a second field source immediately hit internal type errors.
- **Require full 4-vector field tensors in Stage 0 and Stage 2**:
  Generalizing Stage 2 to integrate local Stokes parameters or space-dependent
  electromagnetic tensors along trajectories would solve structured polarization
  physically, but requires paper author derivations that do not currently exist in the
  Compton numerics manuscript.

## Rationale

Decoupling the physical invariants from `fit_gaussian_paraxial` allows any quasi-monochromatic
field (such as future *Spectral-FEM-Fields* pulses or custom beam distributions) to execute
directly through Xigma and Kascade without pretending to be a Gaussian. At the same time,
making `AnalyticalEngine` explicitly reject non-Gaussian inputs prevents silent numerical
misapplication of Gaussian overlap integrals.

## Consequences

- Conforming quasi-monochromatic `LaserField` implementations run through `auto_ranges`,
  `XigmaEngine`, and the `delta` reference without calling `fit_gaussian_paraxial`.
- `AnalyticalEngine` cleanly rejects non-Gaussian inputs with `TypeError`.
- `GRAND_PLAN.md` P15 and §3.3 accurately reflect the engine boundaries without speculative
  claims.

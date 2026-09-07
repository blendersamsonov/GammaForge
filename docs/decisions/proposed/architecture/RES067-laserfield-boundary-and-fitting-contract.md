# RES067 — LaserField boundary and fitting contract

Status: proposed
Class: architecture

## Problem

The architecture in `GRAND_PLAN.md` §3.3 and design principle P15 states that engines
depend strictly on the `LaserField` protocol (`intensity_profile`, `a0_profile`,
`field`, `active_region`), allowing arbitrary pulse representations (such as from
*Spectral-FEM-Fields*) to slot in with "zero changes to any engine."

An audit of the codebase reveals this contract is currently non-functional and
structurally incomplete:

1. **Total Execution Failure on Conforming Wrappers**: Any object implementing
   `LaserField` that is not a concrete `GaussianParaxialLaser` passes
   `isinstance(..., LaserField)` at runtime, but immediately raises `NotImplementedError`
   across all engines (`AnalyticalEngine`, `XigmaEngine`, `KascadeEngine`),
   `auto_ranges`, bunch illumination diagnostics, and drawing utilities.
2. **Hidden Invariant Dependence on `fit_gaussian_paraxial`**: The function
   `fit_gaussian_paraxial` implements an identity path only (RES010). While
   `GRAND_PLAN.md` §3.3 describes this function as providing "descriptive metrics" (waist
   size, Rayleigh range, effective duration) for autoranging and visual sketches,
   downstream code actually consumes it for foundational physical invariants and
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
     propagation vector $\hat{\mathbf{k}}$. For polychromatic pulses (broadband, chirped)
     or structured beams (radial/azimuthal polarization, orbital angular momentum, vector
     vortices), a single carrier frequency and single polarization scalar pair do not
     exist.
   - Stage 0 field sampling via `intensity_profile` is the *only* engine layer truly
     capable of ingesting arbitrary spatio-temporal fields today.

Thus, the claim that arbitrary pulses can be handled with "zero engine changes"
conflated Stage 0 trajectory sampling with the monochromatic, uniformly polarized
plane-wave kinematics hardcoded into Stage 2, Kascade, and Analytical.

### Consumer / Requirement Mapping

| Consumer Component | Location | Direct `LaserField` calls | Direct `fit_gaussian_paraxial` calls | Requirement Category | Physical Role |
|---|---|---|---|---|---|
| Prefilter / Window | `io/bunch.py` | `active_region` | *(None)* | Approximate autoranging | Bounding cone for particle overlap time window `[t0, t1]`. |
| Target Auto-ranges | `io/target.py` | *(None)* | `photon_energy`, `sigma_x`, `sigma_y` | Normalization / autoranging | Compton edge energy for spectral axis; beam/laser minimum spot size for spatial axis. |
| Xigma Stage 0 | `engines/xigma/stages.py` | `intensity_profile` | `intensity_peak`, `omega0`, `theta_xz`, `theta_yz` | Sampling + Normalization + Geometry | Samples $\langle a^2 \rangle$; scales to photon density via $\omega_0$; normalizes $a_0$ shape ratio; relative velocity. |
| Xigma Stage 2 | `engines/xigma/collision.py` | *(None)* | `photon_energy`, `theta_xz`, `theta_yz`, `psi_pol`, `ellipticity` | Physics normalization + Polarization | Incident photon energy with crossing angle factor $\cos^2(\alpha/2)$; polarization basis vectors and ellipticity. |
| Analytical Engine | `engines/analytical/engine.py` | *(None)* | `photon_energy`, `cycle_average_factor`, full laser dataclass | Paraxial Gaussian convolution | Closed-form bivariate Gaussian overlap integrals and spectrum width formulas. |
| Kascade Engine | `engines/kascade/engine.py` | `intensity_profile` | `omega0`, `photon_energy`, `focusing_axes` | Sampling + Normalization + Geometry | Trajectory grid photon density via $\omega_0$; photon energy $\hbar\omega_0 / m_e c^2$; collision angle $\hat{\mathbf{k}} \cdot \hat{\mathbf{z}}$. |
| Illumination Diagnostics | `io/bunch.py` | *(None)* | `focusing_axes`, `spot_sizes`, `sigma_ct`, `beta_ff`, offsets | Paraxial Gaussian optimization | Closed-form quadratic solver for stationary point illumination and curvature. |
| Geometry Drawing | `io/drawing.py` | *(None)* | `focusing_axes`, `polarization_axes`, `sigma_x`, `sigma_y`, offsets, `ellipticity` | Schematic visualization | 3D schematic representation of waist ellipses, focal offsets, and polarization axes. |

## Proposal

1. **Explicit Engine Boundaries and Unsupported-Field Rejection**:
   - Formally declare that `AnalyticalEngine` requires `GaussianParaxialLaser`. It must
     raise a clear, descriptive `TypeError` if given an unsupported `LaserField`,
     rather than failing downstream in `fit_gaussian_paraxial`.
   - Update `GRAND_PLAN.md` §3.3 and P15 to replace the "zero engine changes" claim with
     an honest specification: Stage 0 samples arbitrary fields; Stage 2 and Kascade
     require quasi-monochromatic pulses with known carrier frequency and uniform
     polarization.
2. **Decouple Physical Invariants from Paraxial Gaussian Geometry**:
   - Define the minimal protocol needed by quasi-monochromatic non-Gaussian fields
     (*MonochromaticLaserField* or *PulseMetadata*):
     - `carrier_frequency() -> float` (or central wavelength)
     - `intensity_peak() -> float`
     - `propagation_direction() -> np.ndarray` (unit vector $\hat{\mathbf{k}}$)
     - `polarization_basis() -> tuple[np.ndarray, np.ndarray, float]` ($(\mathbf{e}_1, \mathbf{e}_2, \varepsilon)$)
   - Allow engines (Xigma Stage 0/Stage 2, Kascade) and `auto_ranges` to query these
     physical invariants directly from the laser if exposed, rather than forcing a
     synthetic `GaussianParaxialLaser` conversion.
3. **Decouple `auto_ranges` from Gaussian Spot Sizes**:
   - Modify `auto_ranges` to derive spatial bounds from `laser.active_region(threshold)`
     transverse bounds when `sigma_x`, `sigma_y` are unavailable, and use
     `carrier_frequency()` for the Compton edge.
4. **Scope of Numerical Fitting**:
   - Retain `fit_gaussian_paraxial` strictly for descriptive approximations (e.g.
     providing an equivalent Gaussian sketch for `drawing.py` or feeding
     `AnalyticalEngine` when an approximate estimate is explicitly requested). Acknowledge
     that numerical fitting cannot invent physical invariants (carrier frequency,
     polarization state) from scalar intensity grids alone.
5. **Author Physics Boundaries**:
   - **Broadband / Polychromatic Pulses (Resolved)**: Broadband pulses are explicitly out
     of scope for now. The paper's numerical analysis demonstrates that for realistic
     interaction scenarios, the spectral bandwidth/shape of the laser does not
     significantly affect the emitted photon observables; quasi-monochromatic carrier
     treatment is physically sufficient.
   - **Spatially Inhomogeneous Polarization**: In Stage 2, polarization vectors
     $\mathbf{e}_1, \mathbf{e}_2$ and ellipticity $\varepsilon$ are assumed constant
     across the whole beam. For structured light (e.g. radial/azimuthal polarization,
     vector vortices), can Stage 0 sample local polarization vectors along trajectories
     for Stage 2 consumption, or are such pulses scientifically unsupported?

## Alternatives considered

- **Build a numerical moment-based Gaussian fitter now**:
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

## Acceptance criteria

- A distinct conforming `LaserField` implementation (e.g. a non-Gaussian test wrapper)
  can be tested against the contract boundaries.
- Calling `AnalyticalEngine` on a non-Gaussian laser raises a clear, descriptive rejection
  explaining that the engine requires an analytical paraxial Gaussian pulse.
- Engines query physical invariants through explicit metadata protocols rather than
  asserting `isinstance(..., GaussianParaxialLaser)` or invoking `fit_gaussian_paraxial`.
- `auto_ranges` succeeds for any `LaserField` providing carrier frequency and
  `active_region`.
- `GRAND_PLAN.md` and `AGENTS.md` reflect the exact scope of arbitrary field support
  without speculative claims.

## Risks

- Breaking downstream callers that assume `fit_gaussian_paraxial` always returns a
  `GaussianParaxialLaser`.
- Introducing additional protocol methods onto `LaserField` could increase the
  implementation burden for external field providers (e.g. *Spectral-FEM-Fields*).
- Delaying full arbitrary-pulse execution in Stage 2 until paper authors formulate
  polychromatic and inhomogeneous polarization scattering kernels.

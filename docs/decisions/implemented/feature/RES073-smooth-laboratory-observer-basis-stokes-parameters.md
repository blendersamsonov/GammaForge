# RES073 — Smooth laboratory observer basis Stokes parameters

Status: implemented
Class: feature

## Problem
Characterizing the polarization state of scattered Compton photons across an electron bunch requires evaluating the Stokes parameters $(I, Q, U, V)$ (or polarization degree $P$ and polarization angle $\chi$). However, the coherence matrix and Stokes parameters are reference-frame dependent:
1. An electron-dependent polarization basis tied to the scattering plane $(\mathbf{v}_e, \mathbf{n})$ changes from particle to particle; summing coherence matrices across electrons with finite divergence mixes distinct reference frames and produces unphysical depolarization.
2. The standard spherical meridian basis $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$ tied to the plane $(\mathbf{n}, \mathbf{z}_0)$ places the topological coordinate singularity (per the Hairy Ball theorem) at the North Pole $\mathbf{n} = \mathbf{z}_0$ ($\theta_{\text{obs}} = 0$), the center of the Compton radiation cone. This causes an indeterminate $0/0$ division on axis and imposes an artificial $2\phi$ azimuth vortex that erroneously zeroes linear polarization upon integration across an aperture.

## Decision
Implement the smooth laboratory observer basis $(\mathbf{m}_x, \mathbf{m}_y)$ established in DER007:
1. The basis $(\mathbf{m}_x, \mathbf{m}_y)$ is constructed by parallel-transporting the fixed Cartesian laboratory axes $(\hat{\mathbf{x}}, \hat{\mathbf{y}})$ from $\mathbf{z}_0$ to $\mathbf{n}$ along the connecting great circle (Rodrigues' rotation without torsion). This places the coordinate singularity at the backward South Pole $-\mathbf{z}_0$, leaving the entire forward radiation cone smooth, orthonormal, and singularity-free. At $\mathbf{n} = \mathbf{z}_0$, $(\mathbf{m}_x, \mathbf{m}_y) = (\hat{\mathbf{x}}, \hat{\mathbf{y}})$ identically.
2. Kernel functions `rotated_laser_axes`, `compute_stokes_components`, and `stokes_parameters_vectorized` in `src/gammaforge/engines/xigma/stages.py` compute single-electron Stokes parameters $(I, Q, U, V)$ in the smooth observer basis for arbitrary crossing angles and laser ellipticity.
3. In `src/gammaforge/engines/xigma/stages.py`, `bunch_stokes_parameters` sums $(I, Q, U, V)$ over macroparticles weighted by emission luminosity $w_e = L_e$, yielding bunch polarization degree $P_{\text{bunch}} = \sqrt{Q_{\text{tot}}^2 + U_{\text{tot}}^2 + V_{\text{tot}}^2} / I_{\text{tot}} \le 1$ and polarization angle $\chi = \frac{1}{2}\operatorname{atan2}(U_{\text{tot}}, Q_{\text{tot}})$.
4. In `src/gammaforge/engines/xigma/collision.py`, `Collision.stokes_parameters()` exposes the bunch-integrated Stokes parameters via the `BunchStokes` named tuple.
5. Intensity-only Stage 2 kernels (`polarization_factor_vectorized` and `spectrum_from_table`) bypass Stokes matrix overhead, preserving high-performance scalar execution and machine-precision numerical agreement with existing references.

## Alternatives considered
- *Electron-dependent scattering plane basis*: defining polarization axes $(\mathbf{f}_0, \mathbf{f}_1)$ tied to each electron's velocity vector $\mathbf{v}_e$. Rejected because coherence matrices are tensor representations that cannot be summed componentwise across different coordinate bases without physical distortion.
- *Spherical meridian observer basis*: using standard spherical coordinates $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$. Rejected because the coordinate singularity sits directly at $\theta_{\text{obs}} = 0$, producing singular division on axis and an unphysical $2\phi$ vortex on $(Q, U)$.
- *Full 4D Stokes phase-space tensor*: constructing full $(I, Q, U, V)$ cubes for `OutputKind.COLLIMATED_SPECTRUM` across all spatial and energy grids. Deferred: primary polarimetry queries require integrated bunch Stokes characterization at chosen observation directions; extending spatial/spectral tensor grids can be built on top of `compute_stokes_components` as phase requirements demand.

## Rationale
The smooth basis $(\mathbf{m}_x, \mathbf{m}_y)$ satisfies every required physical and mathematical invariant:
- **Single-electron purity:** $I^2 = Q^2 + U^2 + V^2$ holds identically ($|P^2 - 1| < 10^{-13}$) for every individual electron regardless of divergence, crossing angle, or ellipticity.
- **Physical bunch depolarization:** summing pure-state single-electron Stokes vectors over a bunch with finite transverse emittance correctly models beam depolarization ($P_{\text{bunch}} \le 1$), with depolarization increasing monotonically with $\sigma_{\theta e} \gamma$.
- **Scalar intensity agreement:** the trace $I = M_{xx} + M_{yy}$ matches `polarization_factor` to machine precision ($< 10^{-14}$ relative difference).
- **Collinear limits:** on-axis collinear radiation produces uniform linear polarization $Q/I = \cos(2\psi_{\text{pol}}), U/I = \sin(2\psi_{\text{pol}})$ independent of azimuth, helicity transfer $V/I = \mp 1.0$ for circular polarization ($\varepsilon = \pm 1$), and a dipole null $I \equiv 0$ at $90^\circ$ crossing angle.

## Consequences
- `gammaforge.engines.xigma.stages` exports `rotated_laser_axes`, `compute_stokes_components`, `stokes_parameters_vectorized`, and `bunch_stokes_parameters`.
- `gammaforge.engines.xigma.collision` exports `BunchStokes` and `Collision.stokes_parameters`.
- `spectrum_sampler._polarization_parameters` reuses `rotated_laser_axes`, consolidating 3D rotation trigonometry across CPU and GPU pipelines.
- Unit tests in `tests/test_xigma_stokes.py` verify the smooth-basis limits and bunch polarization invariants of DER007.

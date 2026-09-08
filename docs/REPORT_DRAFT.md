# Scientific Report Draft: Numerical Modeling of High-Brightness Inverse Compton Sources with GammaForge

**Authors:** GammaForge Collaboration / ICS Working Group  
**Date:** September 2026  
**Document Status:** Working Draft & Physics Reference  
**Software Repository:** [`GammaForge`](file:///home/alexander/Work/Code/GammaForge)

---

## Executive Overview

This document provides the mathematical foundations, physical descriptions, literature citations, and proposed figure specifications for the scientific report on the latest developments in **GammaForge**. GammaForge is a high-performance simulation platform for computing the properties of Compton photons generated in relativistic electron-bunch / intense-laser interactions. 

The report focuses on six major capabilities:
1. **Computations using 6D electron distribution functions:** Arbitrary phase spaces and correlation-dependent collision yields.
2. **The flying focus technique:** Extended focal co-propagation for yield enhancement and mitigation of nonlinear ponderomotive spectral broadening.
3. **Elliptical polarization of laser pulses:** Angular emission patterns, cycle-averaged intensity invariants, and crossing-angle integration.
4. **Angle-resolved 3D collimated photon distributions & GPU architecture:** Kinematic annulus reduction and quasi-Monte Carlo acceleration in the `xigma` engine.
5. **Polarization properties of Compton photons:** Stokes vector $(I, Q, U, V)$ derivation, geometric depolarization, and comparison with literature approximations.
6. **Intensity-modulated laser pulses (pulse trains):** Duty cycle scaling, burst format optimization at constant total energy, and mitigation of nonlinear broadening.

---

## 1. Computations Using 6D Electron Distribution Functions

### 1.1 Arbitrary 6D Phase-Space Representation
In experimental and accelerator environments, realistic electron bunches deviate significantly from idealized Gaussian distributions due to RF wakefields, space-charge forces, coherent synchrotron radiation (CSR), and chromatic aberrations. 

In GammaForge, arbitrary distributions are ingested through the macroparticle container [`Bunch`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/bunch.py#L326-L364). Each particle $i \in \{1, \dots, N_p\}$ is represented by 7 flat arrays in canonical CGS-Gaussian units without coordinate normalization ([`RES015`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES015-no-coordinate-normalization.md), [`RES064`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES064-input-boundary-snapshots-and-array-ownership.md)):
$$
\mathbf{\zeta}_i = \left( x_i,\, y_i,\, z_i,\, \theta_{x,i},\, \theta_{y,i},\, \gamma_i,\, w_i \right)
$$
where:
* $(x_i, y_i, z_i)$ are spatial coordinates (cm).
* $(\theta_{x,i}, \theta_{y,i}) \approx (p_{x,i}/p_{z,i}, p_{y,i}/p_{z,i})$ are transverse trajectory angles (rad).
* $\gamma_i$ is the relativistic Lorentz factor.
* $w_i$ is the dimensionless per-particle relative weight ($\sum_i w_i = 1$ for an unfiltered bunch).

Particle momenta are kinematically derived rather than independently sampled, guaranteeing that the relativistic mass-shell condition $\gamma^2 - \mathbf{p}^2 = 1$ holds identically:
$$
p_{z,i} = \sqrt{\frac{\gamma_i^2 - 1}{1 + \theta_{x,i}^2 + \theta_{y,i}^2}}, \qquad p_{x,i} = \theta_{x,i} p_{z,i}, \qquad p_{y,i} = \theta_{y,i} p_{z,i}.
$$

### 1.2 Correlated 6D Gaussian Representation
For parametric analysis and optimization, [`GaussianElectronBeam`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/bunch.py#L91-L148) describes an analytic 6D Gaussian distribution. Rather than using dimensional slopes (which do not preserve marginal energy distributions), correlations are parameterized through dimensionless **correlation coefficients** $\rho \in (-1, 1)$ ([`RES005`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/architecture/RES005-beam-correlations-as-coefficients.md)):
$$
\boldsymbol{\rho}_\gamma = \left( \rho_{x,\gamma},\, \rho_{y,\gamma},\, \rho_{z,\gamma},\, \rho_{\theta_x,\gamma},\, \rho_{\theta_y,\gamma} \right)
$$
* $\rho_{z,\gamma}$: Longitudinal energy chirp ($\operatorname{corr}(z, \gamma)$).
* $\rho_{x,\gamma}, \rho_{y,\gamma}$: Transverse spatial dispersion.
* $\rho_{\theta_x,\gamma}, \rho_{\theta_y,\gamma}$: Transverse angular dispersion derivative ($\operatorname{corr}(\theta_x, \gamma)$).

Including the angle-energy correlations $(\rho_{\theta_x,\gamma}, \rho_{\theta_y,\gamma})$ ensures that spatial drift transport operators $\mathcal{D}(L): x \to x + L\theta_x$ compose consistently ([`RES012`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/bug-fix/RES012-angle-energy-correlations-stored.md)):
$$
\Sigma_{x,\gamma}(L) = \Sigma_{x,\gamma}(0) + L\,\Sigma_{\theta_x,\gamma}(0).
$$

Each macroparticle draws independent standard normal deviates $\mathbf{n} = (n_x, n_y, n_z, n_{\theta_x}, n_{\theta_y}, n_\gamma)$, with the Lorentz factor assembled via:
$$
\gamma = \gamma_0 + \sigma_\gamma \left( a_x n_x + a_y n_y + a_z n_z + a_{\theta_x} n_{\theta_x} + a_{\theta_y} n_{\theta_y} + \sqrt{1 - \sum_k a_k^2}\; n_\gamma \right),
$$
where the explained variance must strictly satisfy $\sum_k a_k^2 < 1$ for physical realizability.

### 1.3 Influence of Energy-z Chirp ($\rho_{z,\gamma}$) on Total Interaction Yield
The total scattered photon yield in a head-on collision is given by the 4D space-time convolution of electron density $n_e$ and laser photon density $n_L$ ([`DER001`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER001-gaussian-luminosity-overlap-integral.md)):
$$
N_\gamma = \sigma_T (1+\beta_0) \int_{-\infty}^\infty \mathrm{d}t \iiint \mathrm{d}^3\mathbf{r}\; n_e(\mathbf{r}, t)\, n_L(\mathbf{r}, t).
$$
For Gaussian profiles, integrating over transverse coordinates $(x, y)$ yields the reduced $(z, ct)$ integral:
$$
N_\gamma = \frac{\sigma_T (1+\beta_0) N_e N_L}{4\pi^2 \sigma_{ez} s_{ct}} \iint \mathrm{d}z\,\mathrm{d}(ct)\; \frac{\exp\left[ -\dfrac{(z - \beta_0 ct)^2}{2\sigma_{ez}^2} - \dfrac{(z + ct)^2}{2 s_{ct}^2} \right]}{\sqrt{\det\left[\mathsf{C}_e(z) + \mathsf{C}_L(-z)\right]}},
$$
where $\mathsf{C}_e(z) = \operatorname{diag}(\sigma_{ex}^2(z), \sigma_{ey}^2(z))$ and $\mathsf{C}_L(z) = \operatorname{diag}(\sigma_{Lx}^2(z), \sigma_{Ly}^2(z))$ describe the transverse beam envelopes along the interaction path.

When a strong longitudinal chirp $\rho_{z,\gamma} \neq 0$ is present:
1. Electrons at the bunch head ($z > 0$) have different Lorentz factors $\gamma(z)$ than those at the bunch tail ($z < 0$):
   $$
   \langle \gamma \rangle(z) = \gamma_0 + \rho_{z,\gamma} \sigma_\gamma \frac{z}{\sigma_{ez}}.
   $$
2. Because beam emittance drives chromatic waist defocusing ($\sigma_{ex}(z) \propto \epsilon_x / \gamma$), the focal waist hourglass shape couples directly to the longitudinal position.
3. If the collision is temporally detuned by an offset $\Delta t$, the interaction selects a specific slice of the chirped bunch, altering both the total integrated yield $N_\gamma$ and the central backscattered energy.

### 1.4 Key Derivations & Decisions
* **Derivations:** [`DER001 — Gaussian luminosity overlap integral`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER001-gaussian-luminosity-overlap-integral.md).
* **Decisions:** [`RES005`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/architecture/RES005-beam-correlations-as-coefficients.md) (Correlations as correlation coefficients), [`RES012`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/bug-fix/RES012-angle-energy-correlations-stored.md) (Angle-energy correlation drift composability), [`RES064`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES064-input-boundary-snapshots-and-array-ownership.md) (Owned macroparticle arrays and boundary immutability).

### 1.5 Literature References
* K. L. Brown, *A First- and Second-Order Matrix Theory for the Design of Beam Transport Systems and Charged Particle Spectrometers*, SLAC-75 (1982).
* C. Curatolo, I. Drebot, V. Petrillo, and L. Serafini, *Analytical description of photon beam phase spaces in inverse Compton scattering sources*, [Phys. Rev. Accel. Beams **20**, 080701 (2017)](https://doi.org/10.1103/PhysRevAccelBeams.20.080701).
* N. Ranjan, B. Terzić, G. A. Krafft, et al., *Simulation of inverse Compton scattering and its implications on the scattered linewidth*, [Phys. Rev. Accel. Beams **21**, 030701 (2018)](https://doi.org/10.1103/PhysRevAccelBeams.21.030701).

### 1.6 Suggested Report Figures
* **Figure 1.1: Compton Photon Yield vs. Energy-z Correlation ($\rho_{z,\gamma}$)**
  * *X-axis:* Correlation coefficient $\rho_{z,\gamma} \in [-0.9, 0.9]$.
  * *Y-axis:* Total scattered photon yield $N_\gamma$ (normalized to $\rho_{z,\gamma} = 0$).
  * *Curves:* Multiple ratios of bunch length to laser Rayleigh range ($\sigma_{ez}/z_R \in \{0.2, 1.0, 3.0\}$) and timing offsets $c\Delta t / \sigma_{ez} \in \{-1.0, 0, 1.0\}$.
  * *Caption/Physics:* Demonstrates the sensitivity of total luminosity to electron bunch chirp when waist hour-glassing is prominent.
* **Figure 1.2: Phase-Space Ingestion: Tracking Particle Distribution vs. Gaussian Fit**
  * *Panels:* (a) 2D scatter/density plot of an exported accelerator bunch ($(x, x')$ and $(z, \gamma)$) showing nonlinear phase-space curvature (e.g. from longitudinal wakefields). (b) Corresponding reconstructed 4D illuminated distribution $\mathcal{H}(\gamma, \theta_x, \theta_y, \hat{a})$. (c) Resulting on-axis photon spectrum comparing direct particle tracking against an equivalent 6D Gaussian fit.
  * *Caption/Physics:* Demonstrates GammaForge's capability to ingest arbitrary, non-ideal phase spaces without loss of non-Gaussian spectral features.
* **Figure 1.3: Transverse Dispersion ($\rho_{x,\gamma}$) Impact on Spot Symmetry and Spectral Bandwidth**
  * *Panels:* Left: Transverse cross-section of emitted X-rays on a downstream target for uncoupled vs. dispersed beams ($\rho_{x,\gamma} = 0.7$). Right: Angle-integrated energy spectrum showing chromatic linewidth broadening induced by dispersion.

---

## 2. The Flying Focus Technique to Improve Brightness

### 2.1 Spatiotemporal Laser Control & Synchronization
In standard Inverse Compton Scattering, maximizing photon yield requires tight laser focusing. However, paraxial diffraction confines high intensity to the Rayleigh range:
$$
z_R = \frac{\pi w_0^2}{\lambda_L}.
$$
If the interaction length $L_{\rm int} \sim c\tau_L \gg z_R$, the electron bunch interacts with an expanding hourglass beam, causing the local intensity to drop rapidly away from the focal plane.

The **flying focus** technique overcomes this limit using chirped laser pulses combined with chromatic focusing optics (e.g., a diffractive lens or axiprecision mirror), causing the focal plane to travel at an arbitrary velocity $v_f = -\beta_{\rm ff} c$.

In GammaForge ([`DER002`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER002-flying-focus.md)), the spot size is evaluated at the moving local coordinate:
$$
u_{\rm spot} = u + \beta_{\rm ff}\, ct, \qquad u = \hat{\mathbf{k}} \cdot \mathbf{r}.
$$
For head-on propagation ($\hat{\mathbf{k}} = -\hat{\mathbf{z}}$), the focal waist moves along $+\hat{\mathbf{z}}$ at velocity $+\beta_{\rm ff} c$. When $\beta_{\rm ff} = 1$, the optical waist **co-propagates synchronously** with the ultrarelativistic electron bunch.

### 2.2 Mitigation of Nonlinear Spectral Broadening
The fundamental Doppler resonance frequency for backscattered photons is:
$$
\omega_R = \frac{4\omega_L \gamma^2}{1 + \gamma^2\theta^2 + \hat{a}},
$$
where $\hat{a}$ is the trajectory-averaged effective laser intensity parameter:
$$
\hat{a} \equiv \langle a^2 \rangle_{\rm peak} \frac{\int |E(t)|^4 \mathrm{d}t}{\int |E(t)|^2 \mathrm{d}t}.
$$
In a conventional focused pulse:
1. Achieving high total yield requires high peak intensity $a_0 \sim 0.8\text{--}1.5$.
2. Because electrons experience large intensity gradients across the waist, the ponderomotive red-shift $1/(1+\hat{a})$ varies widely across the ensemble.
3. This creates severe **nonlinear spectrum broadening**:
   $$
   \left(\frac{\Delta \omega}{\omega}\right)_{\rm NL} \approx \frac{\Delta \hat{a}}{1 + \hat{a}} \propto a_0^2.
   $$

**The Flying Focus Advantage:**  
By extending the high-intensity focal zone over a distance $L_{\rm int} \gg z_R$, a flying focus pulse produces the **same integrated photon yield at significantly lower peak field intensity $a_0$**. Consequently:
* Ponderomotive red-shift variations across the bunch are drastically suppressed ($\Delta \hat{a} \to 0$).
* Peak on-axis spectral brightness $\mathcal{B} \equiv \frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega\,\mathrm{d}t}$ is enhanced by several fold.

### 2.3 Mathematical Properties & Reciprocal Invariance
GammaForge evaluates flying focus collisions via a 2D principal-axis quadrature ([`RES044`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/feature/RES044-flying-focus-2d-grid.md)):
$$
N_\gamma = \frac{\sigma_T(1+\beta_0)N_eN_L}{4\pi^2\sigma_{ez}s_{ct}} \iint \mathrm{d}z\,\mathrm{d}w\; \frac{\exp\left[-\dfrac{(z-\beta_0w)^2}{2\sigma_{ez}^2} -\dfrac{(z+w)^2}{2s_{ct}^2}\right]}{\sqrt{\det\!\big(\mathsf{C}_e(z)+\mathsf{C}_L(-z+\beta_{\rm ff}w)\big)}},
$$
where $w \equiv ct$.

Rigorous analysis in [`DER002`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER002-flying-focus.md) reveals two fundamental properties:
1. **Synchronization Optimum ($\beta_{\rm ff} = 1$):** Yield reaches an absolute maximum when the focus co-moves with the bunch ($2.8\times$ enhancement over $\beta_{\rm ff} = 0$ for a $30\,\mu\text{m}$ bunch).
2. **Exact Reciprocal Symmetry:** For short bunches ($\sigma_{ez} \ll z_R$), the total yield is invariant under:
   $$
   \beta_{\rm ff} \longleftrightarrow \frac{1}{\beta_{\rm ff}}.
   $$
   This follows from paraxial wave solutions: along the collision ridge $z \approx ct$, the spot coordinate scales as $(\beta_{\rm ff} - 1)ct$, while the effective Rayleigh range carries a $(1 + \beta_{\rm ff})$ factor. The spot size depends on $\frac{\beta_{\rm ff}-1}{\beta_{\rm ff}+1}$, which is odd under $\beta_{\rm ff} \to 1/\beta_{\rm ff}$, rendering the squared width and the resulting yield invariant.

### 2.4 Key Derivations & Decisions
* **Derivation:** [`DER002 — Flying focus`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER002-flying-focus.md).
* **Decisions:** [`RES022`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/bug-fix/RES022-active-region-flying-focus-coordinate.md) (Active region coordinate), [`RES044`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/feature/RES044-flying-focus-2d-grid.md) (Mandatory 2D grid quadrature; rejection of unsafe 1D approximations).

### 2.5 Literature References
* D. H. Froula, D. Turnbull, A. S. Davies, et al., *Spatiotemporal control of laser intensity*, [Nat. Photonics **12**, 262–265 (2018)](https://doi.org/10.1038/s41566-018-0121-8).
* J. P. Palastro, D. Turnbull, P. Franke, et al., *Dephasingless laser wakefield acceleration*, [Phys. Rev. Lett. **124**, 134802 (2020)](https://doi.org/10.1103/PhysRevLett.124.134802).
* A. Sainte-Marie, O. Gobert, and F. Quéré, *Controlling the velocity of a femtosecond laser pulse using specialized optics*, [Optica **4**, 1298–1304 (2017)](https://doi.org/10.1364/OPTICA.4.001298).
* D. Ramsey, P. Franke, T. T. Simpson, et al., *Nonlinear Thomson scattering with a flying focus*, [Phys. Rev. E **102**, 043207 (2020)](https://doi.org/10.1103/PhysRevE.102.043207).

### 2.6 Suggested Report Figures
* **Figure 2.1: Relative Yield vs. Flying Focus Velocity $\beta_{\rm ff}$ & Reciprocal Symmetry**
  * *X-axis:* Flying focus parameter $\beta_{\rm ff} \in [-0.5, 2.5]$.
  * *Y-axis:* Yield enhancement factor $N(\beta_{\rm ff}) / N(0)$.
  * *Curves:* Short bunch ($\sigma_{ez} = 30\,\mu\text{m}$) showing the $2.8\times$ peak at $\beta_{\rm ff} = 1.0$ and symmetric agreement at reciprocal points ($\beta_{\rm ff} = 0.5$ and $2.0$), contrasted with a long bunch ($\sigma_{ez} = 3\,\text{mm}$).
  * *Caption/Physics:* Highlights the synchronization maximum and numerically demonstrates the $\beta_{\rm ff} \to 1/\beta_{\rm ff}$ paraxial symmetry.
* **Figure 2.2: Spectral Brightness and Suppression of Nonlinear Broadening**
  * *X-axis:* Normalized photon energy $\omega / \omega_{\max}$.
  * *Y-axis:* On-axis spectral intensity $\mathrm{d}^2N / (\mathrm{d}\omega\,\mathrm{d}\Omega)$.
  * *Curves:* (1) Standard stationary waist at $a_0 = 0.8$ (broadened red-shifted plateau). (2) Synchronized flying focus ($\beta_{\rm ff} = 1.0$) at $a_0 = 0.35$ configured to deliver identical total yield.
  * *Caption/Physics:* Shows dramatic narrowing of the full-width at half-maximum (FWHM) and $>3\times$ increase in peak spectral brightness.
* **Figure 2.3: Spatiotemporal Laser Intensity Map in the Interaction Plane**
  * *Panels:* 2D contour maps of $a^2(z, ct)$ showing (a) stationary Rayleigh hourglass ($\beta_{\rm ff} = 0$) where electrons rapidly leave the waist, vs. (b) moving focus ($\beta_{\rm ff} = 1.0$) where the waist tracks the electron world-line $z = ct$.

---

## 3. Elliptical Polarization of Laser Pulses

### 3.1 Invariant Energy Scaling & Intensity Definitions
In literature, defining the normalized vector potential $a_0$ for arbitrary polarization often introduces confusing $(1+\varepsilon^2)$ factors. GammaForge establishes consistency by distinguishing peak instantaneous amplitude from the **cycle-averaged field energy density** ([`DER004`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER004-ellipticity-in-the-emission-kernel.md), [`RES054`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES054-physics-through-a-squared-not-a0.md)):
$$
\langle a^2 \rangle = C\, a_0^2, \qquad C = \begin{cases} 1/2, & \text{linear polarization}, \\ 1, & \text{circular polarization}. \end{cases}
$$
The cycle-averaged energy density of the laser pulse is:
$$
U = \frac{\langle E^2 + B^2 \rangle}{8\pi} = \frac{\langle E^2 \rangle}{4\pi} = \frac{1}{4\pi}\left( \frac{m_e c \omega_L}{e} \right)^2 \langle a^2 \rangle.
$$
Inverting at fixed total pulse energy $U$:
$$
a_0 = \frac{e}{m_e c \omega_L} \sqrt{\frac{4\pi U}{C}}.
$$
Because total laser photon density $n_{\rm ph} = U / (\hbar \omega_L)$ is determined solely by pulse energy and central frequency, **the total angle-integrated photon yield $N_\gamma$ is strictly independent of laser ellipticity $\varepsilon$ at fixed pulse energy** ([`DER004`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER004-ellipticity-in-the-emission-kernel.md) §1.1).

### 3.2 Emission Kernel Polarization Factor
Laser ellipticity $\varepsilon \in [0, 1]$ (defined as the ratio of minor to major axis) and major-axis azimuth $\psi_{\rm pol}$ enter the differential cross section through the polarization density matrix $\hat{\Xi}$:
$$
\boldsymbol{\epsilon} = \frac{1}{\sqrt{1+\varepsilon^2}} \mathbf{e}_0 + \frac{i\varepsilon}{\sqrt{1+\varepsilon^2}} \mathbf{e}_1, \qquad \Xi_{ij} = \epsilon_i \epsilon_j^*.
$$
In the electron rest frame, the scattered photon polarization vectors are:
$$
\mathbf{u}_i = \frac{(\mathbf{n} - \mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1 - \mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_i.
$$
Evaluating the trace $\operatorname{Tr}(\hat{U}^T \hat{\Xi} \hat{U}) = \sum_{i,j} \Xi_{ij} (\mathbf{u}_i \cdot \mathbf{u}_j)$ for head-on geometry yields ([`DER004`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER004-ellipticity-in-the-emission-kernel.md)):
$$
\boxed{\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right) = 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}\cdot \frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2}}
$$
where $\psi$ is the observation azimuth relative to the laser polarization major axis $\mathbf{e}_0$.

**Limiting Cases:**
* **Linear Polarization ($\varepsilon = 0$):**
  $$
  \operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right) = 1 - \frac{4\gamma^2\theta^2\cos^2\psi}{(1+\gamma^2\theta^2)^2}.
  $$
  Exhibits the classic transverse dipole radiation pattern with strong nodes along the polarization axis ($\psi = 0$).
* **Circular Polarization ($\varepsilon = 1$):**
  $$
  \operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right) = 1 - \frac{2\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}.
  $$
  Azimuthal dependence cancels identically, yielding a completely isotropic annular ring.

### 3.3 Generalization to Arbitrary Crossing Angles
When colliding at crossing angle $\alpha$, the longitudinal velocity projection $\mathbf{v}\cdot\mathbf{e}_i$ can no longer be neglected. As derived in [`DER006`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER006-polarization-matrix-with-ellipticity-and-crossing-angle.md):
$$
\mathbf{u}_i \cdot \mathbf{u}_j = \delta_{ij} - \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} + \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j) + (\mathbf{n}\cdot\mathbf{e}_j)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}}.
$$
Contracting with $\hat{\Xi}$ maintains the zero cross-term identity $\Re(\Xi_{01}) = 0$, giving the exact lab-frame general kernel:
$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right) = \sum_{i=0}^1 \Xi_{ii} \left[ 1 - \frac{(\mathbf{n}\cdot\mathbf{e}_i)^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} + \frac{2(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} \right].
$$
This expression rigorously enforces the physical dipole null at $\alpha = 90^\circ$ where previous literature approximations fail ([`DER005`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER005-crossing-angle-in-the-emission-kernel.md)).

### 3.4 Key Derivations & Decisions
* **Derivations:** [`DER004 — Ellipticity in emission kernel`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER004-ellipticity-in-the-emission-kernel.md), [`DER006 — Polarization matrix with ellipticity and crossing angle`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER006-polarization-matrix-with-ellipticity-and-crossing-angle.md).
* **Decisions:** [`RES053`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/bug-fix/RES053-ahat-polarization-cycle-average.md) (Cycle-average factor), [`RES054`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES054-physics-through-a-squared-not-a0.md) (Physics through $\langle a^2 \rangle$), [`RES069`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/feature/RES069-cupy-incident-polarization-and-crossing-geometry.md) (CuPy incident polarization).

### 3.5 Literature References
* E. Esarey, S. K. Ride, and P. Sprangle, *Nonlinear Thomson scattering of intense laser pulses from beams and plasmas*, [Phys. Rev. E **48**, 3003 (1993)](https://doi.org/10.1103/PhysRevE.48.3003).
* G. A. Krafft, B. Terzić, E. Johnson, and G. Wilson, *Scattered spectra from inverse Compton sources operating at high laser fields and high electron energies*, [Phys. Rev. Accel. Beams **26**, 034401 (2023)](https://doi.org/10.1103/PhysRevAccelBeams.26.034401).

### 3.6 Suggested Report Figures
* **Figure 3.1: 2D Angular Intensity Distributions for Increasing Ellipticity**
  * *Panels:* 2D false-color maps of $d^2N / (d\theta_x d\theta_y)$ for $\varepsilon \in \{0.0\text{ (linear)}, 0.3, 0.7, 1.0\text{ (circular)}\}$.
  * *Caption/Physics:* Visualizes the transformation from a double-lobe dipolar profile to an azimuthally symmetric circular ring.
* **Figure 3.2: Azimuthal Modulation Contrast vs. Ellipticity at the Critical Angle $\theta = 1/\gamma$**
  * *X-axis:* Azimuthal angle $\psi \in [0, 2\pi]$.
  * *Y-axis:* Normalized differential intensity $I(\psi) / I(0)$.
  * *Curves:* Distinct $\varepsilon$ values validating the contrast formula $C = (1-\varepsilon^2)/(1+\varepsilon^2)$.
* **Figure 3.3: Invariance of Total Integrated Yield vs. Ellipticity**
  * *Plot:* Total photon yield as a flat horizontal line across $\varepsilon \in [0, 1]$ at fixed pulse energy, proving the physical correctness of the cycle-averaged formulation.

---

## 4. Angle-Resolved Spectrum of Compton Photons & xigma GPU Strategy

### 4.1 3D Collimated Distribution & Delta-Resonance Model
Experiments typically collimate the scattered beam using round or rectangular apertures. GammaForge outputs the full 3D distribution:
$$
\frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}\theta_x\,\mathrm{d}\theta_y} \equiv \frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}.
$$
Evaluating the 8D collision integral directly for fine grids ($50 \times 50 \times 100$) requires evaluating oscillatory Jackson integrals across millions of macroparticles, which is computationally prohibitive.

The **xigma engine** decouples tracking from spectral synthesis based on the scale separation principle ([`DER009`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER009-reduced-3d-emission-kernel-under-delta-resonance-approximation.md)):
$$
\Delta\omega_{\rm bunch} \gg \delta\omega_{\rm natural}.
$$
Because electron beam emittance and energy spread induce a frequency spread orders of magnitude wider than the single-electron Fourier linewidth, the resonance shape acts as a Dirac delta function:
$$
R(\omega - \omega_R) \longrightarrow \delta(\omega - \omega_R).
$$
1. **Stage 0/1:** Particles stream through the laser pulse, depositing their column densities $\mathcal{L}_i$ and effective intensities $\hat{a}_i$ into a 4D histogram:
   $$
   \mathcal{H}(\gamma, \theta_x, \theta_y, \hat{a}).
   $$
2. **Stage 2:** For an observation direction $\mathbf{n} = (\theta_{x0}, \theta_{y0})$ and frequency $s = \omega / (4\omega_L)$, delta sifting analytically evaluates the energy integral at the unique root:
   $$
   \Gamma(s, \theta, \hat{a}) = \sqrt{\frac{1+\hat{a}}{\frac{1}{s} - \theta^2}}, \qquad \theta = \sqrt{(\theta_x - \theta_{x0})^2 + (\theta_y - \theta_{y0})^2}.
   $$

### 4.2 GPU Strategy: Kinematic Annulus & Quasi-Monte Carlo Quadrature
While brute-force CPU summation scales as $\mathcal{O}(N_{\rm out} \cdot N_{\theta_x} N_{\theta_y} N_{\hat{a}})$ ($\sim 10^7\text{--}10^8$ interpolations per slice), the GPU CuPy rawkernel ([`DER008`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/derived/DER008-ring-annulus-importance-sampling-spectrum-kernel.md)) achieves optimal efficiency through:
1. **Kinematic Annular Domain Reduction:**  
   Electrons capable of emitting into $(s, \mathbf{n})$ are kinematically restricted to a narrow radial ring $[r_{\min}, r_{\max}]$ in relative angle space:
   $$
   r_{\min}^2 = \max\left(0,\, \frac{1}{s} - \frac{1+\hat{a}_{\max}}{\gamma_{\rm lo}^2}\right), \qquad r_{\max}^2 = \frac{1}{s} - \frac{1+\hat{a}_{\min}}{\gamma_{\rm hi}^2}.
   $$
2. **Analytic Bounding-Box Arc Clipping:**  
   Concentric annular rings are analytically intersected with the finite rectangular phase-space window $[-d_x, d_x] \times [-d_y, d_y]$, immediately pruning unpopulated angles.
3. **Prefix-Sum CDF Importance Sampling:**  
   The 2D spatial marginal $\mathcal{H}_{\rm marginal} = \sum_{\gamma, \hat{a}} \mathcal{H}$ guides sample allocation across active arcs using exact shared-memory prefix-sum cumulative inversion ([`RES068`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/bug-fix/RES068-cupy-sampling-measure-and-stable-polarization.md)).
4. **2D Fibonacci Rank-1 Lattice Quasi-Random Sampling:**  
   Samples are drawn on a low-discrepancy Fibonacci lattice with uniform $r^2$ stratification:
   $$
   \theta^2 = \theta_{\min}^2 + \operatorname{frac}(\alpha \Phi) (\theta_{\max}^2 - \theta_{\min}^2), \qquad \Phi = \frac{\sqrt{5}+1}{2}.
   $$
   This eliminates radial metric distortions and provides $\mathcal{O}(N^{-1})$ integration convergence.

### 4.3 GPU vs. CPU Performance Benchmarks
As documented in [`docs/ALPHA_GPU_VALIDATION.md`](file:///home/alexander/Work/Code/GammaForge/docs/ALPHA_GPU_VALIDATION.md) on an NVIDIA GTX 1660 Ti:

| Scenario / Table Size | NumPy CPU Brute-Force | CuPy GPU Rawkernel | Speedup |
| :--- | :--- | :--- | :--- |
| Single $\hat{a}$ slice ($32 \times 32 \times 32 \times 1$) | $5\text{--}12\text{ s}$ | **$3.50\text{ ms}$** (kernel) / $5.44\text{ ms}$ (wall) | **$\sim 1500\times$** |
| Resolved $\hat{a}$ grid ($32 \times 32 \times 32 \times 32$) | $120\text{--}300\text{ s}$ | **$92.57\text{ ms}$** (kernel) / $96.30\text{ ms}$ (wall) | **$\sim 2000\times$** |

### 4.4 Key Derivations & Decisions
* **Derivations:** [`DER008 — Ring/annulus importance sampling`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/derived/DER008-ring-annulus-importance-sampling-spectrum-kernel.md), [`DER009 — Reduced 3D emission kernel`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER009-reduced-3d-emission-kernel-under-delta-resonance-approximation.md).
* **Decisions:** [`RES029`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES029-stage2-numpy-kernel-brute-force.md), [`RES052`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/simplification/RES052-remove-spectral-angular-distribution.md), [`RES062`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES062-cupy-importance-sampler-production-path.md), [`RES068`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/bug-fix/RES068-cupy-sampling-measure-and-stable-polarization.md), [`RES069`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/feature/RES069-cupy-incident-polarization-and-crossing-geometry.md).

### 4.5 Literature References
* C. Schretter, L. Kobbelt, and P.-O. Dehaye, *Golden Ratio Sequences for Low-Discrepancy Sampling*, [J. Graph. Tools **16**, 95–104 (2012)](https://doi.org/10.1080/2165347X.2012.679555).
* R. M. Feshchenko, A. V. Vinogradov, and I. A. Artyukov, *Mathematical model for calculating parameters of X-ray radiation of a laser-electron generator*, Preprint FIAN No. 2, Moscow (2016).

### 4.6 Suggested Report Figures
* **Figure 4.1: Volumetric 3D Collimated Distribution Rendering**
  * *Visualization:* 3D isometric isosurface plot of $d^3N / (d\omega\,d\theta_x\,d\theta_y)$ coupled with 2D cut-planes: $(E_\gamma, \theta_x)$ showing the characteristic off-axis parabolic dispersive redshift $\omega(\theta)$, and $(\theta_x, \theta_y)$ showing detector spatial profile.
* **Figure 4.2: Collimated Energy Spectra vs. Aperture Semi-Angle $\theta_{\rm col}$**
  * *Plot:* Integrated spectrum $\mathrm{d}N/\mathrm{d}\omega$ for acceptance half-angles $\theta_{\rm col} \in [0.1/\gamma, 2.0/\gamma]$, highlighting the trade-off between photon flux and monochromaticity.
* **Figure 4.3: Runtime Scaling and Numerical Convergence (CPU vs. GPU)**
  * *Panels:* Left: Wall-clock execution time vs. detector grid resolution showing flat millisecond-level GPU response vs. steep CPU scaling. Right: Relative error versus subsample count demonstrating quasi-Monte Carlo convergence rates.

---

## 5. Polarization Properties of Compton Photons (Stokes Parameters)

### 5.1 Scattered Polarization Coherence Matrix and Smooth Laboratory Basis
In [`DER007`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md), GammaForge derives the complete Stokes vector $(I, Q, U, V)$ for scattered photons. 

To sum coherence matrices over an electron bunch with angular divergence, all electrons must project onto the **same basis**. 

**Topological Constraint (Hairy Ball Theorem):**  
Any continuous tangent vector field on the 2-sphere $S^2$ must possess a coordinate singularity. Defining the basis via the spherical meridian plane $(\mathbf{n}, \mathbf{z}_0)$ places the singularity directly at the North Pole $\mathbf{n} = \mathbf{z}_0$ ($\theta_{\rm obs} = 0$), the center of the Compton cone. This causes an indeterminate $0/0$ division on axis and imposes an artificial $2\phi$ vortex on $(Q, U)$ that falsely cancels out linear polarization when integrated across a detector aperture.

Because high-energy Compton photons are emitted strictly in a narrow forward cone around $+\mathbf{z}_0$, we place the coordinate singularity at the **South Pole ($-\mathbf{z}_0$, backward scattering)**, where no scattered radiation ever reaches.

We define the **smooth laboratory observer basis $(\mathbf{m}_x, \mathbf{m}_y)$** by parallel-transporting the fixed laboratory Cartesian axes $(\hat{\mathbf{x}}, \hat{\mathbf{y}})$ from $\mathbf{z}_0$ to $\mathbf{n}$ along the great circle connecting them (Rodrigues' rotation without torsion):
$$
\mathbf{m}_x = \begin{pmatrix} 1 - \frac{n_x^2}{1+n_z} \\ -\frac{n_x n_y}{1+n_z} \\ -n_x \end{pmatrix} \approx \begin{pmatrix} 1 - \frac{1}{2}\theta_x^2 \\ -\frac{1}{2}\theta_x\theta_y \\ -\theta_x \end{pmatrix}, \qquad
\mathbf{m}_y = \begin{pmatrix} -\frac{n_x n_y}{1+n_z} \\ 1 - \frac{n_y^2}{1+n_z} \\ -n_y \end{pmatrix} \approx \begin{pmatrix} -\frac{1}{2}\theta_x\theta_y \\ 1 - \frac{1}{2}\theta_y^2 \\ -\theta_y \end{pmatrix},
$$
where $(\theta_x, \theta_y)$ are transverse observation angles. At $\theta_{\rm obs} = 0$, $\mathbf{m}_x = \hat{\mathbf{x}}$ and $\mathbf{m}_y = \hat{\mathbf{y}}$ identically, matching standard laboratory detector pixel axes without singularity.

Projecting the rest-frame polarization vectors $\mathbf{u}_i$:
$$
U_{ik} \equiv \mathbf{u}_i \cdot \mathbf{m}_k = -\mathbf{m}_k \cdot \mathbf{e}_i - \frac{(\mathbf{m}_k \cdot \mathbf{v}_e)(\mathbf{n} \cdot \mathbf{e}_i)}{1 - \mathbf{v}_e\cdot\mathbf{n}}, \qquad k \in \{x, y\}.
$$
The scattered-photon polarization coherence matrix is:
$$
M_{kl} = (\hat{U}^T \hat{\Xi} \hat{U})_{kl} = \sum_{i,j} \Xi_{ij} U_{ik} U_{jl}.
$$
The single-electron Stokes vector $(I, Q, U, V)$ in the laboratory frame is:
$$
I = M_{xx} + M_{yy}, \qquad Q = M_{xx} - M_{yy}, \qquad U = 2\Re(M_{xy}), \qquad V = 2\Im(M_{xy}).
$$
Because $(\mathbf{m}_x, \mathbf{m}_y)$ is identical for all particles, the bunch coherence matrix is the direct sum $\hat{M}_{\rm bunch} = \sum_e w_e \hat{M}^{(e)}$, whose Stokes vector reflects the true physical depolarization ($P_{\rm bunch} \le 1$) induced by bunch emittance.

### 5.2 Explicit Formulae: On-Axis Head-On Limit
In the collinear head-on limit ($\mathbf{v}_e = \beta\hat{\mathbf{z}}$, $\mathbf{n} = \hat{\mathbf{z}}$):
$$
\begin{aligned}
I &\to 1, \\
Q &\to \frac{1 - \varepsilon^2}{1 + \varepsilon^2} \cos(2\psi_{\rm pol}), \\
U &\to \frac{1 - \varepsilon^2}{1 + \varepsilon^2} \sin(2\psi_{\rm pol}), \\
V &\to -\frac{2\varepsilon}{1 + \varepsilon^2}.
\end{aligned}
$$
Notice that $Q$ and $U$ are **strictly independent of observation azimuth $\phi$**, correctly reproducing the uniform laboratory polarization state across the beam center:
- Linear horizontal ($\psi_{\rm pol} = 0, \varepsilon = 0$): $Q = +1, U = 0, V = 0$.
- Linear diagonal ($\psi_{\rm pol} = \pi/4, \varepsilon = 0$): $Q = 0, U = +1, V = 0$.
- Linear vertical ($\psi_{\rm pol} = \pi/2, \varepsilon = 0$): $Q = -1, U = 0, V = 0$.
- Circular ($\varepsilon = 1$): $Q = 0, U = 0, V = -1$.

**Physical Invariants:**
* The degree of polarization is strictly preserved for every single electron:
  $$
  P = \frac{\sqrt{Q^2 + U^2 + V^2}}{I} \equiv 1.
  $$
* Linear polarization ($\varepsilon = 0$): $V = 0$, $Q^2 + U^2 = 1$ (pure linear polarization).
* Circular polarization ($\varepsilon = 1$): $Q = U = 0$, $|V/I| = 1$ (complete helicity transfer).

### 5.3 Finite Crossing Angles: Geometric Depolarization & Literature Comparison
In literature, **Feshchenko et al. (2016)** modeled crossing angles by applying a 2D planar rotation matrix $M_\sigma = \mathsf{O}^T \mathsf{M}_0 \mathsf{O}$ to the head-on scattering matrix. 

As proved in [`DER007`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md) §10, this factorization relies on neglecting the coupling term:
$$
\frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j)}{1 - \mathbf{v}\cdot\mathbf{n}} \sim \frac{\theta \sin\alpha}{1 - \mathbf{v}\cdot\mathbf{n}}.
$$
* **Domain of Failure:** This 2D approximation breaks down whenever $\sin\alpha \gtrsim 1/\gamma$. For typical sources ($\gamma \sim 100$, crossing angle $\alpha \sim 50\text{ mrad}$), $\sin\alpha \approx 0.05 \gg 1/\gamma = 0.01$.
* **Geometric Depolarization ($P < 1$):** GammaForge's 3D lab-frame projection ([`RES060`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/bug-fix/RES060-polarization-uses-per-particle-lab-velocity.md)) shows that a finite crossing angle mixes polarization components in three dimensions. Consequently, the scattered beam becomes **partially depolarized ($P < 1$)** even when the incident laser is in a pure polarization state.
* **Dipole Null Consistency:** At $\alpha = 90^\circ$, GammaForge correctly predicts complete extinction along the dipole oscillation axis, whereas the uncorrected literature formulas yield unphysical negative intensities.

### 5.4 Key Derivations & Decisions
* **Derivations:** [`DER006 — Polarization matrix`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER006-polarization-matrix-with-ellipticity-and-crossing-angle.md), [`DER007 — Stokes parameters`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md).
* **Decisions:** [`RES060`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/bug-fix/RES060-polarization-uses-per-particle-lab-velocity.md) (Lab-frame velocity projection).

### 5.5 Literature References
* R. M. Feshchenko, A. V. Vinogradov, and I. A. Artyukov, *Mathematical model for calculating parameters of X-ray radiation of a laser-electron generator*, Preprint FIAN No. 2, Moscow (2016).
* W. H. McMaster, *Matrix representation of polarization*, [Rev. Mod. Phys. **33**, 8–28 (1961)](https://doi.org/10.1103/RevModPhys.33.8).
* J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, New York (1998), Chap. 14.

### 5.6 Suggested Report Figures
* **Figure 5.1: 2D Detector Maps of the Four Stokes Parameters $(I, Q, U, V)$**
  * *Panels:* Four quadrant heatmaps over $(\theta_x, \theta_y)$ displaying normalized Stokes parameters $(I/I_{\max}, Q/I, U/I, V/I)$ for an incident pulse with $\varepsilon = 0.5$ and $\psi_{\rm pol} = 0$.
  * *Caption/Physics:* Illustrates spatial variation of linear vs. circular polarization across the emission cone.
* **Figure 5.2: Helicity Transfer: Circular Stokes Parameter $V/I$ vs. Laser Ellipticity**
  * *Plot:* On-axis Stokes $V/I$ versus laser ellipticity $\varepsilon \in [0, 1]$, demonstrating exact numerical agreement with the theoretical curve $V/I = 2\varepsilon / (1+\varepsilon^2)$.
* **Figure 5.3: Crossing-Angle Depolarization and Comparison with 2D Approximation**
  * *X-axis:* Laser crossing angle $\alpha \in [0, 90^\circ]$.
  * *Y-axis:* Degree of polarization $P = \sqrt{Q^2+U^2+V^2}/I$ and normalized Stokes $Q/I$.
  * *Curves:* Comparison between GammaForge (exact 3D lab projection, showing physical depolarization and $90^\circ$ dipole null) and the Feshchenko (2016) 2D rotation model (which artificially maintains $P=1$ and diverges at large angles).

---

## 6. Interaction with Intensity-Modulated Laser Pulses (Pulse Trains)

### 6.1 Technological Motivation & Pulse Train Architecture
In practical high-brightness Inverse Compton Scattering facilities, practical limitations often prevent concentrating the entire available laser energy $E_{\rm tot}$ into a single ultra-short, ultra-intense pulse. Such bottlenecks include:
1. **Optical Damage Thresholds:** Laser-induced damage threshold (LIDT) limits on final dielectric compression gratings, turning optics, and vacuum windows.
2. **Nonlinear Phase Accumulation ($B$-integral):** Self-focusing and beam breakup in optical amplifiers when peak power exceeds critical thresholds.
3. **Emerging Laser Architectures:** Advanced high-average-power sources—such as coherent pulse stacking (CPS) in Gires-Tournois cavities, multi-channel fiber laser arrays, or high-repetition-rate burst-mode chains—natively produce **trains of $N_p$ sub-pulses** ($N_p \sim 10\text{--}100$) rather than isolated monolithic pulses.

Here, we consider the interaction of such a pulse train against a **single electron bunch** of duration $\tau_e = \sigma_{ez}/c$, where the total laser energy $E_{\rm tot}$ and focusing optics (waist $w_0$, Rayleigh length $z_R$) are held constant.

### 6.2 Mathematical Representation in GammaForge (`LaserField` Temporal Envelope)
GammaForge’s calculation engines interface with laser pulses via the vectorized, lab-frame [`LaserField`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py#L115-L148) protocol ([`RES067`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES067-laserfield-boundary-and-fitting-contract.md)). Any non-Gaussian temporal structure is represented by modulating the transverse paraxial spatial mode with an arbitrary longitudinal temporal envelope $g(\tau)$:
$$
I(x, y, z; t) = I_{\perp}(x, y, z) \cdot |g(t + z/c)|^2,
$$
where $\tau = t + z/c$ is the retarded light-cone phase coordinate along the propagation direction $-\hat{\mathbf{z}}$.

For an intensity-modulated pulse train composed of $N_p$ Gaussian sub-pulses:
$$
g(\tau) = \sum_{k=1}^{N_p} A_k \exp\left[ -\frac{(\tau - t_k)^2}{2\sigma_{\tau}^2} \right],
$$
where:
* $t_k = \left( k - \frac{N_p + 1}{2} \right) T_{\rm rep}$ is the arrival time of sub-pulse $k \in \{1, \dots, N_p\}$.
* $T_{\rm rep}$ is the inter-pulse temporal period (repetition interval).
* $\sigma_\tau = \tau_p / (2\sqrt{2\ln 2})$ is the RMS duration of each individual sub-pulse (FWHM duration $\tau_p$).
* $T_{\rm burst} \approx (N_p - 1) T_{\rm rep}$ is the total temporal extent of the train.

**Duty Cycle Definition:**
The temporal structure of the pulse train is parameterized by the **duty cycle** $D \in (0, 1]$:
$$
D \equiv \frac{\tau_p}{T_{\rm rep}}.
$$
When $D \to 1$, adjacent sub-pulses merge into a continuous flat-top envelope; as $D \ll 1$, the train fragments into isolated, well-separated intensity spikes.

**Energy Normalization & Amplitude Partitioning:**
Conserving total laser energy $E_{\rm tot} = \text{const}$ across all configurations requires:
$$
E_{\rm tot} = \sum_{k=1}^{N_p} E_k = N_p E_k = \text{const} \implies E_k = \frac{E_{\rm tot}}{N_p}.
$$
Consequently, the peak vector potential $a_{0, k}$ of each individual sub-pulse is scaled down relative to a single pulse with the same sub-pulse duration $\tau_p$:
$$
a_{0, k} = \frac{a_{0, \text{single}}}{\sqrt{N_p}}.
$$

### 6.3 Collision Kinematics & Rayleigh Range Hourglass Effects
In a head-on collision, an electron at $z_e(t) = \beta c t$ encounters the $k$-th sub-pulse (whose center moves as $z_{L, k}(t) = -c(t - t_k)$) at time and space coordinates:
$$
t_{{\rm coll}, k} = \frac{t_k}{1 + \beta} \approx \frac{t_k}{2}, \qquad z_{{\rm coll}, k} \approx \beta c\, t_{{\rm coll}, k} \approx \frac{c\, t_k}{2}.
$$
The collision points with the $N_p$ sub-pulses are therefore longitudinally distributed over a spatial length:
$$
\Delta Z_{\rm coll} \approx \frac{c T_{\rm burst}}{2} = \frac{c (N_p - 1) T_{\rm rep}}{2} = \frac{c (N_p - 1) \tau_p}{2 D}.
$$

This leads to two distinct physical regimes governed by the duty cycle $D$ and burst duration:

1. **High Duty Cycle / Compact Burst ($\Delta Z_{\rm coll} \lesssim z_R$ and $T_{\rm burst} \lesssim \sigma_{ez}/c$):**
   * All sub-pulses collide with the electron bunch while inside the optical depth of focus $z_R = \pi w_0^2 / \lambda_L$.
   * Across all collisions, the transverse spot size remains at its minimum waist $w(z_{{\rm coll}, k}) \approx w_0$.
   * **Luminosity Preservation:** The total photon yield is nearly equal to that of a single monolithic pulse:
     $$
     N_\gamma \approx N_{\gamma, \text{single}}.
     $$
   * **Mitigation of Nonlinear Spectral Broadening:** Because each sub-pulse carries peak intensity $a_{0, k} = a_{0, \text{single}}/\sqrt{N_p}$, the ponderomotive red-shift variation and nonlinear spectral broadening are suppressed by a factor of $N_p$:
     $$
     \left(\frac{\Delta \omega}{\omega}\right)_{\rm NL} \approx \langle a_k^2 \rangle \approx \frac{\langle a^2 \rangle_{\text{single}}}{N_p}.
     $$
     Partitioning a high-energy pulse into a tight train enables high-yield X-ray production with **narrow, quasi-linear spectral bandwidths**, completely avoiding the nonlinear plateau of a single high-$a_0$ pulse.

2. **Low Duty Cycle / Sparse Burst ($\Delta Z_{\rm coll} \gg z_R$ or $T_{\rm burst} \gg \sigma_{ez}/c$):**
   * As $D$ decreases at constant $\tau_p$, $T_{\rm rep}$ expands and peripheral sub-pulses collide at $|z_{{\rm coll}, k}| \gg z_R$.
   * At these locations, paraxial diffraction broadens the laser spot to $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$, causing the local photon density to decay as $n_L(z) \propto 1 / w^2(z) \approx (z_R / z)^2$.
   * Furthermore, if the electron bunch length $\sigma_{ez}/c \ll T_{\rm burst}$, outer pulses arrive before or after the electron bunch has passed the interaction zone, missing the bunch completely.
   * As a result, the total yield $N_\gamma$ drops monotonically as the duty cycle $D$ is reduced.

### 6.4 Key Code References
* **Code Interfaces:** [`LaserField`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py#L115-L148) (Generic sampling contract), [`GaussianParaxialLaser`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py#L173-L250) (Paraxial spatial field backbone).
* **Decisions:** [`RES054`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES054-physics-through-a-squared-not-a0.md) (Formulation via $\langle a^2 \rangle$), [`RES067`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES067-laserfield-boundary-and-fitting-contract.md) (Arbitrary field execution through Stage 0/1 without fitting).

### 6.5 Literature References
* G. A. Krafft, E. Johnson, K. Deitrick, B. Terzić, R. Kelmar, T. Hodges, W. Melnitchouk, and J. R. Delayen, *Laser pulsing in linear Compton scattering*, [Phys. Rev. Accel. Beams **19**, 121302 (2016)](https://doi.org/10.1103/PhysRevAccelBeams.19.121302).
* T. Zhou, J. Rauschenberger, et al., *Coherent pulse stacking amplification using low-finesse Gires-Tournois interferometers*, [Opt. Lett. **40**, 1053–1056 (2015)](https://doi.org/10.1364/OL.4.001053).
* F. V. Hartemann, F. Albert, C. W. Siders, and C. P. J. Barty, *High-energy Compton scattering sources: Theory and design*, [Phys. Rev. Lett. **105**, 130801 (2010)](https://doi.org/10.1103/PhysRevLett.105.130801).

### 6.6 Suggested Report Figures
* **Figure 6.1: Total Photon Yield vs. Duty Cycle ($D = \tau_p / T_{\rm rep}$) at Fixed Total Energy**
  * *X-axis:* Duty cycle $D \in [0.01, 1.0]$ (logarithmic scale) for constant total energy $E_{\rm tot}$ and fixed sub-pulse duration $\tau_p$.
  * *Y-axis:* Normalized total scattered photon yield $N_\gamma(D) / N_{\gamma, \text{single}}$.
  * *Curves:* Multiple burst lengths relative to the Rayleigh range ($c N_p \tau_p / z_R \in \{0.2, 1.0, 5.0\}$) and electron bunch durations ($\sigma_{ez} / z_R$).
  * *Caption/Physics:* Highlights the transition from the compact Rayleigh-confined regime ($D \to 1$, full yield preservation) to the geometric hourglass divergence loss regime ($D \ll 1$).
* **Figure 6.2: Pulse Train Partitioning: Yield and Nonlinear Linewidth vs. Number of Sub-Pulses $N_p$**
  * *X-axis:* Number of sub-pulses $N_p \in [1, 100]$ in the train (at constant total energy $E_{\rm tot}$ and total burst duration constrained within the Rayleigh range $c T_{\rm burst} \le z_R$).
  * *Left Y-axis:* Total photon yield $N_\gamma$ (solid line, showing near-constant yield across $N_p$).
  * *Right Y-axis:* On-axis spectral linewidth FWHM $(\Delta\omega/\omega)_{\rm NL}$ (dashed line, dropping as $1/N_p$).
  * *Caption/Physics:* Demonstrates that pulse stacking circumvents optic damage thresholds and suppresses nonlinear broadening by $N_p$ without compromising photon flux.
* **Figure 6.3: Spatiotemporal Collision Map $(z, ct)$ for Monolithic vs. Modulated Trains**
  * *Panels:* 2D $(z, ct)$ contour maps of local overlap luminosity density $n_e(z, t) n_L(z, t)$:
    (a) Single monolithic pulse ($N_p = 1$, localized high-$a_0$ collision).
    (b) Compact pulse train ($N_p = 20$, $D = 0.5$, collisions clustered inside $|z| < z_R$).
    (c) Sparse pulse train ($N_p = 20$, $D = 0.05$, collisions distributed over $|z| \gg z_R$ displaying geometric luminosity drop in the diffraction wings).

---

## 7. Summary Matrix: Derivations, Code Modules, and Literature

| Section | Topic | Core Derivation | Implementation File | Key Literature |
| :--- | :--- | :--- | :--- | :--- |
| **§1** | 6D Phase Space & Chirp Yield | [`DER001`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER001-gaussian-luminosity-overlap-integral.md) | [`bunch.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/bunch.py), [`formulas.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/analytical/formulas.py) | Curatolo et al. (2017), Brown (1982) |
| **§2** | Flying Focus & Brightness | [`DER002`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER002-flying-focus.md) | [`formulas.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/analytical/formulas.py), [`laser.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py) | Froula et al. (2018), Palastro et al. (2020) |
| **§3** | Elliptical Polarization | [`DER004`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER004-ellipticity-in-the-emission-kernel.md), [`DER006`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER006-polarization-matrix-with-ellipticity-and-crossing-angle.md) | [`stages.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/stages.py), [`laser.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py) | Esarey et al. (1993), Krafft et al. (2023) |
| **§4** | 3D Collimated Spectrum & GPU | [`DER008`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/derived/DER008-ring-annulus-importance-sampling-spectrum-kernel.md), [`DER009`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER009-reduced-3d-emission-kernel-under-delta-resonance-approximation.md) | [`spectrum_sampler.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/spectrum_sampler.py), [`stages.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/stages.py) | Schretter et al. (2012), Feshchenko (2016) |
| **§5** | Stokes Vector & Depolarization | [`DER007`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md), [`DER006`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER006-polarization-matrix-with-ellipticity-and-crossing-angle.md) | [`stages.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/stages.py), [`collision.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/collision.py) | Feshchenko (2016), McMaster (1961) |
| **§6** | Pulse Train Envelope & Duty Cycle | [`DER001`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER001-gaussian-luminosity-overlap-integral.md), [`DER010`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER010-laser-photon-density-scale-and-cycle-averaged-field-energy.md) | [`laser.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py), [`stages.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/engines/xigma/stages.py) | Krafft et al. (2016), Zhou et al. (2015) |

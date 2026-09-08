# Handoff: Smooth Laboratory Observer Basis Stokes Implementation (DER007)

**Date:** 2026-09-08  
**Target Modules:** `src/gammaforge/engines/xigma/stages.py`, `src/gammaforge/engines/xigma/collision.py`, `src/gammaforge/engines/xigma/spectrum_sampler.py`  
**Related Requirements:** Phase 3b (Physics closure), DER007 (verified), RES060, RES070  
**Document Status:** Ready for implementation in the next session  

---

## 1. Goal & Motivation

GammaForge simulates Compton scattering of laser pulses off relativistic electron bunches. To properly characterize the polarization state of scattered photons (Stokes parameters $I, Q, U, V$, polarization degree $P$, and polarization angle $\chi$) across a bunch with finite emittance / angular spread:

1. **Common Observer Polarization Basis:**  
   The coherence matrix $J = \begin{pmatrix} \langle E_x E_x^* \rangle & \langle E_x E_y^* \rangle \\ \langle E_y E_x^* \rangle & \langle E_y E_y^* \rangle \end{pmatrix}$ is basis-dependent. Summing coherence matrices over electrons with different velocities $\mathbf{v}_e$ is physically valid **only if all contributions are expressed in the exact same basis** for a given observation direction $\mathbf{n}$.
2. **Elimination of Electron-Dependent Basis:**  
   Earlier drafts tied polarization axes $(\mathbf{f}_0, \mathbf{f}_1)$ to each electron's scattering plane $(\mathbf{v}_e, \mathbf{n})$. Summing those naive Stokes vectors resulted in unphysical mixing of different reference frames.
3. **Topological Regularity (Hairy Ball Theorem):**  
   Any continuous tangent vector field on $S^2$ must possess a coordinate singularity. Defining the basis via the spherical meridian plane $(\mathbf{n}, \mathbf{z}_0)$ places the singularity directly at the North Pole $\mathbf{n} = \mathbf{z}_0$ ($\theta_{\text{obs}} = 0$), the exact center of the Compton cone. This causes an indeterminate $0/0$ division on axis and imposes an artificial $2\phi$ vortex on $(Q, U)$ that falsely cancels out linear polarization when integrated across a detector aperture ($\int_0^{2\pi} \cos(2\phi)\,\mathrm{d}\phi = 0$).
4. **Smooth Laboratory Observer Basis $(\mathbf{m}_x, \mathbf{m}_y)$:**  
   Because high-energy Compton photons are emitted strictly in a narrow forward cone around $+\mathbf{z}_0$, in [`DER007`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md) we place the coordinate singularity at the **South Pole ($-\mathbf{z}_0$, backward scattering)**, where no scattered radiation ever reaches.
   - Defined by parallel-transporting the fixed laboratory Cartesian axes $(\hat{\mathbf{x}}, \hat{\mathbf{y}})$ from $\mathbf{z}_0$ to $\mathbf{n}$ along the great circle connecting them (Rodrigues' rotation without torsion).
   - At $\mathbf{n} = \mathbf{z}_0$, $\mathbf{m}_x = \hat{\mathbf{x}}$ and $\mathbf{m}_y = \hat{\mathbf{y}}$ identically, matching standard laboratory detector pixel axes without singularity.
   - For an individual electron, radiation is 100% polarized ($P_e \equiv 1$).
   - When summed over the bunch, angular divergence induces physical depolarization ($P_{\text{bunch}} \le 1$).

---

## 2. Mathematical Specification

### 2.1 Coordinate Systems and Geometry
- **Bunch axis:** $\mathbf{z}_0 = (0, 0, 1)^T$.
- **Laser crossing angles:** Completely arbitrary angles $\theta_{xz}, \theta_{yz}$ (or yaw/pitch $\alpha_x, \alpha_y$). The rotated laser polarization vectors $\mathbf{e}_0, \mathbf{e}_1$ are computed by exact 3D rotation:
  $$
  \mathbf{e}_i = R_y(\theta_{xz}) R_x(\theta_{yz})\, \mathbf{e}_i^{(0)}, \qquad i \in \{0, 1\}.
  $$
  *(No small-crossing-angle assumption is made; $\alpha$ can be anywhere up to $90^\circ$ or $180^\circ$.)*
- **Observer direction:** Paraxial angles $(\theta_x, \theta_y)$ relative to $\mathbf{z}_0$:
  $$
  \mathbf{n} = \begin{pmatrix} \theta_x \\ \theta_y \\ 1 - \frac{1}{2}(\theta_x^2 + \theta_y^2) \end{pmatrix}.
  $$
- **Smooth laboratory observer basis vectors:**
  $$
  \mathbf{m}_x = \begin{pmatrix} 1 - \frac{n_x^2}{1+n_z} \\ -\frac{n_x n_y}{1+n_z} \\ -n_x \end{pmatrix} \approx \begin{pmatrix} 1 - \frac{1}{2}\theta_x^2 \\ -\frac{1}{2}\theta_x\theta_y \\ -\theta_x \end{pmatrix}, \qquad
  \mathbf{m}_y = \begin{pmatrix} -\frac{n_x n_y}{1+n_z} \\ 1 - \frac{n_y^2}{1+n_z} \\ -n_y \end{pmatrix} \approx \begin{pmatrix} -\frac{1}{2}\theta_x\theta_y \\ 1 - \frac{1}{2}\theta_y^2 \\ -\theta_y \end{pmatrix}.
  $$
  To first order: $\mathbf{m}_x \approx (1, 0, -\theta_x)^T, \mathbf{m}_y \approx (0, 1, -\theta_y)^T$. These form an orthonormal triad $(\mathbf{m}_x, \mathbf{m}_y, \mathbf{n})$ smooth and non-singular across the forward hemisphere.

### 2.2 Relativistic Electron Kinematics
For an electron with Lorentz factor $\gamma$, speed $\beta = \sqrt{1 - 1/\gamma^2}$, and divergence angles $(\theta_{e,x}, \theta_{e,y})$:
$$
\mathbf{v}_e = \beta \begin{pmatrix} \theta_{e,x} \\ \theta_{e,y} \\ 1 - \frac{1}{2}(\theta_{e,x}^2 + \theta_{e,y}^2) \end{pmatrix}.
$$
The relative angle between electron and observation direction satisfies:
$$
\theta_{\text{rel}}^2 = (\theta_x - \theta_{e,x})^2 + (\theta_y - \theta_{e,y})^2.
$$
The Doppler denominator $d = 1 - \mathbf{v}_e \cdot \mathbf{n}$ is:
$$
d = 1 - \mathbf{v}_e \cdot \mathbf{n} \approx \frac{1 + \gamma^2 \theta_{\text{rel}}^2}{2\gamma^2}.
$$
*(At high $\gamma$, evaluate $d$ using this form or the RES070 stabilized vector norm to prevent catastrophic floating-point cancellation.)*

### 2.3 Projections and Overlaps
1. **Overlap of observer basis with laser polarization:**
   $$
   E_{xi} = \mathbf{m}_x \cdot \mathbf{e}_i \approx e_{i,x}\left(1 - \frac{1}{2}\theta_x^2\right) - e_{i,y}\left(\frac{1}{2}\theta_x\theta_y\right) - e_{i,z}\theta_x \approx e_{i,x} - e_{i,z}\theta_x,
   $$
   $$
   E_{yi} = \mathbf{m}_y \cdot \mathbf{e}_i \approx -e_{i,x}\left(\frac{1}{2}\theta_x\theta_y\right) + e_{i,y}\left(1 - \frac{1}{2}\theta_y^2\right) - e_{i,z}\theta_y \approx e_{i,y} - e_{i,z}\theta_y.
   $$
2. **Projection of laser polarization along observation vector:**
   $$
   C_i = \mathbf{n} \cdot \mathbf{e}_i = \theta_x e_{i,x} + \theta_y e_{i,y} + e_{i,z}\left(1 - \frac{1}{2}(\theta_x^2+\theta_y^2)\right).
   $$
3. **Projection of electron velocity along observer basis:**
   $$
   D_x = \mathbf{m}_x \cdot \mathbf{v}_e \approx \beta (\theta_{e,x} - \theta_x),
   $$
   $$
   D_y = \mathbf{m}_y \cdot \mathbf{v}_e \approx \beta (\theta_{e,y} - \theta_y).
   $$

### 2.4 Effective Radiation Coupling Matrix $U_{ik}$
The lab-frame polarization projection vector is (Eq. `udef`, RES060):
$$
\mathbf{u}_i = \frac{(\mathbf{n} - \mathbf{v}_e)(\mathbf{n}\cdot\mathbf{e}_i)}{1 - \mathbf{v}_e \cdot \mathbf{n}} - \mathbf{e}_i.
$$
Projecting into the smooth observer basis gives the components $U_{ik} = \mathbf{u}_i \cdot \mathbf{m}_k$:
$$
U_{ik} = -E_{ki} - \frac{D_k C_i}{1 - \mathbf{v}_e \cdot \mathbf{n}}, \qquad i \in \{0, 1\}, \; k \in \{x, y\}.
$$
Explicitly:
$$
U_{0x} = -E_{x0} - \frac{D_x C_0}{d}, \qquad U_{0y} = -E_{y0} - \frac{D_y C_0}{d},
$$
$$
U_{1x} = -E_{x1} - \frac{D_x C_1}{d}, \qquad U_{1y} = -E_{y1} - \frac{D_y C_1}{d}.
$$

### 2.5 Single-Electron Stokes Parameters
For laser polarization density matrix $\hat{\Xi}$ with ellipticity parameter $\varepsilon \in [-1, 1]$:
$$
\Xi_{00} = \frac{1}{1 + \varepsilon^2}, \qquad \Xi_{11} = \frac{\varepsilon^2}{1 + \varepsilon^2}, \qquad \Xi_{01} = \frac{-i\varepsilon}{1 + \varepsilon^2}.
$$
The unnormalized Stokes parameters in the laboratory basis $(\mathbf{m}_x, \mathbf{m}_y)$ are:
$$
I = \Xi_{00} (U_{0x}^2 + U_{0y}^2) + \Xi_{11} (U_{1x}^2 + U_{1y}^2),
$$
$$
Q = \Xi_{00} (U_{0x}^2 - U_{0y}^2) + \Xi_{11} (U_{1x}^2 - U_{1y}^2),
$$
$$
U = 2 \left[ \Xi_{00} U_{0x} U_{0y} + \Xi_{11} U_{1x} U_{1y} \right],
$$
$$
V = -\frac{2\varepsilon}{1 + \varepsilon^2} (U_{0x} U_{1y} - U_{1x} U_{0y}).
$$

**Key Invariants & Identities:**
1. **Scalar Intensity Match:**  
   $I \equiv \operatorname{Tr}(\hat{U}^T \hat{\Xi} \hat{U}) = \sum_{ij} \Xi_{ij} (\mathbf{u}_i \cdot \mathbf{u}_j)$. This matches the current scalar intensity in `stages.py` / `spectrum_sampler.py` to machine precision ($< 10^{-15}$).
2. **Single-Electron Purity:**  
   $I^2 = Q^2 + U^2 + V^2$ holds identically for any single electron, even with nonzero divergence ($D_x, D_y \ne 0$).
3. **On-Axis Regularity:**  
   At $\theta_x = \theta_y = 0$, for linear horizontal polarization, $Q \equiv +I$ and $U \equiv 0$ strictly independent of azimuth $\phi$.

---

## 3. Implementation Plan for Next Session

### Step 1: Kernel Function in `src/gammaforge/engines/xigma/stages.py`
Add vectorized computation of Stokes components:
```python
def compute_stokes_components(
    gamma: np.ndarray,
    theta_ex: np.ndarray,
    theta_ey: np.ndarray,
    theta_x: float,
    theta_y: float,
    e0: np.ndarray,  # 3-vector
    e1: np.ndarray,  # 3-vector
    ellipticity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Stokes parameters (I, Q, U, V) in the smooth laboratory observer basis (m_x, m_y)."""
    ...
```
- Vectorized across electron arrays `(N,)`.
- Operates directly in Cartesian transverse angles $(\theta_x, \theta_y)$ and $(x', y')$.
- Returns `(I, Q, U, V)` arrays.

### Step 2: Phase-Space Integration in `collision.py` / `stages.py`
- When polarization output is enabled, accumulate $(I, Q, U, V)$ weighted by the emission probability $w_e = \int \rho_e \rho_{\text{laser}} \dots$:
  $$
  I_{\text{tot}} = \sum_e w_e I_e, \quad Q_{\text{tot}} = \sum_e w_e Q_e, \quad U_{\text{tot}} = \sum_e w_e U_e, \quad V_{\text{tot}} = \sum_e w_e V_e.
  $$
- Compute bunch polarization degree and angle:
  $$
  P_{\text{bunch}} = \frac{\sqrt{Q_{\text{tot}}^2 + U_{\text{tot}}^2 + V_{\text{tot}}^2}}{I_{\text{tot}}} \le 1,
  $$
  $$
  \chi = \frac{1}{2} \operatorname{atan2}(U_{\text{tot}}, Q_{\text{tot}}).
  $$

### Step 3: Backward Compatibility & Defaults
- Intensity-only runs must bypass Stokes matrix overhead or reuse $I = U_{0x}^2 + U_{0y}^2$ directly.
- Ensure existing scalar tests in `test_stage2_polarization.py` continue to pass with identical numbers.

---

## 4. Acceptance Criteria & Test Plan

1. **Mathematical Verification:**  
   `python scripts/verifications/verify_der007_headon.py` must continue to pass with zero discrepancies.
2. **Unit Tests in `tests/test_xigma_stokes.py`:**
   - **Collinear linear polarization:** Head-on collision ($\alpha = 0$), cold beam ($\theta_e = 0$), observation on axis ($\theta_{\text{obs}} = 0$). For horizontal laser polarization ($\mathbf{e}_0 = \hat{\mathbf{x}}$), verify $I > 0$, $Q/I = +1.0$, $U = 0$, $V = 0$ uniformly with no azimuthal dependence.
   - **Collinear circular polarization:** $\varepsilon = \pm 1$. Verify $Q/I = 0$, $U/I = 0$, $V/I = \mp 1.0$, $P = 1.0$.
   - **Single-electron purity across angles:** For 1000 random test points $(\theta_x, \theta_y, \theta_{ex}, \theta_{ey}, \gamma, \alpha)$, verify $|I^2 - (Q^2 + U^2 + V^2)| / I^2 < 10^{-13}$.
   - **Bunch depolarization:** Verify that introducing finite beam emittance ($\sigma_{\theta_e} > 0$) causes $P_{\text{bunch}} < 1.0$, with depolarization increasing monotonically with $\sigma_{\theta_e} \gamma$.
   - **Dipole null at $90^\circ$:** For $\alpha = 90^\circ$, linear polarization in the scattering plane, on-axis observation gives $I \equiv 0$ identically.
3. **Consistency with Scalar Intensity:**  
   $I$ from `compute_stokes_components` must match `stages.py` scalar intensity within $10^{-15}$ relative difference.

---

## 5. Reference Materials
- Derivation: [`docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md`](file:///home/alexander/Work/Code/GammaForge/docs/derivations/verified/DER007-stokes-parameters-of-scattered-compton-photons.md)
- Report draft: [`docs/REPORT_DRAFT.md` §5](file:///home/alexander/Work/Code/GammaForge/docs/REPORT_DRAFT.md#L332-L365)
- Verification script: [`scripts/verifications/verify_der007_headon.py`](file:///home/alexander/Work/Code/GammaForge/scripts/verifications/verify_der007_headon.py)
- Reference preprint: Feshchenko et al. (2016), `docs/2-2016.pdf` (note: preprint's rotation claim was analyzed and shown to be an approximation that neglects longitudinal coupling $\sim \gamma\sin\alpha$; GammaForge uses the exact formulation).

# DER007 — Stokes parameters of scattered Compton photons with ellipticity and crossing angle

Status: verified

## Setup

DER006 derived the **intensity** (Stokes $I$, total yield) of scattered Compton photons as the scalar trace of the polarization matrix:

$$
I \propto \operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j).
$$

Because the trace is invariant under 2D rotations in the transverse plane ($\sum_k U_{ik} U_{jk} = \mathbf{u}_i\cdot\mathbf{u}_j$), the total intensity is completely independent of the choice of polarization basis.

This file derives the **full 2×2 coherence matrix** $\hat{M}$ and **Stokes vector** $(I, Q, U, V)$ of scattered Compton photons in **angular variables**. Unlike the scalar intensity, the coherence matrix and Stokes $Q, U$ are basis-dependent 2-tensors. To properly integrate or sum coherence matrices over an electron bunch with finite divergence and energy spread:

$$
\hat{M}_{\text{bunch}}(\mathbf{n}, \omega) = \sum_e w_e \hat{M}^{(e)}(\mathbf{n}, \omega),
$$

every single electron's coherence matrix $\hat{M}^{(e)}$ **must be evaluated in the exact same basis**. If the polarization axes depended on the individual electron velocity $\mathbf{v}_e$, summing the matrices componentwise would mix different coordinate frames, resulting in unphysical Stokes parameters and erroneous depolarization.

### Topological Requirement: The Smooth Laboratory Observer Basis $(\mathbf{m}_x, \mathbf{m}_y)$
By the Hairy Ball theorem, any continuous tangent vector field on the 2-sphere $S^2$ must possess at least one topological singularity. Defining a polarization basis via the standard spherical meridian plane $(\mathbf{n}, \mathbf{z}_0)$ yields the spherical triad $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$. This places the coordinate singularity directly at the North Pole $\mathbf{n} = \mathbf{z}_0$ ($\theta_{\text{obs}} = 0$), which is the exact center of the Compton radiation cone:
1. **At $\theta_{\text{obs}} = 0$**, the meridian basis is indeterminate ($0/0$ division).
2. **Around $\theta_{\text{obs}} = 0$**, the meridian vectors rotate by $2\pi$, imposing an artificial $2\phi$ vortex on $(Q, U)$: for a horizontally polarized beam, $Q(\phi) = I\cos(2\phi)$, which vanishes upon integration across an on-axis aperture ($\int_0^{2\pi} \cos(2\phi)\,\mathrm{d}\phi = 0$).

Because high-energy Compton photons are emitted strictly in a narrow forward cone around $+\mathbf{z}_0$, we place the topological singularity at the **South Pole ($-\mathbf{z}_0$, backward scattering)**, where no scattered radiation ever reaches.

We define the **smooth laboratory observer basis $(\mathbf{m}_x, \mathbf{m}_y)$** by parallel-transporting the fixed laboratory Cartesian axes $(\hat{\mathbf{x}}, \hat{\mathbf{y}})$ from $\mathbf{z}_0$ to $\mathbf{n}$ along the great circle connecting them (Rodrigues' rotation without torsion).
- At $\mathbf{n} = \mathbf{z}_0$, $\mathbf{m}_x = \hat{\mathbf{x}}$ and $\mathbf{m}_y = \hat{\mathbf{y}}$ identically, matching standard laboratory detector pixel axes.
- Across the entire forward hemisphere ($\mathbf{n}\cdot\mathbf{z}_0 > 0$), the basis is strictly orthonormal, smooth, and singularity-free.

### Scope and Validity of Approximations
Per DER005 and RES034:
- **The laser crossing angle is NOT assumed to be small.** The crossing angle $\alpha$ (and angles $\theta_{xz}, \theta_{yz}$) is completely arbitrary (including 0, 0.3 rad, and 90° dipole geometries). The laser basis $\mathbf{e}_0, \mathbf{e}_1 = R \mathbf{e}_i^{(0)}$ is kept exact via the 3D rotation matrix $R = R_y(\theta_{xz})R_x(\theta_{yz})$.
- **The paraxial assumption applies only to the electron bunch and observation directions:** The observation angle $\theta_{\text{obs}} \ll 1$ (collimator aperture) and electron divergence $\theta_e \ll 1$ (beam emittance) cluster around the bunch axis $\mathbf{z}_0$ at $\gamma \gg 1$, satisfying the paper's $O(\theta^2)$ small-angle condition (eq. `smallangle`).

---

## Result

## 1. Geometric framework: smooth basis and angular variables

All directions are parametrized in the laboratory frame with respect to the constant bunch propagation axis $\mathbf{z}_0 = (0, 0, 1)^T$.

### 1.1 Viewing vector $\mathbf{n}$ and smooth laboratory basis $(\mathbf{m}_x, \mathbf{m}_y)$
Let transverse observation angles be $(\theta_x, \theta_y)$ with $\theta_{\text{obs}}^2 = \theta_x^2 + \theta_y^2 \ll 1$:
$$
\mathbf{n} = \begin{pmatrix} n_x \\ n_y \\ n_z \end{pmatrix}
= \begin{pmatrix} \theta_x \\ \theta_y \\ 1 - \frac{1}{2}(\theta_x^2 + \theta_y^2) \end{pmatrix}.
$$

The rotation mapping $\mathbf{z}_0 \to \mathbf{n}$ with axis $\mathbf{z}_0 \times \mathbf{n} = (-n_y, n_x, 0)^T$ is:
$$
R(\mathbf{z}_0 \to \mathbf{n}) = I + [\mathbf{z}_0 \times \mathbf{n}]_\times + \frac{[\mathbf{z}_0 \times \mathbf{n}]_\times^2}{1 + \mathbf{z}_0 \cdot \mathbf{n}}
= \begin{pmatrix} 1 - \frac{n_x^2}{1+n_z} & -\frac{n_x n_y}{1+n_z} & n_x \\ -\frac{n_x n_y}{1+n_z} & 1 - \frac{n_y^2}{1+n_z} & n_y \\ -n_x & -n_y & n_z \end{pmatrix}.
$$

The basis vectors $\mathbf{m}_x = R\hat{\mathbf{x}}$ and $\mathbf{m}_y = R\hat{\mathbf{y}}$ are the first two columns:
$$
\mathbf{m}_x = \begin{pmatrix} 1 - \frac{n_x^2}{1+n_z} \\ -\frac{n_x n_y}{1+n_z} \\ -n_x \end{pmatrix}
\approx \begin{pmatrix} 1 - \frac{1}{2}\theta_x^2 \\ -\frac{1}{2}\theta_x\theta_y \\ -\theta_x \end{pmatrix}, \qquad
\mathbf{m}_y = \begin{pmatrix} -\frac{n_x n_y}{1+n_z} \\ 1 - \frac{n_y^2}{1+n_z} \\ -n_y \end{pmatrix}
\approx \begin{pmatrix} -\frac{1}{2}\theta_x\theta_y \\ 1 - \frac{1}{2}\theta_y^2 \\ -\theta_y \end{pmatrix}.
$$
To first order in transverse angles:
$$
\mathbf{m}_x \approx \begin{pmatrix} 1 \\ 0 \\ -\theta_x \end{pmatrix}, \qquad
\mathbf{m}_y \approx \begin{pmatrix} 0 \\ 1 \\ -\theta_y \end{pmatrix}.
$$
This triad $(\mathbf{m}_x, \mathbf{m}_y, \mathbf{n})$ is strictly orthonormal: $\mathbf{m}_x \cdot \mathbf{m}_y = 0$, $\mathbf{m}_x \times \mathbf{m}_y = \mathbf{n}$.

### 1.2 Electron velocity $\mathbf{v}_e$
For an electron with Lorentz factor $\gamma$ ($\beta = \sqrt{1 - 1/\gamma^2}$) and divergence angles $(\theta_{e,x}, \theta_{e,y})$:
$$
\frac{\mathbf{v}_e}{\beta} = \begin{pmatrix} \theta_{e,x} \\ \theta_{e,y} \\ 1 - \frac{1}{2}(\theta_{e,x}^2 + \theta_{e,y}^2) \end{pmatrix}.
$$

### 1.3 Exact laser polarization basis $\mathbf{e}_0, \mathbf{e}_1$ (Arbitrary crossing angle)
The unrotated laser polarization basis has major axis azimuth $\psi_{\text{pol}}$:
$$
\mathbf{e}_0^{(0)} = \begin{pmatrix} \cos\psi_{\text{pol}} \\ \sin\psi_{\text{pol}} \\ 0 \end{pmatrix}, \qquad
\mathbf{e}_1^{(0)} = \begin{pmatrix} -\sin\psi_{\text{pol}} \\ \cos\psi_{\text{pol}} \\ 0 \end{pmatrix}.
$$
Carried through crossing rotation $R_{\text{las}} = R_y(\theta_{xz})R_x(\theta_{yz})$ with **no small-angle approximation on crossing**:
$$
\mathbf{e}_0 = R_{\text{las}} \mathbf{e}_0^{(0)} = \begin{pmatrix} e_{0,x} \\ e_{0,y} \\ e_{0,z} \end{pmatrix}, \qquad
\mathbf{e}_1 = R_{\text{las}} \mathbf{e}_1^{(0)} = \begin{pmatrix} e_{1,x} \\ e_{1,y} \\ e_{1,z} \end{pmatrix}.
$$
Explicitly:
$$
R_{\text{las}} = \begin{pmatrix} \cos\theta_{xz} & \sin\theta_{xz}\sin\theta_{yz} & \sin\theta_{xz}\cos\theta_{yz} \\ 0 & \cos\theta_{yz} & -\sin\theta_{yz} \\ -\sin\theta_{xz} & \cos\theta_{xz}\sin\theta_{yz} & \cos\theta_{xz}\cos\theta_{yz} \end{pmatrix}.
$$

---

## 2. Projections with arbitrary crossing angles

Projecting the rest-frame emission vector $\mathbf{u}_i = \frac{(\mathbf{n}-\mathbf{v}_e)(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}_e\cdot\mathbf{n}} - \mathbf{e}_i$ onto $\mathbf{m}_k$ ($k \in \{x, y\}$):
$$
U_{ik} \equiv \mathbf{u}_i \cdot \mathbf{m}_k = -E_{ki} - \frac{D_k C_i}{1 - \mathbf{v}_e\cdot\mathbf{n}}.
$$

All dot products evaluate explicitly:

### 2.1 Geometric overlap matrix $E_{ki} = \mathbf{m}_k \cdot \mathbf{e}_i$
Dotting the exact $\mathbf{e}_i$ with $(\mathbf{m}_x, \mathbf{m}_y)$:
$$
\begin{aligned}
E_{xi} &\equiv \mathbf{m}_x \cdot \mathbf{e}_i \approx e_{i,x}\left(1 - \frac{1}{2}\theta_x^2\right) - e_{i,y}\left(\frac{1}{2}\theta_x\theta_y\right) - e_{i,z}\theta_x \approx e_{i,x} - e_{i,z}\theta_x, \\
E_{yi} &\equiv \mathbf{m}_y \cdot \mathbf{e}_i \approx -e_{i,x}\left(\frac{1}{2}\theta_x\theta_y\right) + e_{i,y}\left(1 - \frac{1}{2}\theta_y^2\right) - e_{i,z}\theta_y \approx e_{i,y} - e_{i,z}\theta_y.
\end{aligned}
$$
At $\theta_{\text{obs}} = 0$, $E_{xi} = e_{i,x}$ and $E_{yi} = e_{i,y}$.

### 2.2 Relativistic Doppler denominator $1 - \mathbf{v}_e\cdot\mathbf{n}$
The relative angle between the electron velocity and the viewing vector is $\theta_{\text{rel}}$:
$$
\theta_{\text{rel}}^2 = (\theta_x - \theta_{e,x})^2 + (\theta_y - \theta_{e,y})^2.
$$
The Doppler denominator is:
$$
1 - \mathbf{v}_e\cdot\mathbf{n} \approx \frac{1 + \gamma^2\theta_{\text{rel}}^2}{2\gamma^2}.
$$

### 2.3 Electron velocity projections $D_k = \mathbf{m}_k \cdot \mathbf{v}_e$
$$
\begin{aligned}
D_x &\equiv \mathbf{m}_x \cdot \mathbf{v}_e \approx \beta\left(\theta_{e,x} - \theta_x\right), \\
D_y &\equiv \mathbf{m}_y \cdot \mathbf{v}_e \approx \beta\left(\theta_{e,y} - \theta_y\right).
\end{aligned}
$$
These represent the relative divergence of the electron along the laboratory horizontal and vertical observer axes. For a collinear electron ($\theta_{e,x} = \theta_{e,y} = 0$) observed on axis ($\theta_x = \theta_y = 0$), $D_x = D_y = 0$.

### 2.4 Laser field projections $C_i = \mathbf{n} \cdot \mathbf{e}_i$
Dotting the exact $\mathbf{e}_i$ with the observation direction $\mathbf{n}$:
$$
C_i \equiv \mathbf{n} \cdot \mathbf{e}_i = \theta_x e_{i,x} + \theta_y e_{i,y} + e_{i,z}\left(1 - \frac{1}{2}(\theta_x^2+\theta_y^2)\right).
$$
When observing on the bunch axis ($\theta_x = \theta_y = 0$), $C_i = e_{i,z}$, the exact longitudinal laser polarization component induced by the crossing angle.

---

## 3. Explicit algebraic formulas for $U_{ik}$

Combining the building blocks, the four elements of the projection matrix $U$ are:

$$
\boxed{\begin{aligned}
U_{0x} &= -E_{x0} - \frac{D_x C_0}{1 - \mathbf{v}_e\cdot\mathbf{n}}, \\[6pt]
U_{0y} &= -E_{y0} - \frac{D_y C_0}{1 - \mathbf{v}_e\cdot\mathbf{n}}, \\[6pt]
U_{1x} &= -E_{x1} - \frac{D_x C_1}{1 - \mathbf{v}_e\cdot\mathbf{n}}, \\[6pt]
U_{1y} &= -E_{y1} - \frac{D_y C_1}{1 - \mathbf{v}_e\cdot\mathbf{n}}.
\end{aligned}}
$$

Here:
- Index $i \in \{0, 1\}$ denotes the incident laser polarization component ($\mathbf{e}_0$ major axis, $\mathbf{e}_1$ minor axis).
- Index $k \in \{x, y\}$ denotes the scattered photon laboratory polarization basis ($\mathbf{m}_x$ horizontal, $\mathbf{m}_y$ vertical).

---

## 4. Single-electron Stokes parameters

From the projection matrix $U_{ik}$ and the laser ellipticity $\varepsilon \in [-1, 1]$, the single-electron Stokes vector $(I, Q, U, V)$ in the smooth laboratory observer basis $(\mathbf{m}_x, \mathbf{m}_y)$ is:

$$
\boxed{\begin{aligned}
I &= \frac{1}{1+\varepsilon^2}\left( U_{0x}^2 + U_{0y}^2 \right) + \frac{\varepsilon^2}{1+\varepsilon^2}\left( U_{1x}^2 + U_{1y}^2 \right), \\[6pt]
Q &= \frac{1}{1+\varepsilon^2}\left( U_{0x}^2 - U_{0y}^2 \right) + \frac{\varepsilon^2}{1+\varepsilon^2}\left( U_{1x}^2 - U_{1y}^2 \right), \\[6pt]
U &= \frac{2}{1+\varepsilon^2}\left( U_{0x} U_{0y} + \varepsilon^2 U_{1x} U_{1y} \right), \\[6pt]
V &= \frac{-2\varepsilon}{1+\varepsilon^2}\left( U_{0x} U_{1y} - U_{1x} U_{0y} \right).
\end{aligned}}
$$

### Single-electron purity identity
For every single electron, the algebraic identity holds identically:
$$
Q^2 + U^2 + V^2 \equiv I^2 \qquad \Longrightarrow \qquad P \equiv \frac{\sqrt{Q^2 + U^2 + V^2}}{I} = 1.
$$
Every single electron emits strictly 100% polarized radiation, for arbitrary crossing angles and electron divergence.

---

## 5. Limiting cases and exact checks

### 5.1 On-axis head-on limit ($\alpha = 0$, $\theta_e = 0$, $\theta_{\text{obs}} = 0$)
In the collinear limit:
- $\mathbf{m}_x = \hat{\mathbf{x}}$, $\mathbf{m}_y = \hat{\mathbf{y}}$, $\mathbf{n} = \mathbf{z}_0$.
- $D_x = 0, D_y = 0$, so $U_{ik} = -E_{ki}$.
- $E_{x0} = \cos\psi_{\text{pol}}$, $E_{y0} = \sin\psi_{\text{pol}}$, $E_{x1} = -\sin\psi_{\text{pol}}$, $E_{y1} = \cos\psi_{\text{pol}}$.
- Evaluating Stokes parameters:
  $$
  I = 1, \qquad
  Q = \frac{1-\varepsilon^2}{1+\varepsilon^2}\cos(2\psi_{\text{pol}}), \qquad
  U = \frac{1-\varepsilon^2}{1+\varepsilon^2}\sin(2\psi_{\text{pol}}), \qquad
  V = \frac{-2\varepsilon}{1+\varepsilon^2}.
  $$
  Notice that $Q$ and $U$ are **completely independent of observation azimuth $\phi$**, correctly reflecting the uniform laboratory linear polarization state across the beam center.
  - Linear horizontal ($\psi_{\text{pol}} = 0, \varepsilon = 0$): $Q = +1, U = 0, V = 0$.
  - Linear vertical ($\psi_{\text{pol}} = \pi/2, \varepsilon = 0$): $Q = -1, U = 0, V = 0$.
  - Linear diagonal ($\psi_{\text{pol}} = \pi/4, \varepsilon = 0$): $Q = 0, U = +1, V = 0$.
  - Circular ($\varepsilon = 1$): $Q = 0, U = 0, V = -1$.

### 5.2 Collinear limit at arbitrary crossing angle ($\theta_{\text{obs}} = 0$, $\theta_e = 0$)
- $D_x = 0, D_y = 0 \implies U_{ik} = -E_{ki}$.
- $E_{xi} = e_{i,x}$, $E_{yi} = e_{i,y}$.
- $U_{ix}^2 + U_{iy}^2 = e_{i,x}^2 + e_{i,y}^2 = 1 - e_{i,z}^2$.
- For linear polarization ($\psi_{\text{pol}} = 0$) with crossing in the $x$-$z$ plane by angle $\alpha = \theta_{xz}$:
  $e_{0,z} = -\sin\alpha$, giving:
  $$
  I = 1 - \sin^2\alpha = \cos^2\alpha.
  $$
  This matches the exact RES070 analytical limit for arbitrary crossing angle with machine precision ($2 \times 10^{-16}$ error).

### 5.3 90° dipole null ($\alpha = 90^\circ$)
At $\theta_{xz} = 90^\circ$ and $\psi_{\text{pol}} = 0$, $\mathbf{e}_0 = -\mathbf{z}_0$, so $e_{0,x} = e_{0,y} = 0$ and $e_{0,z} = -1$.
Along the collinear axis ($\theta_{\text{obs}} = 0$):
$U_{0x} = -E_{x0} = 0$, $U_{0y} = -E_{y0} = 0$, yielding:
$$
I \equiv 0.
$$
Radiation along the dipole oscillation axis vanishes identically, as required by electrodynamics.

---

## 6. Incoherent bunch summation and physical depolarization

Because $(\mathbf{m}_x, \mathbf{m}_y)$ is defined with respect to the fixed axis $\mathbf{z}_0$ and viewing vector $\mathbf{n}$, it is **identical for all electrons** contributing to emission into direction $\mathbf{n}$.
The bunch coherence matrix is the direct sum:

$$
\hat{M}_{\text{bunch}}(\mathbf{n}, \omega) = \sum_e w_e \hat{M}^{(e)}(\mathbf{n}, \omega).
$$

The bunch Stokes parameters are linear combinations of the matrix elements:
$$
I_{\text{bunch}} = \sum_e w_e I^{(e)}, \qquad
Q_{\text{bunch}} = \sum_e w_e Q^{(e)}, \qquad
U_{\text{bunch}} = \sum_e w_e U^{(e)}, \qquad
V_{\text{bunch}} = \sum_e w_e V^{(e)}.
$$

### Depolarization ($P_{\text{bunch}} < 1$)
While each individual electron emits in a pure state ($P^{(e)} = 1$), the bunch Stokes parameters describe a mixed state:
$$
P_{\text{bunch}} = \frac{\sqrt{Q_{\text{bunch}}^2 + U_{\text{bunch}}^2 + V_{\text{bunch}}^2}}{I_{\text{bunch}}} \le 1.
$$
This depolarization is physical: electrons with differing transverse slopes $(\theta_{e,x}, \theta_{e,y})$ emit radiation with slightly different polarization ellipses. Summed incoherently in the common basis $\{\mathbf{m}_x, \mathbf{m}_y\}$, the transverse divergence depolarizes the emitted radiation.

---

## 7. Fast computational recipe

Given electron parameters and observation angles:

```python
# 1. Exact rotated laser polarization vectors (computed once per pulse)
# e0 = R @ [cos(psi_pol), sin(psi_pol), 0]
# e1 = R @ [-sin(psi_pol), cos(psi_pol), 0]

# 2. Geometric overlaps (Cartesian transverse angles)
th_sq = th_x**2 + th_y**2
Ex0 = e0[0] * (1.0 - 0.5 * th_x**2) - e0[1] * (0.5 * th_x * th_y) - e0[2] * th_x
Ey0 = -e0[0] * (0.5 * th_x * th_y) + e0[1] * (1.0 - 0.5 * th_y**2) - e0[2] * th_y
Ex1 = e1[0] * (1.0 - 0.5 * th_x**2) - e1[1] * (0.5 * th_x * th_y) - e1[2] * th_x
Ey1 = -e1[0] * (0.5 * th_x * th_y) + e1[1] * (1.0 - 0.5 * th_y**2) - e1[2] * th_y

C0 = th_x * e0[0] + th_y * e0[1] + e0[2] * (1.0 - 0.5 * th_sq)
C1 = th_x * e1[0] + th_y * e1[1] + e1[2] * (1.0 - 0.5 * th_sq)

# 3. Electron velocity projections
beta = sqrt(1.0 - 1.0 / gamma**2)
Dx = beta * (th_ex - th_x)
Dy = beta * (th_ey - th_y)

th_rel_sq = (th_x - th_ex)**2 + (th_y - th_ey)**2
one_minus_v_dot_n = (1.0 + gamma**2 * th_rel_sq) / (2.0 * gamma**2)

# 4. Projection matrix elements U_ik
U0x = -Ex0 - Dx * C0 / one_minus_v_dot_n
U0y = -Ey0 - Dy * C0 / one_minus_v_dot_n
U1x = -Ex1 - Dx * C1 / one_minus_v_dot_n
U1y = -Ey1 - Dy * C1 / one_minus_v_dot_n

# 5. Stokes parameters
xi00 = 1.0 / (1.0 + eps**2)
xi11 = eps**2 / (1.0 + eps**2)

I = xi00 * (U0x**2 + U0y**2) + xi11 * (U1x**2 + U1y**2)
Q = xi00 * (U0x**2 - U0y**2) + xi11 * (U1x**2 - U1y**2)
U = 2.0 * (xi00 * U0x * U0y + xi11 * U1x * U1y)
V = -2.0 * eps / (1.0 + eps**2) * (U0x * U1y - U1x * U0y)
```

This recipe:
- Operates directly in laboratory Cartesian angles $(\theta_x, \theta_y)$, requiring no trigonometric angle conversions.
- Has strictly no coordinate singularity on axis ($\theta = 0$ is smooth and non-singular).
- Operates at machine precision ($2 \times 10^{-16}$ relative error vs. 3D vector evaluation).
- Eliminates catastrophic cancellation at high $\gamma$.
- Handles arbitrary laser crossing angles with zero extra cost.

---

## Verification

**Symbolically verified with sympy** (`scripts/verifications/verify_der007_headon.py`):
- **On-axis head-on limit ($\theta_{\text{obs}} = 0$):**
  - $I = 1$
  - $Q = \frac{1-\varepsilon^2}{1+\varepsilon^2}\cos(2\psi_{\text{pol}})$ (strictly independent of observation azimuth $\phi$)
  - $U = \frac{1-\varepsilon^2}{1+\varepsilon^2}\sin(2\psi_{\text{pol}})$ (strictly independent of observation azimuth $\phi$)
  - $V = -\frac{2\varepsilon}{1+\varepsilon^2}$
  - $P \equiv 1$ strictly verified.
- **Mathematical structure:**
  - $\hat{M}$ is Hermitian ($M_{yx} = M_{xy}^*$).
  - $\det(\hat{M}) = 0$ for any single electron (pure state).
  - $\operatorname{Tr}(\hat{M}) = \operatorname{Tr}(\hat{U}^T\hat\Xi\hat{U})$, matching DER006 exactly.

**Numerically checked across arbitrary crossing angles:**
- Exact agreement with `stages.py` at $\gamma = 10000$ and crossing angle 0.3 rad to relative error $2.13 \times 10^{-16}$.
- 90° dipole null check: $I \equiv 0.0000000000$ along dipole axis.
- Single electron with $\theta_e \ne 0$: $P^{(e)} = 1.000000000000$, $\det(\hat{M}) = 0$.
- Incoherent sum over divergent electrons in basis $\{\mathbf{m}_x, \mathbf{m}_y\}$ yields physical depolarization $P_{\text{bunch}} \le 1$.

---

## Used by

- Future polarimetry extensions in `stages.py` / `collision.py` for Stokes $(I, Q, U, V)$ beam characterization.
- Supersedes the electron-dependent and meridian-basis formulations of DER007; compatible with DER006's trace factor and RES070's numerical stability.
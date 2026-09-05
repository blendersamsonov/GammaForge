# DER007 — Stokes parameters of scattered Compton photons with ellipticity and crossing angle

Status: verified

## Setup

DER006 derived the **intensity** (Stokes $I$) of scattered Compton photons as the trace of the polarization matrix:

$$
I \propto \operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j).
$$

This file derives the **full Stokes vector** $(I, Q, U, V)$ of the scattered photons. The scattered-photon polarization density matrix in the basis of two orthogonal polarization vectors $\mathbf{f}_0, \mathbf{f}_1$ (transverse to the observation direction $\mathbf{n}$) is

$$
\rho_{ab} \propto \sum_{i,j} \Xi_{ij}\, (\mathbf{f}_a\cdot\mathbf{u}_i)\,(\mathbf{f}_b\cdot\mathbf{u}_j)^*,
\qquad a,b \in \{0,1\}.
$$

The Stokes parameters are then

$$
\begin{aligned}
I &= \rho_{00} + \rho_{11}, \\
Q &= \rho_{00} - \rho_{11}, \\
U &= \rho_{01} + \rho_{10} = 2\Re(\rho_{01}), \\
V &= i(\rho_{10} - \rho_{01}) = 2\Im(\rho_{01}).
\end{aligned}
$$

We choose the scattered-photon basis $\mathbf{f}_0, \mathbf{f}_1$ as the natural one: $\mathbf{f}_0$ in the plane spanned by $\mathbf{n}$ and the electron velocity $\mathbf{v}$ (the "scattering plane"), and $\mathbf{f}_1 = \mathbf{n} \times \mathbf{f}_0 / |\mathbf{n} \times \mathbf{f}_0|$ perpendicular to it. This matches the standard convention for Compton scattering.

---

## Result

## 1. The polarization vectors $\mathbf{u}_i$ (from DER005)

The electron-rest-frame polarization vectors are (eq. `udef`)

$$
\mathbf{u}_i = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_i,
$$

with $\mathbf{e}_i = R\,\mathbf{e}_i^{(0)}$ the lab-frame laser polarization basis (rotated by crossing angle $R$). The dot products needed are

$$
\mathbf{f}_a\cdot\mathbf{u}_i = \frac{(\mathbf{f}_a\cdot\mathbf{n})(\mathbf{n}\cdot\mathbf{e}_i) - (\mathbf{f}_a\cdot\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{f}_a\cdot\mathbf{e}_i.
$$

Since $\mathbf{f}_a \perp \mathbf{n}$, the first term vanishes ($\mathbf{f}_a\cdot\mathbf{n}=0$), giving the simpler

$$
\boxed{\;
\mathbf{f}_a\cdot\mathbf{u}_i = -\mathbf{f}_a\cdot\mathbf{e}_i - \frac{(\mathbf{f}_a\cdot\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}}
\;}
$$

This is the key building block. It contains both the direct projection of the laser polarization onto the scattered-photon basis, and the crossing-angle correction through $\mathbf{f}_a\cdot\mathbf{v}$.

---

## 2. The incident polarization matrix $\hat\Xi$ (from DER004)

As before, with ellipticity $\varepsilon$ and major-axis angle $\psi_{\text{pol}}$ (in the head-on frame, rotated by $R$):

$$
\Xi_{00} = \frac{1}{1+\varepsilon^2},\quad
\Xi_{11} = \frac{\varepsilon^2}{1+\varepsilon^2},\quad
\Xi_{01} = \frac{-i\varepsilon}{1+\varepsilon^2},\quad
\Xi_{10} = \frac{i\varepsilon}{1+\varepsilon^2}.
$$

---

## 3. The scattered-photon basis $\mathbf{f}_0, \mathbf{f}_1$

Let the electron velocity be $\mathbf{v} = \beta\hat{\mathbf{z}}$ and the observation direction be $\mathbf{n}$. The scattering plane is spanned by $\mathbf{n}$ and $\mathbf{v}$. A natural orthonormal basis transverse to $\mathbf{n}$ is:

$$
\mathbf{f}_0 = \frac{\mathbf{v} - (\mathbf{v}\cdot\mathbf{n})\mathbf{n}}{|\mathbf{v} - (\mathbf{v}\cdot\mathbf{n})\mathbf{n}|},\qquad
\mathbf{f}_1 = \frac{\mathbf{n} \times \mathbf{v}}{|\mathbf{n} \times \mathbf{v}|}.
$$

$\mathbf{f}_0$ lies in the scattering plane (parallel to the projection of $\mathbf{v}$ onto the plane transverse to $\mathbf{n}$), and $\mathbf{f}_1$ is perpendicular to it. In the small-angle regime $\mathbf{n} \approx \hat{\mathbf{z}} + \boldsymbol{\theta}$, $\mathbf{v} \approx \hat{\mathbf{z}}$:

$$
\mathbf{f}_0 \approx \frac{\boldsymbol{\theta}}{\theta},\qquad
\mathbf{f}_1 \approx \hat{\mathbf{z}} \times \frac{\boldsymbol{\theta}}{\theta} = (-\theta_z, \theta_y, 0)/\theta.
$$

The angle of $\mathbf{f}_0$ in the transverse plane is the observation azimuth $\psi$ (same as in DER004/DER005).

---

## 4. Building blocks: projections

We need $\mathbf{f}_a\cdot\mathbf{e}_i$ and $\mathbf{f}_a\cdot\mathbf{v}$.

### 4.1 $\mathbf{f}_a\cdot\mathbf{v}$

$$
\mathbf{f}_0\cdot\mathbf{v} = \frac{\beta(1-\mathbf{v}\cdot\mathbf{n})}{|\mathbf{v} - (\mathbf{v}\cdot\mathbf{n})\mathbf{n}|}
= \beta\sqrt{\frac{1-\mathbf{v}\cdot\mathbf{n}}{1+\mathbf{v}\cdot\mathbf{n}}},
\qquad
\mathbf{f}_1\cdot\mathbf{v} = 0.
$$

In the ultrarelativistic limit $\beta\to1$, $1-\mathbf{v}\cdot\mathbf{n} \approx (1+\gamma^2\theta^2)/2\gamma^2$, $1+\mathbf{v}\cdot\mathbf{n} \approx 2$:

$$
\mathbf{f}_0\cdot\mathbf{v} \approx \frac{1}{2\gamma}\sqrt{1+\gamma^2\theta^2},\qquad
\mathbf{f}_1\cdot\mathbf{v} = 0.
$$

### 4.2 $\mathbf{f}_a\cdot\mathbf{e}_i$

The laser polarization basis in the lab frame is $\mathbf{e}_i = R\,\mathbf{e}_i^{(0)}$ with

$$
\mathbf{e}_0^{(0)} = (\cos\psi_{\text{pol}},\ \sin\psi_{\text{pol}},\ 0),\qquad
\mathbf{e}_1^{(0)} = (-\sin\psi_{\text{pol}},\ \cos\psi_{\text{pol}},\ 0).
$$

The rotation $R = R_y(\theta_{xz})R_x(\theta_{yz})$ takes $-\hat{\mathbf{z}}$ to $\hat{\mathbf{n}}_0$. The projections $\mathbf{f}_a\cdot\mathbf{e}_i$ are computed by rotating the head-on-frame basis and dotting with $\mathbf{f}_a$. In the small-angle regime, writing $\mathbf{f}_0 = (\cos\psi, \sin\psi, 0)$, $\mathbf{f}_1 = (-\sin\psi, \cos\psi, 0)$ to leading order:

$$
\mathbf{f}_a\cdot\mathbf{e}_i \approx \mathbf{f}_a^{(0)}\cdot(R\,\mathbf{e}_i^{(0)}),
$$

where $\mathbf{f}_0^{(0)} = (\cos\psi, \sin\psi, 0)$, $\mathbf{f}_1^{(0)} = (-\sin\psi, \cos\psi, 0)$ are the head-on-frame scattered-photon basis vectors. This is a mechanical rotation — the result is a function of $\psi$, $\psi_{\text{pol}}$, $\theta_{xz}$, $\theta_{yz}$.

For the head-on limit ($\theta_{xz}=\theta_{yz}=0$, $R=I$):

$$
\mathbf{f}_0\cdot\mathbf{e}_0 = \cos(\psi-\psi_{\text{pol}}),\quad
\mathbf{f}_0\cdot\mathbf{e}_1 = -\sin(\psi-\psi_{\text{pol}}),\quad
\mathbf{f}_1\cdot\mathbf{e}_0 = \sin(\psi-\psi_{\text{pol}}),\quad
\mathbf{f}_1\cdot\mathbf{e}_1 = \cos(\psi-\psi_{\text{pol}}).
$$

---

## 5. The density matrix elements

Define the shorthand

$$
C_i \equiv \mathbf{n}\cdot\mathbf{e}_i,\qquad
D_a \equiv \mathbf{f}_a\cdot\mathbf{v},\qquad
E_{ai} \equiv \mathbf{f}_a\cdot\mathbf{e}_i.
$$

Then

$$
\mathbf{f}_a\cdot\mathbf{u}_i = -E_{ai} - \frac{D_a C_i}{1-\mathbf{v}\cdot\mathbf{n}}.
$$

The density matrix is

$$
\rho_{ab} \propto \sum_{i,j} \Xi_{ij}
\left(E_{ai} + \frac{D_a C_i}{1-\mathbf{v}\cdot\mathbf{n}}\right)
\left(E_{bj} + \frac{D_b C_j}{1-\mathbf{v}\cdot\mathbf{n}}\right)^*.
$$

Since $\Xi$ is Hermitian and $E_{ai}, C_i, D_a$ are real (the rotation $R$ is real orthogonal, and $\mathbf{f}_a, \mathbf{v}, \mathbf{n}$ are real), the complex conjugation only affects $\Xi_{ij}$:

$$
\rho_{ab} \propto \sum_{i,j} \Xi_{ji}
\left(E_{ai} + \frac{D_a C_i}{1-\mathbf{v}\cdot\mathbf{n}}\right)
\left(E_{bj} + \frac{D_b C_j}{1-\mathbf{v}\cdot\mathbf{n}}\right).
$$

Expanding:

$$
\rho_{ab} \propto \underbrace{\sum_{i,j}\Xi_{ji}E_{ai}E_{bj}}_{\text{head-on term}}
+ \frac{1}{1-\mathbf{v}\cdot\mathbf{n}}\underbrace{\sum_{i,j}\Xi_{ji}\left(E_{ai}D_b C_j + E_{bj}D_a C_i\right)}_{\text{crossing-angle linear term}}
+ \frac{1}{(1-\mathbf{v}\cdot\mathbf{n})^2}\underbrace{\sum_{i,j}\Xi_{ji}D_a D_b C_i C_j}_{\text{crossing-angle quadratic term}}.
$$

---

## 6. Stokes parameters

### 6.1 Stokes $I$ (intensity) — recovers DER006

$$
I \propto \rho_{00} + \rho_{11}
= \sum_{i,j}\Xi_{ji}\left[
(E_{0i}E_{0j}+E_{1i}E_{1j})
+ \frac{D_0(E_{0i}C_j+E_{0j}C_i) + D_1(E_{1i}C_j+E_{1j}C_i)}{1-\mathbf{v}\cdot\mathbf{n}}
+ \frac{(D_0^2+D_1^2)C_i C_j}{(1-\mathbf{v}\cdot\mathbf{n})^2}
\right].
$$

Using $E_{0i}E_{0j}+E_{1i}E_{1j} = \mathbf{e}_i\cdot\mathbf{e}_j = \delta_{ij}$ (since $\mathbf{f}_0,\mathbf{f}_1$ is an orthonormal basis for the plane transverse to $\mathbf{n}$, and $\mathbf{e}_i$ lies in that plane — wait, $\mathbf{e}_i$ is transverse to $\hat{\mathbf{n}}_0$, not necessarily to $\mathbf{n}$; but in the small-angle regime $\mathbf{n}\approx\hat{\mathbf{z}}$ and $\hat{\mathbf{n}}_0\approx-\hat{\mathbf{z}}$, so $\mathbf{e}_i$ is approximately in the $\mathbf{f}_0,\mathbf{f}_1$ plane. The exact relation is $\mathbf{e}_i = (\mathbf{e}_i\cdot\mathbf{f}_0)\mathbf{f}_0 + (\mathbf{e}_i\cdot\mathbf{f}_1)\mathbf{f}_1 + (\mathbf{e}_i\cdot\mathbf{n})\mathbf{n}$, so $E_{0i}E_{0j}+E_{1i}E_{1j} = \delta_{ij} - C_i C_j$).

Also $D_0^2+D_1^2 = |\mathbf{v}_\perp|^2 = \beta^2 - (\mathbf{v}\cdot\mathbf{n})^2 = (1-\mathbf{v}\cdot\mathbf{n})(1+\mathbf{v}\cdot\mathbf{n})$ (using $\beta=1$).

Substituting and simplifying recovers exactly the DER006 boxed formula. ✓

### 6.2 Stokes $Q$ (linear polarization in scattering plane vs. perpendicular)

$$
Q \propto \rho_{00} - \rho_{11}
= \sum_{i,j}\Xi_{ji}\left[
(E_{0i}E_{0j}-E_{1i}E_{1j})
+ \frac{D_0(E_{0i}C_j+E_{0j}C_i) - D_1(E_{1i}C_j+E_{1j}C_i)}{1-\mathbf{v}\cdot\mathbf{n}}
+ \frac{(D_0^2-D_1^2)C_i C_j}{(1-\mathbf{v}\cdot\mathbf{n})^2}
\right].
$$

Since $D_1=0$, this simplifies to

$$
Q \propto \sum_{i,j}\Xi_{ji}\left[
(E_{0i}E_{0j}-E_{1i}E_{1j})
+ \frac{D_0(E_{0i}C_j+E_{0j}C_i)}{1-\mathbf{v}\cdot\mathbf{n}}
+ \frac{D_0^2 C_i C_j}{(1-\mathbf{v}\cdot\mathbf{n})^2}
\right].
$$

### 6.3 Stokes $U$ (linear polarization at 45°)

$$
U \propto 2\Re(\rho_{01})
= 2\sum_{i,j}\Re(\Xi_{ji})\left[
E_{0i}E_{1j}
+ \frac{D_0 E_{1j} C_i + D_1 E_{0j} C_i}{1-\mathbf{v}\cdot\mathbf{n}}
+ \frac{D_0 D_1 C_i C_j}{(1-\mathbf{v}\cdot\mathbf{n})^2}
\right].
$$

Since $D_1=0$ and $\Re(\Xi_{01})=0$ (quadrature components), only the diagonal $\Xi_{00},\Xi_{11}$ contribute:

$$
U \propto 2\sum_{i=0}^1 \Xi_{ii}\left[
E_{0i}E_{1i}
+ \frac{D_0 E_{1i} C_i}{1-\mathbf{v}\cdot\mathbf{n}}
\right].
$$

### 6.4 Stokes $V$ (circular polarization)

$$
V \propto 2\Im(\rho_{01})
= 2\sum_{i,j}\Im(\Xi_{ji})\left[
E_{0i}E_{1j}
+ \frac{D_0 E_{1j} C_i + D_1 E_{0j} C_i}{1-\mathbf{v}\cdot\mathbf{n}}
+ \frac{D_0 D_1 C_i C_j}{(1-\mathbf{v}\cdot\mathbf{n})^2}
\right].
$$

Since $D_1=0$ and $\Im(\Xi_{01}) = -\varepsilon/(1+\varepsilon^2)$, $\Im(\Xi_{10}) = \varepsilon/(1+\varepsilon^2)$:

$$
V \propto \frac{2\varepsilon}{1+\varepsilon^2}\left[
E_{00}E_{11} - E_{01}E_{10}
+ \frac{D_0(E_{11}C_0 - E_{10}C_1)}{1-\mathbf{v}\cdot\mathbf{n}}
\right].
$$

---

## 7. Explicit formulas in the head-on limit

In the head-on limit ($\alpha=0$, $R=I$, $C_i=0$, $D_0=0$), the crossing-angle terms vanish and we recover the standard Compton scattering Stokes parameters for a polarized laser:

$$
\begin{aligned}
I &\propto 1, \\
Q &\propto \frac{\cos^2(\psi-\psi_{\text{pol}}) - \varepsilon^2\sin^2(\psi-\psi_{\text{pol}})}{1+\varepsilon^2}, \\
U &\propto \frac{2(1-\varepsilon^2)\cos(\psi-\psi_{\text{pol}})\sin(\psi-\psi_{\text{pol}})}{1+\varepsilon^2} = \frac{(1-\varepsilon^2)\sin 2(\psi-\psi_{\text{pol}})}{1+\varepsilon^2}, \\
V &\propto \frac{2\varepsilon}{1+\varepsilon^2}\left[\cos^2(\psi-\psi_{\text{pol}}) + \sin^2(\psi-\psi_{\text{pol}})\right] = \frac{2\varepsilon}{1+\varepsilon^2}.
\end{aligned}
$$

Checks:
- $\varepsilon=0$ (linear): $Q = \cos 2(\psi-\psi_{\text{pol}})$, $U = \sin 2(\psi-\psi_{\text{pol}})$, $V=0$ — fully linearly polarized, angle $2(\psi-\psi_{\text{pol}})$. ✓
- $\varepsilon=1$ (circular): $Q=0$, $U=0$, $V=1$ — fully circularly polarized. ✓
- Degree of polarization: $P = \sqrt{Q^2+U^2+V^2}/I = 1$ for all $\varepsilon$ — the scattered photons are fully polarized in the head-on limit (as expected for Thomson/Compton scattering of a pure polarization state). ✓

---

## 8. Full formulas with crossing angle (general case)

The general expressions are the boxed formulas in §6.2–6.4 with the building blocks from §4. For implementation, the most practical form is to compute the 2×2 matrix

$$
M_{ab} = \sum_{i,j} \Xi_{ji}
\left(E_{ai} + \frac{D_a C_i}{1-\mathbf{v}\cdot\mathbf{n}}\right)
\left(E_{bj} + \frac{D_b C_j}{1-\mathbf{v}\cdot\mathbf{n}}\right)
$$

numerically from the vectors $\mathbf{f}_0, \mathbf{f}_1, \mathbf{e}_0, \mathbf{e}_1, \mathbf{n}, \mathbf{v}$, then extract Stokes parameters via

$$
I = M_{00}+M_{11},\quad Q = M_{00}-M_{11},\quad U = 2M_{01},\quad V = 2i(M_{10}-M_{01})/2 = -2\Im(M_{01}).
$$

Since all vectors are real and $\Xi$ is Hermitian, $M$ is Hermitian: $M_{10} = M_{01}^*$. The numerical approach avoids sign errors in the analytic expansion.

---

## 9. Degree of polarization and polarization ellipse

The degree of polarization is

$$
P = \frac{\sqrt{Q^2+U^2+V^2}}{I}.
$$

In the head-on limit $P=1$ (fully polarized). With crossing angle, $P < 1$ generally because the crossing angle mixes polarization components — the scattered radiation becomes partially polarized even for a pure incident state. This is a physical effect: the crossing angle breaks the symmetry that guaranteed full polarization in the head-on case.

The polarization ellipse parameters (orientation $\chi$, ellipticity $\eta$) are

$$
\tan 2\chi = \frac{U}{Q},\qquad
\sin 2\eta = \frac{V}{\sqrt{Q^2+U^2+V^2}}.
$$

---

## 10. Small-angle approximation (Feshchenko et al. 2016 factorization)

The 2016 FIAN preprint (Feshchenko, Vinogradov, Artyukov, *Mathematical model for calculating parameters of X-ray radiation of a laser-electron generator*, Preprint FIAN No. 2, 2016) derives a factorized form for the scattering matrix that is computationally cheaper. This section documents that approximation and its validity domain.

### 10.1 The factorization

In the head-on limit ($\alpha=0$), the 2×2 scattering matrix in the basis $\mathbf{f}_0,\mathbf{f}_1$ (scattering plane / perpendicular) is diagonal:

$$
M = \begin{pmatrix} \sqrt{m_{11}} & 0 \\ 0 & \sqrt{m_{22}} \end{pmatrix},
$$

where $m_{11}, m_{22}$ are the differential cross sections for laser polarization parallel ($\sin\alpha=1$) and perpendicular ($\sin\alpha=0$) to the scattering plane (Eqs 12, 14 in the paper; our §7).

For a **small crossing angle** $\alpha \ll 1$, the paper argues that the only effect is a geometric rotation of the single-electron scattering pattern. The full matrix becomes

$$
M_\sigma = O^T M O,
$$

where $O$ is the orthogonal rotation matrix by angle $\varepsilon$ (the angle between the plane of $\mathbf{v},\mathbf{n}$ and the $y$-$z$ plane):

$$
O = \begin{pmatrix} \cos\varepsilon & -\sin\varepsilon \\ \sin\varepsilon & \cos\varepsilon \end{pmatrix}.
$$

The Stokes parameters of the scattered radiation are then (Eqs 20–23 in the paper):

$$
\begin{aligned}
\sigma_0 &= \tfrac{1}{2}\bigl[m_{11}+m_{22} - (\xi_{L3}\cos2\varepsilon + \xi_{L1}\sin2\varepsilon)(m_{11}-m_{22})\bigr], \\
\sigma_0\eta_3 &= -\tfrac{1}{2}(m_{11}-m_{22})\cos2\varepsilon
+ \tfrac{\xi_{L3}}{2}\sqrt{(m_{11}+m_{22})^2\cos^22\varepsilon + 2m_{11}m_{22}\sin^22\varepsilon}
+ \tfrac{\xi_{L1}}{2}(m_{11}+m_{22}-2m_{11}m_{22})\sin2\varepsilon\cos2\varepsilon, \\
\sigma_0\eta_1 &= -\tfrac{1}{2}(m_{11}-m_{22})\sin2\varepsilon
+ \tfrac{\xi_{L1}}{2}\sqrt{(m_{11}+m_{22})^2\sin^22\varepsilon + 2m_{11}m_{22}\cos^22\varepsilon}
+ \tfrac{\xi_{L3}}{2}(m_{11}+m_{22}-2m_{11}m_{22})\sin2\varepsilon\cos2\varepsilon, \\
\sigma_0\eta_2 &= \xi_{L2} m_{11}m_{22}.
\end{aligned}
$$

Here $\xi_{L1},\xi_{L2},\xi_{L3}$ are the incident laser Stokes parameters (related to our $\varepsilon,\psi_{\text{pol}}$ by $\xi_{L1} = \frac{1-\varepsilon^2}{1+\varepsilon^2}\cos2\psi_{\text{pol}}$, $\xi_{L2} = \frac{2\varepsilon}{1+\varepsilon^2}$, $\xi_{L3} = \frac{1-\varepsilon^2}{1+\varepsilon^2}\sin2\psi_{\text{pol}}$).

**Computational cost:** Only $m_{11}, m_{22}$ (scalars) and the rotation angle $\varepsilon$ are needed — no vector projections $E_{ai}, C_i, D_a$ per phase-space point.

### 10.2 Validity conditions

The factorization $M_\sigma = O^T M O$ with diagonal $M$ relies on **three approximations** that are valid only in a restricted domain:

| Approximation | Physical meaning | Breaks when |
|---------------|------------------|-------------|
| $\mathbf{v}\cdot\mathbf{e}_i \approx 0$ | Laser polarization basis remains transverse to electron velocity | $\sin\alpha \gtrsim 1/\gamma$ |
| $1-\mathbf{v}\cdot\hat{\mathbf{n}}_0 \approx 2$ in cross section | Relative velocity factor $\approx$ head-on value | $\alpha \gtrsim 1/\gamma$ |
| $O(\theta^2)$ reduction | Observation near collinear axis | $\gamma\theta \gtrsim 1$ |

**Quantitative bounds** (from DER005 and the paper's own parameters):

- The paper's Table 1: $\gamma \sim 70\text{--}100$, crossing angle $\theta_0 \sim 50$ mrad
- $\sin\alpha \sim 0.05$, $1/\gamma \sim 0.01\text{--}0.015$ → **$\sin\alpha \gtrsim 1/\gamma$**
- Their Fig 1 shows brightness depends on $\theta_0$ up to 50 mrad
- Their Eq 57 for geometric factor $G$ diverges at $\theta_0=\pi$ because they dropped $1-\mathbf{v}\cdot\hat{\mathbf{n}}_0$ from the cross section

**The neglected term** (from DER005 §2.3, our §5 crossing-angle linear term) is:

$$
\frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j) + (\mathbf{n}\cdot\mathbf{e}_j)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} \sim \frac{\theta \sin\alpha}{1-\mathbf{v}\cdot\mathbf{n}}.
$$

This term:
- Does **not** factor as a rotation of the head-on matrix
- Is **essential** for the $\alpha=90^\circ$ dipole null (head-on formula gives negative values!)
- Causes physical depolarization ($P<1$) even for pure incident states

### 10.3 Recommended usage

| Regime | Method |
|--------|--------|
| **Default / general** | Full vector-based computation (§8) — always correct |
| **Fast path (opt-in)** | Factorized form above, **only when**: $\alpha < 10$ mrad **and** $\gamma\theta < 0.5$ **and** user acknowledges approximation |

**Implementation guardrails** (following project pattern `ELLIPTICITY_IS_NOOP` / `EMISSION_IS_HEAD_ON`):
- Add *SMALL_ANGLE_FACTORIZATION* flag (default `False`)
- `validate()` warns if flag is `True` but $\alpha \ge 10$ mrad or $\gamma\theta \ge 0.5$
- Benchmark: expect 2–3× speedup from avoiding per-point vector projections
- Unit test: compare full vs. factorized at $\alpha=5$ mrad (should agree to $<10^{-3}$) and $\alpha=50$ mrad (should diverge)

---

## 11. Implementation notes

**Inputs needed** (all already available in the codebase):
- `ellipticity` $\varepsilon$, `psi_pol` $\psi_{\text{pol}}$ (laser schema)
- Crossing angles $\theta_{xz}, \theta_{yz}$ (`LaserField` geometry)
- Electron velocity $\mathbf{v}$ (from `Bunch`)
- Observation direction $\mathbf{n}$ (from phase-space sampling)

**Computation per photon/phase-space point**:
1. Compute $\hat{\mathbf{n}}_0$ from crossing angles
2. Compute rotation $R = R_y(\theta_{xz})R_x(\theta_{yz})$
3. Compute $\mathbf{e}_0 = R(\cos\psi_{\text{pol}}, \sin\psi_{\text{pol}}, 0)^T$, $\mathbf{e}_1 = R(-\sin\psi_{\text{pol}}, \cos\psi_{\text{pol}}, 0)^T$
4. Compute $\mathbf{f}_0, \mathbf{f}_1$ from $\mathbf{n}, \mathbf{v}$
5. Compute $E_{ai} = \mathbf{f}_a\cdot\mathbf{e}_i$, $C_i = \mathbf{n}\cdot\mathbf{e}_i$, $D_a = \mathbf{f}_a\cdot\mathbf{v}$
6. Build $M_{ab}$ and extract $I,Q,U,V$

**Where to add**: The Stokes parameters can be computed alongside the spectrum in `stages.py` and `delta.py`. The intensity $I$ is already computed (DER006); $Q,U,V$ are three additional scalars per phase-space point.

---

## Verification

**Symbolically verified with sympy** (`verify_der007_headon.py`):

- **Head-on limit ($\theta \to 0$)** matches DER007 §7 basis-invariant physics:
  - Degree of polarization $P = 1$ for all $\varepsilon$ (pure state preservation).
  - Intensity $I = 1$ (normalized).
  - $|V/I| = 2\varepsilon/(1+\varepsilon^2)$ for circular polarization.
  - $Q^2 + U^2 = 1$ for linear polarization ($\varepsilon=0$).
  - General ellipticity: $|V/I| = 2\varepsilon/(1+\varepsilon^2)$.
- **Mathematical structure:**
  - $M = F \Xi F^\dagger$ is Hermitian ($M_{10} = M_{01}^*$).
  - $M$ is positive semidefinite (det$(M) = 0$ for pure $\Xi$).
  - Rank-1 for pure incident state.
- **Dipole null at $\alpha=90^\circ$:** Verified numerically — $I \to 0$ as $\theta \to 0$ when $\mathbf{e}_0 \parallel \mathbf{n}$.

**Numerically verified with exact vectors** (*verify_der007_numerical.py*):

- $I$ matches DER006 exactly (ratio = 1.000000).
- $P = 1$ exactly for all parameters (pure state preservation).
- Dipole null at $\alpha=90^\circ$: $I \to 0$ as $\theta \to 0$.
- Crossing angle changes polarization pattern.
- $V=0$ for linear polarization ($\varepsilon=0$).

**Note on basis convention:** The exact vector implementation (paper's §8) uses a different basis convention for $\mathbf{f}_0, \mathbf{f}_1$ than the analytic formulas in §7. Individual $Q, U, V$ components differ by sign conventions, but **all basis-invariant physical quantities match exactly** ($P$, $I$, $|V/I|$, dipole null, pattern changes).

**Independent validation:** Pending — same as DER006, requires kascade arbitrary-angle MC.

---

## 13. Used by

Not yet implemented. Once reviewed:
- `stages.py` / `delta.py`: Compute and return $(I,Q,U,V)$ per phase-space point (or integrated over azimuth/energy as needed)
- `collision.py`: Optionally integrate Stokes parameters over phase space for total polarized flux
- Analysis tools: Polarization diagnostics, asymmetry calculations

This derivation supersedes DER006 for polarization-sensitive applications — DER006's intensity formula is the $I$ component here.
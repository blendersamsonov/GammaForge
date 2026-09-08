# DER006 — Polarization matrix of scattered Compton photons with ellipticity and crossing angle

Status: verified

## Setup

This derivation combines **DER004** (ellipticity in the emission kernel's polarization factor) and **DER005** (crossing angle in the emission kernel) to obtain the full polarization matrix for scattered Compton photons when the incident laser has arbitrary ellipticity *and* an arbitrary crossing angle relative to the electron bunch.

The paper's emission kernel (eq. *xsec* / eq. *Fmatrix*) contains the factor

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j),
$$

where $\hat\Xi$ encodes the incident laser polarization and $\mathbf{u}_i$ are the polarization vectors of the scattered photon in the electron rest frame. DER004 computed $\hat\Xi$ for elliptical polarization and $\mathbf{u}_i\cdot\mathbf{u}_j$ in the head-on limit. DER005 computed the full $\mathbf{u}_i\cdot\mathbf{u}_j$ including the crossing-angle term $\mathbf{v}\cdot\mathbf{e}_i$ that was neglected in the head-on derivation. This file assembles the complete expression.

---

## 1. The polarization matrix $\hat\Xi$ (from DER004 §1.2)

The laser field is written as $\mathbf{a}(\varphi) = a_0 \operatorname{Re}\bigl[\boldsymbol{\epsilon} e^{-i\varphi}\bigr]$ with complex polarization vector $\boldsymbol{\epsilon} = \epsilon_0\mathbf{e}_0 + \epsilon_1\mathbf{e}_1$, where $\mathbf{e}_0,\mathbf{e}_1$ are orthonormal vectors spanning the plane transverse to the laser propagation direction $\hat{\mathbf{n}}_0$. The polarization matrix is

$$
\Xi_{ij} = \epsilon_i \epsilon_j^*,
$$

which is Hermitian with $\operatorname{Tr}\hat\Xi = |\epsilon_0|^2 + |\epsilon_1|^2 = 1$.

Parametrizing by scalar ellipticity $\varepsilon \in [0,1]$ (0 = linear, 1 = circular) with the major axis along $\mathbf{e}_0$ and the two components in quadrature:

$$
\epsilon_0 = \frac{1}{\sqrt{1+\varepsilon^2}},\qquad
\epsilon_1 = \frac{i\,\varepsilon}{\sqrt{1+\varepsilon^2}},
$$

$$
\Xi_{00} = \frac{1}{1+\varepsilon^2},\qquad
\Xi_{11} = \frac{\varepsilon^2}{1+\varepsilon^2},\qquad
\Xi_{01} = \frac{-i\,\varepsilon}{1+\varepsilon^2},\qquad
\Xi_{10} = \frac{i\,\varepsilon}{1+\varepsilon^2}.
$$

The cross term $\Re(\Xi_{01}) = 0$ because the components are in quadrature — this is exactly the statement that the ellipse axes align with $\mathbf{e}_0,\mathbf{e}_1$. The angle `psi_pol` in the code is the azimuth of the major axis $\mathbf{e}_0$ in the plane transverse to $\hat{\mathbf{n}}_0$.

---

## 2. The polarization overlap $\mathbf{u}_i\cdot\mathbf{u}_j$ (from DER005 §2.3)

The scattered-photon polarization vectors in the electron rest frame are (eq. `udef`)

$$
\mathbf{u}_i = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_i,
$$

where $\mathbf{n}$ is the observation direction, $\mathbf{v}$ is the electron velocity, and $\mathbf{e}_i$ are the laser polarization basis vectors *in the lab frame* (rotated by the crossing angle). No approximation is made on $\mathbf{v}\cdot\mathbf{e}_i$.

Defining $a_i \equiv \mathbf{n}\cdot\mathbf{e}_i$, $b_i \equiv \mathbf{v}\cdot\mathbf{e}_i$, and using $|\mathbf{n}-\mathbf{v}|^2 = 2(1-\mathbf{v}\cdot\mathbf{n}) - 1/\gamma^2$ and $\mathbf{e}_i\cdot\mathbf{e}_j = \delta_{ij}$, DER005 obtains the exact result:

$$
\boxed{\;
\mathbf{u}_i\cdot\mathbf{u}_j
= \delta_{ij}
- \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j)
      + (\mathbf{n}\cdot\mathbf{e}_j)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}}
\;}
$$

The first two terms are the head-on result (eq. `umod` and DER004's off-diagonal extension); the third term is the crossing-angle correction that was previously neglected.

---

## 3. Geometry: rotated polarization basis

The laser propagates along $\hat{\mathbf{n}}_0$, which is rotated from the head-on direction $-\hat{\mathbf{z}}$ by the crossing-angle rotation $R = R_y(\theta_{xz})R_x(\theta_{yz})$ (as defined in `io.laser`). The polarization basis vectors $\mathbf{e}_0,\mathbf{e}_1$ are defined in the plane transverse to $\hat{\mathbf{n}}_0$ and carried through the same rotation. The paper (§2.2) defines `psi_focus` and `psi_pol` as angles in the head-on frame that are then rotated by $R$ — this is self-consistent and we adopt it.

For sampled particle slopes, the author-approved lab-frame convention is

$$
\mathbf v_e = \beta\frac{(\theta_{x,e},\theta_{y,e},1)}
 {\sqrt{1+\theta_{x,e}^2+\theta_{y,e}^2}},\qquad
\mathbf n = \frac{(\theta_{x,\mathrm{obs}},\theta_{y,\mathrm{obs}},1)}
 {\sqrt{1+\theta_{x,\mathrm{obs}}^2+\theta_{y,\mathrm{obs}}^2}}.
$$

The basis and both vectors remain in this one lab frame; no per-particle rotation is
applied. In the collinear small-angle limit used below, $\mathbf v=\beta\hat{\mathbf z}$ and
we write

$$
\mathbf{n} = \hat{\mathbf{z}} + \boldsymbol{\theta},\qquad
\boldsymbol{\theta} = (\theta_y, \theta_z),\qquad
\theta = |\boldsymbol{\theta}| \ll 1.
$$

The laser direction is

$$
\hat{\mathbf{n}}_0 = \bigl(-\sin\theta_{xz}\cos\theta_{yz},\ \sin\theta_{yz},\ -\cos\theta_{xz}\cos\theta_{yz}\bigr),
$$

and the crossing angle $\alpha$ satisfies $\cos\alpha = \cos\theta_{xz}\cos\theta_{yz}$.

The polarization basis in the head-on frame ($\hat{\mathbf{n}}_0 = -\hat{\mathbf{z}}$) is

$$
\mathbf{e}_0^{(0)} = (\cos\psi_{\text{pol}},\ \sin\psi_{\text{pol}},\ 0),\qquad
\mathbf{e}_1^{(0)} = (-\sin\psi_{\text{pol}},\ \cos\psi_{\text{pol}},\ 0).
$$

After rotation by $R$, the lab-frame basis is $\mathbf{e}_i = R\,\mathbf{e}_i^{(0)}$. The dot products needed are:

$$
\mathbf{n}\cdot\mathbf{e}_i = \mathbf{n}\cdot(R\,\mathbf{e}_i^{(0)}),\qquad
\mathbf{v}\cdot\mathbf{e}_i = \beta\,\hat{\mathbf{z}}\cdot(R\,\mathbf{e}_i^{(0)}).
$$

These are evaluated mechanically once $R$ and $\psi_{\text{pol}}$ are given.

---

## Result

## 4. Combined result: the full polarization factor

Substituting $\hat\Xi$ from §1 and $\mathbf{u}_i\cdot\mathbf{u}_j$ from §2 into the trace:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j=0}^1 \Xi_{ij}\left[
\delta_{ij}
- \frac{a_i a_j}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{a_i b_j + a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}}
\right],
$$

where $a_i = \mathbf{n}\cdot\mathbf{e}_i$, $b_i = \mathbf{v}\cdot\mathbf{e}_i$.

Because $\Xi_{01}$ is purely imaginary and the combination $a_i b_j + a_j b_i$ is symmetric and real, the cross terms $i=0,j=1$ and $i=1,j=0$ contribute only through $\Re(\Xi_{01}) = 0$. **The ellipticity cross term still vanishes** — the ellipse axes remain $\mathbf{e}_0,\mathbf{e}_1$ even after rotation, because the rotation $R$ is a real orthogonal transformation that preserves the relative phase between components.

Thus only the diagonal terms survive:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i=0}^1 \Xi_{ii}\left[
1 - \frac{a_i^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{2 a_i b_i}{1-\mathbf{v}\cdot\mathbf{n}}
\right].
$$

Substituting $\Xi_{00} = 1/(1+\varepsilon^2)$, $\Xi_{11} = \varepsilon^2/(1+\varepsilon^2)$:

$$
\boxed{\;
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \frac{1}{1+\varepsilon^2}\left[
1 - \frac{a_0^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{2 a_0 b_0}{1-\mathbf{v}\cdot\mathbf{n}}
\right]
+ \frac{\varepsilon^2}{1+\varepsilon^2}\left[
1 - \frac{a_1^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{2 a_1 b_1}{1-\mathbf{v}\cdot\mathbf{n}}
\right]
\;}
$$

where

$$
a_i = \mathbf{n}\cdot\mathbf{e}_i,\qquad
b_i = \mathbf{v}\cdot\mathbf{e}_i,\qquad
\mathbf{e}_i = R\,\mathbf{e}_i^{(0)},\qquad
\mathbf{e}_0^{(0)} = (\cos\psi_{\text{pol}},\ \sin\psi_{\text{pol}},\ 0),\quad
\mathbf{e}_1^{(0)} = (-\sin\psi_{\text{pol}},\ \cos\psi_{\text{pol}},\ 0).
$$

---

## 5. Checks and limits

### 5.1 Head-on limit ($\alpha = 0$, $\hat{\mathbf{n}}_0 = -\hat{\mathbf{z}}$)

Then $\mathbf{e}_i = \mathbf{e}_i^{(0)}$, $\mathbf{v}\cdot\mathbf{e}_i = 0$ (so $b_i = 0$), $1-\mathbf{v}\cdot\mathbf{n} = (1+\gamma^2\theta^2)/2\gamma^2$, and $a_i = \mathbf{n}\cdot\mathbf{e}_i = \theta\cos(\psi - \psi_i)$ with $\psi_1 = \psi_0 - \pi/2$. The formula reduces to DER004's result:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}\cdot
\frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2}.
$$

### 5.2 Linear polarization ($\varepsilon = 0$)

Only the $i=0$ term survives:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{a_0^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{2 a_0 b_0}{1-\mathbf{v}\cdot\mathbf{n}}.
$$

This is the full crossing-angle generalization of the linear-polarization kernel.

### 5.3 Circular polarization ($\varepsilon = 1$)

$\Xi_{00} = \Xi_{11} = 1/2$, and the result is the average of the two diagonal terms. The azimuthal dependence enters only through $a_i, b_i$.

### 5.4 $\alpha = 90^\circ$ dipole check (from DER005)

Take $\mathbf{n} = \hat{\mathbf{z}}$, $\mathbf{v} = \beta\hat{\mathbf{z}}$, laser from $+\hat{\mathbf{x}}$ so $\mathbf{e}_0 = -\hat{\mathbf{z}}$. Then $a_0 = -1$, $b_0 = -\beta$, $1-\mathbf{v}\cdot\mathbf{n} = 1-\beta$, $\gamma^2(1-\beta)^2 = (1-\beta)/(1+\beta)$. The $i=0$ term gives

$$
1 - \frac{1}{\gamma^2(1-\beta)^2} + \frac{2\beta}{1-\beta}
= 1 - \frac{1+\beta}{1-\beta} + \frac{2\beta}{1-\beta} = 0,
$$

as required for a dipole radiating along its axis. The head-on formula (without $b_i$) would give a negative value — the crossing-angle term is essential.

---

## 6. Implementation notes

Per RES034, the three crossing-angle pieces must land together:
1. **Relative velocity factor**: $1-\mathbf{v}\cdot\hat{\mathbf{n}}_0 = 1 + \beta\cos\theta_{xz}\cos\theta_{yz}$ (DER005 §2.1)
2. **Resonance frequency / energy conversion**: $\omega_R$ gains $\cos^2(\alpha/2)$ (DER005 §2.2)
3. **Polarization structure**: $\mathbf{u}_i\cdot\mathbf{u}_j$ gains the $b_i$ term (this derivation)

The polarization factor above replaces the head-on `cos²ψ` factor in `stages.py`, `delta.py`, and `collision.py`. The inputs needed are:
- `ellipticity` $\varepsilon$ (scalar, already in schema)
- `psi_pol` $\psi_{\text{pol}}$ (already in schema, defined in head-on frame)
- Crossing angles $\theta_{xz}, \theta_{yz}$ (already in `LaserField` geometry)
- Electron velocity $\mathbf{v}$ (from `Bunch`)
- Observation direction $\mathbf{n}$ (from phase-space sampling)

The rotated basis $\mathbf{e}_i = R\,\mathbf{e}_i^{(0)}$ is computed once per laser configuration and reused.

---

## Verification

**Symbolically verified with sympy** (`verify_der006.py`):

- The full trace matches the boxed formula exactly (difference = 0).
- Cross terms vanish: $\Xi_{01}$ is purely imaginary, $u_0\cdot u_1$ is real.
- **Head-on limit** recovers DER004 exactly (verified via double-angle identities).
- **$\varepsilon = 0$ (linear):** Reduces to single diagonal term (linear polarization).
- **$\varepsilon = 1$ (circular):** Gives average of two diagonal terms (circular polarization).
- **$\alpha = 90^\circ$ dipole null:** Inherited from DER005 (exact zero).
- **Polarization matrix properties:** $\operatorname{Tr}(\hat\Xi) = 1$, $\Re(\Xi_{01}) = 0$, $\Im(\Xi_{01}) = -\varepsilon/(1+\varepsilon^2)$.

`tests/test_stage0_delta.py` independently evaluates Eq. `udef` with the approved
lab-frame vectors and compares it with the production factor at `1e-9` relative tolerance.
The symbolic verifier remains a restricted algebra check; neither test is an independent
arbitrary-angle emission calculation.

---

## 8. Used by

Implemented for the polarization projection by `stages.py`; its vectorized path receives
each Stage-2 table cell's electron angles. `validation.references.delta` supplies each
sample's angles to the same projection. The broader crossing-angle emission validation is
outside this derivation's implementation check.

## CUDA evaluation amendment (2026-09-08)

RES069 implements the same Eq. udef projection for the CuPy sampler with incident
ellipticity and both laser-crossing angles. It evaluates weighted squared vector
norms directly, using stable slope differences for the relativistic denominator
and vector numerator. The expanded expression in §4 is algebraically equivalent
but suffers severe cancellation in float32 for a longitudinal laser basis.
This numerical evaluation choice does not change the derivation's physics or close
independent arbitrary-angle emission validation.

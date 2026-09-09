# Physics Memo: Relativistic Thomson Scattering under Crossing Angles

**To:** External Physics Expert / Theoretical Consultant  
**Date:** September 2026  
**Subject:** Radiation Dipole Projection, Photon Number Conservation, and Angular Symmetry in Relativistic Inverse Compton / Thomson Scattering with Crossing Angles  

---

## 1. Executive Summary & Purpose

In modeling linear Inverse Compton / Thomson scattering between an ultra-relativistic electron bunch ($\gamma \sim 2000$, kinetic energy $\sim 1\text{ GeV}$) and a laser pulse ($\lambda = 1030\text{ nm}$) colliding at a small crossing angle $\alpha$ (e.g., $\alpha = 0.5\text{ mrad} = 1/\gamma$), we encountered a discrepancy involving the angular distribution and photon conservation.

Specifically:
1. An earlier formulation of the differential emission cross section produced an angle-integrated photon yield that scaled as $\sigma_T \left[1 + (\gamma\alpha)^2\right]$. For $\gamma\alpha \sim 1$, this doubled the effective Thomson cross section, causing the number of photons captured by a forward target detector ($N_{\text{tgt}}$) to exceed the total number of emitted photons ($N_{\text{tot}}$) computed from the collision luminosity ($N_{\text{tgt}} / N_{\text{tot}} \approx 110\%$).
2. Furthermore, that earlier formulation exhibited an asymmetry (tilt) in the angular distribution along the crossing axis ($\theta_x$).
3. Setting the radiating dipole to be strictly **transverse to the electron velocity** ($\mathbf{v} \cdot \hat{\mathbf{e}}_\perp \equiv 0$) completely resolved the yield anomaly ($N_{\text{tgt}} \le N_{\text{tot}}$ everywhere, integrating to $\sigma_T$ identically), but restored strict left-right reflection symmetry in the angular distribution ($\theta_x \to -\theta_x$), leaving no angular deflection or tilt toward the laser direction.

We request your review on three specific theoretical physics questions detailed in **Section 4**:
* **Question 1:** Is the transverse dipole condition strictly local to each individual electron velocity vector $\mathbf{v}_e$, or is bunch-axis projection physically sufficient?
* **Question 2:** Does projecting the laser polarization onto the transverse plane discard physical field energy, or is the crossing-angle yield reduction already fully captured by the $(1+\beta\cos\alpha)$ flux factor and spatial beam overlap?
* **Question 3:** In linear Thomson scattering at $\gamma \gg 1$, can a non-zero crossing angle produce an asymmetric/tilted angular distribution, or must the angular distribution remain strictly reflection-symmetric around the electron beam axis?

---

## 2. Physical Setup & Frame Kinematics

Consider the collision in the laboratory frame:

* **Electron Beam:**
  * Mean Lorentz factor: $\gamma = 2000$ ($\beta \approx 1 - \frac{1}{2\gamma^2}$).
  * Nominal propagation axis: $\hat{\mathbf{z}}$ ($\mathbf{v} = v\hat{\mathbf{z}}$).
  * Angular divergence: $\sigma_{\theta} \sim 10^{-4}\text{ rad}$ ($0.1\text{ mrad}$).

* **Laser Pulse:**
  * Wavelength: $\lambda = 1030\text{ nm}$ ($\hbar \omega_0 \approx 1.2\text{ eV}$).
  * Intensity parameter: $a_0 \sim 0.01 \ll 1$ (linear Thomson regime, negligible multiphoton or non-linear ponderomotive deflection).
  * Crossing angle $\alpha$: Defined in the $xz$-plane, such that the laser wavevector is:
    $$\hat{\mathbf{k}}_L = -\sin\alpha\,\hat{\mathbf{x}} - \cos\alpha\,\hat{\mathbf{z}}$$
  * Polarization: Linear in the collision plane ($xz$, p-polarized):
    $$\hat{\mathbf{e}}_L = \cos\alpha\,\hat{\mathbf{x}} - \sin\alpha\,\hat{\mathbf{z}}$$
    such that $\hat{\mathbf{k}}_L \cdot \hat{\mathbf{e}}_L = 0$.

* **Observation Direction:**
  * Unit vector $\mathbf{n} = (\theta_x, \theta_y, 1) / \sqrt{1 + \theta_x^2 + \theta_y^2}$ into solid angle $d\Omega$.
  * Target collimator aperture: Cone of half-angle $\theta_{\text{ap}} = 0.5\text{ mrad} = 1/\gamma$.

---

## 3. Mathematical Mechanism of the Discrepancy

### 3.1. Standard Liénard-Wiechert Radiation

In classical electrodynamics (Jackson Ch. 14, Landau & Lifshitz §73), the far-field radiation emitted by an accelerated charge with velocity $\mathbf{v}$ and acceleration $\dot{\mathbf{v}}$ is:
$$\frac{dP}{d\Omega} = \frac{e^2}{4\pi c^3} \frac{|\mathbf{n} \times [(\mathbf{n} - \mathbf{v}) \times \dot{\mathbf{v}}]|^2}{(1 - \mathbf{v}\cdot\mathbf{n})^5}$$

Using the identity $\mathbf{n} \times [(\mathbf{n}-\mathbf{v}) \times \dot{\mathbf{v}}] = (\mathbf{n}\cdot\dot{\mathbf{v}})(\mathbf{n}-\mathbf{v}) - (1 - \mathbf{v}\cdot\mathbf{n})\dot{\mathbf{v}}$, we define the dimensionless vector:
$$\mathbf{u} = \frac{\mathbf{n} \times [(\mathbf{n}-\mathbf{v}) \times \hat{\mathbf{e}}]}{1 - \mathbf{v}\cdot\mathbf{n}} = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\hat{\mathbf{e}})}{1 - \mathbf{v}\cdot\mathbf{n}} - \hat{\mathbf{e}}$$
where $\hat{\mathbf{e}}$ represents the effective acceleration dipole direction ($\dot{\mathbf{v}} \propto \hat{\mathbf{e}}$).

### 3.2. Expansion of $\mathbf{u}^2$ and the Cross-Term

Squaring $\mathbf{u}$ with $a \equiv \mathbf{n}\cdot\hat{\mathbf{e}}$ and $b \equiv \mathbf{v}\cdot\hat{\mathbf{e}}$:
$$\mathbf{u}^2 = \frac{(\mathbf{n}-\mathbf{v})^2 a^2}{(1 - \mathbf{v}\cdot\mathbf{n})^2} - 2\frac{(\mathbf{n}-\mathbf{v})\cdot\hat{\mathbf{e}}\, a}{1 - \mathbf{v}\cdot\mathbf{n}} + \hat{\mathbf{e}}^2$$

Using $(\mathbf{n}-\mathbf{v})^2 = 2(1 - \mathbf{v}\cdot\mathbf{n}) - \frac{1}{\gamma^2}$ and $(\mathbf{n}-\mathbf{v})\cdot\hat{\mathbf{e}} = a - b$:
$$\mathbf{u}^2 = 1 - \frac{a^2}{\gamma^2(1 - \mathbf{v}\cdot\mathbf{n})^2} + \mathbf{\frac{2 a b}{1 - \mathbf{v}\cdot\mathbf{n}}}$$

### 3.3. The Unphysical Inflation under Laser Tilt

* **In Head-on Collision ($\alpha = 0$):**
  The laser polarization is purely transverse to $\mathbf{v}$: $\hat{\mathbf{e}} = (1, 0, 0)$.
  Thus, $b = \mathbf{v}\cdot\hat{\mathbf{e}} = 0$. The cross-term vanishes identically:
  $$\mathbf{u}^2 = 1 - \frac{a^2}{\gamma^2(1 - \mathbf{v}\cdot\mathbf{n})^2}$$
  Integrating over all solid angle yields the Thomson cross section:
  $$\int d\Omega \frac{3}{8\pi\gamma^2} \frac{\mathbf{u}^2}{(1 - \mathbf{v}\cdot\mathbf{n})^2} \equiv \sigma_T$$

* **With Unprojected Laser Tilt ($\hat{\mathbf{e}} = \hat{\mathbf{e}}_L = \cos\alpha\,\hat{\mathbf{x}} - \sin\alpha\,\hat{\mathbf{z}}$):**
  The longitudinal component along the beam is non-zero:
  $$b = \mathbf{v}\cdot\hat{\mathbf{e}} = -\beta\sin\alpha \approx -\alpha$$
  $$a = \mathbf{n}\cdot\hat{\mathbf{e}} \approx \theta_x - \alpha$$
  In the forward radiation cone ($\theta \sim 1/\gamma$), the Doppler denominator is:
  $$1 - \mathbf{v}\cdot\mathbf{n} \approx \frac{1 + \gamma^2\theta^2}{2\gamma^2}$$
  Substituting $a$ and $b$ into the cross-term flips $\gamma^2$ into the numerator:
  $$\frac{2 a b}{1 - \mathbf{v}\cdot\mathbf{n}} \approx \frac{2(\theta_x - \alpha)(-\alpha)}{\frac{1 + \gamma^2\theta^2}{2\gamma^2}} = \mathbf{+\frac{4(\gamma\alpha)^2}{1 + \gamma^2\theta^2}} - \frac{4\gamma^2\alpha\,\theta_x}{1 + \gamma^2\theta^2}$$

* **Consequences of the Unprojected Formula:**
  1. **Cross-Section Inflation:** When integrated over solid angle $d\Omega$, the $+4(\gamma\alpha)^2$ term adds directly to the cross section:
     $$\sigma_{\text{eff}} = \sigma_T \left[ 1 + (\gamma\alpha)^2 \right]$$
     For $\gamma\alpha = 1$, the integrated cross-section is $2\times \sigma_T$. Because the total yield $N_{\text{tot}}$ in Stage 0 was computed from $\sigma_T$, the target yield within the $1/\gamma$ cone reached $N_{\text{tgt}} \approx 110\% \times N_{\text{tot}}$.
  2. **Angular Asymmetry:** The second term, $-\frac{4\gamma^2\alpha\,\theta_x}{1 + \gamma^2\theta^2}$, is odd in $\theta_x$, creating an apparent deflection/tilt in the forward cone.

### 3.4. Relativistic Dynamics of the Radiating Dipole

In relativistic mechanics, the spatial acceleration $\dot{\mathbf{v}}$ under the Lorentz force $\mathbf{F} = -e(\mathbf{E} + \mathbf{v}\times\mathbf{B})$ is:
$$\dot{\mathbf{v}} = \frac{1}{\gamma m} \left[ \mathbf{F} - \frac{\mathbf{v}}{c^2}(\mathbf{v}\cdot\mathbf{F}) \right]$$

For p-polarization in the $xz$-crossing plane:
* $\mathbf{E} = E_0(\cos\alpha\,\hat{\mathbf{x}} - \sin\alpha\,\hat{\mathbf{z}})$
* $\mathbf{B} = -E_0\,\hat{\mathbf{y}}$
* $\mathbf{v}\times\mathbf{B} = v E_0\,\hat{\mathbf{x}}$
* Transverse force: $F_x = -e E_0(\cos\alpha + \beta)$
* Longitudinal force: $F_z = e E_0\sin\alpha$

Evaluating the longitudinal vs. transverse acceleration components:
$$\dot{v}_x = \frac{F_x}{\gamma m} = -\frac{e E_0(\cos\alpha + \beta)}{\gamma m}$$
$$\dot{v}_z = \frac{F_z}{\gamma^3 m} = +\frac{e E_0\sin\alpha}{\gamma^3 m}$$

The ratio of accelerations is:
$$\frac{\dot{v}_z}{\dot{v}_x} = -\frac{\sin\alpha}{\gamma^2(\cos\alpha + \beta)} \approx -\frac{\alpha}{2\gamma^2}$$
For $\gamma = 2000$ and $\alpha = 0.5\text{ mrad}$:
$$\frac{\dot{v}_z}{\dot{v}_x} \sim 6 \times 10^{-11}$$

Because relativistic longitudinal inertia is $\gamma^3 m$ rather than $\gamma m$, the longitudinal acceleration in the laboratory frame is negligible. The radiation dipole is **strictly transverse to $\mathbf{v}$**.

Projecting the laser polarization unit vector onto the transverse plane ($e_z \to 0$ and renormalizing) sets $b \equiv \mathbf{v}\cdot\hat{\mathbf{e}}_\perp = 0$, identically eliminating the $[1 + (\gamma\alpha)^2]$ inflation and restoring $\int d\Omega \frac{d\sigma}{d\Omega} \equiv \sigma_T$.

---

## 4. Specific Questions for the Consultant

We would appreciate your explicit theoretical feedback on the following three questions:

### Question 1: Per-Electron Velocity vs. Bunch-Averaged Transverse Plane
In our current implementation, the transverse projection is performed relative to the electron bunch central axis $\hat{\mathbf{z}}$:
$$\hat{\mathbf{e}}_{0,\perp} = \frac{(\hat{\mathbf{e}}_{0,x}, \hat{\mathbf{e}}_{0,y}, 0)}{\sqrt{\hat{\mathbf{e}}_{0,x}^2 + \hat{\mathbf{e}}_{0,y}^2}}$$
However, an individual electron has velocity $\mathbf{v}_e / c \approx (\theta_{xe}, \theta_{ye}, 1)$.
* **Question:** Is projecting onto the central bunch axis $\hat{\mathbf{z}}$ completely sufficient for realistic beam emittances ($\sigma_{\theta} \sim 10^{-4}$), or does a rigorous treatment require projecting onto each individual electron's instantaneous velocity plane $\hat{\mathbf{v}}_e^\perp$?
* *Note:* The difference between the two is $O(\sigma_\theta \cdot \alpha) \sim 10^{-8}$.

### Question 2: Does Transverse Projection Double-Count Crossing-Angle Loss?
An intuitive concern was raised: *"If we project the laser electric field onto the transverse plane ($\mathbf{E} \to \mathbf{E}_\perp$), don't we discard the longitudinal field component, thereby reducing the total yield twice?"*
* **Our Analysis:** In the electron's rest frame, the electric field is $\mathbf{E}' = \gamma(\mathbf{E} + \mathbf{v}\times\mathbf{B})_\perp \approx \gamma E_0(1+\cos\alpha)\hat{\mathbf{x}}$ for both in-plane and out-of-plane polarizations because the magnetic force $(\mathbf{v}\times\mathbf{B})_x = v E_0$ adds constructively to $E_x = E_0\cos\alpha$. The total scattering rate in Stage 0 already incorporates the relative velocity factor $c(1+\beta\cos\alpha) = 2c\cos^2(\alpha/2)$ and the spatial overlap integral.
* **Question:** Is it theoretically correct that the Stage 2 angular distribution kernel must integrate to $\sigma_T$ (unit norm), with all crossing-angle yield reduction residing exclusively in the Stage 0 luminosity and kinematic flux?

### Question 3: Symmetry of the Scattered Angular Distribution
Before the fix, the unprojected Eq. `udef` produced a forward angular distribution that was visibly asymmetric/tilted along the crossing plane ($\theta_x$). After enforcing the transverse dipole condition $\mathbf{v}\cdot\hat{\mathbf{e}}_\perp = 0$, the angular distribution becomes strictly reflection-symmetric ($\theta_x \to -\theta_x$):
$$\frac{d^2N}{d\theta_x d\theta_y}(-\theta_x, \theta_y) = \frac{d^2N}{d\theta_x d\theta_y}(\theta_x, \theta_y)$$
* **Question:** In linear Thomson scattering with an ultra-relativistic electron beam ($\gamma = 2000$), can a crossing angle $\alpha$ create an asymmetric or shifted angular distribution of scattered photons relative to the electron beam axis?
* **Our Analysis:** 
  1. Relativistic aberration compresses the incident laser arrival angle in the electron rest frame to $\alpha' \approx \alpha / (2\gamma) \sim 10^{-7}\text{ rad}$ (nearly head-on).
  2. An oscillating electric dipole in the rest frame has radiation pattern $\sin^2\Theta'$, which possesses inversion and reflection symmetry.
  3. A Lorentz boost along $\hat{\mathbf{z}}$ preserves transverse coordinates and azimuthal angles ($\phi_{\text{lab}} = \phi'$), so reflection symmetry across the $yz$-plane must be preserved in the lab frame.
  4. The only parameters that shift with $\alpha$ are the Doppler-shifted Compton edge $E_{\text{max}}(\alpha) = 4\gamma^2\hbar\omega_0\cos^2(\alpha/2)$ and the total bunch luminosity, not the angular centroid.
* **Is this reasoning sound, or is there any physical mechanism in linear Thomson scattering that tilts the far-field angular centroid away from the electron beam axis?**

---

## 5. Summary of Numerical Comparison

For reference, the table below compares simulation results before and after the transverse dipole fix using an identical parameter set ($\gamma = 2000$, $\alpha = 0.5\text{ mrad}$, collimator half-angle $= 0.5\text{ mrad} = 1/\gamma$):

| Configuration | Polarization | Prior Formulation (Eq. udef) | Corrected Formulation (DER012) |
| :--- | :---: | :---: | :---: |
| **Head-on ($\alpha = 0$)** | Linear ($\varepsilon = 0$) | $N_{\text{tgt}} / N_{\text{tot}} = 53.2\%$ | $N_{\text{tgt}} / N_{\text{tot}} = 53.22\%$ |
| **Cross $xz$ ($\alpha = 0.5\,$mrad)** | Linear ($\varepsilon = 0$) | **$N_{\text{tgt}} / N_{\text{tot}} = 110.1\%$** (Anomaly) | **$N_{\text{tgt}} / N_{\text{tot}} = 53.22\%$** (Physical) |
| **Cross $yz$ ($\alpha = 0.5\,$mrad)** | Linear ($\varepsilon = 0$) | $N_{\text{tgt}} / N_{\text{tot}} = 53.2\%$ | $N_{\text{tgt}} / N_{\text{tot}} = 53.22\%$ |
| **Angular profile ($xz$-cross)** | Linear ($\varepsilon = 0$) | Asymmetric / tilted along $\theta_x$ | Symmetric ($+x \leftrightarrow -x$) dipolar pattern |
| **Collinear limit ($\theta = 0, \alpha = 0.3$)** | Linear ($\varepsilon = 0$) | $\mathcal{P} = \cos^2\alpha \approx 0.913$ | $\mathcal{P} = 1.000$ |

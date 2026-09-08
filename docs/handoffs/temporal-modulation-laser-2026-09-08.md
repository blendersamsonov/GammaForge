# Handoff: Temporal Intensity Modulation & Pulse Trains in `gammaforge.io.laser`

**Date:** 2026-09-08  
**Target Module:** `gammaforge.io.laser`  
**Related Requirements:** Phase 1 (Core `gammaforge.io`), P15 (`LaserField` protocol boundary), RES054, RES067  
**Document Status:** Ready for implementation  

---

## 1. Goal & Context

In high-power laser facilities (such as coherent pulse stacking in Gires-Tournois cavities, fiber laser combining networks, or split-and-delay burst lines), damage thresholds (LIDT) on compression optics prevent concentrating total laser energy $E_{\rm tot}$ into a single monolithic pulse. Delivering the energy as a train of $N_p$ sub-pulses ($N_p \sim 10\text{--}100$) colliding with a single electron bunch mitigates optic damage and suppresses nonlinear spectral broadening.

This handoff specifies the addition of **temporal intensity modulation / pulse-train support** to the laser submodule in [`gammaforge.io.laser`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py), enabling simulation of modulated laser fields across downstream engines (`xigma`, `kascade`) without altering engine internals.

---

## 2. Architecture & Design Principles

1. **Protocol Adherence (P15, RES067):**  
   All calculation engines (`xigma.stages`, `kascade`) consume the [`LaserField`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py#L115-L148) protocol, which requires four vectorized lab-frame methods:
   * `intensity_profile(x, y, z, t) -> np.ndarray` (returns cycle-averaged $\langle a^2 \rangle$)
   * `a0_profile(x, y, z, t) -> np.ndarray` (returns peak vector potential $a_0$)
   * `field(x, y, z, t) -> np.ndarray` (returns 3-vector field components)
   * `active_region(threshold) -> ActiveRegion` (spatial bounding cone/cylinder for trajectory prefiltering)
   Any new laser model implementing this protocol integrates seamlessly into Stage 0/1 without engine modification.

2. **Class Design Recommendation:**  
   Implement a dedicated class [`PulseTrainParaxialLaser`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py) in `laser.py` (or subclassing/composing with `GaussianParaxialLaser`):
   * **Why a separate class:** Keeps [`GaussianParaxialLaser`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/laser.py#L246-L290) clean, backward-compatible, and lightweight.
   * **Parameter Contract:** Uses canonical pint `Quantity` fields at construction (`pulse_energy`, `wavelength`, `sigma_x`, `sigma_y`, `subpulse_duration`, `repetition_period`, offsets, angles), converting to CGS in `__post_init__` ([`RES013`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/implemented/architecture/RES013-pint-quantity-boundary.md), [`RES017`](file:///home/alexander/Work/Code/GammaForge/docs/decisions/archived/architecture/RES017-canonical-cgs-on-construction.md)).

3. **Physics Goes Through $\langle a^2 \rangle$ (RES054):**  
   Total pulse energy $E_{\rm tot}$ fixes the cycle-averaged field density and total photon count $N_{\rm ph} = E_{\rm tot} / (\hbar \omega_0)$. The peak $a_0$ of each sub-pulse scales naturally as $a_{0, k} \approx a_{0, \text{single}} / \sqrt{N_p}$.

---

## 3. Mathematical & Algorithmic Specification

### 3.1 Temporal Envelope
Let the single-pulse RMS duration be $\sigma_\tau = \tau_p / (2\sqrt{2\ln 2})$ with length scale $s_{ct} = c \sigma_\tau$.  
For $N_p$ sub-pulses with inter-pulse period $T_{\rm rep}$ (spacing $\Delta \zeta = c T_{\rm rep}$):

$$
t_k = \left( k - \frac{N_p + 1}{2} \right) T_{\rm rep}, \qquad k \in \{1, \dots, N_p\}.
$$

The longitudinal coordinate in the pulse frame is $\zeta = u - ct$, where $u = \hat{\mathbf{k}} \cdot (\mathbf{r} - \mathbf{r}_{\rm off})$.  
The longitudinal photon density envelope $\rho_\parallel(\zeta)$ is:

$$
\rho_\parallel(\zeta) = \frac{1}{N_p \sqrt{2\pi}\, s_{ct}} \sum_{k=1}^{N_p} \exp\left[ -\frac{(\zeta - c\, t_k)^2}{2 s_{ct}^2} \right].
$$

**Normalization Property:**
$$
\int_{-\infty}^\infty \rho_\parallel(\zeta)\, \mathrm{d}\zeta = \frac{1}{N_p} \sum_{k=1}^{N_p} 1 = 1.
$$
Therefore, the 3D spatial integral at any time $t$ satisfies:
$$
\iiint \rho(x, y, z, t)\, \mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z \equiv 1.
$$

### 3.2 3D Photon Density & Field Profiles
Using the paraxial transverse spot sizes $s_1(u_{\rm spot}), s_2(u_{\rm spot})$ along focusing axes $\xi_1, \xi_2$:
$$
\rho(x, y, z, t) = \frac{1}{2\pi s_1(u_{\rm spot}) s_2(u_{\rm spot})} \exp\left[ -\frac{\xi_1^2}{2 s_1^2(u_{\rm spot})} - \frac{\xi_2^2}{2 s_2^2(u_{\rm spot})} \right] \cdot \rho_\parallel(u - ct).
$$
The cycle-averaged intensity profile is:
$$
\langle a^2 \rangle(x, y, z, t) = \left( \frac{e}{m_e c\,\omega_0} \right)^{\!2} 4\pi E_{\rm tot} \cdot \rho(x, y, z, t).
$$

### 3.3 Active Region & Bounding Envelope
`ActiveRegion` determines the temporal integration window and spatial prefiltering cone in Stage 0.  
For a train of $N_p$ pulses, the temporal half-length must span the entire burst:
$$
T_{\rm burst} \approx (N_p - 1) T_{\rm rep}.
$$
The bounding half-length along the propagation axis is:
$$
\text{half\_length} = \frac{c (N_p - 1) T_{\rm rep}}{2} + s_{ct} \sqrt{-2\ln(\text{threshold})}.
$$
Because this expands the temporal bounding box, Stage 0’s overlap time window and trajectory illumination filter ([`bunch.py`](file:///home/alexander/Work/Code/GammaForge/src/gammaforge/io/bunch.py#L63-L70)) will automatically track all macroparticles through the entire train!

---

## 4. Implementation Steps

### Step 1: Data Model in `src/gammaforge/io/laser.py`
Define `PulseTrainParaxialLaser`:
```python
@dataclass(frozen=True)
class PulseTrainParaxialLaser:
    pulse_energy: Quantity        # Total energy summed over all sub-pulses
    wavelength: Quantity
    sigma_x: Quantity
    sigma_y: Quantity
    subpulse_duration: Quantity   # RMS duration of one sub-pulse
    repetition_period: Quantity   # Time between adjacent sub-pulses
    n_subpulses: int = 10         # Number of sub-pulses (N_p >= 1)
    z_fx: Quantity = Quantity(0.0, "cm")
    z_fy: Quantity = Quantity(0.0, "cm")
    x_off: Quantity = Quantity(0.0, "cm")
    y_off: Quantity = Quantity(0.0, "cm")
    t_off: Quantity = Quantity(0.0, "s")
    theta_xz: Quantity = Quantity(0.0, "rad")
    theta_yz: Quantity = Quantity(0.0, "rad")
    psi_focus: Quantity = Quantity(0.0, "rad")
    psi_pol: Quantity = Quantity(0.0, "rad")
    ellipticity: float = 0.0
    beta_ff: float = 0.0
```
* Register `subpulse_duration`, `repetition_period`, and `t_off` in `LIGHT_TIME_FIELDS`.
* Enforce $N_p \ge 1$ and positive scalars in `__post_init__`.

### Step 2: Implementation of `LaserField` Methods
* `photon_density(x, y, z, t)`: Vectorized sum over the $N_p$ Gaussians along the longitudinal coordinate.
* `intensity_profile(x, y, z, t)`: Scale `photon_density` by $\left(\frac{e}{m_e c\omega_0}\right)^2 4\pi E_{\rm tot}$.
* `a0_profile(x, y, z, t)`: Period-averaged amplitude from density.
* `active_region(threshold)`: Return `ActiveRegion` with expanded `half_length`.

### Step 3: Export & Schema
* Add `PulseTrainParaxialLaser` to `__all__` in `src/gammaforge/io/laser.py` and `src/gammaforge/io/__init__.py`.
* Expose duty cycle helper method:
  ```python
  def duty_cycle(self) -> float:
      """Duty cycle D = subpulse_duration / repetition_period."""
      return self.m("subpulse_duration") / self.m("repetition_period")
  ```

---

## 5. Verification & Acceptance Criteria

Write unit and integration tests in `tests/test_laser_pulse_train.py`:

1. **Single-Subpulse Equivalence ($N_p = 1$):**  
   When $N_p = 1$, `PulseTrainParaxialLaser(..., subpulse_duration=D, ...)` must match `GaussianParaxialLaser(..., duration=D, ...)` across `photon_density`, `intensity_profile`, and `active_region` to within machine precision ($10^{-14}$).

2. **Spatial Normalization Check:**  
   Numerically integrate `photon_density(x, y, z, t)` over a 3D grid in $(x, y, z)$ at multiple time points $t \in \{-T_{\rm rep}, 0, T_{\rm rep}\}$. The integral must equal $1.0000 \pm 10^{-4}$.

3. **Total Photon Count Invariance:**  
   `laser.n_photons()` must match $E_{\rm tot} / (\hbar \omega_0)$ regardless of $N_p$ or $T_{\rm rep}$.

4. **Active Region Enclosure:**  
   Verify that `active_region(threshold=1e-4).contains(...)` evaluates to `True` for every sub-pulse center $(0, 0, -c t_k)$ at $t = 0$.

5. **End-to-End Xigma Interaction Test:**  
   Run `stages.track_and_deposit` (Stage 0) with a test `Bunch` and `PulseTrainParaxialLaser(n_subpulses=10)`:
   * Confirms that macroparticles accumulate column densities across all sub-pulses.
   * Verify that total scattered yield decays as duty cycle $D \to 0$ when inter-pulse spacing pushes collisions outside the Rayleigh range.

6. **Doc and Decision Integrity:**  
   * Ensure `pytest tests/test_doc_staleness.py tests/test_decision_format.py` stays green.
   * File an architectural decision in `docs/decisions/implemented/architecture/` once implemented.

---

## 6. Current Status & Readiness

* The report draft document [`docs/REPORT_DRAFT.md`](file:///home/alexander/Work/Code/GammaForge/docs/REPORT_DRAFT.md#L407-L509) already carries Section 6 documenting this physics, equations, and proposed figures.
* The codebase is ready for this implementation; `laser.py` has zero downstream blockers.

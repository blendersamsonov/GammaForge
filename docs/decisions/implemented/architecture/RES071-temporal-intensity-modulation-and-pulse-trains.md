# RES071 — Temporal intensity modulation and pulse trains

Status: implemented
Class: architecture

## Problem

In high-brightness Inverse Compton Scattering facilities, laser-induced damage thresholds on
final dielectric compression optics and nonlinear phase accumulation prevent concentrating
the full pulse energy $E_{\rm tot}$ into a single ultra-short, ultra-intense pulse. Modern
high-repetition-rate laser architectures (such as coherent pulse stacking in Gires-Tournois
cavities, multi-channel fiber combining networks, or split-and-delay burst lines) naturally deliver
laser energy as a train of $N_p$ sub-pulses ($N_p \sim 10\text{--}100$) colliding with an electron bunch.

Until now, `gammaforge.io.laser` only provided `GaussianParaxialLaser`, representing isolated monolithic
pulses. Simulating pulse train interactions was not possible without either manually hacking
offsets or modifying downstream calculation engines.

Furthermore, an audit of `ActiveRegion.origin` revealed that the timing offset `t_off` had an inverted sign
when computing the origin offset along the propagation vector, which caused `ActiveRegion.contains` to evaluate
to False at the pulse arrival time for non-zero `t_off` and caused `overlap_time_window` to miss the
interaction interval.

## Decision

1. **Dedicated `PulseTrainParaxialLaser` Class**:
   Implement `PulseTrainParaxialLaser` in `gammaforge.io.laser`, implementing the full `LaserField` protocol
   (`intensity_profile`, `a0_profile`, `field`, `active_region`).
   - Parameterized by `pulse_energy` ($E_{\rm tot}$), `wavelength`, waists (`sigma_x`, `sigma_y`),
     `subpulse_duration` ($\tau_p$), `repetition_period` ($T_{\rm rep}$), and integer `n_subpulses` ($N_p \ge 1$).
   - Exposes `duty_cycle` returning $\tau_p / T_{\rm rep}$, `subpulse_delays` returning $\{t_k\}$,
     and `subpulses` returning constituent `GaussianParaxialLaser` instances.
   - For $N_p = 1$, `PulseTrainParaxialLaser` reproduces `GaussianParaxialLaser` identically to machine
     precision ($10^{-14}$), and `fit_gaussian_paraxial` returns the constituent sub-pulse directly.

2. **Cycle-Averaged Intensity Superposition**:
   Conserving total energy $E_{\rm tot}$ partitions each sub-pulse to energy $E_k = E_{\rm tot} / N_p$.
   The longitudinal envelope sums $N_p$ Gaussians along the light-cone coordinate $\zeta = u - ct$.
   The cycle-averaged intensity $\langle a^2 \rangle$ is mathematically identical to the sum of the cycle-averaged
   intensities of the constituent sub-pulses.

3. **Active Region Bounding**:
   `ActiveRegion.half_length` expands to cover the burst extent $c(N_p - 1)T_{\rm rep}/2 + 2 s_{ct}\sqrt{\ln(1/\text{threshold})}$,
   guaranteeing all sub-pulses are encompassed across the entire collision.

4. **Timing Offset Origin Sign Correction**:
   Correct the timing offset in `ActiveRegion.origin` to subtract the displacement rather
   than add it, aligning `ActiveRegion.contains` with the lab-frame coordinate evolution in `photon_density`.

## Alternatives considered

*Subclassing GaussianParaxialLaser or overloading its duration field with pulse-train arrays.* Rejected:
violates Single Responsibility Principle and corrupts the clean analytic geometry of `GaussianParaxialLaser`.
A separate dataclass adhering to `LaserField` satisfies principle P15 cleanly without engine modifications.

*Modifying engines (Xigma, Kascade) to loop over pulses.* Rejected: violates the engine decoupling boundary.
Stage 0 integrates the lab-frame `intensity_profile` across any field geometry satisfying `LaserField`.
Supplying the modulated envelope directly to `LaserField` requires zero changes to calculation engines.

*Allowing fit_gaussian_paraxial to invent a speculative equivalent Gaussian for N_p > 1.* Rejected:
violates principle P6 (no speculative abstraction). Closed-form analytical overlap requires an actual
astigmatic Gaussian pulse; a train with $N_p > 1$ is non-Gaussian and executes through Stage 0 / Xigma.

## Rationale

Because Xigma Stage 0 integrates `intensity_profile` along electron trajectories (RES054, RES067), any spatio-temporal
field implementing `LaserField` can be integrated directly.
Summing independent interactions of the constituent sub-pulses provides an exact benchmark for validation:
for wide pulse spacing ($D \ll 1$), independent sub-pulse simulation concentrates integration points
within each sub-pulse window, avoiding undersampling of narrow spikes.

## Consequences

- `PulseTrainParaxialLaser` is available in `gammaforge.io.laser` and exported by `gammaforge.io`.
- `scripts/calculate_duty_cycle_yield.py` provides a turnkey verification tool sweeping duty cycle $D$
  and comparing full-train integration against summed constituent sub-pulses.
- Stage 0 trajectory tracking correctly captures pulse trains across arbitrary burst lengths and duty cycles.

# %% [markdown]
# # GammaForge Interactive Walkthrough: 01 — Core Contracts, Units & I/O
# 
# Welcome! This notebook is designed specifically for **human developers** to build an intuitive,
# rock-solid mental model of GammaForge's core architecture, data contracts, and unit conventions.
# 
# ### What you will learn in this notebook:
# 1. **The CGS-Gaussian Rule & Pint at the boundary** — why units exist on dataclasses but never inside kernels.
# 2. **Electron Beam (`GaussianElectronBeam`) & Macroparticle Bunch (`Bunch`)** — why momentum is derived, why correlations are coefficients, and how arrays are structured.
# 3. **Laser Pulse Protocol (`LaserField`) & `GaussianParaxialLaser`** — geometry angles, $\langle a^2 \rangle$ vs $a_0$, and conventions.
# 4. **Interaction Packaging (`Target`, `SamplingSpec`, `build_interaction`)** — bundling everything into `InteractionParameters` for the engines.
# 5. **Interactive 2D/3D Phase-Space Visualizations** with both Matplotlib and Plotly.

# %% [markdown]
# ## 1. Unit System: CGS-Gaussian & Pint at the Boundary
# 
# In GammaForge (`GRAND_PLAN.md §2.1`, Principle P1):
# - **Canonical internal storage is strictly CGS-Gaussian**: cm, s, g, erg, statC, gauss.
# - **Inputs at the user boundary are dimensioned using Pint `Quantity`**.
#   This guarantees that a caller providing beam energy in MeV, pulse energy in mJ, or spot size in $\mu$m
#   has their units checked and converted automatically.
# - **Pint stops at the engine boundary**: kernels only see raw NumPy float arrays. Per-particle arrays are NOT wrapped in `Quantity`.

# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from gammaforge.io.units import (
    ureg,
    Quantity,
    C_CGS,
    E_ESU,
    MEC2_CGS,
    STATC_PER_COULOMB,
)

print(f"Speed of light c: {C_CGS:.3e} cm/s")
print(f"Electron charge e: {E_ESU:.3e} statC (esu)")
print(f"Electron rest mass energy: {MEC2_CGS:.3e} erg")
print(f"1 Coulomb in statC: {STATC_PER_COULOMB:.3e} statC")

# Boundary conversion example:
user_energy = 50.0 * ureg.megaelectronvolt
cgs_energy = user_energy.to("erg")
print(f"\nUser input: {user_energy} -> CGS: {cgs_energy:.4e} ({cgs_energy.magnitude:.4e} erg)")

# %% [markdown]
# ## 2. Electron Beam & Macroparticle Bunch
# 
# Key architectural guarantees in `gammaforge.io.bunch`:
# 1. **`GaussianElectronBeam`** defines the 6D Gaussian analytic distribution.
# 2. **`Bunch`** holds the macroparticle arrays: `x`, `y`, `z`, `px`, `py`, `pz`, `weight`.
# 3. **Momenta are derived, never sampled**:
#    The sampler draws `(x, y, z, thx, thy, gamma)`.
#    Then:
#    $$p_z = \sqrt{\frac{\gamma^2 - 1}{1 + \theta_x^2 + \theta_y^2}}, \quad p_x = \theta_x p_z, \quad p_y = \theta_y p_z$$
#    Because $p_z$ is obtained this way, $\gamma^2 = 1 + p^2$ is an **algebraic identity** for every particle — no mass-shell enforcement step can go wrong!
# 4. **Correlations are coefficients $\rho \in (-1, 1)$**, not dimensional slopes.

# %%
from gammaforge.io.bunch import (
    GaussianElectronBeam,
    Bunch,
    sample_gaussian_bunch,
    momenta,
)

beam = GaussianElectronBeam(
    bunch_charge=100.0 * ureg.picocoulomb,
    kinetic_energy=50.0 * ureg.megaelectronvolt,
    rel_energy_spread=0.01,
    sigma_x=10.0 * ureg.micrometer,
    sigma_y=10.0 * ureg.micrometer,
    emit_x=1.0 * ureg.millimeter * ureg.milliradian,
    emit_y=1.0 * ureg.millimeter * ureg.milliradian,
    sigma_z=100.0 * ureg.micrometer,
)

print("Beam summary:")
print(f"  Gamma0: {beam.gamma0():.2f}")
print(f"  Number of physical electrons N_e: {beam.n_electrons():.3e}")
print(f"  Sigma X (CGS): {beam.sigma_x.m:.4e} cm")
print(f"  Sigma Z (CGS): {beam.sigma_z.m:.4e} cm")

# Draw a fast toy bunch (100 particles for instant execution)
bunch = sample_gaussian_bunch(beam, n_particles=100, seed=42)

print(f"\nSampled Bunch ({bunch.n_particles} macroparticles):")
print(f"  x range: [{bunch.x.min()*1e4:.2f}, {bunch.x.max()*1e4:.2f}] µm")
print(f"  z range: [{bunch.z.min()*1e4:.2f}, {bunch.z.max()*1e4:.2f}] µm")
print(f"  gamma range: [{bunch.gamma.min():.2f}, {bunch.gamma.max():.2f}]")

# Derive momenta and verify mass-shell identity: gamma^2 == 1 + px^2 + py^2 + pz^2
px, py, pz = momenta(bunch)
p_sq = px**2 + py**2 + pz**2
gamma_sq = bunch.gamma**2
max_diff = np.max(np.abs(gamma_sq - (1.0 + p_sq)))
print(f"  Max deviation from mass shell (|gamma^2 - (1+p^2)|): {max_diff:.2e} (Exact!)")

# %% [markdown]
# ### Visualizing the Bunch: 2D Phase Space (Matplotlib)
# Let's inspect the spatial $(x, y)$ and transverse phase-space $(x, \theta_x)$ projections.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Transverse spot (x vs y)
axes[0].scatter(bunch.x * 1e4, bunch.y * 1e4, c=bunch.gamma, cmap="viridis", alpha=0.8)
axes[0].set_xlabel("x [µm]")
axes[0].set_ylabel("y [µm]")
axes[0].set_title("Transverse Profile (x-y)")
axes[0].grid(True, alpha=0.3)

# 2. Phase space (x vs thx)
axes[1].scatter(bunch.x * 1e4, bunch.thx * 1e3, c=bunch.gamma, cmap="viridis", alpha=0.8)
axes[1].set_xlabel("x [µm]")
axes[1].set_ylabel("θ_x [mrad]")
axes[1].set_title("Horizontal Phase Space (x - θ_x)")
axes[1].grid(True, alpha=0.3)

# 3. Longitudinal phase space (z vs gamma)
axes[2].scatter(bunch.z * 1e4, bunch.gamma, c=bunch.gamma, cmap="viridis", alpha=0.8)
axes[2].set_xlabel("z [µm]")
axes[2].set_ylabel("γ (Lorentz factor)")
axes[2].set_title("Longitudinal Phase Space (z - γ)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualizing the Bunch: Interactive 3D Phase Space (Plotly)
# Hover over particles to inspect individual particle coordinates and energies!

# %%
fig_3d = go.Figure(
    data=[
        go.Scatter3d(
            x=bunch.x * 1e4,
            y=bunch.y * 1e4,
            z=bunch.z * 1e4,
            mode="markers",
            marker=dict(
                size=4,
                color=bunch.gamma,
                colorscale="Viridis",
                colorbar=dict(title="γ"),
                opacity=0.8,
            ),
            text=[f"Particle {i}<br>γ = {g:.2f}<br>θx = {tx*1e3:.2f} mrad" 
                  for i, (g, tx) in enumerate(zip(bunch.gamma, bunch.thx))],
            hoverinfo="text",
        )
    ]
)

fig_3d.update_layout(
    title="Interactive 3D Macroparticle Bunch",
    scene=dict(
        xaxis_title="x [µm]",
        yaxis_title="y [µm]",
        zaxis_title="z [µm]",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    width=700,
    height=500,
)
fig_3d.show()

# %% [markdown]
# ## 3. Laser Field Protocol & `GaussianParaxialLaser`
# 
# Key architectural principles (`GRAND_PLAN.md §3.3`, Principle P15):
# 1. **`LaserField` is a protocol**: Engines sample fields via `intensity_profile()`, `field()`, and `active_region()`.
# 2. **Cycle-averaged intensity $\langle a^2 \rangle$ is the physical quantity**:
#    At fixed pulse energy, $\langle a^2 \rangle$ is identical regardless of polarization state.
#    $a_0$ is merely a reported linear-equivalent convention.
# 3. **Extrinsic Euler angles**:
#    $R = R_y(\theta_{xz}) R_x(\theta_{yz})$ defines laser propagation and roll cleanly.

# %%
from gammaforge.io.laser import GaussianParaxialLaser

laser = GaussianParaxialLaser(
    pulse_energy=10.0 * ureg.millijoule,
    wavelength=800.0 * ureg.nanometer,
    sigma_x=25.0 * ureg.micrometer,
    sigma_y=25.0 * ureg.micrometer,
    duration=30.0 * ureg.femtosecond,
    theta_xz=0.0 * ureg.radian,  # head-on collision
    theta_yz=0.0 * ureg.radian,
)

print("Laser parameters:")
print(f"  Peak a0: {laser.a0_peak():.4f}")
print(f"  Photon count: {laser.n_photons():.3e}")
print(f"  Rayleigh range z_R: {laser.rayleigh_x():.4e} cm")

# Evaluate intensity profile along x at focus and t=0
x_grid = np.linspace(-60e-4, 60e-4, 100)  # -60 to +60 µm in cm
a2_profile = laser.intensity_profile(x_grid, 0.0, 0.0, 0.0)
a0_profile = laser.a0_profile(x_grid, 0.0, 0.0, 0.0)

plt.figure(figsize=(7, 3.5))
plt.plot(x_grid * 1e4, a2_profile, label="⟨a²⟩ (Cycle-averaged intensity)", color="crimson", lw=2)
plt.plot(x_grid * 1e4, a0_profile, label="a₀ (Envelope peak)", color="navy", linestyle="--", lw=2)
plt.xlabel("Transverse coordinate x [µm]")
plt.ylabel("Normalized vector potential")
plt.title("Laser Focus Transverse Profile")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Temporal Intensity Modulation: Pulse Trains (`PulseTrainParaxialLaser`)
# 
# In high-brightness Inverse Compton Scattering facilities, damage thresholds on compression optics
# prevent concentrating total laser energy $E_{\rm tot}$ into a single ultra-intense pulse.
# GammaForge supports **pulse trains** of $N_p$ sub-pulses parameterized by the duty cycle
# $D = \tau_p / T_{\rm rep}$, conserving total energy while suppressing optic damage and nonlinear spectral broadening:

# %%
from gammaforge.io.laser import PulseTrainParaxialLaser

train = PulseTrainParaxialLaser(
    pulse_energy=10.0 * ureg.millijoule,
    wavelength=800.0 * ureg.nanometer,
    sigma_x=25.0 * ureg.micrometer,
    sigma_y=25.0 * ureg.micrometer,
    subpulse_duration=25.0 * ureg.femtosecond,
    repetition_period=100.0 * ureg.femtosecond,
    n_subpulses=5,
)

print("Pulse Train parameters:")
print(f"  Duty cycle D: {train.duty_cycle():.2f}")
print(f"  Number of sub-pulses: {train.n_subpulses}")
print(f"  Total photons: {train.n_photons():.3e}")
print(f"  Sub-pulse delays [fs]: {[f'{t*1e15:.1f}' for t in train.subpulse_delays()]}")

# Longitudinal intensity profile along z at focus (x=0, y=0, t=0)
z_grid = np.linspace(-150e-4, 150e-4, 300)  # -150 to +150 µm in cm
a2_longitudinal = train.intensity_profile(0.0, 0.0, z_grid, 0.0)

plt.figure(figsize=(7, 3.5))
plt.plot(z_grid * 1e4, a2_longitudinal, color="darkorange", lw=2, label=f"5 Sub-pulses (D = {train.duty_cycle():.2f})")
plt.xlabel("Longitudinal coordinate z [µm]")
plt.ylabel("⟨a²⟩ (Cycle-averaged intensity)")
plt.title("Pulse Train Longitudinal Profile at t = 0")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Packaging the Interaction: `Target`, `SamplingSpec`, `build_interaction`
# 
# To run any engine in GammaForge, everything is packaged into an immutable `InteractionParameters` object:
# - **`Target`**: Specifies collimation half-angles (`theta_x_col`, `theta_y_col`) and requested outputs (`OutputRequest`).
#   Temporal output resolution belongs to the request; its range comes from the actual overlap.
#   With no overlap, the zero histogram uses a laser-derived display interval. Every slice axis
#   is auto-ranged by default and can carry a manual override in canonical CGS units; omitted
#   axes remain automatic. Explicit histogram widths retain their captured photon mass.
# - **`SamplingSpec`**: Particle count, random seed, and prefilter threshold fraction.
# - **`build_interaction()`**: The factory function that samples the bunch, prefilters particles outside the laser pulse, and bundles them.

# %%
from gammaforge.io.target import Target, OutputKind, OutputRequest
from gammaforge.io.interaction import SamplingSpec, build_interaction
from gammaforge.io.results import Axis

focused_energy_range = tuple(
    value.to("erg").magnitude for value in (50 * ureg.keV, 250 * ureg.keV)
)

target = Target(
    theta_x_col=1.0 * ureg.milliradian,
    theta_y_col=1.0 * ureg.milliradian,
    outputs=(
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(
            OutputKind.SPECTRUM,
            resolution=(100,),
            manual_ranges={Axis.ENERGY: focused_energy_range},
        ),
    ),
)

sampling = SamplingSpec(n_particles=100, seed=42, prefilter=1e-3)

interaction = build_interaction(beam, laser, target, sampling)

print("Interaction successfully built:")
print(f"  Physical electron count N_e: {interaction.N_e:.3e}")
print(f"  Macroparticles in bunch: {interaction.bunch.n_particles}")
print(f"  Requested outputs: {[out.kind.name for out in interaction.target.outputs]}")
print(f"  Manual spectrum range [keV]: {[value / 1.602176634e-9 for value in focused_energy_range]}")

# %% [markdown]
# ---
# ### Summary of Core Rules Learned:
# 1. **CGS-Gaussian internally**: Pint types protect the inputs; raw floats drive calculations.
# 2. **Mass-shell is guaranteed**: $p_z = \sqrt{(\gamma^2 - 1)/(1 + \theta_x^2 + \theta_y^2)}$ prevents unphysical momenta.
# 3. **Engines consume `InteractionParameters`**: You are now ready for **Notebook 02: Engine Stages & Data Flow**!

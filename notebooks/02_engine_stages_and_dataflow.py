# %% [markdown]
# # GammaForge Interactive Walkthrough: 02 — Engine Stages & Data Flow
# 
# Welcome to the second interactive walkthrough notebook!
# 
# In this notebook, we explore **how GammaForge actually computes Compton scattering**.
# We will demystify the multi-stage pipeline, understand the architectural boundaries between engines,
# inspect intermediate stage representations, and compare results between different physics engines.
# 
# ### What you will learn in this notebook:
# 1. **The 3 Engines in GammaForge**: `AnalyticalEngine`, `XigmaEngine`, and `KascadeEngine`.
# 2. **Engine Architecture Rules**: Why engines have no mutable `Config` (P5) and recompute costs (`RecomputeCost`).
# 3. **The Xigma Pipeline Deep Dive (`Collision`)**:
#    - **Stage 0**: Trajectory integration (`TrajectorySamples`) & active region filtering.
#    - **Stage 1**: Peak-$a_0$-agnostic 4D shape table deposition (`ShapeTable`).
#    - **Stage 1.5**: Retargeting to the physical pulse intensity $\hat{a}$ (`Table`).
#    - **Stage 2**: Emission quadrature & harmonic integration.
# 4. **Live Execution & Side-by-Side Comparison**: Running Xigma and Analytical engines side-by-side in milliseconds.
# 5. **Visualizing Intermediate Diagnostics & Spectra** using Matplotlib and interactive Plotly.
# 6. **Zero-Cost Charge Rescaling**: Scaling yields instantly without re-running engines.

# %% [markdown]
# ## 1. Setting Up the Interaction
# 
# We configure a fast, representative interaction:
# - Electron beam: 50 MeV, 100 pC, 10 µm transverse size.
# - Laser pulse: 10 mJ, 800 nm, 25 µm waist, 30 fs duration.
# - Fast toy sampling: 100 particles for instant (<50ms) execution.

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from gammaforge.io.units import ureg, Quantity, EV_CGS
from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.target import Target, OutputKind, OutputRequest
from gammaforge.io.interaction import SamplingSpec, build_interaction

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

laser = GaussianParaxialLaser(
    pulse_energy=10.0 * ureg.millijoule,
    wavelength=800.0 * ureg.nanometer,
    sigma_x=25.0 * ureg.micrometer,
    sigma_y=25.0 * ureg.micrometer,
    duration=30.0 * ureg.femtosecond,
)

target = Target(
    theta_x_col=1.0 * ureg.milliradian,
    theta_y_col=1.0 * ureg.milliradian,
    outputs=(
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(128,)),
    ),
)

sampling = SamplingSpec(n_particles=100, seed=42)
interaction = build_interaction(beam, laser, target, sampling)
print(f"Interaction built: {interaction.bunch.n_particles} particles, {interaction.N_e:.2e} physical electrons.")

# %% [markdown]
# ## 2. The Engine Architecture & Protocols
# 
# GammaForge defines a uniform `Engine` protocol (`gammaforge.engines.base.Engine`):
# ```python
# class Engine(Protocol):
#     name: str
#     schema: Parameters
#     supported_outputs: tuple[OutputKind, ...]
#     recompute_costs: dict[str, RecomputeCost]
#     def run(self, interaction: InteractionParameters, params: Parameters) -> Results: ...
# ```
# 
# **Critical Design Rules (AGENTS.md & GRAND_PLAN.md):**
# - **No mutable `Config` on an engine (P5)**: Engine knobs live in the typed `Parameters` schema, validated.
# - **Engines never branch the GUI, and the GUI never touches engine internals (P3)**.
# - **Uniform Results (P10)**: Every engine returns a `Results` object holding `PhasespaceSlice` entries.

# %%
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.analytical.engine import AnalyticalEngine

xigma = XigmaEngine()
analytical = AnalyticalEngine()

print(f"Engine 1: '{xigma.name}', outputs: {[o.name for o in xigma.supported_outputs]}")
print(f"Engine 2: '{analytical.name}', outputs: {[o.name for o in analytical.supported_outputs]}")

# %% [markdown]
# ## 3. Inside the Xigma Pipeline: The `Collision` Facade
# 
# While `XigmaEngine.run()` is opaque and stateless across calls, the underlying `Collision`
# object is the facade that executes and memoizes each stage:
# 
# ```
# [ Bunch + Laser ]
#        │
#        ▼
# ┌─────────────────────────────────────────────────────────────┐
# │ Stage 0: integrate_trajectories() -> TrajectorySamples     │
# │ Tracks each macroparticle through the laser pulse:          │
# │ evaluates local a0, angles, and luminosity weights          │
# └──────────────────────────────┬──────────────────────────────┘
#                                │
#                                ▼
# ┌─────────────────────────────────────────────────────────────┐
# │ Stage 1: deposit_shape_table() -> ShapeTable                │
# │ Bins particles into 4D space (gamma, thx, thy, a0_shape).  │
# │ Completely peak-a0-agnostic!                                │
# └──────────────────────────────┬──────────────────────────────┘
#                                │
#                                ▼
# ┌─────────────────────────────────────────────────────────────┐
# │ Stage 1.5: retarget_ahat() -> Table                         │
# │ Scales normalized shape to the physical pulse intensity <a²>│
# └──────────────────────────────┬──────────────────────────────┘
#                                │
#                                ▼
# ┌─────────────────────────────────────────────────────────────┐
# │ Stage 2: angle_integrated_spectrum() -> Results             │
# │ Numerical quadrature over Compton emission harmonics       │
# └─────────────────────────────────────────────────────────────┘
# ```

# %%
from gammaforge.engines.xigma.collision import Collision

# Instantiate a Collision object
collision = Collision(interaction=interaction, params=xigma.schema)

# 1. Run Stage 0: Trajectory integration
t0 = time.perf_counter()
samples = collision.build_overlap()
t1 = time.perf_counter()
print(f"Stage 0 complete in {(t1 - t0)*1000:.2f} ms")
print(f"  Trajectory samples: {len(samples.gamma)} particles")
print(f"  Peak a0 encountered: {samples.a0_shape.max():.4f}")
print(f"  Mean gamma: {samples.gamma.mean():.2f}")
print(f"  Luminosity sum: {samples.luminosity.sum():.4e}")

# %% [markdown]
# ### Visualizing Stage 0 Diagnostics
# Let's inspect the illumination $a_0$ shape and luminosity distribution across macroparticles.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].scatter(samples.gamma, samples.a0_shape, c=samples.luminosity, cmap="plasma", alpha=0.8)
ax[0].set_xlabel("Particle γ")
ax[0].set_ylabel("Encountered a₀ (shape)")
ax[0].set_title("Stage 0: Peak Field vs Particle Energy")
ax[0].grid(True, alpha=0.3)

ax[1].hist(samples.luminosity, bins=25, color="teal", edgecolor="black", alpha=0.7)
ax[1].set_xlabel("Particle Luminosity Weight")
ax[1].set_ylabel("Count")
ax[1].set_title("Stage 0: Luminosity Weight Distribution")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Live Comparison: XigmaEngine vs AnalyticalEngine
# 
# Now let's run both engines on the exact same interaction and compare their execution times,
# total yields, and photon energy spectra!

# %%
# 1. Run Xigma Engine
t_start = time.perf_counter()
xigma_res = xigma.run(interaction, xigma.schema)
t_xigma = (time.perf_counter() - t_start) * 1000

# 2. Run Analytical Engine
t_start = time.perf_counter()
ana_res = analytical.run(interaction, analytical.schema)
t_ana = (time.perf_counter() - t_start) * 1000

xigma_yield = xigma_res.photon_slices[OutputKind.TOTAL_YIELD].integrate()
ana_yield = ana_res.photon_slices[OutputKind.TOTAL_YIELD].integrate()

print(f"Execution Times:")
print(f"  XigmaEngine:      {t_xigma:.2f} ms")
print(f"  AnalyticalEngine: {t_ana:.2f} ms")
print(f"\nTotal Photon Yield Comparison:")
print(f"  Xigma yield:      {xigma_yield:,.1f} photons")
print(f"  Analytical yield: {ana_yield:,.1f} photons")
print(f"  Relative agreement: {abs(xigma_yield - ana_yield)/ana_yield * 100:.2f}%")

# %% [markdown]
# ### Visualizing the Spectra: Side-by-Side Comparison (Matplotlib)
# The `OutputKind.SPECTRUM` slice carries the energy axis in canonical CGS units (erg).
# We convert erg $\to$ keV ($1\text{ keV} \approx 1.602 \times 10^{-9}\text{ erg}$) for standard display.

# %%
from gammaforge.io.results import Axis

# Extract spectrum slices
xigma_spec = xigma_res.photon_slices[OutputKind.SPECTRUM]
ana_spec = ana_res.photon_slices[OutputKind.SPECTRUM]

# Energy axes in keV:
e_kev_xigma = xigma_spec.axes[Axis.ENERGY] / (1e3 * EV_CGS)
e_kev_ana = ana_spec.axes[Axis.ENERGY] / (1e3 * EV_CGS)

# Densities in photons / keV:
# dN/dE [photons/erg] * (1e3 * EV_CGS [erg/keV]) = dN/dE [photons/keV]
dn_de_xigma = xigma_spec.distr * (1e3 * EV_CGS)
dn_de_ana = ana_spec.distr * (1e3 * EV_CGS)

plt.figure(figsize=(8, 4.5))
plt.plot(e_kev_xigma, dn_de_xigma, label=f"XigmaEngine ({xigma_yield:,.0f} ph)", color="navy", lw=2)
plt.plot(e_kev_ana, dn_de_ana, label=f"AnalyticalEngine ({ana_yield:,.0f} ph)", color="crimson", linestyle="--", lw=2)
plt.xlabel("Photon Energy [keV]")
plt.ylabel("Spectral Density dN/dE [photons / keV]")
plt.title("Compton Photon Energy Spectrum Comparison")
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interactive Spectrum Exploration (Plotly)
# Hover over the curves to inspect precise spectral peak locations and Compton edge cutoffs.

# %%
fig_spec = go.Figure()

fig_spec.add_trace(
    go.Scatter(
        x=e_kev_xigma,
        y=dn_de_xigma,
        mode="lines",
        name="XigmaEngine",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="Energy: %{x:.2f} keV<br>dN/dE: %{y:.2e} ph/keV<extra></extra>",
    )
)

fig_spec.add_trace(
    go.Scatter(
        x=e_kev_ana,
        y=dn_de_ana,
        mode="lines",
        name="AnalyticalEngine",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="Energy: %{x:.2f} keV<br>dN/dE: %{y:.2e} ph/keV<extra></extra>",
    )
)

fig_spec.update_layout(
    title="Interactive Spectral Density Comparison",
    xaxis_title="Photon Energy [keV]",
    yaxis_title="Spectral Density dN/dE [photons / keV]",
    template="plotly_white",
    hovermode="x unified",
    width=750,
    height=450,
)
fig_spec.show()

# %% [markdown]
# ## 5. Zero-Cost Operations: Charge Rescaling
# 
# One of GammaForge's key design principles (`GRAND_PLAN.md §5`) is the cost tier hierarchy:
# - Compton scattering at this operating point has **no space charge**.
# - Every output (yield, spectrum, distributions) scales **strictly linearly with $N_e$**.
# - Therefore, changing bunch charge does NOT require re-running any engine!
#   `Results.scaled(factor)` updates all slices instantly (`QUERY_ONLY` cost).

# %%
t_start = time.perf_counter()
# Double the bunch charge:
scaled_res = xigma_res.scaled(2.0)
t_scale = (time.perf_counter() - t_start) * 1e6  # microseconds!

scaled_yield = scaled_res.photon_slices[OutputKind.TOTAL_YIELD].integrate()
print(f"Original yield: {xigma_yield:,.1f} photons")
print(f"Rescaled yield: {scaled_yield:,.1f} photons (Exact factor of 2.0!)")
print(f"Rescaling time: {t_scale:.1f} µs (Instantaneous UI update!)")

# %% [markdown]
# ---
# ### Summary of Engine & Pipeline Rules:
# 1. **Engines are stateless**: Inputs enter via `run(interaction, params)`.
# 2. **Stages memoize cleanly**: `Collision` caches Stage 0 and 1 so multiple output requests reuse work.
# 3. **Linear observables scale for free**: Charge scaling is pure metadata rescaling, avoiding redundant computation.
# 4. In **Notebook 03**, we'll explore how decisions (`RESxxx`), derivations (`DERxxx`), and validation progress are tracked.

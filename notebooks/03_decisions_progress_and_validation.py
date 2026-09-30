# %% [markdown]
# # GammaForge Interactive Walkthrough: 03 — Decisions, Issues & Validation
# 
# Welcome to the third interactive walkthrough notebook!
# 
# When multiple AI agents work on a codebase, humans often feel lost in the stream of
# commits, refactors, and architectural choices. This notebook serves as your **interactive
# control center and briefing room**:
# 
# ### What you will learn & explore in this notebook:
# 1. **The Architecture Knowledge Base**:
#    - How decisions (`docs/decisions/RESxxx`) are permanently recorded and indexed.
#    - How mathematical derivations (`docs/derivations/DERxxx`) move through the confidence pipeline (`derived` $\to$ `validated` $\to$ `verified`).
#    - How GitHub issues track unfinished work while validation reports record measured evidence.
# 2. **Interactive Search & Explorer Widget**:
#    - Query decisions and derivations by topic (e.g., `"laser"`, `"cupy"`, `"polarization"`, `"units"`).
# 3. **Deep Dive into Current Blockers**:
#    - Current independent validation coverage and its limits (RES084).
#    - Arbitrary-angle emission validation.
# 4. **Live Validation Harness in Action**:
#    - Exploring the shared scenario bank (`gammaforge.validation.scenarios.SCENARIOS`).
#    - Running a fast, live validation benchmark on the `BASELINE` scenario.
#    - Comparing engine outputs and plotting residual differences.

# %% [markdown]
# ## 1. Parsing the Living Knowledge Base
# 
# Let's inspect the current state of GammaForge programmatically from the repository's
# authority indexes: `docs/decisions/INDEX.md` and `docs/derivations/INDEX.md`. GitHub issues own unfinished work.

# %%
import re
from pathlib import Path

# Locate repo root
repo_root = Path.cwd()
if not (repo_root / "docs").exists():
    repo_root = repo_root.parent

decisions_index_file = repo_root / "docs" / "decisions" / "INDEX.md"
derivations_index_file = repo_root / "docs" / "derivations" / "INDEX.md"

def parse_markdown_table(file_path: Path) -> list[dict]:
    """Parse a GitHub markdown table into a list of dictionaries."""
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    table_lines = [l.strip() for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []
    headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
    rows = []
    for line in table_lines[2:]:  # skip header separator
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows

decisions = parse_markdown_table(decisions_index_file)
derivations = parse_markdown_table(derivations_index_file)

print(f"Total architecture decisions indexed: {len(decisions)}")
print(f"Total physics derivations indexed:   {len(derivations)}")

# %% [markdown]
# ### Where to find active work and measured evidence
# GitHub issues own unfinished work. Dated validation reports state what was
# measured and the limits of each result; their existence does not imply full
# scientific acceptance.

# %%
validation_reports = sorted((repo_root / "docs" / "validation").glob("*.md"))
print("Open work: https://github.com/blendersamsonov/GammaForge/issues")
print(f"Validation reports in this checkout: {len(validation_reports)}")
for report in validation_reports:
    print(" ", report.relative_to(repo_root))

# %% [markdown]
# ## 2. Interactive Decision & Derivation Search
# 
# Use this query function to quickly catch up on any topic across all decisions and derivations!

# %%
def search_knowledge_base(query: str, full_text: bool = True):
    """Search decisions, derivations, and open threads for a keyword."""
    q = query.lower()
    print(f"=== Search Results for: '{query}' ===\n")
    
    # Decisions
    matching_dec = []
    for d in decisions:
        match = q in d.get("title", "").lower() or q in d.get("id", "").lower() or q in d.get("type", "").lower()
        if not match and full_text and "path" in d:
            doc_file = repo_root / "docs" / "decisions" / d["path"]
            if doc_file.exists() and q in doc_file.read_text(encoding="utf-8").lower():
                match = True
        if match:
            matching_dec.append(d)

    print(f"--- Decisions ({len(matching_dec)} matches) ---")
    for d in matching_dec:
        print(f"  [{d['id']}] ({d['type']} - {d['status']}) {d['title']}")
        
    # Derivations
    matching_der = []
    for d in derivations:
        match = q in d.get("title", "").lower() or q in d.get("id", "").lower()
        if not match and full_text and "path" in d:
            doc_file = repo_root / "docs" / "derivations" / d["path"]
            if doc_file.exists() and q in doc_file.read_text(encoding="utf-8").lower():
                match = True
        if match:
            matching_der.append(d)

    print(f"\n--- Derivations ({len(matching_der)} matches) ---")
    for d in matching_der:
        print(f"  [{d['id']}] ({d['status']}) {d['title']}")
    print()

# Example searches:
search_knowledge_base("polarization")
search_knowledge_base("cupy")

# %% [markdown]
# ## 3. Active Blockers & Known Gaps Deep Dive
# 
# Independent fixed-direction xigma/delta spectra now run through
# `python -m gammaforge.validation.run --production` (RES084). The matrix covers
# head-on linear, crossed elliptical and crossed circular polarization, on/off axis.
# Counts, spectral L1 and centroid use matched finite energy bins without rescaling.
# Energy quadrature, angular-table and retarget refinement have separate gates.
# The initial matrix uses five-axis tables, angular 32/64 bins and retarget 512/1024
# bins. A scenario failing angular-table or retarget refinement is measured once more
# with angular 64/128 bins and retarget 1024/2048 bins (RES092). Initial failures stay
# visible as diagnostics; the complete finer matrix must pass the unchanged budgets.
# The full 2026-09-29 run passes all 388 executed checks; two scientific coverage
# blockers remain. See `docs/validation/delta-production-2026-09-29.md`.
#
# These provisional numerical checks share Stage-0 samples and reduced assumptions.
# Particle/seed, Stage-0, gamma/shape-grid, angular-aperture and independent CUDA
# convergence remain open, as does four-method coverage. Production therefore exits
# nonzero while these blockers remain, even when all numerical checks pass.
# The alpha selector retains its restricted analytical/xigma checks and runtime.

# %% [markdown]
# ## 4. The Validation Scenario Bank in Action
# The matched-bin delta pilot now explicitly uses the direction-Doppler mode (beta=1).
# Its historical nominal and exact finite-speed modes remain diagnostics. DER013 is
# author-verified (2026-09-13). The current nonlinear resonance stores raw `ahat` and
# evaluates the observer-dependent coefficient
# `Q = (1 - n dot n0) / (1 - e dot n0)` at query time. Broader independent arbitrary-angle
# acceptance remains open.
# 
# The Stage-0/1 CUDA agreement gate is `scripts/validate_cupy_stages01.py` (RES083).
# It iterates the shared scenario bank and reports agreement separately from warm
# timing; direct device tests additionally observe CuPy arrays inside both stages.
# The independent GPU delta reference is `gammaforge.validation.references.delta_cupy`
# (RES085). It accepts NumPy trajectory samples, computes float64 emission and weighted
# histograms on CUDA in bounded chunks, and returns NumPy arrays. Its per-particle line
# energy includes the trajectory's carrier mean. Its standalone gate is
# `scripts/validate_delta_cupy.py`; it checks against the long-double CPU reference.
# The xigma release gate also compares the reconstructed `moment2` CUDA spectrum with
# a refined CPU quadrature on a case where finite-line terms visibly change the result.
# The second-order expansion can be signed, so the gate checks finite values and
# quantitative agreement without treating every negative bin as a CUDA error.
# The release command logs each CPU/CUDA case to stderr and records stage timings
# in its JSON report. Numerical agreement does not close scientific acceptance.
# Select `doppler="direction"` for current xigma. Numerical backend agreement does not
# close the broader particle, Stage-0, or arbitrary-angle convergence requirements.
#
# Validation in GammaForge lives in `gammaforge.validation`:
# - `gammaforge.validation.scenarios.SCENARIOS`: A curated bank of physical scenarios (`BASELINE`, `LOW_A0`, `NEAR_A0_MAX`).
# - A `Scenario` contains **physics only**: beam, laser, target.
# - Engine parameters live in the engine's schema, never on the scenario (`Principle P5`).
# - Every maintained validation leg is reproducible from this checkout: closed-form
#   identities, analytical/xigma comparisons, independent CPU/CUDA delta emission,
#   convergence, and invariance checks (RES087).
# 
# Let's load the `BASELINE` scenario and run both `XigmaEngine` and `AnalyticalEngine` on it with a fast toy sampling!

# %%
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from dataclasses import replace
from gammaforge.validation.scenarios import BASELINE, build
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.target import Target, OutputKind, OutputRequest
from gammaforge.io.results import Axis
from gammaforge.io.units import EV_CGS
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.analytical.engine import AnalyticalEngine

print(f"Scenario: '{BASELINE.name}'")
print(f"  Electron energy: {BASELINE.beam.m('kinetic_energy') / 1.602e-6:.1f} MeV")
print(f"  Laser wavelength: {BASELINE.laser.m('wavelength') * 1e7:.1f} nm")
print(f"  Laser peak a0: {BASELINE.laser.a0_peak():.3f}")

# Build a fast interaction instance (100 particles, yield + spectrum for instant <50ms execution)
fast_target = replace(
    BASELINE.target,
    outputs=(
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(100,)),
    ),
)
fast_baseline = replace(BASELINE, target=fast_target)
toy_sampling = SamplingSpec(n_particles=100, seed=42)
interaction = build(fast_baseline, sampling=toy_sampling)

# Run Xigma Engine
xigma = XigmaEngine()
t0 = time.perf_counter()
xigma_res = xigma.run(interaction, xigma.schema.with_values(backend="numpy"))
t_xigma = (time.perf_counter() - t0) * 1000

# Run Analytical Engine
analytical = AnalyticalEngine()
t0 = time.perf_counter()
ana_res = analytical.run(interaction, analytical.schema)
t_ana = (time.perf_counter() - t0) * 1000

y_xigma = xigma_res.photon_slices[OutputKind.TOTAL_YIELD].integrate()
y_ana = ana_res.photon_slices[OutputKind.TOTAL_YIELD].integrate()
rel_diff = abs(y_xigma - y_ana) / y_ana * 100

print(f"\nBenchmark Results on BASELINE:")
print(f"  Xigma time:      {t_xigma:.2f} ms | Yield: {y_xigma:,.1f}")
print(f"  Analytical time: {t_ana:.2f} ms | Yield: {y_ana:,.1f}")
print(f"  Relative difference: {rel_diff:.2f}% (Consistent with 100-particle sampling)")

# %% [markdown]
# ### Independent CPU/GPU delta emission lines
# The same Stage-0 samples let us check emission arithmetic separately from table
# interpolation. Energies are in erg; histogram densities are per erg. The GPU call
# requires actual CUDA and never falls back silently. Skip explicitly when unavailable.

# %%
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.validation.references import delta_emission, delta_cupy
import numpy as np

delta_samples = integrate_trajectories(interaction.bunch, interaction.laser,
                                      interaction.N_e, n_steps=32, backend="numpy")
delta_kwargs = dict(photon_energy=float(interaction.laser.photon_energy()), doppler="direction")
cpu_energy, cpu_weight = delta_emission.emission_lines(delta_samples, 0., 0., **delta_kwargs)
if is_gpu_available():
    gpu_energy, gpu_weight = delta_cupy.emission_lines(delta_samples, 0., 0., chunk=17, **delta_kwargs)
    np.testing.assert_allclose(gpu_energy, cpu_energy, rtol=3e-13)
    np.testing.assert_allclose(gpu_weight, cpu_weight, rtol=3e-8, atol=1e-12*float(cpu_weight.max()))
    edges = np.geomspace(float(cpu_energy.min())*.9, float(cpu_energy.max())*1.1, 17)
    gpu_bins = delta_cupy.bin_emission(gpu_energy, gpu_weight, edges)
    cpu_bins = delta_emission.bin_emission(cpu_energy, cpu_weight, edges)
    np.testing.assert_allclose(gpu_bins["bin_mass"], cpu_bins["bin_mass"],
                               rtol=3e-8, atol=1e-12*float(cpu_weight.sum()))
    print("Independent CPU/GPU delta line and finite-bin agreement passed.")
else:
    print("CUDA unavailable: independent GPU delta example explicitly skipped.")

# %% [markdown]
# ### Visualizing the Baseline Photon Spectrum Comparison
# Let's plot both the Matplotlib publication-style curve and the interactive Plotly overlay.

# %%
spec_xigma = xigma_res.photon_slices[OutputKind.SPECTRUM]
spec_ana = ana_res.photon_slices[OutputKind.SPECTRUM]

# Convert energy from erg to MeV
e_mev_xigma = spec_xigma.axes[Axis.ENERGY] / (1e6 * EV_CGS)
e_mev_ana = spec_ana.axes[Axis.ENERGY] / (1e6 * EV_CGS)

# Convert spectral density dN/dE to photons / MeV
dn_de_xigma = spec_xigma.distr * (1e6 * EV_CGS)
dn_de_ana = spec_ana.distr * (1e6 * EV_CGS)

# Matplotlib Figure
plt.figure(figsize=(8, 4.5))
plt.plot(e_mev_xigma, dn_de_xigma, label="XigmaEngine", color="#1f77b4", lw=2)
plt.plot(e_mev_ana, dn_de_ana, label="AnalyticalEngine", color="#d62728", linestyle="--", lw=2)
plt.xlabel("Photon Energy [MeV]")
plt.ylabel("Spectral Density dN/dE [photons / MeV]")
plt.title(f"BASELINE Scenario: Compton Spectrum Comparison ({BASELINE.name})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interactive Plotly Spectrum
# Inspect the peak and Compton edge cutoffs interactively:

# %%
fig_val = go.Figure()
fig_val.add_trace(go.Scatter(
    x=e_mev_xigma, y=dn_de_xigma,
    mode="lines", name="XigmaEngine",
    line=dict(color="#1f77b4", width=2.5),
    hovertemplate="E: %{x:.3f} MeV<br>dN/dE: %{y:.2e}<extra></extra>"
))
fig_val.add_trace(go.Scatter(
    x=e_mev_ana, y=dn_de_ana,
    mode="lines", name="AnalyticalEngine",
    line=dict(color="#d62728", width=2, dash="dash"),
    hovertemplate="E: %{x:.3f} MeV<br>dN/dE: %{y:.2e}<extra></extra>"
))
fig_val.update_layout(
    title=f"Interactive Validation: {BASELINE.name} Spectrum",
    xaxis_title="Photon Energy [MeV]",
    yaxis_title="Spectral Density dN/dE [photons / MeV]",
    template="plotly_white",
    hovermode="x unified",
    width=750,
    height=450,
)
fig_val.show()

# %% [markdown]
# ---
# ### Summary of Verification & Progress Rules:
# 1. **`RES` IDs are permanent addressing anchors**: Never renumbered, never deleted.
# 2. **Derivations must be verified**: Mathematical proofs back physics before landing.
# 3. **Blockers are explicitly flagged**: Known issues (like CuPy sampler) are documented and pinned rather than swept under the rug.
# 4. Use `search_knowledge_base("keyword")` any time you want to catch up on what agents decided while you were away!

# %% [markdown]
# ### Report Figure 1: delta and two xigma table coordinates
# `scripts/plot_report_figure1.py` measures direct GPU delta, production xigma's
# intensity coordinate, and the report-local effective-gamma proposal against a
# separately sampled matched-bin delta reference. The publication run uses the exact
# linear limit and a 5x5 observer grid. `RES086` and `docs/report-figures.md` document
# the timing boundary, precision differences, reference diagnostics, and why the
# effective-gamma competitor remains outside production routing.

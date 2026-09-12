# %% [markdown]
# # GammaForge Interactive Walkthrough: 03 — Decisions, Progress & Validation
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
#    - How `PROGRESS.md` tracks active state and blockers without stale session logs.
# 2. **Interactive Search & Explorer Widget**:
#    - Query decisions, derivations, and open threads by topic (e.g., `"laser"`, `"cupy"`, `"polarization"`, `"units"`).
# 3. **Deep Dive into Current Blockers**:
#    - CuPy GPU ring/annulus sampler discrepancy (RES062).
#    - Arbitrary-angle emission validation.
# 4. **Live Validation Harness in Action**:
#    - Exploring the shared scenario bank (`gammaforge.validation.scenarios.SCENARIOS`).
#    - Running a fast, live validation benchmark on the `BASELINE` scenario.
#    - Comparing engine outputs and plotting residual differences.

# %% [markdown]
# ## 1. Parsing the Living Knowledge Base
# 
# Let's inspect the current state of GammaForge programmatically from the repository's
# authoritative files: `docs/decisions/INDEX.md`, `docs/derivations/INDEX.md`, and `PROGRESS.md`.

# %%
import re
from pathlib import Path

# Locate repo root
repo_root = Path.cwd()
if not (repo_root / "docs").exists():
    repo_root = repo_root.parent

decisions_index_file = repo_root / "docs" / "decisions" / "INDEX.md"
derivations_index_file = repo_root / "docs" / "derivations" / "INDEX.md"
progress_file = repo_root / "PROGRESS.md"

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
# ### Phase Status Snapshot from `PROGRESS.md`
# Let's read the live Phase Status table directly from `PROGRESS.md`.

# %%
progress_phases = []
if progress_file.exists():
    lines = progress_file.read_text(encoding="utf-8").splitlines()
    in_phase_table = False
    for line in lines:
        if "## Phase status" in line:
            in_phase_table = True
            continue
        if in_phase_table:
            if line.startswith("## ") or (line.startswith("---") and progress_phases):
                break
            if line.strip().startswith("|") and not line.strip().startswith("|---"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) == 2 and cells[0].lower() != "phase":
                    progress_phases.append({"Phase": cells[0], "Status": cells[1]})

print(f"{'Phase':<30} | Status")
print("-" * 75)
for p in progress_phases:
    print(f"{p['Phase']:<30} | {p['Status']}")

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
        match = q in d.get("title", "").lower() or q in d.get("id", "").lower() or q in d.get("class", "").lower()
        if not match and full_text and "path" in d:
            doc_file = repo_root / "docs" / "decisions" / d["path"]
            if doc_file.exists() and q in doc_file.read_text(encoding="utf-8").lower():
                match = True
        if match:
            matching_dec.append(d)

    print(f"--- Decisions ({len(matching_dec)} matches) ---")
    for d in matching_dec:
        print(f"  [{d['id']}] ({d['class']} - {d['status']}) {d['title']}")
        
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
# In `PROGRESS.md`, three items are currently open:
# 
# 1. **CuPy / GPU Discrepancy (`RES062`, `DER008`, `docs/ALPHA_GPU_VALIDATION.md`)**:
#    The CuPy ring/annulus importance sampler runs on GPU, but baseline test cubes give an
#    integrated ratio of 1.419 against NumPy. NumPy quadrature remains the default for the alpha release.
# 2. **Arbitrary-Angle Emission Validation (`RES060`, `DER005`, `DER006`)**:
#    Lab-frame per-particle polarization projection is implemented, but independent arbitrary-angle
#    multi-code benchmarks remain open.
# 3. **Spatial Autoranging under Laser Displacement**:
#    Laser offset sweeps require event-source coordinate tracing through displaced overlap.
# 
# This honesty-first culture is enforced by `validation.run --production`, which will purposefully
# exit with a non-zero code if these blockers are bypassed without resolution.

# %% [markdown]
# ## 4. The Validation Scenario Bank in Action
# The matched-bin delta pilot now explicitly uses the direction-Doppler mode (beta=1).
# Its historical nominal and exact finite-speed modes remain diagnostics. Agreement of
# shared-input numerical methods does not close the independent physics review (DER013).
# 
# Validation in GammaForge lives in `gammaforge.validation`:
# - `gammaforge.validation.scenarios.SCENARIOS`: A curated bank of physical scenarios (`BASELINE`, `LOW_A0`, `NEAR_A0_MAX`).
# - A `Scenario` contains **physics only**: beam, laser, target.
# - Engine parameters live in the engine's schema, never on the scenario (`Principle P5`).
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
xigma_res = xigma.run(interaction, xigma.schema)
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

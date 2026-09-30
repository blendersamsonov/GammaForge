# GammaForge Interactive Walkthrough Notebooks

These modular notebooks are designed specifically for **human developers** to learn, explore, and retain a clear mental model of the entire GammaForge codebase.

## Overview of Notebooks

| Notebook | Focus & Key Concepts |
| :--- | :--- |
| **[`01_units_io_and_contracts`](01_units_io_and_contracts.ipynb)** | **Core data contracts, Pint boundary, & I/O.** Explains why CGS-Gaussian is the canonical unit system, why Pint is used at the boundary but stripped in kernels, how `GaussianElectronBeam` and `Bunch` guarantee the mass shell $p^2 + 1 = \gamma^2$, how `LaserField` / `GaussianParaxialLaser` protocol works, and provides interactive 2D & 3D phase-space plots (Matplotlib & Plotly). |
| **[`02_engine_stages_and_dataflow`](02_engine_stages_and_dataflow.ipynb)** | **Engine architectures & computational stages.** Compares `AnalyticalEngine` vs `XigmaEngine`, explains why engines are stateless with typed schemas (no mutable Config), walks through Xigma's Stages 0, 1, 1.5, and 2 via the `Collision` facade, inspects intermediate arrays, runs fast side-by-side benchmarks (<10ms), and plots overlaid photon energy spectra. |
| **[`03_decisions_progress_and_validation`](03_decisions_progress_and_validation.ipynb)** | **Decision and derivation indexes, issues, & validation.** Bridges the gap between fast-moving agent development and the human architect. Dynamically parses the `RESNNN` and `DERNNN` indexes, points to GitHub issues, and lists validation reports. Includes a live search function to query decisions by keyword (e.g. `laser`, `cupy`, `polarization`), explains measured coverage limits, and runs validation benchmarks against `gammaforge.validation.scenarios.BASELINE`. |

---

## Dual-Format Architecture: `.py` and `.ipynb`

To solve the friction where AI agents struggle with editing large JSON `.ipynb` files and can introduce corruptions or merge conflicts:
- Each notebook is authored as a standard Python script with `# %%` cell markers (e.g., `01_units_io_and_contracts.py`).
- VS Code can open `.py` files with `# %%` markers natively in the **Interactive Window** or run individual cells directly.
- The companion `.ipynb` files are built automatically using standard Python stdlib JSON via `tools/build_notebooks.py`.

### How to Rebuild Notebooks
Whenever you or an agent modifies any of the `.py` scripts:
```bash
python tools/build_notebooks.py
```
To build and verify that all notebooks execute cleanly from start to finish:
```bash
python tools/build_notebooks.py --run
```

---

## Rule for Autonomous Agents

As documented in [`AGENTS.md`](../AGENTS.md):
> Whenever modifying core data structures (`gammaforge.io`), engine interfaces (`gammaforge.engines`), or validation pipelines, agents **must** verify and update the corresponding walkthrough script in `notebooks/` and re-run `python tools/build_notebooks.py --run` so the human mental model is never broken.

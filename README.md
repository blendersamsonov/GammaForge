# GammaForge

Inverse-Compton scattering simulation toolkit: computes properties of Compton
photons produced by the interaction of an electron bunch with a laser pulse.

This is a **ground-up rebuild** of the predecessor framework. The architecture and
phase plan live in [docs/GRAND_PLAN.md](docs/GRAND_PLAN.md); the browser-workspace
contract is in [docs/UI_SPEC.md](docs/UI_SPEC.md).

## Status

- **0.1.0a1: script-first alpha** for Gaussian calculations with analytical and xigma.
  Start with the [alpha guide](docs/ALPHA.md) and the
  [crossing-angle yield example](examples/crossing_angle_yield.py).
- NumPy is the alpha default. CuPy supports incident polarization and crossing
  angles with numerical checks, but remains [experimental](docs/ALPHA_GPU_VALIDATION.md).
- GUI and kascade remain available for development, outside alpha release support.

```bash
python -m pip install -e .
python -m gammaforge.validation.run --alpha
python examples/crossing_angle_yield.py
```

## Layout

```
src/gammaforge/
├── io/            # shared CGS-Gaussian physics core (schema, beam, laser, target, results)
├── engines/       # xigma, analytical, and the minimal validation-only kascade port
├── validation/    # cross-engine suite + old-repo golden references
└── gui/           # optional local NiceGUI browser workspace
```

## Development setup

Core development and tests require no browser dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
make check PYTHON=.venv/bin/python
```

The reusable library ranges live in `pyproject.toml`.
`requirements/developer.lock` pins the recorded CPython 3.14/Linux developer set for
the core, GUI, browser, and symbolic tiers. To reproduce it, first install the local
project without resolving dependencies, then install the pinned set:

```bash
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install -r requirements/developer.lock
```

The lock is intentionally not a cross-platform promise. Regenerate it after dependency
changes in a clean virtual environment with:

```bash
.venv/bin/python -m pip install -e '.[dev,gui,browser,symbolic]'
.venv/bin/python -m pip freeze --exclude-editable > requirements/developer.lock
```

CI separately exercises the declared lower bounds on Python 3.12 and the current
dependency ranges on Python 3.14.

## Local browser workspace

Install the optional GUI dependencies, then start a loopback server:

```bash
.venv/bin/python -m pip install -e '.[dev,gui]'
.venv/bin/python -m gammaforge.gui
```

For this checkout's virtual environment and a fixed port:

```bash
.venv/bin/python -m gammaforge.gui --port 8090
```

The workspace has Inputs and Results tabs, plus an optional split view. Inputs use
equal-height Electron, Laser, and Geometry columns, followed by Target/outputs,
analytical estimates, and engine settings. Calculate is gated on valid schema inputs
and at least one selected calculation engine; analytical estimates remain a preview,
not a substitute for a calculation. Xigma is selected initially; kascade is available
as an opt-in engine tab.

Each browser page owns an in-memory workspace. Refreshing the page starts a new
workspace; restart the server to load code edits. Results can be downloaded as HDF5,
and displayed plots as PNG or PDF. A completed calculation also offers an input-snapshot
ZIP: `inputs.yaml` uses the standard input loader format, while `calculation.yaml`
records the target and selected engine settings.

This is a local-only interface. LAN execution, remote workers, and persistence across
browser refreshes are deferred.

## Tests

`make check` (or bare `pytest`) is the local counterpart to CI's core tier, running Tier 0, Tier 1, and Tier 2 tests in ~1.2 minutes. Execution tiers and optional suites are explicit (RES075):

```bash
# Fast iterative development loop (< 30s): Tier 0 contracts + Tier 1 component physics
.venv/bin/python -m pytest -m fast -q

# Ultra-fast contract, schema, unit, and doc lint checks (< 10s)
.venv/bin/python -m pytest --tier=tier0 -q

# Default core run (~1.2m): Tier 0, Tier 1, and Tier 2 (numerical integration, trajectory tracking)
make check

# Full physics validation (~11m): includes heavy Tier 3 Monte Carlo and 16x quadratures
.venv/bin/python -m pytest --run-heavy

# GUI plotting and import-boundary tests
.venv/bin/python -m pytest -q tests/test_gui_boundary.py tests/test_gui_controller.py tests/test_gui_inputs.py tests/test_gui_plotting.py

# Symbolic derivation checks
.venv/bin/python -m pip install -e '.[dev,symbolic]'
.venv/bin/python -m pytest -q tests/test_verifications.py
.venv/bin/python scripts/verifications/verify_all.py der004
```

The browser tier also needs Playwright's separate browser binary installation:

```bash
.venv/bin/python -m pip install -e '.[dev,gui,browser]'
.venv/bin/python -m playwright install chromium
GAMMAFORGE_BROWSER_TEST=1 .venv/bin/python -m pytest -q tests/test_gui_browser.py
```

The `gpu` extra installs the experimental CuPy angular
sampler. It is opt-in and requires convergence checks for new calculations; see the
alpha guide before use. The `jit` extra remains a dependency placeholder for an
unimplemented backend.

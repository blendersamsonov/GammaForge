# GammaForge

Inverse-Compton scattering simulation toolkit: computes properties of Compton
photons produced by the interaction of an electron bunch with a laser pulse.

GammaForge originated as a ground-up rebuild of ComptonSuite and is now completely
independent: no external checkout is needed to build, test, validate, or use it.
Physics, implementation choices and scientific evidence are recorded in the
[derivation index](docs/derivations/INDEX.md),
[decision index](docs/decisions/INDEX.md) and [validation records](docs/validation/README.md).
Open work is tracked in [GitHub issues](https://github.com/blendersamsonov/GammaForge/issues).

## Status

- **0.1.0a1: script-first alpha** for Gaussian calculations with analytical and xigma.
  Start with the [crossing-angle yield example](examples/crossing_angle_yield.py)
  or the [walkthrough notebooks](notebooks/README.md).
- Xigma's schema defaults to CuPy. On a CPU-only installation, explicitly set
  `backend="numpy"` or choose `backend="auto"` for fallback. The optional `gpu`
  extra installs CuPy; CUDA calculations require their own convergence checks.
  The [GPU measurement record](docs/validation/alpha-gpu-sampler-2026-09.md)
  and [current release gate](docs/validation/cupy-release-2026-09-28.md)
  state what has been checked. Independent arbitrary-angle scientific acceptance
  remains open.
- GUI and kascade remain available for development, outside alpha release support.

```bash
python -m pip install -e .
python -m gammaforge.validation.run --alpha
python examples/crossing_angle_yield.py
```

The alpha selector checks reduced analytical/xigma scenarios and core invariants.
The production selector adds independent xigma/delta emission comparisons and
reports unmeasured scientific coverage as blockers. It can exit nonzero even
when every executed numerical comparison passes. Xigma's `SPECTRUM` is the
table-free linear approximation; angular and collimated spectra use the
tabulated nonlinear model. Returned energy axes use erg and angular axes rad;
the shared inputs use pint quantities stored in CGS-Gaussian units.

## Layout

```
src/gammaforge/
├── io/            # shared CGS-Gaussian physics core (schema, beam, laser, target, results)
├── engines/       # xigma, analytical, and the minimal validation-only kascade port
├── validation/    # identities, invariance, convergence, and cross-engine checks
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

To check the declared lower bounds, install `requirements/minimum.txt` in a clean
Python 3.12 environment and run `make check`.

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

Each input section has its own **Save as default** action, including electrons,
sampling, laser, geometry, target, requested outputs, and each engine. Defaults are
stored in `~/.config/gammaforge/gui-defaults.yaml` (or under `$XDG_CONFIG_HOME`, and
`$GAMMAFORGE_CONFIG_DIR` can override the directory). Output axes show their current
Auto bounds at all times; turn off Auto for one axis to enter a manual Min/Max without
changing the other axes. This is especially useful for narrowing the energy grid of a
collimated spectrum below its conservative zero-to-Compton-edge Auto interval.

Each browser page owns an in-memory workspace. Refreshing the page starts a new
workspace; restart the server to load code edits. Results can be downloaded as HDF5,
and displayed plots as PNG or PDF. A completed calculation also offers an input-snapshot
ZIP: `inputs.yaml` uses the standard input loader format, while `calculation.yaml`
records the target and selected engine settings.

This is a local-only interface. Saved panel defaults survive browser/server restarts;
run history and the rest of the workspace do not. LAN execution and remote workers are
deferred.

## Tests

`make check` (or bare `pytest`) runs the default Tier 0–2 suite. Execution tiers
and optional suites are explicit (RES075):

```bash
# Fast iterative development loop: Tier 0 contracts + Tier 1 component physics
.venv/bin/python -m pytest -m fast -q

# Contract, schema, unit, and format checks
.venv/bin/python -m pytest --tier=tier0 -q

# Default core run: Tier 0, Tier 1, and Tier 2
make check

# Full physics validation: includes heavy Tier 3 tests
.venv/bin/python -m pytest --run-heavy

# Available GUI estimate tests
.venv/bin/python -m pytest -q tests/test_gui_estimates.py

# Symbolic derivation checks
.venv/bin/python -m pip install -e '.[dev,symbolic]'
.venv/bin/python scripts/verifications/verify_all.py der004
```

On a CPU-only machine, tests that invoke Xigma with its CuPy default require
an explicit NumPy setting in their test inputs. The CUDA gate requires actual
hardware and is separate from the default suite.

The `gpu` extra installs the CuPy backend. Run
`python scripts/validate_cupy_release.py --output output/validation/cupy-release.json`
on an actual CUDA device when assessing a new backend configuration. The `jit`
extra remains a dependency placeholder for an unimplemented backend.

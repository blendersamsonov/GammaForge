# GammaForge

Inverse-Compton scattering simulation toolkit: computes properties of Compton
photons produced by the interaction of an electron bunch with a laser pulse.

This is a **ground-up rebuild** of the predecessor framework. The architecture and
phase plan live in [docs/GRAND_PLAN.md](docs/GRAND_PLAN.md); the browser-workspace
contract is in [docs/UI_SPEC.md](docs/UI_SPEC.md).

## Status

- Shared CGS-Gaussian inputs, xigma, and the analytical engine are available.
- The optional local NiceGUI workspace runs those concrete engines; kascade remains
  unimplemented.

## Layout

```
src/gammaforge/
├── io/            # shared CGS-Gaussian physics core (schema, beam, laser, target, results)
├── engines/       # xigma and analytical; kascade is deferred
├── validation/    # cross-engine suite + old-repo golden references
└── gui/           # optional local NiceGUI browser workspace
```

## Dev install

```bash
pip install -e .
pytest
```

## Local browser workspace

Install the optional browser dependencies, then start a loopback server:

```bash
pip install -e '.[gui]'
python -m gammaforge.gui
```

For this checkout's virtual environment and a fixed port:

```bash
.venv/bin/python -m gammaforge.gui --port 8090
```

The workspace has Inputs and Results tabs, plus an optional split view. Inputs use
equal-height Electron, Laser, and Geometry columns, followed by Target/outputs,
analytical estimates, and engine settings. Calculate is gated on valid schema inputs
and at least one selected calculation engine; analytical estimates remain a preview,
not a substitute for a calculation.

Each browser page owns an in-memory workspace. Refreshing the page starts a new
workspace; restart the server to load code edits. Results can be downloaded as HDF5,
and displayed plots as PNG or PDF. A completed calculation also offers an input-snapshot
ZIP: `inputs.yaml` uses the standard input loader format, while `calculation.yaml`
records the target and selected engine settings.

This is a local-only interface. LAN execution, remote workers, and persistence across
browser refreshes are deferred.

## Tests

```bash
.venv/bin/python -m pytest
```

An optional browser test can be run where Playwright and a system Chromium are
available:

```bash
GAMMAFORGE_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_gui_browser.py
```

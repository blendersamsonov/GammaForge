# GammaForge

Inverse-Compton scattering simulation toolkit: computes properties of Compton
photons produced by the interaction of an electron bunch with a laser pulse.

This is a **ground-up rebuild** of the predecessor framework. It is under
active design — start with [docs/GRAND_PLAN.md](docs/GRAND_PLAN.md).

## Status

- Scaffold / design phase. No usable code yet.

## Layout (planned)

```
src/gammaforge/
├── io/            # shared CGS-Gaussian physics core (schema, beam, laser, target, results)
├── engines/       # xigma (first-class), analytical (first-class), kascade (minimal port)
├── validation/    # cross-engine suite + old-repo golden references
└── gui/           # thin schema-driven Tkinter GUI
```

## Dev install

```bash
pip install -e .
pytest
```

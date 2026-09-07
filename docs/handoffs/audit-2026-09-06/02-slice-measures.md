# 02 — Preserve histogram mass and define slice integration

Read [coordination rules](README.md). Own A03, the axis/output-resolution part of
A14, and investigation of A17's spatial range. This is the prerequisite contract
for 05's persistence and the result-related part of 06.

## Entry points and reproduction

`io/results.py` (`PhasespaceSlice.integrate`, `scaled`), `io/target.py`
(`OutputRequest`, `slice_axis_values`, `auto_ranges`), `io/plotting.py`,
`gui/outputs.py`, and `engines/kascade/engine.py::_histogram_slice`, all under
`src/gammaforge/`. Inspect analytical/xigma slice producers too. Tests:
`test_target_results_interaction.py`, `test_gui_plotting.py`, `test_kascade_engine.py`,
`test_formats.py`, `test_analytical.py`.

```python
import numpy as np
from gammaforge.io.results import Axis, PhasespaceSlice
centers = np.array([0.5, 1.5])  # two bins of width 1
print(PhasespaceSlice({Axis.ENERGY: centers}, np.ones(2)).integrate())
axes = {a: centers for a in (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y)}
print(PhasespaceSlice(axes, np.ones((2, 2, 2))).integrate())
# Baseline: 1.0, 1.0. Histogram masses represented here: 2.0, 8.0.
```

Histograms store counts divided by bin volume; trapezoids over centers omit half
the endpoint bins. Smooth point samples and cell-average densities cannot acquire
the same integration meaning merely by sharing an ndarray type.

## Work

1. Document the minimal explicit measure needed by the single slice contract
   (edges, widths or quadrature weights), including point-sampled producers,
   one-bin axes, nonuniform bins, axis ordering, scaling and projected slices.
   Update GRAND_PLAN §3.6 first. Do not add a generic sampled/binned class hierarchy.
2. Implement that contract and update producers/projections together. Existing
   smooth quadrature must not silently change into midpoint integration.
   Coordinate a minimal HDF5 compatibility update with 05 so this landing does
   not make current save/load unusable while the full persistence task waits.
3. Validate finite, ordered one-dimensional axes/measure arrays and genuinely
   integral positive resolutions at construction. Reject malformed ranges early.
   Support a one-bin histogram when its width is explicit; reject an undefined
   point-sample integral clearly. Do not simply force all bins to have ≥2 samples.
4. Investigate spatial autoranging with displaced beam/laser overlap. First make
   a reproducible clipping case and quantify it. Fix only an established geometry
   error consistent with the plan; otherwise leave a precise follow-up, not a
   guessed Gaussian formula or a new user-facing range policy.

## Acceptance

- Weighted histograms integrate to the weights actually captured by their bins in
  1D/2D/3D, including edge bins, unequal widths and one-bin axes.
- Every projection followed by integration agrees with direct integration of the
  original slice, with correct axis order/units and non-unit charge scaling.
- Existing smooth-function quadrature has its documented convergence behavior.
  Include an analytic nonconstant fixture; constant arrays alone are insufficient.
- Invalid descending/duplicate/non-finite coordinates, invalid widths and
  fractional resolutions fail at the boundary. Empty results follow an explicit
  policy coordinated with 03.
- Round-trip tests preserve the new measure; legacy files have explicit semantics
  rather than invented edges. Publish the contract for 05 and 06.

Kascade's deterministic expected yield and a finite MC realization are different
quantities. Never rescale sampled photons to force their histogram to that expected
yield. Do not touch the emission kernel, event RNG or final-electron type here.

Starting check: `.venv/bin/pytest -q tests/test_target_results_interaction.py tests/test_gui_plotting.py tests/test_kascade_engine.py tests/test_formats.py tests/test_analytical.py`.
GUI plotting needs its optional extra.

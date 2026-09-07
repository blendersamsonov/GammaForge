# GammaForge 0.1.0a1 — calculations from Python

This alpha supports Gaussian-beam calculations with the analytical and xigma engines
through their Python APIs. GUI and kascade are outside release support. The library
defaults to NumPy. The merged CuPy angular sampler is experimental: measured disagreement
with the reference prevents treating it as a validated production backend. See
[the GPU validation record](ALPHA_GPU_VALIDATION.md).

## Install and check

From a checkout, with Python 3.12 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m gammaforge.validation.run --alpha
.venv/bin/python examples/crossing_angle_yield.py
```

For another project's environment, install this checkout with
`python -m pip install /path/to/GammaForge`, or install the built wheel from `dist/`.
GUI packages and CUDA are unnecessary. The alpha is prepared locally; these instructions
do not assume a published PyPI release.

The minimum environment uses NumPy 2.0, Pint 0.25, h5py 3.11, Matplotlib 3.8.4
and PyYAML 6.0.1. Pint 0.25 supplies the CODATA 2022 table used by the reference
checks; older Pint versions change the constants enough to fail those tight gates.
See the [Pint release notes](https://pint.readthedocs.io/en/0.25/changes.html).

`--alpha` checks the shared scenario bank's analytical/xigma yields and common head-on
spectral regime, plus core invariants, identities and scalar goldens. Numerical failures
are fatal. `--production` additionally treats unfinished independent angular-emission
and four-method coverage as blockers and currently exits nonzero. A passing alpha check
does not certify arbitrary-angle emission.

## A complete calculation

Widths below are intensity RMS; emittances are geometric, not normalized. Inputs accept
Pint quantities and are stored in CGS. Returned energy axes are in erg and angles in rad.
`theta_xz=0` means head-on; use `dataclasses.replace(laser, theta_xz=Q(20, "mrad"))`
for a crossing-angle scan and rebuild the interaction for each changed geometry.

```python
from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io import (
    GaussianElectronBeam, GaussianParaxialLaser, Target,
    SamplingSpec, OutputKind, OutputRequest, build_interaction,
)
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.formats.hdf5 import save_results, load_results, load_request
from gammaforge.io.units import Quantity as Q

beam = GaussianElectronBeam(
    bunch_charge=Q(100, "pC"), kinetic_energy=Q(100, "MeV"),
    rel_energy_spread=0.005,
    sigma_x=Q(10, "um"), sigma_y=Q(10, "um"), sigma_z=Q(100, "um"),
    emit_x=Q(1e-7, "cm * rad"), emit_y=Q(1e-7, "cm * rad"),
)
laser = GaussianParaxialLaser(
    pulse_energy=Q(0.01, "J"), wavelength=Q(800, "nm"),
    sigma_x=Q(10, "um"), sigma_y=Q(10, "um"), duration=Q(1, "ps"),
)
target = Target(Q(1, "mrad"), Q(1, "mrad"), outputs=(
    OutputRequest(OutputKind.TOTAL_YIELD),
    OutputRequest(OutputKind.SPECTRUM, (128,)),
))
sampling = SamplingSpec(n_particles=20_000, seed=17, prefilter=1e-4)
settings = {
    "analytical": AnalyticalEngine.schema.with_values(n_quad_overlap=4001),
    "xigma": XigmaEngine.schema.with_values(n_steps=400, backend="numpy"),
}
request = CalculationRequest(beam, laser, target, sampling, settings)
interaction = build_interaction(beam, laser, target, sampling)

for name, engine in (("analytical", AnalyticalEngine()), ("xigma", XigmaEngine())):
    result = engine.run(interaction, settings[name])
    print(name, result.photon_slices[OutputKind.TOTAL_YIELD].integrate())
    print(result.model_specific.get("warnings", ()))
    save_results(result, f"{name}.h5", request=request)

restored = load_results("xigma.h5")
submitted = load_request("xigma.h5", {
    "analytical": AnalyticalEngine.schema, "xigma": XigmaEngine.schema,
})
```

An HDF5 file preserves slices, integration measures, warnings and numerical metadata;
the embedded request contains all beam/laser, target, sampling and engine settings.
Pass the actual submitted snapshot, not settings edited after the calculation.
Old files without a request remain readable but cannot supply missing provenance.
Metadata dataclasses load as field dictionaries; output keys load as `OutputKind`
members (`kind_from_name=str` retains the previous string-key behavior).

## Reproduce the crossing-angle figure

[The example](../examples/crossing_angle_yield.py) uses explicit inputs adapted from
the original XIGMA example configuration: gamma 2000, 10 nC, 20 J, 1030 nm and
10 micrometre RMS transverse sizes. It records the original parameter conventions and
their conversion in its provenance JSON. It imports no validation fixtures or GUI code.

The script produces [PNG](../examples/output/crossing_angle_yield/crossing_angle_yield.png)
and [PDF](../examples/output/crossing_angle_yield/crossing_angle_yield.pdf), absolute yields,
individual seed results and numerical convergence tables in
`examples/output/crossing_angle_yield/`. Re-running overwrites those example artifacts.

Across 0–40 mrad, the three-seed xigma mean differs from analytical by at most 0.092%.
Analytical yield falls from approximately 1.196e11 to 2.607e10 photons.
No result is normalized to the other engine. At 0/20/40 mrad, quadrature changes are
below 3e-8 for analytical and 1e-7 for xigma time integration. The fixed-seed
15k-to-30k particle check changes yield by up to 0.26%; the plotted SEM estimates
sampling uncertainty from three independent seeds. Scan points share those seeds and
therefore have correlated errors. This demonstrates total-overlap agreement, not
angle-resolved Stage-2 emission accuracy or a universal sub-0.1% error bound.

## Supported behavior and limits

| Calculation | Alpha behavior |
|---|---|
| Analytical total yield | General Gaussian overlap, including crossing angle, displacement, astigmatism and flying focus; refine quadrature for each new configuration |
| Xigma total yield | Stage-0 trajectory overlap; refine particle count, seeds, time steps and threshold |
| Analytical spectrum | Includes its luminosity-weighted nonlinear redshift; crossed-spectrum shape still uses head-on kinematics and reports a warning |
| Xigma `SPECTRUM` | Table-free linear spectrum; omits the nonlinear redshift and reports a warning |
| Xigma angular/collimated output | NumPy table quadrature by default; retains tabulated nonlinear physics, but independent arbitrary-angle emission validation remains open |
| CuPy | Explicit experimental `backend="cupy"` or `"auto"`; head-on linear polarization only, with recorded backend/settings and a warning |

Request collimated output with
`OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (n_energy, n_theta_x, n_theta_y))`.
The result density has axes `(ENERGY, THETA_X, THETA_Y)` and integrates using
`slice.integrate()`. Start with modest grids: CPU angular quadrature is expensive.
Installing `.[gpu]` only adds CUDA 12 CuPy dependencies; it does not change the default
or make the experimental sampler accurate. Numba remains unimplemented.

The historical `ahat_decades=1.0` default remains unchanged; the tracked roughly 1%
centroid-grid bias is still an author decision. The validation tier uses an explicit
0.3 override. Treat table-grid refinement as part of any new angular-spectrum study.

Use fresh `Engine.run()` calls as above for independent runs. A `Collision` may be reused
for queries of fixed inputs; replacing its inputs is prohibited, and cached trajectory
arrays are read-only. Shared bunch vectors and request containers now own their data.
To change particles, construct a new `Bunch`; do not mutate borrowed arrays.
No cross-run cache or arbitrary mutable-laser snapshot guarantee is introduced.

## Audit disposition

The alpha includes the slice-measure fix (02), bounded xigma spectra and empty-input
handling (03), supported result/request persistence (05), input/cache ownership fixes
(06), and headless setup/checks (08). Physics convention fixes and honest coverage
reporting from 01 are retained. Non-Gaussian laser work (07), GUI export integration,
kascade/four-method validation, cross-run caching and unrelated cleanup remain outside
this release. The CuPy numerical discrepancy is a GPU release blocker, not a reason to
withhold the independently checked NumPy total-yield workflow.

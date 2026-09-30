# Direction-Doppler integration validation — 2026-09-12

RES082 and DER013 apply beta=1 electron directions consistently to Stage-0 encounter
flux and Stage-2 resonance, Jacobian and CUDA support. This record checks implementation
and scoped numerical agreement. Subsequent author verification on 2026-09-13 promoted
DER013 to verified; RES074's full arbitrary-angle scientific acceptance remains open.

## Independent matched-bin pilot

The original run recorded environment, unchanged source SHA256 fingerprints,
direction convention, bin measures and settings. The measured result is below.
Reproduce from the repository root:

```sh
mkdir -p output/validation
PYTHONPATH=src .venv/bin/python scripts/validate_delta_emission.py --backend cupy --particles 16000 --n-steps 64 --table 32,32,32,64,512 --table 32,64,64,64,512 --quadrature-order 16 --rings 64 --subsampling 128 --output output/validation/delta-direction-doppler.json
```

The 24 records cover three shared-bank scenarios, head-on and small two-plane crossed
geometry, two observers and two angular table resolutions. All 24 pass their q16/q32
energy-bin quadrature convergence test. All are within the provisional comparison
budgets (3% yield, 5% spectral L1, 1% centroid). At the finer angular resolution,
maximum absolute relative errors over the four geometry/observer cases per scenario are:

| Scenario | Yield | Spectral L1 | Centroid |
|---|---|---|---|
| baseline | 0.5314% | 0.8230% | 0.0078% |
| low_a0 | 0.4785% | 3.0748% | 0.0616% |
| near_a0_max | 0.7652% | 1.0566% | 0.0246% |

The crossed case uses theta_xz=0.02, theta_yz=-0.015 radians, ellipticity=0.4 and
polarization angle=0.37 radians. These results establish agreement for those inputs;
this packet alone does not establish particle/seed, Stage-0, gamma or full angular
convergence. The raw pilot intentionally retains scientific_pass=false and its
unresolved-numerics caveat. Shared trajectory ahat remains a common-mode dependency.

## Regression coverage

`tests/test_xigma_doppler.py` checks the inverse-resonance derivative symbolically,
Stage-0 per-electron flux against an independent constant-field integral, NumPy photon
mass and centroid against independently evaluated gamma-quadrature lines, linear-spectrum
normalization and energy shift, and conservative angular support. The actual-CUDA
support stress test places all requested energies above the former nominal edge:
nonzero output agrees with NumPy within 5%. Its deliberately amplified angles test
algorithmic support, not validity of the physical approximation in that regime.

`tests/test_xigma_source_diagnostics.py` checks requested/nonuniform source grids,
chunk independence, retry-safe reduction, integrated mass, clipping, empty/displaced
interactions, pulse trains and cache ownership. Temporal outputs describe lab emission
time, and spatial outputs describe source x/y; neither is a detector-arrival prediction.

## Integration checks

The fast suite passed 488 tests. The full `pytest --run-heavy -q` sweep passed
801 tests with 1 skipped in 1012.83 seconds, including real-CUDA comparisons and
symbolic derivation checks. The paired walkthrough notebooks were rebuilt and
executed successfully with `PYTHONPATH=src .venv/bin/python tools/build_notebooks.py --run`.

The alpha command `PYTHONPATH=src .venv/bin/python -m gammaforge.validation.run --alpha`
passed all runnable checks. The final Tier-0 contract/documentation sweep passed 200
tests. Explicit PYTHONPATH selects this checkout when worktrees share an editable
installation; it avoids importing a different branch through the environment.

The real-CUDA release run passed
on an NVIDIA GeForce GTX 1660 Ti with CuPy 14.2.0. All required checks passed across
eight cases (baseline, crossed, low-a0, near-a0-max, wide/narrow off axis and gamma
10,000 circular/crossed). The run recorded 80 convergence diagnostics, including
coarser sampler settings that are diagnostic rather than required gates. Source
fingerprints before and after match. Reproduce with:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_cupy_release.py --output output/validation/cupy-release-direction-doppler.json
```

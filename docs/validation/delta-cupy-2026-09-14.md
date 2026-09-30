# CuPy delta reference validation — 2026-09-14

The GPU delta prototype from the spectrum-proposal worktree was not numerically
usable: a collinear electron had polarization factor zero instead of one. The
prototype also reversed scalar/vector outputs in the linear spectrum and lacked the
current direction-Doppler mode. The repaired reference implements the existing
DER012/DER013 formulas; no new physics convention is introduced (RES085).

## Reproduction and scope

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_delta_cupy.py -q
mkdir -p output/validation
PYTHONPATH=src .venv/bin/python scripts/validate_delta_cupy.py --output output/validation/delta-cupy.json
```

Actual CUDA is mandatory for the standalone gate. Pytest's GPU tests skip on a
CPU-only machine; such a run does not establish GPU validation. The commands use
the checkout explicitly because the local editable environment can point to another
worktree. The recorded run fingerprinted its sources before and after the
measurement and recorded its device and numerical environment.

The full repository run (`pytest --run-heavy -q`) passed 881 tests with one skipped
in 466.14 seconds on actual CUDA. All three walkthrough notebooks were rebuilt and
executed with `python tools/build_notebooks.py --run`; the new GPU delta example ran.
The final nonzero-underflow test refinement was also checked directly on CUDA.

The targeted run passed 56 tests on the NVIDIA GeForce GTX 1660 Ti with CuPy 14.2.0,
NumPy 2.5.1 and CUDA runtime 12.9. Checks include collinear and linear-spectrum
anchors, gamma 2/2000/10000, all three Doppler modes, crossed geometry, signed
ellipticity, circular polarization, independent extended-precision line agreement,
histogram edges and tails, zero and empty inputs, chunk invariance, invalid inputs,
finite-cone integration, and angular photon conservation. The emission and binning
path also runs with production and CPU emission helpers disabled.

The standalone gate passed all 72 cases: every shared scenario, two seeds, head-on
linear/crossed elliptical/crossed circular polarization, on/off axis, and chunk sizes
127 and 65536. Each case uses 4096 requested particles, 64 Stage-0 steps and 64
matched physical-energy bins. Stage 0 is shared and held fixed; its convergence is
not measured by this gate.

| Error versus independent long-double CPU lines | Maximum |
|---|---:|
| Relative resonant energy | 6.32e-16 |
| Absolute line-weight error divided by peak weight | 2.02e-12 |
| Relative total weight | 1.62e-12 |
| Integrated spectral L1 | 1.62e-12 |
| Normalized-spectrum wrapper integrated L1 | 1.62e-12 |

The energy tolerance is 3e-13; weight and spectral tolerances are 3e-8. These are
backend numerical tolerances, not scientific acceptance thresholds. Final-right-edge
inclusion, underflow/overflow and nonuniform-bin densities are checked separately
with explicit exact-edge inputs.

## Execution and remaining limits

The emission kernel is fused float64 arithmetic followed by weighted GPU histogramming.
Particle chunks bound temporary device arrays. Public trajectory data and outputs
remain NumPy. The finite-cone wrapper reuses a transferred chunk across its direction
loop. The physical-energy API mirrors the independent CPU reference; normalized
wrappers preserve nominal Doppler by default, so current xigma comparisons must
explicitly select the direction mode.

Measured warm normalized-spectrum calls took 0.64–15.13 ms across this packet, including
transfers and preparation; small chunks account for substantial overhead. The first
line-plus-bin call took 0.21 s and is recorded separately. Timings exclude Stage 0
and do not establish a speedup over xigma or over a CPU delta implementation. Existing
figures using the faulty prototype cannot be treated as validated comparisons.

The independent long-double CPU oracle remains unchanged. Full arbitrary-angle
acceptance, particle/seed convergence, Stage-0 convergence, and xigma table/sampler
convergence remain separate tasks. Agreement between GPU and CPU delta does not
close those requirements or promote the production runner's scientific status.

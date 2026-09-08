# Sampler integration checkpoint

**Superseded numerical status:** RES068 repairs the proposal/weight mismatch,
float32 cancellation and support/allocation defects described below. The former
expected-failure gate now passes unchanged; exact-CDF, high-gamma and refined-input
scenario-bank regressions are committed alongside the repair. See
`docs/ALPHA_GPU_VALIDATION.md` for current measurements and the reusable benchmark
at `scripts/benchmark_xigma_sampler.py`. NumPy remains the default; broad convergence
and independent emission validation remain open. The text below preserves the
pre-repair integration checkpoint, not the current blocker list.

Merged feature/xigma-importance-sampler into release/alpha-script and then main.
The original audit runtime changes, RES060 polarization and RES061 slice measures
are retained. Sampler decision is RES062; DER008 remains derived.

Current policy: NumPy is the alpha default. Explicit CuPy or auto is experimental,
head-on linear polarization only. Results record the backend, fixed sampler settings
and a warning. Nonfinite GPU output raises. Collision inputs and published Stage-0
arrays are read-only; memoization remains per-instance.

GPU verification was run outside the sandbox on GTX 1660 Ti, driver 580.173.02,
CuPy 14.2.0. Do not infer missing hardware from sandbox cudaErrorNoDevice.
Focused final check: 23 passed, 1 strict expected failure in 18.85 s:
PYTHONPATH=/tmp/gammaforge-alpha-integration/src /home/alexander/Work/Code/GammaForge/.venv/bin/pytest -q tests/test_xigma_gpu_sampler.py tests/test_xigma_engine.py

Numerical blocker: the baseline GPU/NumPy integrated ratio is 1.419 with relative
L1 difference 0.653; subsampling changes did not establish convergence. The original
median-only check misses this discrepancy. Reproduce the stronger test with
--runxfail. See docs/ALPHA_GPU_VALIDATION.md for scope and numbers.

Root owns final main integration, full tests, packaging and release guide.
No physics constants were tuned. Further GPU work must establish convergence of both
reference quadrature and sampler before promoting this backend.

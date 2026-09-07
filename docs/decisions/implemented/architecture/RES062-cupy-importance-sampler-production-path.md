# RES062 — CuPy ring/annulus importance sampler integrated as production path; NumPy grid quadrature retained for reference

Status: implemented
Class: architecture

*(Numbering note: this entry was written as RES061 and renumbered to RES062 on
2026-09-06 to resolve a collision with parallel work on `main`.)*

## Problem

Stage 2 spectral/angular evaluations (`angular_spectrum_from_table`, `spectrum_in_angular_range`) in
`xigma` require integrating emission over the 4D phase space `(gamma, theta_x, theta_y, ahat)`.
In RES029, GammaForge ported the predecessor's direct grid quadrature (*reference.py*) as the
production NumPy path, deliberately postponing the predecessor's ~550-line CuPy rawkernel
importance sampler (*spectrum_kernel_4d* / *spectrum4d.py*) because there was no GPU hardware
accessible to validate it against and its Monte Carlo noise characteristics in sparse particle
regimes were noted.

However, for production runs and interactive exploration with fine angular/spectral grids (such as
3D collimated spectra or 2D angular distributions), the brute-force grid quadrature scales linearly
with the product of the 4D table size and the query grid size (`O(N_gamma * N_tx * N_ty * N_a0 * N_out)`),
requiring tens of seconds per slice. The predecessor's ring/annulus importance sampler evaluates
the energy resonance condition on concentric circular arcs in `(theta_x, theta_y)` and
importance-samples the azimuthal emission distribution, reducing multi-point spectrum queries to
milliseconds on a modern GPU.

## Decision

1. Re-introduce the ring/annulus importance sampling algorithm as a `cupyx.jit.rawkernel` in
   `gammaforge.engines.xigma.spectrum_sampler`.
2. Reconcile the kernel with GammaForge's modern architecture:
   - Normalize with `KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2 pi)` (RES033).
   - Consume 1D device arrays `ahat_centers` and `ahat_widths` supporting the non-uniform
     log-spaced target grid (RES032), replacing the predecessor's uniform `a0` linspace assumption.
3. Expose compute backend selection via `backend: Choice("auto", "cupy", "numpy", default="auto")`
   in `XigmaEngine.schema` (`XIGMA_SPECS`) and optional arguments to Stage 2 functions:
   - `"auto"` (default): Dispatches to CuPy importance sampling if CuPy and a CUDA device are
     available, and the interaction has head-on linear polarization. Otherwise falls back
     gracefully to NumPy brute force.
   - `"cupy"`: Explicitly requests the CuPy kernel. If CuPy or a CUDA device is unavailable, raises
     `RuntimeError`. If non-zero ellipticity or crossing angles are requested, raises
     `NotImplementedError`.
   - `"numpy"`: Always executes the deterministic brute-force grid quadrature.
4. Retain the NumPy brute-force grid quadrature as the golden reference for testing and validation.

## Alternatives considered

**Replace the NumPy brute-force path completely with the CuPy kernel.** Rejected: NumPy brute force
is deterministic, requires no CUDA hardware or drivers, and provides an exact quadrature reference
for unit tests and CPU-only environments.

**Port the CPU ctypes/numpy fallback of the importance sampler.** Rejected: The CPU version of the
importance sampler (*spectrum4d_cpu.py*) was slow and carried Monte Carlo variance; NumPy brute force
is simpler, exact, and vectorizes cleanly on CPU.

**Support arbitrary ellipticity and crossing angles in the initial CuPy kernel.** Deferred: The
predecessor kernel implemented head-on linear polarization. Extending the ring geometry and
polarization projection to arbitrary ellipticity and 3D crossing angles requires a new derivation for
off-axis resonance curves, which remains open (DER004, DER005).

## Rationale

The CuPy rawkernel preserves the ring/annulus reduction while the NumPy quadrature remains
the deterministic reference. Its GPU-versus-NumPy agreement test is part of this change, but
this integration host has no CUDA device, so neither a performance claim nor a numerical
agreement measurement is recorded here. *DER008* remains `derived`; GPU execution and the
independent comparison must pass on CUDA hardware before either is claimed as verified.

## Consequences

- `XigmaEngine` selects the GPU kernel only for head-on linear polarization on systems with
  CUDA and CuPy (`backend="auto"`). Non-zero ellipticity or crossing angles always use the
  NumPy reference; an explicit unsupported `backend="cupy"` request raises.
- Angular/collimated `Results.model_specific` reports the selected `stage2_backend`; GPU
  results additionally report the fixed sampler settings (`samples_total=256`,
  `subsampling=32`).
- Direct stage callers and unit tests can continue to request `backend="numpy"` for deterministic
  verification.
- Code and tests maintain compatibility with environments lacking CUDA GPUs.

## Amendments

> **2026-09-07 — alpha validation keeps NumPy as the default.** On the GTX 1660 Ti,
> the 40,000-particle baseline's 5x5x8 cube gave a GPU/NumPy integrated ratio of
> 1.419 at the fixed 256x32 sampler setting; increasing subsampling to 64, 128, and
> 256 gave 1.230, 1.252, and 1.373. A smooth-table diagnostic also produced non-finite
> GPU values for supported points. The port remains explicitly selectable for diagnosis,
> but `backend="numpy"` is the alpha default, `auto` is an opt-in experimental route,
> and GPU results carry an actionable warning. *DER008* remains derived.

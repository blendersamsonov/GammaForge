# RES069 — CuPy incident polarization and crossing geometry

Status: implemented
Type: feature

Basis update: RES078/DER012 adds local transverse projection before the stable
radiation-vector evaluation below. Stable arithmetic and incident-polarization scope
remain in force; earlier unprojected-basis results are historical.

## Problem

CuPy Stage 2 rejected nonzero ellipticity and laser crossing angles even though
NumPy already implemented the corresponding DER004–DER006/RES060 factors. Simply
porting the expanded polarization norm to float32 would introduce another precision
defect: its large cancelling terms become significant when the laser basis has a
longitudinal component. Stabilizing only the denominator (RES068) is insufficient.

## Decision

CuPy evaluates the manuscript's Eq. udef vectors directly and weights their squared
norms by the existing elliptical-polarization diagonal weights. The host computes
the once-rotated laser basis and weights; the device computes the per-sample lab-frame
projection. The helper is private, not a new public polarization-output API.

With unit electron/observer directions u and n, use

    delta = 1/[gamma^2 (1+beta)]
    Delta = n-u
    d = delta + beta |Delta|^2/2
    Ui = (Delta + delta*u) (n.ei)/d - ei
    factor = (|U0|^2 + epsilon^2 |U1|^2)/(1+epsilon^2)

The longitudinal component of Delta is evaluated from factored differences of the
squared slopes and their normalization factors, not by subtracting two rounded
near-one components. This is an algebraic evaluation of the same Eq. udef, with no
new emission approximation, clipping or normalization adjustment.

The existing angular-query API forwards ellipticity and both crossing angles to
CuPy. Explicit CuPy requests require CUDA; auto uses it when available and otherwise
falls back to NumPy. NumPy remains the default. Existing Stage-0 flux and shared
photon-energy conversion supply their crossing corrections exactly as before;
neither correction is duplicated inside the sampler.

The existing `1/s^2` factor is applied before multiplying the emission prefactor
by H, rather than after summation. Larger crossed polarization factors otherwise
overflow float32 intermediates even when the final scaled density is representable.

## Alternatives considered

- Porting the expanded dot-product norm directly: rejected because collinear
  float32 cancellation survives the stable denominator repair.
- Direct vector evaluation with naive subtraction of longitudinal directions:
  rejected because it loses the small longitudinal difference before forming Ui.
- Switching the entire sampler to float64: unnecessary for this bounded projection
  and would change the memory/performance tradeoff of RES068.
- Outgoing Stokes parameters or spatially varying polarization: separate features,
  not requirements for incident-polarization dependence of intensity.

## Rationale

Direct device tests compare against an independently evaluated extended-precision
Eq. udef and its collinear cosine-squared limit. Spectral comparisons refine the input
angular quadrature of the same H, preserving the distinction between formula checks,
numerical integration agreement and independent emission-physics validation.

The expression-order risk is also covered by NVIDIA's
[floating-point computation guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
The manuscript and author-approved RES060, not external implementation guidance,
remain the physics authority.

## Consequences

Uniform incident linear, elliptical and circular polarization and the existing
two-plane laser-crossing geometry can use CuPy. The backend remains experimental;
fixed-ring convergence and independent arbitrary-angle emission checks remain open.
GUI, kascade, NumPy emission formulas, energy conventions and the ahat default are
unchanged. Timing and numerical evidence belong in `docs/ALPHA_GPU_VALIDATION.md`.

## Amendments

> **2026-09-12 — Direction-dependent Doppler extension.** RES082 extends the nominal
> flux/energy convention with per-electron beta=1 encounter weighting, resonance and
> Jacobian, including conservative CUDA support. This does not replace the polarization
> or geometry construction recorded here.
> **2026-09-29 — Documentation relocation.** The measurement formerly at `docs/ALPHA_GPU_VALIDATION.md` is retained at `docs/validation/alpha-gpu-sampler-2026-09.md`.

# RES085 — Validated CuPy delta emission reference

Status: implemented
Class: feature

## Problem

The untracked GPU delta prototype in the spectrum-proposal worktree returned zero
on-axis emission for a collinear electron, where the documented polarization factor
is one. It squared the wrong projection, omitted the radiation vector's subtraction,
and used incorrect ellipticity weights. Its linear anchor also reversed scalar and
vector output shapes, and its resonance predates RES082's direction Doppler.
Promoting it unchanged would invalidate spectrum and performance comparisons.

## Decision

Implement `src/gammaforge/validation/references/delta_cupy.py` as an explicit CUDA
reference using a fused float64 per-particle emission kernel and weighted CuPy
histograms. Evaluate DER012's local transverse dipole radiation vectors and DER013's
direction Doppler independently of xigma's production helpers. Check individual
energies and weights against the long-double double-cross-product CPU reference.

`emission_lines` and `bin_emission` use the existing physical-energy contract with
NumPy inputs and outputs. Particle chunks bound device temporaries. Normalized
`resonance_spectrum` and finite-cone `angle_integrated_spectrum` retain the historical
nominal default; callers explicitly choose the direction mode for current xigma.
The angle-integrated wrapper transfers each particle chunk once, then reuses it
over its observation loop. The linear head-on `single_electron_spectrum` anchor
retains its documented omission of nonlinear redshift and returns the input grid's
shape. Legacy normalization-report machinery is not duplicated in the accelerator.

Explicit CUDA unavailability raises; no CPU execution can masquerade as a GPU result.
`scripts/validate_delta_cupy.py` records device, source fingerprints, two seeds,
head-on/crossed linear/elliptical/circular cases, on/off-axis directions and two chunk
sizes over the shared scenario bank. Warm query times include transfers; the first
call is reported separately. Non-finite errors, missing CUDA and source mutation fail.

## Alternatives considered

Copy the prototype unchanged: rejected by its failing collinear anchor and obsolete
resonance. The worktree prototype is left intact as historical local work.

Call production polarization on CuPy arrays: avoids duplicate algebra but weakens the
reference's independence. The separate implementation is checked against explicit
extended-precision vector algebra instead.

Keep a long chain of GPU array operations: rejected in favor of fusing per-particle
arithmetic into one elementwise kernel, reducing launches and temporary arrays while
preserving the direct resonance-and-histogram algorithm.

Store device arrays in shared trajectory objects or silently choose a CPU backend:
rejected because the shared boundary remains NumPy and GPU validation must prove
actual device execution.

## Rationale

Analytic on-axis, linear-spectrum and angular-photon-count checks detect errors that
smooth plots or agreement with shared production helpers could conceal. Matched bin
masses and tails retain absolute normalization. Small chunks expose accumulation and
transfer mistakes without requiring an oversized GPU allocation.

## Consequences

The repaired GPU reference is suitable for numerical comparisons within the measured
scope; the validation evidence is in `docs/validation/delta-cupy-2026-09-14.md`.
This extends RES076 and RES077 without replacing the independent CPU oracle or
changing the production runner's acceptance matrix. It does not close RES074,
particle/Stage-0 convergence, or independent xigma CUDA scientific acceptance.
Any performance figures based on the faulty prototype require regeneration.

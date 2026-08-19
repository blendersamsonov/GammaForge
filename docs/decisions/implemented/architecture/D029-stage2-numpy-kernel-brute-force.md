# D029 — Stage 2's numpy kernel ports the predecessor's brute-force grid quadrature, not the GPU importance sampler; `KERNEL_NORMALIZATION_CONSTANT` isolates the pending §9.1 factor

Status: implemented
Class: architecture

## Problem

Stage 2 needs a production numpy kernel, and the predecessor had two candidates: a direct
grid-quadrature sum (*reference.py*) and a ~550-line GPU importance sampler
(*spectrum4d.py*/*spectrum4d_cpu.py*) that the predecessor's own audit already rated
trust-level C, with 3x-30x variance in sparse/narrow-angle configurations.

## Decision

`stages.spectrum_from_table`/`angular_spectrum_from_table`/`spectrum_in_angular_range`
port the predecessor's *reference.py*'s `spectrum_from_table` — a direct sum over the
table's own `(theta_x, theta_y, ahat)` cells with quadrilinear-in-gamma interpolation
(`stages._interp_gamma`) — as the **production** numpy path, not the importance sampler.
Porting the sampler would import a known-noisy path as this phase's only implementation,
with no GPU to validate it against yet.

`cupy`/`numba` backends are gated exactly like Stage 0's `_check_backend` (`stages.py`,
Phase 2.5): declared in `schema.py`'s `scheme`/eventual-backend vocabulary but rejected at
call time until a real kernel exists, rather than silently falling back to numpy under a
GPU-sounding flag.

**The pending §9.1 constant** (`stages.KERNEL_NORMALIZATION_CONSTANT = 1.5`) is pi-free,
matching the predecessor's own kernel math exactly (its `coef=1.5`, explicitly documented
pi-free in both *spectrum4d.py* and *reference.py*). The predecessor's ~2π gap is *not*
inside this constant — it is the same gap D025/D026 already traced, between this kernel's
differential form and the table-free `angle_integrated_spectrum` shape, both of which this
repo's engine now computes (`stages.angle_integrated_spectrum` for `Collision.spectrum`,
the table kernel for everything else). `run.py::identity_checks`'s fourth leg (Stage 2
kernel vs `delta.resonance_spectrum` at one point) measures ~1.00 across the scenario bank
precisely because both sides carry the identical pending factor — evidence the constant is
isolated correctly, not evidence the §9.1 question is closed.

**A resolution artefact, found while testing this constant, not caused by it: evaluating
`spectrum_from_table` exactly at a beam's own angular centre against a `scheme="nearest"`
table aliases against that table's cell boundaries** — measured ratios from 0.48 to 1.67
against `delta` across theta-bin counts 10-150 at fixed particle count
(`tests/test_stage1_stage2.py`). `scheme="cic"` (already in scope, §4.2) removes it,
holding within a few percent from 40 to 250 theta bins. `run.py`'s fourth identity leg and
the `test_stage1_stage2.py` regression test both use CIC for this reason; `nearest` stays
the schema default (cheaper, and Stage 1's own conservation is scheme-independent).

## Alternatives considered

**Port the GPU sampler now, run it CPU-side via cupy's numpy fallback or a hand rewrite.**
Copies a known-noisy algorithm before there is hardware to validate it against, and
duplicates ~550 lines this phase's exit criteria (`GRAND_PLAN.md` §11, "Stage architecture
tests green; placeholders documented") do not ask for.

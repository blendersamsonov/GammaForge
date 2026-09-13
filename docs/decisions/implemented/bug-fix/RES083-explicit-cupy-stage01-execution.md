# RES083 — Explicit CuPy execution for xigma Stages 0 and 1

Status: implemented
Class: bug-fix

## Problem

The merged array-module-aware code inferred its backend from shared NumPy bunch arrays.
An explicit CuPy Stage-0 request still evaluated the laser on NumPy, and Collision did
not pass its backend choice to either particle stage. Stage-1 CuPy paths also called
array reductions on Python shape tuples. The merge title claimed GPU execution that
the public pipeline did not perform.

## Decision

Resolve numpy/cupy/auto explicitly. Collision fixes the selected Stage-0/1 backend for
its cache lifetime, forwards it to both particle stages, and reports the stage backends
in result metadata. Explicit CuPy requires usable CUDA; auto permits initial CPU fallback.

Stage 0 computes conservative geometry windows and small per-particle kinematics on the
host. Each particle chunk transfers positions, velocities, weights and window arrays;
trajectory evaluation, LaserField intensity calls, integration and source histograms run
on the selected array module. Pulse-train delays are transferred to that same module.
Completed chunk reductions return to NumPy before the shared chunk utility frees device
memory. Failed chunks never enter the reduction.

Stage 1 builds fixed edges on the host, transfers particle coordinates/weights per chunk,
and uses CuPy coordinate arithmetic, nearest/CIC deposition and weighted bincounts.
Completed masses transfer back before in-place host accumulation. An allocation failure
before that commit retries without double-counting. Shape products use Python arithmetic
rather than an array reduction on a shape tuple. The particle temporary budget excludes
fixed table buffers, which must still fit at the smallest retry size.

Public Bunch, TrajectorySamples, ShapeTable, cached arrays and Results retain NumPy
storage. Existing retargeting, Stokes, table-free spectra and persistence continue on the
host. CGS values, normalization and the physical formulas are unchanged.

## Alternatives considered

- Infer execution from input arrays: rejected because shared host arrays silently
  defeated the explicit backend selection.
- Store CuPy arrays in shared objects and all cached intermediates: rejected because
  ownership guards, NumPy retargeting and persistence expect host arrays; this would
  expand a backend-wiring fix into a device-resident pipeline redesign.
- Silently fall back for explicit CuPy: rejected because it conceals whether requested
  device execution actually happened. Automatic selection is the explicit fallback mode.
- Accumulate deposits on a persistent device table during retries: rejected because
  partial writes before an OOM could double-count unless another transactional buffer
  were introduced. Completed chunk masses already provide a simple commit boundary.

## Rationale

The backend schema should control real numerical work, while shared engine boundaries
stay compatible with the existing application. Real-device tests observe CuPy inputs
inside laser evaluation and both deposition implementations, compare NumPy outputs,
and inject failures after device work to check retry-safe conservation.

## Consequences

CuPy engine requests now need CUDA even for total-yield-only calculations. CPU-only
callers select numpy or auto. Transfers and launch overhead can outweigh gains for
small bunches; this is not a universal speedup claim. Memory remains bounded by particle
chunks plus the requested fixed table, with the existing limited OOM-retry policy.

The device checks, numerical agreement and timing limits are recorded in
`docs/validation/cupy-stages01-2026-09-13.md`.

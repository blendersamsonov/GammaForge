# RES064 — Input boundary snapshots and owned particle arrays

Status: implemented
Type: architecture

## Problem

Several shared dataclasses were frozen only at the attribute level. A caller could still
mutate an `engine_params` mapping, nested target ranges, or a particle ndarray after
submitting a request, changing the data observed by a runner or cached calculation.
Direct construction also accepted non-finite scalar values, non-integral sampling counts,
unsupported seeds, and inconsistent particle-vector lengths.

## Decision

The `gammaforge.io` input boundary takes ownership of mutable payloads that determine
calculation inputs. `CalculationRequest.engine_params` and `OutputRequest.manual_ranges`
are shallow mapping snapshots with immutable nested tuples; `Target.outputs` and output
resolutions are normalized to tuples. `Bunch` copies all particle vectors to independent,
read-only one-dimensional float arrays and rejects unequal lengths, while preserving
zero-length vectors. Direct constructors validate finite scalar inputs, NumPy real/integer
scalars, positive finite `InteractionParameters.N_e`, and the supported non-negative seed
range `[0, 2**31 - 1]`.

## Alternatives considered

- Rely on `frozen=True` and document that callers must not mutate their containers. This
  leaves in-flight requests vulnerable to ordinary aliasing and was the audited failure.
- Return read-only views of caller arrays. A caller retaining ownership can still mutate
  the backing storage, so views are not a snapshot.
- Deep-copy all arbitrary nested metadata and external `LaserField` objects. This adds
  unbounded cost and would falsely imply that arbitrary mutable field implementations are
  stable cache keys; those ownership and cross-run cache questions remain explicit engine
  concerns.

## Rationale

The copied payloads are small structural containers or the exact arrays crossing into a
calculation that must remain stable. `Parameters` is already immutable, so its values need
no recursive copy. Array copies prevent both mutation through the original caller-owned
storage and mutation through a returned `Bunch`; empty arrays remain useful for analytical
and filtered paths. Physical positivity and correlation admissibility remain in the
existing explicit `validate(beam/laser)` functions to avoid changing established notebook
construction semantics.

## Consequences

Callers cannot use a returned `Bunch` as a mutable workspace; create a new array or a new
`Bunch` when changing particles. Public construction allocates one owned copy, while pure
internal transforms use a private fresh-array constructor to avoid repeating that copy.
Mutable arbitrary `LaserField` instances are not silently treated as immutable cache keys.

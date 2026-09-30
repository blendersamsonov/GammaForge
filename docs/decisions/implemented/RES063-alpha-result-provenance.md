# RES063 — Alpha results preserve metadata and the submitted Gaussian request

Status: implemented
Type: architecture

## Problem

Saving script results discarded engine warnings and numerical metadata. Reloaded output
keys changed type, and the beam/laser spec sidecar omitted target, seed and engine settings.
Those omissions make a saved scientific calculation difficult to interpret or reproduce.

## Decision

`save_results` writes a versioned HDF5 file with slice measures, photon arrays, existing
`Bunch` payloads and safely encoded metadata. It optionally embeds the submitted
`CalculationRequest` as YAML; `load_request` reconstructs it with caller-supplied engine
schemas. The request format records Gaussian inputs, target, sampling and every engine
setting in canonical CGS without rounding. Unsupported field implementations raise.

`load_results` restores recognized `OutputKind` keys by default; an explicit string
converter retains the old behavior. Unknown names remain strings. Metadata supports
string-key mappings, numerical/string arrays, tuples/lists and plain scalar values.
Dataclasses load as field mappings; no engine class is imported by the shared I/O layer.
Unsupported objects and nonfinite JSON numbers fail rather than being silently dropped.

The writer validates metadata before opening its temporary file and replaces the destination
only after HDF5 construction succeeds. A supplied legacy spec sidecar remains a separate
file; the embedded request is the complete provenance record in the atomic HDF5 artifact.

## Alternatives considered

Pickling engine objects would preserve Python types but adds executable deserialization
and ties files to implementation classes. Dynamic imports violate the shared I/O boundary.

Keeping provenance solely in separate ad hoc script files makes it easy to separate
results from the settings that produced them. The optional embedded request retains the
existing small results API while keeping one complete artifact.

Silently accepting an arbitrary laser's fitted Gaussian description would substitute a
different calculation on replay. Such sources require an explicitly designed format.

## Rationale

The supported alpha uses Gaussian inputs and the existing typed engine schemas. Storing
their actual numerical values is sufficient for replay without a new execution framework
or a configuration back-reference on `Results`.

## Consequences

Old unversioned files remain readable; missing request provenance raises when requested,
and missing metadata stays empty. Unsupported future versions fail explicitly.
Callers supply the request that actually produced the saved result; the serializer cannot
infer or verify that association. Loaded dataclass metadata is data, not a live class instance.
GUI run export, arbitrary fields and future electron typing remain separate work.

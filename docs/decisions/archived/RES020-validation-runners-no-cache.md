# RES020 — The validation runners have no result cache

Status: implemented
Type: testing
Archived: 2026-08-10

## Problem

The predecessor cached validation results to disk, keyed by commit hash, because a GPU run
per tier per scenario dominated its suite's cost. Whether that mechanism is still needed
depends on whether either condition still holds here.

## Decision

`validation/runners.py` runs an engine and returns; nothing is memoized to disk.

## Alternatives considered

**Porting the predecessor's `cache.py` — a commit-hash-keyed pickle store that skipped
recomputation when the tree was clean.** Rejected — see Rationale.

## Rationale

That cache existed because a GPU run per tier per scenario dominated the suite's cost, and
the tiers each re-ran the models independently. Neither is true here: the tiers are gone
(one run is shared), and engines cache their *own* intermediates keyed by the exact hash of
the inputs each one consumed (§5), which is both finer-grained and valid by construction
rather than by a clean-tree heuristic. A second, coarser cache on top would be the
speculative machinery P6 warns about. It is worth reinstating the moment a full suite run
is *measured* to be too slow — not before.

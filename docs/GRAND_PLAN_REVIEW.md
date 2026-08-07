# Review of GRAND_PLAN.md — Proposed Changes

**Status:** draft review v0.1 — 2026-08-07
**Reviewer:** Claude, grounded against `/home/alexander/Work/Code/ComptonSuite` and
`~/Work/Papers/2026/Compton-Numerics`

> This is a review, not an edit — nothing in `GRAND_PLAN.md` has been changed. Add your
> own comments inline below each item; we'll fold the agreed changes back into the grand
> plan afterward.

## Context

`docs/GRAND_PLAN.md` (draft v0.1) lays out a ground-up rebuild of the Compton-photon
simulation toolkit, replacing the predecessor at `/home/alexander/Work/Code/ComptonSuite`.
This plan was reviewed in two grounded passes rather than read at face value:

1. Every specific factual claim the plan makes about the predecessor repo (13 claims:
   unit-splitting, `ModelAdapter`, `ModelSpec`/`adapt_to_model`, the abandoned `core/`
   package, `BeamFittedParams`, `Results.cfg`, `ModelCapabilities`, the xigma `Config`
   dataclass, the 1685-line GUI monolith, the ~2π kernel discrepancy, chunking machinery,
   old task notes, and the test suite's character) was checked against the actual code and
   git history at `ComptonSuite`.
2. The plan's physics authority — the paper draft at
   `~/Work/Papers/2026/Compton-Numerics` — was read directly to check whether it actually
   contains settled answers for the three items the plan marks BLOCKING (§9): the ~2π
   normalization, the ellipticity→a0 factor, and the crossing-angle derivation.

**Result: almost every historical claim about the predecessor is accurate and well
evidenced** (details below, mostly no change needed). But the paper audit surfaced a
structural problem that the plan's phasing doesn't currently account for, and the repo
audit surfaced a phase-ordering bug plus several omissions worth folding in before this
becomes v0.2. The changes below are organized by severity: structural/sequencing issues
first, then design-principle refinements, then additions the plan currently omits.

---

## A. Structural issues (should change before implementation starts)

### A1. The paper does not actually settle two of the three §9 BLOCKING items

The plan treats §9.1 (~2π), §9.2 (ellipticity-a0), and §9.3 (crossing angle) uniformly as
"paper-code discrepancies, resolve with the author" (P14). Direct inspection of
`xigma.tex` shows this framing only fits §9.1:

- **§9.1 (~2π):** the paper *does* give one specific, fully-derived, unambiguous
  normalization (`eq:main`/`eq:Fmatrix`) — no flagged ambiguity. But the paper's own
  "Validation study" section is an explicit placeholder ("Expected content: agreement
  within the predicted bound...", not yet written), so **the paper has never checked its
  own formula against independent numerics.** Converging the three code paths to match
  the paper does not guarantee correctness — it guarantees agreement with a formula that
  is itself unvalidated.
- **§9.2 (ellipticity-a0):** the paper's polarization object `Ξ̂` satisfies `Tr Ξ̂ ≡ 1` by
  construction for any single fully-polarized state (it's a normalized 2×2 coherence
  matrix). There is no independent "ellipticity" scalar and no `(1+ε²)/2` identity
  anywhere in the source, and no pulse-energy→a0 formula at all — `a0` is treated as a
  given input, not derived. The plan's assumed `(1+ellipticity²)/2`-style formula isn't
  in the paper; whether "ellipticity" as a schema parameter even maps cleanly onto `Ξ̂`
  is an open modeling question, not a lookup.
- **§9.3 (crossing angle):** genuinely absent. The entire angular-spectrum derivation is
  built for near-backscattering/near-head-on geometry, accurate to `O(θ²)` around a
  collinear axis, and the paper explicitly warns against extending it without revisiting
  the geometry. This is net-new physics derivation, not a documented-but-unimplemented
  formula.

**Proposed change:** reword §9.2/§9.3 and the P14/risk-table framing to distinguish
"discrepancy to resolve" (§9.1) from "derivation that doesn't exist yet and must be done
from scratch with the author" (§9.2, §9.3). Add a risk-table row: *"Crossing angle and
ellipticity-a0 have no existing derivation in the paper (confirmed, not just
undocumented) — open-ended research tasks, not consult-and-implement. Mitigation: build
the schema/architecture to carry `crossing_angle` and `ellipticity` as first-class
parameters now (per §2.2's existing intent), but do not gate Phase 3's exit on their
physics being correct — wire them as identity/no-op until the derivation lands, and let
that derivation proceed in parallel rather than serially blocking downstream phases."*



### A2. Phase 3 exit criteria depend on `delta`, which isn't built until Phase 5

`delta` (§4.5) is described as the independent first-principles check on xigma's Stage 2
— exactly the kind of arbiter needed to resolve §9.1's ~2π factor, especially now that
we know the paper's own formula is unvalidated (A1). But the phase table (§11) puts
`delta` in **Phase 5** ("kascade port + delta"), while **Phase 3**'s exit criteria already
require "~2π resolution" to be done. As written, Phase 3 cannot actually reach its own
exit criterion using the tools the plan assigns to it — the one method capable of
independently arbitrating the normalization doesn't exist yet at that point.

**Proposed change:** either (a) move a minimal `delta` implementation earlier — into
Phase 3 itself or a new Phase 2.5 — scoped just enough to arbitrate Stage 2 normalization,
with the fuller cross-validation role staying in Phase 5; or (b) explicitly soften Phase
3's exit criterion to "~2π resolved to the extent verifiable by closed-form identities
and analytical anchor; final cross-check against delta deferred to Phase 5" and note the
dependency explicitly. (a) is preferable — it's a small, well-scoped piece of Stage-0
reuse per §4.5's own description ("reusing xigma's Stage 0").



### A3. Phase 3 bundles pure engineering with open-ended physics research as one gate

Related to A1/A2: Phase 3's exit criteria conflate "port/rebuild the stage
architecture, kernels, chunking" (bounded, portable engineering work) with "resolve ~2π,
implement crossing angle, fix ellipticity-a0" (one bounded item plus two open research
items with unknown timelines, per A1). Because phases run serially in §11, an
open-ended physics question could stall Phases 4–8 entirely.

**Proposed change:** split Phase 3 into **3a (engineering)** — stages, facade, engine
wrapper, kernels, chunking consolidation (see B1), with normalization/angle/ellipticity
wired as explicit placeholders — and **3b (physics closure)** — the three §9 items,
explicitly allowed to run concurrently with Phase 4 (analytical) and Phase 5 (kascade/
delta) engineering rather than blocking them. Only final validation completion (Phase 7)
should require 3b to be closed.



---

## B. Design-principle refinements (P1–P14 and related sections)

### B1. Chunking machinery: "the code is ported" is the wrong instruction (§4.2)

The plan says the VRAM/RAM-aware halve-and-retry chunking is "preserved from the old
repo — the lessons are requirements, the code is ported." In the actual repo this logic
exists as **three independently-implemented, inconsistent versions**: auto-sizing +
retry in `particles.py` and a separately-duplicated auto-sizing + retry in
`spectrum_from_particles.py` (each with its own copy of `_MAX_OOM_HALVINGS`), plus a
third, differently-designed `build_table_streaming` that takes a manually-specified
chunk size with no retry and is documented as dead/unwired code. There is no single
"the code" to port.

**Proposed change:** reword §4.2's Stage 0 bullet to call for **one shared auto-chunk +
OOM-retry utility** (consuming the already-good, already-shared memory-query primitives
like `available_vram_bytes`/`available_ram_bytes`) used by every chunked stage — port the
*algorithm and the hard-won constants* (the `_MAX_S_CHUNK` cap history, the halving
policy), not the triplicated implementation, and explicitly retire the unwired
manual-chunk design rather than giving it a fourth home.



### B2. P10's "declarative capability data" needs to be explicitly distinguished from the abandoned `ModelCapabilities`

`ModelCapabilities` was built, then explicitly deleted (commit confirmed), with the
justification recorded in the old repo as "the GUI already hardcoded which model is the
background preview... instead of registering a greyed-out placeholder" — i.e. it was
removed as unnecessary registry/protocol machinery, and the GUI now does capability
checks with plain `hasattr`, a pattern the old repo's own conventions doc explicitly
codifies ("never `isinstance()` against a GUI/engine boundary"). P10 proposes bringing
capability data back as static tuples on the `Engine` object (`supported_outputs`,
`cheap_groups`) — plain data, no registry. That's a real, meaningful difference from what
was removed, but the plan doesn't spell out *why this time is different*, and the same
mistake (regrowing a registry/protocol layer around it) is an easy trap during
implementation.

**Proposed change:** add one sentence to P10 or §4.1 making the distinction explicit:
*"Unlike `ModelCapabilities`, there is no registry, no protocol, no `UnavailableAdapter`
placeholder mechanism — `supported_outputs`/`cheap_groups` are plain tuples read directly
off the `Engine` instance the GUI already holds. If this grows a registry, discovery
mechanism, or capability-negotiation protocol, that's the old mistake recurring."*



### B3. GUI thinness needs an enforced boundary, not just a principle (P3, P12, §6)

The predecessor's GUI monolith was deliberately trimmed once — from 1685 lines down to
1174 lines ("GUI-as-thin-consumer + ModelAdapter unification") — and then **regrew back
to 1685 lines** through normal feature additions (angle-resolved spectra panel, output-
spec controls, live collimation wiring). P3/P12 restate the same "GUI is thin, no
physics, no branching" principle the old repo already tried and lost.

**Proposed change:** add a concrete enforcement mechanism alongside P12, not just the
principle — e.g. a lightweight import-boundary check (lint rule or a small CI test
asserting `gammaforge.gui` never imports from `engines/*/stages.py` or any physics
kernel module, only `Engine.run()`/`Results`). Discipline alone already failed once in
this codebase; the rebuild should not repeat that specific failure mode.


### B4. `kascade` has zero dedicated tests today — don't inherit that silently into the ≥4-method validation role (§4.4, §7)

The plan ports `kascade` "as-is, minimal effort" and simultaneously relies on it as one
of "≥4 cross-validation methods" (§7). Current `kascade.py` (794 lines) has no unit
tests of its own in the predecessor — its only indirect correctness signal comes from a
shared laser-envelope test and the validation tier suite. If it's going to anchor
cross-validation in the new suite, "minimal port" shouldn't also mean "port untested."

**Proposed change:** add one line to §4.4 or the Phase 5 exit criteria: kascade needs at
minimum a closed-form sanity check (e.g. Thomson-limit total yield) before Phase 5's
"4-method cross-validation runs" exit criterion is considered met — otherwise the
4-method comparison has one leg that was never independently checked.

---

## C. Omissions worth adding

### C1. Preserve the predecessor's documentation/provenance discipline as an explicit deliverable

The old repo's docstrings and `AGENTS.md` are unusually good — most non-trivial design
decisions record *why* something was tried and rejected (this is in fact where most of
the plan's own P6–P11 provenance claims come from). That's a real asset worth carrying
forward deliberately, and the plan doesn't currently mention it.

**Proposed change:** add a bullet to §0 goals or §11 Phase 8: *"Maintain an `AGENTS.md`-
style 'design decisions, not to be revisited without good reason' doc from Phase 0
onward, in the same spirit as the predecessor's — this was a genuine strength worth
keeping, not just replacing."*


### C2. Guard against doc staleness

Despite the strong documentation discipline (C1), the predecessor's own `AGENTS.md` and
`validation/` modules currently reference types deleted in earlier refactors (a
`Photons` result type, `CollisionParams`/`build_params`) — stale even in commits from the
day before HEAD. Good documentation habits didn't prevent drift during fast refactors.

**Proposed change:** add a lightweight periodic check (even a manual Phase-8 checklist
item, or a grep-based CI smoke test for symbol names mentioned in docs) rather than
assuming the discipline alone will keep docs in sync — it demonstrably didn't last time.


### C3. Scaffold-phase repo hygiene (Phase 0)

The predecessor accumulated large committed binary/data artifacts at repo root
(`final_distribution.ele` ~8.7MB, a `.sync-conflict` duplicate ~17MB, a ~2.8MB notebook)
and multiple sync-conflict file variants, suggesting edits from unsynced machines/cloud
clients without `.gitignore` coverage for them.

**Proposed change:** add to Phase 0's exit criteria: `.gitignore` explicitly covers large
data formats (`.ele`, notebooks with large outputs) and sync-conflict patterns from day
one.

### C4. Check existing in-flight worktrees before rebuilding overlapping work from scratch

The predecessor repo currently has local branches (`worktree-gpu-streaming-chunk`,
`worktree-xigma-oom-fix`, `worktree-gui-audit-fixes`, `worktree-polish-pass`) and remotes
(`worktree-core-simulation-api-refactor`, `worktree-param-framework-collision-params-
refactor`, `worktree-robust-mixing-perlis`, `worktree-doc-cleanup`) addressing exactly
the chunking/OOM and GUI issues this plan wants to fix fresh. These may contain more
current thinking than what's on `master`, including a possibly-already-unified chunking
utility (B1).

**Proposed change:** before finalizing Phase 3/6 scope, skim those branches for salvage-
able work — cheaper than rebuilding a solution that may already exist half-finished.

---

## D. Not proposing to change

Everything else checked against the old repo held up as stated: the SI/CGS boundary
split (P1), `ModelAdapter`'s GUI/physics fusion (P5/P6), the `ModelSpec`/`adapt_to_model`
dead-end (P6), the never-built `core/` package (P7), the successfully-already-merged
`BeamFittedParams` → `Bunch.gaussian_fit` (P8), the already-removed `Results.cfg`/
derived-property duplication/`*_from_shared_fields` factories (P9), the 1685-line GUI
monolith's real responsibilities (P12), and the documented, still-unresolved ~2π residual
in the three named spectrum-from-H implementations (§9.1) are all accurate. No changes
proposed for P1, P2, P4–P9, P13, or §3/§8/§10 as written.

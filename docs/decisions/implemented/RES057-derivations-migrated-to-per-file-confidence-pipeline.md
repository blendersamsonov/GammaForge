# RES057 — *docs/DERIVATIONS.md* is split into `docs/derivations/`, one file per result on a confidence pipeline instead of a lifecycle

Status: implemented
Type: process

## Problem

*docs/DERIVATIONS.md* held five substantially independent physics results (the Gaussian
luminosity overlap integral, flying focus, the cycle-average factor in `ahat`, ellipticity
in the emission kernel, crossing angle in the emission kernel) in one 1165-line file,
mixing settled, code-checked results with two genuinely open research items pending the
author's review. There was no way to cite or track one result's confidence separately
from another's, and the doc-staleness guard (RES055) had no scope covering it at all — its
backticked references were never mechanically checked.

## Decision

*docs/DERIVATIONS.md* is split into `docs/derivations/{status}/DERNNN-topic-title.md`, one
file per result, following the same permanent-id, never-delete convention as decisions
(`docs/decisions/README.md`). Unlike a decision's build lifecycle, a derivation's `status`
is a **confidence pipeline** — `derived` (worked out, unreviewed) → `validated` (a domain
expert checked the algebra) → `verified` (checked against code) — plus `rejected`/
`archived` for the same permanence reasons decisions have them. The skeleton gets
stricter as confidence climbs: `derived` needs only `## Setup`; `validated` adds
`## Result`; `verified` adds `## Verification`.

Five files landed: DER001 (Gaussian overlap, `verified`), DER002 (flying focus,
`verified`), DER003 (the cycle-average factor — a merge of the old file's §C and §0,
which were the same physics derived from opposite ends, `verified`), DER004 (ellipticity
in the emission kernel, `derived`), DER005 (crossing angle in the emission kernel,
`derived`). DER004/DER005 being `derived` rather than higher reflects reality
accurately: both are real, checked algebra, but neither has been reviewed by the author
or checked against production code — that's what's actually blocking §9.2/§9.3, not
missing math.

`tests/test_doc_staleness.py` (RES055) is extended to also check every derivation file
under `docs/derivations/{derived,validated,verified,rejected}/` — the same "an
after-the-fact record is the one place every backtick should resolve" reasoning applies
here for the first time, and it immediately found six pre-existing convention violations
(equation labels and a predecessor filename backticked instead of italicized) that had
never been checked before, now fixed. `tests/test_derivation_format.py` is added,
mechanically enforcing the header/status/heading-set/id-uniqueness/INDEX.md-agreement
rules `docs/derivations/README.md` defines, the same role `test_decision_format.py`
(RES055) plays for decisions.

## Alternatives considered

**Keep one flat file, add a status marker per section.** This is close to what the old
file already did informally (a table at the top marking lettered sections "implemented"
and numbered sections "pending author review"). Rejected for the same reason the decision
log moved off a flat file: no per-result id to cite bare from code, and the reader can't
tell from a citation alone whether the section it points to is settled or open without
opening the whole document and finding the status table.

**A build lifecycle (`proposed`/`implemented`) matching decisions, instead of a
confidence pipeline.** Rejected: a decision's lifecycle tracks whether code was *built*: a
derivation's status tracks whether the *math* is trusted, independent of whether code
exists yet. DER001–DER003 back code that's already shipped; what `verified` records is
that the derivation itself, not the code, has been checked. Conflating the two would make
"verified" ambiguous between "the code works" and "the algebra is right."

**Split §A into several files** (one per subsection: the core overlap integral, crossing
angle, exact 2D form, mean-square `a0`, resolved profiles). Rejected: these are tightly
coupled extensions of one derivation chain, cross-referencing each other constantly
(§A.7 modifies §A.6's approximation; §A.10 validates the whole chain at once) — splitting
them would fragment a single argument for no gain, the opposite of what atomicity is for
here. The five-file split follows the source document's own letter/number top-level
sections, which is where the content was already substantially independent.

## Rationale

The same reasoning that motivated splitting the decision log applies here, adjusted for
what a derivation actually is: an atomic, citable, permanently-addressable record beats a
flat file once there's enough content that a reader wants to jump straight to one result
without reading unrelated ones, and once results have genuinely different confidence
levels that a single-document status table can't make load-bearing at the point of
citation. DER004/DER005's `derived` status is not a demotion — it's an honest,
mechanically-checkable statement of what's actually true today, which the old file's
prose ("pending author review, not implemented") said but nothing enforced.

## Consequences

**Migration fidelity.** Every split file was checked against its source content:
fingerprint searches for distinctive numbers/phrases (DER003, DER005) or full line-level
diffs (DER001, DER002, DER004) confirmed no content was lost, only restructured — new
`## Setup`/`## Result`/`## Verification`/`## Used by` headers inserted, internal
`§A.N`-style subsection labels kept unchanged as they're cited from code and other
derivations, and cross-references between what are now separate files (e.g. "§B" from
inside old §A) rewritten to the new DERNNN bare-citation form. Two genuine content drops
were caught and fixed during review (a dropped sentence in DER001's §A.11, a dropped
synthesis paragraph in DER004) before this decision was recorded.

**Cross-reference sweep.** All *docs/DERIVATIONS.md §X* citations across code
(`laser.py`, `formulas.py`, `test_analytical.py`), decisions (RES039–RES046, RES049, RES053,
RES054), and `docs/GRAND_PLAN.md`'s current-state content (§4.3, the phase-4 table row)
were updated to bare DERNNN/DERNNN §X citations, consistent with RES056. `GRAND_PLAN.md`'s
dated changelog bullets were left untouched, same historical-record reasoning as RES055/RES056
applied to *DECISIONS.md* citations there. `AGENTS.md`'s authoritative-documents list
gains `docs/derivations/INDEX.md` as item 4.

**Guard coverage.** `test_doc_staleness.py` now checks 5 more files it never checked
before; the six pre-existing violations it found are evidence the extension is
doing real work, not just formalizing something already clean.

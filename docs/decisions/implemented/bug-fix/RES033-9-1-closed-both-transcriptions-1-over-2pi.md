# RES033 — §9.1 closed: both transcriptions of the paper's cross-section carry `1/(2 pi)`; the identity harness now expects one

Status: implemented
Class: bug-fix

## Problem

RES026 traced §9.1's ~2π discrepancy to the paper's eq. *(xsec)* and RES025 kept it reported
rather than absorbed until Phase 3a's kernel existed to arbitrate independently. With the
kernel now built (RES029), the question is whether an absolute measurement confirms the
derived factor before applying it.

## Decision

Phase 3b's §9.1 leg is done. `engines.xigma.stages.KERNEL_NORMALIZATION_CONSTANT` is `1.5
/ (2 pi)` and `validation.references.delta.DIFFERENTIAL_PREFACTOR` is `3.0 / (2 pi)` — one
correction, applied at the two places this repo transcribes the paper's differential
cross-section. `NormalizationCheck.expected_ratio` drops its `2 pi` and reports against the
captured fraction alone, so `validation/run.py`'s delta leg is gated at **one** rather than
at a derived `2 pi`. RES025's "report the factor, do not absorb it" is superseded on this
point: the factor is no longer unexplained, so reporting it is no longer the honest option.

**The order this happened in is the whole justification.** RES026 derived the factor from
two elementary integrals, with no code involved. The constants were then set *from that
derivation*. Only afterwards was the discriminating quantity measured: angle-integrate the
table kernel over an 8/gamma cone and compare with Stage 0's elementary `flux x
cross-section x time` photon count. With the uncorrected constant that ratio was `2 pi x
0.9966` on a 49x49 angular grid with 640 `s` bins — the factor and nothing but the factor,
on a path independent of delta's. P14 forbids inserting a constant that makes a test pass;
predicting a constant and then finding the measurement where the prediction put it is the
opposite procedure, and the only one that licenses the change.

**Why an absolute check had to be built for this.** Every Stage-2 test that existed before
this entry — including `run.py`'s fourth identity leg — is a *ratio* between two paths
carrying the same normalization, so all of them were green with the factor present and are
green with it removed. They cannot see normalization at all, by construction (RES029 says so
explicitly, and that was correct). `tests/test_stage1_stage2.py::
test_the_table_kernel_angle_integrates_to_stage_0_total` is the one that can. It runs on a
grid sized for a ten-second test and is therefore loose (+-15%, reading 1.078; refining to
21 angles gives 1.015, and to 21 angles with 480 `s` bins 1.006) — but the distinction it
has to make is between 1 and 6.28, and no plausible grid error touches that.

**One consumer of the fix is `Results` itself, and it was not obvious.** `Collision` fills
`SPECTRUM` from `stages.angle_integrated_spectrum` — Stage 0's closed form, which never
touches the table or its constant — and every angular output from the table kernel. Before
this entry those two paths disagreed by `2 pi` *inside a single `Results` object*, and no
test could see it. `tests/test_xigma_engine.py::
test_the_two_normalization_paths_inside_one_results_object_agree` now integrates
`COLLIMATED_SPECTRUM` back over its angle axes and compares with `SPECTRUM`; it lands at
0.87, the deficit being the auto-range's ~4.6/gamma angular span (a ~94% capture) plus a
25-point trapezoid over a peaked profile.

**What is *not* claimed.** The paper still typesets the uncorrected eq. *(xsec)*, annotated
per RES026. This repo now computes the corrected physics, which is a deliberate, recorded
code/paper divergence rather than the silent kind §0 exists to prevent — a reference
implementation follows the derivation, not the typo.

## Alternatives considered

**Correct only the kernel and leave delta transcribing the paper as typeset.** Keeps delta
a literal transcription, which has some value — but then §11's Phase-3b exit criterion
("the identity harness re-gated against 1.0 rather than 2 pi") is unreachable, and the
suite's headline identity permanently reports a discrepancy that is understood, corrected
elsewhere, and no longer telling anyone anything.

**Fold the factor into `Table.H` at deposition, or into Stage 0's luminosity.** Would make
every downstream number right with one edit, and put the §9.1 constant somewhere §4.2
explicitly says it must not live. Stage 0's total yield is the one quantity here that is
independently, elementarily correct; multiplying it by a cross-section convention would
destroy the arbitration that settled the question in the first place.

## Consequences

**A known unexercised comparison, for Phase 5/7.**
`validation.metrics.compare_slices` compares **absolute** values, not normalized shapes.
The retired *validation.golden.compare_to_golden* distribution path used that metric but
was never enabled by the routine runner. `run_suite` currently applies engine invariance
checks when engines are supplied; the full four-method distribution comparison remains
an explicit Phase-5/7 task.

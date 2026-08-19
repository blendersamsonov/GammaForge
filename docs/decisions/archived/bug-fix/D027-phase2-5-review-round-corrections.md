# D027 — Review-round corrections to Phase 2.5, and one committed file that should not have been

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

A review of `42e9609..HEAD` returned nine findings; all nine were reproduced and fixed.
Four are worth recording because the reasoning, not just the patch, matters.

## Decision

**(a) The window metric's significance threshold was raised 16000x while fixing something
else — reintroducing D023's blindness on the other side.** D023 correctly moved the floor
from a fraction of the *mean window* to a fraction of the *total*, so it stops moving when
the binning is refined. It also, in the same line, changed the fraction from `1e-6` to
`1e-3`, which was never argued for. Measured consequence: a candidate that dropped a real
spectral feature worth 0.07% of the yield scored **exactly zero** on both reported numbers
— a perfect shape match for a lost feature. The two changes are separable and only the
first was justified. The floor is now `SIGNIFICANT_FLUX_FRACTION = 1e-4`, and it serves as
both the significance test and the denominator guard, which keeps the reported number
interpretable at both ends: a window missing entirely reports `1.0`, and spurious flux
against a zero reference reports how many significance-units of it there are.
`tests/test_stage0_delta.py` pins the sensitivity.

**(b) `_captured_fraction` integrated the wrong integrand.** It used the Lorentz factor
alone, giving the tidy `X/(1+X)`, while delta's own integrand carries the polarization
factor too. Since it is the correction the §9.1 arbitration divides by, the error appeared
as a cone-dependent drift in a number that is supposed to be constant. The closed form is
now `1.5 * [X/(1+X) - 1/3 + (1+X)^-2 - (2/3)(1+X)^-3]` (0.917 at four cone widths, against
the 0.941 it claimed). Renamed to `captured_fraction` and made public — it is a statement
about the method, not an implementation detail.

**(c) `check_normalization` integrated bin-centre densities with the trapezoid rule**,
dropping half of the first and last bin. That is why `anchor_ratio` read 0.9937 for a
quantity documented as exactly one, and the same bias sat uncorrected in the §9.1 headline
number. Both integrals are now midpoint sums; the anchor reads 1.000002. With (b) and (c)
fixed, the residual in `deviation` is `+0.41%` at eight cone widths and is **grid
geometry**: the square grid reaches `sqrt(2)` further in its corners than the disc the
correction assumes. A monoenergetic zero-divergence beam reproduces it to within 0.01
percentage points, so it is not beam spread. `identity_checks` now runs at eight cone
widths rather than four, where that residue is a fifth of the gate's budget instead of
three quarters.

**(d) A machine-specific hook file was committed.** `.claude/settings.json` — written by
the graphify tooling, hardcoding `/home/alexander/.local/bin/graphify` as a *PreToolUse*
hook (a Claude Code event name, not a repo symbol) on essentially every tool call — was
swept into a commit by an unreviewed `git add -A`, along with an auto-appended `AGENTS.md`
section asserting the repo "has a knowledge graph at graphify-out/" one commit after that
directory was gitignored. Both are fixed: the settings moved to
`.claude/settings.local.json` (gitignored — `settings.json` is the *shared* file by
convention, so machine-specific hooks do not belong in it), and the `AGENTS.md` claim now
says the graph must be built locally because it is deliberately not committed.

## Alternatives considered

**For (a): keeping `1e-3` and widening the golden tolerance instead.** That treats a
reporting defect as a calibration question. The problem was never that 0.07% passed a 2%
gate — it should — but that the metric announced perfect agreement where there was a real
difference, which makes it useless for tracking drift and a false pass under any tighter
tolerance later.

## Consequences

**The lesson from (d) is the process one:** `git add -A` after running a tool that writes
config is how someone else's checkout ends up invoking a binary that does not exist on
their machine.

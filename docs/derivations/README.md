# Derivations

One derivation, one file. A **derivation** is a mathematical or physical result that
backs specific code — too long for a docstring, and not a *choice* the way a
[decision](../decisions/README.md) is, so it doesn't belong there either. This file
defines where derivations live, the confidence pipeline they move through, and the
in-file format. `tests/test_derivation_format.py` enforces the mechanical parts.

## Layout and naming

`docs/derivations/{status}/DVNNN-topic-title.md`.

- **`DVNNN`** is a permanent, sequential id — distinct from decisions' `DNNN` and the
  plan's principle ids (`PN`) so a bare citation in code is unambiguous at a glance —
  assigned once, never reused.
- **Status** (the folder) is a **confidence level**, not a build lifecycle, and a
  derivation climbs through them as evidence accumulates:
  1. **`derived/`** — worked out, not yet reviewed by a person with the domain expertise
     to catch a wrong assumption. `DER004`/`DER005` are here — pending the author's review.
  2. **`validated/`** — a domain expert has reviewed the algebra and the result; believed
     correct, not yet checked against what the code actually does.
  3. **`verified/`** — checked against code: a test pins the predicted value, a
     closed-form limit matches, an independent numerical method agrees. `DER001`–`DER003`
     are here.
  4. **`rejected/`** — a derivation attempt whose approach turned out wrong. Kept so
     nobody re-derives the same dead end.
  5. **`archived/`** — fully superseded by a later derivation, moved out of the live
     tree, kept because a bare id might already be cited from a code comment.

No class/topic subfolder — this repo's five derivations sit flat within their status
folder; introduce project-specific grouping only if the list grows large enough to need it.

`docs/derivations/INDEX.md` is the map: `id | title | status | path`. Update it in the
same change as any new derivation or any status move.

## The confidence pipeline is not a lifecycle

A decision's lifecycle (`proposed`→`implemented`) tracks whether something was *built*.
A derivation's status tracks whether it's *believed*, independent of whether the code
that uses it exists yet — `DER001`–`DER003` all back code that's already shipped; what
`verified/` records is that the derivation itself, not just the code, has been checked.

Moving up a level is a **fact about verification having happened**, not a promise: don't
move a file to `verified/` because it's about to be checked, move it once the check
exists and passed.

## Citing from code

Same rule as decisions (`docs/decisions/README.md`'s *Citing from code*, RES056): a
comment may cite `(DV0NN)` bare. Keep it a pointer — the formula the code implements,
plus the citation — not the derivation itself.

## Shared notation

The bunch travels along $+\hat{\mathbf z}$ with velocity $\beta_0 c$. The pulse
propagates along $\hat{\mathbf k}$, with transverse focusing axes
$\hat{\mathbf f}_1,\hat{\mathbf f}_2$ and its own longitudinal coordinate
$u=\hat{\mathbf k}\cdot\mathbf r$; head-on $\hat{\mathbf k}=-\hat{\mathbf z}$ and $u=-z$.
Bunch sizes are $\sigma_{ex}(z),\sigma_{ey}(z),\sigma_{ez}$; pulse spot sizes are
$s_1(u),s_2(u)$ and $s_{ct}=c\,\tau$. Crossing angle $\theta$ is measured from exact
counter-propagation, matching `theta_xz`/`theta_yz`.

Math is MathJax (`$…$` / `$$…$$`), so it renders in Obsidian and pastes into
`~/Work/Papers/2026/Compton-Numerics/xigma.tex` with only environment changes.

## The file format

```markdown
# DV0NN — <title>

Status: <status>
```

`Status:` is one of `derived`, `validated`, `verified`, or `rejected — <why, one line>`,
and must agree with the status folder. An archived file adds one more line,
`Archived: YYYY-MM-DD`, and otherwise keeps whichever status it had before archiving.

### Body skeleton — stricter as confidence grows

```markdown
## Setup                    <- always: what's given, what's being derived, key assumptions
…bespoke derivation sections, free-form, MathJax…
## Result                   <- required from validated/ onward: the formula, stated cleanly
## Verification             <- required from verified/ onward: what evidence backs it, and how
## Used by                  <- optional: pointers to the code/decisions that depend on this
```

A `derived/` file only needs `## Setup`. `validated/` adds `## Result`. `verified/` adds
`## Verification`, since that's the entire content of the claim "checked against code" —
a file sitting in `verified/` with nothing describing what was checked is a gap the
format test catches. `rejected/` keeps whatever it had at the point it was abandoned; the
verdict lives on the `Status:` line.

**Worked examples in this repo.** `DER003-cycle-average-factor-in-ahat.md` is a merge of
two things that turned out to be the same physics derived from opposite ends — the
original notes said "read the derivation first if you want the derivation, the
discrepancy report first if you want the history"; the merged file states the physics
once and keeps both the derivation and the history under `## Setup`/`## Verification`,
rather than two files disagreeing about which is authoritative. `DER004` and `DER005` show
the `derived/` status in practice: real, checked algebra with no `## Verification`
required yet, because what's actually blocking them is the author's review, not more
math — see each file's `## Used by` for exactly what code changes once that happens.

Superseding, amendments, and the parallel-branch id-collision escape hatch all work
exactly as they do for decisions — see `docs/decisions/README.md`, same mechanisms.

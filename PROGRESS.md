# GammaForge — Implementation Progress

Tracks what has actually been built, session by session. `docs/GRAND_PLAN.md` is the
plan (with its own changelog of *design* decisions); this file is the log of *execution*
against that plan. Update it every session — append, don't rewrite history.

Phase numbers/names match `docs/GRAND_PLAN.md` §11.

---

## Status at a glance

| Phase | Status |
|-------|--------|
| 0. Scaffold | 🟢 done (pending only the pre-3a/6 worktree re-check, C4) |
| 1. Core | ⚪ not started |
| 2. Validation harness | ⚪ not started |
| 2.5. Stage 0 + minimal delta | ⚪ not started |
| 3a. xigma engineering | ⚪ not started |
| 3b. Physics closure | ⚪ not started |
| 4. analytical engine | ⚪ not started |
| 5. kascade port + delta full role | ⚪ not started |
| 6. GUI | ⚪ not started |
| 7. Validation completion | ⚪ not started |
| 8. Polish | ⚪ not started |

---

## 2026-08-07 — Plan v0.8 + Phase 0 kickoff

**Plan changes (before any code):**
- Grand plan reviewed end-to-end across several rounds (v0.2 → v0.7): fact-checked
  against the predecessor repo (`ComptonSuite`) and the physics paper, phase-ordering
  bug fixed (Phase 2.5 no longer depends on Stage 0 before Stage 0 exists — Stage 0 +
  the shared chunking utility moved into 2.5), RNG substream architecture pinned so the
  bunch resample rule doesn't silently break the `REUSE_INTERMEDIATES` cost tier.
- v0.8: laser field source made pluggable — new `LaserField` protocol is what engines
  depend on; `GaussianParaxialLaser` reframed as its first implementation, not *the*
  laser type; `fit_gaussian_paraxial` added (P8-style: descriptive Gaussian metrics —
  waist, Rayleigh range — extracted from *any* `LaserField`, not just an analytic one).
  Anticipates `~/Work/Code/Spectral-FEM-Fields` (sibling C++ project, Python bindings
  planned) as a future second implementation for arbitrary non-paraxial pulses, at the
  interface level only — no work on it yet.

**Phase 0 work (this session):**
- `pyproject.toml`: hatchling build backend, src layout, Python ≥3.12, core deps
  (numpy/pint/h5py/pyyaml/matplotlib), optional extras `gpu` (cupy) / `jit` (numba) /
  `dev` (pytest). Rationale for hatchling over setuptools: `DECISIONS.md` D001.
- Package skeleton: `src/gammaforge/{io,engines/{xigma,analytical,kascade},
  validation/references,gui}/`, each a docstring-only `__init__.py` pointing back at
  its `GRAND_PLAN.md` section. Deliberately no placeholder module files yet —
  `DECISIONS.md` D003.
- `.gitignore` hardened: `*.ele`/`*.bun`/`*.h5`/`*.hdf5` and sync-conflict filename
  patterns, per C3 (the predecessor accumulated multi-MB committed artifacts this way).
- `DECISIONS.md` started (C1) — D001–D003 so far, with an explicit backticks-vs-italics
  convention so the doc-staleness guard can trust every backtick literally.
- `tests/test_smoke.py`: package + all subpackages import.
- `tests/test_doc_staleness.py` (C2): resolves backticked tokens in `DECISIONS.md`
  against real repo files / package symbols / builtins. Scope is `DECISIONS.md` only,
  not all of `docs/*.md` as originally checklisted below — `GRAND_PLAN.md` is a
  forward-looking roadmap and would fail the check by design, not by drift; see
  `DECISIONS.md` D002 for the full reasoning.
- `AGENTS.md` written (repo orientation + the plan's easy-to-violate rules, condensed
  for a fresh agent) with `CLAUDE.md` as a symlink to it.
- Verified clean: `python -m venv .venv && pip install -e .` succeeds, `pytest` → 3
  passed.

**Not done / explicitly deferred:**
- C4's `git branch -a` re-check on `ComptonSuite`'s worktree branches — not needed
  until Phase 3a/6 kickoff, already audited once (see `GRAND_PLAN.md` §11 note); just a
  reminder not to skip re-checking then.
- No CI wiring yet (the doc-staleness test and any future import-boundary/mypy checks
  run locally via `pytest` only). Add a CI workflow when there's a remote to run it on.

### Phase 0 checklist

- [x] `pyproject.toml` (Python 3.12, package skeleton, pytest)
- [x] `src/gammaforge/{io,engines/{xigma,analytical,kascade},validation,gui}/` skeleton
- [x] `.gitignore` hardened for `.ele` files and sync-conflict patterns (C3)
- [x] `DECISIONS.md` provenance doc started (C1)
- [x] `pytest` green on empty-suite smoke test
- [x] `pip install -e .` works
- [x] Doc-staleness guard scaffolding (C2) — scoped to `DECISIONS.md`, see above
- [x] `AGENTS.md` + `CLAUDE.md` symlink
- [ ] `git branch -a` re-check on `ComptonSuite` before Phase 3a/6 kickoff (C4) — not
      needed yet, noted here so it isn't forgotten

---

## How to update this file

- One dated section per work session (or per meaningful chunk of a session).
- State what actually landed, not what's planned — the plan lives in `GRAND_PLAN.md`.
- If a plan decision changes mid-implementation, record it in `GRAND_PLAN.md`'s own
  changelog (bump the version), and just link back to it here (`plan bumped to vX.Y —
  see its changelog`) rather than duplicating the rationale.
- Flag blockers explicitly (e.g. "waiting on author for §9.1" or "waiting on
  Spectral-FEM-Fields Python bindings") so the next session doesn't have to rediscover
  them.

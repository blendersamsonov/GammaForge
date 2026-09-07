# 04 — Correct negative last-emission times

Read [coordination rules](README.md). Covers A06. This is a small, independently
startable bookkeeping fix, not an extension of kascade's physics.

## Evidence and entry points

`src/gammaforge/engines/kascade/solver.py` reduces photon times with
`time_last = np.zeros(n_electrons)` followed by `np.maximum.at`. All emission
times before t=0 therefore produce the wrong last time. The adapter in
`engines/kascade/engine.py` reconstructs final positions from that value.

The prior audit used a synthetic 64-electron rate grid over [-2 ps, -1 ps],
cumulative rates [0, 100], quantum disabled and one allowed photon. Emissions
occurred near -2 ps, but every last-emission time was zero. Rebuild a tiny
deterministic fixture from the solver's current typed inputs; don't depend on a
rare event in a physical Monte Carlo run. Start with `tests/test_kascade_engine.py`
and RES059.

## Work and acceptance

- Separate “no emission” from the initial value of a maximum reduction. Preserve
  the currently intended no-emitter final-time/position convention explicitly;
  do not let a sentinel leak into exported positions.
- Assert each emitting parent's last time equals the maximum of its own photon
  times for all-negative, mixed-sign, positive and repeated-parent histories.
- Test nonemitters, zero photons, empty input, and final-position reconstruction
  using the selected time convention.
- Keep random draws, event ordering, recoil calculations, weights and photon
  yields unchanged. Seeded outputs other than the erroneous bookkeeping should
  remain unchanged.

Own solver changes and narrowly scoped tests. Coordinate any adapter change with
02, which owns histogram assembly in the same file. Tell 05 exactly where final
time metadata lives so it can be preserved without deciding the future
ElectronMacroparticles-vs-Bunch design. Run the kascade tests and affected format
tests; no new backend, engine capability or generic event framework is needed.

Starting check: `.venv/bin/pytest -q tests/test_kascade_engine.py tests/test_formats.py`.

## Resolution

- **Status**: Done.
- **Solver fix (`src/gammaforge/engines/kascade/solver.py`)**:
  Separated "no emission" from the initial reduction value by initializing emitting
  parents `time_last[parent] = -np.inf` before `np.maximum.at(time_last, parent, time)`.
  Non-emitters retain `0.0` (stay at the focus at $t=0.0$).
- **Adapter fix (`src/gammaforge/engines/kascade/engine.py`)**:
  Explicitly uses `t_final = np.where(raw.n_photons > 0, raw.time_last_emit, 0.0)`
  so non-emitters remain at initial coordinates and no sentinels can leak into `Bunch` coordinates.
- **Metadata for Session 05**:
  Final emission time and count metadata live in `results.electrons.meta["time_last_emit"]`
  (`np.ndarray[float]`, shape `(n_particles,)`) and `results.electrons.meta["n_photons"]`
  (`np.ndarray[int]`, shape `(n_particles,)`).
- **Tests added (`tests/test_kascade_engine.py`)**:
  - `test_negative_emission_times_are_preserved_in_last_emission_reduction`: synthetic 64-electron audit fixture with emissions in `[-2 ps, -1 ps]`.
  - `test_time_last_emit_matches_parent_maximum_across_histories`: all-negative, mixed-sign, all-positive, and repeated-parent histories (`max_photons > 1`).
  - `test_time_last_emit_handles_nonemitters_zero_photons_and_empty_input`: partial emission, zero photons overall, and 0-particle input.
  - `test_final_electron_position_reconstruction_preserves_nonemitters_and_avoids_sentinels`: verifies non-emitter initial position preservation and finite coordinates.

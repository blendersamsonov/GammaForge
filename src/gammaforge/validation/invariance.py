"""Invariance properties: things that must not change an answer.

Four of them, and each is here because it has already gone wrong somewhere:

* **chunk size** — a chunked reduction that drops or double-counts a boundary is invisible
  in every other test;
* **prefilter on/off** — the active-region filter is a pure optimization (§3.2), so it is
  either exactly neutral or it is a bug;
* **backend** (numpy / cupy / numba) — to a tight *relative* tolerance, never bit-equality:
  GPU float32 against CPU float64 makes exact agreement the wrong thing to demand;
* **seed** — same seed and same physical parameters give identical results, a user-facing
  reproducibility guarantee (§3.5), not an implementation detail.

Each check is a plain function over an `Engine`, so it applies to every engine without any
of them knowing this module exists. The two properties that need no engine at all — bunch
seed determinism, and the prefilter discarding only particles the pulse never reaches —
are here too, and they are what makes the mechanism exercisable today, before there is an
engine to run the rest against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from ..engines.base import Engine
from ..io.bunch import Bunch, overlap_time_window
from ..io.interaction import PREFILTER_OFF
from ..io.laser import LaserField, fit_gaussian_paraxial
from ..io.results import Results
from ..io.schema import Parameters
from ..io.units import C_CGS
from .metrics import compare_slices
from .runners import run_engine
from .scenarios import Scenario, build

__all__ = [
    "Check",
    "compare_results",
    "check_bunch_seed_determinism",
    "check_prefilter_discards_only_dark_particles",
    "check_seed_determinism",
    "check_prefilter_invariance",
    "check_chunk_invariance",
    "check_backend_agreement",
    "core_checks",
    "engine_checks",
    "EXACT_TOLERANCE",
    "BACKEND_TOLERANCE",
]

#: Two runs differing only in a bookkeeping knob should agree to round-off, not to a
#: physics tolerance. Loose enough for a different summation order, far tighter than any
#: real disagreement.
EXACT_TOLERANCE = 1e-12

#: Backends genuinely differ — float32 on a GPU against float64 on a CPU. §7 pins ~1e-6.
BACKEND_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Check:
    """One property, whether it held, and enough detail to act on a failure."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'ok  ' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def compare_results(results: Results, reference: Results, tolerance: float) -> tuple[bool, str]:
    """Worst per-slice deviation between two `Results`, against one tolerance.

    A slice present in one and not the other fails rather than being skipped: these are two
    runs of the *same* engine differing only in a knob that must not matter, so a change in
    which outputs came back is itself the defect.
    """
    if set(results.photon_slices) != set(reference.photon_slices):
        return False, (
            f"different outputs: {sorted(str(k) for k in results.photon_slices)} vs "
            f"{sorted(str(k) for k in reference.photon_slices)}"
        )
    worst_kind, worst = None, 0.0
    for kind, reference_slice in reference.photon_slices.items():
        deviation = compare_slices(results.photon_slices[kind], reference_slice).worst()
        if worst_kind is None or deviation > worst:
            worst_kind, worst = kind, deviation
    where = f", worst in {getattr(worst_kind, 'value', worst_kind)}" if worst_kind is not None else ""
    return worst <= tolerance, f"max deviation {worst:.3e} vs tolerance {tolerance:.1e}{where}"


# ---------------------------------------------------------------------------
# Properties of the shared core — no engine required
# ---------------------------------------------------------------------------
def check_bunch_seed_determinism(scenario: Scenario) -> Check:
    """Same seed, same parameters, same bunch — and a different seed, a different one.

    The negative half matters as much as the positive one: a sampler that ignored its seed
    entirely would pass the first assertion perfectly.
    """
    spec = scenario.sampling
    first = build(scenario, spec).bunch
    again = build(scenario, spec).bunch
    other = build(scenario, replace(spec, seed=spec.seed + 1)).bunch

    name = f"{scenario.name} bunch seed determinism"
    if not _bunches_identical(first, again):
        return Check(name, False, "two builds with the same seed produced different bunches")
    if _bunches_identical(first, other):
        return Check(name, False, f"seeds {spec.seed} and {spec.seed + 1} produced the same bunch")
    return Check(name, True, f"seed {spec.seed} reproduces exactly; seed {spec.seed + 1} differs")


def check_prefilter_discards_only_dark_particles(
    scenario: Scenario, n_times: int = 512
) -> Check:
    """No discarded macroparticle ever sees an ``a0`` above the prefilter threshold.

    The prefilter's claim is that it is a pure optimization, and the half of that claim it
    owns alone is this one: whatever it threw away contributed nothing. Checking it
    directly — sample the laser along each discarded trajectory and take the maximum —
    is independent of the closed-form geometry that decided to discard it, so an error in
    that algebra shows up here rather than as a mysterious few-percent yield shift later.

    The other half, that keeping them changes no engine's answer, is
    `check_prefilter_invariance`.
    """
    name = f"{scenario.name} prefilter discards only dark particles"
    threshold = scenario.sampling.prefilter
    if threshold <= PREFILTER_OFF:
        return Check(name, True, "scenario runs with the prefilter off; nothing to check")

    unfiltered = build(scenario, replace(scenario.sampling, prefilter=PREFILTER_OFF)).bunch
    t0, t1 = overlap_time_window(unfiltered, scenario.laser, threshold)
    discarded = unfiltered.select(t0 > t1)
    if discarded.n_particles == 0:
        return Check(name, True,
                     f"the pulse reaches all {unfiltered.n_particles} macroparticles; "
                     f"nothing was discarded")

    limit = threshold * fit_gaussian_paraxial(scenario.laser).a0_peak()
    peak = _peak_a0_along_trajectories(discarded, scenario.laser, threshold, n_times)
    passed = bool(peak <= limit)
    return Check(
        name, passed,
        f"{discarded.n_particles} of {unfiltered.n_particles} discarded; their peak a0 is "
        f"{peak:.3e} against a threshold of {limit:.3e}",
    )


#: Elements per vectorized block when sampling trajectories. Particles times time samples
#: can be tens of millions across the bank, and this check is not worth a large allocation.
_SAMPLE_BLOCK = 1 << 20


def _peak_a0_along_trajectories(
    bunch: Bunch, laser: LaserField, threshold: float, n_times: int
) -> float:
    """Largest ``a0`` any of these particles sees, sampled over its own encounter window.

    Each particle gets its **own** time grid, derived from physics rather than from the
    geometry under test: the pulse centre reaches the particle where ``u(t) = c t``, and
    the longitudinal Gaussian alone puts ``a0`` under the threshold once
    ``|u - c t| > half_length``. So sampling ``|t - t_meet| <= half_length / |c - v_par|``
    covers everything that could clear the threshold, at a resolution set by the pulse
    length instead of by the bunch length — a single shared grid would step over a short
    pulse entirely and report a reassuring number for the wrong reason.

    Straight-line ultrarelativistic motion (§2.3) throughout.
    """
    region = laser.active_region(threshold)
    axis = np.asarray(region.axis, dtype=float)
    origin = np.asarray(region.origin, dtype=float)

    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    vx, vy, vz = C_CGS * bunch.thx / norm, C_CGS * bunch.thy / norm, C_CGS / norm
    u0 = ((bunch.x - origin[0]) * axis[0] + (bunch.y - origin[1]) * axis[1]
          + (bunch.z - origin[2]) * axis[2])
    v_par = vx * axis[0] + vy * axis[1] + vz * axis[2]

    # Closing speed along the axis. It vanishes only for a particle riding along with the
    # pulse, which never leaves it — the prefilter keeps such a particle, so none is in
    # this bunch, and the guard exists only to keep the arithmetic finite.
    closing = C_CGS - v_par
    closing = np.where(np.abs(closing) < 1e-30, 1e-30, closing)
    t_meet = u0 / closing
    half_window = region.half_length / np.abs(closing)

    offsets = np.linspace(-1.0, 1.0, n_times)
    block = max(1, _SAMPLE_BLOCK // n_times)
    peak = 0.0
    for start in range(0, bunch.n_particles, block):
        stop = min(start + block, bunch.n_particles)
        window = slice(start, stop)
        times = t_meet[window, None] + half_window[window, None] * offsets[None, :]
        a0 = laser.a0_profile(
            bunch.x[window, None] + vx[window, None] * times,
            bunch.y[window, None] + vy[window, None] * times,
            bunch.z[window, None] + vz[window, None] * times,
            times,
        )
        peak = max(peak, float(np.max(a0)))
    return peak


def core_checks(scenarios: Iterable[Scenario]) -> list[Check]:
    """Every invariance property that holds of `gammaforge.io` alone."""
    checks: list[Check] = []
    for scenario in scenarios:
        checks.append(check_bunch_seed_determinism(scenario))
        checks.append(check_prefilter_discards_only_dark_particles(scenario))
    return checks


# ---------------------------------------------------------------------------
# Properties that need an engine
# ---------------------------------------------------------------------------
def check_seed_determinism(engine: Engine, scenario: Scenario,
                           params: Parameters | None = None) -> Check:
    """Same seed → identical results, with beam and laser held fixed (§3.5)."""
    first = run_engine(engine, scenario, params).results
    again = run_engine(engine, scenario, params).results
    passed, detail = compare_results(again, first, EXACT_TOLERANCE)
    return Check(f"{engine.name}/{scenario.name} seed determinism", passed, detail)


def check_prefilter_invariance(engine: Engine, scenario: Scenario,
                               params: Parameters | None = None,
                               tolerance: float = EXACT_TOLERANCE) -> Check:
    """Prefiltering is a pure optimization: on and off must agree at the same seed."""
    name = f"{engine.name}/{scenario.name} prefilter invariance"
    spec = scenario.sampling
    if spec.prefilter <= PREFILTER_OFF:
        return Check(name, True, "scenario runs with the prefilter off; nothing to compare")
    with_filter = run_engine(engine, scenario, params, sampling=spec).results
    without = run_engine(engine, scenario, params,
                         sampling=replace(spec, prefilter=PREFILTER_OFF)).results
    passed, detail = compare_results(with_filter, without, tolerance)
    return Check(name, passed, detail)


def check_chunk_invariance(engine: Engine, scenario: Scenario, field: str,
                           values: Sequence[int], params: Parameters | None = None,
                           tolerance: float = EXACT_TOLERANCE) -> Check:
    """Results do not depend on how the work was split into chunks.

    ``field`` is the engine's own chunk-size parameter key: engines name their knobs, the
    harness does not (P10). Every value is compared against the first, so a failure names
    the chunk size that broke it rather than only that some pair disagreed.
    """
    return _sweep(f"{engine.name}/{scenario.name} chunk invariance",
                  engine, scenario, params, field, values, tolerance)


def check_backend_agreement(engine: Engine, scenario: Scenario, field: str,
                            values: Sequence[str], params: Parameters | None = None,
                            tolerance: float = BACKEND_TOLERANCE) -> Check:
    """Backends agree to a relative tolerance — never bit-identically (§7)."""
    return _sweep(f"{engine.name}/{scenario.name} backend agreement",
                  engine, scenario, params, field, values, tolerance)


def _sweep(name: str, engine: Engine, scenario: Scenario, params: Parameters | None,
           field: str, values: Sequence, tolerance: float) -> Check:
    """Run one engine parameter across ``values``; compare every result to the first."""
    base = params if params is not None else engine.schema
    if field not in {spec.key for spec in base.specs}:
        return Check(name, False, f"{engine.name} has no parameter {field!r} to vary")
    if len(values) < 2:
        return Check(name, False, f"need at least two values of {field!r}, got {list(values)}")

    reference = run_engine(engine, scenario, base.with_values(**{field: values[0]})).results
    failures = []
    for value in values[1:]:
        results = run_engine(engine, scenario, base.with_values(**{field: value})).results
        passed, detail = compare_results(results, reference, tolerance)
        if not passed:
            failures.append(f"{field}={value}: {detail}")
    if failures:
        return Check(name, False, "; ".join(failures))
    return Check(name, True, f"{field} over {list(values)} agrees within {tolerance:.1e}")


def engine_checks(engine: Engine, scenarios: Iterable[Scenario],
                  params: Parameters | None = None) -> list[Check]:
    """The invariance legs that need no engine-specific knob names.

    Chunk size and backend are deliberately absent: their parameter keys belong to the
    engine, so a caller that knows the engine passes them to `check_chunk_invariance` and
    `check_backend_agreement` itself.
    """
    checks: list[Check] = []
    for scenario in scenarios:
        checks.append(check_seed_determinism(engine, scenario, params))
        checks.append(check_prefilter_invariance(engine, scenario, params))
    return checks


def _bunches_identical(first: Bunch, second: Bunch) -> bool:
    if first.n_particles != second.n_particles:
        return False
    return all(np.array_equal(left, right) for left, right in zip(first.arrays(), second.arrays()))

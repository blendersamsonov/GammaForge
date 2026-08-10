"""The validation suite's entry point: run everything runnable, report pass/fail (§7).

::

    python -m gammaforge.validation.run

Three sections today, because engines arrive in later phases and the suite should be
honest about what it did rather than silently doing less:

* **core invariants** — properties of `gammaforge.io` alone (bunch seed determinism, the
  prefilter discarding only particles the pulse never reaches). These run now.
* **identities** — methods that must agree on the same number: Stage 0's total yield, the
  closed-form single-electron spectrum, and delta's brute-force angular integral. All
  three now agree on 1: the third disagreed by exactly ``2 pi`` until Phase 3b closed §9.1
  (`DECISIONS.md` D033), and this leg is what would catch that factor coming back. A
  fourth leg compares Stage 2's own table kernel against delta at one point; both carry
  the same §9.1 factor, so it read ~1 before the closure and reads ~1 after — it checks
  deposition and interpolation, deliberately not the normalization.
* **goldens** — every committed snapshot is loaded and its stored closed-form scalars are
  compared against what this repo computes for the same scenario. This is a real
  cross-implementation check with no engine in it: two independent codebases, the same
  physical input, the same number expected out.

The engine sections (cross-engine consistency, chunk/backend/prefilter invariance, the
closed-form identity tests) are `invariance.engine_checks` and `golden.compare_to_golden`
applied to a registered engine. They are not stubbed here: an empty engine list prints as
an empty engine list, and the section appears when an engine does. **`XigmaEngine`
exists** (Phase 3a, `engines/xigma/engine.py`) but is not passed to `run_suite` by
`main()` below: the scenario bank's `_DEFAULT_OUTPUTS` resolution
(`COLLIMATED_SPECTRUM` at 64x16x16, tuned for the predecessor's GPU importance sampler)
takes tens of seconds per slice against this phase's numpy brute-force kernel
(`DECISIONS.md` D031) — real for a deliberate Calculate (§12), not for a suite run meant
to be exercised routinely. `tests/test_xigma_engine.py` and `tests/test_stage1_stage2.py`
exercise it at a suite-appropriate scale instead.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from ..engines.base import Engine
from .golden import available_goldens, compare_to_golden, load_golden
from .invariance import Check, core_checks, engine_checks
from .runners import derived_scalars, run_engine
from .scenarios import SCENARIOS, Scenario

__all__ = ["Report", "run_suite", "golden_scalar_checks", "identity_checks", "main",
           "SCALAR_TOLERANCE", "IDENTITY_PARTICLES"]


class Report:
    """Accumulates lines and a pass/fail verdict, so a section can report as it goes."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"--- {title} " + "-" * max(0, 60 - len(title)))

    def check(self, check: Check) -> None:
        self.lines.append(str(check))
        self.failures += not check.passed

    def note(self, text: str) -> None:
        self.lines.append(text)

    def verdict(self) -> str:
        return "ALL CHECKS PASS" if self.failures == 0 else f"{self.failures} CHECK(S) FAILED"

    def __str__(self) -> str:
        return "\n".join([*self.lines, "", "=" * 62, self.verdict(), "=" * 62])


#: How closely a golden's stored closed-form scalars must match this repo's own. Tight by
#: design: these are not Monte-Carlo numbers. Both repos compute them analytically from the
#: same physical inputs, so a disagreement means a constant, a unit or a convention differs
#: — worth failing over, unlike a spectrum shape.
SCALAR_TOLERANCE = 1e-9


def golden_scalar_checks(scenarios: Sequence[Scenario]) -> list[Check]:
    """Compare each golden's stored closed-form scalars against this repo's own.

    Scalars are recomputed from the `Scenario` **objects passed in**, not from the bank
    entry that happens to share a name — a caller comparing a modified scenario must be
    told about *its* numbers.
    """
    available = available_goldens()
    checks = []
    for scenario in scenarios:
        fresh = derived_scalars(scenario)
        for model in sorted(model for name, model in available if name == scenario.name):
            golden = load_golden(scenario.name, model)
            for name, reference in sorted(golden.scalars.items()):
                if name not in fresh:
                    continue
                error = (abs(fresh[name] - reference) / abs(reference) if reference
                         else abs(fresh[name]))
                checks.append(Check(
                    name=f"{scenario.name}/{model} {name}",
                    passed=error <= SCALAR_TOLERANCE,
                    detail=f"{fresh[name]:.12g} vs golden {reference:.12g} ({error:.2e})",
                ))
    return checks


#: Macroparticles the identity harness runs on. It is a normalization check, not a
#: production run — the ratios it reports converge long before the statistics do.
IDENTITY_PARTICLES = 2000


def identity_checks(scenarios: Sequence[Scenario]) -> list[Check]:
    """The §7 identity harness: methods that must agree, and by how much they do not.

    Four legs exist today. Stage 0's total yield is an elementary
    ``flux x cross-section x time`` count; the closed-form single-electron spectrum
    integrates to that same number as an identity; delta's angular integral is the
    independent brute-force path. All three agree on one.

    **The delta leg is the §9.1 tripwire, and it changed meaning in Phase 3b.** It used to
    report against a *derived* ``2 pi`` — green while the question was open, red if the
    ratio moved — because encoding "expected 1.0" would have meant either a permanently red
    suite or a constant nobody had justified (P14). D026 derived the factor and D033 applied
    it at both transcriptions of the paper's cross-section, so the leg now reads what an
    identity should read. What it watches is unchanged: a ratio that walks away from its
    expected value, which after the closure includes a return to ``2 pi``.

    The fourth leg (Phase 3a) is Stage 2's own table kernel against delta, at one
    observation point. Both sides carry the *same* §9.1 factor, before and after the
    closure, so this ratio was ~1 throughout and is deliberately blind to the
    normalization — what it exercises is `stages.deposit_shape_table`/
    `stages.retarget_ahat`/`angular_spectrum_from_table`, at a scale
    `python -m gammaforge.validation.run` can afford (see the module docstring for why
    the engine itself is not wired in at full scenario-bank resolution yet). The check that
    the *absolute* kernel normalization is right — angle-integrating the table kernel and
    comparing with Stage 0's count, the same arbitration delta gets here — needs a fine
    ``s`` grid to converge and lives in
    `tests/test_stage1_stage2.py::test_the_table_kernel_angle_integrates_to_stage_0_total`.
    """
    from ..engines.xigma.stages import (
        angular_spectrum_from_table,
        deposit_shape_table,
        integrate_trajectories,
        retarget_ahat,
    )
    from .references import delta
    from .references.delta import resonance_spectrum
    from .scenarios import build

    checks = []
    for scenario in scenarios:
        interaction = build(
            scenario, replace(scenario.sampling, n_particles=IDENTITY_PARTICLES)
        )
        samples = integrate_trajectories(
            interaction.bunch, interaction.laser, interaction.N_e, n_steps=64
        )
        # Eight cone widths, not the module default of four: the square grid's corners
        # leave a positive residue in `deviation` that shrinks with the cone, and at four
        # it is +1.5% — three quarters of this gate's budget spent on grid geometry rather
        # than on the normalization the gate is watching. At eight it is +0.4%, for about
        # a second per scenario.
        normalization = delta.check_normalization(samples, n_angles=49, cone_factor=8.0)
        checks.append(Check(
            name=f"{scenario.name} closed form = Stage 0 total",
            passed=abs(normalization.anchor_ratio - 1.0) <= 1e-2,
            detail=f"anchor ratio {normalization.anchor_ratio:.6f} (identity, up to binning)",
        ))
        checks.append(Check(
            name=f"{scenario.name} delta/Stage 0 = 1 (\u00a79.1 closed, D033)",
            passed=abs(normalization.deviation) <= 2e-2,
            detail=normalization.summary(),
        ))

        # Fourth leg: Stage 2's table kernel against delta's particle-based histogram, at
        # one observation point. Both carry the *same* section-9.1 factor
        # (`stages.py`'s `KERNEL_NORMALIZATION_CONSTANT` and delta's `DIFFERENTIAL_PREFACTOR`
        # are the same correction applied to the same equation), so this ratio read ~1
        # before D033 and reads ~1 after: it checks deposition and interpolation, and is
        # deliberately insensitive to the normalization the leg above watches.
        # CIC, not the default `nearest`: evaluating exactly at the beam centre
        # aliases against a nearest-deposited table's own cell boundaries
        # (`tests/test_stage1_stage2.py` measured 0.5x-1.7x at nearest with 40-100 theta
        # bins; CIC holds within a few percent).
        shape_table = deposit_shape_table(samples, n_bins=(32, 48, 48, 64), scheme="cic")
        table = retarget_ahat(shape_table, samples.intensity_peak)
        edge = float(np.max(samples.gamma) ** 2)
        s_edges = np.linspace(0.0, 1.05 * edge, 150)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        kernel_total = float(np.sum(angular_spectrum_from_table(table, [0.0], [0.0], s_centers)))
        delta_total = float(np.sum(resonance_spectrum(samples, s_edges, 0.0, 0.0)))
        ratio = kernel_total / delta_total if delta_total else math.inf
        checks.append(Check(
            name=f"{scenario.name} Stage 2 kernel = delta (single point, pi-agnostic)",
            passed=abs(ratio - 1.0) <= 0.1,
            detail=f"kernel/delta = {ratio:.4f} at (theta_x, theta_y) = (0, 0)",
        ))
    return checks


def run_suite(engines: Iterable[Engine] = (), scenarios: Sequence[Scenario] = SCENARIOS) -> Report:
    report = Report()
    engines = list(engines)

    report.section("core invariants (gammaforge.io)")
    for check in core_checks(scenarios):
        report.check(check)

    report.section("identities (Stage 0 / closed form / delta)")
    for check in identity_checks(scenarios):
        report.check(check)

    report.section("golden scalars (predecessor vs this repo)")
    scalar_checks = golden_scalar_checks(scenarios)
    if not scalar_checks:
        report.note("no goldens committed — run `python -m gammaforge.validation.make_references`")
    for check in scalar_checks:
        report.check(check)

    report.section("engines")
    if not engines:
        report.note(
            "no engines passed to run_suite() — `main()` below does not pass XigmaEngine "
            "by default (see the module docstring: full scenario-bank resolution is slow "
            "against this phase's numpy kernel); pass engines=[...] to exercise this section"
        )
        return report

    for engine in engines:
        for check in engine_checks(engine, scenarios):
            report.check(check)
        for scenario in scenarios:
            models = sorted(model for name, model in available_goldens() if name == scenario.name)
            for model in models:
                golden = load_golden(scenario.name, model)
                comparison = compare_to_golden(
                    run_engine(engine, scenario).results,
                    golden,
                    scalars=derived_scalars(scenario),
                )
                report.check(Check(
                    name=f"{engine.name}/{scenario.name} vs golden {model}",
                    passed=comparison.passed,
                    detail=comparison.summary().split(": ", 1)[1],
                ))
    return report


def main(argv=None) -> int:
    report = run_suite()
    print(report)
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

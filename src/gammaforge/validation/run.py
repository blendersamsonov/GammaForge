"""The validation suite's entry point: run everything runnable, report pass/fail (§7).

::

    python -m gammaforge.validation.run

Three sections today, because engines arrive in later phases and the suite should be
honest about what it did rather than silently doing less:

* **core invariants** — properties of `gammaforge.io` alone (bunch seed determinism, the
  prefilter discarding only particles the pulse never reaches). These run now.
* **identities** — methods that must agree on the same number: Stage 0's total yield, the
  closed-form single-electron spectrum, and delta's brute-force angular integral. The
  third disagrees by exactly ``2 pi`` (§9.1, open) and is reported against that derived
  value rather than against 1.
* **goldens** — every committed snapshot is loaded and its stored closed-form scalars are
  compared against what this repo computes for the same scenario. This is a real
  cross-implementation check with no engine in it: two independent codebases, the same
  physical input, the same number expected out.

The engine sections (cross-engine consistency, chunk/backend/prefilter invariance, the
closed-form identity tests) are `invariance.engine_checks` and `golden.compare_to_golden`
applied to a registered engine. They are not stubbed here: an empty engine list prints as
an empty engine list, and the section appears when an engine does.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Iterable, Sequence

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

    Three legs exist today. Stage 0's total yield is an elementary
    ``flux x cross-section x time`` count; the closed-form single-electron spectrum
    integrates to that same number as an identity; delta's angular integral is the
    independent brute-force path. The first two agree exactly, and delta comes out at
    ``2 pi`` — the §9.1 discrepancy, derived rather than observed (see
    `gammaforge.validation.references.delta`).

    The 2 pi leg is reported against its **derived** value, so this section stays green
    while the question is open and turns red if the ratio ever moves. Encoding it as
    "expected 1.0, fails" would make the suite permanently red and therefore ignored;
    encoding it as "expected 1.0, passes" would require pasting in a constant nobody has
    justified (P14). Neither is what a harness is for.
    """
    from ..engines.xigma.stages import integrate_trajectories
    from .references import delta
    from .scenarios import build

    checks = []
    for scenario in scenarios:
        interaction = build(
            scenario, replace(scenario.sampling, n_particles=IDENTITY_PARTICLES)
        )
        samples = integrate_trajectories(
            interaction.bunch, interaction.laser, interaction.N_e, n_steps=64
        )
        normalization = delta.check_normalization(samples, n_angles=33, cone_factor=4.0)
        checks.append(Check(
            name=f"{scenario.name} closed form = Stage 0 total",
            passed=abs(normalization.anchor_ratio - 1.0) <= 1e-2,
            detail=f"anchor ratio {normalization.anchor_ratio:.6f} (identity, up to binning)",
        ))
        checks.append(Check(
            name=f"{scenario.name} delta/Stage 0 = 2*pi (\u00a79.1, open)",
            passed=abs(normalization.deviation) <= 2e-2,
            detail=normalization.summary(),
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
        report.note("no engines registered yet (Phase 3a onwards) — nothing to run")
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

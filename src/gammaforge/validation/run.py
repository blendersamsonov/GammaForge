"""The validation suite's entry point: run everything runnable, report pass/fail (§7).

::

    python -m gammaforge.validation.run

Two sections today, because engines arrive in later phases and the suite should be honest
about what it did rather than silently doing less:

* **core invariants** — properties of `gammaforge.io` alone (bunch seed determinism, the
  prefilter discarding only particles the pulse never reaches). These run now.
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
from typing import Iterable, Sequence

from ..engines.base import Engine
from .golden import available_goldens, compare_to_golden, load_golden
from .invariance import Check, core_checks, engine_checks
from .runners import derived_scalars, run_engine
from .scenarios import SCENARIOS, Scenario, by_name

__all__ = ["Report", "run_suite", "main"]


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


def _golden_scalar_checks(scenarios: Sequence[Scenario]) -> list[Check]:
    """Compare each golden's stored closed-form scalars against this repo's own.

    Tight by design (1e-9): these are not Monte-Carlo numbers. Both repos compute them
    analytically from the same physical inputs, so a disagreement means a constant, a unit
    or a convention differs — and that is worth failing over, unlike a spectrum shape.
    """
    names = {scenario.name for scenario in scenarios}
    checks = []
    for scenario_name, model in available_goldens():
        if scenario_name not in names:
            continue
        golden = load_golden(scenario_name, model)
        fresh = derived_scalars(by_name(scenario_name))
        for name, reference in sorted(golden.scalars.items()):
            if name not in fresh:
                continue
            error = abs(fresh[name] - reference) / abs(reference) if reference else abs(fresh[name])
            checks.append(Check(
                name=f"{scenario_name}/{model} {name}",
                passed=error <= 1e-9,
                detail=f"{fresh[name]:.12g} vs golden {reference:.12g} ({error:.2e})",
            ))
    return checks


def run_suite(engines: Iterable[Engine] = (), scenarios: Sequence[Scenario] = SCENARIOS) -> Report:
    report = Report()
    engines = list(engines)

    report.section("core invariants (gammaforge.io)")
    for check in core_checks(scenarios):
        report.check(check)

    report.section("golden scalars (predecessor vs this repo)")
    scalar_checks = _golden_scalar_checks(scenarios)
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

"""The validation suite's entry point: run everything runnable, report pass/fail (§7).

::

    python -m gammaforge.validation.run

Three sections today, because engines arrive in later phases and the suite should be
honest about what it did rather than silently doing less:

* **core invariants** — properties of `gammaforge.io` alone (bunch seed determinism, the
  prefilter discarding only particles the pulse never reaches). These run now.
* **identities** — methods that must agree on the same number: Stage 0's total yield, the
  closed-form single-electron spectrum, and delta's brute-force angular integral. All
  three agree on 1 (§9.1, RES033); this leg is what would catch that factor coming back. A
  fourth leg compares Stage 2's own table kernel against delta at one point; both carry
  the same §9.1 factor, so it is deliberately blind to the normalization — it checks
  deposition and interpolation instead.
* **goldens** — every committed snapshot is loaded and its stored closed-form scalars are
  compared against what this repo computes for the same scenario. This is a real
  cross-implementation check with no engine in it: two independent codebases, the same
  physical input, the same number expected out.

The engine sections (cross-engine consistency, chunk/backend/prefilter invariance, the
closed-form identity tests) are `invariance.engine_checks` and `golden.compare_to_golden`
applied to a registered engine. They are not stubbed here: an empty engine list prints as
an empty engine list, and the section appears when an engine does. **`XigmaEngine`
exists** (`engines/xigma/engine.py`) but is not passed to `run_suite` by `main()` below —
the scenario bank's default output resolution is too slow against this phase's numpy
kernel for a suite run meant to be exercised routinely (RES031).
`tests/test_xigma_engine.py` and `tests/test_stage1_stage2.py` exercise it at a
suite-appropriate scale instead.
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

__all__ = [
    "Report", "run_suite", "golden_scalar_checks", "identity_checks", "production_checks", "main",
    "SCALAR_TOLERANCE", "IDENTITY_PARTICLES", "PRODUCTION_PARTICLES",
]


class Report:
    """Accumulates lines and a pass/fail verdict, so a section can report as it goes."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0
        self.blockers = 0

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"--- {title} " + "-" * max(0, 60 - len(title)))

    def check(self, check: Check) -> None:
        self.lines.append(str(check))
        self.failures += not check.passed

    def note(self, text: str) -> None:
        self.lines.append(text)

    def blocked(self, text: str) -> None:
        """Record coverage that prevents this invocation being a full validation pass."""
        self.lines.append(f"[blocked] {text}")
        self.blockers += 1

    def verdict(self) -> str:
        if self.failures:
            return f"{self.failures} CHECK(S) FAILED"
        if self.blockers:
            return f"ALL EXECUTED CHECKS PASS; {self.blockers} COVERAGE BLOCKER(S)"
        return "ALL CHECKS PASS"

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


#: The opt-in production tier uses the scenario bank's physics inputs with a smaller
#: sampling/output request so its Stage-2 checks are runnable on numpy. The ordinary bank
#: remains the production request; this is a named validation measurement, not a default.
PRODUCTION_PARTICLES = 4_000
_PRODUCTION_OUTPUTS = (
    ("total_yield", ()),
    ("spectrum", (96,)),
)
_PRODUCTION_XIGMA_PARAMS = {
    "n_steps": 64,
    "n_bins_gamma": 48,
    "n_bins_theta_x": 32,
    "n_bins_theta_y": 32,
    "n_bins_ahat": 24,
    "ahat_decades": 0.3,
}


def _production_scenario(scenario: Scenario) -> Scenario:
    """A reduced request over one scenario-bank physical configuration."""
    from ..io.target import OutputKind, OutputRequest

    outputs = tuple(
        OutputRequest(getattr(OutputKind, name.upper()), resolution)
        for name, resolution in _PRODUCTION_OUTPUTS
    )
    return replace(scenario, target=replace(scenario.target, outputs=outputs))


def _spectrum_moments(slice_) -> tuple[float, float]:
    """Return the spectral centroid and 99%-contained-energy edge."""
    from ..io.results import Axis

    energy = slice_.axes[Axis.ENERGY]
    density = slice_.distr
    total = float(np.trapezoid(density, energy))
    if not np.isfinite(total) or total <= 0.0:
        return math.nan, math.nan
    centroid = float(np.trapezoid(density * energy, energy) / total)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(energy)))
    )
    return centroid, float(np.interp(0.99 * total, cumulative, energy))


def _relative_deviation(value: float, reference: float) -> float:
    if not (np.isfinite(value) and np.isfinite(reference)) or reference == 0.0:
        return math.inf
    return abs(value / reference - 1.0)


def production_checks(scenarios: Sequence[Scenario]) -> tuple[list[Check], list[str], list[str]]:
    """Opt-in distribution checks over the shared bank's real engine implementations.

    Analytical and xigma agree only in their documented head-on, weakly nonlinear common
    regime. Those yield and spectrum checks are gates. Angular checks are deliberately
    absent: the histogram measure contract exists (RES061), but the independent angular
    comparisons have not been wired. Kascade's Phase-5 four-method comparison is also open.
    """
    from ..engines.analytical.engine import AnalyticalEngine
    from ..engines.xigma.engine import XigmaEngine
    from ..io.target import OutputKind

    checks: list[Check] = []
    notes: list[str] = []
    blockers = [
        "angular xigma/delta validation was not run: the histogram measure contract is "
        "implemented (RES061), but the distribution comparison remains unwired",
        "kascade is not part of this tier: the independent four-method comparison remains unwired",
        "arbitrary-angle emission is not independently validated: the approved per-particle "
        "lab-frame polarization projection has only its direct Eq. udef implementation check",
    ]
    xigma = XigmaEngine()
    analytical = AnalyticalEngine()
    xigma_params = xigma.schema.with_values(**_PRODUCTION_XIGMA_PARAMS)
    for source in scenarios:
        scenario = _production_scenario(source)
        sampling = replace(source.sampling, n_particles=PRODUCTION_PARTICLES)
        xigma_results = run_engine(xigma, scenario, xigma_params, sampling).results
        analytical_results = run_engine(analytical, scenario, sampling=sampling).results

        xigma_yield = float(xigma_results.photon_slices[OutputKind.TOTAL_YIELD].distr)
        analytical_yield = float(analytical_results.photon_slices[OutputKind.TOTAL_YIELD].distr)
        yield_deviation = _relative_deviation(xigma_yield, analytical_yield)
        checks.append(Check(
            f"{source.name} xigma/analytical total yield", yield_deviation <= 0.01,
            f"relative deviation {yield_deviation:.2%} (1% reduced-grid agreement gate)",
        ))

        xigma_centroid, xigma_edge = _spectrum_moments(
            xigma_results.photon_slices[OutputKind.SPECTRUM]
        )
        analytical_centroid, analytical_edge = _spectrum_moments(
            analytical_results.photon_slices[OutputKind.SPECTRUM]
        )
        centroid_deviation = _relative_deviation(xigma_centroid, analytical_centroid)
        edge_deviation = _relative_deviation(xigma_edge, analytical_edge)
        checks.append(Check(
            f"{source.name} xigma/analytical spectrum centroid", centroid_deviation <= 0.035,
            f"relative deviation {centroid_deviation:.2%} (3.5% distribution gate)",
        ))
        checks.append(Check(
            f"{source.name} xigma/analytical 99% spectral edge", edge_deviation <= 0.035,
            f"relative deviation {edge_deviation:.2%} (3.5% distribution gate)",
        ))
    notes.append(
        "The xigma/analytical gates share scenario inputs and the head-on regime, but use "
        "separate overlap/intensity and spectrum constructions. They do not independently "
        "validate the polarization factor or a crossing-angle emission formula. The yield "
        "comparison checks Stage 0 against analytical overlap, while sharing the Gaussian inputs."
    )
    return checks, notes, blockers


def identity_checks(scenarios: Sequence[Scenario]) -> list[Check]:
    """The §7 identity harness: methods that must agree, and by how much they do not.

    Four legs exist today. Stage 0's total yield is an elementary
    ``flux x cross-section x time`` count; the closed-form single-electron spectrum
    integrates to that same number as an identity; delta's angular integral is the
    independent brute-force path. All three agree on one.

    **The delta leg is the §9.1 tripwire.** It watches a ratio that must equal 1; before
    §9.1 closed it read a derived ``2 pi`` instead (RES026, RES033) — this leg is what would
    catch that factor coming back.

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
            name=f"{scenario.name} delta/Stage 0 = 1 (\u00a79.1 closed, RES033)",
            passed=abs(normalization.deviation) <= 2e-2,
            detail=normalization.summary(),
        ))

        # Fourth leg: Stage 2's table kernel against delta's particle-based histogram, at
        # one observation point. Both carry the same normalization constant (RES033), so this
        # ratio is deliberately insensitive to it — it checks deposition and interpolation.
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


def run_suite(
    engines: Iterable[Engine] = (),
    scenarios: Sequence[Scenario] = SCENARIOS,
    *,
    production: bool = False,
    alpha: bool = False,
) -> Report:
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

    report.section("production engine validation (opt-in)")
    if production or alpha:
        if alpha:
            report.note(
                "HEADLESS ALPHA SCOPE: Gaussian analytical/xigma total yields and head-on "
                "weak-field spectrum checks. GUI, kascade and independent arbitrary-angle "
                "emission certification are outside this release gate (RES065)."
            )
        checks, notes, blockers = production_checks(scenarios)
        for check in checks:
            report.check(check)
        for note in notes:
            report.note(note)
        for blocker in blockers:
            if alpha:
                report.note("outside alpha validation scope: " + blocker)
            else:
                report.blocked(blocker)
    else:
        report.note(
            "not run — use `python -m gammaforge.validation.run --production`; this opt-in "
            "tier runs xigma and analytical over every scenario, with delta only as a "
            "shared-input Stage-2 angular reference. Kascade/four-method coverage remains open."
        )

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


def main(argv=None, *, engines: Iterable[Engine] = (), scenarios: Sequence[Scenario] = SCENARIOS) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if set(argv) - {"--production", "--alpha"} or {"--production", "--alpha"} <= set(argv):
        print("usage: python -m gammaforge.validation.run [--production | --alpha]", file=sys.stderr)
        return 2
    report = run_suite(engines, scenarios, production="--production" in argv, alpha="--alpha" in argv)
    print(report)
    return 0 if report.failures == 0 and report.blockers == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

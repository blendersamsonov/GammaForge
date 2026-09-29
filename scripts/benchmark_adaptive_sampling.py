"""Benchmark luminosity-aware adaptive sampling against IID (RES093).

The question this answers is the handoff's, and it is deliberately *not* "is adaptive better
at equal N":

    For a fixed Stage-1/Stage-2 accuracy target, how many expensive Stage-0 particles does
    each strategy need?

Equal-``N`` comparison answers a different and less useful question. What actually costs
anything is Stage 0, so the honest comparison converts an accuracy target into a particle
count, and reports the pilot's own cost separately — the pilot is paid once per plan, and a
plan is reusable for any budget, so it does not scale with ``N`` the way Stage 0 does.

Run from the worktree root (``export PYTHONPATH=$PWD/src`` first -- see NOTES.md).

    python scripts/benchmark_adaptive_sampling.py
    python scripts/benchmark_adaptive_sampling.py --quick     # fewer references, faster
    python scripts/benchmark_adaptive_sampling.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from gammaforge.engines.xigma.stages import (
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
    spectrum_from_table,
)
from gammaforge.io.adaptive_sampling import DEFAULT_PILOT_CONFIG, build_adaptive_plan
from gammaforge.io.interaction import ADAPTIVE, IID, SamplingSpec, build_interaction
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, auto_ranges
from gammaforge.validation import scenarios

#: Bins for the 5D ``ShapeTable``. Deliberately smaller than production (48,48,48,96,8):
#: the question is how quickly the *sampling* converges, and a fine table would make the
#: Stage-1 quadrature dominate the run time being measured.
BINS = (24, 16, 16, 32, 8)

BUDGETS = (2_500, 5_000, 10_000, 20_000, 40_000, 80_000)
STAGE0_STEPS = 200


def spectral_grid(laser) -> np.ndarray:
    """The ``s`` grid Stage 2 is parameterized on, from the target's own energy autorange."""
    ranges = auto_ranges(scenarios.BASELINE.target, scenarios.BASELINE.beam, laser)
    energy_hi = float(ranges[OutputKind.SPECTRUM][Axis.ENERGY][1])
    return np.linspace(0.0, energy_hi / (4.0 * float(laser.photon_energy())), 600)


def run(strategy: str, n_particles: int, seed: int, plan=None, laser=None) -> dict:
    """One full adaptive/IID run through Stage 0, Stage 1, retarget and Stage 2.

    Returns the observables and a phase breakdown. The plan build is timed separately from
    the run itself, because it is paid once and reused across every budget -- folding it into
    the per-run cost would make adaptive look quadratically worse the more budgets a sweep
    uses, which is exactly backwards.
    """
    beam, laser = scenarios.BASELINE.beam, laser or scenarios.BASELINE.laser
    timings: dict[str, float] = {}

    start = time.perf_counter()
    interaction = build_interaction(
        beam, laser, scenarios.BASELINE.target,
        SamplingSpec(n_particles=n_particles, seed=seed, prefilter=1e-3, strategy=strategy),
        plan=plan,
    )
    timings["sampling"] = time.perf_counter() - start

    start = time.perf_counter()
    samples = integrate_trajectories(
        interaction.bunch, laser, interaction.N_e, n_steps=STAGE0_STEPS, threshold=1e-3
    )
    timings["stage0"] = time.perf_counter() - start

    start = time.perf_counter()
    shape_table = deposit_shape_table(samples, n_bins=BINS, scheme="nearest")
    table = retarget_ahat(shape_table, float(samples.intensity_peak))
    grid = spectral_grid(laser)
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    timings["stage1_2"] = time.perf_counter() - start

    total = float(np.sum(samples.luminosity))
    return {
        "yield": total,
        "spectrum": spectrum,
        "centroid": float(np.sum(grid * spectrum) / np.sum(spectrum)),
        "grid": grid,
        "n_particles": interaction.bunch.n_particles,
        "n_eff": float(interaction.bunch.meta.get("n_eff", interaction.bunch.n_particles)),
        "timings": timings,
        "total_seconds": sum(timings.values()),
    }


def errors(result: dict, reference: dict) -> dict:
    """Relative errors against a reference, on grid-independent physical observables."""
    return {
        "yield": abs(result["yield"] - reference["yield"]) / reference["yield"],
        "spectrum": float(
            np.sum(np.abs(result["spectrum"] - reference["spectrum"]))
            / np.sum(np.abs(reference["spectrum"]))
        ),
        "centroid": abs(result["centroid"] - reference["centroid"])
        / abs(reference["centroid"]),
    }


def particles_for(target: float, measured: dict[int, float], budgets=BUDGETS) -> int | None:
    """Smallest measured budget whose error is at or below ``target``.

    Interpolated in log-error between the two bracketing budgets rather than snapped to a
    grid point, so the answer is not an artifact of which budgets happened to be chosen.
    ``None`` means the target was not reached anywhere in the measured range, which is
    itself the answer for a strategy that cannot get there.
    """
    ordered = sorted(measured.items())
    reached = [(n, e) for n, e in ordered if e <= target]
    if not reached:
        return None
    best = reached[0][0]
    previous = [(n, e) for n, e in ordered if n < best]
    if not previous:
        return float(best)
    lower_n, lower_error = previous[-1]
    # Geometric interpolation between the bracketing points.
    fraction = math.log(lower_error / target) / math.log(lower_error / reached[0][1])
    return float(lower_n * (best / lower_n) ** min(max(fraction, 0.0), 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer reference seeds and budgets")
    parser.add_argument("--json", type=str, default=None, help="write results as JSON")
    args = parser.parse_args()

    budgets = BUDGETS[:3] if args.quick else BUDGETS
    reference_seeds = (3, 11) if args.quick else (3, 11, 23)
    sampler_seeds = (3,) if args.quick else (3, 11)
    reference_n = 100_000 if args.quick else 200_000

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser

    print(f"reference: {reference_n} IID particles, seeds {reference_seeds}")
    print(f"budgets:   {list(budgets)}   seeds: {list(sampler_seeds)}\n")

    start = time.perf_counter()
    references = [run(IID, reference_n, seed) for seed in reference_seeds]
    print(f"built {len(references)} references in {time.perf_counter() - start:.1f}s")
    print(
        "reference spread: yield {:.2e}  spectrum {:.2e}".format(
            (max(r["yield"] for r in references) - min(r["yield"] for r in references))
            / np.mean([r["yield"] for r in references]),
            float(
                np.mean(
                    [
                        np.abs(r["spectrum"] - references[0]["spectrum"]).mean()
                        for r in references
                    ]
                )
                / np.abs(references[0]["spectrum"]).mean()
            ),
        )
    )
    print(
        "\nAny absolute tolerance below that spread measures the reference, not the sampler;"
        "\nthe per-target table below is therefore comparative and the target is deliberately"
        "\nloose relative to it.\n"
    )

    # One plan per sampler seed, each reused across every budget. A plan's low-discrepancy
    # shifts are seeded, so a plan is valid for exactly one seed -- reusing one across seeds
    # is refused by `build_adaptive_bunch` rather than silently producing a different
    # estimator. This is also the point of the design: the pilot is paid once per seed, not
    # once per budget.
    plans, plan_seconds = {}, 0.0
    for seed in sampler_seeds:
        start = time.perf_counter()
        plans[seed] = build_adaptive_plan(beam, laser, seed=seed)
        plan_seconds += time.perf_counter() - start
    print(
        f"adaptive plans: {plans[sampler_seeds[0]].n_regions} regions, {len(plans)} built in "
        f"{plan_seconds:.2f}s total (reused across every budget)\n"
    )

    header = f"{'N':>7} {'strategy':>9} {'yield err':>11} {'spectrum err':>13} {'centroid err':>13} {'secs':>7} {'N_eff':>8}"
    print(header)
    print("-" * len(header))

    measured = {IID: {}, ADAPTIVE: {}}
    for n_particles in budgets:
        for strategy in (IID, ADAPTIVE):
            runs = [
                run(
                    strategy, n_particles, seed,
                    plan=plans[seed] if strategy == ADAPTIVE else None,
                )
                for seed in sampler_seeds
            ]
            mean_spectrum = np.mean([r["spectrum"] for r in runs], axis=0)
            mean_result = {
                "yield": float(np.mean([r["yield"] for r in runs])),
                "spectrum": mean_spectrum,
                "centroid": float(np.mean([r["centroid"] for r in runs])),
            }
            error = errors(mean_result, references[0])
            measured[strategy].setdefault("yield", {})[n_particles] = error["yield"]
            measured[strategy].setdefault("spectrum", {})[n_particles] = error["spectrum"]
            print(
                f"{n_particles:>7} {strategy:>9} {error['yield']:>11.2e} "
                f"{error['spectrum']:>13.2e} {error['centroid']:>13.2e} "
                f"{np.mean([r['total_seconds'] for r in runs]):>7.1f} "
                f"{np.mean([r['n_eff'] for r in runs]):>8.0f}"
            )
        print()

    print("Particles needed to reach a target error (geometric interpolation):\n")
    print(f"{'target':>10} {'quantity':>10} {'IID':>10} {'adaptive':>10} {'reduction':>10}")
    print("-" * 54)
    summary = {"plan_seconds": plan_seconds, "reference_n": reference_n, "targets": {}}
    for quantity in ("yield", "spectrum"):
        for target in (1e-2, 3e-3, 1e-3):
            iid_n = particles_for(target, measured[IID][quantity], budgets)
            adaptive_n = particles_for(target, measured[ADAPTIVE][quantity], budgets)
            if iid_n is None or adaptive_n is None:
                reduction = "n/a" if iid_n is None and adaptive_n is None else (
                    "IID only" if adaptive_n is None else "adaptive only"
                )
                iid_s = f"{iid_n:.0f}" if iid_n else "not reached"
                adaptive_s = f"{adaptive_n:.0f}" if adaptive_n else "not reached"
            else:
                iid_s, adaptive_s = f"{iid_n:.0f}", f"{adaptive_n:.0f}"
                reduction = f"{iid_n / adaptive_n:.1f}x"
            print(f"{target:>10.0e} {quantity:>10} {iid_s:>10} {adaptive_s:>10} {reduction:>10}")
            summary["targets"][f"{quantity}@{target:.0e}"] = {
                "iid": iid_n, "adaptive": adaptive_n,
            }
    print(
        "\n'spectrum' and 'yield' answer different questions. The yield is the "
        "luminosity-weighted\nintegral, which is exactly what the allocation optimizes, so it "
        "is where the gain\nis largest. The spectrum's residual error is dominated by how "
        "many particles land\nin each Stage-1 cell, which reallocating particles does not "
        "change -- so parity there\nis the expected result, not a shortfall."
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    **summary,
                    "measured": {
                        strategy: {q: {str(k): v for k, v in values.items()}
                                   for q, values in quantities.items()}
                        for strategy, quantities in measured.items()
                    },
                },
                handle, indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

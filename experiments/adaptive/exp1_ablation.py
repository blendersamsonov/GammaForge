"""Experiment 1+2: ablation, lambda sweep, proposal-scale sweep (baseline scenario).

All arms share one strong reference built on fixed edges, so every number is comparable and
the Stage-1 table metric is well posed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

import harness as H
from gammaforge.io.interaction import IID
from gammaforge.validation import scenarios

BUDGETS = (10_000, 20_000, 40_000)
SEEDS = (3, 11)
REF_N = 800_000
REF_SEEDS = (3, 11, 23)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def main():
    global BUDGETS, SEEDS, REF_N, REF_SEEDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ref-n", type=int, default=REF_N)
    parser.add_argument("--ref-seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    args = parser.parse_args()
    REF_N = args.ref_n
    REF_SEEDS = tuple(3 + 8 * i for i in range(args.ref_seeds))
    SEEDS = REF_SEEDS[:2] if len(REF_SEEDS) >= 2 else REF_SEEDS
    if args.budgets:
        BUDGETS = tuple(args.budgets)
    if args.quick:
        BUDGETS = (5_000, 10_000)
        SEEDS = (3,)
    return _run()


def _run():
    beam, laser, target = scenarios.BASELINE.beam, scenarios.BASELINE.laser, scenarios.BASELINE.target
    grid = H.spectral_grid(beam, laser, target)

    # Reference edges from one large IID run, then the reference itself as a mean over seeds.
    from gammaforge.engines.xigma.stages import integrate_trajectories
    from gammaforge.io.interaction import SamplingSpec, build_interaction

    start = time.perf_counter()
    probe = build_interaction(
        beam, laser, target, SamplingSpec(n_particles=REF_N, seed=REF_SEEDS[0])
    )
    probe_samples = integrate_trajectories(
        probe.bunch, laser, probe.N_e, n_steps=H.STAGE0_STEPS, threshold=1e-3
    )
    edges = H.reference_edges(probe_samples, H.REDUCED_BINS)
    print(f"reference edges from one {REF_N}-particle run ({time.perf_counter()-start:.0f}s)",
          flush=True)

    start = time.perf_counter()
    ref = H.make_reference(None, beam, laser, target, REF_N, REF_SEEDS, edges, grid=grid)
    print(f"reference: {len(REF_SEEDS)} x {REF_N} IID particles, {time.perf_counter()-start:.0f}s")
    print("  noise floor: yield {:.2e} | marginal L1 worst {:.2e} mean {:.2e} | "
          "spectrum L1 {:.2e} | raw 5D table L1 {:.2e}".format(
        ref["yield_spread"], ref["marg_worst_spread"], ref["marg_mean_spread"],
        ref["spectrum_spread"], ref["table_l1_spread"]), flush=True)
    print("  (any error below these is measuring the reference, not the sampler)\n", flush=True)

    arms = H.ablation_variants() + H.lambda_variants() + H.scale_variants()
    # Drop duplicates by name, keeping the first.
    seen, unique = set(), []
    for arm in arms:
        if arm.name not in seen:
            seen.add(arm.name)
            unique.append(arm)

    columns = ("yield", "marg_worst", "marg_mean", "spectrum", "centroid", "table_l1")
    header = f"{'variant':>17} {'N':>7} " + " ".join(f"{c:>11}" for c in columns) + \
             f" {'w_spread':>9} {'N_eff':>8} {'s':>6}"
    print(header)
    print("-" * len(header))
    results = []
    for arm in unique:
        for n in BUDGETS:
            runs = [
                H.run(arm, beam, laser, target, n, seed, edges, grid=grid)
                for seed in SEEDS
            ]
            merged = {
                "yield": float(np.mean([r["yield"] for r in runs])),
                "H": np.mean([r["H"] for r in runs], axis=0),
                "spectrum": np.mean([r["spectrum"] for r in runs], axis=0),
                "centroid": float(np.mean([r["centroid"] for r in runs])),
            }
            err = H.errors(merged, ref)
            row = {
                "variant": arm.name, "n": n, "note": arm.note,
                "weight_spread": float(np.mean([r["weight_spread"] for r in runs])),
                "n_eff": float(np.mean([r["n_eff"] for r in runs])),
                "seconds": float(np.mean([r["seconds"] for r in runs])),
                **err,
            }
            results.append(row)
            print(
                f"{arm.name:>17} {n:>7} " + " ".join(f"{err[c]:>11.2e}" for c in columns)
                + f" {row['weight_spread']:>9.2f} {row['n_eff']:>8.0f} {row['seconds']:>6.1f}",
                flush=True,
            )
        print(flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "exp1_ablation.json", "w") as handle:
        json.dump(
            {"reference": {k: v for k, v in ref.items() if k not in ("H", "spectrum")},
             "bins": H.REDUCED_BINS, "budgets": BUDGETS, "seeds": SEEDS,
             "results": results},
            handle, indent=2,
        )
    print("wrote results/exp1_ablation.json")


if __name__ == "__main__":
    main()

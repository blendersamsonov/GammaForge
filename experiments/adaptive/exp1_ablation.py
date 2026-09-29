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
import stats as S
from gammaforge.io.interaction import IID
from gammaforge.validation import scenarios

CONTROL = "strat-s1-lam0"
BUDGETS = (10_000, 20_000, 40_000)
SEEDS = (3, 11)
REF_N = 800_000
REF_SEEDS = (3, 11, 23)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def main():
    global BUDGETS, SEEDS, REF_N, REF_SEEDS, CONTROL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ref-n", type=int, default=REF_N)
    parser.add_argument("--ref-seeds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=8,
                        help="independent seeds per arm; the paired spread against the control "
                             "is what decides whether an arm differs")
    parser.add_argument("--control", default=CONTROL,
                        help="arm every other arm is compared against, seed by seed. Default is "
                             "plain QMC s=1 (no allocation), which isolates the allocation")
    parser.add_argument("--chunk-mb", type=float, default=2000.0,
                        help="peak MiB for Stage 0's (particle x step) temporaries; see "
                             "harness.chunk_for_mb")
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    args = parser.parse_args()

    import preflight
    preflight.check()
    H.CHUNK = H.chunk_for_mb(args.chunk_mb)
    REF_N = args.ref_n
    REF_SEEDS = tuple(3 + 8 * i for i in range(args.ref_seeds))
    SEEDS = H.arm_seeds(args.replicates)
    CONTROL = CONTROL
    if args.budgets:
        BUDGETS = tuple(args.budgets)
    if args.quick:
        BUDGETS = (5_000, 10_000)
        # --quick shrinks the budgets and the reference, but keeps the requested
        # replicate count: a smoke test that skips the pairing cannot catch a broken one.
        SEEDS = H.arm_seeds(min(args.replicates, 2))
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
        probe.bunch, laser, probe.N_e, n_steps=H.STAGE0_STEPS, threshold=1e-3, chunk=H.CHUNK
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
    by_name = {arm.name: arm for arm in unique}
    if CONTROL not in by_name:
        raise SystemExit(f"--control {CONTROL!r} is not an arm; choose from {sorted(by_name)}")
    results = []
    for n in BUDGETS:
        # All arms at this budget first, so each can be paired seed-by-seed against the control.
        per_arm = {
            arm.name: [H.run(arm, beam, laser, target, n, seed, edges, grid=grid)
                       for seed in SEEDS]
            for arm in unique
        }
        control_runs = per_arm[CONTROL]
        for arm in unique:
            runs = per_arm[arm.name]
            is_control = arm.name == CONTROL
            err = H.errors(H.merge(runs), ref)
            block = S.replicate_block(
                runs, ref, None if is_control else control_runs,
                control_name=CONTROL, label=arm.name,
            )
            results.append({
                "variant": arm.name, "n": n, "note": arm.note,
                "is_control": is_control,
                "weight_spread": float(np.mean([r["weight_spread"] for r in runs])),
                "n_eff": float(np.mean([r["n_eff"] for r in runs])),
                "seconds": float(np.mean([r["seconds"] for r in runs])),
                "replicates": block,
                **err,
            })
            print(
                f"{arm.name:>17} {n:>7} " + " ".join(f"{err[c]:>11.2e}" for c in columns)
                + f" {np.mean([r['weight_spread'] for r in runs]):>9.2f}"
                + f" {np.mean([r['n_eff'] for r in runs]):>8.0f}"
                + f" {np.mean([r['seconds'] for r in runs]):>6.1f}"
                + f"  {S.fmt(block, 'spectrum', 17):>17}  {_delta(block)}",
                flush=True,
            )
        print(flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "exp1_ablation.json", "w") as handle:
        json.dump(
            {"reference": {k: v for k, v in ref.items() if k not in ("H", "spectrum")},
             "bins": H.REDUCED_BINS, "budgets": list(BUDGETS), "seeds": list(SEEDS),
             "control": CONTROL, "reference_n": ref["n_particles"],
             "results": results},
            handle, indent=2,
        )
    print("wrote results/exp1_ablation.json")


def _delta(block) -> str:
    """The paired spectrum difference vs the control, formatted, or a marker for the control."""
    entry = block.get("delta", {}).get("spectrum")
    if not entry or not np.isfinite(entry["sem"]):
        return "(control)"
    return f"{entry['mean']:+.2e}±{entry['sem']:.0e}  {block['verdict']}"


if __name__ == "__main__":
    main()

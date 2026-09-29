"""Experiment 3+4: wider scenario bank, production table bins, nearest vs CIC.

The baseline told us which arms are worth carrying, so this runs the promising ones across
the bank rather than all six. Reference is two independent 400k IID runs per scenario (the
noise floor is reported per scenario, since it varies a lot with how localized the
interaction is).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import harness as H
import stats as S
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.adaptive_sampling import PilotConfig
from gammaforge.io.interaction import IID, SamplingSpec, build_interaction
from gammaforge.validation import scenarios as scen

BUDGETS = (20_000, 40_000)
SEEDS = (3, 11)
REF_N = 400_000
REF_SEEDS = (3, 11)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def arms():
    base = dict(
        initial_regions=256, max_regions=256, pilot_points_per_region=8,
        pilot_quad_nodes=128, pilot_quad_panels=2,
    )
    return [
        H.Variant("iid", IID),
        H.Variant("strat-s1", ADAPTIVE := "adaptive",
                  PilotConfig(proposal_scale=1.0, luminosity_fraction=0.0, **base),
                  "plain QMC, s=1, volume allocation"),
        H.Variant("strat-s1.41", ADAPTIVE,
                  PilotConfig(proposal_scale=np.sqrt(2.0), luminosity_fraction=0.0, **base),
                  "broad reference, volume allocation (the inert default)"),
        H.Variant("s1.41-lam0.75", ADAPTIVE,
                  PilotConfig(proposal_scale=np.sqrt(2.0), luminosity_fraction=0.75, **base),
                  "broad reference + luminosity allocation"),
        H.Variant("s1-lam0.75", ADAPTIVE,
                  PilotConfig(proposal_scale=1.0, luminosity_fraction=0.75, **base),
                  "s=1 + luminosity allocation"),
        H.Variant("s1-lam1", ADAPTIVE,
                  PilotConfig(proposal_scale=1.0, luminosity_fraction=1.0, **base),
                  "s=1, pure luminosity allocation"),
        H.Variant("full", ADAPTIVE,
                  PilotConfig(proposal_scale=np.sqrt(2.0), luminosity_fraction=0.75,
                              initial_regions=64, max_regions=256, pilot_points_per_region=8,
                              pilot_quad_nodes=128, pilot_quad_panels=2),
                  "shipped scheme: broad ref + splitting + lam 0.75"),
    ]


def run_scenario(scenario, bins, scheme="nearest"):
    beam0, laser0, target = scen.BASELINE.beam, scen.BASELINE.laser, scen.BASELINE.target
    beam, laser = H.build(scenario, laser0, beam0)
    grid = H.spectral_grid(beam, laser, target)

    probe = build_interaction(
        beam, laser, target, SamplingSpec(n_particles=REF_N, seed=REF_SEEDS[0])
    )
    probe_samples = integrate_trajectories(
        probe.bunch, laser, probe.N_e, n_steps=H.STAGE0_STEPS, threshold=1e-3
    )
    edges = H.reference_edges(probe_samples, bins)
    ref = H.make_reference(None, beam, laser, target, REF_N, REF_SEEDS, edges,
                           scheme=scheme, grid=grid)
    return beam, laser, target, grid, edges, ref


def main():
    global BUDGETS, SEEDS, REF_N, REF_SEEDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny budgets, plumbing check")
    parser.add_argument("--ref-n", type=int, default=REF_N)
    parser.add_argument("--ref-seeds", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=8,
                        help="independent seeds per arm (default 8). The paired spread across "
                             "these, not the reference floor, is what decides whether an arm "
                             "differs from the control")
    parser.add_argument("--control", default="strat-s1",
                        help="arm every other arm is compared against, seed by seed. Default is "
                             "plain QMC s=1, which isolates the *allocation* from the "
                             "stratification; use --control iid to ask the different question of "
                             "whether the whole scheme beats plain sampling")
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--bins", default="reduced", choices=["reduced", "production", "both"])
    parser.add_argument("--scheme", default="nearest", choices=["nearest", "cic", "both"])
    args = parser.parse_args()

    import preflight
    preflight.check()

    REF_N = args.ref_n
    REF_SEEDS = tuple(3 + 8 * i for i in range(args.ref_seeds))
    SEEDS = H.arm_seeds(args.replicates)
    BUDGETS = tuple(args.budgets) if args.budgets else BUDGETS
    if args.quick:
        BUDGETS = (5_000, 10_000)
        # --quick shrinks the budgets and the reference, but keeps the requested
        # replicate count: a smoke test that skips the pairing cannot catch a broken one.
        SEEDS = H.arm_seeds(min(args.replicates, 2))

    if args.only:
        bank = [s for s in H.scenarios() if s.name in set(args.only)]
    else:
        bank = list(H.scenarios())
    bins_choices = (H.REDUCED_BINS, H.PRODUCTION_BINS) if args.bins == "both" else \
        (H.PRODUCTION_BINS,) if args.bins == "production" else (H.REDUCED_BINS,)
    scheme_choices = ("nearest", "cic") if args.scheme == "both" else (args.scheme,)
    arm_list = arms()
    names = {arm.name for arm in arm_list}
    if args.control not in names:
        parser.error(f"--control {args.control!r} is not an arm; choose from {sorted(names)}")

    out = []
    for scenario in bank:
        for bins, label in zip(bins_choices, ["reduced", "production"] * 2):
            for scheme in scheme_choices:
                start = time.perf_counter()
                beam, laser, target, grid, edges, ref = run_scenario(scenario, bins, scheme)
                floor = (ref["yield_spread"], ref["marg_worst_spread"],
                         ref["marg_mean_spread"], ref["spectrum_spread"])
                print(f"\n=== {scenario.name} / {label} bins / {scheme} "
                      f"({time.perf_counter()-start:.0f}s to reference) ===")
                print(f"  reference: {ref['n_particles']} particles x {len(ref['seeds'])} seeds")
                print("  floor: yield {:.1e} | marg worst {:.1e} mean {:.1e} | spectrum {:.1e}"
                      .format(*floor), flush=True)
                print(f"  {len(SEEDS)} replicates/arm, paired against control "
                      f"{args.control!r}", flush=True)
                header = (f"{'variant':>15} {'N':>7} {'yield':>10} {'marg_worst':>11} "
                          f"{'marg_mean':>10} {'spectrum':>17} {'N_eff':>7} "
                          f"{'Δ spectrum':>18}  verdict")
                print(header)
                print("-" * len(header))
                for n in BUDGETS:
                    # Every arm at this budget first, so each can be paired seed-by-seed
                    # against the control. Running them in lockstep is what makes the pairing
                    # possible; the alternative is a second pass over the same arms.
                    per_arm = {
                        arm.name: [H.run(arm, beam, laser, target, n, seed, edges,
                                         scheme=scheme, grid=grid) for seed in SEEDS]
                        for arm in arm_list
                    }
                    control_runs = per_arm[args.control]
                    for arm in arm_list:
                        runs = per_arm[arm.name]
                        is_control = arm.name == args.control
                        merged = H.merge(runs)
                        err = H.errors(merged, ref)
                        block = S.replicate_block(
                            runs, ref, None if is_control else control_runs,
                            control_name=args.control, label=arm.name,
                        )
                        row = {
                            "scenario": scenario.name, "bins": label, "scheme": scheme,
                            "variant": arm.name, "n": n, "is_control": is_control,
                            "floor_yield": floor[0], "floor_marg_worst": floor[1],
                            "floor_marg_mean": floor[2], "floor_spectrum": floor[3],
                            "reference_n": ref["n_particles"],
                            "reference_seeds": ref["seeds"],
                            "replicate_seeds": list(SEEDS),
                            "yield": err["yield"], "marg_worst": err["marg_worst"],
                            "marg_mean": err["marg_mean"], "spectrum": err["spectrum"],
                            "centroid": err["centroid"],
                            "replicates": block,
                        }
                        out.append(row)
                        print(f"{arm.name:>15} {n:>7} {err['yield']:>10.2e} "
                              f"{err['marg_worst']:>11.2e} {err['marg_mean']:>10.2e} "
                              f"{S.fmt(block, 'spectrum', 17):>17} "
                              f"{err['n_eff'] if 'n_eff' in err else block['n_eff']['mean']:>7.0f} "
                              f"{_delta(block):>18}  {block['verdict']}",
                              flush=True)
    RESULTS.mkdir(exist_ok=True)
    names = sorted({r["scenario"] for r in out})
    suffix = "_".join(names) if len(names) <= 2 else "bank"
    target = RESULTS / f"scenarios_{suffix}.json"
    with open(target, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {target} ({len(out)} rows)")


def _delta(block) -> str:
    """The paired spectrum difference vs the control, formatted, or a marker for the control."""
    entry = block.get("delta", {}).get("spectrum")
    if not entry or not np.isfinite(entry["sem"]):
        return "(control)"
    return f"{entry['mean']:+.2e}±{entry['sem']:.0e}"
    RESULTS.mkdir(exist_ok=True)
    names = sorted({r["scenario"] for r in out})
    suffix = "_".join(names) if len(names) <= 2 else "bank"
    target = RESULTS / f"scenarios_{suffix}.json"
    with open(target, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {target} ({len(out)} rows)")


if __name__ == "__main__":
    main()

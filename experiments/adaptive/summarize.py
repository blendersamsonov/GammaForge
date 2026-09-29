"""Print the experiment results as readable tables.

    python summarize.py                       # everything found
    python summarize.py --ablation            # one section
    python summarize.py --scenario tight_focus

Reads whatever JSON exists in ``results/`` and skips what is missing, so it is useful both
mid-run and after a completed pass.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

ARM_ORDER = [
    "iid", "strat-s1-lam0", "strat-s1", "strat-s1.41", "s1.41-lam0.75", "s1-lam0.75",
    "s1-lam1", "lam0", "lam0.25", "lam0.5", "lam0.75", "lam1", "s1-lam0.75", "full",
    "focus_3um",
]
SHORT = {
    "iid": "IID",
    "strat-s1-lam0": "QMC s=1 lam0",
    "strat-s1": "QMC s=1",
    "strat-s1.41": "QMC s=1.41 lam0",
    "s1.41-lam0.75": "s=1.41 lam.75",
    "s1-lam0.75": "s=1 lam.75",
    "s1-lam1": "s=1 lam1",
    "s1-lam0.75": "s=1 lam.75",
    "lam0": "lam0", "lam0.25": "lam.25", "lam0.5": "lam.5",
    "lam0.75": "lam.75", "lam1": "lam1", "full": "FULL (shipped)",
}


def repl(row, key):
    """``mean±sem`` from a row's replicate block, falling back to the bare point estimate.

    Older result files predate the replicate block, so the fallback keeps them readable --
    with no error bar, which the header says out loud.
    """
    block = row.get("replicates") or {}
    entry = block.get(key)
    if not entry or not math.isfinite(entry.get("sem", float("nan"))):
        value = row.get(key, float("nan"))
        return f"{value:>10.2e}", True
    return f"{entry['mean']:>7.2e}±{entry['sem']:.0e}", False


def delta(row):
    """The paired difference vs the control, and its verdict."""
    block = row.get("replicates") or {}
    if row.get("is_control"):
        return f"{'(control)':>18}", "control"
    entry = (block.get("delta") or {}).get("spectrum")
    if not entry or not math.isfinite(entry.get("sem", float("nan"))):
        return f"{'(no replicates)':>18}", "not measured"
    return f"{entry['mean']:>+9.2e}±{entry['sem']:.0e}", block.get("verdict", "?")


def load(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def show_ablation():
    data = load("exp1_ablation.json")
    if not data:
        print("exp1: no results yet")
        return
    ref = data["reference"]
    ref_n = data.get("reference_n") or ref.get("n_particles")
    ref_seeds = len(ref.get("seeds") or []) or data.get("ref_seeds")
    reps = (data["results"][0].get("replicates") or {}).get("replicates") if data["results"] else 0
    print("== EXP1: ablation / lambda / proposal-scale (baseline, reduced bins, nearest) ==")
    print(f"   reference: {ref_n or 'unknown'} particles x {ref_seeds or 'unknown'} seeds"
          f"   floor yield {ref['yield_spread']:.1e} | marg_worst "
          f"{ref['marg_worst_spread']:.1e} | spectrum {ref['spectrum_spread']:.1e}")
    print(f"   control: {data.get('control', '?')!r}   replicates: {reps or 'none (old file)'}")
    if not reps:
        print("   NOTE: this file has no replicates, so the errors are bare point estimates")
    print("   the floor bounds the REFERENCE, not an arm; arm-vs-arm differences are decided by")
    print("   the paired Δ column, not by whether an error exceeds the floor\n")
    header = (f"{'arm':>16} {'N':>7} {'yield':>10} {'marg_worst':>11} {'marg_mean':>10} "
              f"{'spectrum':>14} {'w_spread':>9} {'N_eff':>7} {'Δ spectrum':>18}  verdict")
    print(header)
    print("-" * len(header))
    for row in data["results"]:
        spectrum, _ = repl(row, "spectrum")
        d, verdict = delta(row)
        print(f"{SHORT.get(row['variant'], row['variant']):>16} {row['n']:>7} "
              f"{row['yield']:>10.2e} {row['marg_worst']:>11.2e} {row['marg_mean']:>10.2e} "
              f"{spectrum:>14} "
              f"{row.get('weight_spread', float('nan')):>9.2f} {row.get('n_eff', 0):>7.0f} "
              f"{d:>18}  {verdict}")
    print()


def show_scenarios(only=None):
    rows = []
    for path in sorted(RESULTS.glob("scenarios_*.json")):
        rows.extend(json.load(open(path)))
    if not rows:
        rows = load("exp3_scenarios.json") or []
    if not rows:
        print("exp3: no results yet")
        return
    if only:
        rows = [r for r in rows if r["scenario"] in set(only)]
    print("== EXP3: scenario bank ==")
    scenarios = sorted({r["scenario"] for r in rows})
    for scenario in scenarios:
        subset = [r for r in rows if r["scenario"] == scenario]
        floor = subset[0]
        budgets = sorted({r["n"] for r in subset})
        reps = (subset[0].get("replicates") or {}).get("replicates")
        print(f"\n-- {scenario}  (reference {floor.get('reference_n', '?')} particles; "
              f"floor marg_worst {floor.get('floor_marg_worst', float('nan')):.1e}"
              f" | spectrum {floor.get('floor_spectrum', float('nan')):.1e}"
              f"{f'; {reps} replicates' if reps else '; no replicates'})")
        header = (f"{'arm':>16} {'N':>7} {'yield':>10} {'marg_worst':>11} "
                  f"{'marg_mean':>10} {'spectrum':>14} {'N_eff':>7} {'Δ spectrum':>18}  verdict")
        print(header)
        print("-" * len(header))
        for n in budgets:
            for row in [r for r in subset if r["n"] == n]:
                spectrum, _ = repl(row, "spectrum")
                d, verdict = delta(row)
                n_eff = (row.get("replicates") or {}).get("n_eff", {}).get(
                    "mean", row.get("n_eff", 0.0))
                print(f"{SHORT.get(row['variant'], row['variant']):>16} {row['n']:>7} "
                      f"{row['yield']:>10.2e} {row['marg_worst']:>11.2e} "
                      f"{row['marg_mean']:>10.2e} {spectrum:>14} {n_eff:>7.0f} "
                      f"{d:>18}  {verdict}")
    print()


def show_cell_aware():
    rows = load("exp5_cell_aware.json")
    if not rows:
        print("exp5: no results yet")
        return
    print("== EXP5: cell-aware Neyman allocation ==")
    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        floor = subset[0]
        reps = (subset[0].get("replicates") or {}).get("replicates")
        print(f"\n-- {scenario}  (reference {floor.get('reference_n', '?')} particles; "
              f"floor marg_worst {floor.get('floor_marg_worst', float('nan')):.1e}"
              f" | spectrum {floor.get('floor_spectrum', float('nan')):.1e}"
              f"{f'; {reps} replicates' if reps else '; no replicates'})")
        header = (f"{'plan':>16} {'allocation':>20} {'N':>7} {'yield':>10} "
                  f"{'marg_worst':>11} {'marg_mean':>10} {'spectrum':>14} {'N_eff':>7} "
                  f"{'Δ spectrum':>18}  verdict")
        print(header)
        print("-" * len(header))
        for row in subset:
            spectrum, _ = repl(row, "spectrum")
            d, verdict = delta(row)
            print(f"{row['plan']:>16} {row['variant']:>20} {row['n']:>7} {row['yield']:>10.2e} "
                  f"{row['marg_worst']:>11.2e} {row['marg_mean']:>10.2e} "
                  f"{spectrum:>14} {row.get('n_eff', 0):>7.0f} {d:>18}  {verdict}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--cell-aware", action="store_true")
    parser.add_argument("--scenario", default=None, help="restrict to one scenario name")
    args = parser.parse_args()
    show_all = not (args.ablation or args.scenarios or args.cell_aware)
    if args.ablation or show_all:
        show_ablation()
    if args.scenarios or show_all:
        show_scenarios(args.scenario)
    if args.cell_aware or show_all:
        show_cell_aware()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

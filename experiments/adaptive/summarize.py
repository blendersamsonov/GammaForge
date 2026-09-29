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
import pathlib
from collections import defaultdict

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
    print("== EXP1: ablation / lambda / proposal-scale (baseline, reduced bins, nearest) ==")
    print(f"   reference: {data['reference_n'] if 'reference_n' in data else '?'} particles, "
          f"floor yield {ref['yield_spread']:.1e} | marg_worst "
          f"{ref['marg_worst_spread']:.1e} | spectrum {ref['spectrum_spread']:.1e}")
    print("   (errors at or below the floor are measuring the reference)\n")
    header = (f"{'arm':>16} {'N':>7} {'yield':>10} {'marg_worst':>11} {'marg_mean':>10} "
              f"{'spectrum':>10} {'centroid':>10} {'w_spread':>9} {'N_eff':>7}")
    print(header)
    print("-" * len(header))
    for row in data["results"]:
        print(f"{SHORT.get(row['variant'], row['variant']):>16} {row['n']:>7} "
              f"{row['yield']:>10.2e} {row['marg_worst']:>11.2e} {row['marg_mean']:>10.2e} "
              f"{row['spectrum']:>10.2e} {row['centroid']:>10.2e} "
              f"{row.get('weight_spread', float('nan')):>9.2f} {row.get('n_eff', 0):>7.0f}")
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
        print(f"\n-- {scenario}  (floor: marg_worst {floor.get('floor_marg_worst', float('nan')):.1e}"
              f" | spectrum {floor.get('floor_spectrum', float('nan')):.1e})")
        header = (f"{'arm':>16} {'N':>7} {'yield':>10} {'marg_worst':>11} "
                  f"{'marg_mean':>10} {'spectrum':>10} {'N_eff':>7}")
        print(header)
        print("-" * len(header))
        for n in budgets:
            for row in [r for r in subset if r["n"] == n]:
                print(f"{SHORT.get(row['variant'], row['variant']):>16} {row['n']:>7} "
                      f"{row['yield']:>10.2e} {row['marg_worst']:>11.2e} "
                      f"{row['marg_mean']:>10.2e} {row['spectrum']:>10.2e} "
                      f"{row.get('n_eff', 0):>7.0f}")
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
        print(f"\n-- {scenario}  (floor: marg_worst {floor.get('floor_marg_worst', float('nan')):.1e}"
              f" | spectrum {floor.get('floor_spectrum', float('nan')):.1e})")
        header = (f"{'plan':>16} {'allocation':>20} {'N':>7} {'yield':>10} "
                  f"{'marg_worst':>11} {'marg_mean':>10} {'spectrum':>10} {'N_eff':>7}")
        print(header)
        print("-" * len(header))
        for row in subset:
            print(f"{row['plan']:>16} {row['variant']:>20} {row['n']:>7} {row['yield']:>10.2e} "
                  f"{row['marg_worst']:>11.2e} {row['marg_mean']:>10.2e} "
                  f"{row['spectrum']:>10.2e} {row.get('n_eff', 0):>7.0f}")
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

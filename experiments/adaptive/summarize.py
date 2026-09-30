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


#: The arms worth reading side by side. `strat-s1` is the control (plain stratified QMC, no
#: allocation), so the question each other arm answers is explicit: iid = is the whole scheme
#: worth anything, strat-s1.41 = does broad stratification alone help or hurt, the lam arms =
#: what does the allocation add, full = what actually ships.
DECISION_ARMS = ("iid", "strat-s1", "strat-s1.41", "s1-lam0.75", "s1.41-lam0.75", "full")


def _pm(block, key, width=10):
    """``mean±sem`` from a replicate block; the bare mean when there is no usable sem."""
    entry = block.get(key, {})
    mean, sem = entry.get("mean", float("nan")), entry.get("sem", float("nan"))
    if not math.isfinite(sem):
        return f"{mean:>{width}.2e}"
    return f"{mean:>{width - 5}.2e}±{sem:.0e}"


def show_decision():
    """One line per (scenario, budget, arm): the paired difference and its verdict.

    The per-scenario tables are unreadable for a decision because the answer is a *sign* and a
    significance, not a magnitude: 10 scenarios x 7 arms x 2 budgets of absolute errors buries
    the two questions that matter. This keeps the arms that isolate a mechanism and prints the
    paired difference against the control, which is the only column with a verdict attached.
    """
    per_scenario = sorted(RESULTS.glob("scenarios_*.json"))
    rows = []
    for path in per_scenario:
        rows.extend(json.load(open(path)))
    combined = not per_scenario  # only when there is nothing else to read
    if combined:
        rows = load("exp3_scenarios.json") or []
    if not rows:
        print("exp3: no results yet")
        return

    # Deduplicate on (scenario, n, variant): a scenario present both per-scenario and in the
    # legacy combined file would otherwise be double-counted, and the stale copy would make a
    # freshly re-run scenario look stale.
    seen, deduped = set(), []
    for r in rows:
        key = (r["scenario"], r["n"], r["variant"], r.get("bins"), r.get("scheme"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rows = deduped
    # Fresh means "has a replicate block", judged on the replicate *count* -- not on the
    # presence of a delta. The control arm has no delta by construction, and testing for one
    # classified it as stale and dropped the one row the whole table is a comparison against.
    def _is_fresh(r):
        return bool((r.get("replicates") or {}).get("replicates"))

    stale = [r for r in rows if not _is_fresh(r)]
    fresh = [r for r in rows if _is_fresh(r)]
    if stale:
        names = sorted({r["scenario"] for r in stale})
        print(f"WARNING: no replicates for {', '.join(names)} -- stale, from an earlier "
              f"reference. Re-run those scenarios before reading their verdicts.\n")

    if not fresh:
        print("no scenario has replicates yet; every row is a bare point estimate")
        return

    control = fresh[0]["replicates"].get("control", "strat-s1")
    reps = fresh[0]["replicates"].get("replicates", "?")
    ref_n = fresh[0].get("reference_n", "?")
    print("== DECISION: what does each mechanism add, paired against the control? ==")
    print(f"   control: {control!r}   {reps} replicates/arm   reference {ref_n} particles")
    print("   Δ is (arm - control) in spectrum L1; negative means the arm beat the control.\n")
    header = (f"{'scenario':>16} {'N':>6}  {'arm':>14} {'spectrum':>16} "
              f"{'Δ vs control':>17} {'N_eff':>7}  verdict")
    print(header)
    print("-" * len(header))

    rollcall = {}
    for scenario in sorted({r["scenario"] for r in fresh}):
        subset = [r for r in fresh if r["scenario"] == scenario]
        for n in sorted({r["n"] for r in subset}):
            for r in subset:
                if r["n"] != n or r["variant"] not in DECISION_ARMS:
                    continue
                block = r["replicates"]
                d = (block.get("delta") or {}).get("spectrum") or {}
                delta = (f"{d['mean']:+.2e}±{d['sem']:.0e}" if d and math.isfinite(d.get("sem", float("nan")))
                         else "(control)")
                print(f"{scenario:>16} {n:>6}  {r['variant']:>14} "
                      f"{_pm(block, 'spectrum', 16):>16} {delta:>17} "
                      f"{block['n_eff']['mean']:>7.0f}  {block['verdict']}")
            if n == max(x["n"] for x in subset):
                for arm in ("full", "s1-lam0.75", "strat-s1.41"):
                    row = next((x for x in subset
                                if x["n"] == n and x["variant"] == arm), None)
                    if row:
                        rollcall.setdefault(scenario, {})[arm] = row["replicates"]["verdict"]

    print("\n-- roll call at the largest budget: what FULL does vs the control " + "-" * 8)
    for scenario, verdicts in rollcall.items():
        v = verdicts.get("full", "?")
        mark = {"control": "  ", "within noise": "  ", "needs >1 replicate": "  ?",
                "no control": "  "}.get(v, "  !")
        print(f"  {mark}{scenario:>16}: {v}")
    print("\n  ! = resolved (a real difference); blank = indistinguishable from the control.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--cell-aware", action="store_true")
    parser.add_argument("--decision", action="store_true",
                        help="compact per-scenario verdicts against the control arm")
    parser.add_argument("--scenario", default=None, help="restrict to one scenario name")
    args = parser.parse_args()
    show_all = not (args.ablation or args.scenarios or args.cell_aware or args.decision)
    if args.decision:
        show_decision()
    if args.decision and not show_all:
        return 0
    if args.ablation or show_all:
        show_ablation()
    if args.scenarios or show_all:
        show_scenarios(args.scenario)
    if args.cell_aware or show_all:
        show_cell_aware()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

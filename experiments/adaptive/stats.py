"""Replicate statistics for the experiment tables.

Every arm is run at several independent seeds. Those replicates were previously averaged and
discarded, which left each table row a bare point estimate. The reference floor does not
substitute for the missing error bar: the floor measures the *reference's* seed-to-seed spread
at ``--ref-n`` particles, while an arm at 20-40k has a seed-to-seed spread several times larger.
Comparing an arm to the floor is a category error, and it is what made an unresolvable
difference look like a result.

What decides a comparison is the **paired** difference against a control arm, computed seed by
seed. All arms see the same seed list, so differencing cancels the common random-number
correlation between arms, and the standard error of that difference is small enough to
resolve effects the floor never could.

Two things this module deliberately does not do: it does not pick a control implicitly, because
which control you pick changes the question being asked (versus IID asks "does any of this
help"; versus plain QMC s=1 asks "does the *allocation* help, holding stratification fixed");
and it does not pool across scenarios, because a pooled test would hide exactly the per-scenario
regressions that decide whether a scheme is safe to ship.

    python stats.py results/scenarios_baseline.json    # inspect one results file
"""
from __future__ import annotations

import math
import sys

import numpy as np

#: Metrics carried through the replicate statistics. All are *errors* (lower is better).
METRICS = ("yield", "marg_worst", "marg_mean", "spectrum", "centroid")

#: Sigma for the "resolved" verdict. Two is the conventional, deliberately conservative choice:
#: with ~8 replicates per arm the false-positive rate per comparison is around 5%, and a
#: 10-scenario sweep is then looking at ~0.5 expected false positives, which is acceptable.
SIGMA = 2.0


def sem(values, ddof: int = 1) -> float:
    """Standard error of the mean. NaN for a single replicate -- not zero."""
    values = np.asarray(list(values), dtype=float)
    if values.size <= ddof:
        return float("nan")
    return float(values.std(ddof=ddof) / math.sqrt(values.size))


def mean_sem(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), sem(values)


def paired_delta(arm, control) -> dict:
    """Elementwise ``arm - control`` over shared replicates, with its standard error.

    Pairing seed by seed is the point: the arms share seeds, so the difference removes the
    component of each arm's error that comes from the seed itself rather than from the
    sampling scheme.
    """
    if arm is None or control is None or len(arm) != len(control):
        return {}
    out = {}
    for key in METRICS:
        diff = np.asarray([a[key] - c[key] for a, c in zip(arm, control)], dtype=float)
        mean, err = mean_sem(diff)
        out[key] = {"mean": mean, "sem": err, "n": int(diff.size)}
    return out


def verdict(delta: dict, sigma: float = SIGMA) -> str:
    """Decide whether a paired difference is resolved. ``delta`` is ``arm - control``."""
    if not delta:
        return "no control"
    mean, err = delta["mean"], delta["sem"]
    if not np.isfinite(mean):
        return "needs >1 replicate"
    if not np.isfinite(err):
        return "needs >1 replicate"
    if abs(mean) <= sigma * err:
        return "within noise"
    # Errors are lower-is-better, so a negative delta means the arm beat the control.
    return f"RESOLVED {'better' if mean < 0 else 'WORSE'}"


def replicate_block(runs, reference, control_runs=None, *, control_name: str = "",
                    label: str = "") -> dict:
    """Replicate spread for one arm, plus its paired difference against a control arm.

    ``runs`` and ``control_runs`` must come from the *same* seed list -- pairing mismatched
    replicates would report a difference dominated by seed choice rather than by scheme.
    """
    import harness as H

    per_seed = [H.errors(run, reference) for run in runs]
    block: dict = {"replicates": len(runs)}
    if label:
        block["label"] = label
    if control_name:
        block["control"] = control_name
    for key in METRICS:
        mean, err = mean_sem([e[key] for e in per_seed])
        block[key] = {"mean": mean, "sem": err}
    control_per_seed = (
        [H.errors(run, reference) for run in control_runs] if control_runs is not None else None
    )
    block["delta"] = paired_delta(per_seed, control_per_seed)
    block["verdict"] = verdict(block["delta"].get("spectrum", {}))
    # weight_spread and n_eff are deterministic functions of the weights, with no sampling
    # noise at all, so they are reported as a plain spread rather than a standard error.
    for key in ("weight_spread", "n_eff"):
        if runs and key in runs[0]:
            values = [r[key] for r in runs]
            block[key] = {"mean": float(np.mean(values)),
                          "spread": float(max(values) - min(values))}
    return block


def fmt(block: dict, key: str, width: int = 10) -> str:
    """``mean±sem`` for one metric of a replicate block, right-aligned to `width`."""
    entry = block.get(key, {})
    mean, err = entry.get("mean", float("nan")), entry.get("sem", float("nan"))
    if not np.isfinite(err):
        return f"{mean:>{width}.2e}"
    return f"{mean:>{width - 5}.2e}±{err:.0e}"


def describe(row: dict) -> str:
    """One-line human summary of a replicate block, for eyeballing a JSON file."""
    bits = [f"{k} {fmt(row, k)}" for k in METRICS]
    det = row.get("delta", {}).get("spectrum")
    if det:
        bits.append(f"Δspectrum {det['mean']:+.2e}±{det['sem']:.0e}  {row['verdict']}")
    return "  ".join(bits)


def main(argv: list[str]) -> int:
    import json
    import pathlib

    if len(argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    rows = json.load(open(path))
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    for row in rows:
        name = f"{row.get('variant', '?')} N={row.get('n', '?')}"
        print(f"{name:34s} {describe(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

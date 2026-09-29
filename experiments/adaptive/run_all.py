"""Parallel scenario runner for the adaptive-sampling experiments.

Every scenario is completely independent — its own reference, its own edges, its own arms —
so the sweep parallelizes across processes with no shared state. On a many-core machine this
is the whole cost model: Stage 0 is CPU-bound numpy, and the only real lever is how many
scenarios run at once.

    python run_all.py --quick          # ~5 min sanity pass
    python run_all.py                  # the full pass
    python run_all.py --only tight_focus crossing
    python run_all.py --workers 8 --ref-n 4000000 --ref-seeds 2 --replicates 8
    python run_all.py --stage all      # scenarios + ablation + cell-aware

The reference sets the floor; the replicates set the uncertainty. They answer different
questions and neither substitutes for the other -- an arm's seed-to-seed spread at 40k is
several times the reference's spread at 4M, so "is this difference bigger than the floor" is
not a significance test. What decides it is the paired difference against a control arm, run
over the same seeds.

Results land in ``results/`` as JSON, one file per (experiment, scenario), so a run that is
interrupted keeps everything already computed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def env() -> dict:
    """Child-process environment.

    ``PYTHONPATH`` is the important one: the venv's editable install points at whichever
    checkout it was installed from, so without this the workers would import the *other*
    GammaForge and silently measure the wrong code (see NOTES.md in the repo root).
    """
    e = dict(os.environ)
    src = str(HERE.parents[1] / "src")
    existing = e.get("PYTHONPATH")
    e["PYTHONPATH"] = f"{src}:{existing}" if existing else src
    # One BLAS thread per worker: the sweep is process-parallel, and letting each worker also
    # spawn a thread pool oversubscribes and slows the whole thing down.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        e[var] = "1"
    return e


def launch(script: str, args: list[str], tag: str) -> tuple[str, int]:
    log = RESULTS / f"{tag}.log"
    handle = open(log, "w")
    process = subprocess.Popen(
        [sys.executable, str(HERE / script), *args],
        stdout=handle, stderr=subprocess.STDOUT, env=env(), cwd=str(HERE),
    )
    return tag, process.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="tiny budgets, for a smoke test of the plumbing")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="parallel scenario workers (default: cpu_count - 2)")
    parser.add_argument("--ref-n", type=int, default=400_000,
                        help="particles per reference run (raise this to tighten the floor)")
    parser.add_argument("--ref-seeds", type=int, default=2,
                        help="independent reference runs to average. This sets the *floor* -- how "
                             "well the reference itself converges -- and is NOT the uncertainty "
                             "of an arm; for that see --replicates")
    parser.add_argument("--replicates", type=int, default=8,
                        help="independent seeds per arm, paired across arms (default 8). This is "
                             "what actually decides whether one arm beats another; the floor "
                             "cannot, and comparing an arm to the floor is a category error")
    parser.add_argument("--control", default="strat-s1",
                        help="control arm for the paired comparison (scenarios only)")
    parser.add_argument("--stage", default="scenarios",
                        choices=["scenarios", "ablation", "cell-aware", "all"],
                        help="which experiment to run. 'all' exists because the stages write to "
                             "different files: running only 'scenarios' leaves exp1 and exp5 "
                             "sitting at whatever reference strength they were last run with, "
                             "which is how a stale 'no benefit' conclusion survives a re-run")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these scenario names")
    parser.add_argument("--skip-completed", action="store_true",
                        help="skip a scenario whose results JSON already exists")
    args = parser.parse_args()

    import preflight
    preflight.check()

    RESULTS.mkdir(exist_ok=True)
    import harness as H  # imported here so --help works without the venv

    bank = [s.name for s in H.scenarios()]
    if args.only:
        bank = [name for name in bank if name in set(args.only)]

    common = []
    if args.quick:
        common += ["--quick"]
    common += ["--ref-n", str(args.ref_n), "--ref-seeds", str(args.ref_seeds)]
    # --only is deliberately NOT in `common`: each per-scenario child gets its own single
    # --only below, and passing it twice makes argparse keep only the last, which silently
    # reduced `--only a b` to a single scenario.

    stages = ["scenarios", "ablation", "cell-aware"] if args.stage == "all" else [args.stage]
    print(f"stage(s): {', '.join(stages)}")
    print(f"scenarios: {len(bank)}  workers: {args.workers}")
    print(f"reference: {args.ref_n} particles x {args.ref_seeds} seeds "
          f"({'quick' if args.quick else 'full'})")
    print(f"replicates: {args.replicates} seeds/arm, paired against control "
          f"{args.control!r}")
    print(f"results -> {RESULTS}\n", flush=True)

    # exp1 and exp5 are single-process, not per-scenario, so they are run here directly rather
    # than through the per-scenario fan-out below.
    if "ablation" in stages or "cell-aware" in stages:
        for stage, script, extra in (
            ("ablation", "exp1_ablation.py", []),
            ("cell-aware", "exp5_cell_aware.py", []),
        ):
            if stage not in stages:
                continue
            # exp1 is baseline-only by construction and has no --only; exp5 takes one.
            only = ["--only", *args.only] if (stage == "cell-aware" and args.only) else []
            argv = [sys.executable, str(HERE / script), *common, "--replicates",
                    str(args.replicates), *only, *extra]
            log = RESULTS / f"{stage.replace('-', '_')}.log"
            print(f"  running {stage} -> {log.name}", flush=True)
            with open(log, "w") as handle:
                code = subprocess.call(argv, stdout=handle, stderr=subprocess.STDOUT,
                                       env=env(), cwd=str(HERE))
            print(f"  {'ok  ' if code == 0 else 'FAIL'} {stage} (exit {code})", flush=True)
            if code != 0:
                print(f"  see {log}")

    if "scenarios" not in stages:
        return 0

    start = time.perf_counter()
    pending = []
    for name in bank:
        out = RESULTS / f"scenarios_{name}.json"
        if args.skip_completed and out.exists():
            print(f"  skip {name} (exists)")
            continue
        tag, pid = launch("exp3_scenarios.py",
                          common + ["--only", name, "--replicates", str(args.replicates),
                                    "--control", args.control],
                          f"scenarios_{name}")
        pending.append((name, pid, out, tag))
        time.sleep(1.0)  # stagger so references do not all peak at once

    running = dict((pid, (name, out)) for name, pid, out, _ in pending)
    done, failed = [], []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures_ = {pool.submit(_wait, pid): pid for _, pid, _, _ in pending}
        for future in futures.as_completed(futures_):
            pid = futures_[future]
            name, out = running[pid]
            code = future.result()
            (done if code == 0 else failed).append(name)
            print(f"  {'ok  ' if code == 0 else 'FAIL'} {name} (exit {code})  "
                  f"[{len(done)+len(failed)}/{len(pending)}, {time.perf_counter()-start:.0f}s]",
                  flush=True)

    print(f"\n{'='*70}")
    print(f"scenarios finished in {time.perf_counter()-start:.0f}s; "
          f"{len(done)} ok, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
        print(f"see {RESULTS}/*.log")
    return 1 if failed else 0


def _wait(pid: int) -> int:
    """Block on a child by pid without busy-waiting."""
    while True:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if done == pid:
            return os.waitstatus_to_exitcode(_)
        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())

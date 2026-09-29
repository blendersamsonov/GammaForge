# Adaptive-sampling experiments (RES092 follow-up)

Diagnostics for `docs/handoffs/luminosity-aware-adaptive-sampling.md` and RES092. **Nothing
here is production code** — the shipped module is `src/gammaforge/io/adaptive_sampling.py`
and these scripts do not modify it. Everything lives outside `src/`, takes its configuration
from CLI flags, and writes JSON.

## Reading the results: the floor is not an error bar

This is the thing worth knowing before you read any table here.

`--ref-n` sets the **floor**: the spread of the reference across its own seeds. It answers
"how well does the reference converge?" It is **not** the uncertainty of an arm. A 40k arm's
seed-to-seed spread is several times the 4M reference's, so an error that sits above the
floor is not thereby significant, and one below it is not thereby noise. Comparing an arm to
the floor is a category error, and doing it is how a 10% difference got read as a result.

What decides an arm-vs-arm comparison is the **paired** difference. Every arm sees the same
`--replicates` seeds, so differencing seed by seed cancels the common random-number
correlation, and `--replicates` sets how many seeds that estimate rests on:

```
spectrum      6.44e-03±4e-04     mean ± standard error over replicates
Δ spectrum   -2.29e-03±6e-04     paired difference vs the control arm
verdict      RESOLVED better     |Δ| > 2σ, so the difference is not seed noise
```

The **control** is chosen, not implicit, because which control you pick changes the question:

| control | question it answers |
|---|---|
| `strat-s1` (default) | does the *allocation* help, holding stratification fixed? |
| `iid` | does the whole scheme beat plain sampling? |

Both are worth asking; they are not the same question, and the λ/scale arms only mean
something relative to the first.

`--stage all` exists because the three experiments write to different files. Running only
`scenarios` leaves exp1 and exp5 sitting at whatever reference strength they were last run
with -- which is how a stale "no benefit for cell-aware" conclusion survives a re-run that was
supposed to revisit it. Each table now prints the reference size and replicate count it was
produced with, so a stale file announces itself.

Old result files have no replicate block; `summarize` still reads them, marks them
`not measured`, and says so in the header rather than quietly printing bare point estimates.

## Running

**Use `./python.sh`.** It exists because two silent-failure modes make a bare `python` run
produce plausible numbers from the wrong code:

1. The venv is created with `python3 -> /usr/bin/python3`. Copied to a machine whose Python
   lives elsewhere, `.venv/bin/python` is a **dangling symlink** — non-executable, so
   `which python` keeps reporting the system interpreter and `source activate` appears to
   succeed while changing nothing usable.
2. The venv's editable install points at whichever checkout it was created from, and
   `PYTHONPATH` loses to it unless set. A run then measures a *different* GammaForge.

`python.sh` selects an interpreter **by capability, not by path**: it collects candidate
venvs (this checkout, the main checkout via `git rev-parse --git-common-dir`, any
*neighbouring* checkout, and a few levels up), keeps the first whose interpreter can
`import pint, numpy`, and falls back to a `python3` on `PATH` that can. It then forces
`PYTHONPATH=$repo/src` on top.

The capability test is what makes it work in the three layouts that actually occur: a real
worktree (venv in the main checkout), a **bundle clone sitting next to the main repo** (the
git lookup points at the clone itself and cannot see the sibling venv — this is the case
that bit), and a machine where the venv was never created. `./python.sh --show-env` prints
what it selected and why.

```bash
cd <repo>/experiments/adaptive

./python.sh run_all.py --quick                 # smoke test
./python.sh run_all.py                         # full bank, parallel
./python.sh run_all.py --skip-completed        # resume
./python.sh run_all.py --ref-n 4000000 --replicates 8
./python.sh run_all.py --stage all             # scenarios + ablation + cell-aware
./python.sh summarize.py                       # print the tables
./python.sh preflight.py                       # verify tree + code freshness
```

If it reports a broken venv, either point at a good one
(`export GAMMAFORGE_VENV=/path/to/venv`) or rebuild it.

**`preflight.py` runs automatically before every experiment** and refuses to start on a
wrong tree or on pre-`970e0f9` code. The staleness check is the non-obvious one: before that
commit every region's pilot second moment was zero, so the allocation silently fell back to
`B_m` and a "lambda = 0.75" experiment actually measured uniform allocation. It is worth
keeping, because that failure looks exactly like a working run.

For reference, the equivalent manual incantation:

```bash
cd <repo>/experiments/adaptive
export PYTHONPATH=$(cd ../.. && pwd)/src
source <repo>/.venv/bin/activate

python run_all.py --quick                      # smoke test, ~10 min here
python run_all.py                              # full scenario bank
python run_all.py --skip-completed             # resume an interrupted run
python run_all.py --only tight_focus crossing  # subset
python run_all.py --ref-n 1500000 --ref-seeds 5   # tighter measurement floor
```

Scenarios are fully independent (own reference, own edges, own arms), so `run_all.py`
parallelizes them across processes. One BLAS thread per worker, set automatically — the
sweep is process-parallel and thread-parallel workers would oversubscribe.

The individual experiments can also be run directly, and each is a superset of what
`run_all.py` calls:

```bash
python exp1_ablation.py  --ref-n 800000 --ref-seeds 3    # ablation + lambda + scale sweeps
python exp3_scenarios.py --only tight_focus --bins both --scheme both
python exp5_cell_aware.py --only baseline tight_focus    # the Neyman cell-aware study
```

## The GPU is not used

Stage 0 runs on the NumPy backend here, so an RTX A5000 is idle. Two options if Stage 0
dominates your wall clock: pass `backend="cupy"` to `integrate_trajectories` in
`harness.run` (the fixed-edge deposition stays NumPy on host arrays, which is what the
RES083 stage boundaries guarantee), or just add workers. Process parallelism is the safer
first move — the GPU path is a correctness risk for a diagnostic harness, since a silent
fallback to CPU would make the timings meaningless without failing.

## Files

| File | What it is |
|---|---|
| `harness.py` | Fixed-edge deposition, marginal metrics, variants, scenario bank |
| `exp1_ablation.py` | 6-arm ablation, lambda sweep, proposal-scale sweep (baseline) |
| `exp3_scenarios.py` | Wider scenario bank x production/reduced bins x nearest/CIC |
| `exp5_cell_aware.py` | Cell-aware Neyman allocation, pilot and oracle |
| `run_all.py` | Parallel driver over scenarios |
| `results/` | JSON + logs, including results already computed |

## Two things to know before reading the numbers

**`table_l1` (raw 5D density L1) is not a usable metric.** A 5D table at `(24,16,16,32,8)`
has 1.57e6 cells; a 400k-particle reference disagrees with *itself* at L1 ~ 0.14-0.20 across
seeds, and a 40k run sits at ~0.45. It measures which cells are empty, not the estimator.
Use `marg_worst` / `marg_mean` (marginals of the 5D table) and the Stage-2 spectrum instead.

**The reference noise floor is reported per run and bounds everything.** Independent
reference seeds disagree by 1e-4 (yield) to 1e-2 (marginals) depending on the scenario. Any
error below the floor is measuring the reference. Raising `--ref-n` / `--ref-seeds` is the
only way to tighten it, and it is usually the highest-value knob.

## Already-computed results

`results/` carries what was computed on the development machine before it was moved:

- `exp1_ablation.json` — 42 rows, full ablation/lambda/scale sweep, 800k x 3 reference
- `exp3_scenarios.json` — 112 rows, 8 scenarios at reduced bins / nearest
- `scenarios_*.json` — 10 rows each, full bank at `--quick`
- `exp5_cell_aware.json` — 12 rows, baseline only, both plan configurations

`scenarios_baseline.json` / `scenarios_tight_focus.json` from the quick pass are placeholders;
delete them or pass `--skip-completed` to force a real run.

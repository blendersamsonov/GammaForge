# Working in this worktree

Branch `feature/luminosity-aware-adaptive-sampling`, based on `main` @ `4540929`.
Implements `docs/handoffs/luminosity-aware-adaptive-sampling.md` (adaptive stratified
bunch sampling, RES092). The feature is **opt-in** via `SamplingSpec.strategy`; `"iid"`
remains the default.

## The venv is shared, and it points at the *main* checkout

`/home/alexander/Work/Code/GammaForge/.venv` is a single shared virtualenv whose editable
install adds the **main** checkout's `src` to `sys.path` (via
`_editable_impl_gammaforge.pth`, a plain path entry — not a meta-path finder).

So running `pytest` here without help imports `gammaforge` from
`/home/alexander/Work/Code/GammaForge/src` and silently tests the wrong tree. Always:

```bash
cd /home/alexander/Work/Code/GammaForge-adaptive-sampling
source ../GammaForge/.venv/bin/activate
export PYTHONPATH=/home/alexander/Work/Code/GammaForge-adaptive-sampling/src
python -m pytest tests/test_adaptive_sampling.py -q
```

Verify with `python -c "import gammaforge; print(gammaforge.__file__)"` before trusting any
test result. `PYTHONPATH` wins because `.pth` entries are appended to `sys.path` after it.

## Do not edit the main checkout

`/home/alexander/Work/Code/GammaForge` holds another session's **uncommitted** work
(RES091 `deposit_shape_table` support, cupy delta reference, notebooks, PROGRESS.md, and
more). It is on `main` with 18 modified + 12 untracked paths. Nothing in this feature may
touch it — in particular `engines/xigma/stages.py` is modified there, so any edit to it
here must be made against the *committed* version, not the working-tree one.

## Bugs this feature's tests caught

Recorded because each is invisible to the obvious invariant:

- **Conditional sampling in the wrong CDF coordinates.** A region is axis-aligned in the
  *reference* coordinate `u = Phi(d/s)`, so its edges in the target's CDF are
  `Phi(s Phi^-1(a))`, not `a` — equal only at `s == 1`. Drawing with `a`/`b` samples the
  *reference* conditional. `sum(P_m) == 1` and `sum(w) == 1` both still hold; only *where*
  the mass sits is wrong. Signature: the weighted error stops shrinking as `N` grows.
- **`_normal_interval_mass` cancelling to exactly zero** for bounds deep in the lower tail
  (`(-38, -36)` returned 0 instead of 4e-284), because it used the `Q` pair there. Only
  reachable with a fine enough lower-tail partition, so the mass sum hid it.
- **`norm_ppf` coefficient transcription.** AS 241's coefficient order and the `+1` on its
  denominators are easy to get wrong, and a wrong-but-monotone quantile function still
  satisfies `sum(P_m) == 1`. Replaced with Halley iteration on libm's `erfc` — no
  coefficients to mistype, ~2.8e-14 over the whole range, and validated against
  `statistics.NormalDist().inv_cdf`.

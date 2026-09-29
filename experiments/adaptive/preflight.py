"""Fail fast if this would measure the wrong GammaForge.

The venv's editable install points at whichever checkout it was created from, and
`PYTHONPATH` loses to it unless set. A run that imports the wrong tree produces numbers
that look entirely normal, so every entry point checks three things before doing any work:

1. ``gammaforge`` resolves to *this* checkout's ``src``;
2. ``gammaforge.io.adaptive_sampling`` is importable at all (it does not exist on ``main``);
3. the allocation bug fixed in 970e0f9 is actually fixed in the imported code.

That last one is the non-obvious check. ``build_adaptive_plan`` used to leave every region's
pilot second moment at zero, which collapsed the allocation to ``B_m`` for every lambda --
and the shipped defaults would then look like a working ``lambda = 0.75`` experiment while
measuring uniform allocation. Verifying the moments are populated is the difference between
"the numbers are wrong" and "the numbers are wrong and I know why".
"""
from __future__ import annotations

import pathlib
import sys


def check(strict_path: bool = True) -> None:
    import gammaforge

    root = pathlib.Path(gammaforge.__file__).resolve().parents[2]
    here = pathlib.Path(__file__).resolve().parents[2]
    if strict_path and root != here:
        sys.exit(
            f"WRONG TREE: gammaforge imported from {root}\n"
            f"          expected {here}\n"
            f"          the venv's editable install is pointing elsewhere. Run through\n"
            f"          experiments/adaptive/python.sh, or export PYTHONPATH={here}/src"
        )

    import numpy as np
    from gammaforge.io.adaptive_sampling import PilotConfig, build_adaptive_plan
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    shares = []
    for lam in (0.0, 1.0):
        plan = build_adaptive_plan(
            beam, laser, seed=3,
            config=PilotConfig(initial_regions=16, max_regions=16,
                               pilot_points_per_region=6, pilot_quad_nodes=64,
                               pilot_quad_panels=2, luminosity_fraction=lam),
        )
        if not all(r.pilot_second_moment > 0.0 for r in plan.regions):
            sys.exit(
                "STALE CODE: plan regions have zero pilot second moments, so the "
                "luminosity allocation is inert (the bug fixed in 970e0f9). You are "
                "running a checkout older than that commit."
            )
        share = np.array([r.allocation_probability for r in plan.regions])
        shares.append(share / share.min())
    if np.allclose(shares[0], shares[1]):
        sys.exit(
            "STALE CODE: lambda=0 and lambda=1 give the same allocation, which is the "
            "signature of the inert-allocation bug (pre-970e0f9)."
        )

    print(f"preflight ok: gammaforge from {root}")
    print(f"              adaptive_sampling importable, allocation responds to lambda")


if __name__ == "__main__":
    check()

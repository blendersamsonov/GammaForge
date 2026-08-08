"""The `Collision` facade and `XigmaEngine` (GRAND_PLAN.md §4.2, Phase 3a exit).

Small table resolutions throughout — these tests check architecture (which outputs get
filled, that caching actually caches, that the facade's identity holds), not spectral
accuracy; `test_stage1_stage2.py` already covers the kernel itself at realistic
resolution.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.base import Engine
from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.validation import scenarios

_SMALL_BINS = dict(
    n_bins_gamma=10, n_bins_theta_x=10, n_bins_theta_y=10, n_bins_a0_shape=16, n_bins_ahat=4
)


def _interaction(n_particles=4000, outputs=()):
    small = replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=n_particles))
    interaction = scenarios.build(small)
    return replace(interaction, target=replace(interaction.target, outputs=outputs))


def _engine_params(**overrides):
    return XigmaEngine.schema.with_values(**_SMALL_BINS, **overrides)


def test_xigma_engine_conforms_to_the_engine_protocol():
    assert isinstance(XigmaEngine(), Engine)


def test_supported_outputs_matches_what_run_actually_fills():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(6,)),
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(3, 3)),
        OutputRequest(OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION, resolution=(4, 3, 3)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(4, 3, 3)),
    )
    interaction = _interaction(outputs=requests)
    results = XigmaEngine().run(interaction, _engine_params())
    assert set(results.photon_slices) == set(XigmaEngine.supported_outputs)


def test_unsupported_output_kinds_are_silently_omitted_not_errored():
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.MACROPARTICLE_DUMP))
    )
    results = XigmaEngine().run(interaction, _engine_params())
    assert set(results.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_build_overlap_is_memoized():
    interaction = _interaction()
    collision = Collision(interaction=interaction, params=_engine_params())
    first = collision.build_overlap()
    second = collision.build_overlap()
    assert first is second


def test_shape_is_memoized():
    """Stage 1's shape deposit runs at most once per `Collision`, regardless of how many
    distinct peak a0 values `_table()` is later asked for (`DECISIONS.md` D032)."""
    interaction = _interaction()
    collision = Collision(interaction=interaction, params=_engine_params())
    first = collision._shape()
    collision._table()
    collision._table(a0_peak=2.0 * collision.build_overlap().a0_peak)
    second = collision._shape()
    assert first is second


def test_table_is_memoized_per_a0_peak():
    interaction = _interaction()
    collision = Collision(interaction=interaction, params=_engine_params())
    table_a = collision._table()
    table_b = collision._table()
    assert table_a is table_b
    table_c = collision._table(a0_peak=2.0 * collision.build_overlap().a0_peak)
    assert table_c is not table_a
    # luminosity retargets as a0_peak**2 along with ahat (TrajectorySamples.
    # retargeted_luminosity, applied through retarget_ahat) — twice the peak a0 is 4x the
    # total weight, exactly (the regrid's overlap weights are row-stochastic, D032).
    assert table_c.total_weight == pytest.approx(4.0 * table_a.total_weight, rel=1e-9)


def test_total_yield_and_spectrum_integral_converge_to_the_same_number():
    """`stages.angle_integrated_spectrum`'s shape integrates to exactly 1 (module
    docstring in `stages.py`/`delta.py`), so `SPECTRUM`'s integral should converge to
    `TOTAL_YIELD` as resolution grows — the same identity §7 checks elsewhere, here
    exercised through the facade and `PhasespaceSlice.integrate`'s own trapezoid rule.
    """
    interaction = _interaction(n_particles=3000)
    errors = []
    for n in (60, 600):
        target = replace(
            interaction.target,
            outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(n,))),
        )
        results = XigmaEngine().run(replace(interaction, target=target), _engine_params())
        total = results.photon_slices[OutputKind.TOTAL_YIELD].integrate()
        spec = results.photon_slices[OutputKind.SPECTRUM].integrate()
        errors.append(abs(spec - total) / total)
    assert errors[1] < errors[0]
    assert errors[1] < 0.01


def test_angular_and_collimated_slices_are_nonnegative():
    requests = (
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(4, 4)),
        OutputRequest(OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION, resolution=(5, 4, 4)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(5, 4, 4)),
    )
    interaction = _interaction(n_particles=3000, outputs=requests)
    results = XigmaEngine().run(interaction, _engine_params())
    for kind in (OutputKind.ANGULAR_DISTRIBUTION, OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION, OutputKind.COLLIMATED_SPECTRUM):
        assert np.all(results.photon_slices[kind].distr >= 0.0)


def test_spectral_angular_distribution_axis_order_matches_slice_axes():
    requests = (OutputRequest(OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION, resolution=(5, 4, 3)),)
    interaction = _interaction(n_particles=2000, outputs=requests)
    results = XigmaEngine().run(interaction, _engine_params())
    sl = results.photon_slices[OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION]
    assert sl.axis_order == (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y)
    assert sl.distr.shape == (5, 4, 3)

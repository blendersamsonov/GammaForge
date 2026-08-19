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
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import sample_gaussian_bunch
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import Quantity as Q
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
    distinct peak a0 values `_table()` is later asked for (RES032)."""
    interaction = _interaction()
    collision = Collision(interaction=interaction, params=_engine_params())
    first = collision._shape()
    collision._table()
    collision._table(intensity_peak=2.0 * collision.build_overlap().intensity_peak)
    second = collision._shape()
    assert first is second


def test_table_is_memoized_per_pulse_strength():
    interaction = _interaction()
    collision = Collision(interaction=interaction, params=_engine_params())
    table_a = collision._table()
    table_b = collision._table()
    assert table_a is table_b
    table_c = collision._table(intensity_peak=2.0 * collision.build_overlap().intensity_peak)
    assert table_c is not table_a
    # luminosity retargets linearly in the peak intensity along with ahat
    # (TrajectorySamples.retargeted_luminosity, applied through retarget_ahat) — twice the
    # peak <a^2> is exactly 2x the total weight (the regrid's overlap weights are
    # row-stochastic, RES032).
    assert table_c.total_weight == pytest.approx(2.0 * table_a.total_weight, rel=1e-9)


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


def test_the_two_normalization_paths_inside_one_results_object_agree():
    """`Results` mixes two independent spectral paths, and only this compares them (RES033).

    `SPECTRUM` comes from `stages.angle_integrated_spectrum` — Stage 0's own closed form,
    which never touches the table or its kernel constant. Every angular output comes from
    the table kernel; the engine tests elsewhere check signs, axis order and shapes, never
    magnitudes across both paths at once.

    Integrating `COLLIMATED_SPECTRUM` back over its two angle axes has to reproduce
    `SPECTRUM` — which needs a collimation window wide enough to stand in for the whole
    radiation cone, so this test manually widens `Target`'s collimation half-angles to
    ``~4.6/gamma0`` instead of the baseline scenario's much narrower default. It lands
    somewhat low, and the deficit is understood window/quadrature slack, not a hidden
    factor of ``2 pi`` (RES033 has the numbers).
    """
    interaction = _interaction(n_particles=4000)
    wide = 4.6 / interaction.beam.gamma0()
    requests = (
        OutputRequest(OutputKind.SPECTRUM, resolution=(80,)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(80, 25, 25)),
    )
    target = replace(
        interaction.target, theta_x_col=Q(wide, "rad"), theta_y_col=Q(wide, "rad"), outputs=requests
    )
    interaction = replace(interaction, target=target)
    params = XigmaEngine.schema.with_values(
        n_bins_gamma=32, n_bins_theta_x=24, n_bins_theta_y=24,
        n_bins_a0_shape=64, n_bins_ahat=8, scheme="cic",
    )
    results = XigmaEngine().run(interaction, params)

    spectrum = results.photon_slices[OutputKind.SPECTRUM]
    cube = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    energy = spectrum.axes[Axis.ENERGY]
    theta_x, theta_y = cube.axes[Axis.THETA_X], cube.axes[Axis.THETA_Y]

    dN_dE = np.trapezoid(np.trapezoid(cube.distr, theta_y, axis=2), theta_x, axis=1)
    ratio = float(np.trapezoid(dN_dE, energy)) / float(np.trapezoid(spectrum.distr, energy))

    assert 0.75 < ratio < 1.05


def test_angular_and_collimated_slices_are_nonnegative():
    requests = (
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(4, 4)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(5, 4, 4)),
    )
    interaction = _interaction(n_particles=3000, outputs=requests)
    results = XigmaEngine().run(interaction, _engine_params())
    for kind in (OutputKind.ANGULAR_DISTRIBUTION, OutputKind.COLLIMATED_SPECTRUM):
        assert np.all(results.photon_slices[kind].distr >= 0.0)


# ---------------------------------------------------------------------------
# Trajectory-sampling window (io.bunch.illumination_window)
# ---------------------------------------------------------------------------
def test_integrate_trajectories_rejects_an_unknown_window():
    bunch = sample_gaussian_bunch(scenarios.BASELINE.beam, 64, 0)
    with pytest.raises(ValueError, match="window must be"):
        integrate_trajectories(bunch, scenarios.BASELINE.laser,
                               scenarios.BASELINE.beam.n_electrons(), window="cone")


def test_illumination_window_resolves_stage_0_better_at_a_coarse_step_budget():
    """The payoff, and its limits. Spending the same steps over the illuminated stretch
    rather than the wider geometric bound converges faster while steps are scarce — 17x at
    50 steps on the baseline. It is *not* a uniform win: the illuminated window truncates at
    its threshold, so past the point where that floor dominates the geometric window keeps
    improving and this one does not. Hence the default stays `active_region`."""
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 4_000, 0)
    n_e = beam.n_electrons()
    reference = integrate_trajectories(bunch, laser, n_e, n_steps=4000, threshold=1e-9).total_yield()

    def error(window, n_steps):
        got = integrate_trajectories(bunch, laser, n_e, n_steps=n_steps,
                                     threshold=1e-6, window=window).total_yield()
        return abs(got / reference - 1.0)

    assert error("illumination", 50) < 0.2 * error("active_region", 50)
    assert error("illumination", 20) < error("active_region", 20)

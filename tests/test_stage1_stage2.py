"""Stage 1 (a0_shape deposition), the ahat retarget, and Stage 2 (spectrum queries),
GRAND_PLAN.md §4.2, Phase 3a (RES032).

Stage 1's job is conservation: every particle's weight lands somewhere in the shape
table, and the table's own total agrees with Stage 0's regardless of resolution or scheme.
The retarget's job is a second, independent conservation (total weight is preserved exactly
through the regrid) plus placing mass where the chosen peak a0 actually puts it — denser
near `ahat_max`, folded into the floor below `ahat_min`. Stage 2's job is the resonance
condition: the spectrum it reports must depend on ahat the way the physics does (the
nonlinear redshift), and it must agree with `delta` — an independent, table-free
implementation of the same differential form — to within grid/interpolation error.

Most of those are ratios between two paths carrying the same normalization, which is what
makes them robust and also what makes them blind: `test_the_table_kernel_angle_integrates_to_stage_0_total`
is the one absolute check, and the one that pins §9.1's constant (RES033).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import stages
from gammaforge.engines.xigma.stages import (
    ShapeTable,
    TrajectorySamples,
    _ahat_target_edges,
    angle_integrated_spectrum,
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
    spectrum_from_table,
    spectrum_in_angular_range,
)
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta import captured_fraction, resonance_spectrum


def _samples(scenario, n_particles=20_000, **kwargs):
    small = replace(scenario, sampling=replace(scenario.sampling, n_particles=n_particles))
    interaction = scenarios.build(small)
    return integrate_trajectories(
        interaction.bunch, interaction.laser, interaction.N_e, n_steps=64, **kwargs
    )


@pytest.fixture(scope="module")
def baseline():
    return _samples(scenarios.BASELINE)


def _synthetic_samples(n=20_000, gamma0=2000.0, seed=0, a0_shape=1.0, intensity_peak=0.045):
    """``intensity_peak`` is the peak cycle-averaged ``<a^2>`` (RES054), so ``ahat`` is just
    ``intensity_peak * a0_shape``. The default reproduces the ``a0_peak=0.3`` linear pulse
    these fixtures used before the rename: ``0.5 * 0.3**2 = 0.045``."""
    rng = np.random.default_rng(seed)
    gamma = gamma0 + rng.normal(0.0, gamma0 * 5e-3, n)
    theta_x = rng.normal(0.0, 3.0 / gamma0, n)
    theta_y = rng.normal(0.0, 3.0 / gamma0, n)
    return TrajectorySamples(
        gamma=gamma,
        theta_x=theta_x,
        theta_y=theta_y,
        a0_shape=np.full(n, a0_shape),
        luminosity=np.full(n, 1e4),
        intensity_peak=intensity_peak,
        n_steps=10,
    )


def _table(samples, *, shape_bins=(16, 16, 16, 16), scheme="nearest", **retarget_kwargs):
    """Shorthand: the two-call chain most tests need, at a modest, fast scale."""
    shape_table = deposit_shape_table(samples, n_bins=shape_bins, scheme=scheme)
    return retarget_ahat(shape_table, samples.intensity_peak, **retarget_kwargs)


def _unchunked_angle_integrated_spectrum(samples, s):
    """The direct formula, retained here only as a small regression reference."""
    s_values = np.atleast_1d(np.asarray(s, dtype=float))
    gamma_squared = (samples.gamma**2)[:, None]
    y = s_values[None, :] / gamma_squared
    shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    out = np.sum(samples.luminosity[:, None] * shape / gamma_squared, axis=0)
    return out if np.ndim(s) else out[0]


def test_angle_integrated_spectrum_chunks_both_reduction_axes(monkeypatch):
    samples = replace(_synthetic_samples(n=17), luminosity=np.linspace(1.0, 4.0, 17))
    s = np.linspace(0.1, 1.1, 7) * samples.gamma.mean() ** 2
    expected = _unchunked_angle_integrated_spectrum(samples, s)

    monkeypatch.setattr(stages, "SPECTRUM_MAX_ENERGY_CHUNK", 3)
    monkeypatch.setattr(
        stages, "SPECTRUM_WORKING_SET_BYTES", 3 * stages._SPECTRUM_BYTES_PER_PARTICLE_ENERGY
    )
    calls = []
    original = stages.run_in_chunks

    def spy(n_items, work, **kwargs):
        def observed(start, stop):
            calls.append((start, stop, kwargs["bytes_per_item"]))
            return work(start, stop)

        return original(n_items, observed, **kwargs)

    monkeypatch.setattr(stages, "run_in_chunks", spy)
    actual = angle_integrated_spectrum(samples, s)

    assert actual == pytest.approx(expected, rel=1e-14)
    assert angle_integrated_spectrum(samples, float(s[2])) == pytest.approx(expected[2], rel=1e-14)
    assert {bytes_per_item // stages._SPECTRUM_BYTES_PER_PARTICLE_ENERGY
            for _, _, bytes_per_item in calls} == {1, 3}
    assert all((stop - start) * bytes_per_item <= stages.SPECTRUM_WORKING_SET_BYTES
               for start, stop, bytes_per_item in calls)


def test_angle_integrated_spectrum_rejects_an_output_larger_than_the_memory_budget(monkeypatch):
    monkeypatch.setattr(stages.chunking, "available_ram_bytes", lambda: 100)
    with pytest.raises(MemoryError, match="requested output grid"):
        angle_integrated_spectrum(_synthetic_samples(n=4), np.linspace(0.0, 1.0, 20))


def test_cupy_request_rejects_geometry_it_does_not_implement():
    table = _table(_synthetic_samples(n=100), shape_bins=(4, 4, 4, 4))
    with pytest.raises(NotImplementedError, match="non-zero ellipticity or crossing angles"):
        angular_spectrum_from_table(
            table, [0.0], [0.0], [table.gamma_centers[1] ** 2],
            backend="cupy", ellipticity=0.5,
        )


def test_auto_uses_the_numpy_reference_for_unsupported_cupy_geometry():
    table = _table(_synthetic_samples(n=100), shape_bins=(4, 4, 4, 4))
    args = (table, [0.0], [0.0], [table.gamma_centers[1] ** 2])
    expected = angular_spectrum_from_table(*args, backend="numpy", theta_xz=0.01)
    actual = angular_spectrum_from_table(*args, backend="auto", theta_xz=0.01)
    assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Stage 1: deposit_shape_table / ShapeTable
# ---------------------------------------------------------------------------
def test_deposit_conserves_total_weight_nearest():
    samples = _synthetic_samples()
    table = deposit_shape_table(samples, n_bins=(24, 24, 24, 8), scheme="nearest")
    assert table.total_weight == pytest.approx(samples.total_yield(), rel=1e-12)
    assert table.H.sum() * table.bin_volume == pytest.approx(samples.total_yield(), rel=1e-9)


def test_deposit_conserves_total_weight_cic():
    samples = _synthetic_samples()
    table = deposit_shape_table(samples, n_bins=(24, 24, 24, 8), scheme="cic")
    assert table.total_weight == pytest.approx(samples.total_yield(), rel=1e-9)


def test_deposit_rejects_an_unknown_scheme():
    with pytest.raises(ValueError):
        deposit_shape_table(_synthetic_samples(n=100), scheme="bogus")


def test_deposit_handles_a_monoenergetic_zero_divergence_beam():
    n = 5000
    samples = TrajectorySamples(
        gamma=np.full(n, 2000.0),
        theta_x=np.zeros(n),
        theta_y=np.zeros(n),
        a0_shape=np.full(n, 1.0),
        luminosity=np.full(n, 1.0),
        intensity_peak=0.045,
        n_steps=10,
    )
    table = deposit_shape_table(samples, n_bins=(8, 8, 8, 4))
    assert np.all(np.isfinite(table.H))
    assert np.all(np.diff(table.gamma_edges) > 0.0)
    assert table.total_weight == pytest.approx(n, rel=1e-12)


def test_deposit_handles_an_empty_bunch():
    empty = TrajectorySamples(
        gamma=np.zeros(0), theta_x=np.zeros(0), theta_y=np.zeros(0),
        a0_shape=np.zeros(0), luminosity=np.zeros(0), intensity_peak=0.045, n_steps=10,
    )
    with pytest.raises(ValueError):
        deposit_shape_table(empty, n_bins=(8, 8, 8, 4))


def test_shape_table_rejects_a_shape_mismatched_H():
    samples = _synthetic_samples(n=100)
    table = deposit_shape_table(samples, n_bins=(4, 4, 4, 2))
    with pytest.raises(ValueError):
        type(table)(
            gamma_edges=table.gamma_edges,
            theta_x_edges=table.theta_x_edges,
            theta_y_edges=table.theta_y_edges,
            a0_shape_edges=table.a0_shape_edges,
            H=np.zeros((3, 4, 4, 2)),
            total_weight=0.0,
            scheme="nearest",
            source_intensity_peak=samples.intensity_peak,
        )


def test_deposition_is_one_vectorized_pass(monkeypatch):
    """Stage 1 calls one array deposit, rather than looping over macroparticles."""
    samples = _synthetic_samples(n=257)
    calls = []
    original = stages._deposit_nearest

    def spy(coords, weight, n_bins):
        calls.append((tuple(values.shape for values in coords), weight.shape, n_bins))
        return original(coords, weight, n_bins)

    monkeypatch.setattr(stages, "_deposit_nearest", spy)
    n_bins = (8, 7, 6, 5)
    table = deposit_shape_table(samples, n_bins=n_bins)

    assert calls == [(((257,), (257,), (257,), (257,)), (257,), n_bins)]
    assert table.H.shape == n_bins


# ---------------------------------------------------------------------------
# The retarget: _ahat_target_edges / retarget_ahat
# ---------------------------------------------------------------------------
def test_ahat_target_edges_starts_at_ahat_min_and_ends_at_ahat_max():
    edges = _ahat_target_edges(0.1, 0.5, 16, decades=1.5)
    assert edges[0] == pytest.approx(0.1)
    assert edges[-1] == pytest.approx(0.5)
    assert edges.size == 17


def test_ahat_target_edges_bin_widths_shrink_toward_the_top():
    edges = _ahat_target_edges(0.0, 0.5, 32, decades=1.0)
    widths = np.diff(edges)
    # Strictly decreasing except the very last bin: snapping the top edge to exactly
    # ahat_max (rather than the raw geometric value, which lands within 10**-decades of
    # it) widens that one bin — a documented, deliberate exception, not a trend break.
    assert np.all(np.diff(widths[:-1]) < 0.0)
    assert widths[-1] > widths[-2]  # the snap-widened top bin, present at decades=1.0


def test_ahat_target_edges_rejects_bad_bounds_or_decades():
    with pytest.raises(ValueError):
        _ahat_target_edges(0.5, 0.5, 16, 1.0)
    with pytest.raises(ValueError):
        _ahat_target_edges(0.0, 0.5, 16, 0.0)


def test_retarget_ahat_conserves_total_weight_exactly():
    """The identity `retarget_ahat`'s cheapness rests on: because the target edges are
    extended to +-inf for overlap purposes, every source bin's mass lands somewhere in the
    target, regardless of how much folds into the floor/ceiling bins.
    """
    samples = _synthetic_samples(a0_shape=1.0)
    shape_table = deposit_shape_table(samples, n_bins=(16, 16, 16, 16))
    for intensity_peak in (0.01, 0.045, 0.18, 0.4):
        table = retarget_ahat(shape_table, intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)
        # Linear, not quadratic: the retarget parameter is now an intensity, which is the
        # already-squared quantity (RES054).
        expected = shape_table.total_weight * (intensity_peak / samples.intensity_peak)
        assert table.total_weight == pytest.approx(expected, rel=1e-9)


def test_retarget_ahat_redistributes_mass_toward_higher_ahat_as_the_pulse_strengthens():
    samples = _synthetic_samples(a0_shape=1.0)
    shape_table = deposit_shape_table(samples, n_bins=(16, 16, 16, 16))
    table_own = retarget_ahat(shape_table, samples.intensity_peak)
    table_other = retarget_ahat(shape_table, 2.0 * samples.intensity_peak)
    # Both total_weight (exact identity above) and where that mass sits move: twice the
    # peak intensity is twice the yield *and* twice the ahat, so the populated range's own
    # top edge (after truncation, §Truncation) reaches a higher ahat (RES054).
    assert table_other.total_weight == pytest.approx(2.0 * table_own.total_weight, rel=1e-9)
    assert table_other.ahat_edges[-1] > table_own.ahat_edges[-1]


def test_retarget_ahat_folds_mass_below_ahat_min_into_the_floor_bin():
    """The floor-fold path — otherwise never exercised, since nothing in the default
    ahat_min=0.0 configuration has any ahat to fold (RES032).
    """
    samples = _synthetic_samples(a0_shape=1.0)  # ahat = 0.045 (RES053: C = 1/2)
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16))
    # ahat_min well above the population's actual ahat (0.045): everything must fold into
    # bin 0, and total weight must still be exactly conserved.
    table = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.2, ahat_max=0.5, n_bins=16, decades=1.0)
    assert table.ahat_edges[0] == pytest.approx(0.2)
    assert table.H[..., 0].sum() * np.diff(table.ahat_edges)[0] * table.gamma_theta_cell_area * (
        table.gamma_edges[-1] - table.gamma_edges[0]
    ) / table.H.shape[0] == pytest.approx(table.total_weight, rel=1e-6)
    assert table.total_weight == pytest.approx(shape_table.total_weight, rel=1e-9)


def test_retarget_ahat_truncates_unpopulated_bins():
    """Trailing target bins the rescaled source never reaches carry exactly zero mass
    (RES032) — `retarget_ahat` drops them rather than returning a table
    padded with zeros out to `ahat_max`.
    """
    samples = _synthetic_samples(a0_shape=1.0, intensity_peak=0.00125)  # ahat = 0.00125, tiny
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16))
    table = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)
    assert table.ahat_edges[-1] < 0.5
    assert table.H.shape[3] < 32


def test_retarget_ahat_truncation_does_not_change_the_kernel_output():
    """Truncation is a pure performance optimization: comparing a truncated table's
    `spectrum_from_table` output against a table padded back out with explicit zero bins
    must agree exactly.
    """
    samples = _synthetic_samples(n=5_000, a0_shape=1.0, intensity_peak=0.00125)
    shape_table = deposit_shape_table(samples, n_bins=(8, 8, 8, 16))
    truncated = retarget_ahat(shape_table, samples.intensity_peak, ahat_min=0.0, ahat_max=0.5, n_bins=32, decades=1.0)

    from gammaforge.engines.xigma.stages import Table

    pad = 5
    padded_edges = np.concatenate([truncated.ahat_edges, truncated.ahat_edges[-1] + np.arange(1, pad + 1) * 1e-3])
    padded_H = np.concatenate([truncated.H, np.zeros((*truncated.H.shape[:3], pad))], axis=3)
    padded = Table(
        gamma_edges=truncated.gamma_edges,
        theta_x_edges=truncated.theta_x_edges,
        theta_y_edges=truncated.theta_y_edges,
        ahat_edges=padded_edges,
        H=padded_H,
        total_weight=truncated.total_weight,
        scheme=truncated.scheme,
    )

    s = np.linspace(1.0, samples.gamma.max() ** 2, 20)
    assert np.allclose(
        spectrum_from_table(truncated, 0.0, 0.0, s), spectrum_from_table(padded, 0.0, 0.0, s)
    )


# ---------------------------------------------------------------------------
# Stage 2: spectrum_from_table / angular_spectrum_from_table / spectrum_in_angular_range
# ---------------------------------------------------------------------------
def test_spectrum_from_table_is_nonnegative_and_zero_past_the_compton_edge():
    samples = _synthetic_samples(gamma0=1000.0)
    table = _table(samples, shape_bins=(20, 20, 20, 16))
    edge = float(np.max(samples.gamma) ** 2)
    s = np.linspace(0.0, 1.5 * edge, 60)
    spec = spectrum_from_table(table, 0.0, 0.0, s)
    assert np.all(spec >= 0.0)
    assert np.all(spec[s > 1.1 * edge] == 0.0)


def test_the_kernel_and_delta_agree_on_where_the_redshift_puts_the_photons():
    """The one check that watches ``ahat`` itself, rather than integrating it away.

    Every other kernel-vs-`delta` comparison here and in `validation.run` compares *photon
    counts* — a sum over ``s``. The nonlinear redshift moves photons along ``s`` and
    conserves that sum exactly, so those checks are structurally blind to ``ahat``: the
    fourth identity leg reads 0.9996 whether ``ahat`` is right, doubled, or halved
    (measured, RES053). The **centroid** in ``s`` is the quantity ``ahat``
    controls, so that is what this compares.

    It cannot catch a wrong ``ahat`` *convention* — the two paths share
    `TrajectorySamples`, so a common-mode factor cancels here as it does everywhere else
    (RES053). What it does catch is the kernel losing the redshift to grid coarseness: the
    ``ahat`` axis
    must actually resolve the population for the table to reproduce `delta`'s centroid, and
    at the production defaults ``NEAR_A0_MAX`` lands in two bins.

    Teeth, measured by biasing the table's ``ahat`` axis alone (a kernel-side error `delta`
    does not share): a 1.2x bias moves the centroid 0.53%, a 2x bias 2.6% — the ``rel=2e-3``
    below catches all of them with margin, against a 0.009% residual when nothing is wrong.
    """
    samples = _samples(scenarios.NEAR_A0_MAX)
    shape_table = deposit_shape_table(samples, n_bins=(32, 48, 48, 64), scheme="cic")
    edge = float(np.max(samples.gamma)) ** 2
    s_edges = np.linspace(0.0, 1.05 * edge, 150)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    def centroid(spectrum):
        return float(np.sum(s_centers * spectrum) / np.sum(spectrum))

    # `ahat_max` narrowed to the bank's own scale: the production 0.5 is headroom for
    # pulses far brighter than any scenario here, and spends 30 of its 32 bins above them.
    fine = retarget_ahat(shape_table, samples.intensity_peak, ahat_max=0.1, n_bins=32)
    assert fine.H.shape[3] >= 8, "the point of this test is a resolved ahat axis"
    kernel = angular_spectrum_from_table(fine, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)
    assert centroid(kernel) == pytest.approx(centroid(reference), rel=2e-3)

    # And the redshift is present, not merely consistent: four times the ahat pulls the
    # centroid down by several percent, on both paths independently. Four times the
    # *intensity* now, where this used to double an amplitude for the same effect (RES054).
    bright = retarget_ahat(shape_table, 4.0 * samples.intensity_peak, ahat_max=0.4, n_bins=32)
    kernel_bright = angular_spectrum_from_table(bright, [0.0], [0.0], s_centers)[0, 0, :]
    reference_bright = resonance_spectrum(
        replace(samples, intensity_peak=4.0 * samples.intensity_peak), s_edges, 0.0, 0.0
    )
    assert centroid(kernel_bright) == pytest.approx(centroid(reference_bright), rel=2e-3)
    assert centroid(kernel_bright) < 0.97 * centroid(kernel)


@pytest.mark.parametrize(
    "scenario, expected_bias, expected_bins",
    [(scenarios.BASELINE, -1.131, 1), (scenarios.LOW_A0, -1.627, 1), (scenarios.NEAR_A0_MAX, -0.036, 2)],
)
def test_the_production_ahat_grid_under_resolves_the_bank_by_a_known_amount(
    scenario, expected_bias, expected_bins
):
    """What the *shipping* configuration actually does, pinned rather than described.

    The test above resolves the ``ahat`` axis deliberately (``ahat_max=0.1``) — but
    `Collision._table` builds the grid from the schema defaults, and nothing else here
    exercises those on the quantity ``ahat`` controls. At RES032's defaults
    (``ahat_max=0.5``, ``n_bins=32``, ``decades=1.0``) the first non-floor edge sits at
    0.035, so the whole scenario bank lands at or near the floor bin and the kernel uses
    that bin's own centre — 0.0174 — in place of population means of 0.0057, 0.00057 and
    0.028. The centroid it reports is biased low by the amounts below.

    This is **not a regression from RES053** and not something to fix by widening a tolerance:
    the bias is dominated by floor-bin coarseness and predates the cycle-average correction
    (``low_a0`` moved -1.59% -> -1.65% across it). It is pinned so that a future change to
    `_ahat_target_edges` or to the defaults has to move these numbers deliberately.
    RES053's last section records the measured alternative (``decades=0.3`` at the same
    ``n_bins``/``ahat_max``), which is the author's call rather than this test's.
    """
    samples = _samples(scenario)
    shape_table = deposit_shape_table(samples, n_bins=(32, 48, 48, 64), scheme="cic")
    table = retarget_ahat(shape_table, samples.intensity_peak)  # production defaults, as Collision does
    assert table.H.shape[3] == expected_bins

    # The floor bin's centre is a fixed property of the grid, the same for every scenario,
    # and it is what the kernel uses for the whole population below 0.035.
    assert table.ahat_centers[0] == pytest.approx(0.5 * _ahat_target_edges(0.0, 0.5, 32, 1.0)[1])

    edge = float(np.max(samples.gamma)) ** 2
    s_edges = np.linspace(0.0, 1.05 * edge, 150)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    kernel = angular_spectrum_from_table(table, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)

    def centroid(spectrum):
        return float(np.sum(s_centers * spectrum) / np.sum(spectrum))

    bias = 100.0 * (centroid(kernel) / centroid(reference) - 1.0)
    assert bias == pytest.approx(expected_bias, abs=0.15)


def test_spectrum_shifts_with_ahat_not_merely_rescales():
    """Regression for the predecessor's fixed bug: g/prefac must be recomputed inside the
    ahat loop. An ahat-independent shortcut would rescale the spectrum's amplitude but
    never move where its edge falls; the nonlinear redshift must move the edge.

    a0_shape=0.05 and 4.0 at intensity_peak=0.045 give ahat = 0.00225 and 0.18 — bins 0 and 6 of
    the production target grid (RES053), still well apart.
    """
    low = _synthetic_samples(a0_shape=0.05)
    high = _synthetic_samples(a0_shape=4.0, seed=0)
    table_low = _table(low, shape_bins=(24, 24, 24, 8))
    table_high = _table(high, shape_bins=(24, 24, 24, 8))

    edge = float(np.max(low.gamma) ** 2)
    s = np.linspace(0.0, 1.2 * edge, 200)
    spec_low = spectrum_from_table(table_low, 0.0, 0.0, s)
    spec_high = spectrum_from_table(table_high, 0.0, 0.0, s)

    def edge_index(spec):
        nonzero = np.nonzero(spec > 1e-6 * spec.max())[0]
        return nonzero[-1]

    # Higher ahat redshifts the resonance (s_res = gamma^2 / (1 + ahat + ...)), so the
    # high-ahat table's populated s range ends at a strictly lower edge than the low-ahat
    # one's — not just a smaller peak at the same edge.
    assert s[edge_index(spec_high)] < s[edge_index(spec_low)]


def test_angular_spectrum_matches_a_manual_grid_of_spectrum_from_table():
    samples = _synthetic_samples(n=5_000)
    table = _table(samples, shape_bins=(16, 16, 16, 16))
    s = np.linspace(1.0, samples.gamma.max() ** 2, 5)
    tx, ty = [-1e-3, 0.0], [0.0, 1e-3]
    cube = angular_spectrum_from_table(table, tx, ty, s)
    for i, x in enumerate(tx):
        for j, y in enumerate(ty):
            assert np.allclose(cube[i, j, :], spectrum_from_table(table, x, y, s))


def test_spectrum_in_angular_range_photon_count_matches_the_cube_integral():
    samples = _synthetic_samples(n=5_000)
    table = _table(samples, shape_bins=(16, 16, 16, 16))
    s_edges = np.linspace(0.0, 1.05 * samples.gamma.max() ** 2, 20)
    cube, dN_ds, n_photons = spectrum_in_angular_range(
        table, (-2e-3, 2e-3), (-2e-3, 2e-3), s_edges, resolution=(9, 9)
    )
    assert cube.shape == (9, 9, 19)
    assert dN_ds.shape == (19,)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    assert n_photons == pytest.approx(float(np.trapezoid(dN_ds, s_centers)), rel=1e-12)
    assert n_photons >= 0.0


# ---------------------------------------------------------------------------
# Stage 2 vs delta: an identity gate insensitive to §9.1 (both carry the same factor, so
# their ratio was ~1 before RES033 set it and is ~1 after). The absolute normalization the
# ratio cannot see gets its own test at the bottom of this section.
# ---------------------------------------------------------------------------
def test_stage2_kernel_agrees_with_delta_at_a_point():
    """CIC, not nearest: evaluating exactly at the beam's own angular centre — the natural
    point to check — lands exactly on a cell boundary of a *nearest*-deposited table, and
    a narrow beam spans few enough theta cells that this aliases into ratios anywhere from
    0.5 to 1.7 depending on resolution alone (measured while writing this test, not a
    hypothetical). That is consistent with the predecessor's own audit, which reports its
    three independent spectrum methods agreeing "within ~15%" generally. CIC deposition
    removes the aliasing (measured stable to +-2% from 40 to 250 theta bins) because it
    never lets a single cell speak for the beam centre alone.
    """
    samples = _synthetic_samples(n=200_000, seed=2)  # a0_shape=1.0, intensity_peak=0.045 -> ahat=0.045
    table = _table(samples, shape_bins=(48, 64, 64, 32), scheme="cic")

    edge = float(np.max(samples.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 200)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    kernel = angular_spectrum_from_table(table, [0.0], [0.0], s_centers)[0, 0, :]
    reference = resonance_spectrum(samples, s_edges, 0.0, 0.0)

    ratio = float(np.sum(kernel)) / float(np.sum(reference))
    assert ratio == pytest.approx(1.0, abs=0.1)


def test_the_table_kernel_angle_integrates_to_stage_0_total(baseline):
    """§9.1's closure, on the one quantity that can actually see it (RES033).

    Every other Stage-2 check in this file is a *ratio* between two paths that carry the
    same normalization constant, so all of them stayed green through a factor of ``2 pi``
    and would stay green through any other. This one is absolute: integrate the table
    kernel over solid angle and over ``s``, and compare with Stage 0's elementary
    ``flux x cross-section x time`` photon count. It is the same arbitration
    `delta.check_normalization` performs for delta, applied to the kernel that actually
    ships. RES033 has the full derivation-then-measurement sequence that justified the fix.

    **The tolerance is resolution, not doubt.** Both integrals are midpoint sums over grids
    sized for a ten-second test: the angular one samples a ``1/gamma``-wide cone, and the
    ``s`` one a spectrum narrower still. Refining either walks the ratio straight toward 1
    (RES033 has the convergence sequence) — ±15% covers the grid this test can afford, and is
    nowhere near wide enough to blur the only distinction it exists to make, which is
    between 1 and 6.28.
    """
    n_angles, cone = 17, 4.0
    table = _table(baseline, shape_bins=(32, 24, 24, 64), scheme="cic")

    edge = float(np.max(baseline.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, 241)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    widths = np.diff(s_edges)

    half = cone / float(np.mean(baseline.gamma))
    step = 2.0 * half / n_angles
    offsets = -half + step * (np.arange(n_angles) + 0.5)
    grid_x = float(np.mean(baseline.theta_x)) + offsets
    grid_y = float(np.mean(baseline.theta_y)) + offsets

    cube = angular_spectrum_from_table(table, grid_x, grid_y, s_centers)
    photons = float(np.sum(cube * widths[None, None, :])) * step * step
    # What the finite cone could not see — the same closed-form correction delta's own
    # arbitration divides by, and for the same reason.
    ratio = photons / baseline.total_yield() / captured_fraction(cone)

    assert ratio == pytest.approx(1.0, rel=0.15)
    assert ratio < 2.0  # i.e. nowhere near the 2*pi this used to be

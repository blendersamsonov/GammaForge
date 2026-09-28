"""Persistent Stage-2 query regressions from the chirp/ponderomotive handoff."""

from __future__ import annotations

import numpy as np
import pytest

from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.engines.xigma.stages import Table, prepare_query


pytestmark = [pytest.mark.tier1, pytest.mark.fast]


def _table() -> Table:
    gamma_edges = np.linspace(80.0, 120.0, 5)
    theta_x_edges = np.linspace(-0.03, 0.03, 7)
    theta_y_edges = np.linspace(-0.03, 0.03, 7)
    ahat_edges = np.array([0.0, 0.02, 0.08])
    chirp_edges = np.array([0.9, 1.0, 1.1])
    gamma = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    theta_x = 0.5 * (theta_x_edges[:-1] + theta_x_edges[1:])
    theta_y = 0.5 * (theta_y_edges[:-1] + theta_y_edges[1:])
    ahat = 0.5 * (ahat_edges[:-1] + ahat_edges[1:])
    chirp = 0.5 * (chirp_edges[:-1] + chirp_edges[1:])
    G, X, Y, A, C = np.meshgrid(
        gamma, theta_x, theta_y, ahat, chirp, indexing="ij"
    )
    H = 2.0 + 0.005 * (G - 80.0) + X - 0.5 * Y + A + 0.2 * C
    return Table(
        gamma_edges,
        theta_x_edges,
        theta_y_edges,
        ahat_edges,
        chirp_edges,
        H,
        0.02 * H,
        0.03 * H,
        0.01 * H,
        float(np.sum(H)),
        "prepared-query",
    )


def test_incremental_query_computes_only_missing_points_and_is_order_independent():
    table = _table()
    full_s = np.array([5_000.0, 7_000.0, 9_000.0, 11_000.0, 13_000.0])
    prepared = prepare_query(table, [0.0], [0.0], backend="numpy")
    calls = []
    original = prepared._evaluate_missing

    def tracked(points):
        calls.append(points.copy())
        return original(points)

    prepared._evaluate_missing = tracked
    coarse = prepared.evaluate_raw(full_s[::2])
    refined = prepared.evaluate_raw(full_s)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], full_s[::2])
    np.testing.assert_array_equal(calls[1], full_s[1::2])
    for old, current in zip(coarse, refined):
        np.testing.assert_array_equal(current[..., ::2], old)
    np.testing.assert_array_equal(prepared.evaluated_s, full_s)

    one_shot = prepare_query(table, [0.0], [0.0], backend="numpy")
    expected_raw = one_shot.evaluate_raw(full_s)
    for actual, expected in zip(refined, expected_raw):
        np.testing.assert_array_equal(actual, expected)

    different_order = prepare_query(table, [0.0], [0.0], backend="numpy")
    different_order.evaluate_raw(full_s[[4, 0]])
    different_order.evaluate_raw(full_s[[2]])
    expected = prepared.evaluate(full_s, line_model="moment2")
    actual = different_order.evaluate(full_s, line_model="moment2")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not is_gpu_available(), reason="requires actual CUDA")
def test_cuda_prepared_query_reuses_device_state_and_qmc_sequence():
    table = _table()
    full_s = np.array([5_000.0, 7_000.0, 9_000.0, 11_000.0, 13_000.0])
    prepared = prepare_query(
        table, [0.0], [0.0], backend="cupy", rings=32, subsampling=32
    )
    state = prepared._backend_state
    qmc_ids = (
        id(state._qmc_regular_numerator), id(state._qmc_radial_fraction)
    )
    channel_ids = tuple(id(values) for values in state._static_kernel_args[:4])
    calls = []
    original = state.evaluate

    def tracked(points):
        calls.append(np.asarray(points).copy())
        return original(points)

    state.evaluate = tracked
    coarse = prepared.evaluate_raw(full_s[::2])
    refined = prepared.evaluate_raw(full_s)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], full_s[::2])
    np.testing.assert_array_equal(calls[1], full_s[1::2])
    assert (
        id(state._qmc_regular_numerator), id(state._qmc_radial_fraction)
    ) == qmc_ids
    assert tuple(id(values) for values in state._static_kernel_args[:4]) == channel_ids
    for old, current in zip(coarse, refined):
        np.testing.assert_array_equal(current[..., ::2], old)

    one_shot = prepare_query(
        table, [0.0], [0.0], backend="cupy", rings=32, subsampling=32
    ).evaluate_raw(full_s)
    for actual, expected in zip(refined, one_shot):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)


def test_collision_reuses_prepared_context_for_matching_query(monkeypatch):
    table = _table()
    collision = Collision(
        interaction=object(),
        params=XigmaEngine.schema.with_values(backend="numpy", line_model="delta"),
    )
    monkeypatch.setattr(Collision, "_table", lambda self: table)
    first = collision.prepare_query([0.0], [0.0], backend="numpy")
    second = collision.prepare_query([0.0], [0.0], backend="numpy")
    assert second is first

    collision.angular_spectrum(
        [5_000.0, 9_000.0], [0.0], [0.0], backend="numpy", line_model="delta"
    )
    collision.angular_spectrum(
        [5_000.0, 7_000.0, 9_000.0], [0.0], [0.0],
        backend="numpy", line_model="delta",
    )
    np.testing.assert_array_equal(first.evaluated_s, [5_000.0, 7_000.0, 9_000.0])

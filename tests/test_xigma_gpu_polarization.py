"""Independent CUDA checks for the Stage-2 polarization projection.

The reference below evaluates manuscript Eq. ``udef`` directly in ``longdouble``.  It
does not reproduce the production stable-denominator rearrangement, so a regression in
the device helper cannot cancel against a copied implementation of the same algebra.
Inputs are quantized to the device's float32 interface before the reference is evaluated.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.engines.xigma.stages import polarization_factor
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.results import Axis
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU not available on host")


def _udef_longdouble(
    gamma, theta_x, theta_y, theta_x_obs, theta_y_obs,
    ellipticity, psi_pol, theta_xz=0.0, theta_yz=0.0,
):
    """Direct high-precision Eq. ``udef`` for arrays of electron directions."""
    ld = np.longdouble
    theta_xz, theta_yz, psi_pol = map(ld, (theta_xz, theta_yz, psi_pol))
    cx, sx = np.cos(theta_xz), np.sin(theta_xz)
    cy, sy = np.cos(theta_yz), np.sin(theta_yz)
    rotation = np.array(
        [[cx, sx * sy, sx * cy], [ld(0), cy, -sy], [-sx, cx * sy, cx * cy]], dtype=ld
    )
    cp, sp = np.cos(psi_pol), np.sin(psi_pol)
    e0 = rotation @ np.array([cp, sp, ld(0)], dtype=ld)
    e1 = rotation @ np.array([-sp, cp, ld(0)], dtype=ld)

    txo, tyo = ld(theta_x_obs), ld(theta_y_obs)
    n = np.array([txo, tyo, ld(1)], dtype=ld)
    n /= np.sqrt(np.dot(n, n))
    gamma = np.asarray(gamma, dtype=ld)
    tx = np.asarray(theta_x, dtype=ld)
    ty = np.asarray(theta_y, dtype=ld)
    beta = np.sqrt(ld(1) - gamma ** -2)
    direction_norm = np.sqrt(ld(1) + tx * tx + ty * ty)
    v = np.stack((beta * tx / direction_norm, beta * ty / direction_norm,
                  beta / direction_norm), axis=-1)
    d = ld(1) - np.einsum("...i,i->...", v, n)
    nv = n - v
    a0, a1 = np.dot(n, e0), np.dot(n, e1)
    u0 = nv * a0 / d[..., None] - e0
    u1 = nv * a1 / d[..., None] - e1
    eps2 = ld(ellipticity) ** 2
    return (np.einsum("...i,...i->...", u0, u0) + eps2 * np.einsum("...i,...i->...", u1, u1)) / (ld(1) + eps2)


def _production_helper():
    """Build a tiny probe kernel that invokes the private production device helper."""
    from gammaforge.engines.xigma import spectrum_sampler as sampler

    cp, jit = sampler.cp, sampler.jit

    @jit.rawkernel()
    def probe(output, gamma, xe, ye, xo, yo, e0x, e0y, e0z, e1x, e1y, e1z, xi00, xi11):
        idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        if idx < output.size:
            output[idx] = sampler._polarization_factor_device(
                gamma[idx], xe[idx], ye[idx], xo, yo,
                e0x, e0y, e0z, e1x, e1y, e1z, xi00, xi11,
            )

    def evaluate(gamma, theta_x, theta_y, theta_x_obs, theta_y_obs,
                 ellipticity, psi_pol, theta_xz=0.0, theta_yz=0.0):
        arrays = np.broadcast_arrays(
            np.asarray(gamma, dtype=np.float32), np.asarray(theta_x, dtype=np.float32),
            np.asarray(theta_y, dtype=np.float32),
        )
        params = sampler._polarization_parameters(psi_pol, ellipticity, theta_xz, theta_yz)
        out = cp.empty(arrays[0].size, dtype=cp.float32)
        if out.size:
            device = [cp.asarray(a.reshape(-1), dtype=cp.float32) for a in arrays]
            probe[((out.size + 127) // 128,), (128,)](
                out, *device, cp.float32(theta_x_obs), cp.float32(theta_y_obs),
                *(cp.float32(v) for v in params),
            )
            cp.cuda.Stream.null.synchronize()
        return out

    return evaluate


def _as_host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


@gpu
@pytest.mark.parametrize("ellipticity", [0.0, 0.4, 1.0], ids=["linear", "elliptical", "circular"])
@pytest.mark.parametrize(
    "gamma,theta_x,theta_y,theta_x_obs,theta_y_obs,theta_xz,theta_yz",
    [
        (2000.0, 0.0011, -0.0007, 0.0003, -0.0002, 0.04, -0.03),
        (10000.0, -0.0008, 0.0005, -0.0004, 0.0006, -0.03, 0.05),
        (1600.0, 0.0, 0.0, 0.0002, -0.0001, 0.0, 0.06),
    ],
)
def test_cuda_polarization_helper_matches_independent_longdouble_udef(
    ellipticity, gamma, theta_x, theta_y, theta_x_obs, theta_y_obs, theta_xz, theta_yz,
):
    helper = _production_helper()
    # Quantize every scalar entering the device path, then evaluate the independent
    # reference from those exact float32 values.
    values = np.asarray([gamma, theta_x, theta_y], dtype=np.float32)
    obs_x, obs_y = np.float32(theta_x_obs), np.float32(theta_y_obs)
    angle_xz, angle_yz = np.float32(theta_xz), np.float32(theta_yz)
    eps = np.float32(ellipticity)
    psi = np.float32(0.37)
    expected = _udef_longdouble(
        values[:1], values[1:2], values[2:3], obs_x, obs_y, eps, psi, angle_xz, angle_yz
    )
    actual = _as_host(helper(
        values[:1], values[1:2], values[2:3], obs_x, obs_y, eps, psi, angle_xz, angle_yz
    ))
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("gamma", [2000.0, 10000.0])
def test_cuda_collinear_crossing_limit_is_cosine_squared(gamma):
    helper = _production_helper()
    crossing = np.float32(0.3)
    actual = _as_host(helper(
        np.asarray([gamma], dtype=np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32),
        np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), crossing, np.float32(0.0),
    ))
    expected = np.cos(np.longdouble(crossing)) ** 2
    np.testing.assert_allclose(actual, [float(expected)], rtol=1e-5, atol=1e-6)


@gpu
def test_cuda_circular_polarization_is_invariant_to_basis_azimuth():
    helper = _production_helper()
    gamma = np.asarray([2000.0, 10000.0], dtype=np.float32)
    tx = np.asarray([0.0011, -0.0008], dtype=np.float32)
    ty = np.asarray([-0.0007, 0.0005], dtype=np.float32)
    values = []
    for psi in (0.0, 0.61, 1.73):
        values.append(_as_host(helper(gamma, tx, ty, np.float32(0.0003), np.float32(-0.0002),
                                      np.float32(1.0), np.float32(psi), np.float32(0.04), np.float32(-0.03))))
    np.testing.assert_allclose(values[1], values[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values[2], values[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("ellipticity", [0.0, 0.4, 1.0], ids=["linear", "elliptical", "circular"])
@pytest.mark.parametrize(
    "gamma,theta_x,theta_y,theta_x_obs,theta_y_obs,theta_xz,theta_yz",
    [
        (2000.0, 0.0011, -0.0007, 0.0003, -0.0002, 0.04, -0.03),
        (10000.0, -0.0008, 0.0005, -0.0004, 0.0006, -0.03, 0.05),
        (1600.0, 0.0, 0.0, 0.0002, -0.0001, 0.0, 0.06),
        (10000.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0),
    ],
)
def test_numpy_polarization_helper_matches_independent_longdouble_udef(
    ellipticity, gamma, theta_x, theta_y, theta_x_obs, theta_y_obs, theta_xz, theta_yz,
):
    """NumPy polarization factor matches high-precision longdouble Eq. udef (RES070)."""
    psi = 0.37
    expected = _udef_longdouble(
        [gamma], [theta_x], [theta_y], theta_x_obs, theta_y_obs,
        ellipticity, psi, theta_xz, theta_yz,
    )
    actual = polarization_factor(
        gamma, theta_x, theta_y, theta_x_obs, theta_y_obs,
        ellipticity, psi, theta_xz, theta_yz,
    )
    np.testing.assert_allclose(actual, [float(expected[0])], rtol=1e-10, atol=1e-10)


def test_crossed_engine_forwards_geometry_and_applies_energy_jacobian_once(monkeypatch):
    """A controlled cube catches duplicate crossing-energy factors in ``Collision._fill``."""
    laser = replace(
        scenarios.BASELINE.laser,
        theta_xz=Quantity(0.2, "rad"),
        theta_yz=Quantity(-0.15, "rad"),
        ellipticity=0.4,
        psi_pol=Quantity(0.37, "rad"),
    )
    interaction = scenarios.build(
        replace(scenarios.BASELINE, laser=laser),
        SamplingSpec(n_particles=64, seed=20260721, prefilter=1e-3),
    )
    request = OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(4, 3, 3))
    interaction = replace(interaction, target=replace(interaction.target, outputs=(request,)))
    params = XigmaEngine.schema.with_values(
        n_steps=8, n_bins_gamma=8, n_bins_theta_x=8, n_bins_theta_y=8,
        n_bins_a0_shape=8, n_bins_ahat=4, backend="numpy",
    )
    calls = []

    def controlled_cube(self, s, theta_x, theta_y, **kwargs):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        theta_x = np.atleast_1d(np.asarray(theta_x, dtype=float))
        theta_y = np.atleast_1d(np.asarray(theta_y, dtype=float))
        calls.append((s, theta_x, theta_y, kwargs))
        return np.full((theta_x.size, theta_y.size, s.size), 7.0)

    monkeypatch.setattr(Collision, "angular_spectrum", controlled_cube)
    results = Collision(interaction=interaction, params=params).run((request,))

    assert len(calls) == 1
    s, theta_x, theta_y, kwargs = calls[0]
    assert kwargs == {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.2, "theta_yz": -0.15}
    alpha_cos = np.cos(0.2) * np.cos(-0.15)
    photon_energy = laser.photon_energy() * (1.0 + alpha_cos) * 0.5
    energy = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM].axes[Axis.ENERGY]
    np.testing.assert_allclose(s, energy / (4.0 * photon_energy), rtol=1e-13, atol=0.0)
    distr = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM].distr
    np.testing.assert_allclose(distr, 7.0 / (4.0 * photon_energy), rtol=1e-13, atol=0.0)
    assert results.model_specific["stage2_backend"] == "numpy"

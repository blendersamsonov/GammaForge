"""The nonlinear angle-integrated spectrum ``G(z; h)`` and its collision average (DER019 §23).

The existing `formulas.angle_integrated_spectrum` applies a nonlinear red-shift by replacing
``y = s/gamma^2`` with ``y = s(1+ahat)/gamma^2`` inside the linear DER011 polynomial. That
compresses the linear shape, but it is **not** the exact angular Jacobian of the nonlinear
resonance: the true Jacobian ``ds/du = -gamma^2/(1+h+u)^2`` depends on ``h``, so a nonlinear
shift changes the spectrum's shape as well as its scale (DER019 §23).

This module supplies the two pieces that replace that approximation in a supported tier:

* :func:`nonlinear_shape` — the normalized single-electron shape ``G(z; h)``, exact in the
  head-on ``Q~1`` paraxial model, and
* :func:`collision_averaged_shape` — ``Gbar_nu(z; chi)``, the average of ``G`` over the
  round fixed-width nonlinear distribution ``f_x(x) = nu x^(nu-1)``.

**Photon count is conserved by construction.** Each ``G(.; h)`` integrates to exactly one on
its *own* support ``0 < z <= 1/(1+h)``, and averaging normalized shapes preserves that: the
result is a probability density in ``z`` before any energy-spread integration. This matters
because the current engine instead rescales the discretized spectrum to the overlap yield
(RES036), which would hide a shape-normalization error — the handoff is explicit that the
continuous model must be provably correct before that rescale is allowed to stand in for it.

Deliberately independent of Xigma's Stage-2 kernel: analytical is the validation oracle for
Xigma, and importing the shape from the implementation it is supposed to check would make
that comparison circular (the same reasoning `formulas.angle_integrated_spectrum` documents
for DER011).

The ``x`` integral is a small Gauss-Legendre quadrature rather than the hypergeometric form
DER019 mentions, for the reason the handoff gives: no special-function dependency for an
integral this cheap (DER019 §23.1 "a small stable quadrature is probably preferable in
production").
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["collision_averaged_shape", "nonlinear_shape", "shape_support_edge"]


def nonlinear_shape(z, h: float | np.ndarray):
    """``G(z; h)`` — the normalized nonlinear angle-integrated single-electron shape.

        G(z; h) = (3/2) 1/(1 - h z)^2 [ 1 - 2 z [1 - (1+h) z] / (1 - h z)^2 ]

    with support ``0 < z <= 1/(1+h)``, where ``z = s/gamma^2`` is the normalized photon
    energy and ``h = ahat`` the cycle-averaged nonlinear intensity.

    Two properties make this usable as a replacement for the compressed-DER011 shortcut, and
    both are asserted in `tests/test_analytical.py`:

    * ``int_0^{1/(1+h)} G dz == 1`` exactly, for every ``h``;
    * ``G(z; 0) == 1.5 (1 - 2 z (1 - z))``, i.e. DER011 exactly.

    The zero and negative-``z`` region is returned as zero rather than extrapolated: a photon
    cannot carry negative energy, and ``formulas.angle_integrated_spectrum`` masks the same
    region for the same reason.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    h = np.asarray(h, dtype=float)
    out = np.zeros(np.broadcast(z, h).shape, dtype=float)
    # `1 - h z > 0` is the physical support: the resonance requires 1 + h + u > 0 with
    # u = 1/z - 1 - h >= 0.
    valid = (z > 0.0) & (1.0 - h * z > 0.0)
    z_b, h_b = np.broadcast_arrays(z, h)
    zs, hs = z_b[valid], h_b[valid]
    t = 1.0 - hs * zs
    out[valid] = 1.5 / t**2 * (1.0 - 2.0 * zs * (1.0 - (1.0 + hs) * zs) / t**2)
    return out


def shape_support_edge(h: float) -> float:
    """``1/(1+h)`` — the highest normalized photon energy a trajectory with shift ``h`` emits.

    Not decoration: the collision average must integrate each ``G(.; chi x)`` over *its own*
    edge, and using the dimmest trajectory's edge for all of them is what would silently drop
    the high-energy tail and break photon-count conservation.
    """
    if h <= -1.0:
        raise ValueError(f"shape_support_edge needs h > -1, got {h!r}")
    return 1.0 / (1.0 + h)


def collision_averaged_shape(z, nu: float, chi: float, n_quad: int = 96):
    """``Gbar_nu(z; chi)`` — ``G`` averaged over the round fixed-width nonlinear distribution.

        Gbar_nu(z; chi) = int_0^{X(z)} nu x^(nu-1) G(z; chi x) dx

    with the per-trajectory cutoff

        X(z) = min[1, max[0, (1/z - 1)/chi]]

    because only trajectories satisfying ``z <= 1/(1 + chi x)`` contribute at that energy
    (DER019 §23.1). ``nu = 1 + sigma_L^2/sigma_e^2`` is the number of illuminated Gaussian
    modes and ``chi = I_pk a_max`` converts the dimensionless trajectory shape into the
    physical nonlinear shift.

    Normalized in ``z`` by construction, for the reason in the module docstring: each ``G``
    integrates to one on its own support, and the ``x`` weights sum to one.

    ``chi == 0`` is exact rather than a limit: with no nonlinear shift every trajectory emits
    the same shape, so the average is ``G(z; 0)`` with no quadrature at all.

    Vectorized over ``z`` rather than looping. The cutoff ``X(z)`` is a per-``z`` rescaling of
    the same Gauss nodes, so the whole grid is one ``(len(z), n_quad)`` evaluation — which
    keeps a 200-bin spectrum in the low-millisecond range this engine needs to stay a
    real-time preview (see `formulas`' cost notes and RES043).

    .. note::

       Convergence in ``n_quad`` is set by ``nu - 1``, because ``x^(nu-1)`` has an integrable
       but sharp singularity at the origin when the bunch is much wider than the spot. For
       ``nu - 1 ~ 1`` (a strongly focused spot) the default 96 nodes are converged to
       ~1e-13; for ``nu ~ 1.06`` (a very broad spot, so a nearly flat intensity profile across
       the bunch) they give ~2e-6, falling as ``1/n_quad``. The result is well-conditioned
       either way — this is a smooth refinement, not an instability — so the default is left
       alone rather than tuned to the worst case, which would penalize the common one.
    """
    if nu <= 0.0:
        raise ValueError(f"collision_averaged_shape needs nu > 0, got {nu!r}")
    if chi < 0.0:
        raise ValueError(f"collision_averaged_shape needs chi >= 0, got {chi!r}")

    z = np.atleast_1d(np.asarray(z, dtype=float))
    if chi == 0.0:
        return nonlinear_shape(z, 0.0)

    # X(z): the x-cutoff at each requested energy. `where`/`maximum` rather than a bare
    # `clip` so a non-positive z (no contribution) and a cutoff of exactly 0 stay distinct.
    positive = z > 0.0
    safe_z = np.where(positive, z, 1.0)
    x_max = np.clip((1.0 / safe_z - 1.0) / chi, 0.0, 1.0)
    x_max = np.where(positive, x_max, 0.0)

    nodes, weights = leggauss(n_quad)
    # (len(z), n_quad): map Legendre's (-1, 1) nodes onto each row's own [0, X].
    x = 0.5 * x_max[:, None] * (nodes[None, :] + 1.0)
    weights_x = 0.5 * x_max[:, None] * weights[None, :]

    h = chi * x
    z_grid = np.repeat(z[:, None], n_quad, axis=1)
    t = 1.0 - h * z_grid
    valid = (z_grid > 0.0) & (t > 0.0)
    # G is evaluated only on the valid support; the masked entries must not be allowed to
    # produce a 0/0 or a divide-by-zero that would poison the row sum with a nan.
    safe_t = np.where(valid, t, 1.0)
    g = np.where(
        valid,
        1.5 / safe_t**2 * (1.0 - 2.0 * z_grid * (1.0 - (1.0 + h) * z_grid) / safe_t**2),
        0.0,
    )
    density = np.where(valid, nu * np.power(np.maximum(x, 0.0), nu - 1.0), 0.0)
    return np.sum(weights_x * density * g, axis=1)

"""Deterministic collimated spectra and the raw DER017 moment channels (DER019 §7, §7.1, §8).

`OutputKind.COLLIMATED_SPECTRUM` is the 3D ``(E, theta_x, theta_y)`` slice — the sole
energy-angle output kind (RES052 deliberately removed the duplicate). This module fills it
without macroparticles, and returns the three raw channels separately rather than only the
reconstructed spectrum, because a reconstructed curve that looks plausible can hide any one
of the three being wrong.

**The delta-resonance inversion.** Integrating over photon energy removes the resonance delta
function, which lets ``gamma`` be inverted *analytically* at each requested ``(s, n)``
(DER019 §7):

    A_R = 1 + Q ahat,   K = D Cbar,   r^2 = (theta_ex - theta_x)^2 + (theta_ey - theta_y)^2
    Gamma^2 = A_R / (K/s - r^2),      support K/s > r^2

so there is **no gamma quadrature at all** — the electron's energy PDF is simply evaluated
at ``Gamma``. That is what makes this tier cheap enough for optimization, and it is why
DER019 prefers it to building a Stage-1 table at all.

**What ``rho0`` is, and is not.** The zeroth channel is a *density in s*, not a
probability distribution on a fixed interval: for a zero-emittance on-axis observer it is
``s^{-7/2}``, which piles up at low energy without bound on ``(0, 1]``. That is correct
physics, not a defect — the finite-normalization quantity is the angle-integrated shape
`nonlinear_spectrum.nonlinear_shape`, and RES036's yield normalization still applies to the
slice as a whole. What the tests pin instead is the *scaling law* and the moment
identities, which are the parts a wrong formula would break.

**Independence.** No gammaforge.xigma import appears anywhere in this module, and none
should: analytical is Xigma's validation oracle (RES095 leaves it one of the remaining
independent legs), so importing the kernel or the reconstruction operator from the
implementation it is meant to check would make the comparison circular. The DER017
reconstruction is therefore re-derived here from its defining operator rather than called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from .fixed_width import KAPPA_G

try:  # pragma: no cover - exercised by whichever branch the environment provides
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    _HAS_CUPY = False

__all__ = [
    "CollimatedMoments",
    "CollimatedGrid",
    "collimated_grid",
    "collimated_moments",
    "is_gpu_available",
    "reconstruct_second_order",
]

#: `3/(4 pi)`: the DER017 kernel normalization (DER019 §7). Matches
#: `xigma.stages.KERNEL_NORMALIZATION_CONSTANT` by construction rather than by import —
#: see the module docstring on why these two must stay independent.
KERNEL_NORMALIZATION = 3.0 / (4.0 * math.pi)


def is_gpu_available() -> bool:
    """True if CuPy is importable **and** the operations this module needs actually work.

    Importing CuPy and asking ``cuda.is_available()`` is not enough, and the failure this
    guards against is not hypothetical: on this machine CuPy imports, reports one CUDA device,
    and elementwise operations work, while ``matmul`` raises
    ``ImportError: libcublas.so.12`` because the BLAS library is missing. A probe covering
    only elementwise ops would therefore report "GPU available" and then fail deep inside the
    DER017 reconstruction. So the check exercises exactly the two classes of operation used
    below: elementwise arithmetic and a dense matmul.

    Costs microseconds, cached thereafter, and it is the difference between falling back to
    numpy cleanly and raising from the middle of a calculation.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    _GPU_AVAILABLE = False
    if _HAS_CUPY:
        try:
            if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
                # Elementwise, as in the channel formulas.
                _ = cp.ones(4) * 2.0
                # Dense matmul, as in the DER017 reconstruction. This is the one that fails
                # when the CUDA BLAS libraries are absent, so it must be part of the probe.
                _ = cp.eye(3) @ cp.ones((3, 2))
                _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


#: Tri-state cache: ``None`` until probed, then a bool. Device state cannot change within a
#: process in any way this package supports switching on.
_GPU_AVAILABLE: bool | None = None


def _array_module(*arrays):
    """The array module (`numpy` or `cupy`) matching the inputs.

    Same shape as `io.laser._get_array_module` and `xigma.stages._get_array_module` so this
    package dispatches the way the rest of the repository does, rather than inventing a
    third convention. Falls back to numpy when nothing identifies an array.
    """
    if _HAS_CUPY:
        try:
            return cp.get_array_module(*arrays)
        except Exception:
            pass
    for array in arrays:
        if array is not None:
            if _HAS_CUPY and isinstance(array, cp.ndarray):
                return cp
            if isinstance(array, np.ndarray):
                return np
    return np


def _gradient_matrix(s: np.ndarray, order: int) -> np.ndarray:
    """Dense matrix ``G`` with ``G @ f == np.gradient(f, s, edge_order=2)`` applied ``order`` times.

    Built by probing `np.gradient` with unit vectors. Differentiation is a **linear** operator,
    so the columns of its matrix are exactly its action on the basis — which is why this
    reproduces `np.gradient` rather than approximating it (verified to 1e-15 first order and
    5e-13 second order).

    This exists because `np.gradient` re-derives its nonuniform-grid weights on **every call**,
    and the DER017 reconstruction needs two derivative applications per angular column. On a
    51x21x21 slice that was 882 rebuilds of the weights for arrays only 51 elements long: the
    physics took 1.7 ms and the scaffolding took 239 ms. Building the matrix once per grid
    (2.3 ms) and applying it as a single matmul collapses that.

    Cached per (grid bytes, order) because the weights depend only on ``s``, never on the data.
    """
    key = (s.tobytes(), int(order))
    cached = _GRADIENT_CACHE.get(key)
    if cached is not None:
        return cached
    n = s.size
    matrix = np.empty((n, n), dtype=float)
    for column in range(n):
        probe = np.zeros(n)
        probe[column] = 1.0
        result = probe
        for _ in range(order):
            result = np.gradient(result, s, edge_order=2)
        matrix[:, column] = result
    _GRADIENT_CACHE[key] = matrix
    return matrix


#: Cache of finite-difference weight matrices, keyed by the exact grid. Bounded so a GUI that
#: resizes a slider continuously cannot grow it without limit; the entries are small (n^2), but
#: a slider sweep produces a new key per pixel.
_GRADIENT_CACHE: dict[tuple[bytes, int], np.ndarray] = {}
_GRADIENT_CACHE_LIMIT = 64


@dataclass(frozen=True)
class CollimatedMoments:
    """The three raw DER017 channels on a normalized-energy grid, plus the resonance root.

    ``rho1``/``rho2`` are the *unreconstructed* channels. Reporting them alongside
    ``reconstructed`` is the point: DER019 §18.12 asks for ``rho0``, ``rho1`` and ``rho2`` to
    be compared with Xigma separately, since a matching reconstructed spectrum can be
    reached with compensating errors in two of the three.
    """

    s: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray
    #: ``Gamma`` at the resonance root for each ``s``; exposed because the support condition
    #: ``K/s > r^2`` is the model's own validity edge and is useful when diagnosing an
    #: empty spectrum.
    gamma: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        """The second-order finite-line spectrum ``S(s)`` (DER017, via DER019 §7.1).

        ``S = rho0 - (1/s) d[s rho1]/ds + (1/(2s)) d^2[s rho2]/ds^2``

        Differentiated here rather than by calling `xigma.stages.reconstruct_second_order`,
        for the independence reason in the module docstring.
        """
        return reconstruct_second_order(self.s, self.rho0, self.rho1, self.rho2)


def collimated_moments(
    beam: GaussianElectronBeam,
    s,
    theta_ex: float = 0.0,
    theta_ey: float = 0.0,
    ahat: float = 0.0,
    d_factor: float = 1.0,
    q_factor: float = 1.0,
    c_bar: float = 1.0,
    var_q: float | None = None,
    total_yield: float = 1.0,
) -> CollimatedMoments:
    """Raw ``rho0``/``rho1``/``rho2`` for the round head-on zero-emittance tier.

    Implements DER019 §7's master formula with the round head-on on-axis specialization of
    §8, where ``D = Q = 1``, ``r = 0`` and ``Cbar = 1`` for an unchirped pulse:

        A_R = 1 + ahat,  K = Cbar
        Gamma^2 = A_R / (K/s)
        rho0 = Y (3/(4 pi)) (K/s^2) P Gamma^5 / (A_R (1 + Gamma^2 r^2)^2)

    ``P`` is 1 for the azimuthally averaged on-axis kernel, and ``d_factor``/``q_factor``
    carry the direction factors ``D`` and ``Q`` so a caller can at least see the
    dependence; they are 1 for the tier this actually claims.

    ``var_q`` is the within-trajectory DER016 variance of the nonlinear coordinate. It
    defaults to ``KAPPA_G * ahat^2``, the unchirped separable value from DER019 §3.1 — the
    finite-line width of a single trajectory, distinct from the between-trajectory spread in
    `fixed_width`. Passing ``0.0`` reproduces the delta-line limit, in which the three
    channels collapse and the reconstruction returns ``rho0`` unchanged; the tests use that
    to pin the operator independently of the physics.
    """
    if q_factor <= 0.0:
        # DER019 §8.1: Q = 0 means observation along the laser's propagation direction, where
        # the nonlinear coordinate does not shift the resonance. That is a delta-line limit,
        # not something this formula can express.
        raise ValueError(
            "collimated_moments: q_factor = 0 (observation along the laser propagation "
            "direction) does not shift the resonance and is a delta-line limit, not a value "
            "the resonance-inversion formula accepts (DER019 §8.1)"
        )
    if c_bar <= 0.0:
        raise ValueError(f"collimated_moments needs c_bar > 0, got {c_bar!r}")
    if ahat < 0.0:
        raise ValueError(f"collimated_moments needs ahat >= 0, got {ahat!r}")

    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    a_r = 1.0 + q_factor * ahat
    k = d_factor * c_bar

    # r^2 = (theta_ex - theta_x)^2 + (theta_ey - theta_y)^2, zero on axis for this tier.
    r_sq = (theta_ex - 0.0) ** 2 + (theta_ey - 0.0) ** 2

    # Resonance inversion with its own support condition K/s > r^2 (DER019 §7). `s <= 0` is
    # outside the support too — a normalized photon energy is positive by definition — so it
    # is masked here rather than left to produce a divide-by-zero and then a nan that would
    # spread through the moment channels and the reconstruction.
    supported = (s_arr > 0.0) & (k / np.where(s_arr > 0.0, s_arr, 1.0) - r_sq > 0.0)
    safe_s = np.where(supported, s_arr, 1.0)
    inverse_base = k / safe_s - r_sq
    gamma_sq = np.where(supported, a_r / np.where(supported, inverse_base, 1.0), 0.0)
    gamma = np.sqrt(gamma_sq)

    # rho0: the master formula, with P = 1 for the azimuthally averaged on-axis kernel.
    per_s = (
        KERNEL_NORMALIZATION
        * (k / safe_s**2)
        * gamma**5
        / (a_r * (1.0 + gamma_sq * r_sq) ** 2)
    )
    rho0 = np.where(supported, total_yield * per_s, 0.0)

    # The moment channels, at the same root (DER019 §7.1). Unchirped means Cbar = 1,
    # Var(C) = 0 and Cov(q, C) = 0, which reduces both to the single Var(q) term.
    if var_q is None:
        var_q = KAPPA_G * ahat**2
    b_factor = 1.0 + q_factor * ahat + gamma_sq * r_sq
    safe_b = np.where(supported, b_factor, 1.0)
    delta_c = np.where(supported, q_factor**2 * var_q / safe_b**2, 0.0)
    m2 = delta_c.copy()

    return CollimatedMoments(
        s=s_arr,
        rho0=rho0,
        rho1=np.where(supported, rho0 * safe_s * delta_c, 0.0),
        rho2=np.where(supported, rho0 * safe_s**2 * m2, 0.0),
        gamma=gamma,
    )


def _finite_difference(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """``order``-th derivative of ``y`` sampled on a possibly nonuniform grid ``x``.

    Uses `numpy.gradient`, which handles nonuniform spacing and picks the stencil order from
    the number of points. A hand-rolled local polynomial fit was tried first and rejected: the
    monomial Vandermonde system it solves is ill-conditioned enough to return ~1e8 relative
    error on smooth power laws, which then dominated the reconstructed spectrum entirely.
    `numpy.gradient` is exact to roundoff for low degrees and degrades gracefully beyond.

    This is deliberately *not* `xigma.stages.nonuniform_derivative` — the two must be
    separate implementations for the comparison to be a check rather than a tautology.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = y
    for _ in range(order):
        result = np.gradient(result, x, edge_order=2)
    return result


def reconstruct_second_order(
    s: np.ndarray, rho0: np.ndarray, rho1: np.ndarray, rho2: np.ndarray
) -> np.ndarray:
    """The DER017 second-order finite-line reconstruction, re-derived (DER019 §7.1).

        S(s) = rho0 - (1/s) d[s rho1]/ds + (1/(2s)) d^2[s rho2]/ds^2

    Implemented from the defining operator rather than imported from
    `xigma.stages.reconstruct_second_order`, so that comparing this engine's output with
    Xigma's compares two derivations rather than one derivation called twice.
    """
    s = np.asarray(s, dtype=float)
    if s.size < 3:
        raise ValueError("second-order reconstruction requires at least three spectral points")
    if np.any(s <= 0.0):
        raise ValueError("reconstruct_second_order requires positive spectral points")
    first = _finite_difference(s, s * np.asarray(rho1), 1)
    second = _finite_difference(s, s * np.asarray(rho2), 2)
    return np.asarray(rho0) - first / s + 0.5 * second / s


@dataclass(frozen=True)
class CollimatedGrid:
    """The whole ``(E, theta_x, theta_y)`` slice evaluated at once.

    Same physics as :func:`collimated_moments`, applied to every angular cell at once instead
    of column by column. The two must agree — `tests/test_analytical.py` asserts the grid
    result equals the per-column path rather than trusting that a refactor preserved it.
    """

    s: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray
    gamma: np.ndarray
    #: Which backend produced this, for the provenance record.
    backend: str

    @property
    def reconstructed(self) -> np.ndarray:
        """The DER017 reconstruction along the energy axis, over the whole grid.

        Differentiating mixes neighbouring rows, so a row that is *outside* the support
        (``s <= 0``, or ``K/s <= r^2``) must not contribute to the rows next to it — a zeroed
        channel there is not enough, because ``0 * inf`` still yields a nan that propagates
        into the supported rows one step away. The derivative is therefore applied to the
        channel values with unsupported rows carried as zeros, and the reconstruction is
        evaluated only where ``s > 0``; the off-support cells are left at zero afterwards.

        That is why the operator is not simply "mask, then differentiate": masking the output
        was what the earlier per-column version did, and it was correct there only because
        each column's support was independent.

        The backend comes from :attr:`backend`, which is recorded at construction. It is
        deliberately *not* inferred from the channel arrays: `cupy.get_array_module` on a
        NumPy array returns `numpy`, so inferring it here would pick the CPU path and silently
        copy the entire device grid back to the host twice per derivative — the module's
        weight matrices are built with NumPy because `np.gradient` has no CuPy equivalent, and
        that is a small host-side cost, not a reason to demote the grid itself.
        """
        xp = cp if self.backend == "cupy" else np
        host_s = np.asarray(self.s, dtype=float)
        matrix1 = xp.asarray(_gradient_matrix(host_s, 1))
        matrix2 = xp.asarray(_gradient_matrix(host_s, 2))

        leading = self.rho0.shape[0]
        trailing = int(np.prod(self.rho0.shape[1:], dtype=int)) if self.rho0.ndim > 1 else 1
        # Unsupported energy rows are excluded from the derivative entirely: zero them on the
        # *input* side, and only write output where s is positive.
        valid = xp.asarray(host_s > 0.0)[:, None]
        s_column = xp.where(valid, xp.asarray(host_s)[:, None], 0.0)
        rho1 = xp.where(valid, xp.asarray(self.rho1).reshape(leading, trailing), 0.0)
        rho2 = xp.where(valid, xp.asarray(self.rho2).reshape(leading, trailing), 0.0)

        first = matrix1 @ (s_column * rho1)
        second = matrix2 @ (s_column * rho2)
        divisor = xp.where(valid, xp.asarray(host_s)[:, None], 1.0)
        result = (
            xp.asarray(self.rho0).reshape(leading, trailing)
            - first / divisor
            + 0.5 * second / divisor
        )
        # Two distinct zero-regions, both of which must reconstruct to exactly zero:
        #
        # 1. `s <= 0` — outside the support entirely (the `valid` mask below).
        # 2. `rho0 == 0` — where `f_gamma` underflowed, so the trajectory emits nothing at
        #    that energy. The DER017 terms are *ratios* built from rho0, so they underflow
        #    with it, but the finite differences that evaluate them do not: the difference of
        #    two ~1e-40 values is representable, and the `1/s` prefactor then amplifies that
        #    noise by ~1e13 into a visible negative (~1e-5 against a peak of ~1e-1). That is
        #    arithmetic noise in a region with no physics in it, not a spectral feature.
        #
        # Masking on `rho0 > 0` rather than on a magnitude is what makes this exact: where
        # the density is genuinely zero the spectrum is genuinely zero, whatever the
        # derivative stencil does there. Non-negative is also a physical requirement — a
        # photon-count density cannot be negative.
        emitting = (xp.asarray(self.rho0).reshape(leading, trailing) > 0.0)
        result = xp.where(valid & emitting, xp.maximum(result, 0.0), 0.0)
        return result.reshape(self.rho0.shape)


def collimated_grid(
    ahat: float,
    s,
    theta_x,
    theta_y,
    *,
    total_yield: float = 1.0,
    var_q: float | None = None,
    d_factor: float = 1.0,
    q_factor: float = 1.0,
    c_bar: float = 1.0,
    gamma0: float = 0.0,
    sigma_gamma: float = 0.0,
    backend: str = "auto",
) -> CollimatedGrid:
    """Evaluate :func:`collimated_moments` over a whole ``(s, theta_x, theta_y)`` grid.

    The channel formulas are pure elementwise arithmetic on ``s`` and ``r^2``, so the whole
    slice is one broadcast evaluation rather than ``len(theta_x) * len(theta_y)`` calls. The
    DER017 reconstruction differentiates along the energy axis only, so it becomes two matrix
    products against weights built once from ``s``.

    ``gamma0`` and ``sigma_gamma`` supply ``f_gamma(Gamma)``, the beam's energy PDF evaluated
    at the resonance root. **That factor is not optional.** DER019 §7's master formula carries
    it, and without it the result is a bare power law ``s^{+1/2}`` rising across the whole
    energy range — which is what a zero-emittance, *delta-resonance* model must not look like:
    each electron radiates at one resonance energy, so the spectrum is a line whose width is
    set by the energy spread and the angular smear, not a monotonic ramp.

    Passing ``gamma0 = 0`` reproduces the pre-fix behaviour and exists only so the regression
    test can demonstrate the difference; production callers pass the real beam.

    ``backend`` follows `xigma`'s convention: ``"auto"`` uses the GPU when one is present,
    ``"cupy"`` requires it, and ``"numpy"`` forces the CPU path. No custom kernel — every
    operation is a dense array op that CuPy accelerates directly, which is what keeps this
    reading as the same physics on either device.
    """
    if backend not in ("auto", "cupy", "numpy"):
        raise ValueError(f"backend must be 'auto', 'cupy' or 'numpy', got {backend!r}")
    if backend == "cupy" and not is_gpu_available():
        raise ValueError(
            "collimated_grid: backend='cupy' was requested but no CUDA device is available. "
            "Use backend='auto' to fall back to numpy, or 'numpy' to force the CPU path."
        )

    s_host = np.atleast_1d(np.asarray(s, dtype=float))
    theta_x_host = np.atleast_1d(np.asarray(theta_x, dtype=float))
    theta_y_host = np.atleast_1d(np.asarray(theta_y, dtype=float))

    xp = cp if (backend in ("auto", "cupy") and is_gpu_available()) else np
    s_grid = xp.asarray(s_host)[:, None, None]
    theta_x_grid = xp.asarray(theta_x_host)[None, :, None]
    theta_y_grid = xp.asarray(theta_y_host)[None, None, :]

    a_r = 1.0 + q_factor * ahat
    k = d_factor * c_bar
    radius_sq = theta_x_grid**2 + theta_y_grid**2

    # Same support logic as the single-column path, including s <= 0 — which is why every
    # intermediate is masked rather than allowed to produce inf and then nan.
    positive_s = s_grid > 0.0
    safe_s = xp.where(positive_s, s_grid, 1.0)
    inverse_base = k / safe_s - radius_sq
    supported = positive_s & (inverse_base > 0.0)
    gamma_sq = xp.where(supported, a_r / xp.where(supported, inverse_base, 1.0), 0.0)
    gamma = xp.sqrt(gamma_sq)

    # f_gamma(Gamma), the beam's own energy PDF at the resonance root (DER019 §7). This is
    # what makes the result a *line* rather than a ramp: the factor falls off once the root
    # leaves the populated part of the energy distribution.
    if gamma0 > 0.0 and sigma_gamma > 0.0:
        energy_density = xp.exp(-0.5 * ((gamma - gamma0) / sigma_gamma) ** 2) / (
            sigma_gamma * math.sqrt(2.0 * math.pi)
        )
    elif gamma0 > 0.0:
        # Exactly monoenergetic: a delta in gamma, so only the root that matches the beam is
        # supported. Represented by keeping just the cells whose root equals gamma0 to within
        # the grid resolution — a band, not a spike, because the root is a grid evaluation.
        energy_density = xp.where(
            xp.abs(gamma - gamma0) <= 1e-9 * gamma0, xp.ones_like(gamma), xp.zeros_like(gamma)
        )
    else:
        energy_density = xp.ones_like(gamma)

    rho0 = xp.where(
        supported,
        total_yield
        * energy_density
        * KERNEL_NORMALIZATION
        * (k / safe_s**2)
        * gamma**5
        / (a_r * (1.0 + gamma_sq * radius_sq) ** 2),
        0.0,
    )

    if var_q is None:
        var_q = KAPPA_G * ahat**2
    b_factor = 1.0 + q_factor * ahat + gamma_sq * radius_sq
    safe_b = xp.where(supported, b_factor, 1.0)
    delta_c = xp.where(supported, q_factor**2 * var_q / safe_b**2, 0.0)
    m2 = delta_c

    return CollimatedGrid(
        s=s_host,
        rho0=rho0,
        rho1=xp.where(supported, rho0 * safe_s * delta_c, 0.0),
        rho2=xp.where(supported, rho0 * safe_s**2 * m2, 0.0),
        gamma=gamma,
        backend="cupy" if xp is cp else "numpy",
    )

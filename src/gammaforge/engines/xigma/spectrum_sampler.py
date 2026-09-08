"""Stage 2: CuPy ring/annulus importance-sampling kernel for xigma.

Ports the predecessor's `spectrum_kernel_4d` (ComptonSuite) onto GammaForge's
architecture:
- Applies `KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2 pi)` (RES033).
- Adapts the ahat quadrature loop to arbitrary non-uniform target grids (RES032),
  consuming 1D device arrays `ahat_centers` and `ahat_widths`.
- Dispatches multi-point angular spectrum queries to GPU rawkernel when CUDA and CuPy
  are available, falling back to NumPy brute-force grid quadrature.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .stages import Table

try:
    import cupy as cp
    from cupyx import jit

    _HAS_CUPY = True
except Exception:
    cp = None
    jit = None
    _HAS_CUPY = False

from .stages import KERNEL_NORMALIZATION_CONSTANT

__all__ = [
    "is_gpu_available",
    "calculate_angular_spectrum_gpu",
    "gamma_bracket",
]

# Sizing and launch constants (matching ComptonSuite config.py)
SINGLE_PRECISION = True
CP_FLOAT = np.float32
CP_UINT = np.uint32
CP_INT = np.int32
CP_PI = CP_FLOAT(np.pi)
CP_TWO_PI = CP_FLOAT(2.0 * np.pi)
CP_ONE = CP_FLOAT(1.0)
CP_ZERO = CP_FLOAT(0.0)

X_THREADS = 128
N_RINGS_MIN = 32
MAX_RINGS = 32
MAX_ARCS = 4 * MAX_RINGS
ARC_STRIDE = 3
RING_STRIDE = 9
RINGS_SIZE = CP_UINT(RING_STRIDE * MAX_RINGS)
SCRATCH_SIZE = (RING_STRIDE + 2) * MAX_RINGS
INVAL = CP_FLOAT(9999.0)

PHI_EDGES = 32
PHI_CELLS = PHI_EDGES - 1
CUM_WEIGHTS_SIZE = MAX_ARCS * PHI_EDGES

SAMPLES_TOTAL = 256
SAMPLES_REPEAT = (SAMPLES_TOTAL + X_THREADS - 1) // X_THREADS
THREAD_STRIDE = 3 * SAMPLES_REPEAT + 1
GOLDEN_PHI = 1.618033988749894848
PROPOSAL_FLOOR_FRACTION = 1e-3


if _HAS_CUPY:
    @jit.rawkernel(device=True)
    def _cdf_cell(cumulative, offset, target):
        """Bracket a quantile in the same piecewise-constant PDF used for weighting."""
        left = CP_UINT(0)
        right = CP_UINT(PHI_CELLS)
        while right - left > 1:
            mid = (left + right) // 2
            if cumulative[offset + mid] <= target:
                left = mid
            else:
                right = mid
        return left
else:
    _cdf_cell = None


def is_gpu_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device is accessible."""
    if not _HAS_CUPY:
        return False
    try:
        return bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def _define_kernel():
    if not _HAS_CUPY:
        return None

    @jit.rawkernel()
    def _spectrum_kernel_4d_impl(
        output,
        params_Arr,
        H,
        H_marginal,
        gamma_min,
        gamma_width,
        n_gamma,
        theta_x_min,
        theta_x_width,
        n_theta_x,
        theta_y_min,
        theta_y_width,
        n_theta_y,
        ahat_centers,
        ahat_widths,
        ahat_min,
        ahat_max,
        n_a0,
        gamma_lo,
        gamma_hi,
        dx,
        dy,
        phi_pol,
        subsampling,
    ):
        thread_idx = jit.threadIdx.x
        out_idx = jit.blockIdx.x

        TMP_FLOAT_ARRAY = jit.shared_memory(CP_FLOAT, SCRATCH_SIZE)

        n_arcs_shared = jit.shared_memory(CP_UINT, 1)
        arcs = jit.shared_memory(CP_FLOAT, ARC_STRIDE * MAX_ARCS)
        cum_cell_weights = jit.shared_memory(CP_FLOAT, CUM_WEIGHTS_SIZE)
        thread_samples = jit.shared_memory(CP_UINT, X_THREADS * THREAD_STRIDE)

        x0 = params_Arr[out_idx, 0]
        y0 = params_Arr[out_idx, 1]
        s = params_Arr[out_idx, 2]
        box_x0 = x0 - (theta_x_min + theta_x_width * n_theta_x / 2)
        box_y0 = y0 - (theta_y_min + theta_y_width * n_theta_y / 2)

        if s <= CP_ZERO:
            skip = True
        else:
            rmin_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + ahat_max) / gamma_lo**2))
            rmax_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + ahat_min) / gamma_hi**2))

            rmin_r = cp.sqrt(max(cp.abs(box_x0) - dx, CP_ZERO) ** 2 + max(cp.abs(box_y0) - dy, CP_ZERO) ** 2)

            diam = 2 * cp.sqrt(dx**2 + dy**2)
            xm = dx + cp.abs(box_x0)
            ym = dy + cp.abs(box_y0)
            rmax_r = cp.sqrt(xm**2 + ym**2)

            rmin = max(rmin_g, rmin_r)
            rmax = min(rmax_g, rmax_r)

            skip = rmin >= rmax

        if not skip:
            r_inside = max(CP_ZERO, min(dx - cp.abs(box_x0), dy - cp.abs(box_y0)))
            n_rings = min(CP_UINT(MAX_RINGS), max(CP_UINT(N_RINGS_MIN), CP_UINT(MAX_RINGS * (rmax - rmin) / diam)))
            dr = (rmax - rmin) / n_rings

            rings = TMP_FLOAT_ARRAY
            phi_cur = TMP_FLOAT_ARRAY
            if thread_idx < n_rings:
                phi_cur[RINGS_SIZE + 2 * thread_idx + 0] = -INVAL
                phi_cur[RINGS_SIZE + 2 * thread_idx + 1] = INVAL

                r_idx = thread_idx
                r = rmin + dr * (CP_FLOAT(r_idx) + CP_FLOAT(0.5))
                n_arcs = CP_UINT(0)

                if r < r_inside:
                    rings[r_idx * RING_STRIDE + 0] = CP_ONE
                    rings[r_idx * RING_STRIDE + 1] = CP_ZERO
                    rings[r_idx * RING_STRIDE + 2] = CP_TWO_PI
                else:
                    for q_idx in jit.range(4):
                        sin_pos = CP_UINT((q_idx // 2))
                        cos_pos = CP_UINT(((q_idx + 1) // 2) % 2)

                        sin_sign = CP_INT(2 * sin_pos - 1)
                        cos_sign = CP_INT(2 * cos_pos - 1)

                        cos_0 = (dx - box_x0) / r
                        sin_0 = cp.sqrt(CP_ONE - cos_0**2)

                        cos_1 = (-dx - box_x0) / r
                        sin_1 = cp.sqrt(CP_ONE - cos_1**2)

                        sin_2 = (dy - box_y0) / r
                        cos_2 = cp.sqrt(CP_ONE - sin_2**2)

                        sin_3 = (-dy - box_y0) / r
                        cos_3 = cp.sqrt(CP_ONE - sin_3**2)

                        if cos_sign * cos_0 > 0 and cp.abs(box_y0 + r * sin_0 * sin_sign) < dy:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (1 - sin_pos)] = cp.arctan2(sin_0 * sin_sign, cos_0)

                        if cos_sign * cos_1 > 0 and cp.abs(box_y0 + r * sin_1 * sin_sign) < dy:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (sin_pos)] = cp.arctan2(sin_1 * sin_sign, cos_1)

                        if sin_sign * sin_2 > 0 and cp.abs(box_x0 + r * cos_2 * cos_sign) < dx:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (cos_pos)] = cp.arctan2(sin_2, cos_2 * cos_sign)

                        if sin_sign * sin_3 > 0 and cp.abs(box_x0 + r * cos_3 * cos_sign) < dx:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (1 - cos_pos)] = cp.arctan2(sin_3, cos_3 * cos_sign)

                        if phi_cur[RINGS_SIZE + 2 * thread_idx + 1] < 1000.0:
                            rings[r_idx * RING_STRIDE + 1 + 2 * n_arcs + 0] = phi_cur[RINGS_SIZE + 2 * thread_idx + 0]
                            rings[r_idx * RING_STRIDE + 1 + 2 * n_arcs + 1] = phi_cur[RINGS_SIZE + 2 * thread_idx + 1]

                            phi_cur[RINGS_SIZE + 2 * thread_idx + 0] = -INVAL
                            phi_cur[RINGS_SIZE + 2 * thread_idx + 1] = INVAL

                            n_arcs += CP_UINT(1)

                    if phi_cur[RINGS_SIZE + 2 * thread_idx + 0] > -1000.0 and rings[r_idx * RING_STRIDE + 1] < -1000.0:
                        rings[r_idx * RING_STRIDE + 1] = phi_cur[RINGS_SIZE + 2 * thread_idx + 0] - CP_TWO_PI

                    rings[r_idx * RING_STRIDE + 0] = CP_FLOAT(n_arcs)

        jit.syncthreads()

        if not skip:
            n_arcs = CP_UINT(0)
            if thread_idx == 0:
                for i in jit.range(CP_INT(n_rings)):
                    n_ring_arcs = CP_INT(rings[i * RING_STRIDE + 0])
                    for j in jit.range(n_ring_arcs):
                        if n_arcs < MAX_ARCS:
                            arcs[n_arcs * ARC_STRIDE + 0] = rmin + dr * (CP_FLOAT(i) + CP_FLOAT(0.5))
                            arcs[n_arcs * ARC_STRIDE + 1] = rings[i * RING_STRIDE + 1 + 2 * j + 0]
                            arcs[n_arcs * ARC_STRIDE + 2] = rings[i * RING_STRIDE + 1 + 2 * j + 1]
                        n_arcs += CP_UINT(1)
                n_arcs_shared[0] = min(n_arcs, CP_UINT(MAX_ARCS))

        jit.syncthreads()

        if not skip:
            n_arcs = n_arcs_shared[0]
            cell_weights = cum_cell_weights
            weights_size = n_arcs * PHI_EDGES
            weights_repeat = CP_INT((weights_size + X_THREADS - 1) // X_THREADS)

            for i in jit.range(weights_repeat):
                sample_idx = CP_UINT(i * X_THREADS) + thread_idx
                phi_idx = sample_idx % PHI_EDGES
                arc_idx = sample_idx // PHI_EDGES

                if arc_idx < n_arcs and phi_idx < PHI_CELLS:
                    r = arcs[arc_idx * ARC_STRIDE + 0]
                    phi_min = arcs[arc_idx * ARC_STRIDE + 1]
                    phi_max = arcs[arc_idx * ARC_STRIDE + 2]

                    phi = phi_min + ((phi_idx + CP_ONE / 2) / PHI_CELLS) * (phi_max - phi_min)
                    x = x0 + r * cp.cos(phi)
                    y = y0 + r * cp.sin(phi)

                    w = CP_ZERO
                    if (
                        x > theta_x_min
                        and x < theta_x_min + theta_x_width * n_theta_x
                        and y > theta_y_min
                        and y < theta_y_min + theta_y_width * n_theta_y
                    ):
                        xi = min(CP_UINT(n_theta_x - 1), CP_UINT(cp.floor((x - theta_x_min) / theta_x_width)))
                        yj = min(CP_UINT(n_theta_y - 1), CP_UINT(cp.floor((y - theta_y_min) / theta_y_width)))
                        w = H_marginal[xi, yj]

                    dphi_cell = (phi_max - phi_min) / PHI_CELLS
                    cell_weights[sample_idx] = w * dphi_cell * r

            jit.syncthreads()

            if thread_idx < n_arcs:
                total = CP_ZERO
                for i in jit.range(PHI_CELLS):
                    tmp = cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)]
                    cum_cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)] = total
                    total += tmp
                cum_cell_weights[thread_idx * PHI_EDGES + PHI_CELLS] = total
            jit.syncthreads()

            if thread_idx == 0:
                TMP_FLOAT_ARRAY[0] = CP_ZERO
                for i in jit.range(CP_INT(n_arcs)):
                    TMP_FLOAT_ARRAY[0] += cum_cell_weights[CP_UINT(i * PHI_EDGES) + (PHI_EDGES - 1)]
            jit.syncthreads()

            total_weight = TMP_FLOAT_ARRAY[0]
            if total_weight <= CP_ZERO:
                return  # Uniform block-wide decision after the reduction barrier.
            thread_samples[thread_idx * THREAD_STRIDE] = CP_UINT(0)
            jit.syncthreads()
            if thread_idx == 0:
                active_arcs = CP_UINT(0)
                for k in jit.range(CP_INT(n_arcs)):
                    if cum_cell_weights[k * PHI_EDGES + (PHI_EDGES - 1)] > CP_ZERO:
                        active_arcs += CP_UINT(1)
                cur_thread = CP_UINT(0)
                for k in jit.range(CP_INT(n_arcs)):
                    arc_weight = cum_cell_weights[k * PHI_EDGES + (PHI_EDGES - 1)]
                    s_add = CP_UINT(0)
                    if arc_weight > CP_ZERO:
                        s_add = CP_UINT(1) + CP_UINT(cp.floor((SAMPLES_TOTAL - active_arcs) * arc_weight / total_weight))
                    for j in jit.range(CP_INT(s_add)):
                        n_samples = thread_samples[cur_thread * THREAD_STRIDE + 0]
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 0] = CP_UINT(k)
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 1] = CP_UINT(j)
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 2] = CP_UINT(s_add)
                        thread_samples[cur_thread * THREAD_STRIDE + 0] += CP_UINT(1)
                        cur_thread = (cur_thread + CP_UINT(1)) % X_THREADS
            jit.syncthreads()

            f_tot = CP_ZERO
            n_thread_samples = thread_samples[thread_idx * THREAD_STRIDE + 0]
            for thread_sample_idx in jit.range(CP_UINT(SAMPLES_REPEAT)):
                if thread_sample_idx < n_thread_samples:
                    arc_idx = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 0]
                    arc_sample_idx = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 1]
                    n_arc_samples = thread_samples[thread_idx * THREAD_STRIDE + 1 + 3 * thread_sample_idx + 2]

                    arc_r = arcs[arc_idx * ARC_STRIDE + 0]
                    phi_min = arcs[arc_idx * ARC_STRIDE + 1]
                    phi_max = arcs[arc_idx * ARC_STRIDE + 2]
                    arc_total_weight = cum_cell_weights[arc_idx * PHI_EDGES + (PHI_EDGES - 1)]
                    dphi_cell = (phi_max - phi_min) / PHI_CELLS
                    arc_area = dphi_cell * arc_r * dr

                    for di in jit.range(subsampling):
                        subsample_idx = arc_sample_idx * subsampling + di
                        reg = (subsample_idx + 0.5) / n_arc_samples / subsampling
                        fib = cp.remainder(subsample_idx * GOLDEN_PHI, 1.0)

                        theta_min = arc_r - dr / 2
                        theta_max = theta_min + dr

                        theta_sq = theta_min**2 + fib * (theta_max**2 - theta_min**2)
                        theta = cp.sqrt(theta_sq)

                        target_cdf = reg * arc_total_weight
                        phi_idx = _cdf_cell(cum_cell_weights, arc_idx * PHI_EDGES, target_cdf)
                        cell_weight = (
                            cum_cell_weights[arc_idx * PHI_EDGES + phi_idx + 1]
                            - cum_cell_weights[arc_idx * PHI_EDGES + phi_idx]
                        )
                        sample_area = CP_ZERO
                        fraction = CP_ZERO
                        if cell_weight > CP_ZERO:
                            fraction = (target_cdf - cum_cell_weights[arc_idx * PHI_EDGES + phi_idx]) / cell_weight
                            sample_area = arc_area / n_arc_samples / subsampling * arc_total_weight / cell_weight
                        phi = phi_min + (CP_FLOAT(phi_idx) + fraction) * dphi_cell

                        x = x0 + theta * cp.cos(phi)
                        y = y0 + theta * cp.sin(phi)

                        if (
                            x > theta_x_min
                            and x < theta_x_min + theta_x_width * n_theta_x
                            and y > theta_y_min
                            and y < theta_y_min + theta_y_width * n_theta_y
                        ):
                            Xf = min(max((x - theta_x_min) / theta_x_width - CP_FLOAT(0.5), CP_ZERO), CP_FLOAT(n_theta_x - 1))
                            Yf = min(max((y - theta_y_min) / theta_y_width - CP_FLOAT(0.5), CP_ZERO), CP_FLOAT(n_theta_y - 1))
                            xi2 = CP_INT(cp.floor(Xf))
                            yj2 = CP_INT(cp.floor(Yf))
                            xi2 = min(max(xi2, CP_INT(0)), CP_INT(n_theta_x - 2))
                            yj2 = min(max(yj2, CP_INT(0)), CP_INT(n_theta_y - 2))
                            xw = Xf - CP_FLOAT(xi2)
                            yw = Yf - CP_FLOAT(yj2)

                            inv_base = CP_ONE / s - theta_sq
                            h_sum = CP_ZERO
                            if inv_base > CP_ZERO:
                                for ai2 in jit.range(CP_INT(n_a0)):
                                    a0_val = ahat_centers[ai2]
                                    a0_width = ahat_widths[ai2]
                                    g_sq = (CP_ONE + a0_val) / inv_base
                                    g = cp.sqrt(g_sq)

                                    if g >= gamma_min + gamma_width / 2 and g <= gamma_min + gamma_width * (CP_FLOAT(n_gamma) - CP_FLOAT(0.5)):
                                        gth_sq_inv = CP_ONE / (CP_ONE + theta_sq * g_sq) ** 2
                                        # Head-on, linearly polarized specialization of the
                                        # lab-frame per-electron projection (RES060/DER006).
                                        # ``x``/``y`` are the sampled electron angles and
                                        # ``x0``/``y0`` are the observer direction.
                                        n_norm = cp.sqrt(CP_ONE + x0**2 + y0**2)
                                        nx = x0 / n_norm
                                        ny = y0 / n_norm
                                        nz = CP_ONE / n_norm
                                        v_norm = cp.sqrt(CP_ONE + x**2 + y**2)
                                        beta = cp.sqrt(CP_ONE - CP_ONE / g_sq)
                                        vx = beta * x / v_norm
                                        vy = beta * y / v_norm
                                        # Stable exact lab-vector identity (RES060); no 1 - near-1 subtraction.
                                        one_minus_vn = CP_ONE / (g_sq * (CP_ONE + beta)) + beta * CP_FLOAT(0.5) * (
                                            (x / v_norm - nx)**2 + (y / v_norm - ny)**2 + (CP_ONE / v_norm - nz)**2
                                        )
                                        cos_pol = cp.cos(phi_pol)
                                        sin_pol = cp.sin(phi_pol)
                                        n_e0 = nx * cos_pol + ny * sin_pol
                                        v_e0 = vx * cos_pol + vy * sin_pol
                                        pol_factor = (
                                            CP_ONE
                                            - n_e0**2 / (g_sq * one_minus_vn**2)
                                            + CP_FLOAT(2.0) * n_e0 * v_e0 / one_minus_vn
                                        )
                                        prefac = pol_factor * g**5 * gth_sq_inv / (CP_ONE + a0_val)

                                        Gf = (g - gamma_min) / gamma_width - CP_FLOAT(0.5)
                                        gi2 = CP_INT(cp.floor(Gf))
                                        gi2 = min(max(gi2, CP_INT(0)), CP_INT(n_gamma - 2))
                                        gw = min(max(Gf - CP_FLOAT(gi2), CP_ZERO), CP_ONE)

                                        h000 = H[gi2, xi2, yj2, ai2]
                                        h100 = H[gi2 + 1, xi2, yj2, ai2]
                                        h010 = H[gi2, xi2 + 1, yj2, ai2]
                                        h110 = H[gi2 + 1, xi2 + 1, yj2, ai2]
                                        h001 = H[gi2, xi2, yj2 + 1, ai2]
                                        h101 = H[gi2 + 1, xi2, yj2 + 1, ai2]
                                        h011 = H[gi2, xi2 + 1, yj2 + 1, ai2]
                                        h111 = H[gi2 + 1, xi2 + 1, yj2 + 1, ai2]

                                        h_yj = (h000 * (CP_ONE - xw) + h010 * xw) * (CP_ONE - yw) + (
                                            h001 * (CP_ONE - xw) + h011 * xw
                                        ) * yw
                                        h_yj1 = (h100 * (CP_ONE - xw) + h110 * xw) * (CP_ONE - yw) + (
                                            h101 * (CP_ONE - xw) + h111 * xw
                                        ) * yw
                                        h_val = h_yj * (CP_ONE - gw) + h_yj1 * gw

                                        h_sum += h_val * a0_width * prefac

                            f = h_sum
                            f_tot += f * sample_area

            jit.atomic_add(output, out_idx, f_tot / s**2)

    return _spectrum_kernel_4d_impl


_kernel = _define_kernel()


def gamma_bracket(table: Table, q: float = 1e-4) -> tuple[float, float]:
    """Lowest and highest gamma populated in `table.H` by quantile."""
    marginal = table.H.sum(axis=(1, 2, 3))
    total = marginal.sum()
    if total <= 0:
        return float(table.gamma_edges[0]), float(table.gamma_edges[-1])
    cdf = np.cumsum(marginal) / total
    lo = float(np.interp(q, cdf, table.gamma_centers))
    hi = float(np.interp(1.0 - q, cdf, table.gamma_centers))
    return max(lo, 1.0), max(hi, 1.0)


def calculate_angular_spectrum_gpu(
    table: Table,
    theta_x,
    theta_y,
    s,
    *,
    psi_pol: float = 0.0,
    subsampling: int = 32,
) -> np.ndarray:
    """Compute `d3N / (ds dtheta_x dtheta_y)` on CUDA device via importance sampling.

    Output shape: `(len(theta_x), len(theta_y), len(s))`.
    """
    if not is_gpu_available():
        raise RuntimeError("calculate_angular_spectrum_gpu: CuPy or a CUDA device is not available")
    if isinstance(subsampling, bool) or not isinstance(subsampling, Integral) or not 1 <= subsampling <= np.iinfo(CP_UINT).max // SAMPLES_TOTAL:
        raise ValueError("subsampling must be a positive integer with uint32-safe sample indices")
    if not math.isfinite(psi_pol):
        raise ValueError("psi_pol must be finite")
    for edges in (table.gamma_edges, table.theta_x_edges, table.theta_y_edges):
        widths = np.diff(edges)
        if len(widths) < 2 or not np.allclose(widths, widths[0], rtol=1e-10, atol=0.0):
            raise ValueError("CuPy requires at least two uniform bins on gamma and angular axes")
    if table.gamma_centers[0] < 1.0 or table.ahat_edges[0] < 0.0:
        raise ValueError("CuPy requires gamma >= 1 and nonnegative ahat")
    if not np.all(np.isfinite(table.H)) or np.any(table.H < 0.0):
        raise ValueError("CuPy table density must be finite and nonnegative")

    tx = np.atleast_1d(np.asarray(theta_x, dtype=np.float32))
    ty = np.atleast_1d(np.asarray(theta_y, dtype=np.float32))
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float32))
    if any(values.ndim != 1 or not np.all(np.isfinite(values)) for values in (tx, ty, s_arr)):
        raise ValueError("CuPy query axes must be finite one-dimensional arrays")

    grid_x = tx.size * ty.size * s_arr.size
    if grid_x == 0 or not np.any(table.H):
        return np.zeros((tx.size, ty.size, s_arr.size), dtype=CP_FLOAT)
    params = cp.stack(
        cp.meshgrid(cp.asarray(tx), cp.asarray(ty), cp.asarray(s_arr), indexing="ij"), 3
    ).reshape(-1, 3).astype(CP_FLOAT)

    gamma_min = CP_FLOAT(table.gamma_edges[0])
    gamma_width = CP_FLOAT(table.gamma_edges[1] - table.gamma_edges[0])
    n_gamma = CP_UINT(table.H.shape[0])

    theta_x_min = CP_FLOAT(table.theta_x_edges[0])
    theta_x_width = CP_FLOAT(table.theta_x_edges[1] - table.theta_x_edges[0])
    n_theta_x = CP_UINT(table.H.shape[1])

    theta_y_min = CP_FLOAT(table.theta_y_edges[0])
    theta_y_width = CP_FLOAT(table.theta_y_edges[1] - table.theta_y_edges[0])
    n_theta_y = CP_UINT(table.H.shape[2])

    ahat_centers = cp.asarray(table.ahat_centers, dtype=CP_FLOAT)
    ahat_widths = cp.asarray(table.ahat_widths, dtype=CP_FLOAT)
    ahat_min = CP_FLOAT(table.ahat_edges[0])
    ahat_max = CP_FLOAT(table.ahat_edges[-1])
    n_a0 = CP_UINT(table.H.shape[3])

    # Match the NumPy interpolator's center-domain support without cutting gamma tails.
    gamma_lo, gamma_hi = table.gamma_centers[[0, -1]]
    dx = float((table.theta_x_edges[-1] - table.theta_x_edges[0]) / 2)
    dy = float((table.theta_y_edges[-1] - table.theta_y_edges[0]) / 2)

    H_gpu = cp.asarray(table.H, dtype=CP_FLOAT)
    H_marginal_gpu = (H_gpu * ahat_widths[None, None, None, :]).sum(axis=(0, 3))
    # A coarse zero must not exclude nonzero interpolated target density between cells.
    H_marginal_gpu += CP_FLOAT(PROPOSAL_FLOOR_FRACTION) * H_marginal_gpu.max()

    spec = cp.zeros((grid_x,), dtype=CP_FLOAT)

    _kernel[grid_x, X_THREADS](
        spec,
        params,
        H_gpu,
        H_marginal_gpu,
        gamma_min,
        gamma_width,
        n_gamma,
        theta_x_min,
        theta_x_width,
        n_theta_x,
        theta_y_min,
        theta_y_width,
        n_theta_y,
        ahat_centers,
        ahat_widths,
        ahat_min,
        ahat_max,
        n_a0,
        CP_FLOAT(gamma_lo),
        CP_FLOAT(gamma_hi),
        CP_FLOAT(dx),
        CP_FLOAT(dy),
        CP_FLOAT(psi_pol),
        CP_UINT(subsampling),
    )
    cp.cuda.Stream.null.synchronize()

    out = (KERNEL_NORMALIZATION_CONSTANT * spec).reshape((tx.size, ty.size, s_arr.size)).get()
    if not np.all(np.isfinite(out)):
        raise RuntimeError(
            "calculate_angular_spectrum_gpu produced non-finite samples; use backend='numpy' "
            "while the experimental CuPy sampler is under validation."
        )
    return out

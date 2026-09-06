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
INVAL = CP_FLOAT(9999.0)

PHI_EDGES = 32
PHI_CELLS = PHI_EDGES - 1
CUM_WEIGHTS_SIZE = MAX_ARCS * PHI_EDGES

CDF_PHI_RESOLUTION = 32
CDF_PHI_REPEAT = (CDF_PHI_RESOLUTION + X_THREADS - 1) // X_THREADS
CDF_SIZE = CDF_PHI_RESOLUTION * MAX_ARCS

SAMPLES_TOTAL = 256
SAMPLES_REPEAT = (SAMPLES_TOTAL + X_THREADS - 1) // X_THREADS
THREAD_STRIDE = 3 * SAMPLES_REPEAT + 1
R_MAX_NUDGE = 128
GOLDEN_PHI = 1.618033988749894848


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

        inv_cdf = jit.shared_memory(CP_FLOAT, CDF_SIZE)
        TMP_FLOAT_ARRAY = inv_cdf

        n_arcs_shared = jit.shared_memory(CP_UINT, 1)
        arcs = jit.shared_memory(CP_FLOAT, ARC_STRIDE * MAX_ARCS)
        cum_cell_weights = jit.shared_memory(CP_FLOAT, CUM_WEIGHTS_SIZE)
        thread_samples = jit.shared_memory(CP_UINT, X_THREADS * THREAD_STRIDE)

        x0 = params_Arr[out_idx, 0]
        y0 = params_Arr[out_idx, 1]
        s = params_Arr[out_idx, 2]

        if s <= CP_ZERO:
            skip = True
        else:
            rmin_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + ahat_max) / gamma_lo**2))
            rmax_g = cp.sqrt(cp.maximum(CP_ZERO, CP_ONE / s - (CP_ONE + ahat_min) / gamma_hi**2))

            rmin_r = cp.sqrt(max(cp.abs(x0) - dx, CP_ZERO) ** 2 + max(cp.abs(y0) - dy, CP_ZERO) ** 2)

            diam = 2 * cp.sqrt(dx**2 + dy**2)
            xm = dx + cp.abs(x0)
            ym = dy + cp.abs(y0)
            rmax_r = cp.sqrt(xm**2 + ym**2) - diam / R_MAX_NUDGE

            rmin = max(rmin_g, rmin_r)
            rmax = min(rmax_g, rmax_r)

            skip = rmin >= rmax

        if not skip:
            r_inside = max(CP_ZERO, min(dx - cp.abs(x0), dy - cp.abs(y0)))
            n_rings = max(N_RINGS_MIN, CP_UINT(MAX_RINGS * (rmax - rmin) / diam))
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

                        cos_0 = (dx - x0) / r
                        sin_0 = cp.sqrt(CP_ONE - cos_0**2)

                        cos_1 = (-dx - x0) / r
                        sin_1 = cp.sqrt(CP_ONE - cos_1**2)

                        sin_2 = (dy - y0) / r
                        cos_2 = cp.sqrt(CP_ONE - sin_2**2)

                        sin_3 = (-dy - y0) / r
                        cos_3 = cp.sqrt(CP_ONE - sin_3**2)

                        if cos_sign * cos_0 > 0 and cp.abs(y0 + r * sin_0 * sin_sign) < dy:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (1 - sin_pos)] = cp.arctan2(sin_0 * sin_sign, cos_0)

                        if cos_sign * cos_1 > 0 and cp.abs(y0 + r * sin_1 * sin_sign) < dy:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (sin_pos)] = cp.arctan2(sin_1 * sin_sign, cos_1)

                        if sin_sign * sin_2 > 0 and cp.abs(x0 + r * cos_2 * cos_sign) < dx:
                            phi_cur[RINGS_SIZE + 2 * thread_idx + (cos_pos)] = cp.arctan2(sin_2, cos_2 * cos_sign)

                        if sin_sign * sin_3 > 0 and cp.abs(x0 + r * cos_3 * cos_sign) < dx:
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
                for i in jit.range(PHI_EDGES):
                    tmp = cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)]
                    cum_cell_weights[thread_idx * PHI_EDGES + CP_UINT(i)] = total
                    total += tmp
            jit.syncthreads()

            if thread_idx == 0:
                TMP_FLOAT_ARRAY[0] = CP_ZERO
                for i in jit.range(CP_INT(n_arcs)):
                    TMP_FLOAT_ARRAY[0] += cum_cell_weights[CP_UINT(i * PHI_EDGES) + (PHI_EDGES - 1)]
            jit.syncthreads()

            total_weight = TMP_FLOAT_ARRAY[0]
            thread_samples[thread_idx * THREAD_STRIDE] = CP_UINT(0)
            if thread_idx == 0:
                cur_thread = CP_UINT(0)
                for k in jit.range(CP_INT(n_arcs)):
                    arc_weight = cum_cell_weights[k * PHI_EDGES + (PHI_EDGES - 1)]
                    s_add = CP_UINT(cp.floor(SAMPLES_TOTAL * arc_weight / total_weight))
                    for j in jit.range(CP_INT(s_add)):
                        n_samples = thread_samples[cur_thread * THREAD_STRIDE + 0]
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 0] = CP_UINT(k)
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 1] = CP_UINT(j)
                        thread_samples[cur_thread * THREAD_STRIDE + 1 + 3 * n_samples + 2] = CP_UINT(s_add)
                        thread_samples[cur_thread * THREAD_STRIDE + 0] += CP_UINT(1)
                        cur_thread = (cur_thread + CP_UINT(1)) % X_THREADS
            jit.syncthreads()

            for arc_idx in jit.range(n_arcs):
                phi_min = arcs[arc_idx * ARC_STRIDE + 1]
                phi_max = arcs[arc_idx * ARC_STRIDE + 2]
                dphi = (phi_max - phi_min) / PHI_CELLS
                for k in jit.range(CDF_PHI_REPEAT):
                    r_idx = CP_UINT(k * X_THREADS) + thread_idx
                    if r_idx < CDF_PHI_RESOLUTION:
                        r = cum_cell_weights[arc_idx * PHI_EDGES + (PHI_EDGES - 1)] * r_idx / (CDF_PHI_RESOLUTION - 1)
                        left = CP_UINT(0)
                        right = CP_UINT(PHI_EDGES - 1)
                        while right - left > 1:
                            mid = (left + right) // 2
                            if cum_cell_weights[arc_idx * PHI_EDGES + mid] <= r:
                                left = mid
                            else:
                                right = mid

                        cdf_i = cum_cell_weights[arc_idx * PHI_EDGES + (left + 0)]
                        cdf_ip1 = cum_cell_weights[arc_idx * PHI_EDGES + (left + 1)]
                        cdf_span = cdf_ip1 - cdf_i
                        fac = CP_ZERO
                        if cdf_span > CP_ZERO:
                            fac = (r - cdf_i) / cdf_span
                        inv_cdf[arc_idx * CDF_PHI_RESOLUTION + r_idx] = phi_min + (CP_FLOAT(left) + fac) * dphi
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

                        il = CP_UINT(cp.floor(reg * (CDF_PHI_RESOLUTION - 1)))
                        fac = reg * (CDF_PHI_RESOLUTION - 1) - CP_FLOAT(il)
                        phi = (
                            inv_cdf[arc_idx * CDF_PHI_RESOLUTION + il] * (CP_ONE - fac)
                            + inv_cdf[arc_idx * CDF_PHI_RESOLUTION + (il + 1)] * fac
                        )

                        phi_idx = min(PHI_CELLS - 1, CP_UINT(PHI_CELLS * (phi - phi_min) / (phi_max - phi_min)))
                        cell_weight = (
                            cum_cell_weights[arc_idx * PHI_EDGES + phi_idx + 1]
                            - cum_cell_weights[arc_idx * PHI_EDGES + phi_idx]
                        )
                        sample_area = CP_ZERO
                        if cell_weight > CP_ZERO:
                            sample_area = arc_area / n_arc_samples / subsampling * arc_total_weight / cell_weight

                        x = x0 + theta * cp.cos(phi)
                        y = y0 + theta * cp.sin(phi)

                        if (
                            x > theta_x_min
                            and x < theta_x_min + theta_x_width * n_theta_x
                            and y > theta_y_min
                            and y < theta_y_min + theta_y_width * n_theta_y
                        ):
                            cos_pol = cp.cos(phi_pol - phi) ** 2

                            Xf = (x - theta_x_min) / theta_x_width - CP_FLOAT(0.5)
                            Yf = (y - theta_y_min) / theta_y_width - CP_FLOAT(0.5)
                            xi2 = CP_INT(cp.floor(Xf))
                            yj2 = CP_INT(cp.floor(Yf))
                            xw = Xf - CP_FLOAT(xi2)
                            yw = Yf - CP_FLOAT(yj2)
                            xi2 = min(max(xi2, CP_INT(0)), CP_INT(n_theta_x - 2))
                            yj2 = min(max(yj2, CP_INT(0)), CP_INT(n_theta_y - 2))

                            inv_base = CP_ONE / s - theta_sq
                            h_sum = CP_ZERO
                            if inv_base > CP_ZERO:
                                for ai2 in jit.range(CP_INT(n_a0)):
                                    a0_val = ahat_centers[ai2]
                                    a0_width = ahat_widths[ai2]
                                    g_sq = (CP_ONE + a0_val) / inv_base
                                    g = cp.sqrt(g_sq)

                                    if g > gamma_min and g < gamma_min + gamma_width * n_gamma:
                                        gth_sq_inv = CP_ONE / (CP_ONE + theta_sq * g_sq) ** 2
                                        a_fac = CP_ONE - 4 * cos_pol * theta_sq * g_sq * gth_sq_inv
                                        prefac = a_fac * g**5 * gth_sq_inv / (CP_ONE + a0_val)

                                        Gf = (g - gamma_min) / gamma_width - CP_FLOAT(0.5)
                                        gi2 = CP_INT(cp.floor(Gf))
                                        gw = Gf - CP_FLOAT(gi2)
                                        gi2 = min(max(gi2, CP_INT(0)), CP_INT(n_gamma - 2))

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

    tx = np.atleast_1d(np.asarray(theta_x, dtype=np.float32))
    ty = np.atleast_1d(np.asarray(theta_y, dtype=np.float32))
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float32))

    grid_x = tx.size * ty.size * s_arr.size
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

    gamma_lo, gamma_hi = (float(v) for v in gamma_bracket(table))
    dx = float(max(abs(float(table.theta_x_edges[0])), abs(float(table.theta_x_edges[-1]))))
    dy = float(max(abs(float(table.theta_y_edges[0])), abs(float(table.theta_y_edges[-1]))))
    dx = max(dx, 1e-12)
    dy = max(dy, 1e-12)

    H_gpu = cp.asarray(table.H, dtype=CP_FLOAT)
    H_marginal_gpu = H_gpu.sum(axis=(0, 3))

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
    return out

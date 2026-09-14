"""Explicit CUDA delta emission and finite-bin spectra, with NumPy boundaries (RES085).

Physical-energy functions mirror ``delta_emission``; the normalized-spectrum
wrappers retain the historical nominal-Doppler default. Select ``direction`` to
compare with current xigma. No production emission helpers are used.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from ...engines.xigma.stages import TrajectorySamples

__all__ = ["emission_lines", "bin_emission", "resonance_spectrum",
           "angle_integrated_spectrum", "single_electron_spectrum"]

DEFAULT_CHUNK = 65536


def _cupy():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA devices")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("delta_cupy requires CuPy and a usable CUDA device") from exc
    return cp


@lru_cache(maxsize=1)
def _line_kernel():
    # Stable double-cross-product radiation vectors, local dipole basis (DER012).
    return _cupy().ElementwiseKernel(
        "float64 g, float64 tx, float64 ty, float64 a, float64 lum, "
        "float64 ox, float64 oy, float64 photon, float64 psi, float64 eps, "
        "float64 xz, float64 yz, int32 mode",
        "float64 energy, float64 weight",
        r"""
        const double ve = sqrt(1.0 + tx*tx + ty*ty);
        const double ux = tx/ve, uy = ty/ve, uz = 1.0/ve;
        const double no = sqrt(1.0 + ox*ox + oy*oy);
        const double nx = ox/no, ny = oy/no, nz = 1.0/no;
        const double beta = sqrt(1.0 - 1.0/(g*g));
        const double slow = 1.0/(g*g*(1.0+beta));
        const double dz = ((tx-ox)*(tx+ox) + (ty-oy)*(ty+oy))
                          / (ve*no*(ve+no));
        const double dx = (ox-tx)/no + tx*dz;
        const double dy = (oy-ty)/no + ty*dz;
        const double den = slow + 0.5*beta*(dx*dx + dy*dy + dz*dz);
        const double qx = (dx+slow*ux)/den;
        const double qy = (dy+slow*uy)/den;
        const double qz = (dz+slow*uz)/den;
        const double cx = cos(xz), sx = sin(xz), cy = cos(yz), sy = sin(yz);
        const double ex = cx*cos(psi) + sx*sy*sin(psi);
        const double ey = cy*sin(psi);
        const double ez = -sx*cos(psi) + cx*sy*sin(psi);
        const double projection = ux*ex + uy*ey + uz*ez;
        double p0x = ex-projection*ux, p0y = ey-projection*uy;
        double p0z = ez-projection*uz;
        const double norm = sqrt(p0x*p0x+p0y*p0y+p0z*p0z);
        if (norm <= 1e-12) {
            energy = nan("");
            weight = nan("");
        } else {
            p0x /= norm; p0y /= norm; p0z /= norm;
            double p1x = uy*p0z-uz*p0y, p1y = uz*p0x-ux*p0z;
            double p1z = ux*p0y-uy*p0x;
            const double norm1 = sqrt(p1x*p1x+p1y*p1y+p1z*p1z);
            p1x /= norm1; p1y /= norm1; p1z /= norm1;
            const double dot0 = nx*p0x+ny*p0y+nz*p0z;
            const double dot1 = nx*p1x+ny*p1y+nz*p1z;
            const double u0x = qx*dot0-p0x, u0y = qy*dot0-p0y, u0z = qz*dot0-p0z;
            const double u1x = qx*dot1-p1x, u1y = qy*dot1-p1y, u1z = qz*dot1-p1z;
            const double pol = (u0x*u0x+u0y*u0y+u0z*u0z
                               +eps*eps*(u1x*u1x+u1y*u1y+u1z*u1z))/(1.0+eps*eps);
            const double incident = -ux*sx*cy + uy*sy - uz*cx*cy;
            const double encounter = mode == 0 ? 1.0+cx*cy
                                   : 1.0-(mode == 1 ? beta : 1.0)*incident;
            const double r2 = (tx-ox)*(tx-ox)+(ty-oy)*(ty-oy);
            energy = 2.0*photon*encounter*g*g/(1.0+a+g*g*r2);
            weight = (3.0/(2.0*3.14159265358979323846))*lum*pol*g*g
                     / ((1.0+g*g*r2)*(1.0+g*g*r2));
        }
        """, "delta_emission_lines_float64",
    )


def _inputs(samples, chunk, photon_energy, theta_x, theta_y, psi_pol, ellipticity,
            theta_xz, theta_yz, doppler):
    if not isinstance(chunk, Integral) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    if doppler not in ("nominal", "particle", "direction"):
        raise ValueError("doppler must be 'nominal', 'particle', or 'direction'")
    vals = (photon_energy, theta_x, theta_y, psi_pol, ellipticity, theta_xz, theta_yz)
    if not all(np.isfinite(v) for v in vals) or photon_energy <= 0 or not -1 <= ellipticity <= 1:
        raise ValueError("finite directions, positive photon_energy and ellipticity in [-1,1] required")
    arrays = tuple(np.asarray(a, dtype=float) for a in
                   (samples.gamma, samples.theta_x, samples.theta_y, samples.ahat(), samples.luminosity))
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError("sample fields must be same-shaped one-dimensional arrays")
    g, tx, ty, a, lum = arrays
    if any(np.any(~np.isfinite(x)) for x in arrays) or np.any(g <= 1) or np.any(a < 0) or np.any(lum < 0):
        raise ValueError("samples contain invalid gamma, ahat, luminosity, or directions")
    args = (theta_x, theta_y, photon_energy, psi_pol, ellipticity, theta_xz, theta_yz,
            np.int32(("nominal", "particle", "direction").index(doppler)))
    return arrays, args


def _chunks(cp, arrays, chunk):
    for start in range(0, arrays[0].size, chunk):
        yield start, tuple(cp.asarray(a[start:start+chunk]) for a in arrays)


def _lines(cp, arrays, args):
    energy, weight = _line_kernel()(*arrays, *args)
    if not bool(cp.all(cp.isfinite(energy) & cp.isfinite(weight))):
        raise ValueError("degenerate transverse polarization or non-finite emission")
    return energy, weight


def emission_lines(samples: TrajectorySamples, theta_x: float, theta_y: float, *,
                   photon_energy: float, psi_pol: float = 0., ellipticity: float = 0.,
                   theta_xz: float = 0., theta_yz: float = 0., doppler: str = "nominal",
                   chunk: int = DEFAULT_CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Return physical energies (erg) and weights, computed in float64 on CUDA.

    ``nominal``, ``particle`` and ``direction`` match the CPU reference modes.
    Inputs and outputs remain NumPy; temporary device work is bounded by ``chunk``.
    """
    arrays, args = _inputs(samples, chunk, photon_energy, theta_x, theta_y,
                           psi_pol, ellipticity, theta_xz, theta_yz, doppler)
    cp = _cupy()
    energies = np.empty(arrays[0].size)
    weights = np.empty_like(energies)
    for start, device in _chunks(cp, arrays, chunk):
        e, w = _lines(cp, device, args)
        energies[start:start+e.size] = cp.asnumpy(e)
        weights[start:start+w.size] = cp.asnumpy(w)
    return energies, weights


def bin_emission(energies, weights, energy_edges, *, chunk: int = DEFAULT_CHUNK) -> dict[str, object]:
    """GPU weighted histogram with NumPy outputs, including both tails.

    The final right edge is included in the last bin, as in the CPU reference.
    """
    e, w, edges = (np.asarray(a, dtype=float) for a in (energies, weights, energy_edges))
    if e.ndim != 1 or e.shape != w.shape or any(np.any(~np.isfinite(a)) for a in (e, w)) or np.any(e < 0) or np.any(w < 0):
        raise ValueError("energies and weights must be finite, nonnegative, same-shaped 1D arrays")
    _edges(edges)
    if not isinstance(chunk, Integral) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    cp = _cupy()
    bins = cp.asarray(edges)
    mass = cp.zeros(edges.size-1, dtype=cp.float64)
    tails = cp.zeros(3, dtype=cp.float64)
    for _, (de, dw) in _chunks(cp, (e, w), chunk):
        mass += cp.histogram(de, bins=bins, weights=dw)[0]
        tails += cp.stack((dw[de < bins[0]].sum(), dw[de > bins[-1]].sum(), dw.sum()))
    host = cp.asnumpy(mass)
    under, over, total = cp.asnumpy(tails)
    return {"bin_mass": host, "density": host/np.diff(edges),
            "underflow": float(under), "overflow": float(over), "total_weight": float(total)}


def _edges(edges):
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges must be finite and strictly increasing")


def resonance_spectrum(samples, s_edges, theta_x, theta_y, psi_pol=0., ellipticity=0.,
                       theta_xz=0., theta_yz=0., *, doppler="nominal", chunk=DEFAULT_CHUNK):
    """GPU-binned dN/(ds dOmega), with s = E/[2 E_laser (1+cos(xz)cos(yz))].

    The default reproduces legacy delta's nominal resonance; use ``direction``
    for current xigma. Histogramming stays on device between particle chunks.
    """
    arrays, args = _inputs(samples, chunk, 1., theta_x, theta_y,
                           psi_pol, ellipticity, theta_xz, theta_yz, doppler)
    edges = np.asarray(s_edges, dtype=float)
    _edges(edges)
    scale = 2*(1+np.cos(theta_xz)*np.cos(theta_yz))
    if scale <= 0:
        raise ValueError("normalized spectrum requires a positive nominal Doppler scale")
    cp = _cupy()
    bins = cp.asarray(edges)
    mass = cp.zeros(edges.size-1, dtype=cp.float64)
    for _, device in _chunks(cp, arrays, chunk):
        e, w = _lines(cp, device, args)
        mass += cp.histogram(e/scale, bins=bins, weights=w)[0]
    return cp.asnumpy(mass)/np.diff(edges)


def angle_integrated_spectrum(samples, s_edges, *, n_angles=33, cone_factor=4.,
                              psi_pol=0., ellipticity=0., theta_xz=0., theta_yz=0.,
                              doppler="nominal", chunk=DEFAULT_CHUNK):
    """Midpoint integration over the same finite square cone as legacy delta.

    Particle chunks transfer once and are reused over the host direction loop.
    This is finite-aperture emission, not an exact full-solid-angle yield.
    """
    arrays, args = _inputs(samples, chunk, 1., 0., 0.,
                           psi_pol, ellipticity, theta_xz, theta_yz, doppler)
    if not isinstance(n_angles, Integral) or isinstance(n_angles, bool) or n_angles < 1 or not np.isfinite(cone_factor) or cone_factor <= 0:
        raise ValueError("positive integer n_angles and finite positive cone_factor required")
    edges = np.asarray(s_edges, dtype=float)
    _edges(edges)
    scale = 2*(1+np.cos(theta_xz)*np.cos(theta_yz))
    if scale <= 0:
        raise ValueError("normalized spectrum requires a positive nominal Doppler scale")
    cp = _cupy()
    bins = cp.asarray(edges)
    mass = cp.zeros(edges.size-1, dtype=cp.float64)
    if not arrays[0].size:
        return cp.asnumpy(mass)
    half = cone_factor/float(np.mean(arrays[0]))
    step = 2*half/n_angles
    offsets = -half+step*(np.arange(n_angles)+.5)
    cx, cy = np.mean(arrays[1]), np.mean(arrays[2])
    for _, device in _chunks(cp, arrays, chunk):
        for dx in offsets:
            for dy in offsets:
                e, w = _lines(cp, device, (cx+dx, cy+dy, *args[2:]))
                mass += cp.histogram(e/scale, bins=bins, weights=w)[0]
    return cp.asnumpy(mass)*step**2/np.diff(edges)


def single_electron_spectrum(samples, s, *, chunk=DEFAULT_CHUNK):
    """GPU linear head-on closed-form anchor, omitting nonlinear redshift.

    Scalar input returns a scalar; a one-dimensional grid returns that grid's shape.
    This historical anchor is not a direction-Doppler or nonlinear reference.
    """
    arrays, _ = _inputs(samples, chunk, 1., 0., 0., 0., 0., 0., 0., "nominal")
    values = np.asarray(s, dtype=float)
    if values.ndim > 1 or np.any(~np.isfinite(values)):
        raise ValueError("s must be a finite scalar or one-dimensional grid")
    cp = _cupy()
    grid = cp.asarray(np.atleast_1d(values))
    result = cp.zeros(grid.size, dtype=cp.float64)
    # Bound the particle-by-frequency temporary as well as the particle vectors.
    step = max(1, chunk//max(1, grid.size))
    for _, (g, lum) in _chunks(cp, (arrays[0], arrays[4]), step):
        g2 = g[:, None]**2
        y = grid[None, :]/g2
        shape = cp.where((y < 0) | (y > 1), 0., 1.5*(1-2*y*(1-y)))
        result += cp.sum(lum[:, None]*shape/g2, axis=0)
    host = cp.asnumpy(result)
    return host if values.ndim else host[0]

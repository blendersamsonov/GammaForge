"""Independent finite-bin emission-line reference for validation (RES074)."""
from __future__ import annotations

import numpy as np

from ...engines.xigma.stages import TrajectorySamples

__all__ = ["emission_lines", "bin_emission"]


def _ld_guard() -> None:
    if np.finfo(np.longdouble).precision < 18:
        raise RuntimeError("delta emission reference requires extended longdouble precision")


def _axes(psi, txz, tyz):
    psi, txz, tyz = map(np.longdouble, (psi, txz, tyz))
    c, s = np.cos(psi), np.sin(psi)
    e0 = np.array([c, s, 0], dtype=np.longdouble)
    e1 = np.array([-s, c, 0], dtype=np.longdouble)
    rx = np.array([[1,0,0],[0,np.cos(tyz),-np.sin(tyz)],[0,np.sin(tyz),np.cos(tyz)]], dtype=np.longdouble)
    ry = np.array([[np.cos(txz),0,np.sin(txz)],[0,1,0],[-np.sin(txz),0,np.cos(txz)]], dtype=np.longdouble)
    return ry @ rx @ e0, ry @ rx @ e1, ry @ rx @ np.array([0,0,-1], dtype=np.longdouble)


def emission_lines(samples: TrajectorySamples, theta_x: float, theta_y: float, *, photon_energy: float,
                   psi_pol: float = 0., ellipticity: float = 0., theta_xz: float = 0., theta_yz: float = 0.,
                   doppler: str = "nominal") -> tuple[np.ndarray, np.ndarray]:
    """Return per-particle resonant energies (erg) and shared-flux weights.

    ``photon_energy`` is the incident laser photon energy in CGS erg. ``nominal``
    uses the historical nominal-axis factor; ``particle`` uses exact finite speed;
    ``direction`` uses each electron's direction at beta=1, matching xigma (RES082).
    All three intentionally share Stage-0
    luminosity weights so this remains a bounded emission-reference check.
    """
    _ld_guard()
    if doppler not in ("nominal", "particle", "direction"):
        raise ValueError("doppler must be 'nominal', 'particle', or 'direction'")
    vals = [photon_energy, theta_x, theta_y, psi_pol, ellipticity, theta_xz, theta_yz]
    if not all(np.isfinite(v) for v in vals) or photon_energy <= 0 or not -1 <= ellipticity <= 1:
        raise ValueError("photon_energy/directions must be finite and positive; ellipticity must be in [-1,1]")
    gamma = np.asarray(samples.gamma, dtype=np.longdouble)
    tx = np.asarray(samples.theta_x, dtype=np.longdouble); ty = np.asarray(samples.theta_y, dtype=np.longdouble)
    ahat = np.asarray(samples.ahat(), dtype=np.longdouble); lum = np.asarray(samples.luminosity, dtype=np.longdouble)
    if not (gamma.ndim == tx.ndim == ty.ndim == ahat.ndim == lum.ndim == 1) or len({a.size for a in (gamma, tx, ty, ahat, lum)}) != 1:
        raise ValueError("sample fields must be same-shaped one-dimensional arrays")
    if any(np.any(~np.isfinite(a)) for a in (gamma, tx, ty, ahat, lum)) or np.any(gamma <= 1) or np.any(ahat < 0) or np.any(lum < 0):
        raise ValueError("samples contain invalid gamma, ahat, luminosity, or directions")
    n = np.array([theta_x, theta_y, 1.], dtype=np.longdouble); n /= np.linalg.norm(n)
    e0_raw, e1_raw, n0 = _axes(*map(np.longdouble, (psi_pol, theta_xz, theta_yz)))
    v = np.stack((tx, ty, np.ones_like(tx)), axis=1); v /= np.linalg.norm(v, axis=1)[:, None]
    direction_factor = 1 - np.sum(v * n0, axis=1)
    # Independently construct the physical transverse dipole basis per electron.
    p0 = e0_raw[None, :] - np.sum(v * e0_raw[None, :], axis=1)[:, None] * v
    p0_norm = np.sqrt(np.sum(p0 * p0, axis=1))
    if np.any(p0_norm <= np.longdouble("1e-12")):
        raise ValueError("transverse polarization projection is degenerate")
    e0 = p0 / p0_norm[:, None]
    e1 = np.cross(v, e0)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    orientation = np.sum(e1 * e1_raw[None, :], axis=1) < 0
    e1[orientation] *= -1
    v *= np.sqrt(1 - gamma[:, None] ** -2)
    r2 = (tx - theta_x) ** 2 + (ty - theta_y) ** 2
    d = 1 - np.sum(v * n, axis=1)
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("reference emission denominator is nonpositive")
    u0 = np.cross(n, np.cross(n - v, e0)) / d[:, None]
    u1 = np.cross(n, np.cross(n - v, e1)) / d[:, None]
    eps2 = np.longdouble(ellipticity) ** 2
    pol = (np.sum(u0*u0, axis=1) + eps2*np.sum(u1*u1, axis=1)) / (1 + eps2)
    den = 1 + ahat + gamma**2*r2
    c = (1 + np.cos(np.longdouble(theta_xz))*np.cos(np.longdouble(theta_yz))) / 2
    if doppler == "nominal":
        energies = 4*np.longdouble(photon_energy)*c*gamma**2/den
    elif doppler == "direction":
        energies = 2*np.longdouble(photon_energy)*direction_factor*gamma**2/den
    else:
        energies = 2*np.longdouble(photon_energy)*(1-np.sum(v*n0, axis=1))*gamma**2/den
    weights = (3/(2*np.pi))*lum*pol*gamma**2/(1+gamma**2*r2)**2
    if np.any(~np.isfinite(energies)) or np.any(~np.isfinite(weights)):
        raise ValueError("reference emission calculation produced non-finite values")
    return np.asarray(energies), np.asarray(weights)


def bin_emission(energies, weights, energy_edges) -> dict[str, object]:
    """Bin nonnegative erg energies/weights, retaining tails and bin-average density."""
    e = np.asarray(energies, dtype=float); w = np.asarray(weights, dtype=float); edges = np.asarray(energy_edges, dtype=float)
    if e.shape != w.shape or np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("energies and weights must be finite, nonnegative, same-shaped arrays")
    if e.ndim != 1 or w.ndim != 1 or edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("energy_edges must be finite and strictly increasing")
    mass, _ = np.histogram(e, bins=edges, weights=w)
    return {"bin_mass": mass, "density": mass / np.diff(edges), "underflow": float(w[e < edges[0]].sum()),
            "overflow": float(w[e > edges[-1]].sum()), "total_weight": float(w.sum())}

"""Conservative finite-energy-bin comparisons for the delta reference.

This module deliberately has no dependency on production spectral machinery.  A
candidate is a callable density ``f(energy)``; its values are integrated in
each supplied physical-energy bin, rather than sampled at bin centres.
"""
from __future__ import annotations

import numpy as np

__all__ = ["integrate_density_bins", "compare_emission_bins"]


def _edges(edges):
    a = np.asarray(edges, dtype=float)
    if a.ndim != 1 or a.size < 2 or np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError("energy_edges must be finite, positive, and strictly increasing")
    return a


def _density_values(density, x):
    y = np.asarray(density(x), dtype=float)
    if y.shape != x.shape:
        raise ValueError("density callable must return the same shape as energy nodes")
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("density must be finite and nonnegative")
    return y


def _integrate(density, edges, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lo, hi = edges[:-1], edges[1:]
    x = (lo[:, None] + hi[:, None]) / 2 + (hi[:, None] - lo[:, None]) * nodes / 2
    y = _density_values(density, x.ravel()).reshape(x.shape)
    scale = (hi - lo) / 2
    mass = scale * (y @ weights)
    moment = scale * ((y * x) @ weights)
    if np.any(~np.isfinite(mass)) or np.any(~np.isfinite(moment)):
        raise ValueError("integrated density contains non-finite values")
    return mass, moment


def integrate_density_bins(density, energy_edges, *, quadrature_order=8, refinement_order=None):
    """Integrate a nonnegative physical-energy density over finite bins.

    Returns bin ``mass`` and first energy ``moment`` (neither is a midpoint
    approximation), plus a refinement metric.  If ``refinement_order`` is not
    supplied it is twice ``quadrature_order``.  ``converged`` is informational
    and never silently turns an unresolved calculation into a pass.
    """
    edges = _edges(energy_edges)
    if not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    q2 = 2 * int(quadrature_order) if refinement_order is None else refinement_order
    if not isinstance(q2, (int, np.integer)) or q2 <= quadrature_order:
        raise ValueError("refinement_order must be an integer greater than quadrature_order")
    mass, moment = _integrate(density, edges, int(quadrature_order))
    refined_mass, refined_moment = _integrate(density, edges, int(q2))
    total = float(refined_mass.sum())
    coarse_total = float(mass.sum())
    moments = (float(moment.sum()), float(refined_moment.sum()))
    if not np.all(np.isfinite([total, coarse_total, *moments])):
        raise ValueError("integrated density totals contain non-finite values")
    if total > 0 and coarse_total > 0:
        yield_error = abs(coarse_total - total) / total
        mass_error = float(np.abs(mass - refined_mass).sum() / total)
        centroid_error = abs((moments[0] / coarse_total) / (moments[1] / total) - 1)
    elif total == coarse_total == 0:
        yield_error = mass_error = centroid_error = 0.0
    else:
        yield_error = mass_error = centroid_error = None
    errors = {"yield": yield_error, "l1": mass_error, "centroid": centroid_error}
    # One third of the provisional outer budgets in RES074; not scientific acceptance.
    convergence_thresholds = {"yield": 0.03 / 3, "l1": 0.05 / 3, "centroid": 0.01 / 3}
    converged = all(value is not None and value <= convergence_thresholds[key]
                    for key, value in errors.items())
    return {"bin_mass": mass, "first_moment": moment, "refined_bin_mass": refined_mass,
            "refined_first_moment": refined_moment,
            "refinement_error": max(errors.values()) if all(v is not None for v in errors.values()) else None,
            "refinement_errors": errors, "converged": bool(converged),
            "convergence_thresholds": convergence_thresholds,
            "quadrature_order": int(quadrature_order), "refinement_order": int(q2)}


def compare_emission_bins(density, energy_edges, line_energies, line_weights, *, quadrature_order=8,
                          refinement_order=None):
    """Compare a smooth candidate density with finite-window delta emission lines."""
    edges = _edges(energy_edges)
    e, w = np.asarray(line_energies, dtype=float), np.asarray(line_weights, dtype=float)
    if e.ndim != 1 or w.shape != e.shape or np.any(~np.isfinite(e)) or np.any(~np.isfinite(w)) or np.any(e < 0) or np.any(w < 0):
        raise ValueError("line energies and weights must be finite, nonnegative, same-shaped 1-D arrays")
    ref_mass = np.histogram(e, edges, weights=w)[0]
    ref_moment = np.histogram(e, edges, weights=e * w)[0]
    if np.any(~np.isfinite(ref_mass)) or np.any(~np.isfinite(ref_moment)):
        raise ValueError("reference histogram contains non-finite values")
    candidate = integrate_density_bins(density, edges, quadrature_order=quadrature_order,
                                       refinement_order=refinement_order)
    cm, ct = candidate["refined_bin_mass"], candidate["refined_first_moment"]
    if np.any(~np.isfinite(cm)) or np.any(~np.isfinite(ct)):
        raise ValueError("integrated candidate contains non-finite values")
    total_ref, total_cand = float(ref_mass.sum()), float(cm.sum())
    if not np.all(np.isfinite([total_ref, total_cand, ref_moment.sum(), ct.sum(), w.sum()])):
        raise ValueError("emission totals contain non-finite values")
    centroid_ref = float(ref_moment.sum() / total_ref) if total_ref else None
    centroid_cand = float(ct.sum() / total_cand) if total_cand else None
    l1 = float(np.abs(cm - ref_mass).sum() / total_ref) if total_ref else None
    rel_yield = float((total_cand - total_ref) / total_ref) if total_ref else None
    cand_zero = float(cm[ref_mass == 0].sum())
    under = float(w[e < edges[0]].sum()); over = float(w[e > edges[-1]].sum())
    return {"candidate": candidate, "reference_bin_mass": ref_mass, "reference_first_moment": ref_moment,
            "candidate_total": total_cand, "reference_total": total_ref, "absolute_yield_error": total_cand - total_ref,
            "relative_yield_error": rel_yield, "l1_mass_error": l1,
            "candidate_centroid": centroid_cand, "reference_centroid": centroid_ref,
            "centroid_relative_error": ((centroid_cand - centroid_ref) / centroid_ref if centroid_ref and centroid_cand is not None else None),
            "candidate_mass_in_reference_zero_bins": cand_zero, "reference_underflow": under,
            "reference_overflow": over,
            "zero_reference_with_candidate_signal": bool(total_ref == 0 and total_cand > 0)}

"""Focused and flying-focus deterministic source maps (DER019 §11, §12).

The tiers so far freeze the laser's transverse width. That is the approximation DER019 §11
identifies as the real boundary: **Stage-0 quantities are nonlinear ratios attached to
individual electron trajectories**, and their distribution cannot be recovered by taking
ratios of the global DER001 overlap integrals — doing so would form the ratio before
integrating, which is a different quantity.

So this module keeps a small set of deterministic electron source labels and evaluates the
DER016 trajectory integrals at each node. It is a *deterministic quadrature over source
coordinates*, not a macroparticle sample: the nodes and weights come from Gauss-Hermite on
the bunch's own Gaussian, so nothing here scales with a particle count.

**What is and is not reimplemented.** The per-trajectory integrals call
`xigma.stages.integrate_trajectories` on a one-particle `Bunch` per node, so the trajectory
physics is Xigma's, not a second copy. That is a deliberate exception to this package's
independence from Xigma (asserted for the *spectrum* in `tests/test_analytical.py`): the
thing being validated here is the deterministic source *reduction*, not the trajectory
moment definitions, which DER016 already fixes and which duplicating would only invite to
drift. The source map's own claim — that its quadrature reproduces the luminosity — is
checked against `formulas.overlap_yield` in the tests, which is a genuinely independent
oracle.

**The acceptance criterion that matters** (DER019 §11.1, §12.1): the source-integrated
luminosity must agree with DER001/DER002's `overlap_yield` for the same geometry. That is an
unusually strong check for a semi-analytical model — a closed form validating a quadrature —
and it is asserted before the spectra are considered at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from ...io.bunch import Bunch, GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser
from ...io.units import Quantity

__all__ = ["SourceMap", "build_source_map"]


@dataclass(frozen=True)
class SourceMap:
    """A prepared deterministic source map: one trajectory evaluation per node.

    Arrays are parallel and carry no particle dimension. ``luminosity`` sums to the total
    yield, which is the property DER019 §11.1 requires before any spectrum built on this map
    is accepted.
    """

    #: Transverse source labels in CGS cm.
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    #: Gauss-Hermite weights, normalized so they sum to one over the whole map.
    weight: np.ndarray
    #: Per-node photon weight; ``sum(luminosity)`` is the total yield (DER001/DER002).
    luminosity: np.ndarray
    #: DER016 trajectory quantities at each node.
    a_shape: np.ndarray
    c_bar: np.ndarray
    var_a_shape: np.ndarray
    intensity_peak: float
    #: Gaussian-label quadrature dimension: 2 for round/head-on (R, z0), 3 otherwise.
    dimension: int

    @property
    def total_luminosity(self) -> float:
        return float(np.sum(self.luminosity))

    def as_metadata(self) -> dict:
        return {
            "nodes": int(self.x0.size),
            "dimension": int(self.dimension),
            "total_luminosity": self.total_luminosity,
            "mean_a_shape": float(np.sum(self.weight * self.a_shape)),
            "rms_a_shape": float(np.sqrt(np.sum(self.weight * (self.a_shape - np.sum(self.weight * self.a_shape)) ** 2))),
            "intensity_peak": float(self.intensity_peak),
        }


def _one_particle_bunch(
    beam: GaussianElectronBeam, x0: float, y0: float, z0: float, weight: float
) -> Bunch:
    """A single-node `Bunch` carrying exactly one deterministic electron label.

    The particle is placed exactly at its quadrature node rather than drawn, which is what
    makes the whole construction particle-free: this is a batch of one, not a sample. The
    weight is carried through as the bunch's relative weight so the usual `N_e` scaling in
    `integrate_trajectories` applies unchanged.
    """
    zeros = np.zeros(1)
    return Bunch(
        x=np.array([x0]),
        y=np.array([y0]),
        z=np.array([z0]),
        thx=zeros.copy(),
        thy=zeros.copy(),
        gamma=np.array([beam.gamma0()]),
        weight=np.array([weight]),
        meta={"source_map_node": True},
    )


def build_source_map(
    beam: GaussianElectronBeam,
    laser: GaussianParaxialLaser,
    n_electrons: float,
    *,
    n_transverse: int = 12,
    n_longitudinal: int = 12,
    n_steps: int = 256,
    threshold: float = 1e-4,
) -> SourceMap:
    """Prepare the deterministic source map for one focused collision.

    Nodes are Gauss-Hermite on the bunch's own Gaussian labels, so the map integrates the
    same distribution `overlap_yield` does while keeping each node's trajectory intact.
    Transverse symmetry decides the dimension, following DER019 §11.1 vs §11.2:

    * round and head-on — cylindrical symmetry leaves ``(R, z0)``: implemented as a 2D
      transverse map, reported as ``dimension == 2``;
    * elliptical, astigmatic, offset or crossed — ``(x0, y0, z0)``, ``dimension == 3``.

    Both are built here with a full 3D node set; ``dimension`` records the symmetry the
    physics admits so a caller can see whether the map is carrying redundant nodes. The
    distinction matters for cost, and DER019 §14 is explicit that beyond three source
    dimensions the deterministic route stops being the cheaper one.

    ``n_steps``/``threshold`` are the trajectory-integration knobs and are passed through
    unchanged; they set the per-node quadrature accuracy, not the source quadrature.

    .. warning::

       **Node count is set by the spot-to-bunch ratio, not by taste.** When the laser's
       transverse width is much *smaller* than the bunch's, the luminosity profile in the
       normalized node variable becomes a narrow Gaussian and Gauss-Hermite converges slowly.
       Measured for a 10 um bunch against a 1 um spot (10:1): the total luminosity reaches
       1e-3 only around ``n_transverse ~ 40`` and is still ~2% low there, versus 1e-5 at ten
       nodes for a matched spot.

       That is a property of the integrand, not a defect in the map, and it is the same
       regime where this tier stops earning its cost — DER019 §14 notes the deterministic
       route losing its advantage as dimensionality grows. The tests assert convergence at a
       matched spot and assert only the *trend* in the anisotropic regime, rather than
       pretending a fixed node count suffices.
    """
    if n_transverse < 2 or n_longitudinal < 2:
        raise ValueError(
            f"build_source_map needs at least 2 nodes per axis, got "
            f"n_transverse={n_transverse}, n_longitudinal={n_longitudinal}"
        )

    transverse_nodes, transverse_weights = hermegauss(n_transverse)
    longitudinal_nodes, longitudinal_weights = hermegauss(n_longitudinal)

    # `hermegauss` nodes are already standard normal and its weights already carry
    # `exp(-z^2/2)`; only the `1/sqrt(2 pi)` is missing. So the labels are `sigma * node`
    # with **no** further rescaling, and the weights are divided once.
    #
    # Rescaling the nodes by `sqrt(2)` here is a silent, plausible-looking mistake: the weight
    # sum stays 1 and `sum(w) == 1` still passes, but `E[x^2]` becomes 2 and the map quietly
    # integrates the wrong Gaussian — worth 25% of the luminosity here. The tests assert
    # `E[x^2] == 1` on these nodes precisely because that check is the only thing standing
    # between this and a plausible wrong answer.
    transverse_weights = transverse_weights / math.sqrt(2.0 * math.pi)
    longitudinal_weights = longitudinal_weights / math.sqrt(2.0 * math.pi)

    sigma_x, sigma_y, sigma_z = beam.m("sigma_x"), beam.m("sigma_y"), beam.m("sigma_z")

    # Tensor nodes over (x0, y0, z0).
    grid_x, grid_y, grid_z = np.meshgrid(
        sigma_x * transverse_nodes,
        sigma_y * transverse_nodes,
        sigma_z * longitudinal_nodes,
        indexing="ij",
    )
    # The weight tensor must be (n, n, n) to match the node grids. Built with `einsum`
    # rather than chained broadcasting: `w[:, None] * w[None, :, None] * w[None, None, :]`
    # broadcasts to shape (1, n, n) — the leading axis collapses and `z0` ends up carrying a
    # single node's weight, which silently costs a fixed fraction of the luminosity (25% at
    # n = 10) rather than failing.
    weight_grid = np.einsum("i,j,k->ijk", transverse_weights, transverse_weights, longitudinal_weights)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    flat_z = grid_z.ravel()
    flat_weight = weight_grid.ravel()

    from ..xigma.stages import integrate_trajectories

    count = flat_x.size
    luminosity = np.empty(count)
    a_shape = np.empty(count)
    var_a_shape = np.empty(count)
    c_bar = np.empty(count)
    intensity_peak = 0.0

    for index in range(count):
        node = _one_particle_bunch(
            beam, float(flat_x[index]), float(flat_y[index]), float(flat_z[index]),
            float(flat_weight[index]),
        )
        samples = integrate_trajectories(
            node, laser, n_electrons, n_steps=n_steps, threshold=threshold
        )
        luminosity[index] = float(samples.luminosity[0])
        a_shape[index] = float(samples.a0_shape[0])
        var_a_shape[index] = float(samples.var_a_shape[0])
        c_bar[index] = float(samples.chirp_mean[0])
        intensity_peak = float(samples.intensity_peak)

    round_and_head_on = (
        abs(beam.m("sigma_x") - beam.m("sigma_y")) < 1e-30
        and abs(laser.m("sigma_x") - laser.m("sigma_y")) < 1e-30
        and abs(laser.m("theta_xz")) < 1e-30
        and abs(laser.m("theta_yz")) < 1e-30
        and abs(laser.m("psi_focus")) < 1e-30
    )
    return SourceMap(
        x0=flat_x,
        y0=flat_y,
        z0=flat_z,
        weight=flat_weight,
        luminosity=luminosity,
        a_shape=a_shape,
        c_bar=c_bar,
        var_a_shape=var_a_shape,
        intensity_peak=intensity_peak,
        dimension=2 if round_and_head_on else 3,
    )

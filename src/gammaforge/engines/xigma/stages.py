"""xigma's pipeline as composable pure functions (GRAND_PLAN.md §4.2).

Stage 0 — :func:`integrate_trajectories` — is what this module holds today. Stages 1 and 2
(H-table deposition and the spectrum kernels) arrive in Phase 3a; Stage 0 is pulled
forward because delta needs it (§4.5).

**Every stage is a pure function.** State lives in the `Collision` facade (Phase 3a), not
here, so validation can call these directly and a stage can be reasoned about without
knowing what cached it.

**The engine calls the laser; it does not model it** (§4.2, P15). Stage 0 samples
``a0_profile`` along each trajectory through the `LaserField` protocol and needs nothing
else from it — no envelope formula, no Gaussian assumption, no spot sizes. The photon
density it needs follows from ``a0`` by inverting the same energy→a0 chain the laser used
to produce it (see :data:`PHOTON_DENSITY_PER_A0_SQUARED`), so a future non-Gaussian
`LaserField` drops in with no change here.

**No coordinate normalization** (§2.1, `DECISIONS.md` D015). The predecessor worked in
``k0_las``-normalized coordinates, where a factor ``k0**2`` in the per-step contribution
was exactly what replaced ``c``. Here everything is CGS and the ``c`` is simply ``c``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import Bunch, overlap_time_window
from ...io.laser import LaserField, fit_gaussian_paraxial
from ...io.units import C_CGS, E_ESU, HBAR_CGS, ME_CGS, SIGMA_T_CGS
from .chunking import run_in_chunks

__all__ = [
    "TrajectorySamples",
    "integrate_trajectories",
    "photon_density_scale",
    "RELATIVE_VELOCITY",
    "BYTES_PER_PARTICLE_STEP",
]

#: Relative-velocity factor for the near-backscattering geometry: electron and photon
#: approach at ``2c``, so the flux an electron sees is twice what its own speed implies.
#: Head-on is the only geometry whose physics is wired (§9.3 is an open derivation), and
#: this constant is where a crossing angle will enter when it lands.
RELATIVE_VELOCITY = 2.0

#: Live bytes per (particle x step) in Stage 0's inner loop, for auto-chunking.
#: Calibrated on the predecessor's equivalent kernel on a GTX 1660 Ti (6 GB): 400,000
#: particles at 64 steps fit, 800,000 did not, with the failing allocation reported at
#: 409,600,000 bytes — about 113 bytes per particle-step, consistent with roughly fourteen
#: concurrently-live float64 temporaries. Rounded up for headroom against other processes
#: on the device.
BYTES_PER_PARTICLE_STEP = 200


def photon_density_scale(laser: LaserField) -> float:
    """Photons per cm^3 per unit ``a0**2`` — the inverse of the laser's energy→a0 chain.

    ``a0 = e sqrt(8 pi U density) / (m_e c omega0)`` with ``density`` the normalized photon
    envelope and ``U`` the pulse energy, and the physical photon density is
    ``N_l * density = (U / hbar omega0) * density``. Solving the first for ``density`` and
    substituting, the pulse energy **cancels**::

        n_photons(r, t) = a0(r, t)**2 * (m_e c)**2 omega0 / (8 pi hbar e**2)

    which is why Stage 0 can take the whole laser through ``a0_profile`` alone. Only
    ``omega0`` remains, and that comes from the descriptive fit (§3.3), not from assuming
    the field is Gaussian.

    Linear polarization, like everything downstream of §9.2's unwritten derivation.
    """
    omega0 = fit_gaussian_paraxial(laser).omega0()
    return (ME_CGS * C_CGS) ** 2 * omega0 / (8.0 * math.pi * HBAR_CGS * E_ESU**2)


@dataclass(frozen=True)
class TrajectorySamples:
    """One sample per macroparticle, ready for Stage 1 deposition. CGS.

    ``gamma``/``theta_x``/``theta_y`` pass straight through from the bunch: the pusher is
    ballistic, so a particle's energy and angles do not change along its trajectory.

    ``luminosity`` is the plan's per-particle ``L`` — the photon weight this macroparticle
    deposits, integrated over its passage through the pulse. Summing it is the total yield.

    ``a0_shape`` is **not** the trajectory-averaged effective intensity ``ahat``; it is
    ``ahat``'s a0-independent shape factor, ``ahat = a0_peak**2 * a0_shape``, computed
    without reference to any actual a0. That is what lets one Stage 0 run be retargeted to
    a different pulse energy without rerunning it (§5's `REUSE_INTERMEDIATES` tier).

    It is one scalar per particle rather than a per-timestep distribution, and that is
    physics, not an optimization: in this weakly nonlinear regime the photon formation
    length spans the whole trajectory, so — unlike synchrotron radiation — the trajectory
    may **not** be split into independently radiating segments. The predecessor learned
    this the hard way and left a warning against going back to per-timestep a0.
    """

    gamma: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    a0_shape: np.ndarray
    luminosity: np.ndarray
    a0_peak: float
    n_steps: int

    @property
    def n_particles(self) -> int:
        return int(self.gamma.shape[0])

    def total_yield(self) -> float:
        return float(np.sum(self.luminosity))

    def ahat(self) -> np.ndarray:
        """The physical trajectory-averaged effective intensity, at this pulse's own a0."""
        return self.a0_peak**2 * self.a0_shape

    def retargeted_ahat(self, a0_peak: float) -> np.ndarray:
        """``ahat`` as it would be for a pulse of a different peak a0, no rerun needed."""
        return a0_peak**2 * self.a0_shape


def integrate_trajectories(
    bunch: Bunch,
    laser: LaserField,
    n_electrons: float,
    *,
    n_steps: int = 200,
    threshold: float = 1e-3,
    backend: str = "numpy",
    chunk: int | None = None,
) -> TrajectorySamples:
    """Stage 0: push every macroparticle through the pulse and sample the overlap.

    Each particle travels a straight line at ``c`` (§2.3) across its own overlap window —
    the closed-form window `gammaforge.io.bunch.overlap_time_window` derives from the
    laser's active region, so a particle is integrated over exactly the interval in which
    it can see the pulse and no longer. The integral is a midpoint sum over ``n_steps``.

    ``n_electrons`` is the interaction's ``N_e``. The bunch's own weights are *relative*
    (§3.2), so absolute photon counts enter here and only here — which is what makes every
    output exactly linear in charge and a charge edit a pure rescale (§5).

    ``threshold`` is the same active-region fraction the prefilter uses. Passing a smaller
    value integrates a longer window; results converge as it shrinks, and the prefilter
    invariance of §7 requires that a particle excluded at this threshold contributes
    nothing at it.

    Chunking is over particles, whose trajectories are independent, so the partition
    cannot change the answer.
    """
    if n_steps < 1:
        raise ValueError(f"integrate_trajectories: n_steps must be >= 1, got {n_steps}")
    _check_backend(backend)

    t0, t1 = overlap_time_window(bunch, laser, threshold)
    span = np.maximum(0.0, t1 - t0)
    # A particle that never enters the pulse gets an empty window, which
    # `overlap_time_window` reports as t0 = +inf, t1 = -inf. Its span is zero and it
    # contributes nothing — but `inf + 0 * 0` is still `inf`, and a trajectory evaluated
    # there hands the laser a NaN that propagates into the sum for *every* particle. So
    # empty windows are anchored at a finite time, and the zero span does the rest.
    # (With the prefilter on such particles are already gone; with it off they are not,
    # which is how the §7 prefilter-invariance property found this.)
    start = np.where(span > 0.0, t0, 0.0)
    # Midpoint rule: no sample sits on the window edge, where the integrand is smallest and
    # the window definition is least meaningful.
    offsets = (np.arange(n_steps) + 0.5) / n_steps

    a0_peak = fit_gaussian_paraxial(laser).a0_peak()
    density_scale = photon_density_scale(laser)
    # Absolute photons per macroparticle-second of overlap. The bunch's weights are
    # relative and sum to 1 over the *unfiltered* population, so scaling by N_e here keeps
    # a prefiltered run and a full run identical (§3.2).
    weight = n_electrons * bunch.weight

    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    velocity = (C_CGS * bunch.thx / norm, C_CGS * bunch.thy / norm, C_CGS / norm)

    def integrate(first_index: int, last_index: int):
        sl = slice(first_index, last_index)
        times = start[sl, None] + offsets[None, :] * span[sl, None]
        a0 = np.asarray(
            laser.a0_profile(
                bunch.x[sl, None] + velocity[0][sl, None] * times,
                bunch.y[sl, None] + velocity[1][sl, None] * times,
                bunch.z[sl, None] + velocity[2][sl, None] * times,
                times,
            )
        )
        # The photon density an electron flies through, and the rate it scatters at.
        # In CGS this is simply flux x cross-section x time; the predecessor's k0**2 was
        # the Jacobian of its coordinate normalization and has no counterpart here (D015).
        dt = span[sl] / n_steps
        rate = RELATIVE_VELOCITY * density_scale * C_CGS * SIGMA_T_CGS
        luminosity = rate * weight[sl] * dt * np.sum(a0**2, axis=1)

        # `ratio` is the local photon density as a fraction of the pulse's peak — equally,
        # (a0/a0_peak)**2. a0_shape is its second moment over the trajectory, normalized
        # by its first: the intensity an electron *effectively* experiences, weighted by
        # where it actually radiated.
        ratio = (a0 / a0_peak) ** 2
        moment_1 = np.sum(ratio, axis=1)
        # A particle with no window saw nothing, whatever the envelope reads at the
        # anchor time above; its effective intensity is zero, not the value at t = 0.
        usable = (moment_1 > 0.0) & (dt > 0.0)
        a0_shape = np.where(
            usable, np.sum(ratio**2, axis=1) / np.maximum(moment_1, 1e-300), 0.0
        )
        return luminosity, a0_shape

    parts = run_in_chunks(
        bunch.n_particles,
        integrate,
        chunk=chunk,
        # No ceiling: the predecessor's measured chunk ceiling was for the *spectrum*
        # path's s-axis, where per-launch overhead amortized by ~8-16. Stage 0 partitions
        # particles, where no such measurement exists, and inventing one would be the
        # cargo-culting the chunking module's own docstring warns the constants against.
        bytes_per_item=BYTES_PER_PARTICLE_STEP * n_steps,
        backend=backend,
    )
    # An empty bunch is reachable, not hypothetical: the prefilter discards every particle
    # for a mistimed pulse or a bunch far wider than the spot. `np.concatenate([])` raises,
    # which would make the prefilter turn a zero yield into an exception — the opposite of
    # the pure optimization §3.2 promises.
    luminosity = np.concatenate([part[0] for part in parts]) if parts else np.zeros(0)
    a0_shape = np.concatenate([part[1] for part in parts]) if parts else np.zeros(0)

    return TrajectorySamples(
        gamma=bunch.gamma,
        theta_x=bunch.thx,
        theta_y=bunch.thy,
        a0_shape=a0_shape,
        luminosity=luminosity,
        a0_peak=a0_peak,
        n_steps=n_steps,
    )


def _check_backend(backend: str) -> None:
    """Stage 0 runs on numpy today; the other two backends land with the kernels (3a).

    Not an oversight, and deliberately an error rather than a silent fallback. P15 puts
    the laser in charge of its own sampling, so a GPU Stage 0 needs
    `LaserField.a0_profile` to be array-module-agnostic — a change to the *protocol's*
    contract that every future implementation inherits, not something Stage 0 can arrange
    on its own by wrapping the call. Accepting ``backend='cupy'`` here and running on the
    host anyway would make the §7 backend-agreement leg pass while comparing numpy with
    numpy, which is worse than not having it.
    """
    if backend != "numpy":
        raise NotImplementedError(
            f"integrate_trajectories(backend={backend!r}): Stage 0 is numpy-only until "
            f"Phase 3a. A GPU path needs LaserField.a0_profile to accept device arrays "
            f"(P15/§3.3), which is a protocol change, not a wrapper here."
        )

"""delta: brute-force per-macroparticle resonance binning (GRAND_PLAN.md §4.5).

A validation reference, never a registered engine and never in the GUI's model list. It
takes xigma's Stage 0 output and computes a spectrum the most direct way there is: work
out the photon energy each macroparticle resonates at when viewed from a given direction,
weight it by the bare differential cross-section, and histogram it. No table, no
interpolation, no importance sampling — nothing to be subtly wrong in the same way as the
thing it checks.

**What it can and cannot arbitrate.** delta reuses Stage 0, so it shares the trajectory
integration and the laser sampling with xigma and is blind to any error living there. It
is an independent check on the **Stage-2 kernel normalization** and nothing more — which
is exactly the job §9.1 needs done, and why a minimal delta is built in Phase 2.5 rather
than waiting for its full cross-validation role in Phase 5.

**Conventions.** ``s`` is the normalized photon energy, ``E = 4 hbar omega0 s``, so a
head-on electron of Lorentz factor ``gamma`` has its Compton edge at ``s = gamma**2``.
Angles are the small-angle observation direction ``(theta_x, theta_y)`` in rad, and
``dOmega = dtheta_x dtheta_y`` to the same order.

**The 2 pi (§9.1) — traced, then closed.** The predecessor recorded that this method's
angle-integrated total ran "consistently ~6.3x" its table-free spectrum, "suspiciously
close to 2*pi, not yet explained". It is not close to ``2 pi``; it is exactly ``2 pi``,
and the integral is elementary. For one macroparticle of weight ``L``, with
``u = gamma**2 r**2``, the azimuthal average ``<a_fac> = 1 - 2u/(1 + u)**2`` and the
paper's own prefactor of ``3``::

    int dOmega  3 L gamma**2 <a_fac> / (1 + u)**2
        = 3 pi L int_0^inf du [ 1/(1+u)**2 - 2u/(1+u)**4 ]
        = 3 pi L [ 1 - 2*(1/6) ]
        = 2 pi L

That is ``2 pi`` times the photon count Stage 0 assigned the particle, not the count
itself, and the two lines above are the whole proof that the paper's eq. *(xsec)* is short
a ``1/(2 pi)`` (RES026). :data:`DIFFERENTIAL_PREFACTOR` now carries that correction, so this
module computes the *corrected* physics rather than the equation as typeset — see RES033 for
why a reference implementation follows the derivation and not the typo, and
:func:`check_normalization`, which as of Phase 3b expects **one**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...engines.xigma.stages import TrajectorySamples

__all__ = [
    "resonance_spectrum",
    "angle_integrated_spectrum",
    "single_electron_spectrum",
    "NormalizationCheck",
    "captured_fraction",
    "check_normalization",
    "DEFAULT_CONE_FACTOR",
    "DIFFERENTIAL_PREFACTOR",
]

#: The bare differential cross-section's prefactor, ``3 / (2 pi)`` (§9.1, RES026/RES033).
#: The paper's eq. *(xsec)* reads ``3`` and is short a factor ``1/(2 pi)``; the module
#: docstring's two elementary integrals are the derivation. delta implements the corrected
#: value so that its angle-integral is a photon count — the same number Stage 0 counts
#: directly — rather than ``2 pi`` of them. `engines.xigma.stages`'s
#: ``KERNEL_NORMALIZATION_CONSTANT`` carries the identical correction for the table kernel,
#: which is why the two remain directly comparable.
DIFFERENTIAL_PREFACTOR = 3.0 / (2.0 * math.pi)

#: How far out to integrate the angular grid, in units of the ``1/gamma`` radiation cone.
#: The tail converges slowly — four cone widths hold about **92%** of the yield, not "all
#: but a fraction of a percent" as this once claimed; see :func:`captured_fraction`, which
#: computes the shortfall in closed form so it is corrected rather than assumed away.
#: Four is a cost compromise: the grid is a double loop, and doubling the reach at fixed
#: resolution quadruples the work to recover the next 6%.
DEFAULT_CONE_FACTOR = 4.0


def resonance_spectrum(
    samples: TrajectorySamples,
    s_edges: np.ndarray,
    theta_x: float,
    theta_y: float,
    psi_pol: float = 0.0,
) -> np.ndarray:
    """``d3N / (ds dOmega)`` seen from the direction ``(theta_x, theta_y)``.

    Each macroparticle radiates at one energy in this direction — its resonance,
    ``s_res = gamma**2 / (1 + ahat + gamma**2 r**2)`` with ``r`` the angle between the
    particle's own direction and the observer's. The nonlinear redshift enters through
    ``ahat``, the trajectory-averaged intensity Stage 0 already produced.

    The weight is the **bare** differential cross-section, ``gamma**2 / (1 + r**2
    gamma**2)**2`` times the polarization factor. Not the ``gamma**5`` form a table-based
    kernel uses: those extra powers are a ``|dGamma/domega|`` Jacobian for evaluating a
    *smooth, already-binned* distribution at an interpolated gamma, and there is nothing
    interpolated here — every particle contributes at its own exact gamma. For the same
    reason no ``1 / (1 + ahat)`` Jacobian appears: that one comes from the ensemble
    gamma-integral collapse a table lookup performs, and delta never performs it.

    The histogram is returned as a density in ``s``, so integrating it over ``s`` and over
    solid angle gives a photon count directly comparable with Stage 0's own total — and
    since Phase 3b, equal to it, because :data:`DIFFERENTIAL_PREFACTOR` carries §9.1's
    ``1/(2 pi)``. Before that correction this integrated to ``2 pi`` times the count.
    """
    gamma = samples.gamma
    ahat = samples.ahat()
    delta_x = samples.theta_x - theta_x
    delta_y = samples.theta_y - theta_y
    r_squared = delta_x**2 + delta_y**2
    gamma_squared = gamma**2

    s_res = gamma_squared / (1.0 + ahat + gamma_squared * r_squared)

    lorentz = 1.0 / (1.0 + r_squared * gamma_squared) ** 2
    cos_polarization = np.cos(psi_pol - np.arctan2(delta_y, delta_x)) ** 2
    polarization = 1.0 - 4.0 * cos_polarization * r_squared * gamma_squared * lorentz

    weights = DIFFERENTIAL_PREFACTOR * samples.luminosity * polarization * gamma_squared * lorentz

    s_edges = np.asarray(s_edges, dtype=float)
    histogram, _ = np.histogram(s_res, bins=s_edges, weights=weights)
    return histogram / np.diff(s_edges)


def angle_integrated_spectrum(
    samples: TrajectorySamples,
    s_edges: np.ndarray,
    *,
    n_angles: int = 33,
    cone_factor: float = DEFAULT_CONE_FACTOR,
    psi_pol: float = 0.0,
) -> np.ndarray:
    """``dN/ds``: :func:`resonance_spectrum` summed over a grid of viewing directions.

    A midpoint Riemann sum over ``(theta_x, theta_y)`` cells of area ``dOmega``. Crude on
    purpose — a quadrature scheme clever enough to be efficient is clever enough to be
    wrong in a way that looks like the answer, and this function's whole value is that
    every step of it can be checked by hand.

    The grid spans ``cone_factor / gamma`` about the beam's mean direction, which is where
    essentially all the radiation is; :func:`check_normalization` reports how much was left
    outside rather than assuming none was.
    """
    half_width = cone_factor / float(np.mean(samples.gamma))
    centre_x, centre_y = float(np.mean(samples.theta_x)), float(np.mean(samples.theta_y))
    step = 2.0 * half_width / n_angles
    offsets = -half_width + step * (np.arange(n_angles) + 0.5)

    total = np.zeros(len(np.asarray(s_edges)) - 1, dtype=float)
    for dx in offsets:
        for dy in offsets:
            total += resonance_spectrum(
                samples, s_edges, centre_x + dx, centre_y + dy, psi_pol
            )
    return total * step * step


def single_electron_spectrum(samples: TrajectorySamples, s) -> np.ndarray:
    """``dN/ds``, angle-integrated in closed form — the anchor delta is checked against.

    A single electron's angle-integrated spectral shape depends only on its own gamma:
    ``1.5 * (1 - 2y(1 - y))`` for ``y = s / gamma**2`` in ``[0, 1]``, zero outside. That
    shape integrates to exactly 1 over ``y``, so **this spectrum's integral over ``s`` is
    exactly the sum of Stage 0's per-particle luminosities** — an identity, not a
    tolerance, and the reason it is the right thing to compare an angular integral against.

    The nonlinear redshift is deliberately absent: this is the linear-Compton shape, and
    at the ``a0 <~ 0.4`` of the scenario bank it is the leading behaviour. Where that
    matters it is a stated approximation of the anchor, not a claim about the physics.
    """
    s_values = np.atleast_1d(np.asarray(s, dtype=float))
    gamma_squared = (samples.gamma**2)[:, None]
    y = s_values[None, :] / gamma_squared
    shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    spectrum = np.sum(samples.luminosity[:, None] * shape / gamma_squared, axis=0)
    return spectrum if np.ndim(s) else spectrum[0]


@dataclass(frozen=True)
class NormalizationCheck:
    """What delta says about the Stage-2 normalization, and how much to trust it.

    ``ratio`` is delta's angle-integrated photon count divided by Stage 0's own total
    yield. It must be **one**, before truncation: two paths counting the same photons.
    Until Phase 3b it was ``2 pi``, which is the §9.1 story in a single number — see the
    module docstring for the derivation and RES033 for the closure.

    ``anchor_ratio`` is the same quotient for the closed-form
    :func:`single_electron_spectrum`, which is **exactly one** by construction. It is
    computed anyway, because a number that is supposed to be exactly one is the cheapest
    check that the comparison itself is sound — if the anchor drifts, the arbitration is
    not measuring what it claims to. It has always read one, and it is now the *second*
    quantity here that does; the two are still independent, since the anchor is a closed
    form and ``ratio`` is a grid sum over a per-particle histogram.

    ``captured_fraction`` is how much of the angular distribution the grid covered, so a
    truncated integral is never mistaken for a normalization defect. It is what
    :attr:`expected_ratio` reduces one by.
    """

    ratio: float
    anchor_ratio: float
    captured_fraction: float
    total_yield: float
    n_angles: int
    cone_factor: float

    @property
    def expected_ratio(self) -> float:
        """One, reduced by what the finite grid could not see."""
        return self.captured_fraction

    @property
    def deviation(self) -> float:
        """Relative departure from the derived value — the number worth watching."""
        return self.ratio / self.expected_ratio - 1.0

    def summary(self) -> str:
        return (
            f"delta/Stage 0 = {self.ratio:.6f} vs capture = {self.expected_ratio:.6f} "
            f"({self.deviation:+.2%}); closed-form anchor {self.anchor_ratio:.6f}; "
            f"{100.0 * self.captured_fraction:.2f}% of the cone captured on a "
            f"{self.n_angles}x{self.n_angles} grid out to {self.cone_factor}/gamma"
        )


def check_normalization(
    samples: TrajectorySamples,
    *,
    n_bins: int = 128,
    n_angles: int = 33,
    cone_factor: float = DEFAULT_CONE_FACTOR,
) -> NormalizationCheck:
    """Compare delta's absolute photon count with Stage 0's. Since Phase 3b: expect one.

    This is the §9.1 arbitration reduced to one number, and it is now a *regression* check
    rather than an open question. The history is worth keeping in view, because it is the
    only reason to trust the answer:

    **Which side was the photon count.** Stage 0's total is
    ``flux x cross-section x time``, summed — an elementary count that needs no
    convention. The closed-form :func:`single_electron_spectrum` integrates to exactly
    that same total, independently. Two agreeing methods against one, and the odd one out
    was the one carrying a differential solid-angle measure — an extra ``2 pi`` is what an
    azimuthal integral counted twice looks like. This function reproduced it from a clean
    CGS reimplementation, so the predecessor's ~6.3 was not an artefact of its coordinate
    normalization.

    **The factor is now applied at the source, not here** (RES033). :data:`DIFFERENTIAL_PREFACTOR`
    carries the ``1/(2 pi)`` RES026 derived, so this ratio reads one and departures from one
    are what the harness watches — including a departure of ``2 pi``, which would mean a
    prefactor got reverted. P14's rule is intact: the constant was predicted from the
    derivation and then confirmed against this number, not tuned until this number
    cooperated.
    """
    edge = float(np.max(samples.gamma) ** 2)
    s_edges = np.linspace(0.0, 1.05 * edge, n_bins + 1)
    widths = np.diff(s_edges)
    s_centres = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Both spectra are densities sampled at bin *centres*, so the integral is the midpoint
    # sum. `np.trapezoid` over centres — which this did — silently drops half of the first
    # and last bin, and that bias landed in `deviation`, the §9.1 headline number, while
    # `expected_ratio` carried no matching correction. It is also why `anchor_ratio`
    # reported 0.9937 for a quantity documented as exactly one.
    total = samples.total_yield()
    delta_total = float(np.sum(
        angle_integrated_spectrum(samples, s_edges, n_angles=n_angles, cone_factor=cone_factor)
        * widths
    ))
    anchor_total = float(np.sum(single_electron_spectrum(samples, s_centres) * widths))

    return NormalizationCheck(
        ratio=delta_total / total if total else math.inf,
        anchor_ratio=anchor_total / total if total else math.inf,
        captured_fraction=captured_fraction(cone_factor),
        total_yield=total,
        n_angles=n_angles,
        cone_factor=cone_factor,
    )


def captured_fraction(cone_factor: float) -> float:
    """Fraction of delta's own angular integrand inside a cone of ``cone_factor / gamma``.

    Closed form, so the truncation is quoted rather than estimated. With ``u = gamma^2 r^2``
    and the azimuthal average ``<a_fac> = 1 - 2u/(1+u)^2``, the partial integral is::

        int_0^X dOmega 3 gamma^2 <a_fac>/(1+u)^2
            = 3 pi [ X/(1+X) - 1/3 + (1+X)^-2 - (2/3)(1+X)^-3 ]

    against the full ``2 pi`` derived in the module docstring, so the fraction is that
    bracket times ``3/2``. At ``X = 16`` (four cone widths) it is 0.917. Both integrals are
    written with the paper's bare prefactor ``3`` rather than
    :data:`DIFFERENTIAL_PREFACTOR`, and may stay that way: this is a *ratio* of the two, so
    §9.1's ``1/(2 pi)`` cancels out of it exactly.

    **The polarization factor has to be in here.** Integrating the Lorentz factor alone —
    which gives the tidier ``X/(1+X)`` — overstates the capture (0.941 at four cone widths)
    and, because it is the correction :func:`check_normalization` divides by, leaves a
    residue in the reported deviation that varies with the cone. That residue then eats the
    budget of the §9.1 tripwire for a reason having nothing to do with normalization.

    This is the fraction inside a **disc**. The grid is square, so it reaches to
    ``cone_factor * sqrt(2)`` in the corners and captures slightly more than this — leaving
    a small *positive* residue in :attr:`NormalizationCheck.deviation` that shrinks as the
    cone widens (measured +1.52% at four cone widths, +0.41% at eight). It is grid
    geometry, not physics: a monoenergetic zero-divergence beam reproduces it to within
    0.01 percentage points, so beam spread is not involved.
    """
    x = cone_factor**2
    bracket = x / (1.0 + x) - 1.0 / 3.0 + (1.0 + x) ** -2 - (2.0 / 3.0) * (1.0 + x) ** -3
    return 1.5 * bracket

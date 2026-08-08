"""Closed-form Compton-source physics: total yield, spectrum-width breakdown, and an
angle-integrated spectrum (GRAND_PLAN.md §4.3). No per-particle Monte Carlo — every
function here costs `O(1)` or `O(n_quad)`, never `O(n_particles)` (the predecessor's
``angle_integrated_spectrum`` used to sum over real macroparticle ``gamma`` samples and
caused a 76.3 GiB allocation at 5,000,000 particles x 2048 energy bins; this module has no
macroparticle argument anywhere, by construction).

Ported (algorithm and constants, not code) from the predecessor's
``ComptonSuite/src/gammaforge/models/analytical.py`` — SI/pint ``CollisionParams``
throughout there, CGS-Gaussian ``GaussianElectronBeam``/``GaussianParaxialLaser`` here
(P1). Round-beam and no-foci-displacement approximations are carried over unchanged;
generalizing them is out of scope for this landing (`DECISIONS.md` D035) — inventing a
non-round or displaced-focus overlap integral the paper does not derive would be the
P14c failure mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...io.bunch import GaussianElectronBeam
from ...io.laser import GaussianParaxialLaser
from ...io.units import C_CGS, SIGMA_T_CGS

__all__ = [
    "estimate_yield",
    "SpectrumWidthBreakdown",
    "estimate_spectrum_width",
    "angle_integrated_spectrum",
]

#: Above this, `math.erfc(nu) * math.exp(nu * nu)` is at real risk of overflowing before
#: `erfc` underflows to zero — the exact regime `erfcx` exists to protect against. Never
#: reached by a physically sane scenario (the baseline scenario has ``nu ~ 0.06``, the
#: predecessor's own worked example ``nu ~ 0.48``), but a GUI quick-estimate panel can see
#: extreme user-entered values, so the asymptotic branch below keeps this finite rather
#: than raising `OverflowError`.
_ERFCX_ASYMPTOTIC_THRESHOLD = 25.0


def _erfcx(nu: float) -> float:
    """``exp(nu**2) * erfc(nu)``, the scaled complementary error function.

    Hand-rolled from `math.erfc` rather than `scipy.special.erfcx` — `pyproject.toml`
    declares only `numpy` today, and `gammaforge.io.bunch._chi2_6_cdf` already sets the
    precedent of writing out a closed-form special function rather than adding `scipy` to
    keep the dependency surface where `pyproject.toml` already draws it (`DECISIONS.md`
    D037). Direct evaluation is exact (to `math.erfc`'s own precision) below the overflow
    threshold; above it, the standard asymptotic expansion
    ``erfcx(x) ~ (1/(x*sqrt(pi))) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6))`` takes over.

    ``estimate_yield`` only ever evaluates this at ``nu >= 0`` (built from sums of squares
    under square roots), so a negative branch is not needed.
    """
    if nu < _ERFCX_ASYMPTOTIC_THRESHOLD:
        return math.exp(nu * nu) * math.erfc(nu)
    inv2 = 1.0 / (nu * nu)
    series = 1.0 - 0.5 * inv2 + 0.75 * inv2**2 - 1.875 * inv2**3
    return series / (nu * math.sqrt(math.pi))


def estimate_yield(beam: GaussianElectronBeam, laser: GaussianParaxialLaser, N_e: float) -> float:
    """Cheap analytic total-photon-yield estimate from a Gaussian-bunch overlap integral.

    Not a replacement for a per-particle computation — a sanity-check anchor (§4.3/§7).
    ``laser`` must be a `GaussianParaxialLaser` — the *fitted* descriptive metrics
    (`gammaforge.io.laser.fit_gaussian_paraxial`), never a raw `LaserField`, mirroring how
    `xigma.collision.Collision.run`/`gammaforge.io.target.auto_ranges` already read laser
    scalars (P15). ``N_e`` is explicit rather than derived from ``beam`` so this honors the
    same cheap charge-only recompute path `XigmaEngine` declares (`RecomputeCost.QUERY_ONLY`
    on ``"n_e"``, handled entirely at the `io` level) — defaulting to
    ``beam.n_electrons()`` would desync from a charge-rescaled `InteractionParameters`.

    The laser's transverse profile is treated as round (geometric-mean effective size
    ``sqrt(sigma_x * sigma_y)``) since the underlying formula assumes a round beam — an
    elliptical laser is only approximated, not modeled exactly (see the module docstring).
    """
    sigma_ex = beam.m("sigma_x")
    sigma_ey = beam.m("sigma_y")
    beta_x = beam.beta_star_x()
    beta_y = beam.beta_star_y()
    sigma_ez = beam.m("sigma_z")
    sigma_lr0 = math.sqrt(laser.m("sigma_x") * laser.m("sigma_y"))
    sigma_lz = laser.m("duration") * C_CGS
    lambda_l = laser.m("wavelength")

    sb_av = math.sqrt(sigma_ex * sigma_ey / beta_x / beta_y)
    sigma0 = math.sqrt(sigma_ex**2 + sigma_lr0**2)
    nu = (
        math.sqrt(2.0)
        * sigma0
        / math.sqrt(sigma_ez**2 + sigma_lz**2)
        / math.sqrt(sb_av**2 + lambda_l**2 / math.pi**2 / sigma_lr0**2)
    )
    return N_e * laser.n_photons() * SIGMA_T_CGS / 2.0 / math.sqrt(math.pi) / sigma0**2 * nu * _erfcx(nu)


@dataclass(frozen=True)
class SpectrumWidthBreakdown:
    """The collimated-spectrum FWHM estimate (units of the Compton edge, dimensionless),
    as four independently-reported components (§4.3: "GUI shows a component table; total
    in quadrature") rather than the predecessor's single summed float.

    Each field already carries the predecessor's ``0.5 * 2.355`` FWHM-from-sigma prefactor
    applied to its own term, so :attr:`total` — `math.hypot` of the four fields — squares
    and re-sums them, exactly reproducing the predecessor's single
    ``0.5 * 2.355 * sqrt(term1 + term2 + term3 + term4)`` formula. The leading ``0.5`` is
    carried unexplained, as it was in the ported source — not retrofitted with a
    justification the original never had.
    """

    collimation: float  #: from angular collimation, ``(gamma * theta_col)^2``
    emittance: float  #: from angular divergence, ``(gamma * sqrt(div_x * div_y))^2``
    energy_spread: float  #: from beam energy spread, ``sigma_gamma / gamma``
    nonlinearity: float  #: from ponderomotive broadening, ``0.5 * a0_peak^2``

    @property
    def total(self) -> float:
        return math.hypot(self.collimation, self.emittance, self.energy_spread, self.nonlinearity)


def estimate_spectrum_width(
    beam: GaussianElectronBeam, laser: GaussianParaxialLaser, theta_col: float
) -> SpectrumWidthBreakdown:
    """Collimated-spectrum FWHM estimate, broken into its four components (§4.3).

    ``theta_col``: collimation half-angle (rad) — a single scalar; a caller combining
    `gammaforge.io.target.Target`'s separate ``theta_x_col``/``theta_y_col`` should use
    their geometric mean (`DECISIONS.md` D038), the same x/y-combining convention this
    module already uses for the laser waist (``sigma_lr0``) and the emittance term below.

    ``laser`` is the fitted `GaussianParaxialLaser` (see :func:`estimate_yield`);
    ``laser.a0_peak()`` stands in for the predecessor's ``pulse.a0_interaction`` — the
    pulse's own maximum a0, not the a0 at the electron bunch's actual position, which is
    the "foci displacement" growth item §4.3 lists as open (`DECISIONS.md` D035).
    """
    gamma0 = beam.gamma0()
    sigma_gamma = beam.sigma_gamma()
    emit_width = math.sqrt(beam.divergence_x() * beam.divergence_y())
    a0 = laser.a0_peak()
    prefactor = 0.5 * 2.355
    return SpectrumWidthBreakdown(
        collimation=prefactor * (gamma0 * theta_col) ** 2,
        emittance=prefactor * (gamma0 * emit_width) ** 2,
        energy_spread=prefactor * (sigma_gamma / gamma0),
        nonlinearity=prefactor * (0.5 * a0**2),
    )


def angle_integrated_spectrum(gamma0: float, sigma_gamma: float, N_e: float, s, n_quad: int = 401):
    """``dN/ds``, integrated over all emission solid angle and over the beam's own
    (assumed Gaussian) energy distribution — a fixed-size quadrature over ``gamma``, no
    macroparticles (§4.3): cost is ``O(n_quad * len(s))``, independent of ``n_particles``
    by construction.

    Implements its **own** copy of the standard linear-Compton kinematic shape
    (``1.5 * (1 - 2y(1-y))`` for ``y = s / gamma**2``) — the same shape
    `gammaforge.engines.xigma.stages.angle_integrated_spectrum` and
    `gammaforge.validation.references.delta.single_electron_spectrum` also implement,
    **deliberately not imported from either**. Analytical is the third independent leg of
    §7's four-method cross-validation (xigma vs delta vs analytical vs kascade); importing
    the shape from either of the other two would make that specific comparison circular,
    the same reasoning `xigma/stages.py`'s own copy already documents.

    ``s``: scalar or array of normalized photon energies (``s = E / (4 * photon_energy)``,
    matching the convention `xigma.collision.Collision` uses for the same axis). ``N_e``
    scales the output to an absolute photon count: the quadrature weights integrate to 1
    (to quadrature precision) over the sampled +-6 ``sigma_gamma`` window, so multiplying
    by ``N_e`` reproduces "one scattering attempt per electron," not yet a photon count —
    see `gammaforge.engines.analytical.engine` for how the caller turns this into a real
    spectrum by construction against `estimate_yield`.

    ``n_quad``: quadrature points spanning +-6 ``sigma_gamma`` around ``gamma0`` —
    independent of ``n_particles``, so a generous default costs nothing.

    Raises ``ValueError`` for ``sigma_gamma <= 0``: `gammaforge.io.bunch.validate` permits
    a beam with exactly zero energy spread (only rejects negative), but the quadrature
    grid this function builds is degenerate there (a zero-width Gaussian divided by its
    own zero width) and would otherwise return `nan` silently rather than raising —
    exactly the silent-fallback failure mode this repo's conventions reject in favor of
    an explicit error (see `engines.xigma.stages._check_backend`'s docstring for the same
    argument in a different context).
    """
    if sigma_gamma <= 0.0:
        raise ValueError(
            f"angle_integrated_spectrum: sigma_gamma must be > 0 (got {sigma_gamma!r}) — "
            "a zero-width energy spread makes the quadrature grid degenerate"
        )
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))

    span = 6.0 * sigma_gamma
    gamma_grid = np.linspace(max(gamma0 - span, 1.0 + 1e-9), gamma0 + span, n_quad)
    pdf = np.exp(-0.5 * ((gamma_grid - gamma0) / sigma_gamma) ** 2) / (sigma_gamma * math.sqrt(2.0 * math.pi))
    quad_weight = pdf * np.gradient(gamma_grid) * N_e

    gamma2 = (gamma_grid**2)[:, None]
    y = s_arr[None, :] / gamma2
    shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    out = np.sum(quad_weight[:, None] * shape / gamma2, axis=0)

    return out if np.ndim(s) else out[0]

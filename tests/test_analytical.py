"""`engines/analytical` (GRAND_PLAN.md §4.3): closed-form yield/width/spectrum formulas
and the `AnalyticalEngine` wrapper.

The worked-example fixture (`_EXAMPLE_BEAM`/`_EXAMPLE_LASER`) is the predecessor's own
test scenario (`ComptonSuite/tests/test_analytical.py`), re-parametrized from SI/pint
`CollisionParams` onto this repo's CGS `GaussianElectronBeam`/`GaussianParaxialLaser` —
100 pC / 200 MeV / 10 um beam, 0.05 J / 0.8 um / 2.5 um / 12.74 fs pulse.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.analytical.formulas import (
    SpectrumWidthBreakdown,
    angle_integrated_spectrum,
    estimate_spectrum_width,
    estimate_yield,
)
from gammaforge.engines.base import Engine
from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge=Quantity(100.0, "pC"),
    kinetic_energy=Quantity(200.0, "MeV"),
    rel_energy_spread=0.001,
    sigma_x=Quantity(10.0, "um"),
    sigma_y=Quantity(10.0, "um"),
    emit_x=Quantity(0.05, "um") * Quantity(1.0, "rad"),
    emit_y=Quantity(0.05, "um") * Quantity(1.0, "rad"),
    sigma_z=Quantity(1.0, "ps"),
)
_EXAMPLE_LASER = GaussianParaxialLaser(
    pulse_energy=Quantity(0.05, "J"),
    wavelength=Quantity(0.8, "um"),
    sigma_x=Quantity(2.5, "um"),
    sigma_y=Quantity(2.5, "um"),
    duration=Quantity(12.74, "fs"),
)

#: `ComptonSuite/models/analytical.py`'s own `estimate_yield`/`estimate_spectrum_width`,
#: evaluated on the identical worked example (SI/pint `CollisionParams`, verified by
#: actually running that repo). The yield differs at ~1e-6 relative rather than float
#: precision because it is the only quantity here that involves `SIGMA_T`, and the two
#: repos are separately-installed `pint` environments with (very slightly) different
#: CODATA constant tables — not a unit-conversion defect in this port. The width uses no
#: such constant and matches to ~1e-11.
_PREDECESSOR_YIELD = 6644238.68637256
_PREDECESSOR_WIDTH_TOTAL = 4.6671292359002505


def test_estimate_yield_is_positive_finite():
    y = estimate_yield(_EXAMPLE_BEAM, _EXAMPLE_LASER, _EXAMPLE_BEAM.n_electrons())
    assert math.isfinite(y) and y > 0


def test_estimate_yield_reproduces_the_predecessors_worked_example():
    y = estimate_yield(_EXAMPLE_BEAM, _EXAMPLE_LASER, _EXAMPLE_BEAM.n_electrons())
    assert y == pytest.approx(_PREDECESSOR_YIELD, rel=1e-4)


def test_estimate_yield_matches_the_thomson_limit_closed_form():
    """§7's own anchor: "Thomson limit: zero-a0 yield ~ N_e . sigma_T . (overlap) closed
    form." As `nu -> infinity`, `nu * erfcx(nu) -> 1/sqrt(pi)` (the standard asymptotic
    limit), collapsing `estimate_yield`'s full expression to the textbook head-on Gaussian
    luminosity `N_e * n_photons * sigma_T / (2 pi (sigma_ex^2 + sigma_lr0^2))`. A short
    bunch/pulse relative to the transverse sizes drives `nu` large without needing the
    predecessor repo at all.
    """
    from gammaforge.io.units import SIGMA_T_CGS

    beam = GaussianElectronBeam(
        bunch_charge=Quantity(100.0, "pC"),
        kinetic_energy=Quantity(200.0, "MeV"),
        rel_energy_spread=0.001,
        sigma_x=Quantity(100.0, "um"),
        sigma_y=Quantity(100.0, "um"),
        emit_x=Quantity(1.0, "um") * Quantity(1.0, "rad"),
        emit_y=Quantity(1.0, "um") * Quantity(1.0, "rad"),
        sigma_z=Quantity(0.03, "um"),
    )
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(0.05, "J"),
        wavelength=Quantity(0.8, "um"),
        sigma_x=Quantity(100.0, "um"),
        sigma_y=Quantity(100.0, "um"),
        duration=Quantity(0.1, "fs"),
    )
    N_e = beam.n_electrons()
    y = estimate_yield(beam, laser, N_e)

    sigma_ex, sigma_ey = beam.m("sigma_x"), beam.m("sigma_y")
    sigma_lr0 = math.sqrt(laser.m("sigma_x") * laser.m("sigma_y"))
    thomson_limit = N_e * laser.n_photons() * SIGMA_T_CGS / (2.0 * math.pi * (sigma_ex**2 + sigma_lr0**2))
    assert y == pytest.approx(thomson_limit, rel=1e-6)


def test_estimate_spectrum_width_is_positive_finite():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert isinstance(w, SpectrumWidthBreakdown)
    assert math.isfinite(w.total) and w.total > 0


def test_estimate_spectrum_width_reproduces_the_predecessors_worked_example():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(_PREDECESSOR_WIDTH_TOTAL, rel=1e-6)


def test_estimate_spectrum_width_grows_with_larger_collimation_angle():
    narrow = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-4)
    wide = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-2)
    assert wide.total > narrow.total


def test_spectrum_width_breakdown_total_is_hypot_of_components():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(
        math.hypot(w.collimation, w.emittance, w.energy_spread, w.nonlinearity)
    )


def test_spectrum_width_breakdown_components_move_independently():
    narrow = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-4)
    wide = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-2)
    assert wide.collimation > narrow.collimation
    for field in ("emittance", "energy_spread", "nonlinearity"):
        assert getattr(wide, field) == pytest.approx(getattr(narrow, field))


def test_angle_integrated_spectrum_shape_and_scalar_input():
    s_array = np.linspace(0.01, 0.99, 16)
    out_array = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), s_array
    )
    assert out_array.shape == s_array.shape
    assert np.all(np.isfinite(out_array)) and np.all(out_array >= 0)

    out_scalar = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), 0.5
    )
    assert np.ndim(out_scalar) == 0 or isinstance(out_scalar, float)


def test_angle_integrated_spectrum_zero_outside_kinematic_range():
    gamma0 = 100.0
    s_far_beyond_edge = np.array([gamma0**2 * 1.5])
    out = angle_integrated_spectrum(gamma0, gamma0 * 1e-6, 1.0, s_far_beyond_edge)
    assert out[0] == 0.0


def test_angle_integrated_spectrum_rejects_zero_energy_spread():
    """`io.bunch.validate` permits `rel_energy_spread == 0` (only `< 0` raises), so
    `sigma_gamma == 0` is a legal beam — but it makes this function's quadrature grid
    degenerate (a zero-width Gaussian divided by its own zero width), which would
    otherwise return `nan` silently. This repo's convention is an explicit error over a
    silent fallback."""
    with pytest.raises(ValueError, match="sigma_gamma"):
        angle_integrated_spectrum(100.0, 0.0, 1.0, 0.5)


def test_angle_integrated_spectrum_fast_at_reported_scale():
    """Regression guard for the OOM bug class the predecessor's refactor fixed (see
    `formulas.py`'s module docstring): no argument here scales with `n_particles`."""
    s_array = np.linspace(1e-3, 1.0 - 1e-3, 2048)
    out = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), s_array
    )
    assert out.shape == s_array.shape
    assert np.all(np.isfinite(out))


def test_angle_integrated_spectrum_matches_monte_carlo_reference():
    """Independent physics check: draw real gamma samples from the same Gaussian, sum the
    per-particle kinematic shape directly here (not via any production code path), and
    compare to the closed-form quadrature."""
    rng = np.random.default_rng(1)
    n = 2_000_000
    gamma0, sigma_gamma, N_e = _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons()
    gamma_samples = rng.normal(gamma0, sigma_gamma, n)
    weight = N_e / n

    s_array = np.linspace(0.05, 0.95, 12)
    gamma2 = (gamma_samples**2)[:, None]
    y = s_array[None, :] / gamma2
    shape = np.where((y < 0) | (y > 1), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    mc_reference = np.sum(weight * shape / gamma2, axis=0)

    quad_result = angle_integrated_spectrum(gamma0, sigma_gamma, N_e, s_array)
    assert np.allclose(quad_result, mc_reference, rtol=0.02)


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------
def _interaction(n_particles=4000, seed=0, outputs=()):
    small = replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=n_particles, seed=seed))
    interaction = scenarios.build(small)
    return replace(interaction, target=replace(interaction.target, outputs=outputs))


def test_analytical_engine_conforms_to_the_engine_protocol():
    assert isinstance(AnalyticalEngine(), Engine)


def test_supported_outputs_matches_what_run_actually_fills():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(64,)),
    )
    interaction = _interaction(outputs=requests)
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == set(AnalyticalEngine.supported_outputs)


def test_unsupported_output_kinds_are_silently_omitted_not_errored():
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(3, 3)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_unsupported_temporal_envelope_request_does_not_crash():
    """`io.target.auto_ranges`'s `TEMPORAL_ENVELOPE` branch requires a bunch and raises
    without one; a `Target` requesting it alongside a supported output must not reach
    that branch just because this engine happens to skip the kind (a regression this
    engine's own review caught: filtering unsupported requests must happen before
    `auto_ranges` runs, not after)."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.TEMPORAL_ENVELOPE, resolution=(8,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_spectrum_integral_equals_total_yield_exactly():
    """§7: "Total yield: integral spectrum = total_yield -- exact identities, not
    tolerances, where the contract guarantees them." `DECISIONS.md` D036 makes this exact
    by construction for analytical."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(200,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    total_yield = float(results.photon_slices[OutputKind.TOTAL_YIELD].distr)
    spectrum_integral = results.photon_slices[OutputKind.SPECTRUM].integrate()
    assert spectrum_integral == pytest.approx(total_yield, rel=1e-9)


def test_spectrum_grid_integral_correction_factor_is_near_one():
    """The rescale in D036 always makes the integral exact; this guards that the factor
    it applies stays close to 1, i.e. the auto-derived energy range is not truncating
    real spectral weight (a distortion the rescale would otherwise mask silently)."""
    from gammaforge.engines.analytical.formulas import angle_integrated_spectrum as ais
    from gammaforge.io.laser import fit_gaussian_paraxial as fit
    from gammaforge.io.results import Axis
    from gammaforge.io.target import auto_ranges, slice_axis_values

    interaction = _interaction(outputs=(OutputRequest(OutputKind.SPECTRUM, resolution=(200,)),))
    metrics = fit(interaction.laser)
    photon_energy = metrics.photon_energy()
    ranges = auto_ranges(interaction.target, interaction.beam, interaction.laser)
    request = interaction.target.outputs[0]
    values = slice_axis_values(request, ranges[OutputKind.SPECTRUM])
    s = values[Axis.ENERGY] / (4.0 * photon_energy)
    raw = ais(interaction.beam.gamma0(), interaction.beam.sigma_gamma(), 1.0, s, 401)
    raw_integral = float(np.trapezoid(raw / (4.0 * photon_energy), values[Axis.ENERGY]))
    assert raw_integral == pytest.approx(1.0, abs=0.05)


def test_analytical_engine_is_independent_of_n_particles():
    """analytical never touches `interaction.bunch` — results must be bit-identical
    across `n_particles`, the structural version of "the old OOM bug class must not
    return" (§4.3)."""
    outputs = (OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(64,)))
    small = AnalyticalEngine().run(_interaction(n_particles=100, outputs=outputs), AnalyticalEngine.schema)
    large = AnalyticalEngine().run(_interaction(n_particles=100_000, outputs=outputs), AnalyticalEngine.schema)
    for request in outputs:
        assert np.array_equal(
            small.photon_slices[request.kind].distr, large.photon_slices[request.kind].distr
        )


def test_model_specific_carries_the_width_breakdown_and_is_charge_independent():
    """`Results.scaled()` copies `model_specific` verbatim (unscaled) — these three
    values must actually be charge-independent, or a charge-rescaled `Results` would
    silently carry a stale number."""
    interaction = _interaction(outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert isinstance(results.model_specific["spectrum_width_fwhm"], SpectrumWidthBreakdown)

    doubled = replace(interaction, N_e=interaction.N_e * 2.0)
    results_doubled = AnalyticalEngine().run(doubled, AnalyticalEngine.schema)
    assert results_doubled.model_specific["spectrum_width_fwhm"] == results.model_specific["spectrum_width_fwhm"]
    assert results_doubled.model_specific["a0_peak"] == results.model_specific["a0_peak"]
    assert results_doubled.model_specific["n_photons"] == results.model_specific["n_photons"]
    # ... while the yield itself does scale.
    y = float(results.photon_slices[OutputKind.TOTAL_YIELD].distr)
    y_doubled = float(results_doubled.photon_slices[OutputKind.TOTAL_YIELD].distr)
    assert y_doubled == pytest.approx(2.0 * y)

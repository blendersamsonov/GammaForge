"""CGS-Gaussian constants and convention algebra.

The constant tests are deliberately *cross-checks against independent CGS identities*
rather than hardcoded digits: `units.py` pulls each value from pint's CODATA table, so
re-asserting the same digits would only test that pint is pint. Checking that r_e, sigma_T
and alpha reproduce their textbook definitions from the *other* constants is what actually
catches a botched SI->CGS conversion — especially the hand-applied charge one.
"""

from __future__ import annotations

import math

import pytest

from gammaforge.io import units as u

pytestmark = [pytest.mark.tier0, pytest.mark.fast]


def test_charge_matches_textbook_esu_value():
    # e = 4.80320471e-10 statC; the one constant pint cannot produce for us, so this is
    # the only place a literal is the real check.
    assert u.E_ESU == pytest.approx(4.80320471e-10, rel=1e-8)


def test_classical_electron_radius_from_charge_and_rest_energy():
    assert u.E_ESU**2 / u.MEC2_CGS == pytest.approx(u.R_E_CGS, rel=1e-6)


def test_thomson_cross_section_from_electron_radius():
    assert 8.0 * math.pi / 3.0 * u.R_E_CGS**2 == pytest.approx(u.SIGMA_T_CGS, rel=1e-6)


def test_fine_structure_constant_in_gaussian_form():
    # alpha = e^2 / (hbar c) has no 4*pi*eps0 in Gaussian units -- a wrong charge
    # conversion shows up here immediately.
    assert u.E_ESU**2 / (u.HBAR_CGS * u.C_CGS) == pytest.approx(u.ALPHA, rel=1e-6)


def test_electron_rest_energy_in_mev():
    assert u.MEC2_CGS / u.EV_CGS / 1e6 == pytest.approx(0.51099895, rel=1e-7)


def test_width_conversions_round_trip():
    sigma = 3.5e-4
    for conv in u.WidthConvention:
        other = u.convert_width(sigma, u.WidthConvention.SIGMA_INTENSITY_RMS, conv)
        back = u.convert_width(other, conv, u.WidthConvention.SIGMA_INTENSITY_RMS)
        assert back == pytest.approx(sigma)


def test_width_conversion_values():
    sigma = 1.0
    fwhm = u.convert_width(sigma, u.WidthConvention.SIGMA_INTENSITY_RMS, u.WidthConvention.FWHM_INTENSITY)
    assert fwhm == pytest.approx(2.0 * math.sqrt(2.0 * math.log(2.0)))
    # The 1/e^2 intensity radius is twice the intensity-profile RMS.
    w0 = u.convert_width(sigma, u.WidthConvention.SIGMA_INTENSITY_RMS, u.WidthConvention.W0_1E2)
    assert w0 == pytest.approx(2.0)
    # A field-RMS width is sqrt(2) wider than the intensity-RMS width it describes.
    sf = u.convert_width(sigma, u.WidthConvention.SIGMA_INTENSITY_RMS, u.WidthConvention.SIGMA_FIELD_RMS)
    assert sf == pytest.approx(math.sqrt(2.0))


def test_time_and_width_conventions_do_not_interoperate():
    with pytest.raises(u.UnknownConversionError):
        u.convert_width(1.0, u.TimeConvention.FWHM_INTENSITY, u.WidthConvention.W0_1E2)
    with pytest.raises(u.UnknownConversionError):
        u.convert_time(1.0, u.TimeConvention.FWHM_INTENSITY, u.WidthConvention.W0_1E2)


def test_light_time_is_opt_in_per_field():
    # It equates a length with a duration, which is right for a longitudinal extent quoted
    # either way (§2.1) and wrong for anything else — a transverse size in femtoseconds is
    # a different physical quantity, not a unit choice. So it is off unless asked for.
    assert u.to_canonical(1.0, "s", "cm", light_time=True) == pytest.approx(u.C_CGS)
    assert u.from_canonical(u.C_CGS, "cm", "s", light_time=True) == pytest.approx(1.0)
    with pytest.raises(Exception):
        u.to_canonical(1.0, "s", "cm")


def test_gaussian_charge_context_is_always_on():
    # Unlike light_time, this one is unconditional: a value either is a charge or is not,
    # and the SI/Gaussian split is notational, not a physical ambiguity.
    assert u.to_canonical(1.0, "C", "statC") == pytest.approx(u.STATC_PER_COULOMB)


def test_ordinary_unit_conversion():
    assert u.to_canonical(1.0, "m", "cm") == pytest.approx(100.0)
    assert u.to_canonical(1.0, "J", "erg") == pytest.approx(1e7)
    assert u.to_canonical(1.0, "MeV", "erg") == pytest.approx(1e6 * u.EV_CGS)

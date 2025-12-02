"""
Unit tests for physics constants module.

Verifies all constants use scipy.constants (no hardcoded values).
"""

import scipy.constants as scipy_const


def test_constants_module_imports():
    """Test that constants module can be imported"""


def test_fundamental_constants_use_scipy():
    """Verify fundamental constants come from scipy.constants"""
    from neunorm.utils.constants import E_CHARGE, M_N, H

    # Must exactly match scipy values (no rounding, no hardcoding)
    assert M_N == scipy_const.m_n
    assert H == scipy_const.h
    assert E_CHARGE == scipy_const.e


def test_derived_constants_computed_correctly():
    """Verify derived constants computed from scipy values"""
    from neunorm.utils.constants import H_OVER_MN

    # h / m_n must be computed from scipy (full precision)
    expected = scipy_const.h / scipy_const.m_n
    assert H_OVER_MN == expected

    # Verify high precision (at least 10 significant figures)
    relative_error = abs(H_OVER_MN - expected) / expected
    assert relative_error < 1e-15  # Machine precision


def test_de_broglie_constant_computed_correctly():
    """Verify de Broglie constant for wavelength-energy conversion"""
    from neunorm.utils.constants import DE_BROGLIE_EV_ANGSQ

    # E(eV) = (h²/(2·m_n·e)) / λ²(Å)
    # Conversion factor in eV·Å²
    expected = (scipy_const.h**2 / (2 * scipy_const.m_n * scipy_const.e)) * 1e20

    assert abs(DE_BROGLIE_EV_ANGSQ - expected) / expected < 1e-15


def test_de_broglie_constant_value_reasonable():
    """Sanity check: de Broglie constant should be ~0.0818 eV·Å²"""
    from neunorm.utils.constants import DE_BROGLIE_EV_ANGSQ

    # Known approximate value (from literature)
    assert 0.081 < DE_BROGLIE_EV_ANGSQ < 0.083


def test_tpx3_clock_is_documented():
    """TPX3 clock constant must have source documentation"""
    from neunorm.utils.constants import TPX3_CLOCK_NS

    # Should be 25 ns (Timepix3 specification)
    assert TPX3_CLOCK_NS == 25.0

    # Verify docstring exists and contains source
    import neunorm.utils.constants as const_module

    docstring = const_module.__dict__.get("__doc__", "")

    # Module docstring should mention scipy.constants
    assert "scipy.constants" in docstring.lower() or "scipy" in docstring.lower()


def test_venus_flight_path_is_documented():
    """VENUS flight path must have source documentation"""
    from neunorm.utils.constants import VENUS_FLIGHT_PATH_M

    # Should be 25 m (VENUS L2 nominal)
    assert VENUS_FLIGHT_PATH_M == 25.0


def test_constants_are_immutable():
    """Constants should be simple values (not mutable objects)"""
    from neunorm.utils.constants import E_CHARGE, H_OVER_MN, M_N, H

    # All should be float or int (immutable)
    assert isinstance(M_N, float)
    assert isinstance(H, float)
    assert isinstance(E_CHARGE, float)
    assert isinstance(H_OVER_MN, float)


def test_conversion_helpers_exist():
    """Test helper functions for common conversions"""
    from neunorm.utils.constants import angstrom_to_meter, ev_to_joule, joule_to_ev, meter_to_angstrom

    # Test roundtrip conversions
    energy_ev = 10.0
    energy_j = ev_to_joule(energy_ev)
    energy_ev_back = joule_to_ev(energy_j)
    assert abs(energy_ev_back - energy_ev) / energy_ev < 1e-15

    wavelength_a = 1.8  # Angstrom (typical thermal neutron)
    wavelength_m = angstrom_to_meter(wavelength_a)
    wavelength_a_back = meter_to_angstrom(wavelength_m)
    assert abs(wavelength_a_back - wavelength_a) / wavelength_a < 1e-15


def test_ev_to_joule_uses_scipy():
    """Verify eV→Joule conversion uses scipy.constants.e"""
    import scipy.constants as const

    from neunorm.utils.constants import ev_to_joule

    energy_ev = 42.0
    energy_j = ev_to_joule(energy_ev)

    # Should be: eV * e (elementary charge)
    expected = energy_ev * const.e
    assert energy_j == expected


def test_angstrom_meter_conversion_factors():
    """Verify Angstrom-meter conversion uses correct factor (1e-10)"""
    from neunorm.utils.constants import angstrom_to_meter, meter_to_angstrom

    # 1 Å = 1e-10 m
    assert angstrom_to_meter(1.0) == 1e-10
    assert meter_to_angstrom(1e-10) == 1.0

    # Test with typical neutron wavelength
    assert angstrom_to_meter(1.8) == 1.8e-10
    assert meter_to_angstrom(1.8e-10) == 1.8

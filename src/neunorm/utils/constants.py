"""
Physical constants for neutron TOF calculations.

All fundamental physical constants sourced from scipy.constants (CODATA 2018).
NO hardcoded values for standard physics quantities - always use scipy.

Detector-specific and facility-specific constants are documented with sources.
"""

import scipy.constants as const

# =============================================================================
# Fundamental Physical Constants (from scipy.constants)
# =============================================================================

M_N = const.m_n  # Neutron mass (kg): 1.67492749804e-27
"""Neutron mass in kg. Source: scipy.constants.m_n (CODATA 2018)"""

H = const.h  # Planck constant (J·s): 6.62607015e-34
"""Planck constant in J·s. Source: scipy.constants.h (exact, 2019 SI definition)"""

E_CHARGE = const.e  # Elementary charge (C): 1.602176634e-19
"""Elementary charge in Coulombs. Source: scipy.constants.e (exact, 2019 SI definition)"""

# =============================================================================
# Derived Constants (computed from scipy.constants for full precision)
# =============================================================================

H_OVER_MN = const.h / const.m_n  # m·s
"""
h / m_n ratio for wavelength conversion.

Used in: λ = (h/m_n) * (t/L)
Value: ~3.956034e-7 m·s
Computed from scipy.constants for full precision (no rounding).
"""

DE_BROGLIE_EV_ANGSQ = (const.h**2 / (2 * const.m_n * const.e)) * 1e20  # eV·Å²
"""
de Broglie constant for wavelength-energy conversion.

Used in: E(eV) = DE_BROGLIE_EV_ANGSQ / λ²(Å)
Value: ~0.0818 eV·Å²
Computed from scipy.constants with unit conversion (m² → Å²).

Derivation:
E = h² / (2·m_n·λ²)     [SI units: J = (J·s)² / (kg·m²)]
E(eV) = E(J) / e        [Convert J to eV]
With λ in Å: multiply by 1e20 (since 1 Å = 1e-10 m, so 1 Å² = 1e-20 m²)
"""

# =============================================================================
# Detector-Specific Constants (NOT in scipy.constants)
# =============================================================================

TPX3_CLOCK_NS = 25.0  # nanoseconds
"""
Timepix3 detector time resolution.

Source:
- Timepix3 Manual, Section 3.2, Medipix Collaboration (2014)
- URL: https://medipix.web.cern.ch/
- Technical specification: 1.5625 ns × 16 (clock divider) = 25 ns

Last verified: 2025-11-17
Precision: Exact (clock specification from manufacturer)
"""

TPX4_CLOCK_NS = 0.195  # nanoseconds
"""
Timepix4 detector time resolution (improved from TPX3).

Source:
- Timepix4 Specifications, Medipix Collaboration (2023)
- URL: https://medipix.web.cern.ch/

Last verified: 2025-11-17
Note: TPX4 not yet deployed at VENUS as of 2025-11-17
"""

# =============================================================================
# Facility-Specific Constants
# =============================================================================

VENUS_FLIGHT_PATH_M = 25.0  # meters
"""
Default VENUS beamline L2 distance (moderator to detector).

Source:
- VENUS beamline design documents, Spallation Neutron Source, ORNL
- URL: https://neutrons.ornl.gov/venus
- Contact: Instrument scientists for precise measurements

Last verified: 2025-11-17

Important Notes:
- Actual flight path varies by detector position and sample position
- Always verify with beamline staff for high-precision measurements
- Typical range: 24.5 - 25.5 m depending on detector mount
"""

MARS_FLIGHT_PATH_M = None  # Not applicable (continuous beam, no TOF)
"""
MARS beamline uses continuous neutron beam (HFIR reactor).

No pulsed source → no time-of-flight capability → flight path concept not applicable.
For MARS normalization, use traditional flat-field correction only.
"""

# =============================================================================
# Unit Conversion Helpers
# =============================================================================


def ev_to_joule(energy_ev: float) -> float:
    """
    Convert electron volts to Joules.

    Uses scipy.constants.e (elementary charge).

    Parameters
    ----------
    energy_ev : float
        Energy in electron volts

    Returns
    -------
    float
        Energy in Joules
    """
    return energy_ev * E_CHARGE


def joule_to_ev(energy_j: float) -> float:
    """
    Convert Joules to electron volts.

    Uses scipy.constants.e (elementary charge).

    Parameters
    ----------
    energy_j : float
        Energy in Joules

    Returns
    -------
    float
        Energy in electron volts
    """
    return energy_j / E_CHARGE


def angstrom_to_meter(wavelength_a: float) -> float:
    """
    Convert Angstrom to meters.

    Parameters
    ----------
    wavelength_a : float
        Wavelength in Angstrom

    Returns
    -------
    float
        Wavelength in meters

    Notes
    -----
    Conversion factor: 1 Å = 1e-10 m (exact, SI definition)
    """
    return wavelength_a * 1e-10


def meter_to_angstrom(wavelength_m: float) -> float:
    """
    Convert meters to Angstrom.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters

    Returns
    -------
    float
        Wavelength in Angstrom

    Notes
    -----
    Conversion factor: 1 m = 1e10 Å (exact, SI definition)
    """
    return wavelength_m * 1e10

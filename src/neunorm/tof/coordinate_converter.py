"""
Convert between TOF and wavelength or energy
"""

import scipp as sc


def convert_tof_to_wavelength(
    tof: sc.Variable, distance: sc.Variable, offset: sc.Variable = sc.scalar(0, unit="us")
) -> sc.Variable:
    """Convert time-of-flight (TOF) data to wavelength.

    TOF → Wavelength:
    λ = (h × TOF) / (m_n × L)

    Parameters
    ----------
    tof : sc.Variable
        Variable containing TOF values with appropriate units (e.g., microseconds).
    distance : sc.Variable
        Variable representing the distance from the source to the detector, with appropriate units (e.g., meters).

    Returns
    -------
    sc.Variable
        Variable containing wavelength values corresponding to the input TOF data in units of Angstroms.
    """

    return sc.to_unit((tof + offset) * sc.constants.h / (sc.constants.m_n * distance), "Angstrom")


def convert_wavelength_to_tof(
    wavelength: sc.Variable, distance: sc.Variable, offset: sc.Variable = sc.scalar(0, unit="us")
) -> sc.Variable:
    """Convert wavelength data to time-of-flight (TOF).

    Wavelength → TOF:
    TOF = (λ × m_n × L) / h

    Parameters
    ----------
    wavelength : sc.Variable
        Variable containing wavelength values with appropriate units (e.g., Angstroms).
    distance : sc.Variable
        Variable representing the distance from the source to the detector, with appropriate units (e.g., meters).
    offset : sc.Variable
        Variable representing the time offset, with appropriate units (e.g., microseconds).


    Returns
    -------
    sc.Variable
        Variable containing TOF values corresponding to the input wavelength data in units of microseconds.
    """

    return sc.to_unit((wavelength * sc.constants.m_n * distance) / sc.constants.h, offset.unit) - offset


def convert_tof_to_energy(
    tof: sc.Variable, distance: sc.Variable, offset: sc.Variable = sc.scalar(0, unit="us")
) -> sc.Variable:
    """Convert time-of-flight (TOF) data to energy.

    TOF → Energy:
    E = (1/2) × m_n × (L / TOF)²

    Parameters
    ----------
    tof : sc.Variable
        Variable containing TOF values with appropriate units (e.g., microseconds).
    distance : sc.Variable
        Variable representing the distance from the source to the detector, with appropriate units (e.g., meters).
    offset : sc.Variable
        Variable representing the time offset, with appropriate units (e.g., microseconds).

    Returns
    -------
    sc.Variable
        Variable containing energy values corresponding to the input TOF data in units of meV.
    """

    velocity = distance / (tof + offset)
    energy_joules = 0.5 * sc.constants.m_n * velocity**2
    return sc.to_unit(energy_joules, "meV")

"""
Binning utilities for TOF/Energy/Wavelength conversions.

Provides functions for:
- Creating TOF bin edges from energy or wavelength specifications
- Converting histograms between TOF/energy/wavelength spaces
- Physics-based conversions using scipy.constants

All conversions preserve variance (uncertainty) information.
"""

from typing import Optional

import scipp as sc
import scipp.constants as sc_const

from neunorm.data_models.tof import BinningConfig


def tof_to_energy(tof: sc.Variable, flight_path: sc.Variable) -> sc.Variable:
    """
    Convert time-of-flight to neutron energy.

    Formula: E = (1/2) * m_n * (L/t)²

    Parameters
    ----------
    tof : sc.Variable
        Time-of-flight with unit compatible with 's' (seconds)
    flight_path : sc.Variable
        Flight path length with unit compatible with 'm' (meters)

    Returns
    -------
    sc.Variable
        Neutron energy with unit 'eV'

    Notes
    -----
    Uses scipy.constants.m_n for neutron mass (no hardcoded values).
    Higher TOF → lower energy (inverse relationship).

    Examples
    --------
    >>> tof = sc.scalar(1e-3, unit='s')  # 1 ms
    >>> L = sc.scalar(25.0, unit='m')
    >>> energy = tof_to_energy(tof, L)
    >>> print(energy)  # ~5.2 eV
    """
    # Velocity: v = L / t
    velocity = flight_path / tof.to(unit="s")

    # Kinetic energy: E = (1/2) * m_n * v²
    energy_j = 0.5 * sc_const.m_n * velocity**2

    # Convert to eV
    energy_ev = energy_j.to(unit="eV")

    return energy_ev


def tof_to_wavelength(tof: sc.Variable, flight_path: sc.Variable) -> sc.Variable:
    """
    Convert time-of-flight to neutron wavelength.

    Formula: λ = h * t / (m_n * L)

    Parameters
    ----------
    tof : sc.Variable
        Time-of-flight with unit compatible with 's'
    flight_path : sc.Variable
        Flight path length with unit compatible with 'm'

    Returns
    -------
    sc.Variable
        Neutron wavelength with unit 'angstrom'

    Notes
    -----
    Uses scipy.constants.h and scipy.constants.m_n (no hardcoded values).
    Linear relationship: TOF ∝ wavelength.

    Examples
    --------
    >>> tof = sc.scalar(1e-3, unit='s')  # 1 ms
    >>> L = sc.scalar(25.0, unit='m')
    >>> wavelength = tof_to_wavelength(tof, L)
    >>> print(wavelength)  # ~1.58 Å
    """
    # λ = h * t / (m_n * L)
    # = (h / m_n) * (t / L)
    h_over_mn = sc_const.h / sc_const.m_n

    wavelength_m = h_over_mn * tof.to(unit="s") / flight_path

    # Convert to Angstrom
    wavelength_angstrom = wavelength_m.to(unit="angstrom")

    return wavelength_angstrom


def wavelength_to_energy(wavelength: sc.Variable) -> sc.Variable:
    """
    Convert neutron wavelength to energy via de Broglie relation.

    Formula: E = h² / (2 * m_n * λ²)

    Parameters
    ----------
    wavelength : sc.Variable
        Wavelength with unit compatible with 'angstrom'

    Returns
    -------
    sc.Variable
        Energy with unit 'eV'

    Notes
    -----
    Uses scipy.constants (no hardcoded values).
    Inverse square relationship: E ∝ 1/λ².

    Examples
    --------
    >>> wl = sc.scalar(1.8, unit='angstrom')  # Thermal neutron
    >>> energy = wavelength_to_energy(wl)
    >>> print(energy)  # ~0.025 eV
    """
    wl_m = wavelength.to(unit="m")

    # E = h² / (2 * m_n * λ²)
    energy_j = sc_const.h**2 / (2 * sc_const.m_n * wl_m**2)

    # Convert to eV
    energy_ev = energy_j.to(unit="eV")

    return energy_ev


def energy_to_wavelength(energy: sc.Variable) -> sc.Variable:
    """
    Convert neutron energy to wavelength via de Broglie relation.

    Formula: λ = h / sqrt(2 * m_n * E)

    Parameters
    ----------
    energy : sc.Variable
        Energy with unit compatible with 'eV'

    Returns
    -------
    sc.Variable
        Wavelength with unit 'angstrom'

    Notes
    -----
    Uses scipy.constants (no hardcoded values).

    Examples
    --------
    >>> energy = sc.scalar(0.025, unit='eV')  # Thermal
    >>> wl = energy_to_wavelength(energy)
    >>> print(wl)  # ~1.8 Å
    """
    energy_j = energy.to(unit="J")

    # λ = h / sqrt(2 * m_n * E)
    wavelength_m = sc_const.h / sc.sqrt(2 * sc_const.m_n * energy_j)

    # Convert to Angstrom
    wavelength_angstrom = wavelength_m.to(unit="angstrom")

    return wavelength_angstrom


def create_tof_bins(config: BinningConfig, flight_path: Optional[sc.Variable] = None) -> sc.Variable:
    """
    Create TOF bin edges from binning configuration.

    Supports three binning strategies:
    1. bin_space='energy': Create energy bins, convert to TOF (reversed)
    2. bin_space='wavelength': Create wavelength bins, convert to TOF (ascending)
    3. bin_space='tof': Create TOF bins directly

    Parameters
    ----------
    config : BinningConfig
        Binning configuration specifying domain and range
    flight_path : sc.Variable, optional
        Flight path in meters. Required for energy/wavelength modes.

    Returns
    -------
    sc.Variable
        TOF bin edges with dimension 'tof' and unit 'ns'

    Examples
    --------
    >>> from neunorm.data_models.tof import BinningConfig
    >>> config = BinningConfig(bins=1000, bin_space='energy', energy_range=(1, 100))
    >>> L = sc.scalar(25.0, unit='m')
    >>> tof_bins = create_tof_bins(config, L)
    """
    if config.bin_space == "energy":
        if flight_path is None:
            raise ValueError("flight_path required for energy binning")
        return _energy_bins_to_tof(config, flight_path)

    elif config.bin_space == "wavelength":
        if flight_path is None:
            raise ValueError("flight_path required for wavelength binning")
        return _wavelength_bins_to_tof(config, flight_path)

    elif config.bin_space == "tof":
        return _create_tof_bins_direct(config)

    else:
        raise ValueError(f"Invalid bin_space: {config.bin_space}")


def _energy_bins_to_tof(config: BinningConfig, flight_path: sc.Variable) -> sc.Variable:
    """Create energy bins and convert to TOF bins (reversed)"""
    emin, emax = config.energy_range

    # Create energy bins
    if config.use_log_bin:
        energy_bins = sc.geomspace("energy", emin, emax, num=config.bins + 1, unit="eV")
    else:
        energy_bins = sc.linspace("energy", emin, emax, num=config.bins + 1, unit="eV")

    # Convert to TOF: t = L * sqrt(m_n / (2*E))
    energy_j = energy_bins.to(unit="J", copy=False)
    velocity = sc.sqrt(2.0 * energy_j / sc_const.m_n)
    tof_s = flight_path / velocity
    tof_ns = tof_s.to(unit="ns")

    # Reverse: high energy = low TOF
    tof_bins_reversed = sc.array(dims=["tof"], values=tof_ns.values[::-1].copy(), unit="ns")

    return tof_bins_reversed


def _wavelength_bins_to_tof(config: BinningConfig, flight_path: sc.Variable) -> sc.Variable:
    """Create wavelength bins and convert to TOF bins (ascending)"""
    wl_min, wl_max = config.wavelength_range

    # Create wavelength bins
    if config.use_log_bin:
        wl_bins = sc.geomspace("wavelength", wl_min, wl_max, num=config.bins + 1, unit="angstrom")
    else:
        wl_bins = sc.linspace("wavelength", wl_min, wl_max, num=config.bins + 1, unit="angstrom")

    # Convert to TOF: t = λ * m_n * L / h
    tof_s = wl_bins.to(unit="m") * sc_const.m_n * flight_path / sc_const.h

    tof_ns = tof_s.to(unit="ns")

    # NO reversal: low wavelength = low TOF (both ascending)
    return tof_ns.rename_dims({"wavelength": "tof"})


def _create_tof_bins_direct(config: BinningConfig) -> sc.Variable:
    """Create TOF bins directly in TOF space"""
    if config.tof_range is not None:
        t_min, t_max = config.tof_range
    else:
        # Default: full range (0 to 16.664 ms for 60 Hz SNS)
        t_min, t_max = 0, 16.664e6  # ns

    if config.use_log_bin:
        # Logarithmic TOF bins
        # Note: t_min must be > 0 for geomspace
        if t_min == 0:
            from neunorm.utils.constants import TPX3_CLOCK_NS

            t_min = TPX3_CLOCK_NS  # Start from one clock tick

        tof_bins = sc.geomspace("tof", t_min, t_max, num=config.bins + 1, unit="ns")
    else:
        # Linear TOF bins
        tof_bins = sc.linspace("tof", t_min, t_max, num=config.bins + 1, unit="ns")

    return tof_bins


def get_energy_histogram(hist_tof: sc.DataArray, flight_path: sc.Variable) -> sc.DataArray:
    """
    Convert TOF histogram to energy histogram.

    Converts TOF bin edges to energy and reverses data order
    (high TOF → low energy).

    Parameters
    ----------
    hist_tof : sc.DataArray
        Histogram with 'tof' dimension
    flight_path : sc.Variable
        Flight path in meters

    Returns
    -------
    sc.DataArray
        Histogram with 'energy' dimension (reversed)

    Notes
    -----
    Preserves variance if present. Both data and variance are reversed.
    """
    # Convert TOF edges to energy
    tof_edges = hist_tof.coords["tof"]
    energy_edges = tof_to_energy(tof_edges, flight_path)

    # Reverse data along TOF dimension (high TOF = low energy)
    hist_reversed = hist_tof.copy()

    # Reverse along first dimension (assume 'tof' is first)
    tof_dim_index = hist_tof.dims.index("tof")

    if tof_dim_index == 0:
        hist_reversed.values = hist_tof.values[::-1, ...].copy()
        if hist_tof.variances is not None:
            hist_reversed.variances = hist_tof.variances[::-1, ...].copy()
    else:
        # Handle other dimension orders if needed
        raise NotImplementedError("TOF must be first dimension for now")

    # Remove TOF coordinate
    del hist_reversed.coords["tof"]

    # Rename dimension
    hist_energy = hist_reversed.rename_dims({"tof": "energy"})

    # Assign energy coordinate (reversed to match data)
    energy_edges_reversed = sc.array(dims=["energy"], values=energy_edges.values[::-1].copy(), unit="eV")
    hist_energy.coords["energy"] = energy_edges_reversed

    return hist_energy


def get_wavelength_histogram(hist_tof: sc.DataArray, flight_path: sc.Variable) -> sc.DataArray:
    """
    Convert TOF histogram to wavelength histogram.

    Converts TOF bin edges to wavelength. NO reversal needed
    (low TOF → low wavelength, both ascending).

    Parameters
    ----------
    hist_tof : sc.DataArray
        Histogram with 'tof' dimension
    flight_path : sc.Variable
        Flight path in meters

    Returns
    -------
    sc.DataArray
        Histogram with 'wavelength' dimension

    Notes
    -----
    Preserves variance if present. No data reversal (unlike energy conversion).
    """
    # Convert TOF edges to wavelength
    tof_edges = hist_tof.coords["tof"]
    wavelength_edges = tof_to_wavelength(tof_edges, flight_path)

    # Copy histogram (no reversal needed)
    hist_wavelength = hist_tof.copy()

    # Remove TOF coordinate
    del hist_wavelength.coords["tof"]

    # Rename dimension
    hist_wavelength = hist_wavelength.rename_dims({"tof": "wavelength"})

    # Assign wavelength coordinate
    hist_wavelength.coords["wavelength"] = wavelength_edges.rename_dims({"tof": "wavelength"})

    return hist_wavelength

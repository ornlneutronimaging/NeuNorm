"""
Uncertainty quantification utilities for NeuNorm 2.0.

Provides functions for:
- Attaching Poisson variance to count data
- Adding systematic uncertainties
- Extracting uncertainties from variance

Uses scipp's automatic variance propagation through arithmetic operations.
"""

import numpy as np
import scipp as sc
from loguru import logger


def attach_poisson_variance(data: sc.DataArray) -> sc.DataArray:
    """
    Attach Poisson variance to count data.

    For Poisson counting statistics: σ²(N) = N (variance equals counts)

    Parameters
    ----------
    data : sc.DataArray
        Count data with unit='counts'

    Returns
    -------
    sc.DataArray
        Copy of data with .variances = values

    Raises
    ------
    ValueError
        If data unit is not 'counts'

    Notes
    -----
    Creates a copy - does not modify original data.
    If data already has variance, it will be overwritten (with warning).

    Examples
    --------
    >>> counts = sc.array(dims=['x'], values=[100, 200, 300], unit='counts')
    >>> data = sc.DataArray(data=counts)
    >>> data_with_var = attach_poisson_variance(data)
    >>> print(data_with_var.variances)  # [100, 200, 300]
    """
    if data.unit != sc.units.counts:
        raise ValueError(
            f"Poisson variance only valid for counts, got unit='{data.unit}'. "
            "Convert data to counts before attaching Poisson variance."
        )

    if data.variances is not None:
        logger.warning("Data already has variance - overwriting with Poisson variance (var = N)")

    data_copy = data.copy()

    # Scipp requires float data for variances - convert if needed
    if not np.issubdtype(data_copy.values.dtype, np.floating):
        logger.debug(f"Converting data from {data_copy.values.dtype} to float64 for variance support")
        data_copy.data = sc.array(dims=data_copy.dims, values=data_copy.values.astype(np.float64), unit=data_copy.unit)

    data_copy.variances = data_copy.values.copy()

    return data_copy


def add_systematic_variance(data: sc.DataArray, relative_uncertainty: float) -> sc.DataArray:
    """
    Add systematic uncertainty to data variance.

    Used for beam monitor corrections (proton charge, shutter counts)
    and other systematic uncertainties.

    Parameters
    ----------
    data : sc.DataArray
        Data with or without existing variance
    relative_uncertainty : float
        Relative uncertainty (fractional). Examples:
        - 0.005 = 0.5% (typical proton charge uncertainty)
        - 0.01 = 1.0%

    Returns
    -------
    sc.DataArray
        Copy with systematic variance added to existing variance

    Notes
    -----
    Systematic variance added in quadrature:
    var_total = var_existing + (relative_unc * value)²

    If no existing variance, only systematic is added.

    Examples
    --------
    >>> data = sc.DataArray(data=sc.array(dims=['x'], values=[100, 200], unit='counts'))
    >>> data.variances = np.array([100, 200])  # Poisson
    >>> # Add 0.5% systematic (e.g., proton charge)
    >>> data_sys = add_systematic_variance(data, 0.005)
    >>> # var_total = 100 + (0.005*100)² = 100.25
    """
    data_copy = data.copy()

    # Compute systematic variance
    systematic_var = (relative_uncertainty * data.values) ** 2

    # Add to existing variance (or create new)
    if data_copy.variances is None:
        data_copy.variances = systematic_var
    else:
        data_copy.variances = data_copy.variances + systematic_var

    return data_copy


def get_uncertainty(data: sc.DataArray) -> sc.DataArray:
    """
    Get standard deviation (σ) from variance (σ²).

    Parameters
    ----------
    data : sc.DataArray
        Data with .variances attached

    Returns
    -------
    sc.DataArray
        Standard deviation (sqrt of variance)

    Raises
    ------
    ValueError
        If data has no variance

    Notes
    -----
    Equivalent to scipp.stddevs() but with better error message.

    Examples
    --------
    >>> data = sc.DataArray(data=sc.array(dims=['x'], values=[100], unit='counts'))
    >>> data.variances = np.array([100])
    >>> uncertainty = get_uncertainty(data)
    >>> print(uncertainty.values)  # [10.0]
    """
    if data.variances is None:
        raise ValueError("Data has no variance. Use attach_poisson_variance() for count data.")

    return sc.stddevs(data)


def get_relative_uncertainty(data: sc.DataArray) -> sc.DataArray:
    """
    Get relative uncertainty (σ / value).

    Parameters
    ----------
    data : sc.DataArray
        Data with .variances attached

    Returns
    -------
    sc.DataArray
        Relative uncertainty (σ / value)

    Raises
    ------
    ValueError
        If data has no variance

    Examples
    --------
    >>> data = sc.DataArray(data=sc.array(dims=['x'], values=[100, 400], unit='counts'))
    >>> data.variances = np.array([100, 400])  # Poisson
    >>> rel_unc = get_relative_uncertainty(data)
    >>> # For Poisson: σ/N = √N/N = 1/√N
    >>> print(rel_unc.values)  # [0.1, 0.05]
    """
    uncertainty = get_uncertainty(data)
    return uncertainty / data

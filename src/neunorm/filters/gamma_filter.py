"""
Gamma filter to remove outliers from neutron imaging data.
"""

import numpy as np
import scipp as sc
from loguru import logger
from scipy import ndimage as ndi


def apply_gamma_filter(
    data: sc.DataArray, threshold_sigma: float = 5.0, kernel_size: int = 3, preserve_variance: bool = True
) -> sc.DataArray:
    """Remove gamma contamination by replacing outliers with local median.

    Formula:
    FOR each pixel (i, j):
    IF value > median(neighborhood) + k * σ(neighborhood):
        Replace with median(neighborhood)
        Update variance estimate

    Handles
    - Detect gamma spikes (statistical outliers above threshold)
    - Replace detected spikes with local median (3x3 or 5x5 neighborhood)
    - Support both single images and stacks
    - Configurable threshold (default: median + k×σ)

    Parameters
    ----------
    data : sc.DataArray
        Input neutron imaging data.
    threshold_sigma : float
        Number of standard deviations to use as the threshold for identifying outliers.
    kernel_size : int
        Size of the local neighborhood for computing the median.
    preserve_variance : bool
        If True, keep the original variance for all pixels.
        If False, update the variance of outliers to the local median variance.

    Returns
    -------
    sc.DataArray
        Gamma-filtered data with propagated variance if requested.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")
    if threshold_sigma < 0:
        raise ValueError("threshold_sigma must be >= 0.")

    logger.debug("Applying gamma filter: threshold_sigma={}, kernel_size={}", threshold_sigma, kernel_size)

    # Apply filter over spatial axes x/y
    ndim = data.data.ndim
    size = [1] * ndim
    dims = data.dims
    if "y" in dims and "x" in dims:
        size[dims.index("y")] = kernel_size
        size[dims.index("x")] = kernel_size
    else:
        raise ValueError("Input data must have 'x' and 'y' dimensions for spatial filtering.")

    # create footprint for neighborhood with center excluded
    footprint = np.full(size, True, dtype=bool)
    footprint[tuple((s // 2) for s in size)] = False

    # calculate local median and std using scipy filters
    values = data.data.values
    local_std = ndi.generic_filter(values, np.std, footprint=footprint, mode="nearest")
    local_median = ndi.median_filter(values, footprint=footprint, mode="nearest")
    local_threshold = local_median + threshold_sigma * local_std

    # Identify outliers
    outlier_mask = values > local_threshold

    logger.debug("Identified {} outliers in data of shape {}", outlier_mask.sum(), values.shape)

    # Replace outliers with local median
    filtered_values = values.copy()
    filtered_values[outlier_mask] = local_median[outlier_mask]

    # Handle variance
    input_variances = data.data.variances
    filtered_variances = input_variances.copy() if input_variances is not None else None

    if not preserve_variance and input_variances is not None:
        # Local variance estimate from existing per-pixel variances
        local_var = ndi.median_filter(filtered_variances, footprint=footprint, mode="nearest")
        filtered_variances[outlier_mask] = local_var[outlier_mask]

    out = data.copy(deep=False)
    out.data = sc.array(
        dims=dims,
        values=filtered_values,
        variances=filtered_variances,
        unit=data.unit,
    )

    return out

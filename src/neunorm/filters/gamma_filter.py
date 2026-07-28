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

    Formula::

        FOR each pixel (i, j):
            IF value > median(neighborhood) + k * σ(neighborhood):
                Replace with median(neighborhood)
                Update variance estimate

    The filter will:

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

    logger.info("Applying gamma filter: threshold_sigma={}, kernel_size={}", threshold_sigma, kernel_size)

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
    # Compute local standard deviation using convolution to avoid per-pixel Python callbacks.
    # Equivalent to local_std = ndi.generic_filter(values, np.std, footprint=footprint, mode="nearest")
    kernel = footprint.astype(float)
    count = kernel.sum()
    local_mean = ndi.convolve(values, kernel, mode="nearest") / count
    local_mean_sq = ndi.convolve(values**2, kernel, mode="nearest") / count
    # Numerical guard: clip small negative variances due to floating point
    local_var = np.clip(local_mean_sq - local_mean**2, 0, None)
    local_std = np.sqrt(local_var)
    # Calculate local median using scipy's median filter
    local_median = ndi.median_filter(values, footprint=footprint, mode="nearest")
    # Calculate threshold for outlier detection
    local_threshold = local_median + threshold_sigma * local_std

    # Identify outliers
    outlier_mask = values > local_threshold
    outlier_count = np.sum(outlier_mask)

    logger.info("Identified {} outliers in data of shape {}", outlier_count, values.shape)

    # Replace outliers with local median
    filtered_values = values.copy()
    filtered_values[outlier_mask] = local_median[outlier_mask]

    # Handle variance
    input_variances = data.data.variances
    filtered_variances = input_variances.copy() if input_variances is not None else None

    if not preserve_variance and input_variances is not None and outlier_count > 0:
        # Recalculate variance for outliers from local neighborhood.
        # This is an approximation of the variance of the median.
        # Use Var(median) ≈ (π / (2n)) * mean_variance

        # Pad input variances to handle edge cases when extracting neighborhood.
        # Matching the 'nearest' mode used in the filters.
        input_variances_padded = np.pad(input_variances, [(s // 2, s // 2) for s in size], mode="edge")

        for idx in np.ndindex(outlier_mask.shape):
            if outlier_mask[idx]:
                neighbor_indices = tuple(slice(i, i + s) for i, s in zip(idx, size))
                # extract variances of the neighbors using the same footprint as the median filter
                neighbor_variances = input_variances_padded[neighbor_indices][footprint]
                mean_variance = neighbor_variances.mean()
                filtered_variances[idx] = (np.pi / (2 * len(neighbor_variances))) * mean_variance
                logger.debug("Updating variance for outlier at index {} to {}", idx, filtered_variances[idx])

    out = data.copy(deep=False)
    out.data = sc.array(
        dims=dims,
        values=filtered_values,
        variances=filtered_variances,
        unit=data.unit,
    )

    return out


def replace_pixels_with_frame_local_median(data: sc.DataArray, pixel_mask: sc.Variable, kernel_size: int = 3) -> int:
    """Replace flagged pixels of every spectral frame with the frame's local spatial median.

    Companion of :func:`apply_gamma_filter` for **sparse TOF stacks** (e.g. VENUS TPX1
    per-bin histograms, median counts per pixel per bin ≈ 0), where a frame-by-frame
    outlier filter cannot work: the local median of a nearly empty neighborhood is 0,
    so every real event would be flagged. Instead the gamma-contaminated pixels are
    detected once on the spectrally *integrated* counts (see
    :func:`neunorm.tof.pixel_detector.detect_hot_pixels`) and this function replaces
    their values — in each spectral frame — with the median of the ``kernel_size ×
    kernel_size`` spatial neighborhood of that frame. Variances, when present, get the
    neighborhood median of the variances the same way.

    The replacement happens **in place** on ``data``.

    Parameters
    ----------
    data : sc.DataArray
        Stack with one spectral dimension (``tof``/``energy``/``wavelength``/``N_image``)
        and two spatial dimensions.
    pixel_mask : sc.Variable
        Boolean mask over the spatial dimensions — True = replace this pixel.
    kernel_size : int
        Odd size (>= 3) of the spatial neighborhood used for the median.

    Returns
    -------
    int
        Number of pixels replaced.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")

    spatial_dims = list(pixel_mask.dims)
    spectral_dims = [dim for dim in data.dims if dim not in spatial_dims]
    if len(spectral_dims) != 1:
        raise ValueError(f"Expected exactly one spectral dimension, got {spectral_dims} (data dims {data.dims}).")

    # Views with the spectral axis first and the spatial axes in the mask's order:
    # np.transpose returns views, so writing through them updates `data` in place.
    order = [data.dims.index(spectral_dims[0])] + [data.dims.index(dim) for dim in spatial_dims]
    values = np.transpose(data.values, order)
    variances = np.transpose(data.variances, order) if data.variances is not None else None
    mask = pixel_mask.values

    half = kernel_size // 2
    n_rows, n_cols = mask.shape
    flagged = np.argwhere(mask)
    for row, col in flagged:
        row_slice = slice(max(0, row - half), min(n_rows, row + half + 1))
        col_slice = slice(max(0, col - half), min(n_cols, col + half + 1))
        # Exclude the flagged pixels themselves from the neighborhood so adjacent
        # gamma pixels do not feed each other's replacement value.
        good = ~mask[row_slice, col_slice]
        if not good.any():
            values[:, row, col] = 0.0
            if variances is not None:
                variances[:, row, col] = 0.0
            continue
        neighborhood = values[:, row_slice, col_slice][:, good]
        values[:, row, col] = np.median(neighborhood, axis=1)
        if variances is not None:
            neighborhood_var = variances[:, row_slice, col_slice][:, good]
            variances[:, row, col] = np.median(neighborhood_var, axis=1)

    logger.info("Replaced {} flagged pixel(s) with the frame-local median (kernel {})", len(flagged), kernel_size)
    return int(len(flagged))

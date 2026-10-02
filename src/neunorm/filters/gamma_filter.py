"""
Gamma filter to remove outliers from neutron imaging data.
"""

import numpy as np
import scipp as sc
from loguru import logger
from scipy import ndimage as ndi

from neunorm.utils.progress import STAGE_GAMMA_FILTER, ProgressLike, resolve_progress

#: Steps `apply_gamma_filter` counts. The optional outlier-variance recomputation is not among them:
#: whether it runs depends on the outlier count, which is only known after step 4, so it is announced
#: with a note instead of being counted — a total that cannot be computed up front would leave the bar
#: either short or overshooting.
#:
#: Public because a caller that hands this function a pre-bound ``ProgressReporter`` — a pipeline
#: reporting this as one stage of a longer run — has to declare the stage total itself:
#: ``resolve_progress`` deliberately does not let a callee re-bind a total. Reading it from here is
#: what keeps the pipeline's declared total and the ticks this function actually emits from drifting
#: apart.
GAMMA_FILTER_STEPS = 4

#: Elements per block when the squared local mean is subtracted from the local mean of squares.
_BLOCK_SIZE = 1 << 20


def _subtract_squared_mean(mean_sq: np.ndarray, local_sum: np.ndarray, count: float) -> None:
    """Subtract ``(local_sum / count) ** 2`` from ``mean_sq`` in place, ``_BLOCK_SIZE`` elements at a time.

    Both arrays must be C-contiguous and of one shape, so that their flat reshapes are views. Each block
    of the mean is computed as ``local_sum / count`` with that expression's dtype, so the result is
    bit-identical to ``mean_sq - (local_sum / count) ** 2`` while only one block of the mean exists.
    """
    flat_var = mean_sq.reshape(-1)
    flat_sum = local_sum.reshape(-1)
    for start in range(0, flat_var.size, _BLOCK_SIZE):
        block = slice(start, start + _BLOCK_SIZE)
        local_mean = flat_sum[block] / count
        flat_var[block] -= np.square(local_mean, out=local_mean)
        del local_mean


def apply_gamma_filter(
    data: sc.DataArray,
    threshold_sigma: float = 5.0,
    kernel_size: int = 3,
    preserve_variance: bool = True,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_GAMMA_FILTER,
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
    progress : bool or callable, optional
        Progress reporting, off by default. ``True`` draws a :mod:`tqdm` bar; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent`. There is no item axis — the filter is a
        sequence of whole-array operations — so it reports four named steps, the third of which is
        the ``scipy.ndimage.median_filter`` that dominates the cost. Each is named before it runs and
        counted after it returns. This stage is enabled by default on the CCD and MARS TPX3
        pipelines and is the slowest per frame there, so without reporting it is a long silence.
        See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_GAMMA_FILTER``.

    Returns
    -------
    sc.DataArray
        Gamma-filtered data with propagated variance if requested.

    Notes
    -----
    Working memory, beyond the input, peaks at one float64 array, one array of the data's dtype and
    one boolean array of the data's shape, about 3.3 times the size of a float32 stack's values. That
    bound leaves out the scratch for one block of the local mean, a fixed 2**20 float64 values (8 MiB)
    whatever the data's size. While the output, one copy of the values and variances, is made, the
    boolean array and the outliers' replacement values, one element of the data's dtype per outlier,
    are still alive. ``preserve_variance=False`` with outliers also pads a copy of the variances.
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
    with resolve_progress(progress, stage, total=GAMMA_FILTER_STEPS) as report:
        # Compute local standard deviation using convolution to avoid per-pixel Python callbacks.
        # Equivalent to local_std = ndi.generic_filter(values, np.std, footprint=footprint, mode="nearest")
        kernel = footprint.astype(float)
        count = kernel.sum()
        report.note("local mean")
        # The float64 local mean of squares becomes the variance, then the deviation, then the threshold, in place.
        local_var = ndi.convolve(values**2, kernel, mode="nearest") / count
        local_sum = ndi.convolve(values, kernel, mode="nearest")
        report()
        report.note("local deviation")
        _subtract_squared_mean(local_var, local_sum, count)
        del local_sum
        # Numerical guard: clip small negative variances due to floating point
        local_std = np.sqrt(np.clip(local_var, 0, None, out=local_var), out=local_var)
        del local_var
        report()
        # Calculate local median using scipy's median filter. Named on its own because it dominates
        # the stage — roughly three quarters of it — so a bar parked here is not stuck, just slow.
        report.note("local median")
        local_median = ndi.median_filter(values, footprint=footprint, mode="nearest")
        report()
        # Calculate threshold for outlier detection, in the local deviation's buffer
        report.note("detecting and replacing outliers")
        local_threshold = local_std
        del local_std
        local_threshold *= threshold_sigma
        local_threshold += local_median

        # Identify outliers
        outlier_mask = values > local_threshold
        del local_threshold
        outlier_count = np.sum(outlier_mask)

        logger.info("Identified {} outliers in data of shape {}", outlier_count, values.shape)

        # Replace outliers with local median, in the one copy of the data that becomes the output
        replacement = local_median[outlier_mask]
        del local_median
        filtered = data.data.copy()
        filtered.values[outlier_mask] = replacement
        del replacement
        report()

        # Handle variance
        input_variances = data.data.variances

        if not preserve_variance and input_variances is not None and outlier_count > 0:
            # Recalculate variance for outliers from local neighborhood.
            # This is an approximation of the variance of the median.
            # Use Var(median) ≈ (π / (2n)) * mean_variance
            #
            # Announced rather than counted: whether this runs depends on the outlier count, which is
            # not known until the step above finishes, so it cannot be part of a total fixed up front.
            # It is a per-outlier Python loop, so it can be the slowest part of the call.
            report.note(f"recomputing variance for {outlier_count} outliers")

            # Pad input variances to handle edge cases when extracting neighborhood.
            # Matching the 'nearest' mode used in the filters.
            input_variances_padded = np.pad(input_variances, [(s // 2, s // 2) for s in size], mode="edge")
            filtered_variances = filtered.variances

            for idx in np.ndindex(outlier_mask.shape):
                if outlier_mask[idx]:
                    neighbor_indices = tuple(slice(i, i + s) for i, s in zip(idx, size))
                    # extract variances of the neighbors using the same footprint as the median filter
                    neighbor_variances = input_variances_padded[neighbor_indices][footprint]
                    mean_variance = neighbor_variances.mean()
                    filtered_variances[idx] = (np.pi / (2 * len(neighbor_variances))) * mean_variance
                    logger.debug("Updating variance for outlier at index {} to {}", idx, filtered_variances[idx])

        out = data.copy(deep=False)
        out.data = filtered

        return out

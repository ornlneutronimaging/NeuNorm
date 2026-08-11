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
        local_mean = ndi.convolve(values, kernel, mode="nearest") / count
        local_mean_sq = ndi.convolve(values**2, kernel, mode="nearest") / count
        report()
        # Numerical guard: clip small negative variances due to floating point
        report.note("local deviation")
        local_var = np.clip(local_mean_sq - local_mean**2, 0, None)
        local_std = np.sqrt(local_var)
        report()
        # Calculate local median using scipy's median filter. Named on its own because it dominates
        # the stage — roughly three quarters of it — so a bar parked here is not stuck, just slow.
        report.note("local median")
        local_median = ndi.median_filter(values, footprint=footprint, mode="nearest")
        report()
        # Calculate threshold for outlier detection
        report.note("detecting and replacing outliers")
        local_threshold = local_median + threshold_sigma * local_std

        # Identify outliers
        outlier_mask = values > local_threshold
        outlier_count = np.sum(outlier_mask)

        logger.info("Identified {} outliers in data of shape {}", outlier_count, values.shape)

        # Replace outliers with local median
        filtered_values = values.copy()
        filtered_values[outlier_mask] = local_median[outlier_mask]
        report()

        # Handle variance
        input_variances = data.data.variances
        filtered_variances = input_variances.copy() if input_variances is not None else None

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

"""
Bad pixel detection for TOF neutron imaging.

Provides tools for detecting dead and hot pixels in event-mode and histogram
neutron imaging data. Uses MAD (Median Absolute Deviation) for robust outlier
detection.

Ported from venus_tof.masking with generalizations for tof/energy/wavelength dimensions.
"""

from typing import Dict

import numpy as np
import scipp as sc
from loguru import logger


def detect_dead_pixels(hist: sc.DataArray) -> sc.Variable:
    """
    Detect dead pixels (zero counts across all spectral bins).

    Dead pixels can be:
    - Permanently damaged (physical failure)
    - Temporarily disabled (radiation damage, recoverable via power cycle)

    Parameters
    ----------
    hist : sc.DataArray
        3D histogram with dimensions (tof/energy/wavelength, x, y)
        Spectral dimension can be any of: 'tof', 'energy', 'wavelength'

    Returns
    -------
    sc.Variable
        Boolean mask with dimensions (x, y) where True = dead pixel

    Examples
    --------
    >>> dead_mask = detect_dead_pixels(hist_ob)
    >>> print(f"Found {dead_mask.values.sum()} dead pixels")
    """
    # Detect spectral dimension (first dim that's not x or y)
    spectral_dims = [dim for dim in hist.dims if dim not in ["x", "y"]]

    if len(spectral_dims) == 0:
        raise ValueError("Histogram must have a spectral dimension (tof, energy, or wavelength)")

    spectral_dim = spectral_dims[0]  # Use first spectral dimension

    # Sum over spectral dimension to get total counts per pixel
    spatial = hist.sum(spectral_dim)

    # Dead pixels have exactly zero counts
    dead = spatial.data == sc.scalar(0, unit=spatial.unit)

    return dead


def detect_hot_pixels(hist: sc.DataArray, sigma: float = 5.0) -> sc.Variable:
    """
    Detect hot pixels using MAD (Median Absolute Deviation) threshold.

    Hot pixels are caused by radiation damage and generate fake events uniformly
    across all spectral bins, resulting in abnormally high spatial sum values.

    MAD is more robust than standard deviation for outlier detection because it
    is not affected by the outliers themselves.

    Parameters
    ----------
    hist : sc.DataArray
        3D histogram with dimensions (tof/energy/wavelength, x, y)
    sigma : float, optional
        Threshold in units of MAD (default: 5.0)
        Common values:
        - 3.0: Aggressive (catches more pixels, may have false positives)
        - 5.0: Balanced (recommended for most cases)
        - 10.0: Conservative (only catches extreme outliers)

    Returns
    -------
    sc.Variable
        Boolean mask with dimensions (x, y) where True = hot pixel

    Notes
    -----
    The MAD threshold is converted to approximate standard deviations using
    the scale factor 1.4826, which makes MAD equivalent to sigma for normally
    distributed data.

    Formula: threshold = median + sigma × MAD × 1.4826

    Examples
    --------
    >>> hot_mask = detect_hot_pixels(hist_ta, sigma=5.0)
    >>> print(f"Found {hot_mask.values.sum()} hot pixels")

    >>> # Try different thresholds
    >>> hot_conservative = detect_hot_pixels(hist_ta, sigma=10.0)
    >>> hot_aggressive = detect_hot_pixels(hist_ta, sigma=3.0)
    """
    # Detect spectral dimension
    spectral_dims = [dim for dim in hist.dims if dim not in ["x", "y"]]

    if len(spectral_dims) == 0:
        raise ValueError("Histogram must have a spectral dimension (tof, energy, or wavelength)")

    spectral_dim = spectral_dims[0]

    # Sum over spectral dimension to get total counts per pixel
    spatial = hist.sum(spectral_dim)
    values = spatial.values.flatten()

    # Remove zeros to avoid skewing statistics
    values_nonzero = values[values > 0]

    if len(values_nonzero) == 0:
        # All pixels are dead, no hot pixels possible
        logger.warning("All pixels have zero counts, cannot detect hot pixels")
        return sc.array(dims=["x", "y"], values=np.zeros(spatial.shape, dtype=bool))

    # Calculate median and MAD
    median = np.median(values_nonzero)
    mad = np.median(np.abs(values_nonzero - median))

    # Scale factor 1.4826 converts MAD to approximate standard deviation
    # for normally distributed data
    threshold = median + sigma * mad * 1.4826

    logger.debug(f"Hot pixel detection: median={median:.1f}, MAD={mad:.1f}, threshold={threshold:.1f} (sigma={sigma})")

    # Hot pixels exceed threshold
    hot = spatial.data > sc.scalar(threshold, unit=spatial.unit)

    return hot


def detect_bad_pixels_for_transmission(
    sample: sc.DataArray,
    ob: sc.DataArray,
    sigma: float = 5.0,
) -> Dict[str, sc.Variable]:
    """
    Detect bad pixels from both sample and open beam for transmission imaging.

    For transmission imaging (T = Sample/OB), a pixel is invalid if it's
    problematic in EITHER dataset:
    - Hot pixel in OB → denominator wrong → T wrong
    - Hot pixel in sample → numerator wrong → T wrong
    - Dead pixel in OB → division by zero → T undefined
    - Dead pixel in sample → zero numerator → T=0 (looks like perfect attenuation)

    This function applies 4 separate masks to each histogram:
    - dead_pixels_sample
    - hot_pixels_sample
    - dead_pixels_ob
    - hot_pixels_ob

    Scipp automatically combines all masks with OR during operations, so any
    pixel flagged by any mask will be excluded from calculations.

    Parameters
    ----------
    sample : sc.DataArray
        Sample histogram (tof/energy/wavelength, x, y)
    ob : sc.DataArray
        Open beam reference histogram (tof/energy/wavelength, x, y)
    sigma : float, optional
        MAD threshold for hot pixel detection (default: 5.0)

    Returns
    -------
    dict
        Dictionary containing individual masks for diagnostics:
        - 'dead_sample': Dead pixels in sample
        - 'hot_sample': Hot pixels in sample
        - 'dead_ob': Dead pixels in open beam
        - 'hot_ob': Hot pixels in open beam

    Examples
    --------
    >>> masks = detect_bad_pixels_for_transmission(hist_ta, hist_ob, sigma=5.0)
    >>>
    >>> # Both histograms now have all 4 masks applied
    >>> print(list(hist_ta.masks.keys()))
    >>>
    >>> # Calculate transmission (masks automatically applied by scipp)
    >>> transmission = hist_ta / hist_ob

    Notes
    -----
    The function modifies the input histograms in-place by adding masks to their
    .masks dictionaries. The original data values are not changed.
    """
    logger.info("Starting bad pixel detection for transmission imaging")

    # Detect from sample
    logger.info("Detecting bad pixels in sample...")
    dead_sample = detect_dead_pixels(sample)
    hot_sample = detect_hot_pixels(sample, sigma=sigma)

    # Detect from open beam
    logger.info("Detecting bad pixels in open beam...")
    dead_ob = detect_dead_pixels(ob)
    hot_ob = detect_hot_pixels(ob, sigma=sigma)

    # Report findings
    n_dead_sample = int(dead_sample.values.sum())
    n_hot_sample = int(hot_sample.values.sum())
    n_dead_ob = int(dead_ob.values.sum())
    n_hot_ob = int(hot_ob.values.sum())

    logger.info("Mask detection results:")
    logger.info("  Sample:")
    logger.info(f"    Dead pixels: {n_dead_sample}")
    logger.info(f"    Hot pixels:  {n_hot_sample}")
    logger.info("  Open beam:")
    logger.info(f"    Dead pixels: {n_dead_ob}")
    logger.info(f"    Hot pixels:  {n_hot_ob}")

    # Calculate total unique bad pixels (union of all masks)
    combined = dead_sample.values | hot_sample.values | dead_ob.values | hot_ob.values
    n_total = int(combined.sum())
    total_pixels = dead_sample.values.size
    fraction = n_total / total_pixels * 100
    logger.info(f"  Total bad pixels: {n_total} ({fraction:.2f}% of detector)")

    # Apply all 4 masks to both histograms
    # Sample gets all 4 masks
    sample.masks["dead_pixels_sample"] = dead_sample
    sample.masks["hot_pixels_sample"] = hot_sample
    sample.masks["dead_pixels_ob"] = dead_ob
    sample.masks["hot_pixels_ob"] = hot_ob

    # Open beam gets all 4 masks
    ob.masks["dead_pixels_sample"] = dead_sample
    ob.masks["hot_pixels_sample"] = hot_sample
    ob.masks["dead_pixels_ob"] = dead_ob
    ob.masks["hot_pixels_ob"] = hot_ob

    logger.success("Masks applied to both histograms")

    # Return masks for diagnostics/visualization
    return {
        "dead_sample": dead_sample,
        "hot_sample": hot_sample,
        "dead_ob": dead_ob,
        "hot_ob": hot_ob,
    }

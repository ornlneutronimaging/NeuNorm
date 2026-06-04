"""
Air region correction.
"""

import scipp as sc
from loguru import logger


def apply_air_region_correction(
    transmission: sc.DataArray,
    air_roi: tuple[
        int, int, int, int
    ],  # (x0, y0, x1, y1) with x1, y1 as exclusive stop indices (Python slice semantics)
) -> sc.DataArray:
    """Scale transmission so air region has mean = 1.0.

    Requirements
    ------------

    - Calculate mean transmission in user-specified air region
    - Scale entire image so air region = 1.0
    - Support per-image correction (radiography) and per-TOF-bin correction (hyperspectral)
    - Propagate uncertainty from air region mean

    Formula
    -------

    T_final = T / mean(T[air_ROI])

    Uncertainty:
    σ_T_final = T_final × √[(σ_T/T)² + (σ_air/<T_air>)²]


    Parameters
    ----------
    transmission : sc.DataArray
        Normalized transmission (after OB correction)
    air_roi : tuple[int, int, int, int]
        Coordinates of the air region (x0, y0, x1, y1), where x1 and y1 are exclusive upper bound

    """

    logger.info(f"Applying air region correction with ROI: {air_roi}")

    if len(air_roi) != 4 or not all(isinstance(i, int) for i in air_roi):
        raise ValueError("ROI must be a tuple of 4 integers (x0, y0, x1, y1)")

    x0, y0, x1, y1 = air_roi

    # Validate ROI
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid ROI: (x0, y0, x1, y1) must satisfy 0 <= x0 < x1 and 0 <= y0 < y1")

    # Get current dimensions
    if "x" not in transmission.dims or "y" not in transmission.dims:
        raise ValueError("DataArray must have 'x' and 'y' dimensions for ROI cropping")

    # Validate ROI against current sizes
    if x1 > transmission.sizes["x"] or y1 > transmission.sizes["y"]:
        raise ValueError(
            f"ROI (x1={x1}, y1={y1}) exceeds data size (x={transmission.sizes['x']}, y={transmission.sizes['y']})"
        )

    # Extract air region
    air_region = transmission["x", slice(x0, x1)]["y", slice(y0, y1)]

    # Calculate mean transmission in air region
    mean_air = sc.mean(air_region, dim=["x", "y"])
    if transmission.variances is not None:
        mean_air_variance = sc.variances(mean_air)
        mean_air.variances = None  # Temporarily remove variance to avoid issues with division

    # Scale entire image so mean of the air region = 1.0
    corrected_transmission = transmission / mean_air

    # Propagate uncertainty from air region mean
    if transmission.variances is not None:
        variances = corrected_transmission**2 * (
            sc.variances(transmission) / transmission**2 + mean_air_variance / mean_air**2
        )

        corrected_transmission.variances = variances.values

    logger.success("✓ Air region correction applied")
    return corrected_transmission

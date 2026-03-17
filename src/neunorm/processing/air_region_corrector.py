"""
Air region correction.
"""

import scipp as sc
from loguru import logger


def apply_air_region_correction(
    transmission: sc.DataArray,
    air_roi: tuple[int, int, int, int],  # (x0, y0, x1, y1)
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
        Coordinates of the air region (x0, y0, x1, y1)
    """

    logger.info(f"Applying air region correction with ROI: {air_roi}")

    x0, y0, x1, y1 = air_roi

    # Extract air region
    air_region = transmission["x", slice(x0, x1)]["y", slice(y0, y1)]

    # Calculate mean transmission in air region
    mean_air = sc.mean(air_region, dim=["x", "y"])
    mean_air_variance = sc.variances(mean_air)
    mean_air.variances = None  # Temporarily remove variance to avoid issues with division

    # Scale entire image so mean of the air region = 1.0
    corrected_transmission = transmission / mean_air

    # Propagate uncertainty from air region mean
    variances = corrected_transmission**2 * (
        sc.variances(transmission) / transmission**2 + mean_air_variance / mean_air**2
    )

    corrected_transmission.variances = variances.values

    logger.success("✓ Air region correction applied")
    return corrected_transmission

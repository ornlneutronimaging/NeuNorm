"""
Function for cropping spatial dimensions to a region of interest (ROI).
"""

import scipp as sc
from loguru import logger


def apply_roi(
    data: sc.DataArray,
    roi: tuple[int, int, int, int],  # (x0, y0, x1, y1)
) -> sc.DataArray:
    """Crop spatial dimensions to ROI.

    Crop to specified ROI: (x0, y0, x1, y1)
    Work with 2D, 3D, and 4D arrays (preserve other dimensions)
    Update coordinate arrays if present
    Validate ROI is within bounds

    Parameters
    ----------
    data : sc.DataArray
        Input data array to be cropped.
    roi : tuple[int, int, int, int]
        Region of interest specified as (x0, y0, x1, y1).

    Returns
    -------
    sc.DataArray
        Cropped data array with updated coordinates.
    """

    logger.info("Applying ROI: {}", roi)

    if len(roi) != 4:
        raise ValueError("ROI must be a tuple of 4 integers (x0, y0, x1, y1)")

    x0, y0, x1, y1 = roi

    if not all(isinstance(i, int) for i in roi):
        raise ValueError("ROI must be a tuple of 4 integers (x0, y0, x1, y1)")

    # Validate ROI
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid ROI: (x0, y0, x1, y1) must satisfy 0 <= x0 < x1 and 0 <= y0 < y1")

    # Get current dimensions
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError("DataArray must have 'x' and 'y' dimensions for ROI cropping")

    # Validate ROI against current sizes
    if x1 > data.sizes["x"] or y1 > data.sizes["y"]:
        raise ValueError(f"ROI (x1={x1}, y1={y1}) exceeds data size (x={data.sizes['x']}, y={data.sizes['y']})")

    # Create slices for cropping
    x_slice = slice(x0, x1)
    y_slice = slice(y0, y1)

    # Crop the DataArray
    return data["x", x_slice]["y", y_slice].copy()  # return a copy so it's not read-only

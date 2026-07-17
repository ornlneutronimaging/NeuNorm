"""
Function for cropping spatial dimensions to a region of interest (ROI).
"""

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import MaskROI, RegionLike, as_roi_bounds


def apply_roi(
    data: sc.DataArray,
    roi: RegionLike,  # (x0, y0, x1, y1) tuple, an ROI, or a MaskROI
) -> sc.DataArray:
    """Crop spatial dimensions to a region of interest.

    Crop to specified ROI: (x0, y0, x1, y1)
    Work with 2D, 3D, and 4D arrays (preserve other dimensions)
    Update coordinate arrays if present
    Validate ROI is within bounds

    Arrays are rectangular, so an arbitrary-shape :class:`~neunorm.data_models.roi.MaskROI` crops
    to the selection's **bounding box** and attaches the outside-selection pixels as a scipp
    exclusion mask named ``"outside_roi"``: every downstream mask-aware statistic (background-ROI
    pooling, air-region correction, mean/sum reductions) then automatically ignores pixels outside
    the region. Cropping an already-mask-cropped array ORs the exclusions (the selections
    intersect). The mask is persisted by ``write_hdf5`` (``/masks/outside_roi``) and folded into
    the merged TIFF ``scitiff-mask``. Rectangular ROIs crop exactly as before, with no mask
    attached.

    Parameters
    ----------
    data : sc.DataArray
        Input data array to be cropped.
    roi : ROI, MaskROI, or tuple[int, int, int, int]
        Region of interest as an :class:`~neunorm.data_models.roi.ROI` (e.g.
        ``ROI(x0=10, y0=20, x1=30, y1=40)`` or ``ROI(x0=10, y0=20, width=20, height=20)``), an
        arbitrary-shape :class:`~neunorm.data_models.roi.MaskROI` (selection mask: 1 = pixel in the
        region), or a bare ``(x0, y0, x1, y1)`` tuple with exclusive stop indices.

    Returns
    -------
    sc.DataArray
        Cropped data array with updated coordinates (and, for a ``MaskROI``, the ``outside_roi``
        exclusion mask).
    """
    if isinstance(roi, MaskROI):
        return _apply_mask_roi(data, roi)
    roi = as_roi_bounds(roi)

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


def _apply_mask_roi(data: sc.DataArray, roi: MaskROI) -> sc.DataArray:
    """Bounding-box crop + ``outside_roi`` exclusion mask for an arbitrary-shape region."""
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError("DataArray must have 'x' and 'y' dimensions for ROI cropping")
    ny, nx = roi.shape
    if ny != data.sizes["y"] or nx != data.sizes["x"]:
        raise ValueError(
            f"roi MaskROI selection shape (ny={ny}, nx={nx}) does not match data size "
            f"(y={data.sizes['y']}, x={data.sizes['x']})"
        )
    x0, y0, x1, y1 = roi.bounding_box()
    logger.info(
        "Applying MaskROI: bbox ({}, {}, {}, {}), {} selected pixels; outside pixels masked as 'outside_roi'",
        x0,
        y0,
        x1,
        y1,
        roi.n_selected,
    )
    cropped = data["x", x0:x1]["y", y0:y1].copy()
    outside = sc.array(dims=["y", "x"], values=np.ascontiguousarray(~roi.selection[y0:y1, x0:x1]))
    if "outside_roi" in cropped.masks:
        # repeat mask-crop: exclusions OR together (the selections intersect)
        cropped.masks["outside_roi"] = cropped.masks["outside_roi"] | outside
    else:
        cropped.masks["outside_roi"] = outside
    return cropped

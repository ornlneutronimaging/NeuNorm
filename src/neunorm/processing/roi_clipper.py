"""
Function for cropping spatial dimensions to a region of interest (ROI).
"""

import scipp as sc
from loguru import logger

from neunorm.data_models.roi import ROILike, _checked_crop_bounds, _crop_fit_message


def apply_roi(
    data: sc.DataArray,
    roi: ROILike,  # (x0, y0, x1, y1) tuple or an ROI
) -> sc.DataArray:
    """Crop spatial dimensions to a rectangular region of interest.

    Crop to specified ROI: (x0, y0, x1, y1)
    Work with 2D, 3D, and 4D arrays (preserve other dimensions)
    Update coordinate arrays if present
    Validate ROI is within bounds

    Cropping always yields a rectangular array, so only a rectangular ROI is accepted here. An
    arbitrary-shape :class:`~neunorm.data_models.roi.MaskROI` is not a crop region — pass it to the
    region-statistics operations instead (``background_roi=`` / ``air_roi=`` /
    :func:`~neunorm.processing.air_region_corrector.apply_air_region_correction` /
    :func:`~neunorm.processing.normalizer.normalize_transmission`), which average over the selected
    pixels without resizing the image.

    Parameters
    ----------
    data : sc.DataArray
        Input data array to be cropped.
    roi : ROI or tuple[int, int, int, int]
        Region of interest as an :class:`~neunorm.data_models.roi.ROI` (e.g.
        ``ROI(x0=10, y0=20, x1=30, y1=40)`` or ``ROI(x0=10, y0=20, width=20, height=20)``) or a bare
        ``(x0, y0, x1, y1)`` tuple with exclusive stop indices.

    Returns
    -------
    sc.DataArray
        Cropped (rectangular) data array with updated coordinates.
    """
    x0, y0, x1, y1 = _checked_crop_bounds(roi, caller="apply_roi")

    logger.info("Applying ROI: {}", (x0, y0, x1, y1))

    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError("DataArray must have 'x' and 'y' dimensions for ROI cropping")

    message = _crop_fit_message((x0, y0, x1, y1), ny=data.sizes["y"], nx=data.sizes["x"])
    if message is not None:
        raise ValueError(message)

    # Create slices for cropping
    x_slice = slice(x0, x1)
    y_slice = slice(y0, y1)

    # Crop the DataArray
    return data["x", x_slice]["y", y_slice].copy()  # return a copy so it's not read-only

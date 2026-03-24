"""
TIFF loader for NeuNorm.

Loads TIFF stacks as scipp DataArrays.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger
from PIL import ExifTags, Image


def load_tiff_stack(paths: Sequence[str | Path], tof_edges: Optional[np.ndarray] = None) -> sc.DataArray:  # noqa: C901
    """Load TIFF stack as scipp DataArray with variance tracking.

    Uses Pillow (PIL) to read TIFF images and constructs a scipp DataArray.

    Parameters
    ----------
    paths : Sequence[str | Path]
        List of paths to TIFF files
    tof_edges : Optional[np.ndarray]
        Time-of-flight values for the first dimension.
        Accepts either bin edges (N+1) or bin centers (N), where N is the
        number of images in the loaded stack.

    Returns
    -------
    sc.DataArray
        DataArray with dimensions (TOF/image, y, x)
        - dims: ['TOF', 'y', 'x'] if tof_edges provided, else ['N_image', 'y', 'x']
        - coords: y, x pixel indices, and optionally TOF.
          Additionally, TIFF metadata is added as coordinates with dimension of the stack.
    """

    if not paths:
        raise ValueError("No file paths provided")

    data_list = []
    metadata_list = []

    try:
        for path in paths:
            with Image.open(path) as img:
                data_list.append(np.asanyarray(img, dtype=np.float64))
                metadata_list.append(img.tag_v2)
    except Exception as e:
        logger.error("Error loading TIFF stack: {}", e)
        raise

    # Check shapes consistency
    first_shape = data_list[0].shape
    # Verify other shapes match
    for i, arr in enumerate(data_list[1:]):
        if arr.shape != first_shape:
            raise ValueError(f"Shape mismatch in file {paths[i + 1]}: expected {first_shape}, got {arr.shape}")

    # Stack
    full_data = np.stack(data_list, axis=0)

    n_images, ny, nx = full_data.shape

    # Determine dimension names
    # If tof_edges provided, use 'TOF', else uses 'N_image'
    dim_name = "TOF" if tof_edges is not None else "N_image"
    dims = [dim_name, "y", "x"]

    # Validate data for Poisson statistics: counts must be non-negative.
    if np.any(full_data < 0):
        raise ValueError(
            "Loaded TIFF data contains negative counts; cannot attach Poisson "
            "variances (variance = counts) to negative data."
        )

    # Create DataArray
    # Assuming variance = counts (Poisson) if not provided.
    da = sc.DataArray(
        data=sc.array(dims=dims, values=full_data, unit=sc.units.counts, variances=full_data.copy()),
        coords={"y": sc.arange("y", ny, unit=None), "x": sc.arange("x", nx, unit=None)},
    )

    # Add TOF coordinate if provided
    if tof_edges is not None:
        tof_values = np.asarray(tof_edges)
        if tof_values.ndim != 1:
            raise ValueError(f"tof_edges must be a 1D array, got shape {tof_values.shape}")

        if tof_values.size in (n_images, n_images + 1):
            da.coords[dim_name] = sc.array(dims=[dim_name], values=tof_values, unit=sc.units.us)
        else:
            raise ValueError(
                "Length of tof_edges must be number of images (bin centers) "
                f"or number of images + 1 (bin edges), got {tof_values.size} "
                f"with {n_images} images"
            )

    if metadata_list:
        # Process metadata and add as coordinates
        # Assuming all images have the same metadata keys.
        for key in metadata_list[0]:
            if (key_name := ExifTags.TAGS.get(key)) is not None:
                values = [metadata_list[i][key] for i in range(n_images)]
            else:
                # Check if value is a key value pair separated by a column, e.g. "ExposureTime:0.01"
                try:
                    key_name = str(metadata_list[0][key]).split(":")[0]
                    values = [str(metadata_list[i][key]).split(":")[1] for i in range(n_images)]
                except IndexError:
                    key_name = str(key)
                    values = [str(metadata_list[i][key]) for i in range(n_images)]

            # Try converting to float if possible, otherwise keep as string
            try:
                values = [float(v) for v in values]
                da.coords[key_name] = sc.array(dims=[dim_name], values=values, unit=sc.units.dimensionless)
            except (ValueError, TypeError):
                if len(set(v for v in values)) == 1:
                    # If all values are the same string, store as scalar
                    da.coords[key_name] = sc.scalar(value=values[0], unit=sc.units.dimensionless)
                else:
                    # Values differ across files, store as array with dimension of the stack
                    da.coords[key_name] = sc.array(dims=[dim_name], values=values, unit=sc.units.dimensionless)
            da.coords.set_aligned(key_name, False)

    return da

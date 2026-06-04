"""
FITS loader for NeuNorm based on astropy.

Loads FITS files into scipp DataArrays.
"""

import io
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from astropy.io import fits
from loguru import logger


def load_fits_stack(paths: Sequence[str | Path], tof_edges: Optional[np.ndarray] = None) -> sc.DataArray:  # noqa: C901
    """
    Load FITS stack as scipp DataArray with metadata and optional TOF coordinates.

    Handles:
    - List of FITS files (stacked along the first dimension)
    - Metadata extraction from FITS headers

    Parameters
    ----------
    paths : Sequence[str | Path]
        List of paths to FITS files
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
          Additionally, FITS header keys are added as coordinates with dimension of the stack.
    """

    if not paths:
        raise ValueError("No file paths provided")

    # Load data and metadata
    data_list = []
    headers = []

    try:
        # Load all files
        for path in paths:
            with fits.open(path) as hdul:
                info_buf = io.StringIO()
                hdul.info(output=info_buf)
                logger.debug("FITS info for {}:\n{}", path, info_buf.getvalue().rstrip())

                # Assume data is in primary HDU
                arr = hdul[0].data.astype(np.float64)
                data_list.append(arr)

                # Store header from first file
                headers.append(hdul[0].header)
    except Exception as e:
        logger.error(f"Failed to load FITS files: {e}")
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
            "Loaded FITS data contains negative counts; cannot attach Poisson "
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

    # Process header
    if headers:
        # Assume all headers have the same keys.
        # Storing all as coords with dimension of the stack (e.g. 'N_image' or 'TOF')
        for key in headers[0].keys():
            if key not in ("COMMENT", "HISTORY"):  # Skip multi-line text fields
                values = [hdr.get(key) for hdr in headers]
                if len(set(str(v) for v in values)) == 1:
                    # If all values are the same, store as scalar
                    da.coords[key] = sc.scalar(value=values[0])
                else:
                    # Values differ across files, store as array with dimension of the stack
                    da.coords[key] = sc.array(dims=[dim_name], values=values)
                da.coords.set_aligned(key, False)

    return da

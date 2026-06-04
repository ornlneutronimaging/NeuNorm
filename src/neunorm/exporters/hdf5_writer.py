"""
HDF5 writer for Neunorm outputs, including provenance metadata.
"""

import os
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import scipp as sc
from loguru import logger


def write_hdf5(  # noqa: C901
    output_path: Union[Path, str],
    transmission: sc.DataArray,
    dead_pixel_mask: str = "dead",
    hot_pixel_mask: str = "hot",
    metadata: Optional[dict] = None,
) -> None:
    """Write processed data to HDF5 with full provenance.

    Write transmission array (3D or 4D depending on workflow)
    Write uncertainty array (same shape as transmission)
    Write pixel masks (dead, hot)
    Write TOF bin edges (for hyperspectral data)
    Store full processing metadata/provenance

    Output Structure

    /transmission     # (θ, [TOF,] y, x) float32
    /uncertainty      # (θ, [TOF,] y, x) float32
    /masks/dead       # (y, x) bool
    /masks/hot        # (y, x) bool (optional)
    /tof_bin_edges    # (N+1,) float64 (optional)
    /metadata/        # processing provenance

    Metadata contents:
    - Input file paths
    - Processing timestamp
    - Gamma filter parameters used
    - ROI applied (if any)
    - Number of runs combined (if any)
    - Software version

    Parameters
    ----------
    output_path : Union[Path, str]
        Path to output HDF5 file
    transmission : sc.DataArray
        Processed transmission data to write. Variances, coordinates, and masks should be included in this DataArray.
    dead_pixel_mask : str
        Name of the dead pixel mask in transmission.masks (default: "dead")
    hot_pixel_mask : str
        Name of the hot pixel mask in transmission.masks (default: "hot")
    metadata : Optional[dict]
        Dictionary of metadata to store in the file (default: None)
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if path is writeable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_path.parent}")

    with h5py.File(output_path, "w") as f:
        # Write transmission data and unit
        f.create_dataset("transmission", data=transmission.values.astype("float32"))
        f["transmission"].attrs["units"] = str(transmission.unit)

        # Write uncertainty if available
        if transmission.variances is not None:
            f.create_dataset("uncertainty", data=np.sqrt(transmission.variances).astype("float32"))

        # Write coordinates
        for coord in transmission.coords:
            try:
                f.create_dataset(f"/{coord}", data=transmission.coords[coord].values)
                if transmission.coords[coord].unit is not None:
                    f[f"/{coord}"].attrs["units"] = str(transmission.coords[coord].unit)
            except TypeError as e:
                logger.warning(f"Could not write coordinate '{coord}' to HDF5: {e}")

        # Write masks
        if dead_pixel_mask in transmission.masks:
            f.create_dataset("masks/dead", data=transmission.masks[dead_pixel_mask].values)
        if hot_pixel_mask in transmission.masks:
            f.create_dataset("masks/hot", data=transmission.masks[hot_pixel_mask].values)

        # Write metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    f.create_dataset(f"metadata/{key}", data=value, dtype=h5py.string_dtype(encoding="utf-8"))
                else:
                    f.create_dataset(f"metadata/{key}", data=value)

    logger.info("HDF5 file written to {}", output_path)

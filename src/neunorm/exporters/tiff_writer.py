"""
TIFF writing utilities using scitiff to preserve scipp metadata and coordinates.
"""

import collections.abc
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc
from scitiff import DAQMetadata
from scitiff.io import save_scitiff


def _json_default(value):
    """JSON fallback for metadata leaves: keep NumPy scalars numeric; stringify the rest (e.g. Path)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def convert_metadata_to_scitiff_coords(metadata: dict) -> sc.DataGroup:
    """Convert metadata dictionary to scitiff-compatible DataGroup.

    This function takes a metadata dictionary and converts it into a scitiff DataGroup format,
    which can be embedded in TIFF tags when saving with scitiff. It handles various data types
    and ensures that the resulting DataGroup is compatible with scitiff's requirements.

    Sequence values (e.g. lists of source paths, an ROI tuple) are stored as JSON strings
    rather than object-dtype scipp scalars: scitiff's metadata schema only accepts scalar and
    typed 1-D/2-D variables, so an object-dtype (``PyObject``) scalar fails serialization.
    JSON encoding mirrors the HDF5 writer's provenance convention; read a value back with
    ``json.loads(extra[key])``.

    Parameters
    ----------
    metadata : dict
        Dictionary containing metadata key-value pairs to convert.

    Returns
    -------
    sc.DataGroup
        A DataGroup containing the converted metadata, ready for embedding in scitiff.
    """
    extra = sc.DataGroup()
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            extra[key] = value
        elif isinstance(value, collections.abc.Sequence):
            # JSON-encode sequences as strings; preserve nested structure (e.g. per-run path
            # groups ``[[...], [...]]``) so TIFF provenance matches the HDF5 writer exactly.
            extra[key] = json.dumps(value, default=_json_default)
        else:
            raise ValueError(f"Unsupported metadata type for key '{key}': {type(value)}")
    return extra


def write_tiff_stack(
    output_path: Union[Path, str],
    transmission: sc.DataArray,
    metadata: Optional[dict] = None,
    daqmetadata: Optional[dict] = None,
) -> None:
    """Write transmission stack as TIFF files using scitiff.

    Uses scitiff to preserve scipp DataArray metadata and coordinates.

    Requirements

    Write transmission as TIFF stack (one file per image or multi-page TIFF)
    Write uncertainty (stdevs) and a mask packed into the same stack via the
    channel dimension (``concat_stdevs_and_mask=True``), not as a separate file
    Support 32-bit float output
    Embed metadata in scitiff format (JSON in TIFF tags)
    Preserve scipp coordinate information through scitiff

    Scitiff Metadata Schema https://scipp.github.io/scitiff/index.html#scitiff-metadata-schema

    Parameters
    ----------
    output_path : Union[Path, str]
        Path to save TIFF files
    transmission : sc.DataArray
        DataArray containing transmission data, variances, coordinates
    metadata : Optional[dict]
        Additional metadata to embed in the TIFF files (default: None). Can include any key-value pairs.
    daqmetadata : Optional[dict]
        Additional DAQ metadata to embed in the TIFF files (default: None).
        Created as scitiff.DAQMetadata with keys:
        'facility', 'instrument', 'detector_type', 'source_type', 'source', 'simulated'
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if path is writeable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_path.parent}")

    # build scitiff DataGroup with image and metadata
    image = transmission.astype("float32")
    # scitiff serializes the image's coords/masks but its schema only accepts scalar and
    # typed 1-D/2-D variables. Drop object-dtype (PyObject) coords/masks — e.g. tuple-valued
    # TIFF header tags (BitsPerSample, StripOffsets, ...) carried over from the input files by
    # the loader — which scitiff cannot serialize. They are input-file header provenance, not
    # analysis data; the HDF5 writer likewise does not persist object-dtype coords.
    for _name in [n for n, c in image.coords.items() if c.dtype == sc.DType.PyObject]:
        del image.coords[_name]
    for _name in [n for n, m in image.masks.items() if m.dtype == sc.DType.PyObject]:
        del image.masks[_name]
    dg = sc.DataGroup(image=image)

    # Add DAQ metadata if provided
    if daqmetadata:
        dg["daq"] = DAQMetadata(**daqmetadata)

    # Add extra metadata if provided
    if metadata:
        dg["extra"] = convert_metadata_to_scitiff_coords(metadata)

    # Write transmission, stdevs, and masks to TIFF using scitiff
    save_scitiff(dg, output_path, concat_stdevs_and_mask=True)

"""
TIFF writing utilities using scitiff to preserve scipp metadata and coordinates.
"""

import collections.abc
import os
from pathlib import Path
from typing import Optional, Union

import scipp as sc
from scitiff import DAQMetadata
from scitiff.io import save_scitiff


def convert_metadata_to_scitiff_coords(metadata: dict) -> sc.DataGroup:
    """Convert metadata dictionary to scitiff-compatible DataGroup.

    This function takes a metadata dictionary and converts it into a scitiff DataGroup format,
    which can be embedded in TIFF tags when saving with scitiff. It handles various data types
    and ensures that the resulting DataGroup is compatible with scitiff's requirements.

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
            if isinstance(value[0], collections.abc.Sequence) and not isinstance(value[0], str):
                # Handle list of lists (e.g. list of sample paths)
                extra[key] = sc.scalar(value=[item for sublist in value for item in sublist])
            else:
                extra[key] = sc.scalar(value=value)
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
    Write uncertainty as separate TIFF stack
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
    dg = sc.DataGroup(image=transmission.astype("float32"))

    # Add DAQ metadata if provided
    if daqmetadata:
        dg["daq"] = DAQMetadata(**daqmetadata)

    # Add extra metadata if provided
    if metadata:
        dg["extra"] = convert_metadata_to_scitiff_coords(metadata)

    # Write transmission, stdevs, and masks to TIFF using scitiff
    save_scitiff(dg, output_path, concat_stdevs_and_mask=True)

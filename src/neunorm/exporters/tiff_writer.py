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


#: Dimensions treated as spatial (a single image plane). Any OTHER dimension is the spectral
#: (per-image) axis used by ``one_file_per_image`` — e.g. ``tof``, or ``t`` after the pipelines
#: rename it for scitiff. More than one non-spatial dimension is ambiguous and is rejected.
_SPATIAL_DIMS = ("x", "y")


def _to_scitiff_image(transmission: sc.DataArray) -> sc.DataArray:
    """Cast to float32 and drop coords/masks scitiff cannot serialize.

    scitiff serializes the image's coords/masks but its schema only accepts scalar and typed
    1-D/2-D variables. Object-dtype (``PyObject``) coords/masks — e.g. tuple-valued TIFF header
    tags (BitsPerSample, StripOffsets, ...) carried over from the input files by the loader — are
    dropped. They are input-file header provenance, not analysis data; the HDF5 writer likewise
    does not persist object-dtype coords.
    """
    image = transmission.astype("float32")
    for _name in [n for n, c in image.coords.items() if c.dtype == sc.DType.PyObject]:
        del image.coords[_name]
    for _name in [n for n, m in image.masks.items() if m.dtype == sc.DType.PyObject]:
        del image.masks[_name]
    return image


def _build_scitiff_datagroup(
    image: sc.DataArray, metadata: Optional[dict], daqmetadata: Optional[dict]
) -> sc.DataGroup:
    """Assemble the scitiff DataGroup (image + optional DAQ/extra metadata)."""
    dg = sc.DataGroup(image=image)
    if daqmetadata:
        dg["daq"] = DAQMetadata(**daqmetadata)
    if metadata:
        dg["extra"] = convert_metadata_to_scitiff_coords(metadata)
    return dg


def write_tiff_stack(
    output_path: Union[Path, str],
    transmission: sc.DataArray,
    metadata: Optional[dict] = None,
    daqmetadata: Optional[dict] = None,
    *,
    one_file_per_image: bool = False,
    concat_stdevs_and_mask: Optional[bool] = None,
) -> list[Path]:
    """Write transmission as TIFF using scitiff.

    Uses scitiff to preserve scipp DataArray metadata and coordinates.

    Requirements

    Write transmission as a single multi-page TIFF stack (default) or, with
    ``one_file_per_image=True``, as one scitiff file per spectral image
    Write uncertainty (stdevs) and a mask packed into the same stack via the
    channel dimension (``concat_stdevs_and_mask=True``), not as a separate file
    Support 32-bit float output
    Embed metadata in scitiff format (JSON in TIFF tags)
    Preserve scipp coordinate information through scitiff

    Scitiff Metadata Schema https://scipp.github.io/scitiff/index.html#scitiff-metadata-schema

    Parameters
    ----------
    output_path : Union[Path, str]
        Path to save TIFF files. With ``one_file_per_image=True`` this is the naming template:
        each image is written next to it as ``<stem>_<index><suffix>`` (see below).
    transmission : sc.DataArray
        DataArray containing transmission data, variances, coordinates
    metadata : Optional[dict]
        Additional metadata to embed in the TIFF files (default: None). Can include any key-value pairs.
    daqmetadata : Optional[dict]
        Additional DAQ metadata to embed in the TIFF files (default: None).
        Created as scitiff.DAQMetadata with keys:
        'facility', 'instrument', 'detector_type', 'source_type', 'source', 'simulated'
    one_file_per_image : bool
        When ``False`` (default) write one multi-page TIFF containing the whole stack — unchanged
        behavior. When ``True`` write **one scitiff file per spectral image** (one normalization per
        file), named ``<stem>_00000<suffix>``, ``<stem>_00001<suffix>``, … in spectral order (the
        index is zero-padded to at least 5 digits, widening if there are more than 100000 images, so
        a lexicographic listing always matches spectral order). Each file holds a single
        ``(y, x)`` normalization — a **single-page** TIFF by default (see
        ``concat_stdevs_and_mask``) — and keeps its own slice's coordinates, including that bin's
        ``tof`` bounds and ``spectra_tof`` time, plus the same metadata as the stack. Data with no
        spectral dimension (a plain ``(y, x)`` radiograph) is already one image and is written as a
        single file.

        Raises ``ValueError`` if the data has more than one non-spatial dimension (ambiguous split)
        or if the spectral dimension is empty.
    concat_stdevs_and_mask : bool, optional
        Whether to pack standard deviations and the mask alongside the intensities as extra scitiff
        channels. ``None`` (default) picks the right behavior per mode:

        - **stack mode** → ``True``, unchanged: one multi-page file whose three channels are
          intensities, standard deviations and the mask.
        - **per-image mode** → ``False``: each file is a **single-page image containing only the
          normalization**, which is the point of the mode — a 3-channel file would make a viewer show
          three times as many images, the extra two being an uncertainty plane and an all-zero mask.

        Pass ``True`` explicitly to keep the stdev/mask channels in per-image mode. Note the cost of
        the default: scitiff writes variances only as that channel, so a single-page per-image file
        carries **no uncertainty** — read it from the HDF5 output (the primary format) instead. Masks
        and metadata travel either way.

    Returns
    -------
    list[Path]
        The files written, in order.
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if path is writeable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_path.parent}")

    image = _to_scitiff_image(transmission)

    spectral_dims = [d for d in image.dims if d not in _SPATIAL_DIMS]
    if one_file_per_image and spectral_dims:
        if len(spectral_dims) > 1:
            raise ValueError(
                f"one_file_per_image needs exactly one non-spatial dimension to split on; got "
                f"{spectral_dims} in dims {image.dims}. Reduce or select the extra dimension(s) first."
            )
        spectral_dim = spectral_dims[0]
        n_images = image.sizes[spectral_dim]
        if n_images == 0:
            raise ValueError(f"cannot write one file per image: dimension '{spectral_dim}' is empty (size 0)")
        # One normalization per file. Slice with an integer index so each file holds a plain
        # (y, x) image: scipp reduces that dim's bin-edge coord to the slice's two bounds and a
        # point coord (spectra_tof) to a scalar, so every file records which bin it came from.
        # Pad the index to the width of the largest index (at least 5) so a lexicographic listing
        # always matches spectral order, even past 99999 images.
        width = max(5, len(str(n_images - 1)))
        # One normalization per file means ONE page per file: packing stdevs + mask as extra channels
        # would make a viewer report three times as many images, the extra two being an uncertainty
        # plane and an all-zero mask. Uncertainty lives in the HDF5 output; opt back in explicitly.
        concat = False if concat_stdevs_and_mask is None else concat_stdevs_and_mask
        written: list[Path] = []
        for index in range(n_images):
            frame_path = output_path.with_name(f"{output_path.stem}_{index:0{width}d}{output_path.suffix}")
            dg = _build_scitiff_datagroup(image[spectral_dim, index], metadata, daqmetadata)
            save_scitiff(dg, frame_path, concat_stdevs_and_mask=concat)
            written.append(frame_path)
        return written

    # Stack mode: write transmission, stdevs, and masks into one multi-page file (unchanged default).
    concat = True if concat_stdevs_and_mask is None else concat_stdevs_and_mask
    save_scitiff(_build_scitiff_datagroup(image, metadata, daqmetadata), output_path, concat_stdevs_and_mask=concat)
    return [output_path]

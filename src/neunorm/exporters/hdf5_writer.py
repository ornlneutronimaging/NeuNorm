"""
HDF5 writer for Neunorm outputs, including provenance metadata.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import scipp as sc
from loguru import logger

from neunorm.utils.progress import STAGE_EXPORT, ProgressLike, resolve_progress


def _is_nested_sequence(value) -> bool:
    """Return True if ``value`` is a list/tuple that contains a list/tuple element.

    Nested sequences (e.g. per-run file-path provenance ``[[...], [...]]``) cannot be
    stored natively by h5py when ragged, so they are JSON-serialized instead.
    """
    return isinstance(value, (list, tuple)) and any(isinstance(item, (list, tuple)) for item in value)


def _write_json_metadata(f: h5py.File, name: str, value) -> None:
    """Store ``value`` as a JSON string dataset tagged with ``encoding="json"``.

    JSON-native leaves (strings, numbers, bools, ``null``) keep their JSON types; only
    non-JSON-serializable leaves are coerced via ``str()`` (``json.dumps(..., default=str)``).
    The round-trip is therefore lossless for the string-valued per-run file-path provenance the
    pipelines emit. May raise (e.g. on a circular reference, or an invalid dataset name); the
    caller backstops any failure so a single bad key never aborts the write.
    """
    ds = f.create_dataset(name, data=json.dumps(value, default=str), dtype=h5py.string_dtype("utf-8"))
    ds.attrs["encoding"] = "json"


def _write_metadata_value(f: h5py.File, name: str, value) -> None:
    """Write one metadata value, choosing the best HDF5 representation.

    Strings, scalars, and flat arrays are stored natively; nested list/tuple provenance is
    stored as a round-trippable JSON string. May raise; the caller backstops failures so a
    single un-writable key never aborts the write or corrupts the file.
    """
    if isinstance(value, str):
        f.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    elif _is_nested_sequence(value):
        _write_json_metadata(f, name, value)
    else:
        try:
            f.create_dataset(name, data=value)
        except (TypeError, ValueError):
            # h5py cannot store this value natively; remove the partial dataset it registered
            # before failing (only a dataset, never a pre-existing group), then JSON-encode it.
            if name in f and isinstance(f[name], h5py.Dataset):
                del f[name]
            _write_json_metadata(f, name, value)


def _discard_failed_metadata(f: h5py.File, name: Optional[str], key, exc: Exception, created_here: bool) -> None:
    """Best-effort cleanup + warning for a metadata key that could not be written.

    Never raises: provenance is best-effort and must not abort the bulk-data write. Removes a
    leftover dataset only when this key's write actually created it (``created_here``) — never a
    pre-existing dataset or group — so a colliding or duplicate name cannot drop earlier
    metadata. Logs a pre-formatted string (no live key/exception objects) so a pathological key
    or exception cannot make logging itself raise.
    """
    try:
        if created_here and name is not None and name in f and isinstance(f[name], h5py.Dataset):
            del f[name]
    except Exception:  # noqa: BLE001 - cleanup is best-effort; an h5py name error here must not propagate
        pass
    try:
        detail = f"key {key!r}: {exc}"
    except Exception:  # noqa: BLE001 - a pathological key/exception repr must not break logging
        detail = "an un-formattable key or error"
    logger.warning(
        "Skipping unwritable metadata in HDF5 output ({}). Array data and other metadata are still "
        "written. See https://github.com/ornlneutronimaging/NeuNorm/issues/140",
        detail,
    )


def write_hdf5(  # noqa: C901
    output_path: Union[Path, str],
    transmission: sc.DataArray,
    dead_pixel_mask: str = "dead",
    hot_pixel_mask: str = "hot",
    metadata: Optional[dict] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_EXPORT,
) -> None:
    """Write processed data to HDF5 with full provenance.

    Write transmission array (3D or 4D depending on workflow)
    Write uncertainty array (same shape as transmission)
    Write pixel masks (dead, hot)
    Write TOF bin edges (for hyperspectral data)
    Store full processing metadata/provenance

    Output Structure::

        /transmission     # (θ, [TOF,] y, x) float32
        /uncertainty      # (θ, [TOF,] y, x) float32 (only if variances are present)
        /masks/dead       # (y, x) bool (only if the named dead-pixel mask exists)
        /masks/hot        # (y, x) bool (only if the named hot-pixel mask exists)
        /masks/<name>     # any other mask under its own name (e.g. a 1-D TOF mask)
        /tof              # (N+1,) float64 (only if the data carries a "tof" coordinate)
        /spectra_tof      # (N,) float64 per-bin mean time (only for bin-list / mean / median rebins)
        /metadata/        # processing provenance (only when metadata is supplied)

    Metadata values are stored as native datasets (strings, scalars, flat arrays).
    Nested values such as per-run file-path lists (``sample_paths=[[...], [...]]``) are
    stored as a JSON string with a ``encoding="json"`` dataset attribute; read them back
    with ``json.loads(dataset.asstr()[()])``. The round-trip is lossless for string-valued
    provenance (what the pipelines emit); only non-JSON-serializable leaves are coerced via
    ``str()`` (``json.dumps(..., default=str)``), so JSON-native numbers/bools/null keep their types.

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
    progress : bool or callable, optional
        Progress reporting, off by default. HDF5 is the primary output format and this function has no
        item axis — the bulk data goes out in one or two whole-array writes — so it reports named
        steps, never per image: the transmission dataset, the uncertainty dataset when the data carries
        variances, the coordinate/mask section, and the metadata section when metadata is supplied.
        See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_EXPORT``.
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if path is writeable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_path.parent}")

    # HDF5 is the primary output format and this function has no item axis: the bulk data goes out in
    # one or two whole-array `create_dataset` calls, so it can never be reported per image. What is
    # worth naming is which of those writes is running — they are the only expensive parts — plus the
    # small coordinate/mask and metadata sections. The total is computed because the uncertainty write
    # happens only for variance-bearing data and the metadata section only when metadata is supplied.
    n_steps = 2  # the transmission dataset, then coordinates and masks
    if transmission.variances is not None:
        n_steps += 1
    if metadata:
        n_steps += 1

    with resolve_progress(progress, stage, total=n_steps) as report, h5py.File(output_path, "w") as f:
        # Write transmission data and unit
        report.note("writing transmission")
        f.create_dataset("transmission", data=transmission.values.astype("float32"))
        f["transmission"].attrs["units"] = str(transmission.unit)
        report()

        # Write uncertainty if available
        if transmission.variances is not None:
            report.note("writing uncertainty")
            f.create_dataset("uncertainty", data=np.sqrt(transmission.variances).astype("float32"))
            report()

        # Write coordinates
        report.note("writing coordinates and masks")
        for coord in transmission.coords:
            try:
                f.create_dataset(f"/{coord}", data=transmission.coords[coord].values)
                if transmission.coords[coord].unit is not None:
                    f[f"/{coord}"].attrs["units"] = str(transmission.coords[coord].unit)
            except TypeError as e:
                logger.warning(f"Could not write coordinate '{coord}' to HDF5: {e}")

        # Write masks. The dead/hot masks keep their canonical /masks/dead and /masks/hot names;
        # any other mask (e.g. a 1-D per-bin TOF mask) is written under its own name so that
        # mask provenance is not lost.
        if dead_pixel_mask in transmission.masks:
            f.create_dataset("masks/dead", data=transmission.masks[dead_pixel_mask].values)
        if hot_pixel_mask in transmission.masks:
            f.create_dataset("masks/hot", data=transmission.masks[hot_pixel_mask].values)
        for mask_name in transmission.masks:
            if mask_name in (dead_pixel_mask, hot_pixel_mask):
                continue
            if "/" in mask_name:  # a '/' would create nested HDF5 groups and clash unpredictably
                raise ValueError(f"mask name '{mask_name}' must not contain '/'")
            path = f"masks/{mask_name}"
            if path in f:
                # A mask genuinely named 'dead'/'hot' distinct from the designated dead/hot mask
                # collides with the canonical path. Fail loudly rather than silently drop its data.
                raise ValueError(
                    f"mask '{mask_name}' collides with the already-written HDF5 path '{path}'; "
                    "rename it (it must not reuse the canonical 'dead'/'hot' names unless it is the designated mask)"
                )
            f.create_dataset(path, data=transmission.masks[mask_name].values)
        report()

        # Write metadata. Provenance is best-effort: a single un-writable key — a bad value
        # (ragged/unserializable) or a malformed/colliding name — must never abort the write or
        # leave a corrupt, partially-written file. Nested list/tuple values are
        # serialized as round-trippable JSON (read back with json.loads(dataset.asstr()[()])).
        #
        # The progress emits stay OUTSIDE the per-key try below. That handler deliberately swallows
        # everything so one bad key cannot abort the bulk-data write — which would also swallow a
        # cancelling callback's exception, turning a user's abort into a silently skipped metadata key.
        if metadata:
            report.note("writing metadata")
            for key, value in metadata.items():
                name = None
                created_here = False
                try:
                    name = f"metadata/{key}"
                    if name in f:
                        # Two metadata keys collide under str() (e.g. int 1 and str "1" both ->
                        # "metadata/1"). Keep the first and skip the later one — never overwrite or
                        # delete the earlier value. `created_here` stays False so cleanup won't touch it.
                        raise ValueError(f"duplicate metadata path {name!r}")
                    created_here = True
                    _write_metadata_value(f, name, value)
                except Exception as exc:  # noqa: BLE001 - metadata is best-effort; never abort the bulk-data write
                    _discard_failed_metadata(f, name, key, exc, created_here)
            report()

    logger.info("HDF5 file written to {}", output_path)

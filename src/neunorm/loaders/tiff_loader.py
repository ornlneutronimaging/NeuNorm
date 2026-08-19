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

from neunorm.utils.progress import STAGE_LOAD_SAMPLE, ProgressLike, resolve_progress


def load_tiff_stack(  # noqa: C901
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
) -> sc.DataArray:
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
    progress : bool or callable, optional
        Progress reporting, off by default. ``True`` draws a :mod:`tqdm` bar; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` per file read, then one for each of the two
        whole-stack allocations that follow the read loop. A pipeline normally passes a pre-bound
        reporter here instead, so its per-file count spans every run rather than restarting.
        See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_LOAD_SAMPLE``; pass ``STAGE_LOAD_OB`` or
        ``STAGE_LOAD_DARK`` when loading those, so a callback can tell the loads of a run apart.

    Returns
    -------
    sc.DataArray
        DataArray with dimensions (TOF/image, y, x)

        - dims: ['TOF', 'y', 'x'] if tof_edges provided, else ['N_image', 'y', 'x']
        - coords: y, x pixel indices, and optionally TOF.
          Additionally, TIFF metadata is added as coordinates. Each metadata
          coordinate may be scalar (when its value is constant across the stack
          and not float-convertible) or stack-dimensioned (when values are
          float-convertible or differ across files).
    """

    data_list = []
    metadata_list = []

    # A non-sized iterable (Path.glob(), a generator) was accepted before this function reported
    # progress and must still be: materialise once so the count has a denominator. Wrapped so an
    # iterator that raises is logged like any other read failure. This runs BEFORE the emptiness
    # check because a generator is always truthy — an empty glob would otherwise skip the check and
    # die later with IndexError on `data_list[0]`.
    if not hasattr(paths, "__len__"):
        try:
            paths = list(paths)
        except Exception as e:
            logger.error("Error loading TIFF stack: {}", e)
            raise

    if not paths:
        raise ValueError("No file paths provided")

    with resolve_progress(progress, stage, total=len(paths)) as report:
        for path in paths:
            # The try covers only the read: an exception raised by a progress callback (which is how a
            # caller cancels) must not be logged as a failed TIFF read, so the tick is emitted outside.
            try:
                with Image.open(path) as img:
                    # float32 is sufficient for neutron imaging (16-bit detectors) and
                    # halves the in-memory footprint of large stacks.
                    data_list.append(np.asanyarray(img, dtype=np.float32))
                    metadata_list.append(img.tag_v2)
            except Exception as e:
                logger.error("Error loading TIFF stack: {}", e)
                raise
            report(detail=Path(path).name)

        # Check shapes consistency
        first_shape = data_list[0].shape
        # Verify other shapes match
        for i, arr in enumerate(data_list[1:]):
            if arr.shape != first_shape:
                raise ValueError(f"Shape mismatch in file {paths[i + 1]}: expected {first_shape}, got {arr.shape}")

        # The read loop only appended to a list; the memory peak is here and in the variances copy
        # below, which together hold several full-size copies of the stack. Reporting them keeps a bar
        # moving through the part of the load that can actually exhaust RAM and start swapping —
        # otherwise it reaches 100% at the last file and then sits silent through the worst of it.
        report.note(f"stacking {len(data_list)} frames")
        full_data = np.stack(data_list, axis=0)

        n_images, ny, nx = full_data.shape

        # Determine dimension names
        # If tof_edges provided, use 'TOF', else uses 'N_image'
        dim_name = "TOF" if tof_edges is not None else "N_image"
        dims = [dim_name, "y", "x"]

        # Poisson statistics (variance = counts) needs non-negative data. Real
        # acquisitions occasionally contain a few negative pixels — e.g. a
        # glitching Timepix chip writing wrapped values around ±32k into the
        # autoreduced frames. Those pixels carry no physical information, so
        # they are zeroed (count 0 → variance 0) instead of aborting the load.
negative = full_data < 0
if negative.any():
    frame_has_negative = negative.reshape(n_images, -1).any(axis=1)
    n_frames = int(frame_has_negative.sum())
                "Loaded TIFF data contains {} negative pixel(s) across {} of {} frame(s) "
                "(most negative value: {:.1f}); zeroing them to keep Poisson variances valid.",
                int(negative.sum()),
                n_frames,
                n_images,
                float(full_data.min()),
            )
            full_data[negative] = 0.0

        report.note(f"attaching variances ({full_data.nbytes / 1024**2:.1f} MiB)")

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
                    da.coords[key_name] = sc.array(dims=[dim_name], values=values)
                except (ValueError, TypeError):
                    if len(set(v for v in values)) == 1:
                        # If all values are the same string, store as scalar
                        da.coords[key_name] = sc.scalar(value=values[0])
                    else:
                        # Values differ across files, store as array with dimension of the stack
                        da.coords[key_name] = sc.array(dims=[dim_name], values=values)
                da.coords.set_aligned(key_name, False)

        return da

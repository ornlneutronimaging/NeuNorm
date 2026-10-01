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

from neunorm.data_models.roi import ROILike, _checked_crop_bounds
from neunorm.loaders._frame_stack import decode_frames, variances_note, with_poisson_variances
from neunorm.utils.progress import STAGE_LOAD_SAMPLE, ProgressLike, resolve_progress


def _read_fits_frame(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Decode one FITS frame, returning ``(values, header)``.

    Split out of the read loop so it can be called per frame from a worker thread: it shares no
    state and opens its own file handle.

    **astropy releases the GIL on read**, which is what makes a thread pool worth using here and
    was measured before this was written rather than assumed: 24 frames of 2048x2048 big-endian
    int16 took 0.089 s serially and 0.028 s across 8 threads, a 3.2x speedup. The work that
    overlaps is the read syscall plus the byte swap and cast, all of which drop the GIL.

    The cast must happen inside the ``with``: ``hdul[0].data`` is memory-mapped, so the values are
    only valid while the file is open. The header is fully materialised and outlives the close.

    float32 is sufficient for neutron imaging (16-bit detectors) and halves the in-memory footprint
    of a large stack.
    """
    with fits.open(path) as hdul:
        info_buf = io.StringIO()
        hdul.info(output=info_buf)
        logger.debug("FITS info for {}:\n{}", path, info_buf.getvalue().rstrip())

        # Assume data is in the primary HDU.
        values = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    return values, header


def load_fits_stack(
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
    roi: Optional[ROILike] = None,
) -> sc.DataArray:
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
    progress : bool or callable, optional
        Progress reporting, off by default. ``True`` draws a :mod:`tqdm` bar; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` per file read, plus a note naming the
        stack's frame count, frame size and memory before its variances are filled. A pipeline
        normally passes a
        pre-bound reporter here instead, so its per-file count spans every run rather than
        restarting. See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_LOAD_SAMPLE``; pass ``STAGE_LOAD_OB`` or
        ``STAGE_LOAD_DARK`` when loading those, so a callback can tell the loads of a run apart.
    max_workers : int, optional
        Threads used to decode the stack, 8 by default. Frames are decoded concurrently into a
        pre-allocated array, each written at its own input index, so the result is identical to a
        serial read whatever order the decodes finish in. ``max_workers=1`` reads serially.

        The default is deliberately modest rather than tuned: the useful number depends on whether
        the files sit on local disk or a mounted analysis filesystem, which has not been measured
        here, and a large pool from every concurrent user is worse for a shared mount than a small
        one. Raise it once there are numbers.
    roi : ROI or tuple[int, int, int, int], optional
        Keep only this rectangle of each frame: an :class:`~neunorm.data_models.roi.ROI` or a bare
        ``(x0, y0, x1, y1)`` tuple with exclusive stop indices. The result is identical to
        :func:`~neunorm.processing.roi_clipper.apply_roi` applied to the full load, including the
        ``x`` and ``y`` coordinates, which hold detector pixel indices starting at ``x0`` and
        ``y0``. Each frame is still decoded whole, and the shape check and the non-negative-counts
        check still cover the whole frame, but only the region is stored: the stack's memory scales
        with the region, plus up to ``max_workers`` whole frames being decoded at once.

        A malformed ROI, or a :class:`~neunorm.data_models.roi.MaskROI`, raises before any file is
        read; an ROI that extends past the frames raises once the first frame is decoded.

    Returns
    -------
    sc.DataArray
        DataArray with dimensions (TOF/image, y, x)

        - dims: ['TOF', 'y', 'x'] if tof_edges provided, else ['N_image', 'y', 'x']
        - coords: y, x detector pixel indices (offset by the ROI origin when ``roi`` is given),
          and optionally TOF.
          Additionally, FITS header keys are added as (unaligned) coordinates.
          The ``COMMENT`` and ``HISTORY`` keys are skipped. A key whose value is
          constant across the stack is stored as a scalar coordinate; a key
          whose value differs across files is stored as an array coordinate
          along the stack dimension.
    """
    return _load_fits_stack(paths, tof_edges, progress=progress, stage=stage, max_workers=max_workers, roi=roi)[0]


def _load_fits_stack(  # noqa: C901
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
    roi: Optional[ROILike] = None,
) -> tuple[sc.DataArray, tuple[int, ...]]:
    """:func:`load_fits_stack`, also returning the uncropped stack shape ``(n_frames, ny, nx)``."""
    if max_workers is not None:
        # Same shape as `_check_advance` in utils/progress.py: bool is rejected explicitly because
        # it is an int subclass and `True` would otherwise pass as 1, and numpy integers are
        # accepted because a caller deriving a worker count from an array shape produces one.
        #
        # The type check is not tidiness. ThreadPoolExecutor compares its live thread count against
        # this value rather than truncating it, so max_workers=1.5 creates two threads and 2.9
        # creates three -- silently exceeding the cap the caller asked for, which is the one thing
        # this parameter exists to set.
        if isinstance(max_workers, bool) or not isinstance(max_workers, (int, np.integer)):
            raise TypeError(f"max_workers must be an int, got {type(max_workers).__name__}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")

    # Validated before any file is read, so a malformed ROI fails at once.
    bounds = None if roi is None else _checked_crop_bounds(roi, caller="The roi argument")

    # A generator, Path.glob() or a set was accepted before this function reported progress and
    # must still be: materialise once so the count has a denominator, and because `decode_frames`
    # addresses frames by index — a set has `__len__` but no `__getitem__`, so testing only for
    # length would let it through to a TypeError. Wrapped so an iterator that raises is logged
    # like any other read failure. This runs BEFORE the emptiness check because a generator is
    # always truthy — an empty glob would otherwise skip the check and die indexing `paths[0]`.
    if not hasattr(paths, "__len__") or not hasattr(paths, "__getitem__"):
        try:
            paths = list(paths)
        except Exception as e:
            logger.error("Failed to load FITS files: {}", e)
            raise

    if not paths:
        raise ValueError("No file paths provided")

    # If tof_edges provided, use 'TOF', else uses 'N_image'
    dim_name = "TOF" if tof_edges is not None else "N_image"

    with resolve_progress(progress, stage, total=len(paths)) as report:
        # `decode_frames` owns the read logging, because only it can tell a failed decode from a
        # cancelling progress callback.
        stack = decode_frames(
            paths,
            report,
            max_workers,
            read_frame=_read_fits_frame,
            read_error="Failed to load FITS files",
            thread_name_prefix="neunorm-fits",
            dim=dim_name,
            bounds=bounds,
        )
        headers = stack.meta

        n_images, _, _ = stack.shape

        # Validate data for Poisson statistics: counts must be non-negative, over every whole frame.
        if stack.negative:
            raise ValueError(
                "Loaded FITS data contains negative counts; cannot attach Poisson "
                "variances (variance = counts) to negative data."
            )

        report.note(variances_note(stack.data))

        # Assuming variance = counts (Poisson) if not provided.
        da = with_poisson_variances(stack)

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

        return da, stack.shape

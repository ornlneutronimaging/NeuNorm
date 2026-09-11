"""
FITS loader for NeuNorm based on astropy.

Loads FITS files into scipp DataArrays.
"""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from astropy.io import fits
from loguru import logger

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


class _ShapeMismatchError(ValueError):
    """A frame whose shape differs from the first frame's.

    A ``ValueError`` so the exception a caller sees is unchanged, but its own type so the pool
    loop can tell it apart from a read failure that also happens to be a ``ValueError``.
    """


#: Threads used to decode a stack when the caller does not say. Deliberately modest: the
#: right number depends on whether the files are local or on a mounted analysis filesystem,
#: and that was not measured, so this trades some of the available speedup for not swamping
#: a shared mount from every concurrent user. Raise it via ``max_workers`` once measured.
_DEFAULT_MAX_WORKERS = 8


def _decode_stack(
    paths: Sequence[str | Path],
    report,
    max_workers: Optional[int],
) -> tuple[np.ndarray, list[fits.Header]]:
    """Decode every frame into one pre-allocated array, in input order.

    **Frame order is the spectral axis.** The stack's first dimension becomes ``TOF`` or
    ``N_image``, and the ``tof`` coordinate is matched to it positionally, so a frame landing at
    the wrong index mislabels the time axis and produces a plausible-looking wrong spectrum.
    Workers therefore write ``out[i]`` for their own input index: order is preserved by
    construction rather than by collecting results carefully. Nothing here depends on the order
    in which decodes finish.

    Pre-allocating also removes one of the roughly five full-size copies resident at peak. It
    built a list of ``n`` frames, stacked that into a second copy, then copied the result for the
    variances; decoding straight into ``out`` collapses the first two into one, leaving the output
    and the variances copy, with only the in-flight decode buffers on top. One copy, not two: the
    measured peak drops from 5.26x the stack to 4.46x, which is what removing one of roughly five
    resident copies looks like once scipp's own copies are counted.

    Progress is emitted **from this thread**, never from a worker, which is what keeps the
    contract in :mod:`neunorm.utils.progress`: events stay synchronous and on the calling
    thread, a caller's callback still need not be thread-safe, and raising from it still
    cancels the run. Two consequences of the pool that are not the progress contract and are
    worth knowing:

    - ``detail`` names files in completion order, so it no longer tracks input order. The count
      itself is unaffected.
    - the per-file ``logger.debug`` line in :func:`_read_fits_frame` *is* emitted from the worker,
      so debug-level log order no longer follows input order either. Only progress events are
      promised to be ordered.

    Cancelling is prompt but not instant: raising from the callback propagates out of this loop
    and the ``finally`` cancels every queued file, but it waits for the decodes already in flight.
    And when more than one frame is unreadable, which one is named in the error depends on which
    decode finishes first, where the serial version always reported the first in input order.
    """
    n = len(paths)

    # Frame 0 is decoded here, alone, because its shape is what the output array is allocated
    # from and what every other frame is checked against.
    #
    # Every `report(...)` below sits OUTSIDE these try blocks, and that placement is load-bearing:
    # raising from a progress callback is how a caller cancels, and a cancel must not be logged as
    # "Failed to load FITS files". tests/unit/test_progress_load_path.py pins it.
    try:
        first_values, first_header = _read_fits_frame(paths[0])
    except Exception as e:
        logger.error("Failed to load FITS files: {}", e)
        raise
    out = np.empty((n, *first_values.shape), dtype=np.float32)
    out[0] = first_values
    headers: list[Optional[fits.Header]] = [None] * n
    headers[0] = first_header
    report(detail=Path(paths[0]).name)

    if n == 1:
        return out, headers

    def decode(index: int):
        values, header = _read_fits_frame(paths[index])
        if values.shape != first_values.shape:
            raise _ShapeMismatchError(
                f"Shape mismatch in file {paths[index]}: expected {first_values.shape}, got {values.shape}"
            )
        out[index] = values
        return index, header

    workers = max(1, min(_DEFAULT_MAX_WORKERS if max_workers is None else max_workers, n - 1))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="neunorm-fits")
    try:
        futures = [pool.submit(decode, i) for i in range(1, n)]
        for future in as_completed(futures):
            # `.result()` re-raises a worker's exception here, in the calling thread, so a bad
            # file still surfaces as it did when the read was serial.
            try:
                index, header = future.result()
            except _ShapeMismatchError:
                # A shape mismatch is the caller's data being inconsistent, not a read failure,
                # and carried no log line before. Keyed on this private type rather than on
                # ValueError, which astropy raises for a malformed file — that would otherwise
                # reach the caller with no log line, while the identical failure on frame 0 was
                # logged.
                raise
            except Exception as e:
                logger.error("Failed to load FITS files: {}", e)
                raise
            headers[index] = header
            report(detail=Path(paths[index]).name)
    finally:
        # cancel_futures so a raised error — including a cancelling progress callback — does not
        # wait for every queued file to be read first.
        pool.shutdown(wait=True, cancel_futures=True)

    return out, headers


def load_fits_stack(  # noqa: C901
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
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
        :class:`~neunorm.utils.progress.ProgressEvent` per file read, plus a note before the
        whole-stack variances copy that follows the read loop. A pipeline normally passes a
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

    Returns
    -------
    sc.DataArray
        DataArray with dimensions (TOF/image, y, x)

        - dims: ['TOF', 'y', 'x'] if tof_edges provided, else ['N_image', 'y', 'x']
        - coords: y, x pixel indices, and optionally TOF.
          Additionally, FITS header keys are added as (unaligned) coordinates.
          The ``COMMENT`` and ``HISTORY`` keys are skipped. A key whose value is
          constant across the stack is stored as a scalar coordinate; a key
          whose value differs across files is stored as an array coordinate
          along the stack dimension.
    """

    if max_workers is not None and max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")

    # A generator, Path.glob() or a set was accepted before this function reported progress and
    # must still be: materialise once so the count has a denominator, and because `_decode_stack`
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

    with resolve_progress(progress, stage, total=len(paths)) as report:
        full_data, headers = _decode_stack(paths, report, max_workers)

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

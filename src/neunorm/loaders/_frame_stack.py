"""
Frame-by-frame stack decoding shared by the TIFF and FITS loaders.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import _crop_fit_message

#: Threads used to decode a stack when the caller does not pass ``max_workers``. Kept modest so
#: concurrent users do not swamp a shared analysis filesystem.
DEFAULT_MAX_WORKERS = 8


class _ShapeMismatchError(ValueError):
    """A frame whose shape differs from the first frame's.

    A ``ValueError`` so the exception a caller sees is unchanged, but its own type so the pool
    loop can tell it apart from a read failure that also happens to be a ``ValueError``.
    """


class _ROIFitError(ValueError):
    """An ROI that extends past the frames of the stack being loaded.

    Raised as soon as the first frame is decoded. ``stack_shape`` is the uncropped
    ``(n_frames, ny, nx)``, so a caller loading several runs can tell runs of different detector
    size apart from an ROI that fits none of them.
    """

    def __init__(self, message: str, stack_shape: tuple[int, ...]):
        super().__init__(message)
        self.stack_shape = stack_shape


class DecodedStack(NamedTuple):
    """Result of :func:`decode_frames`."""

    #: float32 values in counts, with an allocated but unfilled variances buffer. ``None`` when the
    #: frames are not 2-D, in which case nothing was stored.
    data: Optional[sc.Variable]
    #: ``y`` and ``x`` pixel-index coordinates of the stored window.
    coords: dict[str, sc.Variable]
    #: Per-frame metadata from the reader, in input order.
    meta: list
    #: Whether any decoded frame, over its whole area, holds a negative value.
    negative: bool
    #: Uncropped shape, ``(n_frames, *frame_shape)``.
    shape: tuple[int, ...]


def format_bytes(n_bytes: int) -> str:
    """Format a byte count in binary units with one decimal, e.g. ``1.5 GiB``."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    size = float(n_bytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
    raise AssertionError("unreachable")


def variances_note(data: sc.Variable) -> str:
    """Progress note naming the stack whose variances are about to be filled."""
    n, ny, nx = data.shape
    frames = "frame" if n == 1 else "frames"
    return f"attaching variances to {n} {frames} of {nx} x {ny} px ({format_bytes(n * ny * nx * 4)})"


def has_negative(values: np.ndarray) -> bool:
    """Whether a decoded frame holds a negative value once stored as float32.

    Gives the same answer as ``np.any(stack < 0)`` on the float32 stack: a float wider than float32
    is cast first, so a tiny negative that rounds to ``-0.0`` does not count. ``np.any`` rather than
    ``min`` because a NaN in the frame would hide a negative from ``min``.
    """
    if values.dtype.kind in "ub":
        return False
    if values.dtype.kind == "f" and values.dtype.itemsize > 4:
        values = values.astype(np.float32)
    return bool(np.any(values < 0))


def allocate_stack(dims: Sequence[str], shape: Sequence[int]) -> sc.Variable:
    """Allocate an uninitialised float32 stack in counts, with room for variances.

    scipp reports a failed allocation as a bare ``std::bad_alloc``; this re-raises it naming the
    shape and the size, which is what a user needs to pick a smaller ROI or fewer images.
    """
    try:
        return sc.empty(dims=list(dims), shape=list(shape), dtype="float32", with_variances=True, unit=sc.units.counts)
    except MemoryError as e:
        size = format_bytes(2 * 4 * math.prod(shape))
        raise MemoryError(
            f"Unable to allocate {size} for a float32 stack of shape {tuple(shape)} with variances"
        ) from e


def with_poisson_variances(stack: DecodedStack) -> sc.DataArray:
    """Fill the variances of a decoded stack from its counts and wrap it, without copying the values."""
    np.copyto(stack.data.variances, stack.data.values)
    return sc.DataArray(data=stack.data, coords=stack.coords)


def _window(
    frame_shape: tuple[int, ...], n: int, bounds: Optional[tuple[int, int, int, int]]
) -> Optional[tuple[slice, slice]]:
    """The ``(y, x)`` slices of each frame to keep, or ``None`` if the frames are not 2-D.

    Raises :class:`_ROIFitError` when ``bounds`` extends past the frame.
    """
    if len(frame_shape) != 2:
        return None
    ny, nx = frame_shape
    if bounds is None:
        return slice(0, ny), slice(0, nx)
    message = _crop_fit_message(bounds, ny=ny, nx=nx)
    if message is not None:
        raise _ROIFitError(message, (n, ny, nx))
    x0, y0, x1, y1 = bounds
    return slice(y0, y1), slice(x0, x1)


def decode_frames(
    paths: Sequence[str | Path],
    report,
    max_workers: Optional[int],
    *,
    read_frame: Callable[[str | Path], tuple[np.ndarray, Any]],
    read_error: str,
    thread_name_prefix: str,
    dim: str,
    bounds: Optional[tuple[int, int, int, int]] = None,
) -> DecodedStack:
    """Decode every frame into one pre-allocated stack, in input order, keeping only ``bounds``.

    ``read_frame(path)`` returns one frame and its metadata. Each frame is decoded whole into its
    own buffer; only the window given by ``bounds`` (exclusive ``(x0, y0, x1, y1)``, the whole frame
    when ``None``) is copied into the stack, so the stack is sized by the window, not the detector.
    The shape and sign checks still see every pixel of every frame.

    **Frame order is the spectral axis.** The stack's first dimension becomes ``TOF`` or
    ``N_image``, and the ``tof`` coordinate is matched to it positionally, so a frame landing at
    the wrong index mislabels the time axis and produces a plausible-looking wrong spectrum.
    Workers therefore write ``out[i]`` for their own input index: order is preserved by
    construction rather than by collecting results carefully. Nothing here depends on the order
    in which decodes finish.

    The stack is allocated once, values and variances together, and the frames are written straight
    into scipp's buffer, so no second full-size copy is made when the variances are filled.

    Progress is emitted **from this thread**, never from a worker, which is what keeps the
    contract in :mod:`neunorm.utils.progress`: events stay synchronous and on the calling
    thread, a caller's callback still need not be thread-safe, and raising from it still
    cancels the run. Two consequences of the pool that are not the progress contract:

    - ``detail`` names files in completion order, so it does not track input order. The count
      itself is unaffected.
    - a reader's own log lines are emitted from the worker, so their order does not follow input
      order either. Only progress events are promised to be ordered.

    Cancelling is prompt but not instant: raising from the callback propagates out of this loop
    and the ``finally`` cancels every queued file, but it waits for the decodes already in flight.
    And when more than one frame is unreadable or mis-shaped, which one is named in the error
    depends on which decode finishes first.

    Raises :class:`_ROIFitError` once frame 0 is decoded if ``bounds`` does not fit it. Frames that
    are not 2-D are read and checked but not stored (``data`` is ``None``).
    """
    n = len(paths)

    # Frame 0 is decoded here, alone, because its shape is what the stack is allocated from and
    # what every other frame is checked against.
    #
    # Every `report(...)` below sits OUTSIDE these try blocks, and that placement is load-bearing:
    # raising from a progress callback is how a caller cancels, and a cancel must not be logged as
    # a read failure. tests/unit/test_progress_load_path.py pins it. The ROI fit check is outside
    # them for the same reason: a misfit ROI is the caller's argument, not a read failure.
    try:
        first, first_meta = read_frame(paths[0])
    except Exception as e:
        logger.error("{}: {}", read_error, e)
        raise
    frame_shape = first.shape
    window = _window(frame_shape, n, bounds)
    data, out, coords = None, None, {}
    if window is not None:
        y, x = window
        data = allocate_stack([dim, "y", "x"], [n, y.stop - y.start, x.stop - x.start])
        out = data.values
        out[0] = first[window]
        coords = {"y": sc.arange("y", y.start, y.stop, unit=None), "x": sc.arange("x", x.start, x.stop, unit=None)}
    negative = has_negative(first)
    del first
    meta: list = [None] * n
    meta[0] = first_meta
    report(detail=Path(paths[0]).name)

    if n > 1:
        negative |= _decode_rest(
            paths, report, max_workers, read_frame, read_error, thread_name_prefix, frame_shape, window, out, meta
        )

    return DecodedStack(data, coords, meta, negative, (n, *frame_shape))


def _decode_rest(
    paths, report, max_workers, read_frame, read_error, thread_name_prefix, frame_shape, window, out, meta
):
    """Decode frames 1..n-1 from a thread pool into ``out``; return whether any was negative."""

    def decode(index: int):
        values, frame_meta = read_frame(paths[index])
        if values.shape != frame_shape:
            raise _ShapeMismatchError(
                f"Shape mismatch in file {paths[index]}: expected {frame_shape}, got {values.shape}"
            )
        if out is not None:
            out[index] = values[window]
        return index, frame_meta, has_negative(values)

    n = len(paths)
    negative = False
    workers = max(1, min(DEFAULT_MAX_WORKERS if max_workers is None else max_workers, n - 1))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix)
    try:
        futures = [pool.submit(decode, i) for i in range(1, n)]
        for future in as_completed(futures):
            # `.result()` re-raises a worker's exception here, in the calling thread, so a bad
            # file still surfaces as it did when the read was serial.
            try:
                index, frame_meta, frame_negative = future.result()
            except _ShapeMismatchError:
                # A shape mismatch is the caller's data being inconsistent, not a read failure,
                # and carries no log line. Keyed on this private type rather than on ValueError,
                # which tifffile and astropy both raise for a malformed file — that would
                # otherwise reach the caller with no log line, while the identical failure on
                # frame 0 is logged.
                raise
            except Exception as e:
                logger.error("{}: {}", read_error, e)
                raise
            meta[index] = frame_meta
            negative |= frame_negative
            report(detail=Path(paths[index]).name)
    finally:
        # cancel_futures so a raised error — including a cancelling progress callback — does not
        # wait for every queued file to be read first.
        pool.shutdown(wait=True, cancel_futures=True)
    return negative

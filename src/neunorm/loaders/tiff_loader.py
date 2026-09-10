"""
TIFF loader for NeuNorm.

Loads TIFF stacks as scipp DataArrays.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
import tifffile
from loguru import logger
from PIL import ExifTags, Image

from neunorm.utils.progress import STAGE_LOAD_SAMPLE, ProgressLike, resolve_progress

#: TIFF tag codes read directly. Spelled out because ``ExifTags.TAGS`` maps the other way, code
#: to name.
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_PHOTOMETRIC = 262
_TAG_ORIENTATION = 274

#: Bit depths where Pillow's decode is measurably not the stored samples and tifffile cannot
#: decode at all, so there is no way to load the frame correctly. Measured on hand-written files,
#: since neither library can encode these here: a 4-bit ramp 0,1,2,3,4,5 loads from Pillow as
#: 0,17,34,51,68,85 (scaled to full 8-bit range), and a 2-bit ramp 0,1,2,3,3,2,1,0 loads as all
#: zeros. Multiplying counts by 17, or zeroing them, is worse than refusing the file. 1-bit is
#: **not** in this set: both readers return the stored bits there, faithfully.
_UNLOADABLE_BIT_DEPTHS = frozenset({2, 4})


def _as_scalar(value):
    """First element of a TIFF tag value, which Pillow reports as a tuple for some tags."""
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _needs_pillow_pixels(tags: dict) -> bool:
    """Whether this frame must be decoded by Pillow even though tifffile *can* decode it.

    This is only for cases where tifffile succeeds and returns **different values**; a file
    tifffile cannot decode at all needs no prediction, because :func:`_read_tiff_frame` simply
    falls back after the attempt fails. Enumerating what tifffile cannot do was tried first and
    was the wrong shape: the first attempt listed compression codes, and review then found the
    same hole for the floating-point predictor, for packed bit depths and for chroma subsampling.
    Anything that raises is now handled by the fallback rather than by this list growing.

    The one case that does not raise:

    - **WhiteIsZero at 8 bits or fewer.** Pillow inverts the samples for photometric 0 at that
      depth — a stored 10 loads as 245 — and tifffile returns them raw. Counts feed the Poisson
      variances, so the difference would silently change both the data and its stated uncertainty.
      At 16 bits neither library inverts, so the depth test is doing real work.

    Two other measured differences are deliberately **not** routed here, because Pillow is the one
    that is wrong and reproducing it would corrupt data:

    - **Signed samples at 8 bits.** Pillow maps ``SampleFormat`` 2 at that depth to unsigned, so a
      stored -12 loaded as 244. tifffile returns -12, which the caller's non-negative-counts guard
      then rejects. Refusing to attach Poisson variances to negative counts is the correct outcome
      and the guard already existed; silently reinterpreting them as large positive counts is not.
    - **An ``Orientation`` tag**; :func:`_read_tiff_frame` says why.
    """
    return _as_scalar(tags.get(_TAG_PHOTOMETRIC)) == 0 and (_as_scalar(tags.get(_TAG_BITS_PER_SAMPLE)) or 0) <= 8


def _pillow_pixels(img, tags: dict, path: str | Path) -> np.ndarray:
    """Decode with Pillow, refusing the two cases where its own decode is not the stored samples.

    Both refusals are load failures where the previous version returned an array, and both are
    deliberate: the array it returned was wrong. See :data:`_UNLOADABLE_BIT_DEPTHS` and
    :func:`_read_tiff_frame`.
    """
    bits = _as_scalar(tags.get(_TAG_BITS_PER_SAMPLE))
    if bits in _UNLOADABLE_BIT_DEPTHS:
        raise ValueError(
            f"Cannot load {path}: {bits}-bit samples cannot be decoded without altering them. "
            f"tifffile needs the 'imagecodecs' package to unpack them, and Pillow rescales "
            f"{bits}-bit data to full range rather than returning the stored counts. Re-save the "
            f"file at 8, 12, 16 or 32 bits per sample."
        )

    orientation = _as_scalar(tags.get(_TAG_ORIENTATION))
    if orientation not in (None, 1):
        raise ValueError(
            f"Cannot load {path}: it can only be decoded by Pillow, which does not read a file "
            f"carrying Orientation {orientation} at its stored width. Re-save it uncompressed, or "
            f"without the Orientation tag."
        )

    return np.asanyarray(img, dtype=np.float32)


def _read_tiff_frame(path: str | Path) -> tuple[np.ndarray, dict]:
    """Decode one TIFF frame, returning ``(values, {tag_code: tag_value})``.

    Split out of the read loop so it can be called per frame from a worker thread: it shares
    no state and opens its own file handle.

    **Pixels normally come from tifffile, tags always from Pillow, and the split is deliberate.**
    tifffile releases the GIL while decompressing, which is what lets a thread pool actually
    overlap decodes, and it is already installed by way of scitiff. Its *tags*, however, are not
    interchangeable with Pillow's, so reading them from tifffile would silently change the
    coordinates this loader publishes. Measured on ``tests/data/tif/sample``:

    - tag 1 is ``InteropIndex`` in ``PIL.ExifTags.TAGS`` and absent from ``tifffile.TIFF.TAGS``,
      so the coordinate would be renamed to ``"1"`` by the fallback below;
    - ``BitsPerSample`` is ``(32,)`` from Pillow and ``32`` from tifffile;
    - ``SampleFormat`` is ``(3,)`` from Pillow and the ``IntEnum`` ``SAMPLEFORMAT.IEEEFP``
      from tifffile. That one converts cleanly under ``float()``, so it would take the numeric
      branch below and become a per-frame array where it is currently a scalar.

    Reproducing Pillow's tag semantics on top of tifffile would mean reproducing its quirks for
    no speed gain — the tags are a header-only IFD read, not the cost. Pillow is a required
    dependency regardless, for ``MaskROI.from_file``.

    Only page 0 is read, matching what ``PIL.Image.open`` yields for a multi-page file;
    ``tifffile.imread`` would return every page stacked and turn a 2-D frame into 3-D.

    float32 is sufficient for neutron imaging (16-bit detectors) and halves the in-memory
    footprint of a large stack. ``copy=False`` makes the cast free for the float32 files the
    VENUS TPX1 auto-reduction writes, while still converting the integer files a CCD writes.

    **Pillow decodes the pixels whenever tifffile cannot.** That is decided by trying tifffile and
    falling back, not by predicting which files it will refuse: predicting was tried and the list
    was wrong three times over — it missed the floating-point predictor, packed bit depths and
    chroma subsampling after already covering LZW. Whatever tifffile raises on, Pillow gets, which
    is exactly what the previous version did with it. Measured cases that reach the fallback and
    load correctly through it: LZW, JPEG and CCITT compression, and 12-bit packed samples (a
    stored 0, 137, 274, 411 comes back exactly). ``_needs_pillow_pixels`` covers the one case
    where tifffile succeeds but disagrees.

    **A file carrying an Orientation tag now loads differently, on purpose, and this is the change
    to look at hardest.** Pillow applies the tag when it loads pixels. Measured against
    ``np.rot90``: for a **square** raster its result is a correct rotation at every orientation
    value, and for a non-square one it is correct at orientation 3 but not at 6 or 8, where it
    swaps width and height from the tag before decoding, reads the strips at the wrong width and
    returns interleaved values in the original shape. So Pillow was right for square frames — which
    is every real detector frame — and wrong for the shape-changing cases on non-square ones.

    This loader now returns the stored raster in every case, with the tag published as an
    ``Orientation`` coordinate for a display step to apply once, matching this project's rule that
    orientation is applied near the end for display and never implicitly inside the pipeline. The
    consequence is not only un-scrambling: **a square orientation-tagged stack now loads un-rotated
    relative to 2.4.0.** Anything calibrated against that rotation — an ROI, a mask, a dark or
    open-beam image — has to be re-checked. Orientation 1 and files with no such tag, which is
    every fixture here and everything the VENUS and MARS writers produce, are unaffected.

    ``MaskROI.from_file`` still decodes with Pillow, so it *would* apply an orientation the data
    no longer has — and on a square frame the shapes would match, making the misalignment silent.
    It therefore refuses an orientation-tagged mask outright.
    """
    # `dict(img.tag_v2)` produced {tag_code: value}; reproduce that exactly so the metadata
    # block in the caller is untouched. Pillow's open is lazy — this reads the IFD, not pixels.
    # Opened first because the tags decide whether tifffile can be trusted with the pixels, and
    # kept open so the fallback below does not have to reopen the file.
    with Image.open(path) as img:
        tags = dict(img.tag_v2)
        if _needs_pillow_pixels(tags):
            return _pillow_pixels(img, tags, path), tags

        try:
            with tifffile.TiffFile(path) as handle:
                values = handle.pages[0].asarray().astype(np.float32, copy=False)
        except Exception:  # noqa: BLE001 - a capability probe: anything tifffile refuses, Pillow gets
            # tifffile cannot decode this one. Pillow could, before this change, so let it — and if
            # it cannot either, its error is the one the previous version raised. Deliberately not
            # narrowed to the exception types seen so far: narrowing is what made the first two
            # attempts at this miss the predictor, the packed depths and chroma subsampling.
            return _pillow_pixels(img, tags, path), tags

    return values, tags


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
) -> tuple[np.ndarray, list[dict]]:
    """Decode every frame into one pre-allocated array, in input order.

    **Frame order is the spectral axis.** The stack's first dimension becomes ``TOF`` or
    ``N_image``, and the ``tof`` coordinate is matched to it positionally, so a frame landing at
    the wrong index mislabels the time axis and produces a plausible-looking wrong spectrum.
    Workers therefore write ``out[i]`` for their own input index: order is preserved by
    construction rather than by collecting results carefully. Nothing here depends on the order
    in which decodes finish.

    Pre-allocating also removes one of the three full-size copies the serial version held. It
    built a list of ``n`` frames, stacked that into a second copy, then copied the result for the
    variances; decoding straight into ``out`` collapses the first two into one, leaving the output
    and the variances copy, with only the in-flight decode buffers on top. One copy, not two: the
    measured peak drops from 5.37x the stack to 4.45x, which is what removing one of roughly five
    resident copies looks like once scipp's own copies are counted.

    Progress is emitted **from this thread**, never from a worker, which is what keeps the
    contract in :mod:`neunorm.utils.progress`: events stay synchronous and on the calling
    thread, a caller's callback still need not be thread-safe, and raising from it still
    cancels the run — the raise propagates out of the loop and the pool is shut down on the
    way out. The one visible change is that ``detail`` names files in completion order, so it
    no longer tracks input order; the count itself is unaffected.
    """
    n = len(paths)

    # Frame 0 is decoded here, alone, because its shape is what the output array is allocated
    # from and what every other frame is checked against.
    #
    # Every `report(...)` below sits OUTSIDE these try blocks, and that placement is load-bearing:
    # raising from a progress callback is how a caller cancels, and a cancel must not be logged as
    # "Error loading TIFF stack". tests/unit/test_progress_load_path.py pins it.
    try:
        first_values, first_tags = _read_tiff_frame(paths[0])
    except Exception as e:
        logger.error("Error loading TIFF stack: {}", e)
        raise
    out = np.empty((n, *first_values.shape), dtype=np.float32)
    out[0] = first_values
    tags: list[Optional[dict]] = [None] * n
    tags[0] = first_tags
    report(detail=Path(paths[0]).name)

    if n == 1:
        return out, tags

    def decode(index: int):
        values, frame_tags = _read_tiff_frame(paths[index])
        if values.shape != first_values.shape:
            raise _ShapeMismatchError(
                f"Shape mismatch in file {paths[index]}: expected {first_values.shape}, got {values.shape}"
            )
        out[index] = values
        return index, frame_tags

    workers = max(1, min(_DEFAULT_MAX_WORKERS if max_workers is None else max_workers, n - 1))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="neunorm-tiff")
    try:
        futures = [pool.submit(decode, i) for i in range(1, n)]
        for future in as_completed(futures):
            # `.result()` re-raises a worker's exception here, in the calling thread, so a bad
            # file still surfaces as it did when the read was serial.
            try:
                index, frame_tags = future.result()
            except _ShapeMismatchError:
                # A shape mismatch is the caller's data being inconsistent, not a read failure,
                # and carried no log line before. Keyed on this private type rather than on
                # ValueError, which tifffile's own TiffFileError subclasses — a malformed file
                # would otherwise reach the caller with no log line, while the identical failure
                # on frame 0 was logged.
                raise
            except Exception as e:
                logger.error("Error loading TIFF stack: {}", e)
                raise
            tags[index] = frame_tags
            report(detail=Path(paths[index]).name)
    finally:
        # cancel_futures so a raised error — including a cancelling progress callback — does not
        # wait for every queued file to be read first.
        pool.shutdown(wait=True, cancel_futures=True)

    return out, tags


def load_tiff_stack(  # noqa: C901
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
) -> sc.DataArray:
    """Load TIFF stack as scipp DataArray with variance tracking.

    Pixels are decoded with :mod:`tifffile` and the TIFF tags read with Pillow; see
    :func:`_read_tiff_frame` for why the two are split.

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
          Additionally, TIFF metadata is added as coordinates. Each metadata
          coordinate may be scalar (when its value is constant across the stack
          and not float-convertible) or stack-dimensioned (when values are
          float-convertible or differ across files).
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
            logger.error("Error loading TIFF stack: {}", e)
            raise

    if not paths:
        raise ValueError("No file paths provided")

    with resolve_progress(progress, stage, total=len(paths)) as report:
        # `_decode_stack` owns the read logging, because only it can tell a failed decode from a
        # cancelling progress callback; wrapping the whole call here would log a cancel as an I/O
        # error, which is exactly what test_cancelling_is_not_reported_as_a_read_failure forbids.
        full_data, metadata_list = _decode_stack(paths, report, max_workers)

        n_images, ny, nx = full_data.shape

        # Determine dimension names
        # If tof_edges provided, use 'TOF', else uses 'N_image'
        dim_name = "TOF" if tof_edges is not None else "N_image"
        dims = [dim_name, "y", "x"]

        # Validate data for Poisson statistics: counts must be non-negative.
        if np.any(full_data < 0):
            raise ValueError(
                "Loaded TIFF data contains negative counts; cannot attach Poisson "
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

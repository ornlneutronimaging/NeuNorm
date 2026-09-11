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
_TAG_SAMPLE_FORMAT = 339


def _as_scalar(value):
    """First element of a TIFF tag value, which Pillow reports as a tuple for some tags."""
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _needs_pillow_pixels(tags: dict) -> bool:
    """Whether Pillow, not tifffile, must decode this frame's pixels.

    **The rule is compatibility, not correctness.** This loader replaced a Pillow-only read, and
    every file that loaded before must still load, with the same values. Pillow and tifffile
    disagree about several TIFF variants, and where they disagree Pillow wins here — including the
    cases where what Pillow does is, on its own merits, wrong. Changing any of those is a separate
    decision about the library's behaviour; it is not something a change whose purpose is "load
    faster" gets to make on the way past.

    Each entry below was established by loading a purpose-written file through both the previous
    version and this one and comparing the arrays, not by reading documentation:

    - **WhiteIsZero at 8 bits or fewer**, including an absent ``PhotometricInterpretation`` tag,
      which Pillow defaults to 0. Pillow inverts the samples at that depth (a stored 10 loads as
      245); tifffile returns them raw. At 16 bits neither inverts.
    - **Signed samples at 8 bits or fewer.** Pillow reads ``SampleFormat`` 2 at that depth as
      unsigned, so a stored -12 loads as 244. tifffile returns -12, which the caller's
      non-negative-counts guard would then reject — turning a file that used to load into an error.
    - **Bit depths that are not a whole number of bytes** (1, 2, 4, 12 and so on). Pillow rescales
      sub-byte samples to the full 8-bit range: 4-bit 0,1,2,3 loads as 0,85,170,255 at 2 bits and
      x17 at 4 bits. tifffile mostly cannot unpack them at all without ``imagecodecs``.
    - **An ``Orientation`` tag other than 1.** Pillow applies it while decoding; tifffile returns
      the stored raster. Pillow's result is an exact rotation for compressed frames and for square
      uncompressed ones, and a mis-strided read for uncompressed non-square frames at the four
      shape-changing values — but either way it is what callers' ROIs and masks were built against.

    Anything tifffile simply *raises* on needs no entry here: :func:`_read_tiff_frame` falls back
    after the attempt fails, which is how LZW, JPEG, CCITT, the floating-point predictor and chroma
    subsampling are covered without this list having to predict them.
    """
    bits = _as_scalar(tags.get(_TAG_BITS_PER_SAMPLE)) or 0

    photometric = _as_scalar(tags.get(_TAG_PHOTOMETRIC))
    if photometric is None:
        # Pillow's own default for a missing tag. Not written as `tags.get(..., 0)` because an
        # empty tuple value also arrives here as None.
        photometric = 0
    if photometric == 0 and bits <= 8:
        return True

    if _as_scalar(tags.get(_TAG_SAMPLE_FORMAT)) == 2 and bits <= 8:
        return True

    if bits % 8:
        return True

    return _as_scalar(tags.get(_TAG_ORIENTATION)) not in (None, 1)


def _pillow_pixels(img) -> np.ndarray:
    """Decode with Pillow, exactly as the Pillow-only version of this loader did.

    Nothing is refused here. Pillow's decode of some variants is not the stored samples — it
    rescales sub-byte depths, inverts WhiteIsZero, reads signed 8-bit as unsigned, applies an
    Orientation tag — and all of that is reproduced rather than corrected, because those files
    loaded before and must keep loading identically. Whether any of it *should* change is a
    separate decision, not one for a change whose purpose is decoding speed.
    """
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

    - tifffile has no *name* for tag 1, which ``PIL.ExifTags.TAGS`` calls ``InteropIndex``. It
      reads the value identically, and this loader keys tags by numeric code, so that difference
      only bites if the tags were ever keyed by tifffile's names instead — then the coordinate
      would be published as ``"1"``. It is the weakest of the three reasons, not a live one;
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

    **An Orientation tag sends the frame to Pillow, so it loads exactly as it always has.**
    Pillow applies the tag while decoding and deletes tag 274 from ``tag_v2`` as a side effect, so
    the Pillow-only version this replaced returned reoriented pixels and never published an
    ``Orientation`` coordinate. Both halves are reproduced: the pixels come from Pillow, and tag
    274 is dropped from the published tags for every file, including the ones tifffile decodes.

    How faithfully Pillow reorients depends on which of its decoders runs, and it is reproduced
    either way. Measured against ``np.rot90`` over all eight values, square and non-square: its
    **libtiff** decoder, which handles compressed files, returns an exact transform every time;
    its **raw** decoder is exact for a square raster at every value and for any raster at the
    shape-preserving values 2, 3 and 4, but at 5, 6, 7 and 8 on a non-square raster it swaps width
    and height from the tag *before* decoding, reads the strips at the wrong width, and returns
    interleaved values in the original shape.

    Returning the stored raster instead, and publishing the tag for a display step, is arguably
    what this project's orientation rule wants — and it was tried here and reverted, because it
    silently changed the pixels of every orientation-tagged stack and added a coordinate to the
    HDF5 product. That is a deliberate behaviour change to make on its own terms, with a migration
    note, rather than a side effect of a change whose purpose is decoding speed.
    """
    # This opens the file twice: Pillow for the IFD, tifffile for the pixels. Reading it once and
    # parsing the buffer twice was tried and reverted -- it removes the second open, but holds the
    # raw bytes plus a BytesIO copy per in-flight frame and pushed peak memory from 4.48x the stack
    # to 4.98x, giving back most of the reduction pre-allocating is otherwise measured to deliver.
    #
    # `dict(img.tag_v2)` produced {tag_code: value}; reproduce that exactly so the metadata
    # block in the caller is untouched. Pillow's open is lazy — this reads the IFD, not pixels.
    # Opened first because the tags decide whether tifffile can be trusted with the pixels, and
    # kept open so the fallback below does not have to reopen the file.
    with Image.open(path) as img:
        tags = dict(img.tag_v2)
        # Pillow deleted the Orientation tag as a side effect of loading pixels, so the previous
        # version could never publish it as a coordinate. Dropping it unconditionally reproduces
        # that for every file, including the ones tifffile decodes.
        tags.pop(_TAG_ORIENTATION, None)
        if _needs_pillow_pixels(dict(img.tag_v2)):
            return _pillow_pixels(img), tags

        try:
            with tifffile.TiffFile(path) as handle:
                values = handle.pages[0].asarray().astype(np.float32, copy=False)
        except Exception:  # noqa: BLE001 - a capability probe: anything tifffile refuses, Pillow gets
            # tifffile cannot decode this one. Pillow could, before this change, so let it — and if
            # it cannot either, its error is the one the previous version raised. Deliberately not
            # narrowed to the exception types seen so far: narrowing is what made the first two
            # attempts at this miss the predictor, the packed depths and chroma subsampling.
            return _pillow_pixels(img), tags

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

    Pre-allocating also removes one of the roughly five full-size copies resident at peak. It
    built a list of ``n`` frames, stacked that into a second copy, then copied the result for the
    variances; decoding straight into ``out`` collapses the first two into one, leaving the output
    and the variances copy, with only the in-flight decode buffers on top. One copy, not two: the
    measured peak drops from 5.37x the stack to 4.45x, which is what removing one of roughly five
    resident copies looks like once scipp's own copies are counted.

    Progress is emitted **from this thread**, never from a worker, which is what keeps the
    contract in :mod:`neunorm.utils.progress`: events stay synchronous and on the calling
    thread, a caller's callback still need not be thread-safe, and raising from it still
    cancels the run. The visible change is that ``detail`` names files in completion order, so
    it no longer tracks input order; the count itself is unaffected.

    Cancelling is prompt but not instant: raising from the callback propagates out of this loop
    and the ``finally`` cancels every queued file, but it waits for the decodes already in flight.
    And when more than one frame is unreadable or mis-shaped, which one is named in the error
    depends on which decode finishes first, where the serial version always reported the first in
    input order.
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
            # Process metadata and add as coordinates.
            #
            # Only keys every frame carries become coordinates. The loop below indexes each frame
            # with frame 0's keys, so a key frame 0 has and a later frame lacks used to escape as a
            # bare `KeyError: <tag code>` from a public function. That was unreachable for the
            # Orientation tag until this loader stopped reading tags through Pillow's pixel load,
            # which deletes tag 274 as a side effect: a stack mixing one oriented frame with one
            # plain frame failed outright if the oriented one came first, and silently dropped the
            # coordinate if it came second.
            #
            # Published coordinates come from the INTERSECTION, but the warning is driven by the
            # UNION: a tag only a later frame carries is dropped just as surely as one only frame 0
            # carries, and warning from frame 0's keys alone would make which tags get mentioned
            # depend on the order the files were passed in.
            shared_keys = [key for key in metadata_list[0] if all(key in tags for tags in metadata_list)]
            shared = set(shared_keys)
            dropped = {key for tags in metadata_list for key in tags} - shared
            for key in sorted(dropped, key=str):
                carriers = [Path(p).name for p, tags in zip(paths, metadata_list, strict=True) if key in tags]
                logger.warning(
                    "TIFF tag {} ({}) is present in {} of {} files of the stack (e.g. {}) but not "
                    "all of them; it is not published as a coordinate.",
                    key,
                    ExifTags.TAGS.get(key, "unknown"),
                    len(carriers),
                    len(metadata_list),
                    carriers[0],
                )
            for key in shared_keys:
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

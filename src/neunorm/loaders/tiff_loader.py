"""
TIFF loader for NeuNorm.

Loads TIFF stacks as scipp DataArrays.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
import tifffile
from loguru import logger
from PIL import ExifTags, Image

from neunorm.data_models.roi import ROILike, _checked_crop_bounds
from neunorm.loaders._frame_stack import decode_frames, variances_note, with_poisson_variances
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

    tifffile pixels are returned in the file's own dtype and cast to float32 as each frame is copied
    into the stack, which is the same conversion an up-front ``astype(np.float32)`` makes. A 16-bit
    frame's decode buffer is then half the size of a float32 one, and that buffer, held once per
    worker, is what bounds the memory of a cropped load. Pillow-decoded frames are float32 already.

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
    # This opens the file twice: Pillow for the IFD, tifffile for the pixels. Reading the bytes once
    # and parsing them twice would remove the second open, but would hold the raw bytes plus a
    # BytesIO copy per in-flight frame, which costs more memory than the open saves time.
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
                values = handle.pages[0].asarray()
        except Exception:  # noqa: BLE001 - a capability probe: anything tifffile refuses, Pillow gets
            # tifffile cannot decode this one. Pillow could, before this change, so let it — and if
            # it cannot either, its error is the one the previous version raised. Deliberately not
            # narrowed to the exception types seen so far: narrowing is what made the first two
            # attempts at this miss the predictor, the packed depths and chroma subsampling.
            return _pillow_pixels(img), tags

    return values, tags


def load_tiff_stack(
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
    roi: Optional[ROILike] = None,
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
          Additionally, TIFF metadata is added as coordinates. Each metadata
          coordinate may be scalar (when its value is constant across the stack
          and not float-convertible) or stack-dimensioned (when values are
          float-convertible or differ across files).
    """
    return _load_tiff_stack(paths, tof_edges, progress=progress, stage=stage, max_workers=max_workers, roi=roi)[0]


def _load_tiff_stack(  # noqa: C901
    paths: Sequence[str | Path],
    tof_edges: Optional[np.ndarray] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
    roi: Optional[ROILike] = None,
) -> tuple[sc.DataArray, tuple[int, ...]]:
    """:func:`load_tiff_stack`, also returning the uncropped stack shape ``(n_frames, ny, nx)``."""
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
            logger.error("Error loading TIFF stack: {}", e)
            raise

    if not paths:
        raise ValueError("No file paths provided")

    # If tof_edges provided, use 'TOF', else uses 'N_image'
    dim_name = "TOF" if tof_edges is not None else "N_image"

    with resolve_progress(progress, stage, total=len(paths)) as report:
        # `decode_frames` owns the read logging, because only it can tell a failed decode from a
        # cancelling progress callback; wrapping the whole call here would log a cancel as an I/O
        # error, which is exactly what test_cancelling_is_not_reported_as_a_read_failure forbids.
        stack = decode_frames(
            paths,
            report,
            max_workers,
            read_frame=_read_tiff_frame,
            read_error="Error loading TIFF stack",
            thread_name_prefix="neunorm-tiff",
            dim=dim_name,
            bounds=bounds,
        )
        metadata_list = stack.meta

        n_images, _, _ = stack.shape

        # Validate data for Poisson statistics: counts must be non-negative, over every whole frame.
        if stack.negative:
            raise ValueError(
                "Loaded TIFF data contains negative counts; cannot attach Poisson "
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

        return da, stack.shape

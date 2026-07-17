"""Region-of-interest data models for NeuNorm 2.0: rectangular ``ROI`` and mask-based ``MaskROI``."""

import collections.abc
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipp as sc
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator


class ROI(BaseModel):
    """Rectangular region of interest with named, self-documenting bounds.

    Define it either by explicit stop indices or by size — the two forms are equivalent::

        ROI(x0=10, y0=20, x1=30, y1=40)          # exclusive stops
        ROI(x0=10, y0=20, width=20, height=20)   # the same 20x20 region

    Stop indices are **exclusive** (Python slice semantics), matching ``apply_roi``,
    ``apply_air_region_correction`` and the ``background_roi`` flux proxy. An ``ROI`` may be passed
    anywhere those APIs accept an ``(x0, y0, x1, y1)`` tuple; the bare-tuple form keeps working
    unchanged for backward compatibility.

    Parameters
    ----------
    x0, y0 : int
        Lower (inclusive) pixel bounds in x and y.
    x1, y1 : int, optional
        Upper (exclusive) pixel bounds. Provide these **or** ``width``/``height``.
    width, height : int, optional
        Extent in x and y; ``x1 = x0 + width`` and ``y1 = y0 + height``. Provide these **or**
        ``x1``/``y1``.
    inclusive : bool, optional
        Interpret the upper bounds as **inclusive** (default ``False`` = exclusive Python-slice
        semantics). When ``True``, the resolved ``x1``/``y1`` are bumped by one so the region spans
        ``(width + 1) x (height + 1)`` pixels (and an explicit ``x1``/``y1`` is included). This is the
        legacy NeuNorm 1.x / iBeatles convention; ``as_bounds()`` always returns exclusive stops, so
        the rest of the library stays exclusive. Bare tuples are always exclusive — use the ``ROI``
        type to opt into inclusive extents.

    Examples
    --------
    >>> ROI(x0=10, y0=20, x1=30, y1=40).as_bounds()
    (10, 20, 30, 40)
    >>> ROI(x0=10, y0=20, width=20, height=20).as_bounds()
    (10, 20, 30, 40)
    >>> ROI(x0=10, y0=20, width=20, height=20, inclusive=True).as_bounds()
    (10, 20, 31, 41)
    """

    x0: int
    y0: int
    x1: Optional[int] = None
    y1: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    inclusive: bool = False

    @model_validator(mode="after")
    def _resolve_bounds(self):
        """Resolve width/height to exclusive stops and validate the rectangle (runs automatically)."""
        if (self.x1 is None) == (self.width is None):
            raise ValueError("ROI requires exactly one of 'x1' or 'width'")
        if (self.y1 is None) == (self.height is None):
            raise ValueError("ROI requires exactly one of 'y1' or 'height'")
        if self.x1 is None:
            self.x1 = self.x0 + self.width
        if self.y1 is None:
            self.y1 = self.y0 + self.height
        if self.inclusive:
            # inclusive upper bound -> exclusive stop covers one more pixel on each axis
            self.x1 += 1
            self.y1 += 1
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"Invalid ROI {self.as_bounds()}: need 0 <= x0 < x1 and 0 <= y0 < y1")
        return self

    def as_bounds(self) -> tuple[int, int, int, int]:
        """Return the ROI as an ``(x0, y0, x1, y1)`` tuple with exclusive stop indices."""
        return (self.x0, self.y0, self.x1, self.y1)


class MaskROI(BaseModel):
    """Arbitrary-shape region of interest defined by a pixel **selection** mask.

    ``selection`` is a 2D boolean array the same spatial size as the images, where **True (or any
    nonzero value) means the pixel IS in the region** and False/0 means it is not.

    .. warning::
        This is the **opposite polarity of scipp masks**: ``DataArray.masks`` mark pixels to
        *exclude* from reductions, while ``MaskROI.selection`` marks pixels to *include* in the
        region. Use :meth:`from_dataarray_mask` (with its required ``invert`` flag) to convert
        between the two conventions explicitly.

    A ``MaskROI`` is accepted by the **region-statistics** operations that take a rectangular
    :class:`ROI` (``background_roi``, air-region correction, and the pipeline
    ``air_roi``/``background_roi`` parameters), and may be pooled together with rectangles in one
    region list. Selected pixels that are masked in the data (dead/hot pixels) are excluded from
    region statistics exactly as they are for rectangular ROIs. Cropping (``apply_roi`` / the
    pipeline ``roi=``) stays rectangle-only — an arbitrary shape has no rectangular crop.

    The selection is canonicalized at construction to a C-contiguous, read-only boolean array in
    ``(y, x)`` row-major order (the same layout as loaded TIFF/FITS frames), regardless of the
    input form. Any nonzero value selects: integer or float arrays are thresholded ``!= 0``.

    Parameters
    ----------
    selection : numpy.ndarray or scipp.Variable
        2D selection array. A scipp Variable must have dims ``y`` and ``x`` (either order — it is
        transposed by name). A bare numpy array is interpreted per ``dims``.
    dims : tuple[str, str], optional
        Axis order of a bare numpy ``selection`` — ``("y", "x")`` (default, row-major image
        convention) or ``("x", "y")``. Ignored for scipp input (named dims win). Note that for a
        square detector a transposed numpy mask cannot be detected — prefer scipp Variables or the
        default row-major layout.
    source : str, optional
        Provenance label (set automatically by :meth:`from_file` / :meth:`from_dataarray_mask`).

    Examples
    --------
    >>> sel = np.zeros((256, 256), dtype=bool)
    >>> sel[100:120, 40:200] = True    # any shape works, not just rectangles
    >>> region = MaskROI(selection=sel)
    >>> region.n_selected
    3200
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    selection: Any
    dims: tuple[str, str] = ("y", "x")
    source: str = "array"

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data):
        """Coerce any accepted input to a read-only C-contiguous bool array in (y, x) order."""
        if not isinstance(data, collections.abc.Mapping):
            return data
        # Pydantic accepts any Mapping (e.g. UserDict), not just dict; copy into a mutable dict so
        # the canonicalization below (bool dtype, non-empty check, owned writable buffer) is
        # enforced for every Mapping input, not only literal dicts.
        data = dict(data)
        sel = data.get("selection")
        dims = tuple(data.get("dims", ("y", "x")))
        if isinstance(sel, sc.Variable):
            if set(sel.dims) != {"y", "x"}:
                raise ValueError(f"MaskROI scipp selection must have dims {{'y', 'x'}}; got {sel.dims}")
            arr = np.asarray(sel.transpose(("y", "x")).values)
        else:
            arr = np.asarray(sel)
            if dims == ("x", "y"):
                arr = arr.T
            elif dims != ("y", "x"):
                raise ValueError(f"MaskROI dims must be ('y', 'x') or ('x', 'y'); got {dims}")
        if arr.ndim != 2:
            raise ValueError(f"MaskROI selection must be a 2D (y, x) array; got shape {arr.shape}")
        # One thresholding rule for every input path: nonzero selects. `!= 0` also copies, so the
        # caller's array can never mutate the (frozen, read-only) canonical buffer.
        canonical = np.ascontiguousarray(arr != 0)
        if not canonical.any():
            raise ValueError("MaskROI selection must select at least one pixel (it is all zeros/False)")
        canonical.setflags(write=False)
        data["selection"] = canonical
        data["dims"] = ("y", "x")
        return data

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MaskROI":
        """Load a selection mask from an image file (TIFF/PNG/...; any nonzero pixel selects).

        Matches the mask images users draw in e.g. ImageJ and save alongside their data. RGB(A)
        images select where any channel is nonzero; only the first frame of a multi-frame file is
        read. The file's row-major ``(y, x)`` layout is used as-is.
        """
        from PIL import Image

        p = Path(path)
        with Image.open(p) as img:
            arr = np.asarray(img)
        if arr.ndim == 3:  # RGB / RGBA: any nonzero channel selects
            arr = arr.any(axis=-1)
        return cls(selection=arr, source=str(p))

    @classmethod
    def from_dataarray_mask(cls, da: sc.DataArray, name: str, *, invert: bool) -> "MaskROI":
        """Build a selection from a mask already carried on a scipp DataArray.

        scipp masks are **exclusion** masks (True = pixel excluded), while ``MaskROI`` is a
        **selection** (True = pixel in the region) — so ``invert`` is required, with no default:

        - ``invert=True``: select the pixels the scipp mask *keeps* (region = NOT masked).
        - ``invert=False``: select the pixels the scipp mask *flags* (region = the masked pixels).
        """
        if name not in da.masks:
            raise ValueError(f"DataArray has no mask named {name!r}; available: {list(da.masks.keys())}")
        mask = da.masks[name]
        if set(mask.dims) != {"y", "x"}:
            raise ValueError(f"mask {name!r} must be spatial with dims {{'y', 'x'}}; got {mask.dims}")
        sel = ~mask if invert else mask
        return cls(selection=sel, source=f"dataarray_mask:{name}")

    @property
    def n_selected(self) -> int:
        """Number of selected pixels."""
        return int(self.selection.sum())

    @property
    def shape(self) -> tuple[int, int]:
        """Selection shape as ``(ny, nx)``."""
        return tuple(self.selection.shape)

    def sha256(self) -> str:
        """Hex digest of the canonical (y, x) boolean buffer — stable across input forms."""
        return hashlib.sha256(self.selection.tobytes()).hexdigest()

    def bounding_box(self) -> tuple[int, int, int, int]:
        """Tight bounding box of the selection as ``(x0, y0, x1, y1)`` with exclusive stops."""
        ys, xs = np.nonzero(self.selection)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def provenance_summary(self) -> dict:
        """JSON-safe summary for output-file provenance (shape, n_selected, source, sha256)."""
        return {
            "shape": [int(s) for s in self.selection.shape],
            "n_selected": self.n_selected,
            "source": self.source,
            "sha256": self.sha256(),
        }

    @field_serializer("selection")
    def _serialize_selection(self, _value: np.ndarray, _info):
        """Serialize the array field as its provenance summary so model_dump_json works."""
        return self.provenance_summary()

    def __eq__(self, other) -> bool:
        if not isinstance(other, MaskROI):
            return NotImplemented
        return np.array_equal(self.selection, other.selection)

    def __hash__(self) -> int:
        return hash((self.shape, self.sha256()))

    def __repr__(self) -> str:
        ny, nx = self.shape
        return f"MaskROI(shape=(ny={ny}, nx={nx}), n_selected={self.n_selected}, source={self.source!r})"

    __str__ = __repr__


# An ``ROI`` or a bare 4-element ``(x0, y0, x1, y1)`` tuple/list — accepted interchangeably by
# ROI-taking APIs (``as_roi_bounds`` coerces either form to a bounds tuple).
ROILike = Union[ROI, tuple[int, int, int, int], list[int]]

# Any single region: a rectangle (in any ROILike spelling) or a mask.
RegionLike = Union[ROILike, MaskROI]

# One region or a sequence of regions (pooled). A bare 4-int tuple/list is a single rectangle.
RegionsLike = Union[RegionLike, Sequence[RegionLike]]


def as_roi_bounds(roi: ROILike) -> tuple[int, int, int, int]:
    """Coerce an :class:`ROI` (or a bare 4-element ``(x0, y0, x1, y1)`` sequence) to a bounds tuple.

    A bare sequence is returned as a plain tuple so downstream code sees a consistent
    ``(x0, y0, x1, y1)`` form regardless of how the ROI was specified. Element-type/bounds validation
    is left to the consumer (and to :class:`ROI` for the named form); only the 4-element length is
    enforced here so a malformed bare sequence fails fast at one place. Pydantic models other than
    :class:`ROI` (e.g. a :class:`MaskROI`) are rejected explicitly — iterating a model would yield
    meaningless ``(field, value)`` pairs.
    """
    if isinstance(roi, ROI):
        return roi.as_bounds()
    if isinstance(roi, BaseModel):
        raise ValueError(
            f"expected a rectangular ROI or an (x0, y0, x1, y1) sequence; got {type(roi).__name__} — "
            "mask regions are only accepted by APIs documented to take a MaskROI"
        )
    bounds = tuple(roi)
    if len(bounds) != 4:
        raise ValueError(f"ROI must be a tuple of 4 integers (x0, y0, x1, y1), or an ROI; got {bounds!r}")
    return bounds


def _plain_int_bounds(bounds: tuple) -> tuple[int, int, int, int]:
    """Coerce NumPy integer bounds to built-in ``int`` (JSON provenance stays numeric)."""
    return tuple(int(v) if isinstance(v, np.integer) else v for v in bounds)


def as_region_list(regions: RegionsLike, arg_name: str = "region") -> list[Union[tuple[int, int, int, int], MaskROI]]:
    """Normalize a region argument to a list of exclusive ``(x0, y0, x1, y1)`` bounds and/or ``MaskROI``.

    The unified-region generalization of ``as_roi_bounds_list``: accepts a single region — an
    :class:`ROI`, a :class:`MaskROI`, or a bare 4-int ``(x0, y0, x1, y1)`` sequence — or a
    **sequence** of those (pooled). A bare 4-int sequence is always ONE rectangle (so ``[0, 0, 1, 1]``
    is a rectangle, never a tiny mask). Raw numpy/scipp arrays and file paths are **not** accepted
    here — wrap them explicitly: ``MaskROI(selection=...)`` / ``MaskROI.from_file(...)``.
    Idempotent on its own output. NumPy integer bounds are coerced to built-in ``int``.

    Parameters
    ----------
    regions : RegionsLike
        The region argument to normalize.
    arg_name : str, optional
        Name used in error messages (e.g. ``"background_roi"``, ``"air_roi"``).
    """
    if isinstance(regions, MaskROI):
        return [regions]
    if isinstance(regions, ROI):
        return [_plain_int_bounds(regions.as_bounds())]
    if isinstance(regions, (str, bytes)) or not isinstance(regions, collections.abc.Sequence):
        raise ValueError(
            f"{arg_name} must be an ROI, a MaskROI, an (x0, y0, x1, y1) tuple, or a sequence of those; got {regions!r}"
        )
    if len(regions) == 4 and all(isinstance(i, (int, np.integer)) for i in regions):
        return [_plain_int_bounds(as_roi_bounds(tuple(regions)))]
    if len(regions) == 4 and all(isinstance(i, (int, float, np.integer, np.floating)) for i in regions):
        # a 4-number sequence with non-integer entries is a malformed rectangle, not a region list
        raise ValueError(f"{arg_name} must be a tuple of 4 integers (x0, y0, x1, y1); got {regions!r}")
    if len(regions) == 0:
        raise ValueError(f"{arg_name} list must contain at least one ROI")
    # a bare sequence of ints is a SINGLE rectangle (handled above when len == 4); an int element
    # here means a malformed single ROI (wrong length), not a sequence of regions.
    if any(isinstance(e, (int, np.integer)) for e in regions):
        raise ValueError(f"{arg_name} must be a tuple of 4 integers (x0, y0, x1, y1); got {regions!r}")
    return [r if isinstance(r, MaskROI) else _plain_int_bounds(as_roi_bounds(r)) for r in regions]


def region_provenance(region: Union[tuple, MaskROI]):
    """JSON-writer-safe provenance for ONE normalized region.

    A rectangle passes through unchanged (the pinned flat ``(x0, y0, x1, y1)`` form); a
    :class:`MaskROI` becomes a **JSON string** ``'{"mask": {...summary...}}'`` — never a bare dict,
    which the TIFF metadata writer rejects at export time.
    """
    import json

    if isinstance(region, MaskROI):
        return json.dumps({"mask": region.provenance_summary()})
    return region


def as_region_provenance(regions: list):
    """JSON-writer-safe provenance for a normalized region list (from :func:`as_region_list`).

    A single rectangle stays a flat ``[x0, y0, x1, y1]`` int list (the pinned HDF5 native form);
    anything else becomes a list whose entries are ``[x0, y0, x1, y1]`` lists or
    ``{"mask": {shape, n_selected, source, sha256}}`` dicts — a **list** of dicts JSON-encodes
    losslessly in both the HDF5 and TIFF writers (a bare top-level dict would not).
    """
    entries = [({"mask": r.provenance_summary()} if isinstance(r, MaskROI) else list(r)) for r in regions]
    if len(entries) == 1 and not isinstance(regions[0], MaskROI):
        return entries[0]
    return entries

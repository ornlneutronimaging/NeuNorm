"""Region-of-interest data model for NeuNorm 2.0."""

from typing import Optional, Union

from pydantic import BaseModel, model_validator


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

    Examples
    --------
    >>> ROI(x0=10, y0=20, x1=30, y1=40).as_bounds()
    (10, 20, 30, 40)
    >>> ROI(x0=10, y0=20, width=20, height=20).as_bounds()
    (10, 20, 30, 40)
    """

    x0: int
    y0: int
    x1: Optional[int] = None
    y1: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

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
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"Invalid ROI {self.as_bounds()}: need 0 <= x0 < x1 and 0 <= y0 < y1")
        return self

    def as_bounds(self) -> tuple[int, int, int, int]:
        """Return the ROI as an ``(x0, y0, x1, y1)`` tuple with exclusive stop indices."""
        return (self.x0, self.y0, self.x1, self.y1)


# An ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple — accepted interchangeably by ROI-taking APIs.
ROILike = Union[ROI, tuple[int, int, int, int]]


def as_roi_bounds(roi: ROILike) -> tuple[int, int, int, int]:
    """Coerce an :class:`ROI` (or a bare ``(x0, y0, x1, y1)`` tuple/list) to a bounds tuple.

    A bare sequence is returned as a plain tuple so downstream code sees a consistent
    ``(x0, y0, x1, y1)`` form regardless of how the ROI was specified.
    """
    if isinstance(roi, ROI):
        return roi.as_bounds()
    return tuple(roi)

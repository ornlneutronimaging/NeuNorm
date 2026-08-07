"""
Configuration for the pre-normalization moving window.

One object rather than four or five loose parameters, following the :class:`BinningConfig`
precedent: the three VENUS TOF pipelines are already at thirteen to fifteen parameters each, and a
kind, three sizes, a dimension and an edge mode would be six more, three times over.

Sizes are addressed by **dimension name**. iBeatles takes a positional ``(y, x, lambda)`` tuple,
which cannot be carried across: NeuNorm's event path produces ``(tof, x, y)`` — x before y — so a
tuple copied from an iBeatles configuration would silently transpose the two spatial axes, and for a
non-square kernel that is a wrong answer which still looks entirely plausible.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from neunorm.processing.moving_window import EDGE_MODES, MovingWindowKind


class MovingWindow(BaseModel):
    """A box window applied to both stacks after they are combined and before normalization.

    Each pixel is replaced by the average — or, with ``kind="sum"``, the total — of the pixels in a
    box around it.

    Parameters
    ----------
    kind : {"average", "sum"}
        Whether to divide by the number of pixels collected. Applied before normalization the two
        give indistinguishable transmission, because the kernel count cancels in the sample/open-beam
        ratio; ``"sum"`` differs only in the intermediate counts.
    x, y : int
        Window length along each spatial axis, in **post-crop, post-spatial-rebin** pixels. A window
        of 1 leaves that axis untouched.
    tof : int
        Window length along the time-of-flight axis. Requires ``dimension="3D"``.
    dimension : {"2D", "3D"}
        ``"2D"`` filters each spectral frame independently, as iBeatles' 2-D kernel does. ``"3D"``
        also averages along TOF, which blurs spectral features — a resonance dip or a Bragg edge —
        and so has to be asked for explicitly rather than inferred from a ``tof`` size.
    mode : str
        Edge policy, passed to :func:`scipy.ndimage.uniform_filter`. Defaults to mirroring the frame
        edge, which is what iBeatles does.

    Raises
    ------
    ValueError
        If every size is 1 (the identity window), or if a ``tof`` size is given without
        ``dimension="3D"``.

    Examples
    --------
    >>> MovingWindow(x=3, y=3).sizes()
    {'x': 3, 'y': 3}
    >>> MovingWindow(x=5, y=5, kind="sum").kernel_pixels
    25
    >>> MovingWindow(x=3, y=3, tof=5, dimension="3D").sizes()
    {'x': 3, 'y': 3, 'tof': 5}
    """

    kind: MovingWindowKind = Field(default="average", description="Divide by the pixel count, or do not")
    x: int = Field(default=1, ge=1, description="Window length along x, in post-crop pixels")
    y: int = Field(default=1, ge=1, description="Window length along y, in post-crop pixels")
    tof: int = Field(default=1, ge=1, description="Window length along TOF; requires dimension='3D'")
    dimension: Literal["2D", "3D"] = Field(default="2D", description="Filter each frame, or also along TOF")
    # Subscripting Literal with the tuple keeps the accepted set in one place — the filter's own —
    # so the config cannot drift from what the filter will actually take.
    mode: Literal[EDGE_MODES] = Field(default="reflect", description="Edge policy, as scipy's")

    @model_validator(mode="after")
    def _check_the_window_does_something(self) -> "MovingWindow":
        """Reject the two configurations that would quietly not be what was asked for."""
        if self.dimension == "2D" and self.tof != 1:
            raise ValueError(
                f"tof={self.tof} needs dimension='3D'. A 2-D window filters each spectral frame on "
                "its own; averaging along TOF as well blurs resonance dips and Bragg edges, so it "
                "has to be asked for rather than inferred."
            )
        if self.x == 1 and self.y == 1 and self.tof == 1:
            raise ValueError(
                "a moving window of 1 in every dimension is the identity and would filter nothing. "
                "Give a size greater than 1 (e.g. MovingWindow(x=3, y=3)), or leave moving_window unset."
            )
        return self

    def sizes(self) -> dict:
        """The window as ``{dim: length}`` for :func:`~neunorm.processing.moving_window.moving_window`.

        The ``tof`` entry is present only for a 3-D window, so a 2-D one cannot reach the spectral
        axis even if a size was set and then the dimension changed.
        """
        window = {"x": self.x, "y": self.y}
        if self.dimension == "3D":
            window["tof"] = self.tof
        return window

    @property
    def kernel_pixels(self) -> int:
        """How many pixels one window covers — the number a ``"sum"`` does not divide by."""
        product = 1
        for length in self.sizes().values():
            product *= length
        return product

    def provenance(self) -> dict:
        """What to record in the output file so a filtered result is identifiable later.

        A log line scrolls away; the pixels of a filtered image are no longer independent and nothing
        in the array itself shows it, so the window travels with the data.
        """
        return {
            "kind": self.kind,
            "sizes": self.sizes(),
            "dimension": self.dimension,
            "mode": self.mode,
            "kernel_pixels": self.kernel_pixels,
        }

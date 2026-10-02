"""
Run loading and combining shared by the MARS and VENUS CCD/CMOS pipelines.
"""

from typing import Optional, Sequence

import scipp as sc
from loguru import logger

from neunorm.loaders._frame_stack import _ROIFitError
from neunorm.loaders.stack_loader import _load_stack
from neunorm.processing.run_combiner import _require_same_shape, combine_runs
from neunorm.utils.progress import ProgressLike


class FrameSize:
    """The uncropped frame size ``(ny, nx)`` that every input family of one pipeline run must share.

    Sample, open-beam and dark frames are subtracted and divided pixel by pixel, so each family's
    frames must be the size of the first family's; only the number of frames may differ.
    """

    def __init__(self) -> None:
        self._family: Optional[str] = None
        self._size: Optional[tuple[int, int]] = None

    def check(self, family: str, stack_shape: tuple[int, ...]) -> None:
        """Record the frame size of the first family checked; raise ``ValueError`` if a later one differs.

        ``stack_shape`` is the family's uncropped ``(n_frames, ny, nx)``. The error is logged before it
        is raised and names both families and both sizes.
        """
        _, ny, nx = stack_shape
        if self._size is None:
            self._family, self._size = family, (ny, nx)
            return
        if (ny, nx) != self._size:
            base_ny, base_nx = self._size
            message = (
                f"{family.capitalize()} frames have size (y={ny}, x={nx}), but {self._family} frames have size"
                f" (y={base_ny}, x={base_nx}); sample, open-beam and dark frames must be the same size"
            )
            logger.error(message)
            raise ValueError(message)


def load_runs(
    groups: Sequence[Sequence],
    *,
    roi: Optional[tuple[int, int, int, int]],
    progress: ProgressLike,
    family: str,
    frame_size: FrameSize,
) -> list[sc.DataArray]:
    """Load every run of one input family (sample, open beam or dark), cropping each frame to ``roi``.

    Only the ROI of each frame is kept, so memory scales with the region rather than the detector.
    With or without ``roi``, each run's uncropped shape is checked against the first run's as it
    loads, raising and logging the error :func:`~neunorm.processing.run_combiner.combine_runs` raises
    for runs of different shape (for example, taken at different binning), so a mismatch within the
    family is reported before any later family is read. The first run's uncropped frame size is
    checked against the families loaded before it with ``frame_size``. An ROI that does not fit a run
    is reported as a mismatch with the frames loaded before it, and as an ROI error when no frames
    were loaded before.

    Parameters
    ----------
    groups : Sequence[Sequence[str | Path]]
        One sequence of image paths per run.
    roi : tuple[int, int, int, int] or None
        Crop bounds ``(x0, y0, x1, y1)``; ``None`` keeps whole frames.
    progress : bool, callable or ProgressReporter
        The family's load-stage reporter, shared by every run so the count spans the family.
    family : str
        The family's name in error messages: ``"sample"``, ``"open-beam"`` or ``"dark"``.
    frame_size : FrameSize
        Shared by every family of one pipeline run.

    Returns
    -------
    list[sc.DataArray]
        One stack per run, in input order.
    """
    if roi is not None:
        logger.info("Cropping frames to ROI {} as they are loaded", roi)
    runs: list[sc.DataArray] = []
    base_shape = base_dims = None
    for i, paths in enumerate(groups):
        try:
            run, shape = _load_stack(paths, progress=progress, roi=roi)
        except _ROIFitError as error:
            if base_shape is not None:
                _require_same_shape(i, error.stack_shape, base_dims, base_shape, base_dims)
            else:
                frame_size.check(family, error.stack_shape)
            raise
        if base_shape is None:
            frame_size.check(family, shape)
            base_shape, base_dims = shape, run.dims
        else:
            _require_same_shape(i, shape, run.dims, base_shape, base_dims)
        runs.append(run)
    return runs


def combine_owned_runs(runs: list[sc.DataArray], **kwargs) -> sc.DataArray:
    """:func:`~neunorm.processing.run_combiner.combine_runs` for runs no caller holds.

    ``combine_runs`` copies a single run so that its result never aliases the caller's data. The
    runs a pipeline loads itself are dropped once combined, so a single run is returned as is
    rather than copied. Several runs are combined by ``combine_runs`` with ``kwargs``.
    """
    if len(runs) == 1:
        return runs[0]
    return combine_runs(runs, **kwargs)

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


def load_runs(
    groups: Sequence[Sequence],
    *,
    roi: Optional[tuple[int, int, int, int]],
    progress: ProgressLike,
) -> list[sc.DataArray]:
    """Load every run of one input family (sample, open beam or dark), cropping each frame to ``roi``.

    Only the ROI of each frame is kept, so memory scales with the region rather than the detector.
    Cropping before combining hides each run's detector size from
    :func:`~neunorm.processing.run_combiner.combine_runs`, whose shape check is what rejects runs of
    different size (for example, taken at different binning). With ``roi`` set, each run's uncropped
    shape is therefore checked against the first run's here, raising the error ``combine_runs``
    raises. An ROI that does not fit a run is reported as a shape mismatch when it fits the first
    run, and as an ROI error otherwise.

    Parameters
    ----------
    groups : Sequence[Sequence[str | Path]]
        One sequence of image paths per run.
    roi : tuple[int, int, int, int] or None
        Crop bounds ``(x0, y0, x1, y1)``; ``None`` keeps whole frames.
    progress : bool, callable or ProgressReporter
        The family's load-stage reporter, shared by every run so the count spans the family.

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
            raise
        if base_shape is None:
            base_shape, base_dims = shape, run.dims
        elif roi is not None:
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

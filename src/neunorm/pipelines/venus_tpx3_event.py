"""
VENUS TPX3 event pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

import scipp as sc

from neunorm import __version__
from neunorm.data_models.moving_window import MovingWindow
from neunorm.data_models.roi import RegionLike, RegionsLike, ROILike
from neunorm.data_models.tof import BinningConfig
from neunorm.loaders.event_loader import LOAD_EVENT_NEXUS_STEPS, load_event_nexus
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.pipelines._tof_spine import (
    TofPipelineProfile,
    coerce_roi_arguments,
    reduce_tof_stacks,
)
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.event_converter import convert_events_to_histogram
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_HISTOGRAM,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    Progress,
    resolve_progress,
    total_across_groups,
)

#: Event mode detects hot pixels, re-detects from the open beam after a spatial rebin, and hard-codes
#: its TIFF detector model rather than reading a ``detector`` coordinate.
_TPX3_EVENT_PROFILE = TofPipelineProfile(
    label="VENUS TPX3 event",
    detect_hot=True,
    remask_after_spatial_rebin_from="ob",
    hdf5_hot_pixel_mask="hot_pixels",
    tiff_detector_model="TPX3",
)


def run_venus_tpx3_event_pipeline(
    sample_paths: Sequence[str | Path],
    ob_paths: Sequence[str | Path],
    binning: BinningConfig,
    output_path: Path,
    roi: Optional[ROILike] = None,
    air_roi: Optional[RegionLike] = None,
    rebin_by_tof: Optional[bool | int | list | tuple] = False,
    rebin_by_spatial: Optional[int | tuple[int, int]] = None,
    detector_shape: tuple[int, int] = (514, 514),
    event_id_offset: int = 1_000_000,
    bank_name: str = "bank100",
    flight_path: sc.Variable = sc.scalar(VENUS_FLIGHT_PATH_M, unit="m"),
    *,
    rebin_reduction: Optional[Literal["mean", "sum", "median"]] = None,
    tiff_one_file_per_image: bool = False,
    spectrum_roi: Optional[RegionsLike] = None,
    spectrum_roi_strict: bool = True,
    moving_window: Optional[MovingWindow] = None,
    progress: Progress = False,
) -> sc.DataArray:
    """Execute VENUS TPX3 event normalization pipeline.

    Pipeline Steps
    - Load event data
    - Run combine (optional)
    - ROI clip (optional)
    - Dead pixel detection
    - Hot pixel detection
    - Statistics analysis (only when ``rebin_by_tof=True``)
    - Coarsening strategy (spatial/TOF)
    - Event → histogram conversion (flexible binning)
    - Beam correction (p_charge)
    - Normalization (TOF-resolved)
    - Air region correction (optional)
    - Error propagation
    - Output


    Parameters
    ----------
    sample_paths : Sequence[str | Path]
        List of paths to sample HDF5 files.
    ob_paths : Sequence[str | Path]
        List of paths to open beam HDF5 files
    binning : BinningConfig
        Configuration for TOF/energy/wavelength binning. Required for event → histogram conversion.
    output_path : Path
        Path to save the output file (HDF5 or TIFF)
    roi : Optional[tuple]
        Region of interest to crop to — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple.
    air_roi : ROI, MaskROI, or tuple, optional
        Region of interest for air correction — an ``ROI``, a bare ``(x0, y0, x1, y1)`` tuple, or an
        arbitrary-shape ``MaskROI`` selection. If None, air correction is not applied.
    rebin_by_tof : bool, int, or list/tuple of [start, stop], optional
        TOF rebinning applied to the histogrammed event stack (so the bin list indexes the TOF
        histogram bins, exactly as in the histogram pipeline). ``True`` uses the statistics-based
        recommended factor; an ``int`` is a uniform factor; a ``[[start, stop], ...]`` list defines
        explicit half-open bins (variable width). One output image per range.
        Frames covered by no range are dropped silently (a deliberate, requested behavior). Note
        that dropping frames leaves the output images covering disjoint time bands, which the
        ``N+1`` bin-edge ``tof`` axis cannot describe exactly — the bin before a dropped span has its
        closing edge (and derived ``wavelength``/``energy`` edge) widened by the omitted span, and the
        result is not a continuous spectrum. Prefer contiguous ranges when the data will be analysed
        as a spectrum; see :mod:`neunorm.tof.histogram_rebinner` for details. Values, variances and
        the per-bin ``spectra_tof`` are exact either way.
    rebin_by_spatial : Optional[int | tuple[int, int]]
        Whether to apply spatial rebinning. If an integer is provided, it is used as the
        rebinning factor for both spatial axes. A ``(x, y)`` tuple selects per-axis rebinning
        factors. If None, no spatial rebinning is applied.
    detector_shape : tuple[int, int]
        Shape of the TPX3 detector (default: (514, 514))
    event_id_offset : int
        Offset to apply when unrolling event_id to x, y coordinates.
        This accounts for any non-zero starting point in the event_ids.
    bank_name : str
        Name of the detector bank in the NeXus file to load (default: "bank100")
    flight_path : sc.Variable
        Source-to-detector flight path used for both energy/wavelength binning and the
        TOF→energy/wavelength coordinate labeling. Defaults to ``VENUS_FLIGHT_PATH_M`` (25 m);
        set it per detector/sample position (the VENUS L2 varies ~24.5–25.5 m).

    rebin_reduction : {"mean", "sum", "median"}, optional
        How frames combine within each TOF bin. ``None`` (default) preserves existing behavior — a
        uniform factor **sums**, a bin list takes the **mean** — while an explicit value applies to
        either. A bin list or a mean/median reduction also attaches a ``spectra_tof`` per-bin
        mean-time coordinate.
    tiff_one_file_per_image : bool
        TIFF output only. When ``False`` (default) the stack is written as one multi-page scitiff
        file. When ``True`` each spectral image is written as its own scitiff file
        (``<stem>_00000.tiff``, ``<stem>_00001.tiff``, …, one normalization per file), which suits
        tools such as ImageJ that expect individual images. Ignored for HDF5 output.
    moving_window : MovingWindow, optional
        Replace each pixel by the average — or, with ``kind="sum"``, the total — of a box of pixels
        around it, applied to **both** stacks immediately before they are divided. Off by default.
        Sizes are given by dimension name (``MovingWindow(x=3, y=3)``) in **post-crop,
        post-spatial-rebin** pixels, so a window of 3 on a ``rebin_by_spatial=2`` stack spans 6
        detector pixels. Dead and hot pixels are excluded from the window rather than averaged into
        it. A ``k x k`` window improves per-pixel precision by ``k`` and coarsens spatial resolution
        by ``k`` while the array keeps its shape; ``docs/moving_window.md`` has the measured trade.
        Cannot be combined with ``spectrum_roi`` or ``air_roi``: both reduce over a region
        assuming its pixels are independent, which a window has just stopped being true.
    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The event path reports differently from the CCD one, because reading one NeXus file is not one
        cheap item: each file is named as it is opened and then counted in the four full-event-length
        allocations it performs, so a single huge file still shows movement. Histogramming is counted per
        event chunk with no total — the chunk count follows from a file's event count, which is not known
        until it is read. Then the run combine, the TOF rebin when one is requested, the normalization
        and the export, which is per file with ``tiff_one_file_per_image=True``. Not every operation in
        between is reported: the metadata reads, the ROI crop, the dead/hot pixel detection, the
        statistics analysis, the spatial rebin and the air-region correction are single passes that run
        between named stages. See :mod:`neunorm.utils.progress`.

    Notes
    -----
    This function writes the normalized transmission data to disk in either HDF5 or TIFF format,
    depending on the file extension of `output_path`. Metadata and detector masks are included in the output.

    Returns
    -------
    sc.DataArray
        Final normalized transmission DataArray with metadata and masks
    """
    # Accept an ROI or a bare (x0, y0, x1, y1) tuple for every ROI argument; coerce to bounds
    # tuples up front so cropping and provenance see a consistent form.
    roi, air_roi = coerce_roi_arguments(roi, air_roi)

    # One reporter for the whole run, resolved exactly once: a second resolve of `progress=True`
    # would build a second tqdm sink and a duplicate set of bars. Each stage below takes its own
    # view via `run_progress.for_stage(...)`, and the leaves it calls borrow that view, so only this
    # context manager retires the bars — on the way out of a clean run and of a failed one alike.
    with resolve_progress(progress) as run_progress:
        x_bins, y_bins = detector_shape

        # Load metadata before histogramming so the detector time offset can be applied to
        # energy/wavelength bin edges; a missing offset defaults to zero.
        # The load total counts ALLOCATIONS, not files: `load_event_nexus` reports its four
        # full-event-length allocations per file, which is where the event path peaks in memory, and one
        # of these files takes minutes. Each file is named as it is opened. Histogramming gets no total —
        # the chunk count follows from a file's event count, which is not known until it is read — and one
        # reporter serves both families, whose borrowed views share its counter cell so chunks accumulate
        # into a single monotonic count.
        #
        # The file counts go through `total_across_groups`, which returns None for an input with no
        # length rather than calling len() on it. A bare len() here made a generator or `Path.glob(...)`
        # abort the whole run — on the default `progress=False` path too — where before it simply ran.
        # `[paths]` wraps each flat sequence as a single group, since this pipeline takes one file per
        # run rather than a group per run.
        n_sample_files = total_across_groups([sample_paths])
        n_ob_files = total_across_groups([ob_paths])
        load_sample = run_progress.for_stage(
            STAGE_LOAD_SAMPLE,
            total=None if n_sample_files is None else LOAD_EVENT_NEXUS_STEPS * n_sample_files,
        )
        load_ob = run_progress.for_stage(
            STAGE_LOAD_OB, total=None if n_ob_files is None else LOAD_EVENT_NEXUS_STEPS * n_ob_files
        )
        histogram = run_progress.for_stage(STAGE_HISTOGRAM)

        samples = []
        for run_path in sample_paths:
            metadata = load_metadata(run_path)
            time_offset = metadata.get("detector_time_offset", sc.scalar(0.0, unit="us"))
            load_sample.note(Path(run_path).name)
            sample = convert_events_to_histogram(
                load_event_nexus(
                    run_path,
                    detector_bank=bank_name,
                    detector_shape=detector_shape,
                    event_id_offset=event_id_offset,
                    progress=load_sample,
                ),
                binning,
                flight_path,
                x_bins,
                y_bins,
                detector_time_offset=time_offset,
                progress=histogram,
            )
            for key, value in metadata.items():
                sample.coords[key] = value
                sample.coords.set_aligned(key, False)
            samples.append(sample)

        obs = []
        for run_path in ob_paths:
            metadata = load_metadata(run_path)
            time_offset = metadata.get("detector_time_offset", sc.scalar(0.0, unit="us"))
            load_ob.note(Path(run_path).name)
            ob = convert_events_to_histogram(
                load_event_nexus(
                    run_path,
                    detector_bank=bank_name,
                    detector_shape=detector_shape,
                    event_id_offset=event_id_offset,
                    progress=load_ob,
                ),
                binning,
                flight_path,
                x_bins,
                y_bins,
                detector_time_offset=time_offset,
                progress=histogram,
            )
            for key, value in metadata.items():
                ob.coords[key] = value
                ob.coords.set_aligned(key, False)
            obs.append(ob)

        # Combine runs if there are multiple runs
        combine = run_progress.for_stage(STAGE_COMBINE_RUNS, total=2)
        combine.note(f"combining {len(samples)} sample run(s)")
        sample = combine_runs(
            samples,
            metadata_keys_to_sum=["proton_charge", "duration"],
            metadata_check_match=["detector_time_offset", "detector"],
            normalize_by_runs=True,
        )
        combine()

        combine.note(f"combining {len(obs)} open-beam run(s)")
        ob = combine_runs(
            obs,
            metadata_keys_to_sum=["proton_charge", "duration"],
            metadata_check_match=["detector_time_offset", "detector"],
            normalize_by_runs=True,
        )
        combine()

        metadata = {
            "sample_paths": [str(run) for run in sample_paths],
            "ob_paths": [str(run) for run in ob_paths],
            "processing_timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

        return reduce_tof_stacks(
            sample,
            ob,
            output_path=output_path,
            profile=_TPX3_EVENT_PROFILE,
            metadata=metadata,
            roi=roi,
            air_roi=air_roi,
            rebin_by_tof=rebin_by_tof,
            rebin_by_spatial=rebin_by_spatial,
            rebin_reduction=rebin_reduction,
            flight_path=flight_path,
            tiff_one_file_per_image=tiff_one_file_per_image,
            spectrum_roi=spectrum_roi,
            spectrum_roi_strict=spectrum_roi_strict,
            moving_window_config=moving_window,
            run_progress=run_progress,
        )

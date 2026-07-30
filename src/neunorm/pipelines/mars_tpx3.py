"""
MARS TPX3 normalization pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.data_models.roi import (
    ROILike,
    as_region_list,
    as_region_provenance,
    as_roi_bounds,
    region_provenance,
)
from neunorm.exporters.hdf5_writer import hdf5_export_step_count, write_hdf5
from neunorm.exporters.tiff_writer import tiff_export_step_count, write_tiff_stack
from neunorm.filters.gamma_filter import GAMMA_FILTER_STEPS, apply_gamma_filter
from neunorm.loaders.event_loader import LOAD_EVENT_NEXUS_STEPS, load_event_nexus
from neunorm.processing.normalizer import BackgroundROILike, normalize_step_count, normalize_transmission
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.event_converter import convert_events_to_2d_histogram
from neunorm.tof.pixel_detector import detect_dead_pixels, detect_hot_pixels
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_EXPORT,
    STAGE_GAMMA_FILTER,
    STAGE_HISTOGRAM,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    Progress,
    resolve_progress,
    total_across_groups,
)


def run_mars_tpx3_pipeline(  # noqa: C901
    sample_paths: Sequence[Sequence[str | Path]],
    ob_paths: Sequence[Sequence[str | Path]],
    output_path: Path,
    roi: Optional[ROILike] = None,
    gamma_filter: bool = True,
    detector_shape: tuple[int, int] = (514, 514),
    background_roi: Optional[BackgroundROILike] = None,
    *,
    progress: Progress = False,
) -> sc.DataArray:
    """Execute MARS TPX3 normalization pipeline.

    Pipeline Step
    1. Load event data
    2. Convert events to 2D histogram
    3. Run combine (optional)
    4. ROI clip (optional)
    5. Dead pixel detection
    6. Hot pixel detection
    7. Gamma filtering
    8. Normalization
    9. Output

    Parameters
    ----------
    sample_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to sample HDF5 files.
        Each inner list corresponds to one run and will be combined before processing.
    ob_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to open beam HDF5 files
        Each inner list corresponds to one run and will be combined before processing.
    output_path : Path
        Path to save the output file (HDF5 or TIFF)
    roi : Optional[tuple]
        Region of interest to crop to — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple.
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    detector_shape : tuple[int, int]
        Shape of the TPX3 detector (default: (514, 514))
    background_roi : ROI, MaskROI, tuple, or a sequence of them
        Sample-free background region(s) — one region or a pooled sequence (rectangles and/or
        arbitrary-shape ``MaskROI`` selections; see ``normalize_transmission``) — for flux-proxy
        normalization when proton charge is unavailable. Mutually exclusive with proton-charge
        correction. If ``roi`` is also given the detector is cropped first, so ``background_roi``
        indices are resolved in the post-crop frame.
    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The event path reports differently from the CCD one, because reading one NeXus file is not one
        cheap item: each file is named as it is opened and then counted in the four full-event-length
        allocations it performs, so a single huge file still shows movement. Histogramming is counted per
        event chunk with no total — the chunk count is only known once a file's event count is read.
        Then the run combine, the gamma filter, the normalization and the export. Not every operation in
        between is reported: the ROI crop, the open-beam averaging and the dead/hot pixel detection are
        single whole-array passes that run between named stages. See :mod:`neunorm.utils.progress`.

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
    if roi is not None:
        roi = as_roi_bounds(roi)
    if background_roi is not None:
        background_roi = as_region_list(background_roi, arg_name="background_roi")

    # One reporter for the whole run, resolved exactly once: a second resolve of `progress=True`
    # would build a second tqdm sink and a duplicate set of bars. Each stage below takes its own
    # view via `run_progress.for_stage(...)`, and the leaves it calls borrow that view, so only this
    # context manager retires the bars — on the way out of a clean run and of a failed one alike.
    with resolve_progress(progress) as run_progress:
        # Load data and convert to histogram. Written as loops rather than comprehensions so each file
        # can be named as it is opened: on the event path one file is not one cheap item, and a run of
        # 40-million-event files spends minutes per file.
        #
        # The load total counts ALLOCATIONS, not files — `load_event_nexus` reports its four
        # full-event-length allocations per file, which is where the event path peaks in memory — so the
        # stage total is that constant times the file count. Histogramming gets no total: the chunk count
        # follows from each file's event count, which is not known until the file is read.
        n_sample_files = total_across_groups(sample_paths)
        n_ob_files = total_across_groups(ob_paths)
        load_sample = run_progress.for_stage(
            STAGE_LOAD_SAMPLE,
            total=None if n_sample_files is None else LOAD_EVENT_NEXUS_STEPS * n_sample_files,
        )
        load_ob = run_progress.for_stage(
            STAGE_LOAD_OB, total=None if n_ob_files is None else LOAD_EVENT_NEXUS_STEPS * n_ob_files
        )
        # One histogram reporter for both families: a borrowed view shares its counter cell, so chunks
        # accumulate into a single monotonic count across every file of the run.
        histogram = run_progress.for_stage(STAGE_HISTOGRAM)

        def _load_runs(path_groups, load_report):
            """Read each file, histogram it, and concatenate one image stack per run."""
            stacks = []
            for group in path_groups:
                frames = []
                for path in group:
                    load_report.note(Path(path).name)
                    events = load_event_nexus(path, detector_shape=detector_shape, progress=load_report)
                    frames.append(convert_events_to_2d_histogram(events, detector_shape, progress=histogram))
                stacks.append(sc.concat(frames, dim="N_image"))
            return stacks

        samples = _load_runs(sample_paths, load_sample)
        obs = _load_runs(ob_paths, load_ob)

        combine = run_progress.for_stage(STAGE_COMBINE_RUNS, total=2)

        # Combine runs if there are multiple runs
        combine.note(f"combining {len(samples)} sample run(s)")
        sample = combine_runs(samples, metadata_keys_to_sum=[], metadata_check_match=[], normalize_by_runs=True)
        combine()
        combine.note(f"combining {len(obs)} open-beam run(s)")
        ob = combine_runs(obs, metadata_keys_to_sum=[], metadata_check_match=[], normalize_by_runs=True)
        combine()

        # Apply ROI if specified
        if roi:
            sample = apply_roi(sample, roi)
            ob = apply_roi(ob, roi)

        # Average OB
        ob = prepare_reference(ob, dim="N_image")

        # Dead pixel detection
        sample.masks["dead_pixels"] = detect_dead_pixels(sample)

        # Hot pixel detection
        sample.masks["hot_pixels"] = detect_hot_pixels(sample)

        # Gamma filtering (optional)
        if gamma_filter:
            sample = apply_gamma_filter(
                sample, progress=run_progress.for_stage(STAGE_GAMMA_FILTER, total=GAMMA_FILTER_STEPS)
            )

        # Normalization (background_roi flux proxy when provided)
        transmission = normalize_transmission(
            sample,
            ob,
            background_roi=background_roi,
            progress=run_progress.for_stage(STAGE_NORMALIZE, total=normalize_step_count(background_roi)),
        )

        # Write output
        metadata = {
            "sample_paths": [[str(p) for p in run] for run in sample_paths],
            "ob_paths": [[str(p) for p in run] for run in ob_paths],
            "gamma_filter_applied": gamma_filter,
            "processing_timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

        if roi:
            metadata["roi_applied"] = region_provenance(roi)

        if background_roi is not None:
            metadata["background_roi"] = as_region_provenance(background_roi)

        if output_path.suffix.lower() in (".hdf5", ".h5"):
            write_hdf5(
                output_path,
                transmission,
                dead_pixel_mask="dead_pixels",
                hot_pixel_mask="hot_pixels",
                metadata=metadata,
                progress=run_progress.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(transmission, metadata)),
            )
        elif output_path.suffix.lower() in (".tiff", ".tif"):
            rename_map = {}
            if "N_image" in transmission.dims:
                rename_map["N_image"] = "z"  # TIFF stacks typically use 'z' for the stack dimension
            if rename_map:
                transmission = transmission.rename_dims(rename_map)

            daqmetadata = {
                "facility": "HFIR",
                "instrument": "MARS",
                "detector_type": "TPX3",
                "source_type": "neutron",
            }

            # Combine all masks and broadcast to the shape of the transmission data.
            # Mask must be same shape as the image data for scitiff.
            if transmission.masks:
                combined_mask = np.zeros_like(transmission.values, dtype=bool)
                for mask in transmission.masks.values():
                    combined_mask |= mask.values

                # remove other masks
                transmission.masks.clear()
                # add combined mask back in with name "scitiff-mask"
                transmission.masks["scitiff-mask"] = sc.array(dims=transmission.dims, values=combined_mask, dtype=bool)

            write_tiff_stack(
                output_path,
                transmission,
                metadata=metadata,
                daqmetadata=daqmetadata,
                progress=run_progress.for_stage(STAGE_EXPORT, total=tiff_export_step_count(transmission)),
            )
        else:
            raise ValueError(f"Unsupported output file format: {output_path.suffix}")

        logger.success("MARS TPX3 pipeline completed successfully. Output written to {}", output_path)
        return transmission

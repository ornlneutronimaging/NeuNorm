"""
MARS CCD/CMOS normalization pipeline.
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
from neunorm.loaders.stack_loader import load_stack
from neunorm.processing.normalizer import (
    BackgroundROILike,
    normalize_step_count,
    normalize_transmission,
    normalize_with_dark,
    normalize_with_dark_step_count,
)
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.pixel_detector import detect_dead_pixels
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_EXPORT,
    STAGE_GAMMA_FILTER,
    STAGE_LOAD_DARK,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    Progress,
    resolve_progress,
    total_across_groups,
)


def run_mars_ccd_pipeline(  # noqa: C901
    sample_paths: Sequence[Sequence[str | Path]],
    ob_paths: Sequence[Sequence[str | Path]],
    dark_paths: Optional[Sequence[Sequence[str | Path]]] = None,
    output_path: Optional[Path] = None,
    roi: Optional[ROILike] = None,
    gamma_filter: bool = True,
    background_roi: Optional[BackgroundROILike] = None,
    metadata_match_atol: float = 0.0,
    *,
    progress: Progress = False,
) -> sc.DataArray:
    """Execute MARS CCD/CMOS normalization pipeline.

    Pipeline Steps (10 total)
    - Load TIFF/FITS (sample, OB, dark [optional])
    - Run combine (optional)
    - ROI clip (optional)
    - Average dark (optional) / OB
    - Dead pixel detection (existing tof/pixel_detector.py)
    - Gamma filtering (filters/gamma_filter.py)
    - Dark correction (optional, processing/dark_corrector.py)
    - Normalization (existing processing/normalizer.py)
    - Output (exporters/hdf5_writer.py, exporters/tiff_writer.py)

    Parameters
    ----------
    sample_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to sample TIFF or FITS files.
        Each inner list represents a run that should be combined before processing.
    ob_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to open beam TIFF or FITS files.
        Each inner list represents a run that should be combined before processing.
    dark_paths : Optional[Sequence[Sequence[str | Path]]]
        List of lists of paths to dark current TIFF or FITS files.
        Each inner list represents a run that should be combined before processing.
        Optional (default: None). If omitted (None or an empty list), dark
        correction is skipped and the dark-frame variance does not contribute to
        the propagated uncertainty.
    output_path : Optional[Path]
        Path to save the output file (HDF5 or TIFF). Required; a value of None
        raises ``ValueError`` (the default exists only so ``dark_paths`` can keep
        its positional slot).
    roi : Optional[tuple]
        Region of interest to crop to — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple.
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    background_roi : ROI, MaskROI, tuple, or a sequence of them
        Sample-free background region(s) — one region or a pooled sequence (rectangles and/or
        arbitrary-shape ``MaskROI`` selections; see ``normalize_transmission``) — for flux-proxy
        normalization when proton charge is unavailable. Mutually exclusive with proton-charge
        correction. If ``roi`` is also given the detector is cropped first, so ``background_roi``
        indices are resolved in the post-crop frame.
    metadata_match_atol : float, optional
        Absolute tolerance for the metadata match check when combining runs
        (exposure time and MotSlit aperture readback positions), by default
        0.0 (exact match). Detector names always require an exact match.
    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The stages reported are the sample, open-beam and dark loads — **one event per file**, counted
        across all input runs rather than restarting per run — then the run combine, the gamma filter,
        the normalization and the export. Not every operation in between is reported: the ROI crop,
        the dark/open-beam averaging and the dead-pixel detection are single whole-array passes that
        run between named stages. See :mod:`neunorm.utils.progress`.

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

    if output_path is None:
        raise ValueError("output_path is required")

    # One reporter for the whole run, resolved exactly once: a second resolve of `progress=True`
    # would build a second tqdm sink and a duplicate set of bars. Each stage below takes its own
    # view via `run.for_stage(...)`, and the leaves it calls borrow that view, so only this
    # context manager retires the bars — on the way out of a clean run and of a failed one alike.
    with resolve_progress(progress) as run:
        # Load data. One reporter per input family, reused for every run in it: a borrowed view shares
        # its counter cell, so N calls accumulate into one count across the whole run instead of
        # restarting per run. The `stage` argument of the leaf is not used here — a handed-down reporter
        # carries its own label, and passing one would be silently ignored.
        load_sample = run.for_stage(STAGE_LOAD_SAMPLE, total=total_across_groups(sample_paths))
        samples = [load_stack(paths, progress=load_sample) for paths in sample_paths]
        load_ob = run.for_stage(STAGE_LOAD_OB, total=total_across_groups(ob_paths))
        ob = [load_stack(paths, progress=load_ob) for paths in ob_paths]

        # Combining runs is the largest operation here that no instrumented leaf covers: it copies the
        # first run's values and variances, then adds each further run in place. Reported as named steps
        # so a multi-run job does not go silent between the loads and the normalization.
        combine = run.for_stage(STAGE_COMBINE_RUNS, total=3 if dark_paths else 2)

        # Before combining, check that all sample runs have the same shape and some metadata keys match
        # Keys to check [ManufacturerStr, MotSlitVB.RBV, MotSlitVT.RBV, MotSlitHR.RBV, MotSlitHL.RBV].
        # MotSlit does not need to match for dark. ExposureTime is included in metadata checks and is
        # effectively averaged/normalized across runs (not summed) when normalize_by_runs=True.

        combine.note(f"combining {len(samples)} sample run(s)")
        sample = combine_runs(
            samples,
            metadata_keys_to_sum=("ExposureTime",),
            metadata_check_match=[
                "ExposureTime",
                "ManufacturerStr",
                "MotSlitVB.RBV",
                "MotSlitVT.RBV",
                "MotSlitHR.RBV",
                "MotSlitHL.RBV",
            ],
            normalize_by_runs=True,
            metadata_match_atol=metadata_match_atol,
        )
        combine()

        combine.note(f"combining {len(ob)} open-beam run(s)")
        ob = combine_runs(
            ob,
            metadata_keys_to_sum=("ExposureTime",),
            metadata_check_match=[
                "ExposureTime",
                "ManufacturerStr",
                "MotSlitVB.RBV",
                "MotSlitVT.RBV",
                "MotSlitHR.RBV",
                "MotSlitHL.RBV",
            ],
            normalize_by_runs=True,
            metadata_match_atol=metadata_match_atol,
        )
        combine()

        # Dark current is optional: only load/combine it when dark paths are provided.
        dark = None
        if dark_paths:
            load_dark = run.for_stage(STAGE_LOAD_DARK, total=total_across_groups(dark_paths))
            dark_runs = [load_stack(paths, progress=load_dark) for paths in dark_paths]
            combine.note(f"combining {len(dark_runs)} dark run(s)")
            dark = combine_runs(
                dark_runs,
                metadata_keys_to_sum=("ExposureTime",),
                metadata_check_match=[
                    "ExposureTime",
                    "ManufacturerStr",
                ],
                normalize_by_runs=True,
                metadata_match_atol=metadata_match_atol,
            )
            combine()

        # Apply ROI if specified
        if roi:
            sample = apply_roi(sample, roi)
            ob = apply_roi(ob, roi)
            if dark is not None:
                dark = apply_roi(dark, roi)

        # Average dark and OB
        if dark is not None:
            dark = prepare_reference(dark, dim="N_image")
        ob = prepare_reference(ob, dim="N_image")

        # Dead pixel detection
        sample.masks["dead_pixels"] = detect_dead_pixels(sample)

        # Gamma filtering (optional). The step count comes from the filter's own constant: a
        # handed-down reporter keeps the total its caller bound, so a literal here could drift.
        if gamma_filter:
            sample = apply_gamma_filter(sample, progress=run.for_stage(STAGE_GAMMA_FILTER, total=GAMMA_FILTER_STEPS))

        # Dark correction (optional) + normalization. With a shared dark frame, normalize_with_dark
        # subtracts the dark and normalizes in one step so the dark variance is not double-counted
        # in the transmission uncertainty. Without dark, normalize directly.
        # Each branch declares its own normalization total from the same helpers the normalizers use,
        # so the bar reaches its end whichever correction runs. MARS passes no proton charge.
        if background_roi is not None:
            # Flux-proxy normalization from a sample-free ROI, in place of proton charge.
            # With a shared dark, route through normalize_with_dark so the shared-dark variance
            # double-count is corrected (k = co/cs); without dark, normalize directly.
            if dark is not None:
                transmission = normalize_with_dark(
                    sample,
                    ob,
                    dark,
                    background_roi=background_roi,
                    progress=run.for_stage(STAGE_NORMALIZE, total=normalize_with_dark_step_count(background_roi)),
                )
            else:
                transmission = normalize_transmission(
                    sample,
                    ob,
                    background_roi=background_roi,
                    progress=run.for_stage(STAGE_NORMALIZE, total=normalize_step_count(background_roi)),
                )
        elif dark is not None:
            transmission = normalize_with_dark(
                sample,
                ob,
                dark,
                progress=run.for_stage(STAGE_NORMALIZE, total=normalize_with_dark_step_count()),
            )
        else:
            logger.info("No dark current provided; skipping dark correction")
            transmission = normalize_transmission(
                sample, ob, progress=run.for_stage(STAGE_NORMALIZE, total=normalize_step_count())
            )

        # Guarantee a float32 normalized data product, regardless of any
        # intermediate dtype promotion. .astype converts values and variances. MARS has
        # no proton-charge division, so this is already float32; the cast keeps the two
        # CCD pipelines symmetric and is robust to future changes.
        transmission = transmission.astype("float32")

        # Write output
        metadata = {
            "sample_paths": [[str(p) for p in run] for run in sample_paths],
            "ob_paths": [[str(p) for p in run] for run in ob_paths],
            "gamma_filter_applied": gamma_filter,
            "dark_correction_applied": dark is not None,
            "processing_timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

        # Only record dark_paths when dark correction was actually applied.
        if dark_paths:
            metadata["dark_paths"] = [[str(p) for p in run] for run in dark_paths]

        if roi:
            metadata["roi_applied"] = region_provenance(roi)

        if background_roi is not None:
            metadata["background_roi"] = as_region_provenance(background_roi)

        if output_path.suffix.lower() in (".hdf5", ".h5"):
            write_hdf5(
                output_path,
                transmission,
                dead_pixel_mask="dead_pixels",
                metadata=metadata,
                progress=run.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(transmission, metadata)),
            )
        elif output_path.suffix.lower() in (".tiff", ".tif"):
            rename_map = {}
            if "N_image" in transmission.dims:
                rename_map["N_image"] = "z"  # TIFF stacks typically use 'z' for the stack dimension
            if rename_map:
                transmission = transmission.rename_dims(rename_map)

            model = "Unknown"
            if "ManufacturerStr" in sample.coords:
                model = sample.coords["ManufacturerStr"].value
            elif "ModelStr" in sample.coords:
                model = sample.coords["ModelStr"].value
            elif "Model" in sample.coords:
                model = sample.coords["Model"].value

            daqmetadata = {
                "facility": "HFIR",
                "instrument": "MARS",
                "detector_type": model,
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
                progress=run.for_stage(STAGE_EXPORT, total=tiff_export_step_count(transmission)),
            )
        else:
            raise ValueError(f"Unsupported output file format: {output_path.suffix}")

        logger.success("MARS CCD pipeline completed successfully. Output written to {}", output_path)
        return transmission

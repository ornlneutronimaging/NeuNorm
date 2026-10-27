"""
VENUS CCD/CMOS normalization pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.data_models.roi import (
    MaskROI,
    RegionLike,
    ROILike,
    as_region_list,
    as_region_provenance,
    as_roi_bounds,
    region_provenance,
)
from neunorm.exporters.hdf5_writer import hdf5_export_step_count, write_hdf5
from neunorm.exporters.tiff_writer import tiff_export_step_count, write_tiff_stack
from neunorm.filters.gamma_filter import GAMMA_FILTER_STEPS, apply_gamma_filter
from neunorm.loaders.metadata_loader import PROTON_CHARGE_TAG, find_nexus_file, load_proton_charge_coulombs
from neunorm.pipelines._output_path import HDF5_SUFFIXES, TIFF_SUFFIXES, resolve_output_path, unsupported_suffix_error
from neunorm.pipelines._run_loading import FrameSize, combine_owned_runs, load_runs
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import (
    BackgroundROILike,
    normalize_step_count,
    normalize_transmission,
    normalize_with_dark,
    normalize_with_dark_step_count,
)
from neunorm.processing.reference_preparer import prepare_reference
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


def _proton_charge_from_nexus(
    runs: Sequence[sc.DataArray],
    groups: Sequence[Sequence[str | Path]],
    *,
    family: str,
    nexus_dir: Optional[str | Path],
) -> list[list[Optional[str]]]:
    """Give every run of one input family its ``IntegratedPCharge`` coordinate.

    The loader publishes the coordinate when every frame of the run carries the TIFF tag (the
    iKon-XL files do); a run without it gets one value per frame from the NeXus file of that frame's
    run, located by :func:`~neunorm.loaders.metadata_loader.find_nexus_file`. The coordinate is the
    same either way — dimensionless coulombs along the frame dimension, unaligned — so everything
    downstream is indifferent to the source.

    Returns, per run, the NeXus file read for each frame, or ``None`` for a frame whose tag was used:
    the provenance recorded in the output metadata.

    Raises
    ------
    ValueError
        A frame carries no tag and no NeXus file was found for it. The message names the frame,
        where the file was looked for, and the two ways out (``nexus_dir``, ``background_roi``).
    """
    provenance: list[list[Optional[str]]] = []
    for run, paths in zip(runs, groups, strict=True):
        if PROTON_CHARGE_TAG in run.coords:
            provenance.append([None] * len(paths))
            continue
        values: list[float] = []
        used: list[Optional[str]] = []
        for path in paths:
            nexus = find_nexus_file(path, nexus_dir=nexus_dir)
            if nexus is None:
                looked = (
                    str(Path(nexus_dir))
                    if nexus_dir is not None
                    else "the nexus/ folder of the image's IPTS-<n> directory"
                )
                raise ValueError(
                    f"{family} frame {Path(path).name} carries no {PROTON_CHARGE_TAG} TIFF tag and no NeXus file "
                    f"*_<run>.nxs.h5 of its run was found in {looked}. Pass nexus_dir= to point at the NeXus files "
                    "of the experiment, or background_roi= to normalize by a sample-free region instead of the "
                    "proton charge."
                )
            values.append(load_proton_charge_coulombs(nexus))
            used.append(str(nexus))
        dim = run.dims[0]
        run.coords[PROTON_CHARGE_TAG] = sc.array(dims=[dim], values=values)
        run.coords.set_aligned(PROTON_CHARGE_TAG, False)
        logger.info(
            "{}: no {} tag in the {} frame(s) — proton charge read from the NeXus files ({} …)",
            family,
            PROTON_CHARGE_TAG,
            len(paths),
            used[0],
        )
        provenance.append(used)
    return provenance


def _proton_charge_source(provenances: Sequence[Sequence[Sequence[Optional[str]]]]) -> str:
    """``"tiff"``, ``"nexus"`` or ``"mixed"``: where the proton charges of the sample and OB came from."""
    flags = {nexus is not None for family in provenances for run in family for nexus in run}
    if flags == {True}:
        return "nexus"
    if flags == {False}:
        return "tiff"
    return "mixed"


def run_venus_ccd_pipeline(  # noqa: C901
    sample_paths: Sequence[Sequence[str | Path]],
    ob_paths: Sequence[Sequence[str | Path]],
    dark_paths: Optional[Sequence[Sequence[str | Path]]] = None,
    output_path: Optional[str | Path] = None,
    roi: Optional[ROILike] = None,
    gamma_filter: bool = True,
    air_roi: Optional[RegionLike] = None,
    background_roi: Optional[BackgroundROILike] = None,
    *,
    nexus_dir: Optional[str | Path] = None,
    progress: Progress = False,
) -> sc.DataArray:
    """Execute VENUS CCD/CMOS normalization pipeline.

    Pipeline Steps (12 total)
    - Load TIFF/FITS (sample, OB, dark [optional]), cropping each frame to the ROI (optional)
    - Load p_charge metadata: the ``IntegratedPCharge`` TIFF tag, else the NeXus file of each run
    - Run combine (critical for VENUS)
    - Average dark (optional) / OB
    - Dead pixel detection
    - Gamma filtering (optional, less critical than MARS)
    - Dark correction (optional)
    - p_charge beam correction
    - Normalization
    - Air region correction (optional)
    - Error propagation (includes p_charge uncertainty)
    - Output

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
    output_path : Optional[str | Path]
        Path to save the output file: ``.hdf5``/``.h5`` for HDF5 or ``.tiff``/``.tif`` for TIFF.
        Required; a value of None raises ``ValueError`` (the default exists only so ``dark_paths``
        can keep its positional slot). Any other suffix raises ``ValueError`` before any input is read.
    roi : Optional[tuple]
        Region of interest to crop to — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple. Each sample,
        open-beam and dark frame is cropped as it is read, so memory scales with the region and the
        number of images rather than with the detector size; the output, including the ``x`` and ``y``
        coordinates, is the same as cropping after the load. Sizes are checked on the uncropped frames,
        with or without an ROI: sample, open-beam and dark frames must be the same size, though their
        number may differ, and the runs of one family must have the same number of frames of that size.
        A mismatch raises ``ValueError`` while the mismatched family loads, before any later family is read.
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    air_roi : ROI, MaskROI, or tuple, optional
        Region of interest for air correction — an ``ROI``, a bare ``(x0, y0, x1, y1)`` tuple, or an
        arbitrary-shape ``MaskROI`` selection. If None, air correction is not applied.
    background_roi : ROI, MaskROI, tuple, or a sequence of them
        Sample-free background region(s) — one region or a pooled sequence (rectangles and/or
        arbitrary-shape ``MaskROI`` selections; see ``normalize_transmission``) — for flux-proxy
        normalization when proton charge is unavailable. Mutually exclusive with proton-charge
        correction. If ``roi`` is also given the detector is cropped first, so ``background_roi``
        indices are resolved in the post-crop frame.
    nexus_dir : str or Path, optional
        Where to find the NeXus files of the runs when a frame carries no ``IntegratedPCharge`` TIFF
        tag. The proton charge is read from the tag when the DAQ wrote it (iKon-XL frames) and
        otherwise from ``entry/proton_charge`` of the NeXus file of the frame's run (QHY411 frames),
        ``*_<run>.nxs.h5`` with the run number taken from the ``_Run_<n>`` token of the file name. By
        default that file is looked for in the ``nexus`` folder of the experiment the image sits in
        (``/SNS/VENUS/IPTS-<m>/nexus``); ``nexus_dir`` replaces that folder for images kept elsewhere.
        A frame with neither source is an error, unless ``background_roi`` is used — then no proton
        charge is read at all. The dark's proton charge is not used by the normalization and is never
        required. The source is recorded in the output metadata (``proton_charge_source``:
        ``"tiff"``, ``"nexus"`` or ``"mixed"``, and the NeXus file of every frame read that way).

    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The stages reported are the sample, open-beam and dark loads — **one event per file**, counted
        across all input runs rather than restarting per run — then the run combine, the gamma filter,
        the normalization and the export. The ROI crop happens inside the loads. Not every operation in
        between is reported: the dark/open-beam averaging, the dead-pixel detection and the air-region
        correction are single whole-array passes that run between named stages. See
        :mod:`neunorm.utils.progress`.

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
    if air_roi is not None:
        air_roi = air_roi if isinstance(air_roi, MaskROI) else as_roi_bounds(air_roi)
    if background_roi is not None:
        background_roi = as_region_list(background_roi, arg_name="background_roi")

    output_path = resolve_output_path(output_path)

    # One reporter for the whole run, resolved exactly once: a second resolve of `progress=True`
    # would build a second tqdm sink and a duplicate set of bars. Each stage below takes its own
    # view via `run_progress.for_stage(...)`, and the leaves it calls borrow that view, so only this
    # context manager retires the bars — on the way out of a clean run and of a failed one alike.
    with resolve_progress(progress) as run_progress:
        # Load data. One reporter per input family, reused for every run in it: a borrowed view shares
        # its counter cell, so N calls accumulate into one count across the whole run instead of
        # restarting per run.
        load_sample = run_progress.for_stage(STAGE_LOAD_SAMPLE, total=total_across_groups(sample_paths))
        frame_size = FrameSize()
        samples = load_runs(sample_paths, roi=roi, progress=load_sample, family="sample", frame_size=frame_size)
        load_ob = run_progress.for_stage(STAGE_LOAD_OB, total=total_across_groups(ob_paths))
        ob = load_runs(ob_paths, roi=roi, progress=load_ob, family="open-beam", frame_size=frame_size)

        # Proton charge: the TIFF tag when the DAQ wrote it, else the NeXus file of each run. Not
        # needed at all when a background region stands in for it.
        proton_charge_provenance: dict[str, list[list[Optional[str]]]] = {}
        if background_roi is None:
            proton_charge_provenance["sample"] = _proton_charge_from_nexus(
                samples, sample_paths, family="sample", nexus_dir=nexus_dir
            )
            proton_charge_provenance["ob"] = _proton_charge_from_nexus(
                ob, ob_paths, family="open-beam", nexus_dir=nexus_dir
            )

        # Combining runs is the largest operation here that no instrumented leaf covers, and VENUS
        # relies on it, so it is reported as named steps rather than left silent.
        combine = run_progress.for_stage(STAGE_COMBINE_RUNS, total=3 if dark_paths else 2)

        # Before combining, check that all sample runs have the same shape and some metadata keys match
        # Keys to check ManufacturerStr. IntegratedPCharge is included in metadata checks and is
        # effectively averaged/normalized across runs (not summed) when normalize_by_runs=True.

        # When background_roi is used (proton-charge proxy), don't require/aggregate IntegratedPCharge.
        pc_keys = () if background_roi is not None else ("IntegratedPCharge",)

        combine.note(f"combining {len(samples)} sample run(s)")
        sample = combine_owned_runs(
            samples,
            metadata_keys_to_sum=pc_keys,
            metadata_check_match=["ManufacturerStr"],
            normalize_by_runs=True,
        )
        combine()
        # Release the per-run stacks once combined rather than holding them to the end of the run.
        del samples

        combine.note(f"combining {len(ob)} open-beam run(s)")
        ob = combine_owned_runs(
            ob,
            metadata_keys_to_sum=pc_keys,
            metadata_check_match=["ManufacturerStr"],
            normalize_by_runs=True,
        )
        combine()

        # Dark current is optional: only load/combine it when dark paths are provided.
        dark = None
        if dark_paths:
            load_dark = run_progress.for_stage(STAGE_LOAD_DARK, total=total_across_groups(dark_paths))
            dark_runs = load_runs(dark_paths, roi=roi, progress=load_dark, family="dark", frame_size=frame_size)
            combine.note(f"combining {len(dark_runs)} dark run(s)")
            # The dark's proton charge plays no part in the normalization (only the sample's and the
            # open beam's do), so it is neither required nor aggregated.
            dark = combine_owned_runs(
                dark_runs,
                metadata_keys_to_sum=(),
                metadata_check_match=["ManufacturerStr"],
                normalize_by_runs=True,
            )
            combine()
            del dark_runs

        # Average dark and OB
        if dark is not None:
            dark = prepare_reference(dark, dim="N_image")
        ob = prepare_reference(ob, dim="N_image")

        # Dead pixel detection
        sample.masks["dead_pixels"] = detect_dead_pixels(sample)

        # Gamma filtering (optional). The step count comes from the filter's own constant: a
        # handed-down reporter keeps the total its caller bound, so a literal here could drift.
        if gamma_filter:
            sample = apply_gamma_filter(
                sample, progress=run_progress.for_stage(STAGE_GAMMA_FILTER, total=GAMMA_FILTER_STEPS)
            )

        # Dark correction (optional) + normalization. The proton-charge coords are cast to float32
        # so the division does not silently re-promote the float32 image data to float64 (the coord
        # is float64 because metadata is parsed via float()). With a shared dark
        # frame, normalize_with_dark subtracts the dark and normalizes in one step so the dark
        # variance is not double-counted in the transmission uncertainty.
        if background_roi is not None:
            # Flux-proxy normalization from a sample-free ROI, replacing the proton-charge
            # correction. With a shared dark, route through normalize_with_dark so the shared-dark
            # variance double-count is corrected (k = co/cs).
            if dark is not None:
                transmission = normalize_with_dark(
                    sample,
                    ob,
                    dark,
                    background_roi=background_roi,
                    progress=run_progress.for_stage(
                        STAGE_NORMALIZE, total=normalize_with_dark_step_count(background_roi)
                    ),
                )
            else:
                transmission = normalize_transmission(
                    sample,
                    ob,
                    background_roi=background_roi,
                    progress=run_progress.for_stage(STAGE_NORMALIZE, total=normalize_step_count(background_roi)),
                )
            # IntegratedPCharge was neither used nor aggregated in this mode (pc_keys=()), so the
            # combined array still carries the first run's loaded value as an unaligned coord. Drop it
            # so a stale, never-aggregated proton charge does not reach the output coords/provenance.
            if "IntegratedPCharge" in transmission.coords:
                del transmission.coords["IntegratedPCharge"]
        else:
            proton_charge_sample = sample.coords["IntegratedPCharge"].astype("float32")
            proton_charge_ob = ob.coords["IntegratedPCharge"].astype("float32")
            if dark is not None:
                transmission = normalize_with_dark(
                    sample,
                    ob,
                    dark,
                    proton_charge_sample=proton_charge_sample,
                    proton_charge_ob=proton_charge_ob,
                    progress=run_progress.for_stage(
                        STAGE_NORMALIZE,
                        total=normalize_with_dark_step_count(proton_charge_sample=proton_charge_sample),
                    ),
                )
            else:
                logger.info("No dark current provided; skipping dark correction")
                transmission = normalize_transmission(
                    sample=sample,
                    ob=ob,
                    proton_charge_sample=proton_charge_sample,
                    proton_charge_ob=proton_charge_ob,
                    progress=run_progress.for_stage(
                        STAGE_NORMALIZE,
                        total=normalize_step_count(proton_charge_sample=proton_charge_sample),
                    ),
                )

        # Air region correction (optional)
        if air_roi is not None:
            transmission = apply_air_region_correction(transmission, air_roi)

        # Guarantee a float32 normalized data product, regardless of any
        # intermediate dtype promotion. .astype converts values and variances.
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

        if proton_charge_provenance:
            metadata["proton_charge_source"] = _proton_charge_source(list(proton_charge_provenance.values()))
            metadata["proton_charge_nexus_sample"] = proton_charge_provenance["sample"]
            metadata["proton_charge_nexus_ob"] = proton_charge_provenance["ob"]

        if roi:
            metadata["roi_applied"] = region_provenance(roi)

        if air_roi is not None:
            metadata["air_roi"] = region_provenance(air_roi)

        if background_roi is not None:
            metadata["background_roi"] = as_region_provenance(background_roi)

        if output_path.suffix.lower() in HDF5_SUFFIXES:
            write_hdf5(
                output_path,
                transmission,
                dead_pixel_mask="dead_pixels",
                metadata=metadata,
                progress=run_progress.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(transmission, metadata)),
            )
        elif output_path.suffix.lower() in TIFF_SUFFIXES:
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
                "facility": "SNS",
                "instrument": "VENUS",
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
                progress=run_progress.for_stage(STAGE_EXPORT, total=tiff_export_step_count(transmission)),
            )
        else:
            raise unsupported_suffix_error(output_path)

        logger.success("VENUS CCD pipeline completed successfully. Output written to {}", output_path)
        return transmission

"""
VENUS TPX1 pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.data_models.roi import (
    MaskROI,
    RegionLike,
    ROILike,
    as_roi_bounds,
    region_provenance,
)
from neunorm.exporters.hdf5_writer import hdf5_export_step_count, write_hdf5
from neunorm.exporters.tiff_writer import tiff_export_step_count, write_tiff_stack
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_step_count, normalize_transmission
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.processing.spatial_rebinner import rebin_spatial
from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength
from neunorm.tof.histogram_rebinner import rebin_tof
from neunorm.tof.pixel_detector import detect_dead_pixels
from neunorm.tof.statistics_analyzer import analyze_statistics
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_EXPORT,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    STAGE_REBIN_TOF,
    Progress,
    resolve_progress,
    total_across_groups,
)


def _tof_bin_edges_from_left_edges(spectra_tof: sc.Variable) -> sc.Variable:
    """Build N+1 TOF bin edges from the N per-frame LEFT edges.

    The VENUS TPX1 ``*_Spectra.txt`` ``shutter_time`` column gives the LEFT (opening) edge of each
    frame's TOF bin — one value per image, i.e. N left edges for N frames. scipp histograms need
    N+1 bin edges, so the closing edge (right edge of the last bin) is appended. Its width is
    extrapolated from the last observed step, which is exact for VENUS's fixed-width TOF grid.
    This makes ``tof`` a proper bin-edge axis so ``rebin_by_tof`` works (GitHub #187).

    ``spectra_tof`` always has >= 2 rows here: ``load_spectra_tof`` rejects a single-row sidecar
    before this is reached, and real TPX1 acquisitions have thousands of frames.
    """
    values = spectra_tof.values
    closing = values[-1] + (values[-1] - values[-2])
    return sc.array(dims=["tof"], values=np.append(values, closing), unit=spectra_tof.unit)


def run_venus_tpx1_pipeline(  # noqa: C901
    sample_hdf5_paths: Sequence[str | Path],
    ob_hdf5_paths: Sequence[str | Path],
    sample_tiff_paths: Sequence[Sequence[str | Path]],
    ob_tiff_paths: Sequence[Sequence[str | Path]],
    output_path: Path,
    roi: Optional[ROILike] = None,
    air_roi: Optional[RegionLike] = None,
    rebin_by_tof: Optional[bool | int | list | tuple] = False,
    rebin_by_spatial: Optional[int | tuple[int, int]] = None,
    flight_path: sc.Variable = sc.scalar(VENUS_FLIGHT_PATH_M, unit="m"),
    *,
    rebin_reduction: Optional[Literal["mean", "sum", "median"]] = None,
    tiff_one_file_per_image: bool = False,
    progress: Progress = False,
) -> sc.DataArray:
    """Execute VENUS TPX1 normalization pipeline.

    Pipeline Steps (11 total)
    - Load TIFF stack (pre-binned histograms from auto-reduction)
    - Load TOF bin edges
    - Load metadata (including proton charge and detector time offset)
    - Run combine
    - ROI clip (optional)
    - Dead pixel detection
    - Statistics analysis + rebinning recommendation (only when ``rebin_by_tof=True``)
    - Rebinning (TOF and/or spatial, optional)
    - Beam correction (proton charge)
    - Normalization (TOF-resolved)
    - Air region correction (optional)
    - Error propagation
    - Output


    Parameters
    ----------
    sample_hdf5_paths : Sequence[str | Path]
        List of paths to sample HDF5 files containing metadata.
    ob_hdf5_paths : Sequence[str | Path]
        List of paths to open beam HDF5 files containing metadata.
    sample_tiff_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to sample TIFF files.
        Each inner list represents a run that should be combined before processing.
    ob_tiff_paths : Sequence[Sequence[str | Path]]
        List of lists of paths to open beam TIFF files.
        Each inner list represents a run that should be combined before processing.
    output_path : Path
        Path to save the output file (HDF5 or TIFF)
    roi : Optional[tuple]
        Region of interest to crop to — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple.
    air_roi : ROI, MaskROI, or tuple, optional
        Region of interest for air correction — an ``ROI``, a bare ``(x0, y0, x1, y1)`` tuple, or an
        arbitrary-shape ``MaskROI`` selection. If None, air correction is not applied.
    rebin_by_tof : bool, int, or list/tuple of [start, stop], optional
        TOF rebinning. ``True`` uses the statistics-based recommended factor; an ``int`` is a uniform
        factor (frames per bin); a ``[[start, stop], ...]`` list defines explicit half-open
        frame-index bins (variable width). One output image per range.
        Frames covered by no range are dropped silently (a deliberate, requested behavior). Note
        that dropping frames leaves the output images covering disjoint time bands, which the
        ``N+1`` bin-edge ``tof`` axis cannot describe exactly — the bin before a dropped span has its
        closing edge (and derived ``wavelength``/``energy`` edge) widened by the omitted span, and the
        result is not a continuous spectrum. Prefer contiguous ranges when the data will be analysed
        as a spectrum; see :mod:`neunorm.tof.histogram_rebinner` for details. Values, variances and
        the per-bin ``spectra_tof`` are exact either way.
    rebin_by_spatial : Optional[int | tuple[int, int]]
        Whether to apply spatial rebinning. If an integer is provided, it is used as the
        rebinning factor for both spatial axes. A ``(x, y)`` tuple selects per-axis
        rebinning factors (x and y). If None, no spatial rebinning is applied.
    flight_path : sc.Variable
        Source-to-detector flight path used for TOF→energy/wavelength coordinate labeling.
        Defaults to ``VENUS_FLIGHT_PATH_M`` (25 m); set it per detector/sample position.

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
    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The stages reported are the sample and open-beam loads — **one event per TIFF**, counted across
        all input runs rather than restarting per run — then the run combine, the TOF rebin when one is
        requested, the normalization, and the export, which is per file with
        ``tiff_one_file_per_image=True``. Not every operation in between is reported: the metadata
        reads, the ROI crop, the dead/hot pixel detection and the spatial rebin are single passes that
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
    if air_roi is not None:
        air_roi = air_roi if isinstance(air_roi, MaskROI) else as_roi_bounds(air_roi)

    # length of hdf5 paths and tiff paths should match for both sample and OB
    if len(sample_hdf5_paths) != len(sample_tiff_paths):
        raise ValueError(
            f"Number of sample HDF5 paths ({len(sample_hdf5_paths)}) does not match number of sample TIFF path groups "
            f"({len(sample_tiff_paths)})."
        )
    if len(ob_hdf5_paths) != len(ob_tiff_paths):
        raise ValueError(
            f"Number of OB HDF5 paths ({len(ob_hdf5_paths)}) does not match number of OB TIFF path groups "
            f"({len(ob_tiff_paths)})."
        )

    # One reporter for the whole run, resolved exactly once: a second resolve of `progress=True`
    # would build a second tqdm sink and a duplicate set of bars. Each stage below takes its own
    # view via `run_progress.for_stage(...)`, and the leaves it calls borrow that view, so only this
    # context manager retires the bars — on the way out of a clean run and of a failed one alike.
    with resolve_progress(progress) as run_progress:
        samples = []
        ob = []

        # One reporter per input family, reused for every run in it: a borrowed view shares its counter
        # cell, so N calls accumulate into one count across the whole run instead of restarting per run.
        load_sample = run_progress.for_stage(STAGE_LOAD_SAMPLE, total=total_across_groups(sample_tiff_paths))
        load_ob = run_progress.for_stage(STAGE_LOAD_OB, total=total_across_groups(ob_tiff_paths))
        combine = run_progress.for_stage(STAGE_COMBINE_RUNS, total=2)

        # Load data from TIFF files and metadata from HDF5 files
        for hdf5_path, tiff_paths in zip(sample_hdf5_paths, sample_tiff_paths):
            if not tiff_paths:
                raise ValueError("Each sample TIFF path group must contain at least one TIFF file.")
            # Read the spectra TOF sidecar from the directory the images actually came from (the
            # auto-reduction tree), not the raw-acquisition path in the DAS log — see GitHub #187.
            metadata = load_metadata(hdf5_path, read_spectra_tof=True, image_dir=Path(tiff_paths[0]).parent)
            sample = load_tiff_stack(tiff_paths, progress=load_sample)
            # Attach metadata as coordinates to the sample DataArray for later use in normalization and rebinning
            for key, value in metadata.items():
                if key == "spectra_tof":
                    # spectra_tof holds per-frame LEFT bin edges; build N+1 bin edges so tof is a
                    # proper bin-edge axis and rebin_by_tof works (#187).
                    sample = sample.rename_dims({"N_image": "tof"})
                    sample.coords["tof"] = _tof_bin_edges_from_left_edges(value)
                else:
                    sample.coords[key] = value
                    sample.coords.set_aligned(key, False)

            samples.append(sample)

        # Load data from TIFF files and metadata from HDF5 files
        for hdf5_path, tiff_paths in zip(ob_hdf5_paths, ob_tiff_paths):
            if not tiff_paths:
                raise ValueError("Each OB TIFF path group must contain at least one TIFF file.")
            # Read the spectra TOF sidecar from the directory the images actually came from (the
            # auto-reduction tree), not the raw-acquisition path in the DAS log — see GitHub #187.
            metadata = load_metadata(hdf5_path, read_spectra_tof=True, image_dir=Path(tiff_paths[0]).parent)
            ob_run = load_tiff_stack(tiff_paths, progress=load_ob)
            # Attach metadata as coordinates to the OB DataArray for later use in normalization and rebinning
            for key, value in metadata.items():
                if key == "spectra_tof":
                    # spectra_tof holds per-frame LEFT bin edges; build N+1 bin edges so tof is a
                    # proper bin-edge axis and rebin_by_tof works (#187).
                    ob_run = ob_run.rename_dims({"N_image": "tof"})
                    ob_run.coords["tof"] = _tof_bin_edges_from_left_edges(value)
                else:
                    ob_run.coords[key] = value
                    ob_run.coords.set_aligned(key, False)

            ob.append(ob_run)

        combine.note(f"combining {len(samples)} sample run(s)")
        sample = combine_runs(
            samples,
            metadata_keys_to_sum=["proton_charge", "duration"],
            metadata_check_match=["detector_time_offset", "detector"],
            normalize_by_runs=True,
        )

        combine()
        combine.note(f"combining {len(ob)} open-beam run(s)")
        ob = combine_runs(
            ob,
            metadata_keys_to_sum=["proton_charge", "duration"],
            metadata_check_match=["detector_time_offset", "detector"],
            normalize_by_runs=True,
        )
        combine()

        # Apply ROI if specified
        if roi:
            sample = apply_roi(sample, roi)
            ob = apply_roi(ob, roi)

        # Dead pixel detection
        sample.masks["dead_pixels"] = detect_dead_pixels(ob)

        # Spatial rebinning (optional)
        if rebin_by_spatial is not None:
            sample = rebin_spatial(sample, rebin_by_spatial)
            ob = rebin_spatial(ob, rebin_by_spatial)
            # redo mask after rebinning
            sample.masks["dead_pixels"] = detect_dead_pixels(ob)

        # TOF rebinning (optional): an integer factor, ``True`` for the statistics-based recommended
        # factor, or an explicit ``[[start, stop], ...]`` bin list. ``rebin_reduction`` selects how
        # frames combine (default: sum for a factor, mean for a bin list); see ``rebin_tof``.
        # A bin list/tuple (even empty) is an explicit rebin request; an empty one must surface as an error
        # from ``rebin_tof`` rather than be silently skipped by the plain falsy check.
        if rebin_by_tof or isinstance(rebin_by_tof, (list, tuple)):
            spec = rebin_by_tof
            if spec is True:
                spec = analyze_statistics(ob).recommended_rebinning
                logger.info(f"Recommended TOF rebinning factor based on statistics analysis: {spec}")
            if isinstance(spec, bool) or not isinstance(spec, (int, np.integer, list, tuple)):
                raise ValueError(
                    f"rebin_by_tof must be a bool, an int factor, or a list/tuple of [start, stop] pairs; got {spec!r}"
                )
            # rebin_tof takes no progress argument of its own, so the pipeline names the two calls
            # around it: with a median reduction this is one of the slowest stages in the run.
            rebin = run_progress.for_stage(STAGE_REBIN_TOF, total=2)
            rebin.note("rebinning sample TOF")
            sample = rebin_tof(sample, spec, reduction=rebin_reduction)
            rebin()
            rebin.note("rebinning open beam TOF")
            ob = rebin_tof(ob, spec, reduction=rebin_reduction)
            rebin()

        # Normalization
        transmission = normalize_transmission(
            sample=sample,
            ob=ob,
            proton_charge_sample=sample.coords["proton_charge"],
            proton_charge_ob=ob.coords["proton_charge"],
            progress=run_progress.for_stage(
                STAGE_NORMALIZE,
                total=normalize_step_count(proton_charge_sample=sample.coords["proton_charge"]),
            ),
        )

        # Air region correction (optional)
        if air_roi is not None:
            transmission = apply_air_region_correction(transmission, air_roi)

        # Add wavelength and energy coordinates converted from TOF using the configurable flight
        # path and the time offset from the metadata.
        if "detector_time_offset" in sample.coords:
            time_offset = sample.coords["detector_time_offset"]
            transmission.coords["wavelength"] = convert_tof_to_wavelength(
                transmission.coords["tof"], flight_path, time_offset
            )
            transmission.coords["energy"] = convert_tof_to_energy(transmission.coords["tof"], flight_path, time_offset)
        else:
            logger.warning("Time offset not found in metadata. Cannot add wavelength and energy coordinates.")

        # Write output
        metadata = {
            "sample_hdf5_paths": [str(run) for run in sample_hdf5_paths],
            "ob_hdf5_paths": [str(run) for run in ob_hdf5_paths],
            "sample_tiff_paths": [[str(p) for p in run] for run in sample_tiff_paths],
            "ob_tiff_paths": [[str(p) for p in run] for run in ob_tiff_paths],
            "processing_timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

        if roi:
            metadata["roi_applied"] = region_provenance(roi)

        if air_roi is not None:
            metadata["air_roi"] = region_provenance(air_roi)

        output_description = str(output_path)
        if output_path.suffix.lower() in (".hdf5", ".h5"):
            write_hdf5(
                output_path,
                transmission,
                dead_pixel_mask="dead_pixels",
                metadata=metadata,
                progress=run_progress.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(transmission, metadata)),
            )
        elif output_path.suffix.lower() in (".tiff", ".tif"):
            rename_map = {}
            if "tof" in transmission.dims:
                rename_map["tof"] = "t"  # TIFF stacks typically use 't' for the time dimension
            if rename_map:
                transmission = transmission.rename_dims(rename_map)

            model = "Unknown"
            if "detector" in sample.coords:
                model = sample.coords["detector"].value

            daqmetadata = {
                "facility": "SNS",
                "instrument": "VENUS",
                "detector_type": model,
                "source_type": "neutron",
            }

            # Combine all masks and broadcast to the shape of the transmission data.
            # Mask must be same shape as the image data for scitiff. Broadcast each mask by DIM NAME
            # (scipp), so both a spatial (y, x) mask and a 1-D per-frame (t) mask expand correctly to
            # the full (t, y, x) stack.
            if transmission.masks:
                combined_mask = np.zeros_like(transmission.values, dtype=bool)
                for mask in transmission.masks.values():
                    combined_mask |= sc.broadcast(mask, sizes=transmission.sizes).values

                # remove other masks
                transmission.masks.clear()
                # add combined mask back in with name "scitiff-mask"
                transmission.masks["scitiff-mask"] = sc.array(dims=transmission.dims, values=combined_mask, dtype=bool)

            written_paths = write_tiff_stack(
                output_path,
                transmission,
                metadata=metadata,
                daqmetadata=daqmetadata,
                one_file_per_image=tiff_one_file_per_image,
                progress=run_progress.for_stage(
                    STAGE_EXPORT,
                    total=tiff_export_step_count(transmission, one_file_per_image=tiff_one_file_per_image),
                ),
            )
            # In per-image mode ``output_path`` is only a naming template and is never written, so
            # report what actually landed on disk rather than a file that does not exist.
            if len(written_paths) > 1:
                output_description = f"{len(written_paths)} files, {written_paths[0].name} .. {written_paths[-1].name}"
            else:
                output_description = str(written_paths[0])

        else:
            raise ValueError(f"Unsupported output file format: {output_path.suffix}")

        logger.success("VENUS TPX1 pipeline completed successfully. Output written to {}", output_description)
        return transmission

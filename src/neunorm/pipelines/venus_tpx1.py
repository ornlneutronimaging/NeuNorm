"""
VENUS TPX1 pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import scipp as sc

from neunorm import __version__
from neunorm.data_models.moving_window import MovingWindow
from neunorm.data_models.roi import RegionLike, RegionsLike, ROILike
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.pipelines._tof_spine import (
    TofPipelineProfile,
    coerce_roi_arguments,
    reduce_tof_stacks,
    require_matching_group_counts,
)
from neunorm.processing.run_combiner import combine_runs
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    Progress,
    resolve_progress,
    total_across_groups,
)

#: TPX1 detects no hot pixels, re-detects its masks from the open beam after a spatial rebin, passes
#: no ``hot_pixel_mask`` to the HDF5 writer, and reads its TIFF detector model from the metadata.
_TPX1_PROFILE = TofPipelineProfile(
    label="VENUS TPX1",
    detect_hot=False,
    remask_after_spatial_rebin_from="ob",
    hdf5_hot_pixel_mask=None,
    tiff_detector_model=None,
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


def run_venus_tpx1_pipeline(
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
    spectrum_roi: Optional[RegionsLike] = None,
    spectrum_roi_strict: bool = True,
    moving_window: Optional[MovingWindow] = None,
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
    moving_window : MovingWindow, optional
        Replace each pixel by the average — or, with ``kind="sum"``, the total — of a box of pixels
        around it, applied to **both** stacks immediately before they are divided. Off by default.
        Sizes are given by dimension name (``MovingWindow(x=3, y=3)``) in **post-crop,
        post-spatial-rebin** pixels, so a window of 3 on a ``rebin_by_spatial=2`` stack spans 6
        detector pixels. Dead and hot pixels are excluded from the window rather than averaged into
        it. A ``k x k`` window improves per-pixel precision by ``k`` and coarsens spatial resolution
        by ``k`` while the array keeps its shape; ``docs/moving_window.md`` has the measured trade.
        Cannot be combined with ``spectrum_roi``.
    progress : bool or callable, optional
        Progress reporting for the whole run, off by default (and free when off). ``True`` lets
        NeuNorm draw one :mod:`tqdm` bar per stage; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent` for every item or step and is how any progress
        library is driven. Raising from the callback cancels the run.

        The stages reported are the sample and open-beam loads — **one event per TIFF**, counted across
        all input runs rather than restarting per run — then the run combine, the TOF rebin when one is
        requested, the normalization, and the export, which is per file with
        ``tiff_one_file_per_image=True``. Not every operation in between is reported: the metadata
        reads, the ROI crop, the dead-pixel detection, the statistics analysis, the spatial rebin and the
        air-region correction are single passes that run between named stages. See
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
    roi, air_roi = coerce_roi_arguments(roi, air_roi)

    # length of hdf5 paths and tiff paths should match for both sample and OB
    require_matching_group_counts(sample_hdf5_paths, sample_tiff_paths, ob_hdf5_paths, ob_tiff_paths)

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

        metadata = {
            "sample_hdf5_paths": [str(run) for run in sample_hdf5_paths],
            "ob_hdf5_paths": [str(run) for run in ob_hdf5_paths],
            "sample_tiff_paths": [[str(p) for p in run] for run in sample_tiff_paths],
            "ob_tiff_paths": [[str(p) for p in run] for run in ob_tiff_paths],
            "processing_timestamp": datetime.now().isoformat(),
            "version": __version__,
        }

        return reduce_tof_stacks(
            sample,
            ob,
            output_path=output_path,
            profile=_TPX1_PROFILE,
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

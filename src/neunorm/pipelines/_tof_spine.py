"""
The shared middle of the three VENUS TOF pipelines.

``venus_tpx1``, ``venus_tpx3_histogram`` and ``venus_tpx3_event`` differ only in how they get their
two stacks — a TIFF stack plus a spectra sidecar, a TIFF stack plus NeXus TOF binning, or events
histogrammed on the fly — and in a handful of per-detector details. Everything from the ROI crop to
the written file was the same 150 lines copied three times.

That middle lives here, once. Each entry point keeps its own loading and metadata and calls
:func:`reduce_tof_stacks`; the per-detector differences are named in a :class:`TofPipelineProfile`
rather than encoded by which copy of the code you are reading.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import (
    MaskROI,
    RegionLike,
    RegionsLike,
    ROILike,
    as_region_list,
    as_roi_bounds,
    region_provenance,
)
from neunorm.exporters.ascii_writer import ascii_spectrum_export_step_count, write_ascii_spectrum
from neunorm.exporters.hdf5_writer import hdf5_export_step_count, write_hdf5
from neunorm.exporters.tiff_writer import tiff_export_step_count, write_tiff_stack
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_step_count, normalize_transmission
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.spatial_rebinner import rebin_spatial
from neunorm.processing.spectrum_reducer import (
    normalize_roi_spectrum,
    normalize_roi_spectrum_step_count,
    spectrum_reduction_provenance,
)
from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength
from neunorm.tof.histogram_rebinner import _parse_bin_list, linear_bin_list, rebin_tof
from neunorm.tof.pixel_detector import detect_dead_pixels, detect_hot_pixels
from neunorm.tof.statistics_analyzer import analyze_statistics
from neunorm.utils.progress import (
    STAGE_EXPORT,
    STAGE_NORMALIZE,
    STAGE_REBIN_TOF,
    STAGE_REDUCE_SPECTRUM,
    ProgressReporter,
)


@dataclass(frozen=True)
class TofPipelineProfile:
    """What genuinely differs between the three VENUS TOF pipelines.

    Every field records a difference that exists in the code today and is preserved verbatim, including
    the two that look like oversights. They are named here rather than silently unified, because
    unifying them would change published output.

    Parameters
    ----------
    label : str
        How the pipeline names itself in its completion log line.
    detect_hot : bool
        Whether a hot-pixel mask is detected alongside the dead-pixel one. TPX1 does not.
    remask_after_spatial_rebin_from : {"ob", "sample"}
        Which stack the masks are re-detected from after a spatial rebin. Two pipelines use the open
        beam; ``venus_tpx3_histogram`` uses the sample, which disagrees with its own pre-rebin
        detection (that one reads the open beam). Preserved as-is: changing it changes that
        pipeline's masks.
    hdf5_hot_pixel_mask : str, optional
        The mask name handed to :func:`~neunorm.exporters.hdf5_writer.write_hdf5` as
        ``hot_pixel_mask``, or ``None`` to leave the writer's default. TPX1 passes nothing, so a mask
        literally named ``hot_pixels`` would land at ``/masks/hot_pixels`` there and at ``/masks/hot``
        in the other two. Preserved: it is the on-disk layout.
    tiff_detector_model : str, optional
        The ``detector_type`` written into the TIFF DAQ metadata. ``None`` reads it from the sample's
        ``detector`` coordinate, falling back to ``"Unknown"``; the event pipeline hard-codes
        ``"TPX3"``.
    """

    label: str
    detect_hot: bool
    remask_after_spatial_rebin_from: Literal["ob", "sample"]
    hdf5_hot_pixel_mask: Optional[str]
    tiff_detector_model: Optional[str]


def coerce_roi_arguments(
    roi: Optional[ROILike], air_roi: Optional[RegionLike]
) -> tuple[Optional[tuple], Optional[RegionLike]]:
    """Coerce the ROI arguments every TOF entry point accepts to the forms downstream code expects.

    A crop is always a bounds tuple; an air region may also be an arbitrary-shape ``MaskROI``, which
    passes through untouched.
    """
    if roi is not None:
        roi = as_roi_bounds(roi)
    if air_roi is not None:
        air_roi = air_roi if isinstance(air_roi, MaskROI) else as_roi_bounds(air_roi)
    return roi, air_roi


def require_matching_group_counts(sample_hdf5_paths, sample_tiff_paths, ob_hdf5_paths, ob_tiff_paths) -> None:
    """One metadata file per TIFF group, for the two pipelines that take both."""
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


def _attach_pixel_masks(sample: sc.DataArray, source: sc.DataArray, profile: TofPipelineProfile) -> None:
    """Attach the dead (and, per profile, hot) pixel masks to ``sample``, detected from ``source``."""
    sample.masks["dead_pixels"] = detect_dead_pixels(source)
    if profile.detect_hot:
        sample.masks["hot_pixels"] = detect_hot_pixels(source)


def _resolve_tof_rebin_spec(spec, ob: sc.DataArray):
    """Resolve ``rebin_by_tof`` to a concrete factor or bin list, rejecting anything else.

    ``True`` means "ask the statistics analysis", which is why the open beam is needed here.
    """
    if spec is True:
        spec = analyze_statistics(ob).recommended_rebinning
        logger.info(f"Recommended TOF rebinning factor based on statistics analysis: {spec}")
    if isinstance(spec, bool) or not isinstance(spec, (int, np.integer, list, tuple)):
        raise ValueError(
            f"rebin_by_tof must be a bool, an int factor, or a list/tuple of [start, stop] pairs; got {spec!r}"
        )
    return spec


def _apply_rebinning(
    sample: sc.DataArray,
    ob: sc.DataArray,
    *,
    profile: TofPipelineProfile,
    rebin_by_spatial,
    rebin_by_tof,
    rebin_reduction,
    run_progress: ProgressReporter,
) -> tuple[sc.DataArray, sc.DataArray, object]:
    """Spatial then TOF rebinning, each optional, with the masks re-detected after a spatial rebin.

    Also returns the TOF rebin spec as actually resolved — the concrete factor or bin list, with
    ``rebin_by_tof=True`` already turned into the statistics-recommended factor — or ``None`` when no
    TOF rebin ran. A spectrum run needs it to label each output point with the input frame index it
    came from, and re-deriving it at the call site would mean resolving ``True`` twice.
    """
    if rebin_by_spatial is not None:
        sample = rebin_spatial(sample, rebin_by_spatial)
        ob = rebin_spatial(ob, rebin_by_spatial)
        # redo mask after rebinning
        source = ob if profile.remask_after_spatial_rebin_from == "ob" else sample
        _attach_pixel_masks(sample, source, profile)

    # TOF rebinning (optional): an integer factor, ``True`` for the statistics-based recommended
    # factor, or an explicit ``[[start, stop], ...]`` bin list. ``rebin_reduction`` selects how
    # frames combine (default: sum for a factor, mean for a bin list); see ``rebin_tof``.
    # A bin list/tuple (even empty) is an explicit rebin request; an empty one must surface as an error
    # from ``rebin_tof`` rather than be silently skipped by the plain falsy check.
    resolved_spec = None
    if rebin_by_tof or isinstance(rebin_by_tof, (list, tuple)):
        spec = _resolve_tof_rebin_spec(rebin_by_tof, ob)
        resolved_spec = spec
        # rebin_tof takes no progress argument of its own, so the pipeline names the two calls
        # around it: with a median reduction this is one of the slowest stages in the run.
        rebin = run_progress.for_stage(STAGE_REBIN_TOF, total=2)
        rebin.note("rebinning sample TOF")
        sample = rebin_tof(sample, spec, reduction=rebin_reduction)
        rebin()
        rebin.note("rebinning open beam TOF")
        ob = rebin_tof(ob, spec, reduction=rebin_reduction)
        rebin()

    return sample, ob, resolved_spec


def spectrum_bin_indices(n_frames: Optional[int], spec) -> Optional[list[int]]:
    """The input frame index each output spectrum point came from.

    The ASCII spectrum's first column is documented as "the same as the file index if no binning", so
    it is each bin's FIRST input frame — which degenerates to the row index exactly when no rebinning
    ran, and stays traceable when one did. A gapped bin list therefore produces a column with gaps
    (``[[0, 2], [4, 6]]`` -> ``0, 4``) rather than renumbered rows, and the dropped 2-3 span has no
    row at all.

    Returns ``None`` when no TOF rebin ran, which the writer renders as the plain row index.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        ranges = _parse_bin_list(spec)
    else:
        if n_frames is None:
            return None
        ranges = linear_bin_list(n_frames, int(spec))
    return [int(start) for start, _ in ranges]


def _warn_on_a_gapped_spectrum(spec, n_frames: Optional[int]) -> None:
    """Warn when a spectrum's frames are not contiguous, because the axis stops being a spectrum.

    `docs/workflows/venus_tpx1.md` and the rebinner's own module warning already tell users that once
    frames are dropped the axis is no longer a continuous spectrum and that Bragg-edge fitting,
    resonance analysis and integrating over a wavelength range are not valid on it. That warning
    matters more here than in image mode, because a spectrum is precisely the thing those tools fit.

    Warned rather than refused: dropping frames between bins is a deliberate, requested behaviour and a
    user may well want a gapped spectrum for something other than fitting. Silence is the option not
    taken — the file looks entirely well-formed either way.
    """
    if not isinstance(spec, (list, tuple)):
        return
    ranges = _parse_bin_list(spec)
    gaps = [(prev[1], nxt[0]) for prev, nxt in zip(ranges, ranges[1:]) if nxt[0] > prev[1]]
    leading = ranges[0][0] if ranges and ranges[0][0] > 0 else None
    trailing = ranges[-1][1] if ranges and n_frames is not None and ranges[-1][1] < n_frames else None
    if not gaps and leading is None and trailing is None:
        return
    dropped = [f"{start}-{stop - 1}" for start, stop in gaps]
    if leading is not None:
        dropped.insert(0, f"0-{leading - 1}")
    if trailing is not None:
        dropped.append(f"{trailing}-{n_frames - 1}")
    logger.warning(
        "spectrum_roi with a gapped bin list: frames {} are covered by no range and are dropped, so the "
        "output axis is NOT a continuous spectrum. Resonance and Bragg-edge fitting, and integrating "
        "over a wavelength range, are not valid on it. Use contiguous ranges if the spectrum will be "
        "fitted.",
        ", ".join(dropped),
    )


def _add_derived_coords(transmission: sc.DataArray, sample: sc.DataArray, flight_path: sc.Variable) -> None:
    """Label the result with wavelength and energy converted from TOF, when the offset is known."""
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


def _write_tiff_output(
    output_path: Path,
    transmission: sc.DataArray,
    sample: sc.DataArray,
    *,
    profile: TofPipelineProfile,
    metadata: dict,
    tiff_one_file_per_image: bool,
    run_progress: ProgressReporter,
) -> tuple[str, sc.DataArray]:
    """Write the transmission stack as scitiff.

    Returns what to report as the output **and the array as scitiff needed it** — dims renamed
    ``tof`` -> ``t`` and every mask combined into one ``scitiff-mask``. That rewritten array is what
    a TIFF run has always returned to its caller, and tests pin it, so it is handed back rather than
    left as a local.
    """
    rename_map = {}
    if "tof" in transmission.dims:
        rename_map["tof"] = "t"  # TIFF stacks typically use 't' for the time dimension
    if rename_map:
        transmission = transmission.rename_dims(rename_map)

    model = profile.tiff_detector_model
    if model is None:
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
        description = f"{len(written_paths)} files, {written_paths[0].name} .. {written_paths[-1].name}"
    else:
        description = str(written_paths[0])
    return description, transmission


def _export_transmission(
    output_path: Path,
    transmission: sc.DataArray,
    sample: sc.DataArray,
    *,
    profile: TofPipelineProfile,
    metadata: dict,
    tiff_one_file_per_image: bool,
    run_progress: ProgressReporter,
) -> tuple[str, sc.DataArray]:
    """Write the result in the format ``output_path``'s suffix selects.

    Returns the description to log and the transmission as written — the TIFF path rewrites it (see
    :func:`_write_tiff_output`), the HDF5 path leaves it alone.
    """
    if output_path.suffix.lower() in (".hdf5", ".h5"):
        hot_kwargs = {} if profile.hdf5_hot_pixel_mask is None else {"hot_pixel_mask": profile.hdf5_hot_pixel_mask}
        write_hdf5(
            output_path,
            transmission,
            dead_pixel_mask="dead_pixels",
            metadata=metadata,
            progress=run_progress.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(transmission, metadata)),
            **hot_kwargs,
        )
        return str(output_path), transmission
    if output_path.suffix.lower() in (".tiff", ".tif"):
        return _write_tiff_output(
            output_path,
            transmission,
            sample,
            profile=profile,
            metadata=metadata,
            tiff_one_file_per_image=tiff_one_file_per_image,
            run_progress=run_progress,
        )
    raise ValueError(f"Unsupported output file format: {output_path.suffix}")


def _export_spectrum(
    output_path: Path,
    spectrum: sc.DataArray,
    *,
    metadata: dict,
    bin_indices: Optional[list[int]],
    run_progress: ProgressReporter,
) -> str:
    """Write a 1-D transmission spectrum: the ASCII table, and HDF5 alongside it.

    Takes no :class:`TofPipelineProfile`: the profile's fields are all about image output — the
    dead/hot mask names the HDF5 writer is given, and the TIFF detector model — and a region mean has
    consumed the spatial masks by this point, so none of them apply.

    ``.txt`` writes both — the three-column table the downstream fitting tools read, plus an HDF5
    sibling carrying the provenance, the masks and the time axis that three columns cannot hold.
    ``.hdf5``/``.h5`` writes only the HDF5.

    ``.tiff``/``.tif`` is refused, and refused HERE rather than left to the writer. A 1-D spectrum
    reaching :func:`~neunorm.exporters.tiff_writer.write_tiff_stack` after the pipelines' usual
    ``tof`` -> ``t`` rename does not fail: it writes a multi-page TIFF of 1x1-pixel images, one per
    bin, which even reads back cleanly. Nothing downstream would flag it.
    """
    suffix = output_path.suffix.lower()
    if suffix in (".tiff", ".tif"):
        raise ValueError(
            f"spectrum_roi produces a 1-D spectrum, which cannot be written as a TIFF image stack "
            f"(got {output_path.name}). Use '.txt' for the three-column ASCII spectrum (an HDF5 file "
            "is written alongside it) or '.hdf5' for HDF5 only."
        )
    if suffix in (".hdf5", ".h5"):
        write_hdf5(
            output_path,
            spectrum,
            metadata=metadata,
            progress=run_progress.for_stage(STAGE_EXPORT, total=hdf5_export_step_count(spectrum, metadata)),
        )
        return str(output_path)
    if suffix == ".txt":
        hdf5_path = output_path.with_suffix(".hdf5")
        _refuse_to_overwrite_an_input(hdf5_path, metadata)
        # ONE reporter shared by both writers: each borrows it and shares its counter cell, so the
        # export bar advances continuously across the two files. That also means each writer's own
        # `total=` is ignored, so the combined total has to be declared here.
        export = run_progress.for_stage(
            STAGE_EXPORT,
            total=ascii_spectrum_export_step_count(spectrum) + hdf5_export_step_count(spectrum, metadata),
        )
        write_ascii_spectrum(output_path, spectrum, bin_indices, progress=export)
        write_hdf5(hdf5_path, spectrum, metadata=metadata, progress=export)
        return f"{output_path} (+ {hdf5_path.name})"
    raise ValueError(f"Unsupported output file format: {output_path.suffix}")


def _input_paths(metadata: dict) -> set:
    """Every input file path the run recorded, resolved, flattened out of the nested per-run lists."""
    found = set()

    def walk(value):
        if isinstance(value, str):
            try:
                found.add(Path(value).resolve())
            except OSError:  # pragma: no cover - a path the filesystem cannot resolve is not an input
                pass
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for key, value in metadata.items():
        if key.endswith("_paths"):
            walk(value)
    return found


def _refuse_to_overwrite_an_input(hdf5_path: Path, metadata: dict) -> None:
    """Refuse to write the HDF5 sibling on top of one of the run's own input files.

    This is the one place in the package that writes a file the user did not name: a ``.txt`` output
    also produces ``<stem>.hdf5``. ``write_hdf5`` opens with mode ``"w"``, which truncates, and a run's
    metadata file is plausibly named ``<stem>.hdf5`` beside where a user would ask for ``<stem>.txt``, so
    the derived path is checked against the run's inputs before anything is written.

    Overwriting a file that is NOT an input is left alone: replacing your own previous output is normal
    and is what every other writer here does.
    """
    resolved = hdf5_path.resolve() if hdf5_path.exists() else None
    if resolved is not None and resolved in _input_paths(metadata):
        raise ValueError(
            f"writing the spectrum to {hdf5_path.name} would overwrite one of this run's own input "
            f"files ({hdf5_path}). A '.txt' output also writes '<stem>.hdf5' alongside it, and that "
            "path is an input here. Choose a different output name or an output directory of its own."
        )


def _warn_on_spectrum_roi_frame(roi, rebin_by_spatial) -> None:
    """Say out loud which pixel frame ``spectrum_roi`` was resolved in, when it is not the detector's.

    The crop and the spatial rebin both run before the region is collapsed, so the indices are
    resolved against the array as it is at that point. ``background_roi`` carries the same caveat in
    its docstrings, and this one is more surprising because a rebin factor rescales the indices as
    well as shifting them. The region still selects real pixels either way, and the spectrum still
    looks entirely plausible, which is exactly why this is said rather than left in the docs.
    """
    if roi:
        logger.warning(
            "spectrum_roi indices are resolved AFTER the roi={} crop, so they are offsets into the "
            "cropped image, not detector pixels.",
            tuple(roi),
        )
    if rebin_by_spatial is not None:
        logger.warning(
            "spectrum_roi indices are resolved AFTER rebin_by_spatial={}, so one index step is that "
            "many detector pixels.",
            rebin_by_spatial,
        )


def reduce_tof_stacks(
    sample: sc.DataArray,
    ob: sc.DataArray,
    *,
    output_path: Path,
    profile: TofPipelineProfile,
    metadata: dict,
    roi: Optional[tuple] = None,
    air_roi: Optional[RegionLike] = None,
    rebin_by_tof=False,
    rebin_by_spatial=None,
    rebin_reduction=None,
    flight_path: sc.Variable,
    tiff_one_file_per_image: bool = False,
    spectrum_roi: Optional[RegionsLike] = None,
    spectrum_roi_strict: bool = True,
    run_progress: ProgressReporter,
) -> sc.DataArray:
    """Crop, mask, rebin, normalize, label and write — the part every VENUS TOF pipeline shares.

    Called with both stacks already loaded and run-combined. The steps run in the order the three
    pipelines have always run them: crop, dead/hot detection, spatial rebin (re-detecting the masks),
    TOF rebin, normalization, air-region correction, wavelength/energy labelling, export.

    That order is why an ROI resolved at normalization time is expressed in **post-crop,
    post-spatial-rebin** pixels rather than detector pixels.

    With ``spectrum_roi`` the last three steps change: the region is collapsed to one value per bin
    and divided once, giving a 1-D transmission spectrum written as a three-column ASCII file — the
    "resonance mode" reduction. Everything before it, including the rebinning, is identical, so a
    spectrum run and an image run of the same data see the same counts.

    Parameters
    ----------
    sample, ob : sc.DataArray
        The combined sample and open-beam stacks.
    output_path : Path
        Where to write; the suffix selects the writer.
    profile : TofPipelineProfile
        The per-detector differences.
    metadata : dict
        Provenance assembled by the entry point. ``roi``/``air_roi``/``spectrum_roi`` provenance is
        added here, so every pipeline records it identically.
    roi : tuple, optional
        Crop bounds, already coerced by :func:`coerce_roi_arguments`.
    air_roi : ROI, MaskROI, or tuple, optional
        Air region for the post-normalization scale correction. Image mode only — it scales an image
        so its air region reads 1.0, which is meaningless once the output is one number per bin.
    rebin_by_tof, rebin_by_spatial, rebin_reduction
        Rebinning arguments, forwarded as the pipelines document them.
    flight_path : sc.Variable
        Source-to-detector distance for the TOF conversions.
    tiff_one_file_per_image : bool
        TIFF output only: one file per spectral image. Image mode only.
    spectrum_roi : ROI, MaskROI, tuple, or a sequence of them, optional
        Switches the run to spectrum mode over this region.
    spectrum_roi_strict : bool, optional
        Whether a non-positive or non-finite open-beam region mean raises. See
        :func:`~neunorm.processing.spectrum_reducer.normalize_roi_spectrum`.
    run_progress : ProgressReporter
        The run's reporter, already resolved by the entry point.

    Returns
    -------
    sc.DataArray
        The normalized transmission that was written — an image stack, or a 1-D spectrum under
        ``spectrum_roi``.
    """
    if spectrum_roi is not None:
        if air_roi is not None:
            raise ValueError(
                "air_roi and spectrum_roi cannot be combined: the air correction rescales an image so "
                "its air region reads 1.0, which has no meaning for a spectrum of one value per bin. "
                "Drop air_roi, or run in image mode."
            )
        if tiff_one_file_per_image:
            raise ValueError(
                "tiff_one_file_per_image applies to TIFF image output, which spectrum_roi does not "
                "produce. Drop it, or run in image mode."
            )
        _warn_on_spectrum_roi_frame(roi, rebin_by_spatial)

    # Apply ROI if specified
    if roi:
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)

    # Dead (and hot) pixel detection
    _attach_pixel_masks(sample, ob, profile)

    n_frames_before_rebin = sample.sizes.get("tof")

    sample, ob, resolved_spec = _apply_rebinning(
        sample,
        ob,
        profile=profile,
        rebin_by_spatial=rebin_by_spatial,
        rebin_by_tof=rebin_by_tof,
        rebin_reduction=rebin_reduction,
        run_progress=run_progress,
    )

    if roi:
        metadata["roi_applied"] = region_provenance(roi)

    if spectrum_roi is not None:
        regions = as_region_list(spectrum_roi, arg_name="spectrum_roi")
        # Collapse the region to one value per bin BEFORE dividing: (Sum a)/(Sum b) != Sum(a/b), so
        # a region-level measurement has to be built from region-level counts.
        spectrum = normalize_roi_spectrum(
            sample,
            ob,
            regions,
            proton_charge_sample=sample.coords["proton_charge"],
            proton_charge_ob=ob.coords["proton_charge"],
            spectrum_roi_strict=spectrum_roi_strict,
            progress=run_progress.for_stage(
                STAGE_REDUCE_SPECTRUM,
                total=normalize_roi_spectrum_step_count(proton_charge_sample=sample.coords["proton_charge"]),
            ),
        )
        _add_derived_coords(spectrum, sample, flight_path)
        # `resolved_spec`, not `rebin_by_tof`: with rebin_by_tof=True the factor came from the
        # statistics analysis, and recording the literal True would leave the file unable to say how
        # many frames went into a point.
        metadata.update(spectrum_reduction_provenance(regions, reduction=rebin_reduction, rebin_by_tof=resolved_spec))
        _warn_on_a_gapped_spectrum(resolved_spec, n_frames_before_rebin)
        bin_indices = spectrum_bin_indices(n_frames_before_rebin, resolved_spec)
        if bin_indices is not None:
            metadata["spectrum_bin_first_frame"] = bin_indices
        output_description = _export_spectrum(
            output_path,
            spectrum,
            metadata=metadata,
            bin_indices=bin_indices,
            run_progress=run_progress,
        )
        logger.success("{} pipeline completed successfully. Output written to {}", profile.label, output_description)
        return spectrum

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

    _add_derived_coords(transmission, sample, flight_path)

    if air_roi is not None:
        metadata["air_roi"] = region_provenance(air_roi)

    output_description, transmission = _export_transmission(
        output_path,
        transmission,
        sample,
        profile=profile,
        metadata=metadata,
        tiff_one_file_per_image=tiff_one_file_per_image,
        run_progress=run_progress,
    )

    logger.success("{} pipeline completed successfully. Output written to {}", profile.label, output_description)
    return transmission

"""
VENUS TPX3 event pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.data_models.roi import ROILike, as_roi_bounds
from neunorm.data_models.tof import BinningConfig
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.loaders.event_loader import load_event_nexus
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.processing.spatial_rebinner import rebin_spatial
from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength
from neunorm.tof.event_converter import convert_events_to_histogram
from neunorm.tof.histogram_rebinner import rebin_tof
from neunorm.tof.pixel_detector import detect_dead_pixels, detect_hot_pixels
from neunorm.tof.statistics_analyzer import analyze_statistics
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M


def run_venus_tpx3_event_pipeline(  # noqa: C901
    sample_paths: Sequence[str | Path],
    ob_paths: Sequence[str | Path],
    binning: BinningConfig,
    output_path: Path,
    roi: Optional[ROILike] = None,
    air_roi: Optional[ROILike] = None,
    rebin_by_tof: Optional[bool | int] = False,
    rebin_by_spatial: Optional[int | tuple[int, int]] = None,
    detector_shape: tuple[int, int] = (514, 514),
    event_id_offset: int = 1_000_000,
    bank_name: str = "bank100",
    flight_path: sc.Variable = sc.scalar(VENUS_FLIGHT_PATH_M, unit="m"),
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
    air_roi : Optional[tuple]
        Region of interest for air correction — an ``ROI`` or a bare ``(x0, y0, x1, y1)`` tuple.
    rebin_by_tof : Optional[bool,int]
        Whether to apply TOF rebinning based on statistics analysis. If an integer is provided,
        it will be used as the rebinning factor instead of the recommended one.
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
        air_roi = as_roi_bounds(air_roi)

    x_bins, y_bins = detector_shape

    # Load metadata before histogramming so the detector time offset can be applied to
    # energy/wavelength bin edges; a missing offset defaults to zero.
    samples = []
    for run in sample_paths:
        metadata = load_metadata(run)
        time_offset = metadata.get("detector_time_offset", sc.scalar(0.0, unit="us"))
        sample = convert_events_to_histogram(
            load_event_nexus(
                run, detector_bank=bank_name, detector_shape=detector_shape, event_id_offset=event_id_offset
            ),
            binning,
            flight_path,
            x_bins,
            y_bins,
            detector_time_offset=time_offset,
        )
        for key, value in metadata.items():
            sample.coords[key] = value
            sample.coords.set_aligned(key, False)
        samples.append(sample)

    obs = []
    for run in ob_paths:
        metadata = load_metadata(run)
        time_offset = metadata.get("detector_time_offset", sc.scalar(0.0, unit="us"))
        ob = convert_events_to_histogram(
            load_event_nexus(
                run, detector_bank=bank_name, detector_shape=detector_shape, event_id_offset=event_id_offset
            ),
            binning,
            flight_path,
            x_bins,
            y_bins,
            detector_time_offset=time_offset,
        )
        for key, value in metadata.items():
            ob.coords[key] = value
            ob.coords.set_aligned(key, False)
        obs.append(ob)

    # Combine runs if there are multiple runs
    sample = combine_runs(
        samples,
        metadata_keys_to_sum=["proton_charge", "duration"],
        metadata_check_match=["detector_time_offset", "detector"],
        normalize_by_runs=True,
    )

    ob = combine_runs(
        obs,
        metadata_keys_to_sum=["proton_charge", "duration"],
        metadata_check_match=["detector_time_offset", "detector"],
        normalize_by_runs=True,
    )

    # Apply ROI if specified
    if roi:
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)

    # Dead pixel detection
    sample.masks["dead_pixels"] = detect_dead_pixels(ob)

    # Hot pixel detection
    sample.masks["hot_pixels"] = detect_hot_pixels(ob)

    # Spatial rebinning (optional)
    if rebin_by_spatial is not None:
        sample = rebin_spatial(sample, rebin_by_spatial)
        ob = rebin_spatial(ob, rebin_by_spatial)
        # redo mask after rebinning
        sample.masks["dead_pixels"] = detect_dead_pixels(ob)
        sample.masks["hot_pixels"] = detect_hot_pixels(ob)

    # TOF rebinning (optional)
    if rebin_by_tof:
        if rebin_by_tof is True:
            # Analyze statistics to get recommended rebinning factor
            recommended_factor = analyze_statistics(ob)
            logger.info(f"Recommended TOF rebinning factor based on statistics analysis: {recommended_factor}")
            sample = rebin_tof(sample, recommended_factor.recommended_rebinning)
            ob = rebin_tof(ob, recommended_factor.recommended_rebinning)
        elif isinstance(rebin_by_tof, int):
            logger.info(f"Applying TOF rebinning with user-specified factor: {rebin_by_tof}")
            sample = rebin_tof(sample, rebin_by_tof)
            ob = rebin_tof(ob, rebin_by_tof)
        else:
            raise ValueError(f"Invalid value for rebin_by_tof: {rebin_by_tof}. Must be bool or int.")

    # Normalization
    transmission = normalize_transmission(
        sample=sample,
        ob=ob,
        proton_charge_sample=sample.coords["proton_charge"],
        proton_charge_ob=ob.coords["proton_charge"],
    )

    # Air region correction (optional)
    if air_roi is not None:
        transmission = apply_air_region_correction(transmission, air_roi)

    # Add wavelength and energy coordinates converted from TOF using the same flight path as
    # the binning step and the time offset from the metadata.
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
        "sample_paths": [str(run) for run in sample_paths],
        "ob_paths": [str(run) for run in ob_paths],
        "processing_timestamp": datetime.now().isoformat(),
        "version": __version__,
    }

    if roi:
        metadata["roi_applied"] = roi

    if output_path.suffix.lower() in (".hdf5", ".h5"):
        write_hdf5(
            output_path, transmission, dead_pixel_mask="dead_pixels", hot_pixel_mask="hot_pixels", metadata=metadata
        )
    elif output_path.suffix.lower() in (".tiff", ".tif"):
        rename_map = {}
        if "tof" in transmission.dims:
            rename_map["tof"] = "t"  # TIFF stacks typically use 't' for the time dimension
        if rename_map:
            transmission = transmission.rename_dims(rename_map)

        daqmetadata = {
            "facility": "SNS",
            "instrument": "VENUS",
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

        write_tiff_stack(output_path, transmission, metadata=metadata, daqmetadata=daqmetadata)
    else:
        raise ValueError(f"Unsupported output file format: {output_path.suffix}")

    logger.success("VENUS TPX3 event pipeline completed successfully. Output written to {}", output_path)
    return transmission

"""
VENUS TPX1 pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.processing.spatial_rebinner import rebin_spatial
from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength
from neunorm.tof.histogram_rebinner import rebin_tof
from neunorm.tof.pixel_detector import detect_dead_pixels
from neunorm.tof.statistics_analyzer import analyze_statistics
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M


def run_venus_tpx1_pipeline(  # noqa: C901
    sample_hdf5_paths: Sequence[str | Path],
    ob_hdf5_paths: Sequence[str | Path],
    sample_tiff_paths: Sequence[Sequence[str | Path]],
    ob_tiff_paths: Sequence[Sequence[str | Path]],
    output_path: Path,
    roi: Optional[tuple] = None,
    air_roi: Optional[tuple] = None,
    rebin_by_tof: Optional[bool | int] = False,
    rebin_by_spatial: Optional[int | tuple[int, int]] = None,
    flight_path: sc.Variable = sc.scalar(VENUS_FLIGHT_PATH_M, unit="m"),
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
        Region of interest to apply (x_start, y_start, x_end, y_end)
    air_roi : Optional[tuple]
        Region of interest to use for air correction (x_start, y_start, x_end, y_end).
        If None, air correction is not applied.
    rebin_by_tof : Optional[Union[bool,int]]
        Whether to apply TOF rebinning based on statistics analysis. If an integer is provided,
        it will be used as the rebinning factor instead of the recommended one.
    rebin_by_spatial : Optional[int | tuple[int, int]]
        Whether to apply spatial rebinning. If an integer is provided, it is used as the
        rebinning factor for both spatial axes. A ``(x, y)`` tuple selects per-axis
        rebinning factors (x and y). If None, no spatial rebinning is applied.
    flight_path : sc.Variable
        Source-to-detector flight path used for TOF→energy/wavelength coordinate labeling.
        Defaults to ``VENUS_FLIGHT_PATH_M`` (25 m); set it per detector/sample position.

    Notes
    -----
    This function writes the normalized transmission data to disk in either HDF5 or TIFF format,
    depending on the file extension of `output_path`. Metadata and detector masks are included in the output.

    Returns
    -------
    sc.DataArray
        Final normalized transmission DataArray with metadata and masks
    """

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

    samples = []
    ob = []

    # Load data from TIFF files and metadata from HDF5 files
    for hdf5_path, tiff_paths in zip(sample_hdf5_paths, sample_tiff_paths):
        metadata = load_metadata(hdf5_path, read_spectra_tof=True)
        sample = load_tiff_stack(tiff_paths)
        # Attach metadata as coordinates to the sample DataArray for later use in normalization and rebinning
        for key, value in metadata.items():
            if key == "spectra_tof":
                # replace N_image dim with TOF from spectra_tof
                sample = sample.rename_dims({"N_image": "tof"})
                sample.coords["tof"] = metadata["spectra_tof"].rename_dims({"N_image": "tof"})
            else:
                sample.coords[key] = value
                sample.coords.set_aligned(key, False)

        samples.append(sample)

    # Load data from TIFF files and metadata from HDF5 files
    for hdf5_path, tiff_paths in zip(ob_hdf5_paths, ob_tiff_paths):
        metadata = load_metadata(hdf5_path, read_spectra_tof=True)
        ob_run = load_tiff_stack(tiff_paths)
        # Attach metadata as coordinates to the OB DataArray for later use in normalization and rebinning
        for key, value in metadata.items():
            if key == "spectra_tof":
                # replace N_image dim with TOF from spectra_tof
                ob_run = ob_run.rename_dims({"N_image": "tof"})
                ob_run.coords["tof"] = metadata["spectra_tof"].rename_dims({"N_image": "tof"})
            else:
                ob_run.coords[key] = value
                ob_run.coords.set_aligned(key, False)

        ob.append(ob_run)

    sample = combine_runs(
        samples,
        metadata_keys_to_sum=["proton_charge", "duration"],
        metadata_check_match=["detector_time_offset", "detector"],
        normalize_by_runs=True,
    )

    ob = combine_runs(
        ob,
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

    # Spatial rebinning (optional)
    if rebin_by_spatial is not None:
        sample = rebin_spatial(sample, rebin_by_spatial)
        ob = rebin_spatial(ob, rebin_by_spatial)
        # redo mask after rebinning
        sample.masks["dead_pixels"] = detect_dead_pixels(ob)

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

    # Add wavelength and energy coordinates converted from TOF using the configurable flight
    # path and the time offset from the metadata (issue #141).
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
        metadata["roi_applied"] = roi

    if output_path.suffix.lower() in (".hdf5", ".h5"):
        write_hdf5(output_path, transmission, dead_pixel_mask="dead_pixels", metadata=metadata)
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

    logger.success("VENUS TPX1 pipeline completed successfully. Output written to {}", output_path)
    return transmission

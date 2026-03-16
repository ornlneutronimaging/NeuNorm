"""
MARS CCD/CMOS normalization pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm import __version__
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.filters.gamma_filter import apply_gamma_filter
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.processing.dark_corrector import subtract_dark
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.tof.pixel_detector import detect_dead_pixels


def run_mars_ccd_pipeline(  # noqa: C901
    sample_paths: Sequence[str | Path],
    ob_paths: Sequence[str | Path],
    dark_paths: Sequence[str | Path],
    output_path: Path,
    roi: Optional[tuple] = None,
    gamma_filter: bool = True,
) -> sc.DataArray:
    """Execute MARS CCD/CMOS normalization pipeline.

    Pipeline Steps (10 total)
    - Load TIFF/FITS (sample, OB, dark) TODO: support FITS
    - Run combine (optional) TODO!
    - ROI clip (optional)
    - Average dark/OB
    - Dead pixel detection (existing tof/pixel_detector.py)
    - Gamma filtering (filters/gamma_filter.py)
    - Dark correction (processing/dark_corrector.py)
    - Normalization (existing processing/normalizer.py)
    - Output (exporters/hdf5_writer.py, exporters/tiff_writer.py)

    Parameters
    ----------
    sample_paths : Sequence[str | Path]
        List of paths to sample TIFF files
    ob_paths : Sequence[str | Path]
        List of paths to open beam TIFF files
    dark_paths : Sequence[str | Path]
        List of paths to dark current TIFF files
    output_path : Path
        Path to save the output file (HDF5 or TIFF)
    roi : Optional[tuple]
        Region of interest to apply (x_start, y_start, x_end, y_end)
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)

    Notes
    -----
    This function writes the normalized transmission data to disk in either HDF5 or TIFF format,
    depending on the file extension of `output_path`. Metadata and detector masks are included in the output.

    Returns
    -------
    sc.DataArray
        Final normalized transmission DataArray with metadata and masks
    """

    # Load data
    sample = load_tiff_stack(sample_paths)
    ob = load_tiff_stack(ob_paths)
    dark = load_tiff_stack(dark_paths)

    # Apply ROI if specified
    if roi:
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)
        dark = apply_roi(dark, roi)

    # Average dark and OB
    dark = prepare_reference(dark, dim="N_image")
    ob = prepare_reference(ob, dim="N_image")

    # Dead pixel detection
    sample.masks["dead_pixels"] = detect_dead_pixels(sample)

    # Gamma filtering (optional)
    if gamma_filter:
        sample = apply_gamma_filter(sample)

    # Dark correction
    sample_dark_corrected = subtract_dark(sample, dark)
    ob_dark_corrected = subtract_dark(ob, dark)

    # Normalization
    transmission = normalize_transmission(sample_dark_corrected, ob_dark_corrected)

    # Write output
    metadata = {
        "sample_paths": [str(p) for p in sample_paths],
        "ob_paths": [str(p) for p in ob_paths],
        "dark_paths": [str(p) for p in dark_paths],
        "gamma_filter_applied": gamma_filter,
        "processing_timestamp": datetime.now().isoformat(),
        "version": __version__,
    }

    if roi:
        metadata["roi_applied"] = roi

    if output_path.suffix.lower() in (".hdf5", ".h5"):
        write_hdf5(output_path, transmission, dead_pixel_mask="dead_pixels", metadata=metadata)
    elif output_path.suffix.lower() in (".tiff", ".tif"):
        rename_map = {}
        if "N_image" in transmission.dims:
            rename_map["N_image"] = "z"  # TIFF stacks typically use 'z' for the stack dimension
        if rename_map:
            transmission = transmission.rename_dims(rename_map)

        model = "Unknown"
        if "ModelStr" in sample.coords:
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

        write_tiff_stack(output_path, transmission, metadata=metadata, daqmetadata=daqmetadata)
    else:
        raise ValueError(f"Unsupported output file format: {output_path.suffix}")

    logger.success("MARS CCD pipeline completed successfully. Output written to {}", output_path)
    return transmission

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
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.filters.gamma_filter import apply_gamma_filter
from neunorm.loaders.stack_loader import load_stack
from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.pixel_detector import detect_dead_pixels


def run_mars_ccd_pipeline(  # noqa: C901
    sample_paths: Sequence[Sequence[str | Path]],
    ob_paths: Sequence[Sequence[str | Path]],
    dark_paths: Optional[Sequence[Sequence[str | Path]]] = None,
    output_path: Optional[Path] = None,
    roi: Optional[tuple] = None,
    gamma_filter: bool = True,
    background_roi: Optional[tuple] = None,
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
        Region of interest to apply (x_start, y_start, x_end, y_end)
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    background_roi : Optional[tuple]
        Sample-free background ROI (x0, y0, x1, y1) for flux-proxy normalization when proton
        charge is unavailable (issue #159). Mutually exclusive with proton-charge correction.

    Notes
    -----
    This function writes the normalized transmission data to disk in either HDF5 or TIFF format,
    depending on the file extension of `output_path`. Metadata and detector masks are included in the output.

    Returns
    -------
    sc.DataArray
        Final normalized transmission DataArray with metadata and masks
    """

    if output_path is None:
        raise ValueError("output_path is required")

    # Load data
    samples = [load_stack(paths) for paths in sample_paths]
    ob = [load_stack(paths) for paths in ob_paths]

    # Before combining, check that all sample runs have the same shape and some metadata keys match
    # Keys to check [ManufacturerStr, MotSlitVB.RBV, MotSlitVT.RBV, MotSlitHR.RBV, MotSlitHL.RBV].
    # MotSlit does not need to match for dark. ExposureTime is included in metadata checks and is
    # effectively averaged/normalized across runs (not summed) when normalize_by_runs=True.

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
    )

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
    )

    # Dark current is optional (issue #146): only load/combine it when dark paths are provided.
    dark = None
    if dark_paths:
        dark_runs = [load_stack(paths) for paths in dark_paths]
        dark = combine_runs(
            dark_runs,
            metadata_keys_to_sum=("ExposureTime",),
            metadata_check_match=[
                "ExposureTime",
                "ManufacturerStr",
            ],
            normalize_by_runs=True,
        )

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

    # Gamma filtering (optional)
    if gamma_filter:
        sample = apply_gamma_filter(sample)

    # Dark correction (optional) + normalization. With a shared dark frame, normalize_with_dark
    # subtracts the dark and normalizes in one step so the dark variance is not double-counted
    # in the transmission uncertainty (issue #142). Without dark, normalize directly.
    if background_roi is not None:
        # Flux-proxy normalization from a sample-free ROI (issue #159), in place of proton charge.
        # With a shared dark, route through normalize_with_dark so the #142 shared-dark variance
        # double-count is corrected (k = co/cs); without dark, normalize directly.
        if dark is not None:
            transmission = normalize_with_dark(sample, ob, dark, background_roi=background_roi)
        else:
            transmission = normalize_transmission(sample, ob, background_roi=background_roi)
    elif dark is not None:
        transmission = normalize_with_dark(sample, ob, dark)
    else:
        logger.info("No dark current provided; skipping dark correction")
        transmission = normalize_transmission(sample, ob)

    # Guarantee a float32 normalized data product (issue #147), regardless of any
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
        metadata["roi_applied"] = roi

    if background_roi is not None:
        metadata["background_roi"] = list(background_roi)

    if output_path.suffix.lower() in (".hdf5", ".h5"):
        write_hdf5(output_path, transmission, dead_pixel_mask="dead_pixels", metadata=metadata)
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

        write_tiff_stack(output_path, transmission, metadata=metadata, daqmetadata=daqmetadata)
    else:
        raise ValueError(f"Unsupported output file format: {output_path.suffix}")

    logger.success("MARS CCD pipeline completed successfully. Output written to {}", output_path)
    return transmission

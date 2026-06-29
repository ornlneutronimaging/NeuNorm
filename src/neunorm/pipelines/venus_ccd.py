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
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.filters.gamma_filter import apply_gamma_filter
from neunorm.loaders.stack_loader import load_stack
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.pixel_detector import detect_dead_pixels


def run_venus_ccd_pipeline(  # noqa: C901
    sample_paths: Sequence[Sequence[str | Path]],
    ob_paths: Sequence[Sequence[str | Path]],
    dark_paths: Optional[Sequence[Sequence[str | Path]]] = None,
    output_path: Optional[Path] = None,
    roi: Optional[tuple] = None,
    gamma_filter: bool = True,
    air_roi: Optional[tuple] = None,
    background_roi: Optional[tuple] = None,
) -> sc.DataArray:
    """Execute VENUS CCD/CMOS normalization pipeline.

    Pipeline Steps (12 total)
    - Load TIFF/FITS (sample, OB, dark [optional])
    - Load p_charge metadata
    - Run combine (critical for VENUS)
    - ROI clip (optional)
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
    output_path : Optional[Path]
        Path to save the output file (HDF5 or TIFF). Required; a value of None
        raises ``ValueError`` (the default exists only so ``dark_paths`` can keep
        its positional slot).
    roi : Optional[tuple]
        Region of interest to apply (x_start, y_start, x_end, y_end)
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    air_roi : Optional[tuple]
        Region of interest to use for air correction (x_start, y_start, x_end, y_end).
        If None, air correction is not applied.
    background_roi : Optional[tuple]
        Sample-free background ROI (x0, y0, x1, y1) for flux-proxy normalization when proton
        charge is unavailable (issue #159). Mutually exclusive with proton-charge correction. If
        ``roi`` is also given the detector is cropped first, so ``background_roi`` indices are
        resolved in the post-crop frame.

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
    # Keys to check ManufacturerStr. IntegratedPCharge is included in metadata checks and is
    # effectively averaged/normalized across runs (not summed) when normalize_by_runs=True.

    # When background_roi is used (proton-charge proxy), don't require/aggregate IntegratedPCharge.
    pc_keys = () if background_roi is not None else ("IntegratedPCharge",)

    sample = combine_runs(
        samples,
        metadata_keys_to_sum=pc_keys,
        metadata_check_match=["ManufacturerStr"],
        normalize_by_runs=True,
    )

    ob = combine_runs(
        ob,
        metadata_keys_to_sum=pc_keys,
        metadata_check_match=["ManufacturerStr"],
        normalize_by_runs=True,
    )

    # Dark current is optional (issue #146): only load/combine it when dark paths are provided.
    dark = None
    if dark_paths:
        dark_runs = [load_stack(paths) for paths in dark_paths]
        dark = combine_runs(
            dark_runs,
            metadata_keys_to_sum=pc_keys,
            metadata_check_match=["ManufacturerStr"],
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

    # Dark correction (optional) + normalization. The proton-charge coords are cast to float32
    # so the division does not silently re-promote the float32 image data to float64 (issue
    # #147; the coord is float64 because metadata is parsed via float()). With a shared dark
    # frame, normalize_with_dark subtracts the dark and normalizes in one step so the dark
    # variance is not double-counted in the transmission uncertainty (issue #142).
    if background_roi is not None:
        # Flux-proxy normalization from a sample-free ROI (issue #159), replacing the proton-charge
        # correction. With a shared dark, route through normalize_with_dark so the #142 shared-dark
        # variance double-count is corrected (k = co/cs).
        if dark is not None:
            transmission = normalize_with_dark(sample, ob, dark, background_roi=background_roi)
        else:
            transmission = normalize_transmission(sample, ob, background_roi=background_roi)
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
            )
        else:
            logger.info("No dark current provided; skipping dark correction")
            transmission = normalize_transmission(
                sample=sample,
                ob=ob,
                proton_charge_sample=proton_charge_sample,
                proton_charge_ob=proton_charge_ob,
            )

    # Air region correction (optional)
    if air_roi is not None:
        transmission = apply_air_region_correction(transmission, air_roi)

    # Guarantee a float32 normalized data product (issue #147), regardless of any
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

    logger.success("VENUS CCD pipeline completed successfully. Output written to {}", output_path)
    return transmission

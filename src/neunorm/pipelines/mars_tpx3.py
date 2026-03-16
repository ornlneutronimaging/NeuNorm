"""
MARS TPX3 normalization pipeline.
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
from neunorm.loaders.event_loader import load_event_data
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.tof.event_converter import convert_events_to_2d_histogram
from neunorm.tof.pixel_detector import detect_dead_pixels, detect_hot_pixels


def run_mars_tpx3_pipeline(  # noqa: C901
    sample_paths: Sequence[str | Path],
    ob_paths: Sequence[str | Path],
    output_path: Path,
    roi: Optional[tuple] = None,
    gamma_filter: bool = True,
    detector_shape: tuple[int, int] = (514, 514),
) -> sc.DataArray:
    """Execute MARS TPX3 normalization pipeline.

    Pipeline Step
    1. Load event data
    2. Convert events to 2D histogram
    3. Run combine (optional) # TODO
    4. ROI clip (optional)
    5. Dead pixel detection
    6. Hot pixel detection
    7. Gamma filtering
    8. Normalization
    9. Output

    Parameters
    ----------
    sample_paths : Sequence[str | Path]
        List of paths to sample HDF5 files
    ob_paths : Sequence[str | Path]
        List of paths to open beam HDF5 files
    output_path : Path
        Path to save the output file (HDF5 or TIFF)
    roi : Optional[tuple]
        Region of interest to apply (x_start, y_start, x_end, y_end)
    gamma_filter : bool
        Whether to apply gamma filtering to the sample data (default: True)
    detector_shape : tuple[int, int]
        Shape of the TPX3 detector (default: (514, 514))

    Notes
    -----
    This function writes the normalized transmission data to disk in either HDF5 or TIFF format,
    depending on the file extension of `output_path`. Metadata and detector masks are included in the output.

    Returns
    -------
    sc.DataArray
        Final normalized transmission DataArray with metadata and masks
    """

    # Load data and convert to histogram
    sample = sc.concat(
        [convert_events_to_2d_histogram(load_event_data(p), detector_shape) for p in sample_paths], dim="N_image"
    )
    ob = sc.concat(
        [convert_events_to_2d_histogram(load_event_data(p), detector_shape) for p in ob_paths], dim="N_image"
    )

    # Apply ROI if specified
    if roi:
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)

    # Average OB
    ob = prepare_reference(ob, dim="N_image")

    # Dead pixel detection
    sample.masks["dead_pixels"] = detect_dead_pixels(sample)

    # Hot pixel detection
    sample.masks["hot_pixels"] = detect_hot_pixels(sample)

    # Gamma filtering (optional)
    if gamma_filter:
        sample = apply_gamma_filter(sample)

    # Normalization
    transmission = normalize_transmission(sample, ob)

    # Write output
    metadata = {
        "sample_paths": [str(p) for p in sample_paths],
        "ob_paths": [str(p) for p in ob_paths],
        "gamma_filter_applied": gamma_filter,
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
        if "N_image" in transmission.dims:
            rename_map["N_image"] = "z"  # TIFF stacks typically use 'z' for the stack dimension
        if rename_map:
            transmission = transmission.rename_dims(rename_map)

        daqmetadata = {
            "facility": "HFIR",
            "instrument": "MARS",
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

    logger.success("MARS TPX3 pipeline completed successfully. Output written to {}", output_path)
    return transmission

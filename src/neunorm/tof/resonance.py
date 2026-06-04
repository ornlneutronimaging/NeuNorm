"""
Resonance detection for neutron transmission imaging.

Provides automatic detection of resonance dips in energy-space transmission
spectra using tiered filtering (background subtraction, SNR, peak shape).

Ported from venus_tof.resonance with minimal modifications.
"""

from typing import Dict, Optional

import numpy as np
import scipp as sc
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class ResonanceDetectionConfig(BaseModel):
    """
    Configuration for automatic resonance detection in transmission spectra.

    This class encapsulates parameters for tiered filtering:
    - Background subtraction (Gaussian filter)
    - Initial peak detection (find_peaks)
    - SNR filtering (Poisson statistics)
    - Peak shape filtering (width and prominence/width ratio)

    Parameters
    ----------
    background_sigma_fraction : float
        Gaussian filter width as fraction of spectrum length (default: 0.05 = 5%)
    initial_prominence : float
        Minimum prominence for initial peak detection (default: 0.01)
    initial_width : int
        Minimum peak width in bins for initial detection (default: 3)
    min_snr : float
        Minimum signal-to-noise ratio using Poisson statistics (default: 50.0)
    snr_window_fraction : float
        Relative energy window for SNR calculation (default: 0.15 = ±15% of E)
    min_peak_width : int
        Minimum allowed peak width in bins (default: 3)
    max_peak_width : int
        Maximum allowed peak width in bins (default: 60)
    min_prom_width_ratio : float
        Minimum prominence/width ratio (default: 0.001)

    Examples
    --------
    >>> config = ResonanceDetectionConfig(min_snr=100.0, max_peak_width=40)
    >>> result = detect_resonances(hist_ta, hist_ob, config=config)
    """

    background_sigma_fraction: float = Field(default=0.05, gt=0, le=0.2)
    initial_prominence: float = Field(default=0.01, gt=0)
    initial_width: int = Field(default=3, ge=2)
    min_snr: float = Field(default=50.0, gt=0)
    snr_window_fraction: float = Field(default=0.15, gt=0, lt=0.5)
    min_peak_width: int = Field(default=3, ge=2)
    max_peak_width: int = Field(default=60, ge=3)
    min_prom_width_ratio: float = Field(default=0.001, gt=0)

    @field_validator("max_peak_width")
    @classmethod
    def validate_max_greater_than_min(cls, v, info):
        if "min_peak_width" in info.data and v <= info.data["min_peak_width"]:
            raise ValueError(f"max_peak_width ({v}) must be > min_peak_width ({info.data['min_peak_width']})")
        return v


def _calculate_snr_poisson(
    energy: np.ndarray,
    peak_indices: np.ndarray,
    spectrum_ta_counts: np.ndarray,
    spectrum_ob_counts: np.ndarray,
    window_fraction: float = 0.15,
) -> np.ndarray:
    """
    Calculate SNR using Poisson statistics with relative energy windows.

    Uses proper Poisson uncertainty propagation: σ_T = T × √(1/S + 1/OB)
    Background windows scale with energy (ΔE/E = constant) to match TOF physics.

    Parameters
    ----------
    energy : np.ndarray
        Energy bin centers (eV)
    peak_indices : np.ndarray
        Indices of detected peaks in spectrum
    spectrum_ta_counts : np.ndarray
        Sample counts (integrated over spatial dimensions)
    spectrum_ob_counts : np.ndarray
        Open beam counts (integrated over spatial dimensions)
    window_fraction : float
        Relative window size (default: 0.15 = ±15% of peak energy)

    Returns
    -------
    np.ndarray
        SNR value for each peak
    """
    snr_values = []

    for idx in peak_indices:
        peak_energy = energy[idx]

        # Get counts at peak
        s_peak = spectrum_ta_counts[idx]
        ob_peak = spectrum_ob_counts[idx]

        if ob_peak <= 0:
            snr_values.append(0.0)
            continue

        # Transmission and uncertainty at peak
        t_peak = s_peak / ob_peak
        sigma_t_peak = t_peak * np.sqrt(1.0 / max(s_peak, 1) + 1.0 / max(ob_peak, 1))

        # Define background windows in RELATIVE energy
        gap = window_fraction

        left_region = (energy >= peak_energy * (1 - 2 * window_fraction)) & (energy < peak_energy * (1 - gap))
        right_region = (energy > peak_energy * (1 + gap)) & (energy <= peak_energy * (1 + 2 * window_fraction))
        bg_region = left_region | right_region

        if np.sum(bg_region) == 0:
            snr_values.append(0.0)
            continue

        # Get background counts
        s_bg = spectrum_ta_counts[bg_region]
        ob_bg = spectrum_ob_counts[bg_region]

        # Only use bins with OB counts > 0
        valid = ob_bg > 0
        if np.sum(valid) == 0:
            snr_values.append(0.0)
            continue

        # Calculate transmission and uncertainties in background
        t_bg = s_bg[valid] / ob_bg[valid]
        t_background = np.median(t_bg)

        sigma_t_bg_array = t_bg * np.sqrt(1.0 / np.maximum(s_bg[valid], 1) + 1.0 / np.maximum(ob_bg[valid], 1))
        sigma_t_background = np.median(sigma_t_bg_array)

        # Signal = resonance depth
        signal = abs(t_background - t_peak)

        # Noise = quadrature sum of uncertainties
        noise = np.sqrt(sigma_t_peak**2 + sigma_t_background**2)

        snr = signal / noise if noise > 0 else 0.0
        snr_values.append(snr)

    return np.array(snr_values)


def detect_resonances(
    hist_ta: sc.DataArray,
    hist_ob: sc.DataArray,
    config: Optional[ResonanceDetectionConfig] = None,
    known_resonances: Optional[np.ndarray] = None,
    validation_tolerance: float = 0.05,
) -> Dict:
    """
    Auto-detect resonance dips in neutron transmission data.

    Uses tiered filtering approach:
    1. Background subtraction (Gaussian filter)
    2. Initial peak detection (scipy.signal.find_peaks)
    3. SNR filtering (Poisson statistics with relative energy windows)
    4. Peak shape filtering (width and prominence/width ratio)

    Parameters
    ----------
    hist_ta : sc.DataArray
        Sample histogram with dimensions (energy, x, y)
    hist_ob : sc.DataArray
        Open beam histogram with dimensions (energy, x, y)
    config : ResonanceDetectionConfig, optional
        Detection parameters. If None, uses defaults.
    known_resonances : np.ndarray, optional
        Known resonance energies (eV) for validation
    validation_tolerance : float
        Relative tolerance for matching known resonances (default: 0.05 = ±5%)

    Returns
    -------
    dict
        Detection results containing:
        - 'resonance_energies': np.ndarray of detected energies (eV)
        - 'resonance_indices': np.ndarray of bin indices
        - 'snr_values': np.ndarray of SNR for each resonance
        - 'n_initial': int, peaks after initial detection
        - 'n_snr_filtered': int, peaks after SNR filter
        - 'n_shape_filtered': int, peaks after shape filter
        - 'validation': dict (if known_resonances provided)

    Examples
    --------
    >>> result = detect_resonances(hist_ta, hist_ob)
    >>> print(f"Detected {len(result['resonance_energies'])} resonances")
    """
    if config is None:
        config = ResonanceDetectionConfig()

    logger.info("Starting automatic resonance detection")
    logger.info(f"  Background sigma: {config.background_sigma_fraction * 100:.0f}% of spectrum")
    logger.info(f"  SNR window: ±{config.snr_window_fraction * 100:.0f}% of E (relative)")
    logger.info(f"  Min SNR: {config.min_snr}")

    # Step 1: Compute integrated transmission spectrum
    logger.info("Computing integrated transmission spectrum...")
    spectrum_ta = hist_ta.sum(["x", "y"])
    spectrum_ob = hist_ob.sum(["x", "y"])
    integrated_transmission = spectrum_ta / spectrum_ob

    # Extract numpy arrays
    energy_edges = integrated_transmission.coords["energy"].values
    energy_centers = (energy_edges[:-1] + energy_edges[1:]) / 2
    transmission_spectrum = integrated_transmission.values
    transmission_spectrum = np.nan_to_num(transmission_spectrum, nan=1.0, posinf=1.0, neginf=0.0)

    # Step 2: Background subtraction
    logger.info("Applying background subtraction...")
    sigma = int(config.background_sigma_fraction * len(transmission_spectrum))
    background = gaussian_filter1d(transmission_spectrum, sigma=sigma, mode="nearest")
    baseline_corrected = transmission_spectrum - background
    logger.info(f"  Gaussian sigma: {sigma} bins")

    # Step 3: Initial peak detection
    logger.info("Initial peak detection...")
    inverted = -baseline_corrected
    peaks_initial, properties = find_peaks(inverted, prominence=config.initial_prominence, width=config.initial_width)
    logger.info(f"  Initial detection: {len(peaks_initial)} peaks")

    if len(peaks_initial) == 0:
        logger.warning("No peaks detected in initial detection")
        return {
            "resonance_energies": np.array([]),
            "resonance_indices": np.array([]),
            "snr_values": np.array([]),
            "n_initial": 0,
            "n_snr_filtered": 0,
            "n_shape_filtered": 0,
        }

    # Step 4: SNR filtering with Poisson statistics
    logger.info("Applying SNR filter (Poisson statistics)...")
    spectrum_ta_counts = spectrum_ta.values
    spectrum_ob_counts = spectrum_ob.values

    snr_values = _calculate_snr_poisson(
        energy_centers,
        peaks_initial,
        spectrum_ta_counts,
        spectrum_ob_counts,
        window_fraction=config.snr_window_fraction,
    )

    snr_mask = snr_values >= config.min_snr
    peaks_snr = peaks_initial[snr_mask]
    snr_values_filtered = snr_values[snr_mask]
    logger.info(f"  SNR filter (>= {config.min_snr}): {len(peaks_initial)} → {len(peaks_snr)} peaks")

    if len(peaks_snr) == 0:
        logger.warning("No peaks passed SNR filter")
        return {
            "resonance_energies": np.array([]),
            "resonance_indices": np.array([]),
            "snr_values": np.array([]),
            "n_initial": len(peaks_initial),
            "n_snr_filtered": 0,
            "n_shape_filtered": 0,
        }

    # Step 5: Peak shape filtering
    logger.info("Applying peak shape filter...")
    widths_at_peaks = properties["widths"][snr_mask]
    prominences_at_peaks = properties["prominences"][snr_mask]

    width_mask = (widths_at_peaks >= config.min_peak_width) & (widths_at_peaks <= config.max_peak_width)

    prom_width_ratio = prominences_at_peaks / widths_at_peaks
    ratio_mask = prom_width_ratio >= config.min_prom_width_ratio

    shape_mask = width_mask & ratio_mask
    peaks_final = peaks_snr[shape_mask]
    snr_values_final = snr_values_filtered[shape_mask]

    logger.info(
        f"  Shape filter (width {config.min_peak_width}-{config.max_peak_width}): "
        f"{len(peaks_snr)} → {len(peaks_final)} peaks"
    )

    # Extract final resonance energies
    resonance_energies = energy_centers[peaks_final]

    logger.success(f"Detected {len(resonance_energies)} resonances")
    if len(resonance_energies) > 0:
        logger.info(f"  Energy range: {resonance_energies.min():.2f} - {resonance_energies.max():.2f} eV")

    # Build result dictionary
    result = {
        "resonance_energies": resonance_energies,
        "resonance_indices": peaks_final,
        "snr_values": snr_values_final,
        "n_initial": len(peaks_initial),
        "n_snr_filtered": len(peaks_snr),
        "n_shape_filtered": len(peaks_final),
        "widths": widths_at_peaks[shape_mask],
        "prominences": prominences_at_peaks[shape_mask],
    }

    # Optional validation against known resonances
    if known_resonances is not None:
        logger.info(f"Validating against {len(known_resonances)} known resonances...")
        validation = _validate_resonances(resonance_energies, known_resonances, tolerance=validation_tolerance)
        result["validation"] = validation

        logger.info(f"  Matched: {validation['n_matched']}/{len(known_resonances)}")
        logger.info(f"  Recall: {validation['recall'] * 100:.1f}%")
        logger.info(f"  Precision: {validation['precision'] * 100:.1f}%")
        logger.info(f"  False positives: {validation['n_false_positives']}")

    return result


def _validate_resonances(detected_energies: np.ndarray, known_energies: np.ndarray, tolerance: float = 0.05) -> Dict:
    """
    Validate detected resonances against known values.

    Parameters
    ----------
    detected_energies : np.ndarray
        Detected resonance energies (eV)
    known_energies : np.ndarray
        Known resonance energies (eV)
    tolerance : float
        Relative tolerance for matching (default: 0.05 = ±5%)

    Returns
    -------
    dict
        Validation metrics (matched_pairs, recall, precision, f1_score)
    """
    matched_pairs = []
    unmatched_known = []

    for known_e in known_energies:
        if len(detected_energies) == 0:
            unmatched_known.append(known_e)
            continue

        errors = np.abs(detected_energies - known_e) / known_e
        min_error = np.min(errors)

        if min_error < tolerance:
            idx = np.argmin(errors)
            matched_pairs.append((known_e, detected_energies[idx], min_error))
        else:
            unmatched_known.append(known_e)

    n_matched = len(matched_pairs)
    n_false_positives = len(detected_energies) - n_matched

    recall = n_matched / len(known_energies) if len(known_energies) > 0 else 0
    precision = n_matched / len(detected_energies) if len(detected_energies) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "matched_pairs": matched_pairs,
        "unmatched_known": unmatched_known,
        "n_matched": n_matched,
        "n_false_positives": n_false_positives,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
    }


def aggregate_resonance_image(
    hist_ta: sc.DataArray, hist_ob: sc.DataArray, resonance_indices: np.ndarray
) -> sc.DataArray:
    """
    Create aggregated 2D transmission image from resonance bins.

    Sums raw counts over detected resonance bins, THEN computes transmission.
    This is mathematically correct: (Σa)/(Σb) ≠ Σ(a/b)

    Parameters
    ----------
    hist_ta : sc.DataArray
        Sample histogram with dimensions (energy, x, y)
    hist_ob : sc.DataArray
        Open beam histogram with dimensions (energy, x, y)
    resonance_indices : np.ndarray
        Energy bin indices corresponding to detected resonances

    Returns
    -------
    sc.DataArray
        Aggregated transmission image with dimensions (x, y)

    Examples
    --------
    >>> result = detect_resonances(hist_ta, hist_ob)
    >>> trans_image = aggregate_resonance_image(hist_ta, hist_ob, result['resonance_indices'])
    """
    logger.info(f"Aggregating transmission over {len(resonance_indices)} resonance bins...")

    # Use numpy advanced indexing to select resonance bins
    ta_values = hist_ta.values[resonance_indices, :, :]  # (n_resonances, x, y)
    ob_values = hist_ob.values[resonance_indices, :, :]

    # Sum counts over energy dimension (axis 0)
    ta_summed = ta_values.sum(axis=0)  # (x, y)
    ob_summed = ob_values.sum(axis=0)

    # Compute transmission (after aggregation)
    transmission_values = ta_summed / ob_summed

    # Create scipp DataArray with spatial coordinates only
    transmission_aggregated = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=transmission_values, unit=sc.units.one),
        coords={"x": hist_ta.coords["x"], "y": hist_ta.coords["y"]},
    )

    # Preserve masks from input histograms
    if hist_ta.masks:
        for mask_name, mask_data in hist_ta.masks.items():
            transmission_aggregated.masks[mask_name] = mask_data

    logger.success(f"Aggregated transmission image created: {transmission_aggregated.sizes}")

    return transmission_aggregated


def print_detection_summary(result: Dict) -> None:
    """
    Print human-readable summary of detection results.

    Parameters
    ----------
    result : dict
        Output from detect_resonances()
    """
    print("=" * 60)
    print("RESONANCE DETECTION SUMMARY")
    print("=" * 60)
    print("Filtering stages:")
    print(f"  Initial detection:  {result['n_initial']:3d} peaks")
    print(f"  After SNR filter:   {result['n_snr_filtered']:3d} peaks")
    print(f"  After shape filter: {result['n_shape_filtered']:3d} peaks")
    print()
    print(f"Final detected resonances: {len(result['resonance_energies'])}")

    if len(result["resonance_energies"]) > 0:
        print(f"Energy range: {result['resonance_energies'].min():.2f} - {result['resonance_energies'].max():.2f} eV")
        print(f"SNR range: {result['snr_values'].min():.1f} - {result['snr_values'].max():.1f}")

    if "validation" in result:
        val = result["validation"]
        print()
        print("Validation Results:")
        print(f"  Known resonances: {len(val['matched_pairs']) + len(val['unmatched_known'])}")
        print(f"  Matched: {val['n_matched']}")
        print(f"  False positives: {val['n_false_positives']}")
        print(f"  Recall: {val['recall'] * 100:.1f}%")
        print(f"  Precision: {val['precision'] * 100:.1f}%")
        print(f"  F1 Score: {val['f1_score']:.3f}")

    print("=" * 60)

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

from neunorm.data_models.roi import RegionsLike
from neunorm.processing.spectrum_reducer import _symmetrize_masks, roi_mean_spectrum


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
        """Validate that ``max_peak_width`` exceeds ``min_peak_width``."""
        if "min_peak_width" in info.data and v <= info.data["min_peak_width"]:
            raise ValueError(f"max_peak_width ({v}) must be > min_peak_width ({info.data['min_peak_width']})")
        return v


def _calculate_snr_from_variances(
    energy: np.ndarray,
    peak_indices: np.ndarray,
    transmission: sc.DataArray,
    spectrum_ob: sc.DataArray,
    window_fraction: float = 0.15,
) -> np.ndarray:
    """SNR per peak, taking sigma_T from the **propagated** variance instead of assuming raw counts.

    Same estimator as the Poisson-counts version this replaced — resonance depth over the quadrature
    sum of the peak's and the background's transmission uncertainty, with background windows that scale
    as ``dE/E`` — and the same branch structure, bin for bin. Only where sigma_T comes from differs.

    That substitution is why it exists. The old form hard-coded ``sigma_T = T*sqrt(1/S + 1/OB)``, i.e.
    ``Var(S) = S``, which holds only for raw summed counts: hand it region **means** and the bracket
    grows by the pixel count, shrinking every SNR by ``1/sqrt(N_pixels)`` — measured 282.8 -> 28.3 over
    a 10x10 detector, which takes four detected peaks to zero against the default ``min_snr=50``.
    Reading the variance the reduction already propagated is scale-free, so the same spectrum gives the
    same SNR whether it is expressed as sums or as means.

    It is also numerically neutral where the old formula was right. For Poisson pixels
    ``Var(mean) = sum(Var)/N**2``, so ``Var(T)/T**2 = 1/S_sum + 1/OB_sum`` — algebraically identical to
    the hard-coded bracket. Measured on synthetic Poisson data the detected peaks are unchanged and the
    SNR values agree to 1.9e-16 relative.

    Where the two DO differ is input whose variances are not Poisson — dark-subtracted or run-combined
    counts, whose variances are inflated relative to their values. There the old formula understated
    sigma_T and this one reports the uncertainty the data actually carries, so the SNR comes out lower,
    and correctly so. That is a deliberate change, not an incidental one.

    ``spectrum_ob`` is taken only for the ``> 0`` validity tests, so a bin the old code skipped is
    still skipped for the same reason rather than for a different one.
    """
    values = transmission.values
    stdevs = np.sqrt(transmission.variances)
    ob_values = spectrum_ob.values
    snr_values = []

    for idx in peak_indices:
        peak_energy = energy[idx]

        if ob_values[idx] <= 0:
            snr_values.append(0.0)
            continue

        t_peak = values[idx]
        sigma_t_peak = stdevs[idx]

        # Define background windows in RELATIVE energy
        gap = window_fraction

        left_region = (energy >= peak_energy * (1 - 2 * window_fraction)) & (energy < peak_energy * (1 - gap))
        right_region = (energy > peak_energy * (1 + gap)) & (energy <= peak_energy * (1 + 2 * window_fraction))
        bg_region = left_region | right_region

        if np.sum(bg_region) == 0:
            snr_values.append(0.0)
            continue

        # Only use bins with OB counts > 0
        valid = ob_values[bg_region] > 0
        if np.sum(valid) == 0:
            snr_values.append(0.0)
            continue

        t_background = np.median(values[bg_region][valid])
        sigma_t_background = np.median(stdevs[bg_region][valid])

        # Signal = resonance depth
        signal = abs(t_background - t_peak)

        # Noise = quadrature sum of uncertainties
        noise = np.sqrt(sigma_t_peak**2 + sigma_t_background**2)

        snr = signal / noise if noise > 0 else 0.0
        snr_values.append(snr)

    return np.array(snr_values)


def _with_poisson_variances(data: sc.DataArray) -> sc.DataArray:
    """``data`` with ``Var = counts`` filled in when it carries no variances.

    Counting data has Poisson variance whether or not anything recorded it, and the SNR step needs a
    variance to read. Synthesizing it here is what makes the variance-routed SNR reproduce the old
    counts formula exactly on inputs that never carried variances — which is all of the existing
    resonance test data. Negative values (dark-subtracted counts scattering below zero) are floored at
    zero rather than given a negative variance.

    The values and the variances are held in the SAME float dtype, because scipp requires it. That is
    not a detail: ``float32`` is what ``event_converter`` and ``tiff_loader`` produce natively and
    integer counts are what a hand-built histogram carries, and a mismatched pair raises a scipp dtype
    error that names nothing in NeuNorm. Integer input is promoted to ``float64``; a float input keeps
    its own precision.
    """
    if data.variances is not None:
        return data
    values = np.asarray(data.values)
    # float32 stays float32; integer and everything else become float64
    dtype = values.dtype if values.dtype in (np.float32, np.float64) else np.float64
    values = values.astype(dtype, copy=False)
    out = data.copy(deep=False)
    out.data = sc.array(
        dims=data.dims,
        values=values,
        variances=np.clip(values, 0.0, None).astype(dtype, copy=False),
        unit=data.unit,
    )
    return out


def _region_spectra(
    hist_ta: sc.DataArray,
    hist_ob: sc.DataArray,
    spectrum_roi: Optional[RegionsLike],
) -> tuple[sc.DataArray, sc.DataArray]:
    """Sample and open-beam spectra as mask-aware pooled region **means** over the same region.

    ``spectrum_roi=None`` uses the whole detector, which is what this function replaced —
    ``hist.sum(["x", "y"])`` on each side independently. Two changes come with it, and only the second
    is about the mean:

    Both sides are given the union of both sides' masks first. A region mean divides by its own count
    of unmasked pixels, so masking a dead pixel on the sample alone makes the numerator and the
    denominator average over different pixel sets — and the dead pixel still inflates the open beam.
    Measured with one of four pixels dead under non-uniform flux: 0.400 from the ratio of sums, 0.533
    from the ratio of means, 0.800 once the masks match, against a true 0.800. **The mean alone does
    not fix this**; it removes the ``N_s != N_o`` scale error, and mask symmetry removes the bias.

    A stack with no ``x``/``y`` dims is already a spectrum and is passed through, so callers handing in
    pre-reduced data keep working.
    """
    if "x" not in hist_ta.dims or "y" not in hist_ta.dims:
        if spectrum_roi is not None:
            raise ValueError(
                f"spectrum_roi needs 'x' and 'y' dimensions to select a region; got dims {hist_ta.dims}. "
                "The input looks already reduced to a spectrum."
            )
        return _with_poisson_variances(hist_ta), _with_poisson_variances(hist_ob)

    ta = _with_poisson_variances(hist_ta)
    ob = _with_poisson_variances(hist_ob)
    ta, ob = _symmetrize_masks(ta, ob)

    regions = spectrum_roi if spectrum_roi is not None else (0, 0, ta.sizes["x"], ta.sizes["y"])
    # strict=False on both sides: a fully absorbing bin (transmission 0) is exactly what a resonance
    # dip looks like, and an empty open-beam bin is handled downstream by the `> 0` validity tests
    # rather than by aborting a detection run.
    spectrum_ta = roi_mean_spectrum(ta, regions, strict=False, region_arg="spectrum_roi", name="hist_ta")
    spectrum_ob = roi_mean_spectrum(ob, regions, strict=False, region_arg="spectrum_roi", name="hist_ob")
    return spectrum_ta, spectrum_ob


def detect_resonances(
    hist_ta: sc.DataArray,
    hist_ob: sc.DataArray,
    config: Optional[ResonanceDetectionConfig] = None,
    known_resonances: Optional[np.ndarray] = None,
    validation_tolerance: float = 0.05,
    *,
    spectrum_roi: Optional[RegionsLike] = None,
) -> Dict:
    """
    Auto-detect resonance dips in neutron transmission data.

    Uses tiered filtering approach:
    1. Background subtraction (Gaussian filter)
    2. Initial peak detection (scipy.signal.find_peaks)
    3. SNR filtering (from the propagated transmission variance, with relative energy windows)
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
    spectrum_roi : ROI, MaskROI, tuple, or a sequence of them, optional
        Restrict the integrated spectrum to a region instead of the whole detector, so a resonance is
        looked for where the sample actually is. ``None`` (default) uses the whole detector, which is
        the historical behaviour. Indices are resolved against the arrays as passed.

    Notes
    -----
    The integrated spectrum is a mask-aware pooled region **mean**, and both inputs are given the union
    of both inputs' masks before it is taken. Two independent reductions —
    ``hist_ta.sum(["x", "y"])`` and ``hist_ob.sum(["x", "y"])`` — are each mask-aware, but a mask
    present on only one side excludes a pixel from that side alone, so the numerator and the
    denominator cover different pixels and the ratio is biased. Measured with one of four pixels dead
    under non-uniform flux: 0.400 from the ratio of sums against a true 0.800.

    The SNR is computed from the propagated transmission variance rather than from a hard-coded Poisson
    formula over raw counts, which is what lets the spectrum be a mean at all. For Poisson counts the
    two agree to floating-point round-off, so detection on counting data is unchanged.

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
        - 'validation': dict, only when ``known_resonances`` is given and at least one peak passes the SNR filter

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

    # Step 1: Compute integrated transmission spectrum — the mask-aware pooled region MEAN of each
    # side, over the same region, divided once. The region is collapsed before the division because
    # (Sum a)/(Sum b) != Sum(a/b), the same identity aggregate_resonance_image states for the
    # spectral direction.
    logger.info("Computing integrated transmission spectrum...")
    if spectrum_roi is not None:
        logger.info("  Restricting the spectrum to region(s) {}", spectrum_roi)
    spectrum_ta, spectrum_ob = _region_spectra(hist_ta, hist_ob, spectrum_roi)
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

    # Step 4: SNR filtering from the propagated transmission variance
    logger.info("Applying SNR filter (propagated transmission variance)...")
    snr_values = _calculate_snr_from_variances(
        energy_centers,
        peaks_initial,
        integrated_transmission,
        spectrum_ob,
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

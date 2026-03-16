"""
Transmission normalization for neutron imaging.

Implements the core neutron imaging equation: T = Sample / OpenBeam
with proper uncertainty propagation and beam corrections.
"""

from typing import Optional

import scipp as sc
from loguru import logger


def normalize_transmission(
    sample: sc.DataArray,
    ob: sc.DataArray,
    proton_charge_sample: Optional[float] = None,
    proton_charge_ob: Optional[float] = None,
    pc_uncertainty: float = 0.005,
) -> sc.DataArray:
    """
    Normalize sample by open beam to compute transmission.

    Formula: T = (Sample / pc_sample) / (OB / pc_ob)

    Handles:
    - Variance propagation (automatic via scipp)
    - Proton charge corrections (for SNS pulsed beam)
    - Systematic uncertainties
    - Mask preservation

    Parameters
    ----------
    sample : sc.DataArray
        Sample histogram with variance
    ob : sc.DataArray
        Open beam histogram with variance
    proton_charge_sample : float, optional
        Integrated proton charge during sample acquisition (Coulombs)
        If provided, normalizes by beam intensity
    proton_charge_ob : float, optional
        Integrated proton charge during OB acquisition (Coulombs)
    pc_uncertainty : float, optional
        Relative proton charge uncertainty (default: 0.005 = 0.5%)
        From PLEIADES measurements

    Returns
    -------
    sc.DataArray
        Transmission with dimensions matching input
        Unit: dimensionless
        Includes propagated variance and systematic uncertainties

    Examples
    --------
    >>> # Basic normalization
    >>> transmission = normalize_transmission(hist_sample, hist_ob)

    >>> # With proton charge correction (SNS)
    >>> transmission = normalize_transmission(
    ...     hist_sample, hist_ob,
    ...     proton_charge_sample=500.0,
    ...     proton_charge_ob=505.0
    ... )

    Notes
    -----
    - Scipp automatically propagates variance through division
    - Masks are preserved from both sample and OB (OR combination)
    - Zero OB counts produce inf/nan (handle with masks before calling)
    """
    logger.info("Normalizing transmission: T = Sample / OB")

    # Apply proton charge corrections if provided
    if proton_charge_sample is not None:
        logger.info(f"  Applying proton charge correction: Sample pc={proton_charge_sample:.1f} C")
        sample_corrected = sample / sc.scalar(proton_charge_sample, unit="C")

        # Add proton charge systematic uncertainty
        if sample_corrected.variances is not None:
            pc_contribution = (pc_uncertainty * sample_corrected.values) ** 2
            sample_corrected.variances = sample_corrected.variances + pc_contribution
    else:
        sample_corrected = sample

    if proton_charge_ob is not None:
        logger.info(f"  Applying proton charge correction: OB pc={proton_charge_ob:.1f} C")
        ob_corrected = ob / sc.scalar(proton_charge_ob, unit="C")

        # Add proton charge systematic uncertainty
        if ob_corrected.variances is not None:
            pc_contribution = (pc_uncertainty * ob_corrected.values) ** 2
            ob_corrected.variances = ob_corrected.variances + pc_contribution
    else:
        ob_corrected = ob

    # Normalize
    if sample_corrected.dims == ob_corrected.dims:
        transmission = sample_corrected / ob_corrected
    else:
        # Need to broadcast to match dimensions
        ob_corrected_broadcast = ob_corrected.copy()
        ob_var = ob_corrected_broadcast.variances.copy() if ob_corrected_broadcast.variances is not None else None
        ob_corrected_broadcast.variances = None
        transmission = sample_corrected / ob_corrected_broadcast
        if ob_var is not None and sample_corrected.variances is not None:
            # Var(T) = (T^2) * (Var(Sample)/Sample^2 + Var(OB)/OB^2)
            # which is equivalent to:
            # Var(T) = (Var(Sample) / OB^2) + (Sample^2 * Var(OB) / OB^4)
            transmission.variances = (sample_corrected.variances / ob_corrected_broadcast.values**2) + (
                sample_corrected.values**2 * ob_var / ob_corrected_broadcast.values**4
            )

    # copy dropped unaligned coordinates from input
    for coord in sample.coords:
        if not sample.coords[coord].aligned:
            transmission.coords[coord] = sample.coords[coord]

    logger.success("✓ Transmission normalized")

    return transmission

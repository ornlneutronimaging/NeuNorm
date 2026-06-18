"""
Transmission normalization for neutron imaging.

Implements the core neutron imaging equation: T = Sample / OpenBeam
with proper uncertainty propagation and beam corrections.
"""

from typing import Optional, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.processing.dark_corrector import subtract_dark


def normalize_transmission(  # noqa: C901
    sample: sc.DataArray,
    ob: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
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
    proton_charge_sample : float or sc.Variable, optional
        Integrated proton charge during sample acquisition (Coulombs)
        If provided, normalizes by beam intensity
    proton_charge_ob : float or sc.Variable, optional
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
        if isinstance(proton_charge_sample, sc.Variable):
            logger.info(
                f"Applying proton charge correction: Sample mean pc={proton_charge_sample.mean().value} "
                f"{proton_charge_sample.unit}"
            )
            sample_corrected = sample / proton_charge_sample
        else:
            logger.info(f"  Applying proton charge correction: Sample pc={proton_charge_sample} C")
            sample_corrected = sample / sc.scalar(proton_charge_sample, unit="C")

        # Add proton charge systematic uncertainty
        if sample_corrected.variances is not None:
            pc_contribution = (pc_uncertainty * sample_corrected.values) ** 2
            sample_corrected.variances = sample_corrected.variances + pc_contribution
    else:
        sample_corrected = sample

    if proton_charge_ob is not None:
        if isinstance(proton_charge_ob, sc.Variable):
            logger.info(
                f"Applying proton charge correction: OB mean pc={proton_charge_ob.mean().value} {proton_charge_ob.unit}"
            )
            ob_corrected = ob / proton_charge_ob
        else:
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


def normalize_with_dark(
    sample: sc.DataArray,
    ob: sc.DataArray,
    dark: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
) -> sc.DataArray:
    """Dark-correct and normalize in one step, treating the shared dark frame correctly.

    Computes ``T = (sample - dark) / (ob - dark)`` (with the optional proton-charge
    correction) where the **same** averaged ``dark`` is subtracted from both sample and open
    beam. ``subtract_dark`` + ``normalize_transmission`` would treat the numerator and
    denominator as statistically independent and propagate ``Var(dark)`` twice; this function
    removes that spurious shared-dark covariance term (issue #142).

    The transmission **values** are identical to
    ``normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark), ...)`` — only
    the propagated variance is corrected (reduced) by ``2 * k**2 * (sample-dark) * Var(dark) /
    (ob-dark)**3``, with ``k = pc_ob / pc_sample`` (1 when no proton charge). The proton-charge
    systematic and the sample/open-beam Poisson terms are unchanged.

    Parameters
    ----------
    sample, ob, dark : sc.DataArray
        Sample, open-beam and (averaged) dark-current frames, each carrying Poisson variance.
        The same ``dark`` is subtracted from both ``sample`` and ``ob``.
    proton_charge_sample, proton_charge_ob : float or sc.Variable, optional
        Integrated proton charge for the SNS beam correction (see ``normalize_transmission``).
    pc_uncertainty : float, optional
        Relative proton-charge uncertainty (default 0.005).

    Returns
    -------
    sc.DataArray
        Transmission with correctly-propagated variance (no shared-dark double-counting).
    """
    sample_dc = subtract_dark(sample, dark)
    ob_dc = subtract_dark(ob, dark)
    transmission = normalize_transmission(sample_dc, ob_dc, proton_charge_sample, proton_charge_ob, pc_uncertainty)

    # Correct the shared-dark double-count (issue #142). normalize_transmission propagated
    # Var(dark) through BOTH numerator and denominator as if they were independent; the true
    # propagation (dark appears once) is smaller by 2*k^2*s*Var(D)/o^3. Subtract that term.
    if transmission.variances is None or dark.variances is None:
        return transmission

    # Use scipp's unit-carrying value/variance accessors so (a) the (3D sample) vs (2D ob/dark)
    # broadcast and the per-image proton-charge ratio align by dimension name and (b) scipp
    # validates units: counts * counts**2 / counts**3 = dimensionless, matching Var(T).
    s_v = sc.values(sample_dc)  # counts
    o_v = sc.values(ob_dc)  # counts
    var_d_v = sc.variances(dark)  # counts**2
    over_count = 2.0 * s_v * var_d_v / (o_v**3)

    k_squared = _proton_charge_ratio_squared(proton_charge_sample, proton_charge_ob)
    if k_squared is not None:
        over_count = k_squared * over_count

    over_values = sc.to_unit(over_count, "dimensionless").transpose(transmission.dims).values
    # Match the variance dtype so the correction never promotes a float32 pipeline to float64.
    over_values = over_values.astype(transmission.variances.dtype, copy=False)
    # Zero the correction where ob-dark == 0 (those pixels are already inf/nan in T).
    over_values = np.where(np.isfinite(over_values), over_values, 0.0)
    # Clamp to >= 0 defensively; the corrected variance is a true (non-negative) variance.
    transmission.variances = np.clip(transmission.variances - over_values, 0.0, None)
    logger.success("✓ Shared-dark variance double-count corrected (issue #142)")

    return transmission


def _proton_charge_ratio_squared(
    proton_charge_sample: Optional[Union[float, sc.Variable]],
    proton_charge_ob: Optional[Union[float, sc.Variable]],
) -> Optional[sc.Variable]:
    """Return ``(pc_ob / pc_sample)**2`` as a dimensionless scipp Variable, or None if either
    proton charge is absent (k = 1, the MARS / no-beam-correction case)."""
    if proton_charge_sample is None or proton_charge_ob is None:
        return None
    pc_s = (
        proton_charge_sample
        if isinstance(proton_charge_sample, sc.Variable)
        else sc.scalar(float(proton_charge_sample), unit="C")
    )
    pc_o = (
        proton_charge_ob if isinstance(proton_charge_ob, sc.Variable) else sc.scalar(float(proton_charge_ob), unit="C")
    )
    k = sc.to_unit(pc_o / pc_s, "dimensionless")
    return k * k
